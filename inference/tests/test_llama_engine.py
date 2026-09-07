from __future__ import annotations

import hashlib
import http.client
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_inference.contracts import Capabilities, normalize_request
from jarvis_inference.engine import EngineProtocolFailure, PreparedPrompt
from jarvis_inference.gemma4 import GEMMA4_PROMPT_PROFILE
from jarvis_inference.llama_engine import REQUIRED_FLAGS, LlamaServerEngine
from jarvis_inference.private_temp import PrivateRuntimeDirectory, PrivateTempFailure
from jarvis_inference.strict_json import canonical_bytes, parse_body
from jarvis_inference.windows_process import WindowsProcessFailure


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.pid = 43210

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.terminated = True
        return 0


class FakeHTTP:
    def __init__(self, port, secret):
        self.calls = []

    def request(self, method, path, value, deadline):
        self.calls.append((method, path, value))
        return {"status": "ok"}


class FakeStream:
    def __init__(self):
        self.closed = False

        self.lines = iter([
            b'data: {"content":"hi","stop":false,"tokens":[101]}\n',
            b'data: {"content":"","stop":true,"tokens":[2],"tokens_predicted":2,"stop_type":"eos"}\n',
        ])

    def readline(self, limit):
        return next(self.lines, b"")

    def close(self):
        self.closed = True


def manifest(prompt_profile=None):
    return SimpleNamespace(
        root=Path("/runtime"), context_window=32,
        backend={"executable": {"path": "runtime/llama-server.exe"}, "parameters": {}},
        model={"artifacts": [{"path": "models/model.gguf"}]},
        prompt_profile=prompt_profile,
    )


class LlamaEngineTests(unittest.TestCase):
    def test_failed_temp_cleanup_retains_owner_until_a_successful_retry(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as temporary:
            owner = PrivateRuntimeDirectory(Path(temporary), prefix="cyrus-test-")
            path = owner.open()
            engine = LlamaServerEngine(manifest())
            engine._runtime_temp = owner
            with patch("jarvis_inference.private_temp.shutil.rmtree", side_effect=PermissionError("busy")):
                with self.assertRaises(PrivateTempFailure):
                    engine._close_runtime_temp()
            self.assertIs(engine._runtime_temp, owner)
            self.assertEqual(owner.path, path)
            engine._close_runtime_temp()
            self.assertIsNone(engine._runtime_temp)
            self.assertIsNone(owner.path)
            self.assertFalse(path.exists())

    def test_cancel_interrupts_a_blocked_response_read(self) -> None:
        for operation in ("cancel", "inject_cancel"):
            with self.subTest(operation=operation):
                client, peer = socket.socketpair()
                peer.sendall(b"HTTP/1.1 200 OK\r\n\r\n")
                response = http.client.HTTPResponse(client)
                response.begin()
                engine = LlamaServerEngine(manifest())
                engine.process = FakeProcess()
                engine._active_response = response
                entered = threading.Event()
                failures = []

                def read():
                    entered.set()
                    try:
                        response.readline()
                    except Exception as exc:
                        failures.append(exc)

                reader = threading.Thread(target=read, daemon=True)
                canceller = threading.Thread(target=getattr(engine, operation), daemon=True)
                try:
                    reader.start()
                    self.assertTrue(entered.wait(1))
                    time.sleep(0.02)
                    canceller.start()
                    canceller.join(0.5)
                    self.assertFalse(canceller.is_alive(), "cancel waited for the remote body")
                    reader.join(0.5)
                    self.assertFalse(reader.is_alive())
                    self.assertEqual(failures, [])
                finally:
                    peer.close()
                    reader.join(1)
                    if canceller.ident is not None:
                        canceller.join(1)
                    response.close()
                    client.close()

    def test_spawn_rehashes_and_uses_frozen_flags_and_environment_secret(self) -> None:
        process = FakeProcess()
        with patch("jarvis_inference.llama_engine.verify_artifacts") as verify, \
             patch("jarvis_inference.llama_engine.resolve_local", side_effect=[Path("/runtime/server.exe"), Path("/runtime/model.gguf")]), \
             patch("jarvis_inference.llama_engine._free_port", return_value=43123), \
             patch("jarvis_inference.llama_engine.BackendHTTP", FakeHTTP), \
             patch("jarvis_inference.llama_engine.subprocess.Popen", return_value=process) as popen:
            engine = LlamaServerEngine(manifest())
            engine.start(time.monotonic() + 1)
        verify.assert_called_once()
        command = popen.call_args.args[0]; environment = popen.call_args.kwargs["env"]
        for flag in REQUIRED_FLAGS:
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "43123")
        self.assertNotIn(engine.secret, command)
        self.assertEqual(environment["LLAMA_API_KEY"], engine.secret)
        self.assertNotIn("JARVIS_INFERENCE_API_KEY", environment)
        self.assertNotIn("PATH", environment)
        self.assertEqual(popen.call_args.kwargs["cwd"], "/runtime")

    def test_windows_runtime_temp_closes_only_after_job_tree_drains(self) -> None:
        events = []

        class TreeProcess(FakeProcess):
            def terminate(self):
                events.append("terminate")
                super().terminate()

            def close_tree(self, timeout):
                self.assert_timeout = timeout
                events.append("tree-drained")

        class TempOwner:
            def close(self):
                self.assert_tree_drained = "tree-drained" in events
                events.append("temp-closed")

        process = TreeProcess()
        owner = TempOwner()
        engine = LlamaServerEngine(manifest())
        engine.process = process
        engine._runtime_temp = owner
        engine.phase = "idle"
        engine._terminate("test")
        self.assertEqual(events, ["terminate", "tree-drained", "temp-closed"])
        self.assertTrue(owner.assert_tree_drained)

    def test_failed_tree_drain_retains_process_and_temp_for_retry(self) -> None:
        events = []

        class RetryTreeProcess(FakeProcess):
            def close_tree(self, timeout):
                events.append("drain-attempt")
                if events.count("drain-attempt") == 1:
                    raise WindowsProcessFailure("injected drain race")
                events.append("tree-drained")

        class TempOwner:
            def close(self):
                events.append("temp-closed")

        process = RetryTreeProcess()
        owner = TempOwner()
        engine = LlamaServerEngine(manifest())
        engine.process = process
        engine._runtime_temp = owner
        engine._process_tree_drained = False
        engine.phase = "idle"

        with self.assertRaises(WindowsProcessFailure):
            engine._terminate("first")
        self.assertIs(engine.process, process)
        self.assertIs(engine._runtime_temp, owner)
        self.assertNotIn("temp-closed", events)

        engine._terminate("retry")
        self.assertIsNone(engine.process)
        self.assertIsNone(engine._runtime_temp)
        self.assertEqual(
            events,
            ["drain-attempt", "drain-attempt", "tree-drained", "temp-closed"],
        )

    def test_generation_consumes_preflight_token_ids(self) -> None:
        request = normalize_request(
            {"model": "jarvis-local", "messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 4},
            alias="jarvis-local", capabilities=Capabilities(), context_window=32,
        )
        prepared = PreparedPrompt((10, 11, 12), "must-not-be-retokenized", request)
        engine = LlamaServerEngine(manifest()); engine.process = FakeProcess(); engine.http = object(); engine.port = 43123; engine.secret = "secret"
        stream = FakeStream()
        with patch("jarvis_inference.llama_engine.urllib.request.urlopen", return_value=stream) as opened:
            events = list(engine.generate(prepared, threading.Event(), time.monotonic() + 2))
        payload = parse_body(opened.call_args.args[0].data)
        self.assertEqual(payload["prompt"], [10, 11, 12])
        self.assertIs(payload["cache_prompt"], False)
        self.assertIs(payload["return_tokens"], True)
        self.assertEqual([(event.kind, event.text, event.finish_reason, event.completion_tokens) for event in events], [
            ("text", "hi", None, None), ("final", "", "stop", 2),
        ])
        stream = FakeStream()
        stream.lines = iter([b"data: " + b"[" * 2000 + b"0" + b"]" * 2000 + b"\n"])
        with patch("jarvis_inference.llama_engine.urllib.request.urlopen", return_value=stream):
            with self.assertRaises(EngineProtocolFailure):
                list(engine.generate(prepared, threading.Event(), time.monotonic() + 2))

    def test_qualification_metric_is_serial_bound_content_free_and_one_shot(self) -> None:
        request = normalize_request(
            {"model": "jarvis-local", "messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 4},
            alias="jarvis-local", capabilities=Capabilities(), context_window=32,
        )
        prepared = PreparedPrompt((10, 11, 12), "sensitive prompt", request)
        engine = LlamaServerEngine(manifest())
        engine.process = FakeProcess(); engine.http = object(); engine.port = 43123; engine.secret = "secret"

        class TimedStream(FakeStream):
            def __init__(self):
                self.closed = False
                self.lines = iter([
                    b'data: {"content":"sensitive output","stop":false,"tokens":[101]}\n',
                    b'data: {"content":"","stop":true,"tokens":[2],"tokens_predicted":2,"stop_type":"eos","timings":{"prompt_n":3,"prompt_ms":250.0,"predicted_n":2,"predicted_ms":500.0}}\n',
                ])

        serial = engine.arm_qualification_generation()
        self.assertEqual(serial, 1)
        # The bound gateway timestamp precedes prepare/tokenize; the native
        # first-token clock therefore includes that delay in request TTFT.
        engine.bind_qualification_request_start(9.0)
        clock_values = iter((10.5, 11.0))
        engine._qualification_clock = lambda: next(clock_values)
        with patch("jarvis_inference.llama_engine.urllib.request.urlopen", return_value=TimedStream()):
            list(engine.generate(prepared, threading.Event(), time.monotonic() + 2))
        with self.assertRaisesRegex(EngineProtocolFailure, "does not match"):
            engine.consume_qualification_generation_metric(2)
        metric = engine.consume_qualification_generation_metric(1)
        self.assertEqual(
            set(metric),
            {
                "generation_serial", "prompt_token_count", "prompt_token_ids_sha256",
                "accepted_token_count", "accepted_token_ids_sha256",
                "request_ttft_seconds", "backend_prompt_seconds",
                "backend_decode_seconds", "backend_decode_tokens_per_second",
                "request_to_engine_final_seconds",
            },
        )
        self.assertEqual(metric["generation_serial"], 1)
        self.assertEqual(metric["prompt_token_count"], 3)
        self.assertEqual(
            metric["prompt_token_ids_sha256"],
            hashlib.sha256(canonical_bytes([10, 11, 12])).hexdigest(),
        )
        self.assertEqual(metric["accepted_token_count"], 2)
        self.assertEqual(
            metric["accepted_token_ids_sha256"],
            hashlib.sha256(canonical_bytes([101, 2])).hexdigest(),
        )
        self.assertEqual(metric["request_ttft_seconds"], 1.5)
        self.assertEqual(metric["backend_prompt_seconds"], 0.25)
        self.assertEqual(metric["backend_decode_seconds"], 0.5)
        self.assertEqual(metric["backend_decode_tokens_per_second"], 4.0)
        self.assertEqual(metric["request_to_engine_final_seconds"], 2.0)
        serialized = canonical_bytes(metric).decode("ascii")
        self.assertNotIn("sensitive", serialized)
        self.assertNotIn("prompt\"", serialized)
        self.assertNotIn("output", serialized)
        self.assertNotIn("token_ids\"", serialized)
        with self.assertRaisesRegex(EngineProtocolFailure, "already consumed"):
            engine.consume_qualification_generation_metric(1)

    def test_qualification_metric_rejects_malformed_final_timings(self) -> None:
        request = normalize_request(
            {"model": "jarvis-local", "messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 4},
            alias="jarvis-local", capabilities=Capabilities(), context_window=32,
        )
        prepared = PreparedPrompt((10, 11, 12), "sensitive prompt", request)
        engine = LlamaServerEngine(manifest())
        engine.process = FakeProcess(); engine.http = object(); engine.port = 43123; engine.secret = "secret"

        class MalformedTimedStream(FakeStream):
            def __init__(self):
                self.closed = False
                self.lines = iter([
                    b'data: {"content":"x","stop":false,"tokens":[101]}\n',
                    b'data: {"content":"","stop":true,"tokens":[2],"tokens_predicted":2,"stop_type":"eos","timings":{"prompt_n":3,"prompt_ms":250.0,"predicted_n":1,"predicted_ms":500.0}}\n',
                ])

        serial = engine.arm_qualification_generation()
        engine.bind_qualification_request_start(9.0)
        clock_values = iter((10.5, 11.0))
        engine._qualification_clock = lambda: next(clock_values)
        with patch("jarvis_inference.llama_engine.urllib.request.urlopen", return_value=MalformedTimedStream()):
            with self.assertRaisesRegex(EngineProtocolFailure, "timings"):
                list(engine.generate(prepared, threading.Event(), time.monotonic() + 2))
        with self.assertRaisesRegex(EngineProtocolFailure, "not complete"):
            engine.consume_qualification_generation_metric(serial)
        with self.assertRaisesRegex(EngineProtocolFailure, "already pending"):
            engine.arm_qualification_generation()

    def test_armed_qualification_generation_requires_gateway_request_start(self) -> None:
        request = normalize_request(
            {"model": "jarvis-local", "messages": [{"role": "user", "content": "x"}]},
            alias="jarvis-local", capabilities=Capabilities(), context_window=32,
        )
        engine = LlamaServerEngine(manifest())
        engine.process = FakeProcess(); engine.http = object()
        engine.arm_qualification_generation()
        with self.assertRaisesRegex(EngineProtocolFailure, "state"):
            list(
                engine.generate(
                    PreparedPrompt((10,), "x", request),
                    threading.Event(),
                    time.monotonic() + 1,
                )
            )

    def test_gemma_profile_owns_bos_and_native_tool_parser(self) -> None:
        class TokenizeHTTP:
            def __init__(self): self.value = None
            def request(self, method, path, value, deadline):
                self.value = value
                return {"tokens": [10, 11]}

        definition = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
        request = normalize_request(
            {
                "model": "jarvis-local",
                "messages": [{"role": "user", "content": "weather"}],
                "max_completion_tokens": 8,
                "tools": [definition],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            },
            alias="jarvis-local",
            capabilities=Capabilities(tools=True),
            context_window=32,
        )
        engine = LlamaServerEngine(manifest(GEMMA4_PROMPT_PROFILE))
        engine.process = FakeProcess(); engine.http = TokenizeHTTP(); engine.port = 43123; engine.secret = "secret"
        prepared = engine.prepare(request, time.monotonic() + 1)
        self.assertTrue(prepared.rendered_prompt.startswith("<bos>"))
        self.assertIs(engine.http.value["add_special"], False)
        self.assertIs(engine.http.value["parse_special"], True)

        class ToolStream(FakeStream):
            def __init__(self):
                self.closed = False
                self.lines = iter([
                    b'data: {"content":"<|tool_call>call:get_weather{city:<|\\\"|>Athens<|\\\"|>}<tool_call|>","stop":false,"tokens":[101]}\n',
                    b'data: {"content":"","stop":true,"tokens":[2],"tokens_predicted":2,"stop_type":"eos"}\n',
                ])

        with patch("jarvis_inference.llama_engine.urllib.request.urlopen", return_value=ToolStream()) as opened:
            events = list(engine.generate(prepared, threading.Event(), time.monotonic() + 2))
        payload = parse_body(opened.call_args.args[0].data)
        self.assertNotIn("json_schema", payload)
        self.assertEqual(
            [(event.kind, event.tool_name, event.tool_arguments, event.finish_reason) for event in events],
            [("tool", "get_weather", '{"city":"Athens"}', None), ("final", None, None, "tool_calls")],
        )

    def test_legacy_profile_retains_backend_added_special_tokens(self) -> None:
        class TokenizeHTTP:
            def __init__(self): self.value = None
            def request(self, method, path, value, deadline):
                self.value = value
                return {"tokens": [1]}

        request = normalize_request(
            {"model": "jarvis-local", "messages": [{"role": "user", "content": "x"}]},
            alias="jarvis-local", capabilities=Capabilities(), context_window=32,
        )
        engine = LlamaServerEngine(manifest())
        engine.process = FakeProcess(); engine.http = TokenizeHTTP()
        engine.prepare(request, time.monotonic() + 1)
        self.assertIs(engine.http.value["add_special"], True)

    def test_cleanup_erases_only_slot_and_verifies_idle(self) -> None:
        class SlotsHTTP:
            def __init__(self): self.paths = []
            def request(self, method, path, value, deadline):
                self.paths.append((method, path))
                return {} if method == "POST" else [{"is_processing": False}]

        engine = LlamaServerEngine(manifest()); engine.http = SlotsHTTP()
        engine.cleanup(time.monotonic() + 1)
        self.assertEqual(engine.http.paths, [("POST", "/slots/0?action=erase"), ("GET", "/slots")])
        self.assertEqual(engine.runtime_snapshot()["phase"], "idle")

    def test_native_qualification_phase_and_crash_are_observable(self) -> None:
        process = FakeProcess()
        engine = LlamaServerEngine(manifest())
        engine.process = process
        engine.http = object()
        engine.phase = "stream"

        result = engine.inject_crash()

        self.assertEqual(
            result,
            {
                "pid": 43210,
                "phase": "stream",
                "elapsed_seconds": engine.last_termination_elapsed,
                "exit_ok": True,
            },
        )
        snapshot = engine.runtime_snapshot()
        self.assertEqual(snapshot["phase"], "stopped")
        self.assertEqual(snapshot["termination_serial"], 1)
        self.assertEqual(snapshot["last_termination_kind"], "crash")
        self.assertEqual(snapshot["last_termination_phase"], "stream")
        self.assertTrue(snapshot["last_termination_exit_ok"])

    def test_native_qualification_cancel_forces_bounded_exit(self) -> None:
        process = FakeProcess()
        engine = LlamaServerEngine(manifest())
        engine.process = process
        engine.http = object()
        engine.phase = "prefill"

        result = engine.inject_cancel()

        self.assertEqual(result["pid"], 43210)
        self.assertEqual(result["phase"], "prefill")
        self.assertTrue(result["exit_ok"])
        self.assertTrue(engine.restart_recorded)
        self.assertEqual(engine.runtime_snapshot()["last_termination_kind"], "cancel")


if __name__ == "__main__":
    unittest.main()
