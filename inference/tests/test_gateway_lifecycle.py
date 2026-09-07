from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_qualification import TEST_HARDWARE_ENV, passing_report

from jarvis_inference.durable import durable_replace
from jarvis_inference.engine import EngineFailure, GenerationEvent, PreparedPrompt
from jarvis_inference.gateway import (
    EventQueue,
    GatewayServer,
    Production,
    effective_configuration,
    package_digest,
)
from jarvis_inference.gemma4 import (
    GEMMA4_PROMPT_PROFILE,
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
)
from jarvis_inference.manifest import load_manifest, runtime_identity
from jarvis_inference.qualification import _QualificationProduction


def artifact(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def make_manifest(root: Path, capabilities: dict | None = None):
    executable = root / "llama-server.exe"; executable.write_bytes(b"exe")
    selected_capabilities = capabilities or {"text": True, "stream": True, "tools": False, "json_object": False, "json_schema": False}
    structured = any(selected_capabilities[name] for name in ("tools", "json_object", "json_schema"))
    model = root / (MODEL_FILENAME if structured else "model.gguf"); model.write_bytes(b"model")
    model_artifact = (
        {"path": MODEL_FILENAME, "size_bytes": MODEL_SIZE_BYTES, "sha256": MODEL_SHA256}
        if structured else artifact(model, root)
    )
    value = {
        "manifest_schema_version": 1, "alias": "jarvis-local", "platform": "windows-x86_64",
        "backend": {"type": "llama-server", "llama_cpp_commit": "a" * 40, "build_id": "fake", "executable": artifact(executable, root), "dlls": [], "parameters": {}},
        "model": {"format": "gguf", "context_window": 16, "artifacts": [model_artifact]},
        "python": {},
        "capabilities": selected_capabilities,
        "qualification_report": "qualification-report.json",
    }
    if structured:
        value["prompt_profile"] = GEMMA4_PROMPT_PROFILE
        value["profile_qualification_report"] = "qualification-profile-local-primary.json"
    path = root / "manifest.json"; path.write_text(json.dumps(value), encoding="utf-8")
    manifest = load_manifest(path)
    runner = root / "fixture-runner"; runner.write_bytes(b"runner")
    durable_replace(manifest.qualification_report, passing_report(manifest, runtime_identity(manifest, package_digest(), effective_configuration()), artifact(runner, root)))
    return manifest


class ConfigurableEngine:
    def __init__(self, state: dict):
        self.state = state
        self.open = True

    def start(self, deadline=None):
        self.open = True

    def ready(self):
        return self.open

    def prepare(self, request, deadline):
        self.state["prepare_calls"] = self.state.get("prepare_calls", 0) + 1
        if self.state.get("fail_prepares", 0):
            self.state["fail_prepares"] -= 1
            raise EngineFailure("injected prefill failure")
        return PreparedPrompt(tuple(range(self.state.get("prompt_tokens", 3))), "prompt", request)

    def generate(self, prompt, cancel, deadline):
        entered = self.state.get("entered")
        if entered is not None:
            entered.set()
        release = self.state.get("release")
        if release is not None:
            release.wait(2)
        for event in self.state.get("events", [GenerationEvent("text", text="ok"), GenerationEvent("final", finish_reason="stop", completion_tokens=1)]):
            if isinstance(event, BaseException):
                raise event
            yield event

    def cleanup(self, deadline):
        self.state["cleanup_calls"] = self.state.get("cleanup_calls", 0) + 1
        if self.state.get("cleanup_failure"):
            raise EngineFailure("injected cleanup failure")

    def cancel(self):
        self.state["cancel_calls"] = self.state.get("cancel_calls", 0) + 1
        if self.state.get("cancel_failure"):
            raise EngineFailure("injected cancellation failure")

    def close(self):
        self.open = False


class Harness:
    def __init__(self, state: dict):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = state
        self.manifest = make_manifest(self.root, self.state.get("capabilities"))
        self.state["factory_calls"] = 0

        def factory(_manifest):
            self.state["factory_calls"] += 1
            return ConfigurableEngine(self.state)

        # Lifecycle tests isolate transport/engine behavior. Exact primary
        # profile-report admission has its own strict test module.
        self.production = Production(
            self.manifest,
            "k" * 40,
            factory,
            profile_validator=lambda _manifest, _identity, _report: True,
        )
        self.server = GatewayServer(self.production, ("127.0.0.1", 0))
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        stop = time.monotonic() + 3
        while not self.production.is_ready() and time.monotonic() < stop:
            time.sleep(0.01)
        if not self.production.is_ready():
            raise RuntimeError("harness failed to become ready")

    def request(self, *, stream: bool = False, maximum: int = 2):
        document = self.state.get("request_document") or {
            "model": "jarvis-local", "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": maximum, "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        }
        body = json.dumps(document).encode()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=4)
        connection.request("POST", "/v1/chat/completions", body, {
            "Authorization": "Bearer " + "k" * 40, "Content-Type": "application/json",
        })
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        return response, raw

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.production.close(); self.thread.join(timeout=2)
        self.temporary.cleanup()


class GatewayLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, TEST_HARDWARE_ENV))

    def test_supervisor_failure_revokes_readiness(self) -> None:
        production = Production(None, "test", None)
        production.engine = ConfigurableEngine({})
        production.ready_state = True
        failure = OSError("qualification artifact unavailable")
        with patch.object(Production, "_supervise_loop", side_effect=failure):
            production._supervise()
        self.assertFalse(production.is_ready())
        self.assertIs(production.loader_failure, failure)
        self.assertTrue(production.loader_exited.is_set())

    def test_rendered_context_overflow_is_client_error(self) -> None:
        harness = Harness({"prompt_tokens": 15})
        try:
            response, raw = harness.request(maximum=2)
            self.assertEqual(response.status, 400)
            self.assertEqual(json.loads(raw)["error"]["code"], "context_length_exceeded")
        finally:
            harness.close()

    def test_nonstream_cleanup_failure_never_returns_success(self) -> None:
        harness = Harness({"cleanup_failure": True})
        try:
            response, raw = harness.request()
            self.assertEqual(response.status, 502)
            self.assertEqual(json.loads(raw)["error"]["code"], "backend_protocol_error")
        finally:
            harness.close()

    def test_post_output_failure_aborts_without_usage_or_done_and_no_retry(self) -> None:
        harness = Harness({"events": [GenerationEvent("text", text="visible"), EngineFailure("after output")]})
        try:
            response, raw = harness.request(stream=True)
            self.assertEqual(response.status, 200)
            self.assertIn(b'"visible"', raw)
            self.assertNotIn(b'"usage":{', raw)
            self.assertNotIn(b"[DONE]", raw)
            self.assertEqual(harness.state["factory_calls"], 1)
            self.assertEqual(harness.state.get("cleanup_calls"), 1)
        finally:
            harness.close()

    def test_failed_stream_cancellation_cannot_restart_the_response(self) -> None:
        harness = Harness({
            "events": [GenerationEvent("text", text="visible"), EngineFailure("after output")],
            "cancel_failure": True,
        })
        try:
            with patch.object(Production, "restart_engine") as restart:
                response, raw = harness.request(stream=True)
            self.assertEqual(response.status, 200)
            self.assertNotIn(b"HTTP/1.1", raw)
            self.assertNotIn(b"[DONE]", raw)
            restart.assert_not_called()
        finally:
            harness.close()

    def test_stream_without_usage_still_validates_token_limit(self) -> None:
        harness = Harness({
            "request_document": {
                "model": "jarvis-local", "messages": [{"role": "user", "content": "x"}],
                "stream": True, "max_completion_tokens": 2,
            },
            "events": [GenerationEvent("text", text="visible"), GenerationEvent("final", finish_reason="stop", completion_tokens=3)],
        })
        try:
            response, raw = harness.request(stream=True)
            self.assertEqual(response.status, 200)
            self.assertNotIn(b"[DONE]", raw)
            self.assertNotIn(b'"finish_reason":"stop"', raw)
        finally:
            harness.close()

    def test_unexpected_backend_failure_returns_typed_error_and_cleans_up(self) -> None:
        for failure in (TypeError("malformed backend value"), OSError("pipe failed")):
            with self.subTest(failure=type(failure).__name__):
                harness = Harness({"events": [failure]})
                try:
                    response, raw = harness.request()
                    self.assertEqual(response.status, 502)
                    self.assertEqual(json.loads(raw)["error"]["code"], "backend_protocol_error")
                    self.assertEqual(harness.state["factory_calls"], 1)
                    self.assertEqual(harness.state.get("cleanup_calls"), 1)
                finally:
                    harness.close()

    def test_preoutput_retry_is_exactly_once(self) -> None:
        harness = Harness({"fail_prepares": 1})
        try:
            response, _ = harness.request()
            self.assertEqual(response.status, 200)
            self.assertEqual(harness.state["factory_calls"], 2)
            self.assertEqual(harness.state["prepare_calls"], 2)
        finally:
            harness.close()

    def test_one_active_request_returns_429_without_queue(self) -> None:
        entered = threading.Event(); release = threading.Event()
        harness = Harness({"entered": entered, "release": release})
        result: list[tuple[object, bytes]] = []
        worker = threading.Thread(target=lambda: result.append(harness.request()), daemon=True)
        try:
            worker.start(); self.assertTrue(entered.wait(2))
            response, raw = harness.request()
            self.assertEqual(response.status, 429)
            self.assertEqual(response.getheader("Retry-After"), "1")
            self.assertEqual(json.loads(raw)["error"]["code"], "server_busy")
            release.set(); worker.join(timeout=3)
            self.assertFalse(worker.is_alive())
        finally:
            release.set(); harness.close()

    def test_terminal_event_is_unique_and_last(self) -> None:
        harness = Harness({"events": [
            GenerationEvent("final", finish_reason="stop", completion_tokens=0),
            GenerationEvent("text", text="late"),
        ]})
        try:
            response, raw = harness.request()
            self.assertEqual(response.status, 502)
            self.assertEqual(json.loads(raw)["error"]["code"], "backend_protocol_error")
        finally:
            harness.close()

    def test_streamed_tool_call_has_one_canonical_buffered_delta(self) -> None:
        schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}
        state = {
            "capabilities": {"text": True, "stream": True, "tools": True, "json_object": False, "json_schema": False},
            "request_document": {
                "model": "jarvis-local", "messages": [{"role": "user", "content": "weather"}],
                "max_completion_tokens": 8, "stream": True, "stream_options": {"include_usage": True},
                "tools": [{"type": "function", "function": {"name": "weather", "parameters": schema, "strict": True}}],
                "tool_choice": "required",
            },
            "events": [
                GenerationEvent("tool", tool_name="weather", tool_arguments='{ "city" : "Athens" }'),
                GenerationEvent("final", finish_reason="tool_calls", completion_tokens=4),
            ],
        }
        harness = Harness(state)
        try:
            response, raw = harness.request(stream=True)
            self.assertEqual(response.status, 200)
            frames = raw.decode().strip().split("\n\n")
            self.assertEqual(frames[-1], "data: [DONE]")
            values = [json.loads(frame[6:]) for frame in frames[:-1]]
            self.assertEqual(values[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
            calls = [value for value in values if value["choices"] and "tool_calls" in value["choices"][0]["delta"]]
            self.assertEqual(len(calls), 1)
            call = calls[0]["choices"][0]["delta"]["tool_calls"][0]
            self.assertRegex(call["id"], r"^call_[0-9a-f]{32}$")
            self.assertEqual(call["function"], {"name": "weather", "arguments": '{"city":"Athens"}'})
            self.assertEqual(values[-2]["choices"][0]["finish_reason"], "tool_calls")
            self.assertEqual(values[-1]["choices"], [])
        finally:
            harness.close()

    def test_outbound_queue_is_bounded_by_event_size(self) -> None:
        queue = EventQueue(); cancel = threading.Event()
        queue.put(GenerationEvent("text", text="x" * 1_100_000), cancel, time.monotonic() + 1)
        value = queue.get(0.1)
        self.assertIsInstance(value, BaseException)

    def test_disconnect_watcher_cancels_without_waiting_for_write(self) -> None:
        entered = threading.Event(); release = threading.Event()
        harness = Harness({"entered": entered, "release": release})
        client = socket.socket()
        try:
            body = b'{"model":"jarvis-local","messages":[{"role":"user","content":"x"}],"max_completion_tokens":2}'
            request = (
                f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{harness.port}\r\n"
                f"Authorization: Bearer {'k' * 40}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n\r\n"
            ).encode() + body
            client.connect(("127.0.0.1", harness.port)); client.sendall(request)
            self.assertTrue(entered.wait(2)); client.close()
            stop = time.monotonic() + 2
            while harness.state.get("cancel_calls", 0) == 0 and time.monotonic() < stop:
                time.sleep(0.01)
            self.assertGreaterEqual(harness.state.get("cancel_calls", 0), 1)
            self.assertEqual(harness.state["factory_calls"], 1)
        finally:
            release.set(); client.close(); harness.close()

    def test_readyz_does_not_block_on_engine_load(self) -> None:
        temporary = tempfile.TemporaryDirectory(); root = Path(temporary.name)
        manifest = make_manifest(root)
        gate = threading.Event()

        class SlowStart(ConfigurableEngine):
            def start(self, deadline=None):
                gate.wait(2); self.open = True

        production = Production(manifest, "k" * 40, lambda _: SlowStart({}))
        server = GatewayServer(production, ("127.0.0.1", 0)); port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True); thread.start()
        try:
            started = time.monotonic()
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            connection.request("GET", "/readyz")
            response = connection.getresponse(); raw = response.read(); connection.close()
            elapsed = time.monotonic() - started
            self.assertEqual((response.status, json.loads(raw)), (503, {"ready": False}))
            self.assertLess(elapsed, 0.5)
            gate.set()
            stop = time.monotonic() + 2
            while not production.is_ready() and time.monotonic() < stop:
                time.sleep(0.01)
            self.assertTrue(production.is_ready())
        finally:
            gate.set(); server.shutdown(); server.server_close(); production.close(); thread.join(timeout=2); temporary.cleanup()

    def test_in_progress_report_is_unready_and_does_not_spawn(self) -> None:
        temporary = tempfile.TemporaryDirectory(); root = Path(temporary.name)
        manifest = make_manifest(root)
        durable_replace(manifest.qualification_report, {"report_schema_version": 1, "status": "in_progress", "identity": runtime_identity(manifest, package_digest(), effective_configuration())})
        calls = []
        production = Production(manifest, "k" * 40, lambda _: calls.append(1))
        server = GatewayServer(production, ("127.0.0.1", 0)); port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True); thread.start()
        try:
            time.sleep(0.05)
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            connection.request("GET", "/readyz"); response = connection.getresponse(); raw = response.read(); connection.close()
            self.assertEqual((response.status, json.loads(raw)), (503, {"ready": False}))
            self.assertEqual(calls, [])
        finally:
            server.shutdown(); server.server_close(); production.close(); thread.join(timeout=2); temporary.cleanup()

    def test_changed_fixture_runner_invalidates_existing_pass(self) -> None:
        harness = Harness({})
        try:
            (harness.root / "fixture-runner").write_bytes(b"changed")
            stop = time.monotonic() + 2
            while harness.production.is_ready() and time.monotonic() < stop:
                time.sleep(0.02)
            self.assertFalse(harness.production.is_ready())
        finally:
            harness.close()

    def test_shutdown_cancels_engine_loading(self) -> None:
        temporary = tempfile.TemporaryDirectory(); root = Path(temporary.name)
        manifest = make_manifest(root); entered = threading.Event(); stopped = threading.Event()

        class BlockingLoad(ConfigurableEngine):
            def start(self, deadline=None):
                entered.set(); stopped.wait(5)
                if stopped.is_set(): raise EngineFailure("load cancelled")
            def close(self):
                self.open = False; stopped.set()

        production = Production(manifest, "k" * 40, lambda _: BlockingLoad({}))
        server = GatewayServer(production, ("127.0.0.1", 0))
        try:
            self.assertTrue(entered.wait(2))
            started = time.monotonic(); production.close()
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertTrue(stopped.is_set())
        finally:
            server.server_close(); production.close(); temporary.cleanup()

    def test_supervisor_and_request_restart_never_load_engines_concurrently(self) -> None:
        self._assert_serialized_engine_loads(Production)

    def test_qualification_supervisor_serializes_request_restarts(self) -> None:
        self._assert_serialized_engine_loads(_QualificationProduction)

    def _assert_serialized_engine_loads(self, production_type) -> None:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        manifest = make_manifest(root)
        entered = threading.Event()
        release = threading.Event()
        counts_lock = threading.Lock()
        counts = {"active": 0, "maximum": 0, "factory": 0}

        class SerializedStart(ConfigurableEngine):
            def start(self, deadline=None):
                with counts_lock:
                    counts["active"] += 1
                    counts["maximum"] = max(counts["maximum"], counts["active"])
                entered.set()
                release.wait(2)
                with counts_lock:
                    counts["active"] -= 1
                self.open = True

        def factory(_manifest):
            with counts_lock:
                counts["factory"] += 1
            return SerializedStart({})

        production = production_type(
            manifest,
            "k" * 40,
            factory,
            profile_validator=lambda _manifest, _identity, _report: True,
        )
        failures: list[BaseException] = []
        restart = threading.Thread(
            target=lambda: self._capture_restart(production, failures),
            daemon=True,
        )
        try:
            production.start_background()
            self.assertTrue(entered.wait(2))
            restart.start()
            time.sleep(0.1)
            with counts_lock:
                self.assertEqual(counts["factory"], 1)
                self.assertEqual(counts["maximum"], 1)
            release.set()
            restart.join(timeout=3)
            self.assertFalse(restart.is_alive())
            self.assertEqual(failures, [])
            with counts_lock:
                self.assertEqual(counts["maximum"], 1)
        finally:
            release.set()
            restart.join(timeout=3)
            production.close()
            temporary.cleanup()

    @staticmethod
    def _capture_restart(
        production: Production,
        failures: list[BaseException],
    ) -> None:
        try:
            production.restart_engine(time.monotonic() + 2)
        except BaseException as exc:  # noqa: BLE001 - asserted by the owner thread
            failures.append(exc)


if __name__ == "__main__":
    unittest.main()
