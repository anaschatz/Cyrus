from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_inference.contracts import Capabilities, normalize_request
from jarvis_inference.engine import PreparedPrompt
from jarvis_inference.llama_engine import LlamaServerEngine, REQUIRED_FLAGS
from jarvis_inference.strict_json import parse_body


class FakeProcess:
    def __init__(self):
        self.terminated = False

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


def manifest():
    return SimpleNamespace(
        root=Path("/runtime"), context_window=32,
        backend={"executable": {"path": "runtime/llama-server.exe"}, "parameters": {}},
        model={"artifacts": [{"path": "models/model.gguf"}]},
    )


class LlamaEngineTests(unittest.TestCase):
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

    def test_cleanup_erases_only_slot_and_verifies_idle(self) -> None:
        class SlotsHTTP:
            def __init__(self): self.paths = []
            def request(self, method, path, value, deadline):
                self.paths.append((method, path))
                return {} if method == "POST" else [{"is_processing": False}]

        engine = LlamaServerEngine(manifest()); engine.http = SlotsHTTP()
        engine.cleanup(time.monotonic() + 1)
        self.assertEqual(engine.http.paths, [("POST", "/slots/0?action=erase"), ("GET", "/slots")])


if __name__ == "__main__":
    unittest.main()
