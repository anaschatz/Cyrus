from __future__ import annotations

import hashlib
import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from jarvis_inference.durable import durable_replace
from jarvis_inference.engine import GenerationEvent, PreparedPrompt
from jarvis_inference.gateway import GatewayServer, Production, effective_configuration, package_digest
from jarvis_inference.manifest import load_manifest, runtime_identity
from jarvis_inference.rendering import render_prompt


def artifact(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class FakeEngine:
    def __init__(self, manifest):
        self.open = True

    def start(self, deadline=None):
        self.open = True

    def ready(self):
        return self.open

    def prepare(self, request, deadline):
        prompt = render_prompt(request)
        return PreparedPrompt((11, 12, 13), prompt, request)

    def generate(self, prompt, cancel, deadline):
        yield GenerationEvent("text", text="γειά")
        yield GenerationEvent("final", finish_reason="stop", completion_tokens=2)

    def cleanup(self, deadline):
        return None

    def cancel(self):
        return None

    def close(self):
        self.open = False


class GatewayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        executable = root / "llama-server.exe"; executable.write_bytes(b"exe")
        model = root / "model.gguf"; model.write_bytes(b"model")
        manifest_document = {
            "manifest_schema_version": 1, "alias": "jarvis-local", "platform": "windows-x86_64",
            "backend": {"type": "llama-server", "llama_cpp_commit": "a" * 40, "build_id": "fake", "executable": artifact(executable, root), "dlls": [], "parameters": {}},
            "model": {"format": "gguf", "context_window": 4096, "artifacts": [artifact(model, root)]},
            "python": {},
            "capabilities": {"text": True, "stream": True, "tools": False, "json_object": False, "json_schema": False},
            "qualification_report": "qualification-report.json",
        }
        manifest_path = root / "manifest.json"; manifest_path.write_text(json.dumps(manifest_document), encoding="utf-8")
        cls.manifest = load_manifest(manifest_path)
        identity = runtime_identity(cls.manifest, package_digest(), effective_configuration())
        runner = root / "fixture-runner"; runner.write_bytes(b"runner")
        durable_replace(cls.manifest.qualification_report, {
            "report_schema_version": 1, "status": "pass", "identity": identity,
            "fixture_runner": artifact(runner, root),
        })
        cls.production = Production(cls.manifest, "s" * 40, FakeEngine)
        cls.server = GatewayServer(cls.production, ("127.0.0.1", 0))
        cls.port = int(cls.server.server_address[1])
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        cls.thread.start()
        stop = time.monotonic() + 3
        while not cls.production.is_ready() and time.monotonic() < stop:
            time.sleep(0.01)
        if not cls.production.is_ready():
            raise RuntimeError("test production did not become ready")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "server"):
            cls.server.shutdown(); cls.server.server_close(); cls.production.close(); cls.thread.join(timeout=1)
        cls.temporary.cleanup()

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        return response, raw

    def raw_request(self, headers: list[tuple[str, str]], body: bytes = b"{}"):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.putrequest("POST", "/v1/chat/completions", skip_host=True, skip_accept_encoding=True)
        for name, value in headers:
            connection.putheader(name, value)
        connection.putheader("Content-Length", str(len(body)))
        connection.endheaders(body)
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        return response, raw

    def test_ready_shape_is_content_free(self) -> None:
        response, raw = self.request("GET", "/readyz")
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(raw), {"ready": True})
        self.assertEqual(response.getheader("Cache-Control"), "no-store")

    def test_canonical_nonstream_response(self) -> None:
        body = json.dumps({"model": "jarvis-local", "messages": [{"role": "user", "content": "hello"}], "max_completion_tokens": 8}).encode()
        response, raw = self.request("POST", "/v1/chat/completions", body, {
            "Authorization": "Bearer " + "s" * 40,
            "Content-Type": "application/json; charset=utf-8",
        })
        self.assertEqual(response.status, 200)
        value = json.loads(raw)
        self.assertEqual(set(value), {"id", "object", "created", "model", "choices", "usage"})
        self.assertEqual(value["object"], "chat.completion")
        self.assertRegex(value["id"], r"^chatcmpl-[0-9a-f]{32}$")
        self.assertEqual(value["model"], "jarvis-local")
        self.assertEqual(value["choices"][0]["message"], {"role": "assistant", "content": "γειά"})
        self.assertEqual(set(value["choices"][0]), {"index", "message", "logprobs", "finish_reason"})
        self.assertEqual(value["usage"], {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})

    def test_sse_usage_frame_and_done(self) -> None:
        body = json.dumps({
            "model": "jarvis-local", "messages": [{"role": "user", "content": "hello"}],
            "stream": True, "stream_options": {"include_usage": True},
        }).encode()
        response, raw = self.request("POST", "/v1/chat/completions", body, {
            "Authorization": "Bearer " + "s" * 40, "Content-Type": "application/json",
        })
        self.assertEqual(response.status, 200)
        frames = raw.decode().strip().split("\n\n")
        self.assertEqual(frames[-1], "data: [DONE]")
        values = [json.loads(frame[6:]) for frame in frames[:-1]]
        self.assertEqual(values[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
        self.assertEqual(values[-1]["choices"], [])
        self.assertEqual(values[-1]["usage"]["total_tokens"], 5)
        self.assertTrue(all(value.get("usage") is None for value in values[:-1]))

    def test_auth_and_content_encoding_fail_closed(self) -> None:
        response, _ = self.request("GET", "/v1/models")
        self.assertEqual(response.status, 401)
        body = b'{"model":"jarvis-local","messages":[]}'
        response, raw = self.request("POST", "/v1/chat/completions", body, {
            "Authorization": "Bearer " + "s" * 40,
            "Content-Type": "application/json", "Content-Encoding": "gzip",
        })
        self.assertEqual(response.status, 415)
        self.assertEqual(json.loads(raw)["error"]["code"], "unsupported_content_encoding")

    def test_strict_json_encoding_and_literals(self) -> None:
        cases = [
            (b'{"model":"jarvis-local","model":"x","messages":[]}', "duplicate_json_key"),
            (b'{"model":"jarvis-local","messages":[],"x":{"a":1,"a":2}}', "duplicate_json_key"),
            (b'{"model":"jarvis-local","messages":[],"x":NaN}', "invalid_json"),
            (b"\xef\xbb\xbf{}", "invalid_json_encoding"),
            (b'{"x":"\xff"}', "invalid_json_encoding"),
        ]
        for body, code in cases:
            with self.subTest(code=code):
                response, raw = self.raw_request([
                    ("Host", f"127.0.0.1:{self.port}"),
                    ("Authorization", "Bearer " + "s" * 40),
                    ("Content-Type", "application/json"),
                ], body)
                self.assertEqual(response.status, 400)
                self.assertEqual(json.loads(raw)["error"]["code"], code)

    def test_media_headers_are_exact(self) -> None:
        valid = b'{"model":"jarvis-local","messages":[{"role":"user","content":"x"}]}'
        cases = [
            ([], 415, "unsupported_media_type"),
            ([("Content-Type", "text/json")], 415, "unsupported_media_type"),
            ([("Content-Type", "application/json; charset=ascii")], 415, "unsupported_media_type"),
            ([("Content-Type", "application/json"), ("Content-Type", "application/json")], 415, "unsupported_media_type"),
            ([("Content-Type", "application/json"), ("Content-Encoding", "gzip")], 415, "unsupported_content_encoding"),
            ([("Content-Type", "application/json"), ("Content-Encoding", "gzip, identity")], 415, "unsupported_content_encoding"),
        ]
        for additions, status, code in cases:
            with self.subTest(additions=additions):
                response, raw = self.raw_request([
                    ("Host", f"127.0.0.1:{self.port}"),
                    ("Authorization", "Bearer " + "s" * 40),
                    *additions,
                ], valid)
                self.assertEqual(response.status, status)
                self.assertEqual(json.loads(raw)["error"]["code"], code)

    def test_host_forwarding_and_authorization_are_exact(self) -> None:
        body = b'{"model":"jarvis-local","messages":[{"role":"user","content":"x"}]}'
        base = [("Host", f"127.0.0.1:{self.port}"), ("Content-Type", "application/json")]
        cases = [
            ([("Host", "evil.invalid"), ("Content-Type", "application/json"), ("Authorization", "Bearer " + "s" * 40)], 400, "invalid_host"),
            (base, 401, "invalid_api_key"),
            (base + [("Authorization", "Bearer wrong")], 401, "invalid_api_key"),
            (base + [("Authorization", "Bearer " + "s" * 40), ("Authorization", "Bearer " + "s" * 40)], 401, "invalid_api_key"),
            (base + [("Authorization", "Bearer " + "s" * 40), ("Origin", "http://localhost")], 400, "forwarded_request_forbidden"),
            (base + [("Authorization", "Bearer " + "s" * 40), ("Forwarded", "for=127.0.0.1")], 400, "forwarded_request_forbidden"),
            (base + [("Authorization", "Bearer " + "s" * 40), ("X-Forwarded-Anything", "x")], 400, "forwarded_request_forbidden"),
        ]
        for headers, status, code in cases:
            with self.subTest(code=code):
                response, raw = self.raw_request(headers, body)
                self.assertEqual(response.status, status)
                self.assertEqual(json.loads(raw)["error"]["code"], code)
