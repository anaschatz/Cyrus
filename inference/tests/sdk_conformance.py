"""Official OpenAI Python SDK conformance against an ephemeral local gateway.

Run in the pinned qualification environment; this file is intentionally
separate from dependency-free unit discovery.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

from jarvis_inference.durable import durable_replace
from jarvis_inference.engine import GenerationEvent, PreparedPrompt
from jarvis_inference.gateway import GatewayServer, Production, effective_configuration, package_digest
from jarvis_inference.manifest import load_manifest, runtime_identity
from jarvis_inference.rendering import render_prompt


def artifact(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class Engine:
    def __init__(self, manifest): self.open = True
    def start(self, deadline=None): self.open = True
    def ready(self): return self.open
    def prepare(self, request, deadline): return PreparedPrompt((1, 2, 3), render_prompt(request), request)
    def generate(self, prompt, cancel, deadline):
        yield GenerationEvent("text", text="ok")
        yield GenerationEvent("final", finish_reason="stop", completion_tokens=1)
    def cleanup(self, deadline): return None
    def cancel(self): return None
    def close(self): self.open = False


class OfficialSdkConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(); root = Path(cls.temporary.name)
        exe = root / "server.exe"; exe.write_bytes(b"x")
        model = root / "model.gguf"; model.write_bytes(b"x")
        value = {
            "manifest_schema_version": 1, "alias": "jarvis-local", "platform": "windows-x86_64",
            "backend": {"type": "llama-server", "llama_cpp_commit": "a" * 40, "build_id": "sdk", "executable": artifact(exe, root), "dlls": [], "parameters": {}},
            "model": {"format": "gguf", "context_window": 4096, "artifacts": [artifact(model, root)]},
            "python": {}, "capabilities": {"text": True, "stream": True, "tools": False, "json_object": False, "json_schema": False},
            "qualification_report": "report.json",
        }
        path = root / "manifest.json"; path.write_text(json.dumps(value))
        manifest = load_manifest(path)
        runner = root / "fixture-runner"; runner.write_bytes(b"runner")
        durable_replace(manifest.qualification_report, {
            "report_schema_version": 1, "status": "pass",
            "identity": runtime_identity(manifest, package_digest(), effective_configuration()),
            "fixture_runner": artifact(runner, root),
        })
        cls.production = Production(manifest, "sdk-secret-" + "x" * 32, Engine)
        cls.server = GatewayServer(cls.production, ("127.0.0.1", 0)); cls.port = int(cls.server.server_address[1])
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True); cls.thread.start()
        stop = time.monotonic() + 3
        while not cls.production.is_ready() and time.monotonic() < stop: time.sleep(0.01)
        if not cls.production.is_ready(): raise RuntimeError("SDK gateway did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.production.close(); cls.thread.join(timeout=2); cls.temporary.cleanup()

    def test_sync_nonstream_and_stream(self):
        client = OpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="sdk-secret-" + "x" * 32, max_retries=0, timeout=5)
        request = {"model": "jarvis-local", "messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 4}
        try:
            response = client.chat.completions.create(**request)
            self.assertEqual((response.model, response.choices[0].message.content), ("jarvis-local", "ok"))
            chunks = list(client.chat.completions.create(**request, stream=True, stream_options={"include_usage": True}))
            self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
            self.assertEqual(chunks[-1].usage.total_tokens, 4)
        finally:
            client.close()

    def test_async_nonstream_and_stream(self):
        async def check():
            client = AsyncOpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="sdk-secret-" + "x" * 32, max_retries=0, timeout=5)
            request = {"model": "jarvis-local", "messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 4}
            try:
                response = await client.chat.completions.create(**request)
                self.assertEqual(response.model, "jarvis-local")
                stream = await client.chat.completions.create(**request, stream=True, stream_options={"include_usage": True})
                chunks = [chunk async for chunk in stream]
                self.assertEqual(chunks[-1].usage.total_tokens, 4)
            finally:
                await client.close()
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
