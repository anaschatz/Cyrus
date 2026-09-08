from __future__ import annotations

import asyncio
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import _bootstrap  # noqa: F401

from test_cloud_release_qualification import DEEPSEEK_MODEL, endpoint
from jarvis_orchestrator.cloud_contract import build_dispatch_body
from jarvis_orchestrator.cloud_provider import (
    GuardedOpenRouterTransport, OpenRouterCloudAdapter, PhysicalCloudPermit,
    _cloud_permit, load_endpoint_registry, load_openrouter_secret,
)
from jarvis_orchestrator.domain import OrchestratorError
from jarvis_orchestrator.jcs import domain_sha256


class CloudFileBoundaryTests(unittest.TestCase):
    def test_secret_requires_private_regular_file_and_valid_header_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'secret'
            path.write_text('synthetic-token\n')
            path.chmod(0o600)
            self.assertEqual(load_openrouter_secret(path), 'synthetic-token')
            link = root / 'link'
            link.symlink_to(path)
            with self.subTest(kind='symlink'), self.assertRaises(RuntimeError):
                load_openrouter_secret(link)
            for value in ('bad\rtoken', 'bad\ttoken', 'μη-ascii', ' ' * (256 * 1024) + 'token'):
                path.write_text(value)
                with self.subTest(value=value[:20]), self.assertRaises(RuntimeError):
                    load_openrouter_secret(path)

    def test_registry_uses_bounded_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'registry.json'
            valid = '{"version":"endpoint-registry/1","endpoints":[]}'
            path.write_text(valid)
            self.assertEqual(load_endpoint_registry(path), ())
            for raw in (valid[:-1] + ',"endpoints":[]}', valid + ' ' * (1024 * 1024),
                        '{"x":' + '[' * 1500 + '0' + ']' * 1500 + '}'):
                path.write_text(raw)
                with self.subTest(raw=raw[:30]), self.assertRaises(RuntimeError):
                    load_endpoint_registry(path)

    def test_registry_fifo_is_rejected_without_a_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'registry.json'
            os.mkfifo(path, 0o600)
            script = """import sys
from jarvis_orchestrator.cloud_provider import load_endpoint_registry
try:
    load_endpoint_registry(sys.argv[1])
except RuntimeError:
    sys.exit(0)
sys.exit(1)
"""
            result = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True, timeout=2)
            self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_constructor_validates_before_creating_clients(self):
        for field, value in (('bearer_token', True), ('bearer_token', 'bad\r\ntoken'),
                ('bearer_token', 'bad token'), ('timeout_seconds', True), ('timeout_seconds', float('nan')),
                ('timeout_seconds', float('inf')), ('timeout_seconds', 10 ** 1000), ('timeout_seconds', 0)):
            with self.subTest(field=field, value=value), patch('jarvis_orchestrator.cloud_provider.httpx.AsyncClient') as client:
                with self.assertRaises(ValueError):
                    OpenRouterCloudAdapter(Mock(), **(dict(bearer_token='synthetic-key', timeout_seconds=1) | {field: value}))
                client.assert_not_called()


class Chunks(httpx.AsyncByteStream):
    def __init__(self, raw, *, delay=0, chunk_size=16384):
        self.raw, self.delay, self.chunk_size = raw, delay, chunk_size
        self.closed = False
        self.read_bytes = 0

    async def __aiter__(self):
        for offset in range(0, len(self.raw), self.chunk_size):
            if self.delay:
                await asyncio.sleep(self.delay)
            chunk = self.raw[offset:offset + self.chunk_size]
            self.read_bytes += len(chunk)
            yield chunk

    async def aclose(self):
        self.closed = True


class CloudHTTPBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def run_dispatch(self, *, target='post', transform=lambda raw: raw, encoding='identity', delay=0, timeout=1):
        selected = endpoint(DEEPSEEK_MODEL)
        body = build_dispatch_body(selected, messages=({'role': 'user', 'content': 'public fixture'},), output_token_limit=256)
        post = dict(id='gen-fixture', choices=[dict(finish_reason='stop', message=dict(content='synthetic answer'))],
                    usage=dict(prompt_tokens=100, completion_tokens=10))
        receipt = dict(data=dict(id='gen-fixture', model=selected.model_canonical_id,
            provider_name=selected.runtime_provider_name, native_tokens_prompt=100,
            native_tokens_completion=10, native_tokens_reasoning=0, total_cost='0.0001'))
        self.stream = Chunks(transform(json.dumps(post if target == 'post' else receipt).encode()),
                             delay=delay, chunk_size=20 if delay else 16384)
        requests = []
        async def response(request):
            requests.append(request.method)
            if (request.method == 'POST') == (target == 'post'):
                return httpx.Response(200, headers={'content-type': 'application/json', 'content-encoding': encoding}, stream=self.stream)
            return httpx.Response(200, json=post if request.method == 'POST' else receipt)
        store = Mock()
        store.reconcile_cloud_dispatch.return_value = 100
        self.store = store
        adapter = OpenRouterCloudAdapter(store, bearer_token='synthetic-key', timeout_seconds=timeout,
            transport=httpx.MockTransport(response), control_transport=httpx.MockTransport(response))
        try:
            return await adapter.dispatch(task_id='synthetic-task', epoch=1, endpoint=selected,
                permit_id='synthetic-permit', reservation_id='synthetic-reservation', dispatch_body=body,
                dispatch_body_sha256=domain_sha256('cloud-dispatch-body/1', body),
                input_token_limit=1024, output_token_limit=256, max_reserved_microusd=1000)
        finally:
            await adapter.aclose()
            self.assertEqual(requests.count('POST'), 1)

    async def test_identity_gzip_and_deflate_responses_remain_supported(self):
        for encoding, transform in (('identity', lambda raw: raw), ('gzip', gzip.compress), ('deflate', zlib.compress)):
            for target in ('post', 'receipt'):
                with self.subTest(encoding=encoding, target=target):
                    result = await self.run_dispatch(target=target, encoding=encoding, transform=transform)
                    self.assertEqual(result.text, 'synthetic answer')
                    self.assertTrue(self.stream.closed)

    async def test_duplicate_json_is_rejected_for_both_response_paths(self):
        for target, extra in (('post', b',"id":"gen-fixture"}'), ('receipt', b',"extra":1,"extra":1}')):
            with self.subTest(target=target), self.assertRaises(OrchestratorError):
                await self.run_dispatch(target=target, transform=lambda raw: raw[:-1] + extra)
            self.assertTrue(self.stream.closed)
            self.store.reconcile_cloud_dispatch.assert_called_with('synthetic-reservation', actual_microusd=None, ambiguous=True)

    async def test_plain_and_compressed_bodies_are_bounded(self):
        def padded(raw):
            return raw[:-1] + b',"padding":"' + b'x' * (5 * 1024 * 1024) + b'"}'
        for target in ('post', 'receipt'):
            for encoding, transform in (('identity', padded), ('gzip', lambda raw: gzip.compress(padded(raw)))):
                with self.subTest(target=target, encoding=encoding), self.assertRaises(OrchestratorError):
                    await self.run_dispatch(target=target, encoding=encoding, transform=transform)
                self.assertTrue(self.stream.closed)

    async def test_incomplete_compressed_streams_are_rejected(self):
        with self.assertRaises(OrchestratorError):
            await self.run_dispatch(encoding='gzip', transform=lambda raw: gzip.compress(raw)[:-8])
        self.assertTrue(self.stream.closed)

    async def test_invalid_control_bodies_close_both_requests(self):
        for raw in (b'{"data":[],"data":[]}', b' ' * (5 * 1024 * 1024)):
            streams = []
            async def control(request):
                stream = Chunks(raw)
                streams.append(stream)
                return httpx.Response(200, headers={'content-type': 'application/json'}, stream=stream)
            adapter = OpenRouterCloudAdapter(Mock(), bearer_token='synthetic-key',
                transport=httpx.MockTransport(control), control_transport=httpx.MockTransport(control))
            try:
                with self.assertRaisesRegex(OrchestratorError, 'revalidation_unavailable'):
                    await adapter.revalidate(endpoint(DEEPSEEK_MODEL))
                self.assertEqual(len(streams), 2)
                self.assertTrue(all(stream.closed for stream in streams))
                adapter._store.store_endpoint_health.assert_not_called()
            finally:
                await adapter.aclose()

    async def test_total_post_deadline_closes_a_trickling_stream(self):
        started = time.monotonic()
        with self.assertRaises(OrchestratorError):
            await self.run_dispatch(delay=0.01, timeout=0.03)
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertTrue(self.stream.closed)
        self.store.reconcile_cloud_dispatch.assert_called_with('synthetic-reservation', actual_microusd=None, ambiguous=True)

    async def test_physical_permit_cannot_authorize_another_origin(self):
        seen = []
        async def wrapped(request):
            seen.append(request.url)
            return httpx.Response(200)
        transport = GuardedOpenRouterTransport(httpx.MockTransport(wrapped))
        permit = PhysicalCloudPermit('synthetic-permit')
        token = _cloud_permit.set(permit)
        try:
            with self.assertRaises(OrchestratorError):
                await transport.handle_async_request(httpx.Request('POST', 'https://other.invalid/api/v1/chat/completions'))
            self.assertEqual(seen, [])
            self.assertFalse(permit.consumed)
        finally:
            _cloud_permit.reset(token)
            await transport.aclose()


if __name__ == '__main__':
    unittest.main()
