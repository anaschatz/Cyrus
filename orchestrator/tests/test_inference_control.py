from __future__ import annotations

import asyncio
import json
import time
import unittest

import httpx
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.domain import OrchestratorError
from jarvis_orchestrator.inference import (
    InferenceControlClient,
    InferenceIdentityEndpointUnavailable,
)


class InferenceControlIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = InferenceControlClient(
            "http://127.0.0.1:8766/v1", "x" * 32
        )

    def test_ready_requires_a_boolean_and_not_an_integer_alias(self) -> None:
        with patch.object(self.client, "_get", return_value={"ready": 1}):
            self.assertFalse(self.client.ready())
        with patch.object(self.client, "_get", return_value={"ready": True}):
            self.assertTrue(self.client.ready())

    def test_redirect_does_not_forward_authentication_or_use_environment_proxies(self) -> None:
        requests = []
        client_type = httpx.AsyncClient

        def respond(request):
            requests.append(request)
            return httpx.Response(302, headers={"Location": "https://unrelated.invalid/identity"})

        def client(**kwargs):
            return client_type(transport=httpx.MockTransport(respond), **kwargs)

        with patch("httpx.AsyncClient", side_effect=client) as clients:
            with self.assertRaises(OrchestratorError):
                self.client.runtime_identity()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.host, "127.0.0.1")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer " + "x" * 32)
        self.assertFalse(clients.call_args.kwargs["trust_env"])
        self.assertFalse(clients.call_args.kwargs["follow_redirects"])

    def test_inference_json_rejects_ambiguous_and_nested_documents(self) -> None:
        client_type = httpx.AsyncClient
        for raw in (b'{"ready":false,"ready":true}', b'[' * 1500 + b'0' + b']' * 1500,
                    json.dumps({"ready": True, "private": "\ud800"}).encode()):
            with self.subTest(size=len(raw)), patch("httpx.AsyncClient", side_effect=lambda **kwargs: client_type(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, content=raw)), **kwargs
            )):
                self.assertFalse(self.client.ready())

    def test_deadline_covers_headers_and_body_and_closes_timed_out_stream(self) -> None:
        client_type = httpx.AsyncClient

        async def slow_headers(request):
            await asyncio.sleep(10)
            return httpx.Response(200, json={"ready": True})

        class Trickle(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                for _ in range(1000):
                    await asyncio.sleep(0.005)
                    yield b" "

            async def aclose(self):
                self.closed = True

        body = Trickle()
        client = InferenceControlClient("http://127.0.0.1:8766/v1", "test", timeout_seconds=0.03)
        for handler in (slow_headers, lambda request: httpx.Response(200, stream=body)):
            started = time.monotonic()
            with patch("httpx.AsyncClient", side_effect=lambda **kwargs: client_type(
                transport=httpx.MockTransport(handler), **kwargs
            )):
                self.assertFalse(client.ready())
            self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(body.closed)

    def test_valid_control_responses_preserve_authentication_scope(self) -> None:
        client_type = httpx.AsyncClient
        paths = []

        def respond(request):
            paths.append(request.url.path)
            if request.url.path == "/readyz":
                self.assertNotIn("Authorization", request.headers)
                return httpx.Response(200, json={"ready": True})
            self.assertEqual(request.headers["Authorization"], "Bearer " + "x" * 32)
            if request.url.path == "/jarvis/v1/capabilities":
                return httpx.Response(200, json={"model": "jarvis-local", "capabilities": {
                    "text": True, "stream": True, "tools": False, "json_object": True, "json_schema": True,
                }})
            return httpx.Response(200, json={"manifest_digest": "a" * 64, "runtime_identity_sha256": "b" * 64,
                                            "schema_version": "jarvis-inference-live-identity/1"})

        with patch("httpx.AsyncClient", side_effect=lambda **kwargs: client_type(
            transport=httpx.MockTransport(respond), **kwargs
        )):
            self.assertTrue(self.client.ready())
            self.assertTrue(self.client.capabilities().json_schema)
            self.assertEqual(self.client.runtime_identity().manifest_digest, "a" * 64)
        self.assertEqual(paths, ["/readyz", "/jarvis/v1/capabilities", "/jarvis/v1/identity"])

    def test_exact_live_identity_is_accepted(self) -> None:
        with patch.object(
            self.client,
            "_get",
            return_value={
                "manifest_digest": "a" * 64,
                "runtime_identity_sha256": "b" * 64,
                "schema_version": "jarvis-inference-live-identity/1",
            },
        ):
            identity = self.client.runtime_identity()

        self.assertEqual(identity.manifest_digest, "a" * 64)
        self.assertEqual(identity.runtime_identity_sha256, "b" * 64)

    def test_malformed_or_extended_identity_fails_closed(self) -> None:
        invalid = (
            {},
            {
                "manifest_digest": "a" * 64,
                "runtime_identity_sha256": "b" * 64,
                "schema_version": "jarvis-inference-live-identity/1",
                "extra": True,
            },
            {
                "manifest_digest": "not-a-digest",
                "runtime_identity_sha256": "b" * 64,
                "schema_version": "jarvis-inference-live-identity/1",
            },
        )
        for value in invalid:
            with self.subTest(value=value), patch.object(
                self.client, "_get", return_value=value
            ), self.assertRaises(OrchestratorError):
                self.client.runtime_identity()

    def test_only_explicit_missing_identity_route_is_distinguished(self) -> None:
        client_type = httpx.AsyncClient
        for status in (404, 405, 503):
            with self.subTest(status=status), patch("httpx.AsyncClient", side_effect=lambda **kwargs: client_type(
                transport=httpx.MockTransport(lambda request: httpx.Response(status)), **kwargs
            )):
                expected = InferenceIdentityEndpointUnavailable if status in (404, 405) else OrchestratorError
                with self.assertRaises(expected):
                    self.client.runtime_identity()

        def unavailable(request):
            raise httpx.ConnectError("synthetic failure", request=request)

        with patch("httpx.AsyncClient", side_effect=lambda **kwargs: client_type(
            transport=httpx.MockTransport(unavailable), **kwargs
        )), self.assertRaises(OrchestratorError):
            self.client.runtime_identity()


if __name__ == "__main__":
    unittest.main()
