from __future__ import annotations

import asyncio
import io
import json
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest.mock import patch

import httpx
import _bootstrap  # noqa: F401
from jarvis_orchestrator import cli
from jarvis_orchestrator.config import OrchestratorConfig


class CliControlClientTests(unittest.TestCase):
    def setUp(self):
        self.config = OrchestratorConfig.from_test_mapping(
            {
                "database_path": "/tmp/unused-cli-control-review.sqlite3",
                "control_bearer_token": "synthetic-control",
                "inference_bearer_token": "synthetic-inference",
            }
        )

    def exchange(
        self,
        body=b'{"ready":true}',
        *,
        status=200,
        headers=None,
        stream=None,
        failure=None,
        main=False,
    ):
        responses, requests, clients = [], [], []
        media = headers if headers is not None else {"content-type": "application/json"}

        class LegacyResponse(io.BytesIO):
            def __init__(self):
                super().__init__(body)
                self.status = status

        legacy_response = LegacyResponse()

        def legacy(request, **kwargs):
            requests.append(request)
            if failure:
                raise urllib.error.URLError("synthetic private diagnostic")
            if status >= 400:
                raise urllib.error.HTTPError(
                    request.full_url, status, "error", media, legacy_response
                )
            return legacy_response

        def respond(request):
            requests.append(request)
            if failure:
                raise httpx.ConnectError(
                    "synthetic private diagnostic", request=request
                )
            response = httpx.Response(
                status,
                headers=media,
                stream=stream if stream is not None else httpx.ByteStream(body),
            )
            responses.append(response)
            return response

        original = httpx.AsyncClient

        class Client(original):
            def __init__(self, **kwargs):
                clients.append(kwargs)
                super().__init__(transport=httpx.MockTransport(respond), **kwargs)

        with (
            patch("urllib.request.urlopen", side_effect=legacy),
            patch.object(httpx, "AsyncClient", Client),
        ):
            if main:
                output = io.StringIO()
                with (
                    patch.object(
                        cli.OrchestratorConfig, "load", return_value=self.config
                    ),
                    redirect_stdout(output),
                ):
                    result = (
                        cli.main(["--config", "/synthetic/config", "ready"]),
                        output,
                    )
            else:
                result = cli._request(
                    self.config,
                    "POST",
                    "/v1/conversations",
                    body={},
                    epoch=7,
                    key="synthetic-key",
                )
        return result, responses, requests, clients, legacy_response

    def test_valid_json_and_empty_responses_keep_status_and_request_headers(self):
        for body, status in (
            (b'{"ready":true}', 200),
            (b"", 204),
            (b'{"error":{"code":"stale_epoch"}}', 409),
        ):
            result, responses, requests, clients, legacy = self.exchange(
                body, status=status
            )
            self.assertEqual(result, (status, json.loads(body) if body else None))
            self.assertEqual(len(requests), 1)
            if clients:
                self.assertFalse(clients[0]["trust_env"])
                self.assertFalse(clients[0]["follow_redirects"])
                self.assertTrue(responses[0].is_closed)
                self.assertEqual(requests[0].headers["if-match-epoch"], "7")
                self.assertEqual(
                    requests[0].headers["idempotency-key"], "synthetic-key"
                )
            else:
                self.assertTrue(legacy.closed)

    def test_wrong_media_redirects_and_malformed_json_fail_closed(self):
        for body, status, headers in (
            (b'{"ready":true}', 302, {"location": "https://example.invalid/"}),
            (b'{"ready":true}', 200, {"content-type": "text/html"}),
            (b'{"ready":false,"ready":true}', 200, None),
            (b"", 200, None),
            (b"[]", 200, None),
            (b"null", 200, None),
            (b'{"value":NaN}', 200, None),
            (b'{"value":"\\ud800"}', 200, None),
            (b"[" * 2000 + b"0" + b"]" * 2000, 200, None),
        ):
            with self.subTest(status=status, size=len(body)):
                result, _, requests, *_ = self.exchange(
                    body, status=status, headers=headers
                )
                self.assertEqual(len(requests), 1)
                self.assertEqual(result[0], 503)
                self.assertEqual(result[1]["error"]["code"], "control_request_failed")

    def test_response_body_is_bounded(self):
        result, *_ = self.exchange(b'{"value":"' + b"x" * (1024 * 1024) + b'"}')
        self.assertEqual(result[0], 503)

    def test_connection_failure_returns_content_free_cli_failure(self):
        result, *_ = self.exchange(failure=True, main=True)
        status, output = result
        self.assertEqual(status, 1)
        value = json.loads(output.getvalue())
        self.assertEqual(
            value,
            {
                "status": 503,
                "body": {
                    "error": {"code": "control_request_failed", "retryable": True}
                },
            },
        )
        self.assertNotIn("private diagnostic", output.getvalue())

    def test_total_deadline_closes_trickling_stream(self):
        class Stream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b" "
                await asyncio.sleep(10)

            async def aclose(self):
                self.closed = True

        stream = Stream()
        original = asyncio.timeout
        with patch.object(asyncio, "timeout", side_effect=lambda delay: original(0.02)):
            result, *_ = self.exchange(stream=stream)
        self.assertEqual(result[0], 503)
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
