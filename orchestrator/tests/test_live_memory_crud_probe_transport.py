"""Exercise the actual urllib routing policy without opening a network socket."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
import urllib.response
import uuid
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from test_live_memory_crud_probe_safety import MODULE


class LiveMemoryCrudTransportTests(unittest.TestCase):
    def exercise(self, method, *, proxy=False, redirect=None):
        requests = []

        def respond(handler, request):
            requests.append(request)
            headers = Message()
            status = 204 if method == "DELETE" else 200
            if redirect is not None and len(requests) == 1:
                status = redirect
                headers["Location"] = "https://outside.example.invalid/capture"
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, status
            )
            response.msg = "synthetic response"
            return response

        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(
                json.dumps({"control_bearer_token": "synthetic-control-key"})
            )
            operation = (
                MODULE._delete_conversation
                if method == "DELETE"
                else MODULE._cancel_task
            )
            with (
                patch.object(urllib.request, "_opener", None),
                patch.object(
                    urllib.request,
                    "getproxies",
                    return_value={"http": "http://127.0.0.1:19999"} if proxy else {},
                ),
                patch.object(urllib.request, "proxy_bypass", return_value=False),
                patch.object(urllib.request.HTTPHandler, "http_open", respond),
                patch.object(urllib.request.HTTPSHandler, "https_open", respond),
            ):
                if redirect is not None:
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        operation("http://127.0.0.1:8867", config, "synthetic-id", 7)
                    self.assertEqual(error.exception.code, redirect)
                else:
                    self.assertEqual(
                        operation("http://127.0.0.1:8867", config, "synthetic-id", 7),
                        204 if method == "DELETE" else 200,
                    )
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.host, "127.0.0.1:8867")
        self.assertEqual(request.get_method(), method)
        self.assertEqual(
            request.get_header("Authorization"), "Bearer synthetic-control-key"
        )
        self.assertEqual(request.get_header("If-match-epoch"), "7")
        self.assertEqual(uuid.UUID(request.get_header("Idempotency-key")).version, 4)
        self.assertEqual(request.timeout, 3)
        expected_path = (
            "/v1/conversations/synthetic-id"
            if method == "DELETE"
            else "/v1/tasks/synthetic-id/cancel"
        )
        self.assertEqual(request.full_url, "http://127.0.0.1:8867" + expected_path)
        self.assertEqual(
            request.data,
            None if method == "DELETE" else b'{"reason_code":"user_cancelled"}',
        )

    def test_direct_control_calls_preserve_protocol(self):
        for method in ("POST", "DELETE"):
            with self.subTest(method=method):
                self.exercise(method)

    def test_environment_proxy_cannot_receive_control_token(self):
        for method in ("POST", "DELETE"):
            with self.subTest(method=method):
                self.exercise(method, proxy=True)

    def test_redirect_cannot_receive_control_token(self):
        for method in ("POST", "DELETE"):
            for status in (302, 307):
                with self.subTest(method=method, status=status):
                    self.exercise(method, redirect=status)


if __name__ == "__main__":
    unittest.main()
