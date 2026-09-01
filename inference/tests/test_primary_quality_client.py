from __future__ import annotations

import unittest
from unittest.mock import patch

from jarvis_inference.primary_profile import PRIMARY_SYSTEM_PROMPT
from jarvis_inference.primary_quality_client import PrimaryQualityClient
from jarvis_inference.profile_preflight import FALLBACK_SYSTEM_PROMPT, Fixture
from jarvis_inference.strict_json import canonical_bytes, parse_body


class _Response:
    status = 200

    def __init__(self) -> None:
        self._lines = iter(
            (
                b"data: "
                + canonical_bytes(
                    {
                        "choices": [
                            {"delta": {"content": "yes"}, "finish_reason": None}
                        ]
                    }
                )
                + b"\n",
                b"data: "
                + canonical_bytes(
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                )
                + b"\n",
                b"data: "
                + canonical_bytes(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 1,
                            "total_tokens": 4,
                        },
                    }
                )
                + b"\n",
                b"data: [DONE]\n",
            )
        )

    def getheader(self, name: str) -> str | None:
        if name.casefold() == "content-type":
            return "text/event-stream; charset=utf-8"
        return None

    def readline(self) -> bytes:
        return next(self._lines, b"")


class _Connection:
    def __init__(self) -> None:
        self.request_body: bytes | None = None
        self.closed = False

    def request(self, method, path, body=None, headers=None) -> None:
        if method != "POST" or path != "/v1/chat/completions":
            raise AssertionError("unexpected primary quality request")
        self.request_body = body

    def getresponse(self) -> _Response:
        return _Response()

    def close(self) -> None:
        self.closed = True


class PrimaryQualityClientTests(unittest.TestCase):
    def test_primary_prompt_is_used_without_mutating_frozen_fallback_scorer(self) -> None:
        client = PrimaryQualityClient("http://127.0.0.1:8766/v1", "k" * 48)
        connection = _Connection()
        fixture = Fixture(
            "primary-exact", "en", "Reply exactly yes", 16, exact="yes"
        )

        with patch.object(client, "_connection", return_value=connection):
            result = client.stream(fixture)

        self.assertTrue(result["quality_pass"])
        self.assertTrue(connection.closed)
        self.assertIsNotNone(connection.request_body)
        document = parse_body(connection.request_body or b"")
        self.assertEqual(document["messages"][0]["content"], PRIMARY_SYSTEM_PROMPT)
        self.assertNotEqual(PRIMARY_SYSTEM_PROMPT, FALLBACK_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
