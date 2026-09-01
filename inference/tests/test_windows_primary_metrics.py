from __future__ import annotations

import hashlib
import unittest
from unittest.mock import patch

from jarvis_inference.strict_json import canonical_bytes
from jarvis_inference.windows_primary_metrics import (
    PrimaryQualificationClient,
    PrimaryRunnerFailure,
)


def _tokenization() -> dict[str, object]:
    return {
        "token_count": 3,
        "token_ids_sha256": hashlib.sha256(
            canonical_bytes([7, 11, 13])
        ).hexdigest(),
    }


def _metric(serial: int = 1) -> dict[str, object]:
    tokenization = _tokenization()
    return {
        "generation_serial": serial,
        "prompt_token_count": 3,
        "prompt_token_ids_sha256": tokenization["token_ids_sha256"],
        "accepted_token_count": 2,
        "accepted_token_ids_sha256": hashlib.sha256(
            canonical_bytes([17, 19])
        ).hexdigest(),
        "request_ttft_seconds": 0.75,
        "backend_prompt_seconds": 0.5,
        "backend_decode_seconds": 0.5,
        "backend_decode_tokens_per_second": 4.0,
        "request_to_engine_final_seconds": 1.25,
    }


class WindowsPrimaryMetricClientTests(unittest.TestCase):
    def client(self) -> PrimaryQualificationClient:
        return PrimaryQualificationClient(
            "http://127.0.0.1:8766/v1", "k" * 48
        )

    def test_arm_public_request_and_exact_one_shot_metric_are_cross_bound(self) -> None:
        client = self.client()
        calls: list[tuple[str, str, object]] = []

        def request(method, path, payload=None):
            calls.append((method, path, payload))
            if path.endswith("/arm"):
                return 200, {"generation_serial": 1}
            return 200, _metric()

        public = {
            "end_to_end_seconds": 1.5,
            "completion_tokens": 2,
            "finish_reason": "stop",
        }
        with patch.object(client, "json_request", side_effect=request), patch.object(
            client, "_observe_buffered", return_value=public
        ):
            observed = client.observe(
                {"stream": False}, _tokenization()  # type: ignore[arg-type]
            )

        self.assertEqual(
            [path for _method, path, _payload in calls],
            [
                "/jarvis/qualification/generation/arm",
                "/jarvis/qualification/generation/metric",
            ],
        )
        self.assertEqual(calls[1][2], {"expected_serial": 1})
        self.assertEqual(observed["generation_serial"], 1)
        self.assertEqual(observed["ttft_seconds"], 0.75)
        self.assertEqual(observed["accepted_decode_tokens"], 2)
        serialized = canonical_bytes(observed).decode("ascii")
        self.assertNotIn('"content"', serialized)
        self.assertNotIn('"token_ids"', serialized)

    def test_serial_gap_extra_content_and_impossible_timings_fail_closed(self) -> None:
        client = self.client()
        with patch.object(
            client,
            "json_request",
            return_value=(200, {"generation_serial": 2}),
        ):
            with self.assertRaisesRegex(PrimaryRunnerFailure, "serial"):
                client._arm_generation()

        extra = _metric()
        extra["content"] = "must never escape"
        with patch.object(client, "json_request", return_value=(200, extra)):
            with self.assertRaisesRegex(PrimaryRunnerFailure, "shape"):
                client._consume_generation_metric(1)

        impossible = _metric()
        impossible["backend_prompt_seconds"] = 2.0
        impossible["backend_decode_seconds"] = 2.0
        impossible["backend_decode_tokens_per_second"] = 1.0
        impossible["request_to_engine_final_seconds"] = 3.0
        with patch.object(client, "json_request", return_value=(200, impossible)):
            with self.assertRaisesRegex(PrimaryRunnerFailure, "inconsistent"):
                client._consume_generation_metric(1)


if __name__ == "__main__":
    unittest.main()
