from __future__ import annotations

import hashlib
import os
import threading
import unittest
from types import SimpleNamespace

from jarvis_inference.errors import ApplianceError
from jarvis_inference.gateway import Production
from jarvis_inference.primary_qualification import (
    _PROFILE_RUNNER_SYSTEM_ENV_KEYS,
    _PrimaryQualificationProduction,
)
from jarvis_inference.strict_json import canonical_bytes


def _metric(serial: int = 1) -> dict[str, object]:
    return {
        "generation_serial": serial,
        "prompt_token_count": 3,
        "prompt_token_ids_sha256": hashlib.sha256(
            canonical_bytes([7, 11, 13])
        ).hexdigest(),
        "accepted_token_count": 2,
        "accepted_token_ids_sha256": hashlib.sha256(
            canonical_bytes([17, 19])
        ).hexdigest(),
        "request_ttft_seconds": 0.5,
        "backend_prompt_seconds": 0.25,
        "backend_decode_seconds": 0.5,
        "backend_decode_tokens_per_second": 4.0,
        "request_to_engine_final_seconds": 1.0,
    }


class MetricEngine:
    def __init__(self, value: dict[str, object] | None = None) -> None:
        self.value = value or _metric()
        self.arm_calls = 0
        self.bind_calls: list[float] = []
        self.consume_calls: list[int] = []

    def ready(self) -> bool:
        return True

    def arm_qualification_generation(self) -> int:
        self.arm_calls += 1
        return 1

    def bind_qualification_request_start(self, started: float) -> None:
        self.bind_calls.append(started)

    def consume_qualification_generation_metric(
        self, expected_serial: int
    ) -> dict[str, object]:
        self.consume_calls.append(expected_serial)
        return dict(self.value)

    def runtime_snapshot(self) -> dict[str, object]:
        return {"pid": 1234, "phase": "idle"}


def _production(engine: MetricEngine) -> _PrimaryQualificationProduction:
    production = object.__new__(_PrimaryQualificationProduction)
    production.admission = threading.Lock()
    production.state_lock = threading.Lock()
    production.engine = engine
    production.ready_state = True
    production._qualification_history = []
    production.manifest = SimpleNamespace(context_window=4096)
    return production


class PrimaryQualificationMetricTests(unittest.TestCase):
    def test_profile_runner_never_inherits_the_user_path(self) -> None:
        self.assertNotIn("PATH", _PROFILE_RUNNER_SYSTEM_ENV_KEYS)

    def test_private_boundary_is_exact_content_free_and_delegated(self) -> None:
        engine = MetricEngine()
        production = _production(engine)

        self.assertEqual(
            production.qualification_arm_generation(), {"generation_serial": 1}
        )
        production.qualification_bind_request_start(123.5)
        result = production.qualification_generation_metric(1)

        self.assertEqual(result, _metric())
        self.assertEqual(engine.arm_calls, 1)
        self.assertEqual(engine.bind_calls, [123.5])
        self.assertEqual(engine.consume_calls, [1])
        serialized = canonical_bytes(result).decode("ascii")
        self.assertNotIn("token_ids\"", serialized)
        self.assertNotIn("content", serialized)
        self.assertNotIn("text", serialized)
        self.assertNotIn("piece", serialized)
        self.assertNotIn("secret", serialized)

    def test_private_boundary_rejects_extra_or_malformed_metric_fields(self) -> None:
        extra = _metric()
        extra["content"] = "must never escape"
        with self.assertRaisesRegex(ApplianceError, "unavailable"):
            _production(MetricEngine(extra)).qualification_generation_metric(1)

        malformed = _metric()
        malformed["backend_decode_seconds"] = 0
        with self.assertRaisesRegex(ApplianceError, "unavailable"):
            _production(MetricEngine(malformed)).qualification_generation_metric(1)

        impossible = _metric()
        impossible["request_ttft_seconds"] = 2.0
        impossible["backend_prompt_seconds"] = 2.0
        impossible["backend_decode_seconds"] = 2.0
        impossible["backend_decode_tokens_per_second"] = 1.0
        impossible["request_to_engine_final_seconds"] = 3.0
        with self.assertRaisesRegex(ApplianceError, "unavailable"):
            _production(MetricEngine(impossible)).qualification_generation_metric(1)

    def test_runtime_binds_gateway_pid_and_public_production_has_no_boundary(self) -> None:
        runtime = _production(MetricEngine()).qualification_runtime()
        self.assertEqual(runtime["gateway_pid"], os.getpid())
        self.assertEqual(runtime["engine"]["pid"], 1234)
        self.assertFalse(hasattr(Production, "qualification_arm_generation"))
        self.assertFalse(hasattr(Production, "qualification_bind_request_start"))
        self.assertFalse(hasattr(Production, "qualification_generation_metric"))


if __name__ == "__main__":
    unittest.main()
