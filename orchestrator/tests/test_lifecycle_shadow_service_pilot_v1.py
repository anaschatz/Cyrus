from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_lifecycle_shadow_service_pilot_v1.py"
RESULT = ROOT / "results/lifecycle_shadow_service_pilot_v1.2026-08-10.json"
SPEC = importlib.util.spec_from_file_location(
    "lifecycle_shadow_service_pilot_v1",
    RUNNER,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LifecycleShadowServicePilotV1Tests(unittest.TestCase):
    def test_frozen_service_pilot_is_complete_and_non_authoritative(self) -> None:
        computed = MODULE.evaluate()
        stored = json.loads(RESULT.read_text())
        self.assertEqual(stored, computed)
        self.assertEqual(computed["scenario_passed"], 8)
        self.assertEqual(computed["scenario_total"], 8)
        self.assertEqual(computed["turn_passed"], 30)
        self.assertEqual(computed["turn_total"], 30)
        self.assertEqual(computed["offline_fallback_calls"], 2)
        self.assertEqual(computed["lifecycle_model_calls"], 0)
        self.assertEqual(computed["provider_api_cost_usd"], 0.0)
        self.assertEqual(computed["authoritative_memory_records"], 0)
        self.assertFalse(computed["authoritative_memory_enabled"])
        self.assertEqual(computed["failures"], [])
        self.assertEqual(
            computed["promotion_decision"],
            "service_shadow_pilot_pass",
        )


if __name__ == "__main__":
    unittest.main()
