from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_lifecycle_shadow_natural_eval.py"
)
SPEC = importlib.util.spec_from_file_location("lifecycle_shadow_natural_eval", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LifecycleShadowNaturalEvalTests(unittest.TestCase):
    def test_frozen_gate_v1_result_is_complete_and_safety_preserving(self) -> None:
        result = MODULE.evaluate()
        self.assertEqual((result["passed"], result["total"]), (37, 48))
        self.assertEqual(
            result["category_scores"],
            {
                "cancel": {"passed": 6, "total": 10},
                "create": {"passed": 7, "total": 12},
                "fallback": {"passed": 6, "total": 6},
                "ignore": {"passed": 10, "total": 10},
                "replace": {"passed": 8, "total": 10},
            },
        )
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["provider_api_cost_usd"], 0.0)
        self.assertEqual(len(result["failures"]), 11)
        self.assertTrue(
            all(failure["actual"] == "ignore" for failure in result["failures"])
        )
        self.assertEqual(
            result["promotion_decision"],
            "gate_v1_rejected_for_pilot_due_to_command_recall",
        )


if __name__ == "__main__":
    unittest.main()
