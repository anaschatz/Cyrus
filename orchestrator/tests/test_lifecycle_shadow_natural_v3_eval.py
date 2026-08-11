from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_lifecycle_shadow_natural_v3_eval.py"
)
SPEC = importlib.util.spec_from_file_location("lifecycle_shadow_natural_v3_eval", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LifecycleShadowNaturalV3EvalTests(unittest.TestCase):
    def test_first_frozen_v3_result_passes_all_shadow_promotion_gates(self) -> None:
        result = MODULE.evaluate()
        self.assertEqual((result["passed"], result["total"]), (48, 48))
        self.assertEqual(
            result["category_scores"],
            {
                "cancel": {"passed": 10, "total": 10},
                "create": {"passed": 12, "total": 12},
                "fallback": {"passed": 6, "total": 6},
                "ignore": {"passed": 10, "total": 10},
                "replace": {"passed": 10, "total": 10},
            },
        )
        self.assertEqual(
            result["promotion_gates"],
            {
                "explicit_command_recall": {"passed": 28, "total": 28},
                "ignore_precision": {"passed": 12, "total": 12},
                "fallback_routing": {"passed": 8, "total": 8},
            },
        )
        self.assertEqual(result["language_scores"]["en"], {"passed": 24, "total": 24})
        self.assertEqual(result["language_scores"]["el"], {"passed": 24, "total": 24})
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["promotion_decision"], "shadow_pilot_candidate")


if __name__ == "__main__":
    unittest.main()
