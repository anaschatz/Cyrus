from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_lifecycle_shadow_natural_v2_eval.py"
)
SPEC = importlib.util.spec_from_file_location("lifecycle_shadow_natural_v2_eval", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LifecycleShadowNaturalV2EvalTests(unittest.TestCase):
    def test_first_frozen_v2_result_is_complete_and_not_promoted(self) -> None:
        result = MODULE.evaluate()
        self.assertEqual((result["passed"], result["total"]), (59, 64))
        self.assertEqual(
            result["category_scores"],
            {
                "cancel": {"passed": 10, "total": 12},
                "create": {"passed": 16, "total": 16},
                "fallback": {"passed": 6, "total": 8},
                "ignore": {"passed": 16, "total": 16},
                "replace": {"passed": 11, "total": 12},
            },
        )
        self.assertEqual(
            result["promotion_gates"],
            {
                "explicit_command_recall": {"passed": 33, "total": 36},
                "ignore_precision": {"passed": 16, "total": 16},
                "fallback_routing": {"passed": 6, "total": 8},
            },
        )
        self.assertEqual(result["language_scores"]["en"], {"passed": 32, "total": 32})
        self.assertEqual(result["language_scores"]["el"], {"passed": 27, "total": 32})
        self.assertEqual(result["promotion_decision"], "gate_v2_rejected")
        self.assertEqual(len(result["failures"]), 5)


if __name__ == "__main__":
    unittest.main()
