from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_lifecycle_shadow_service_pilot_v1.py"
PROTOCOL = ROOT / "eval/lifecycle_shadow_service_pilot_v1.protocol.json"
DATASET = ROOT / "eval/lifecycle_shadow_service_pilot_v1.jsonl"
MANIFEST = ROOT / "eval/lifecycle_shadow_service_pilot_v1.manifest.json"
RESULT = ROOT / "results/lifecycle_shadow_service_pilot_v1.2026-08-10.json"
FROZEN_SHA256 = {
    RUNNER: "973b981438a3b6e0332a92023935d8c1498dec4afea1922321d7d3e9246ffe78",
    PROTOCOL: "00669e1ff12b84475ad65d5af0bfff49f6152bef7c55309fe83e3747c9c5f012",
    DATASET: "320ea83b76bc91d302713846961d8e8b5ce5d47a9abda16698ed90f5f53942ae",
    MANIFEST: "c5a24b801b75cae24274a43f6014e3d4538711fa6eabfeabcf012e6dab878de5",
    RESULT: "e5518f78ba8fdcc2d3e6c2257c33a6736bf6d292334f0bbda222a0b7347ba3a7",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LifecycleShadowServicePilotV1Tests(unittest.TestCase):
    def test_frozen_service_pilot_artifacts_remain_byte_exact(self) -> None:
        # This is historical 2026-08-10 evidence.  It is source-bound to the
        # frozen v1 files and is deliberately not re-executed against current
        # service semantics, which now bypass the assistant for explicit memory
        # commands.  Current executable evidence will be versioned separately
        # only after the remaining backend architecture has stabilized.
        self.assertEqual(
            {path: _sha256(path) for path in FROZEN_SHA256},
            FROZEN_SHA256,
        )

    def test_frozen_service_pilot_claims_remain_historical_and_bounded(self) -> None:
        stored = json.loads(RESULT.read_text())
        self.assertEqual(stored["assistant_stub_calls"], 30)
        self.assertEqual(stored["run_date"], "2026-08-10")
        self.assertEqual(stored["scenario_passed"], 8)
        self.assertEqual(stored["scenario_total"], 8)
        self.assertEqual(stored["turn_passed"], 30)
        self.assertEqual(stored["turn_total"], 30)
        self.assertEqual(stored["offline_fallback_calls"], 2)
        self.assertEqual(stored["lifecycle_model_calls"], 0)
        self.assertEqual(stored["provider_api_cost_usd"], 0.0)
        self.assertEqual(stored["authoritative_memory_records"], 0)
        self.assertFalse(stored["authoritative_memory_enabled"])
        self.assertFalse(stored["hypothesis_test_eligible"])
        self.assertEqual(stored["failures"], [])
        self.assertEqual(
            stored["promotion_decision"],
            "service_shadow_pilot_pass",
        )


if __name__ == "__main__":
    unittest.main()
