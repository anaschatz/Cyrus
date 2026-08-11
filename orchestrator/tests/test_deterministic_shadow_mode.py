from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from jarvis_orchestrator.anamnesis_shadow import (
    DETERMINISTIC_ONLY_LIFECYCLE_WRITER_V3_CONTRACT_SHA256,
    DeterministicOnlyLifecycleWriterV3,
    LifecycleCandidate,
    LifecycleObservation,
    LifecycleOperation,
)
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.memory_release import MEMORY_RELEASE_POLICY_VERSION


EXPECTED_CONTRACT_SHA256 = (
    "140e894a9f36c73c2d76deed7bd76729da88b2c81f350c9fb92472225959d717"
)


class DeterministicShadowModeTests(unittest.TestCase):
    def test_contract_and_conservative_ambiguity_policy_are_frozen(self) -> None:
        writer = DeterministicOnlyLifecycleWriterV3()
        active = (
            LifecycleCandidate(0, "Remember that I prefer tea.", "en"),
            LifecycleCandidate(1, "Remember that I live in Patras.", "en"),
        )
        clear = asyncio.run(
            writer.decide(
                LifecycleObservation("Remember that I prefer coffee.", "en"),
                active,
            )
        )
        ambiguous = asyncio.run(
            writer.decide(
                LifecycleObservation("Delete my preference from memory.", "en"),
                active,
            )
        )
        hedged = asyncio.run(
            writer.decide(
                LifecycleObservation(
                    "Remember that maybe my city will change.",
                    "en",
                ),
                (),
            )
        )
        self.assertEqual(clear.operation, LifecycleOperation.CREATE)
        self.assertEqual(ambiguous.operation, LifecycleOperation.IGNORE)
        self.assertEqual(hedged.operation, LifecycleOperation.IGNORE)
        self.assertEqual(writer.contract_sha256, EXPECTED_CONTRACT_SHA256)
        self.assertEqual(
            DETERMINISTIC_ONLY_LIFECYCLE_WRITER_V3_CONTRACT_SHA256,
            EXPECTED_CONTRACT_SHA256,
        )

    def test_config_accepts_only_closed_shadow_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = {
                "database_path": str(Path(directory) / "state.sqlite3"),
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
                "memory_shadow_enabled": True,
            }
            deterministic = OrchestratorConfig.from_mapping(
                {**base, "memory_shadow_mode": "deterministic_only"}
            )
            structured = OrchestratorConfig.from_mapping(
                {**base, "memory_shadow_mode": "structured_fallback"}
            )
            self.assertEqual(
                deterministic.memory_shadow_mode,
                "deterministic_only",
            )
            self.assertEqual(
                structured.memory_shadow_mode,
                "structured_fallback",
            )
            with self.assertRaisesRegex(ValueError, "memory_shadow_mode"):
                OrchestratorConfig.from_mapping(
                    {**base, "memory_shadow_mode": "model_auto"}
                )
            self.assertEqual(deterministic.final_response_output_tokens, 256)
            self.assertEqual(
                deterministic.memory_release_policy,
                MEMORY_RELEASE_POLICY_VERSION,
            )
            with self.assertRaisesRegex(ValueError, "memory_release_policy"):
                OrchestratorConfig.from_mapping(
                    {**base, "memory_release_policy": "generative-production"}
                )
            for invalid in (31, 1025, True):
                with self.subTest(final_response_output_tokens=invalid):
                    with self.assertRaisesRegex(
                        ValueError,
                        "final_response_output_tokens",
                    ):
                        OrchestratorConfig.from_mapping(
                            {
                                **base,
                                "final_response_output_tokens": invalid,
                            }
                        )


if __name__ == "__main__":
    unittest.main()
