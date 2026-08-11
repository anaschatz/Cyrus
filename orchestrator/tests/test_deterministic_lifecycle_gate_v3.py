from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401
from jarvis_orchestrator.anamnesis_shadow import (
    DETERMINISTIC_FIRST_LOCAL_LIFECYCLE_WRITER_V3_CONTRACT_SHA256,
    DETERMINISTIC_LIFECYCLE_GATE_CONTRACT_SHA256,
    DETERMINISTIC_LIFECYCLE_GATE_V2_CONTRACT_SHA256,
    DETERMINISTIC_LIFECYCLE_GATE_V3_CONTRACT_SHA256,
    LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256,
    DeterministicFirstLifecycleWriter,
    DeterministicLifecycleGateV3,
    LifecycleCandidate,
    LifecycleDecision,
    LifecycleObservation,
    LifecycleOperation,
)


class Fallback:
    contract_sha256 = LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256

    async def decide(self, observation, active_memories):
        del observation, active_memories
        return LifecycleDecision(LifecycleOperation.IGNORE)


class DeterministicLifecycleGateV3Tests(unittest.TestCase):
    def test_contracts_are_additive_and_frozen(self) -> None:
        self.assertEqual(
            DETERMINISTIC_LIFECYCLE_GATE_CONTRACT_SHA256,
            "51ff7b97ac3f8a04fcb584b88eb717f5de827e9dc4dde44a38564f505bda311c",
        )
        self.assertEqual(
            DETERMINISTIC_LIFECYCLE_GATE_V2_CONTRACT_SHA256,
            "a5f522a1107fee9089a06a65c70eaadbe0d95a26894312b93259f482f0518915",
        )
        self.assertEqual(
            DETERMINISTIC_LIFECYCLE_GATE_V3_CONTRACT_SHA256,
            "18d80ac7e750625cb22990979d64d868b799d4fce59eda6e09c72fa0f8644a33",
        )
        writer = DeterministicFirstLifecycleWriter(
            Fallback(),
            gate=DeterministicLifecycleGateV3(),
        )
        self.assertEqual(
            writer.contract_sha256,
            DETERMINISTIC_FIRST_LOCAL_LIFECYCLE_WRITER_V3_CONTRACT_SHA256,
        )
        self.assertEqual(
            writer.contract_sha256,
            "fddfab1bccd089b941ae0e8cc2f28a2b65567525cee82c9e55751818deac2b1b",
        )

    def test_all_rule_parts_share_accent_insensitive_normalization(self) -> None:
        gate = DeterministicLifecycleGateV3()
        active = (LifecycleCandidate(0, "stored preference", "el"),)
        cases = (
            ("Ενημέρωσε τη μνήμη για το ρόφημα σε νερό.", "replace_0"),
            ("Ενημερωσε τη μνημη για το ροφημα σε νερο.", "replace_0"),
            ("Διέγραψε το ρόφημα από τη μνήμη.", "cancel_0"),
            ("Διεγραψε το ροφημα απο τη μνημη.", "cancel_0"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                decision = gate.decide(LifecycleObservation(text, "el"), active)
                assert decision is not None
                self.assertEqual(
                    f"{decision.operation.value}_{decision.active_index}",
                    expected,
                )

    def test_accented_and_unaccented_hedges_route_to_fallback(self) -> None:
        gate = DeterministicLifecycleGateV3()
        for text in (
            "Να θυμάσαι ότι ίσως άλλαξε το πρόγραμμα.",
            "Να θυμασαι οτι ισως αλλαξε το προγραμμα.",
            "Διόρθωση: μάλλον προτιμώ νερό.",
            "Διορθωση: μαλλον προτιμω νερο.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(
                    gate.decide(LifecycleObservation(text, "el"), ())
                )


if __name__ == "__main__":
    unittest.main()
