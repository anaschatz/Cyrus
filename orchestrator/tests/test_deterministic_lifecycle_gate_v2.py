from __future__ import annotations

import asyncio
import unittest

import _bootstrap  # noqa: F401
from jarvis_orchestrator.anamnesis_shadow import (
    DETERMINISTIC_FIRST_LOCAL_LIFECYCLE_WRITER_V2_CONTRACT_SHA256,
    DETERMINISTIC_LIFECYCLE_GATE_CONTRACT_SHA256,
    DETERMINISTIC_LIFECYCLE_GATE_V2_CONTRACT_SHA256,
    LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256,
    DeterministicFirstLifecycleWriter,
    DeterministicLifecycleGateV2,
    LifecycleCandidate,
    LifecycleDecision,
    LifecycleObservation,
    LifecycleOperation,
)


V1_SHA256 = "51ff7b97ac3f8a04fcb584b88eb717f5de827e9dc4dde44a38564f505bda311c"
V2_SHA256 = "a5f522a1107fee9089a06a65c70eaadbe0d95a26894312b93259f482f0518915"
V2_WRITER_SHA256 = (
    "3604c9ae2b87821085d5bdcdd0c5f5e0643aceb16558171656c4235bbc390bce"
)


class RejectingFallback:
    contract_sha256 = LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256

    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, observation, active_memories):
        del observation, active_memories
        self.calls += 1
        return LifecycleDecision(LifecycleOperation.IGNORE)


class DeterministicLifecycleGateV2Tests(unittest.TestCase):
    def test_contract_is_additive_and_v1_remains_frozen(self) -> None:
        self.assertEqual(DETERMINISTIC_LIFECYCLE_GATE_CONTRACT_SHA256, V1_SHA256)
        self.assertEqual(DETERMINISTIC_LIFECYCLE_GATE_V2_CONTRACT_SHA256, V2_SHA256)
        fallback = RejectingFallback()
        writer = DeterministicFirstLifecycleWriter(
            fallback,
            gate=DeterministicLifecycleGateV2(),
        )
        self.assertEqual(writer.contract_sha256, V2_WRITER_SHA256)
        self.assertEqual(
            DETERMINISTIC_FIRST_LOCAL_LIFECYCLE_WRITER_V2_CONTRACT_SHA256,
            V2_WRITER_SHA256,
        )

    def test_closed_bilingual_families_and_normalized_rules(self) -> None:
        gate = DeterministicLifecycleGateV2()
        active = (LifecycleCandidate(0, "stored preference", "en"),)
        cases = (
            ("Could you remember that I prefer tea?", "en", (), "create"),
            ("Κράτα στη μνήμη σου ότι προτιμώ τσάι.", "el", (), "create"),
            (
                "Update my remembered drink preference to coffee.",
                "en",
                active,
                "replace_0",
            ),
            (
                "Άλλαξε την αποθηκευμένη προτίμησή μου σε καφέ.",
                "el",
                active,
                "replace_0",
            ),
            ("Delete my drink preference from memory.", "en", active, "cancel_0"),
            ("Θέλω να ξεχάσεις την προτίμησή μου.", "el", active, "cancel_0"),
            ("Θελω να ξεχασεις την προτιμηση μου.", "el", active, "cancel_0"),
        )
        for text, language, candidates, expected in cases:
            with self.subTest(text=text):
                decision = gate.decide(
                    LifecycleObservation(text, language),
                    candidates,
                )
                assert decision is not None
                actual = decision.operation.value
                if decision.active_index is not None:
                    actual += f"_{decision.active_index}"
                self.assertEqual(actual, expected)

    def test_hedges_multiple_targets_and_incomplete_commands_fallback(self) -> None:
        gate = DeterministicLifecycleGateV2()
        one = (LifecycleCandidate(0, "stored preference", "en"),)
        two = one + (LifecycleCandidate(1, "stored timezone", "en"),)
        for text, active in (
            ("Correction: maybe I prefer coffee.", one),
            ("Delete my preference from memory.", two),
            ("Remember that ...", ()),
        ):
            with self.subTest(text=text):
                self.assertIsNone(
                    gate.decide(LifecycleObservation(text, "en"), active)
                )

    def test_quoted_negated_and_descriptive_text_remains_ignore(self) -> None:
        fallback = RejectingFallback()
        writer = DeterministicFirstLifecycleWriter(
            fallback,
            gate=DeterministicLifecycleGateV2(),
        )
        active = (LifecycleCandidate(0, "stored preference", "en"),)
        for text in (
            'Translate "delete this from memory".',
            "I do not want you to forget anything.",
            "The article describes how memory works.",
        ):
            with self.subTest(text=text):
                decision = asyncio.run(
                    writer.decide(LifecycleObservation(text, "en"), active)
                )
                self.assertEqual(decision.operation, LifecycleOperation.IGNORE)
        self.assertEqual(fallback.calls, 0)


if __name__ == "__main__":
    unittest.main()
