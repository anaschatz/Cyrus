from __future__ import annotations

import asyncio
import unittest
import uuid
from collections.abc import AsyncIterator

import _bootstrap  # noqa: F401
from helpers import StoreCase
from jarvis_orchestrator.anamnesis_shadow import (
    DETERMINISTIC_FIRST_LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256,
    DETERMINISTIC_LIFECYCLE_GATE_CONTRACT_SHA256,
    LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256,
    AnamnesisShadowAdapter,
    DeterministicFirstLifecycleWriter,
    DeterministicLifecycleGate,
    DeterministicLifecycleGateV3,
    LifecycleCandidate,
    LifecycleDecision,
    LifecycleObservation,
    LifecycleOperation,
    reconstruct_active_shadow_memories,
)
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import ModelChunk, ModelUsage, TaskState
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService


EXPECTED_GATE_SHA256 = (
    "51ff7b97ac3f8a04fcb584b88eb717f5de827e9dc4dde44a38564f505bda311c"
)
EXPECTED_DETERMINISTIC_FIRST_SHA256 = (
    "962491120810d815a7058407f6043b0464bdf2987db2ec7243158e0860a5820d"
)


class RejectingFallback:
    contract_sha256 = LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256

    def __init__(self, decision: LifecycleDecision | None = None) -> None:
        self.decision = decision
        self.calls: list[
            tuple[LifecycleObservation, tuple[LifecycleCandidate, ...]]
        ] = []

    async def decide(self, observation, active_memories):
        self.calls.append((observation, active_memories))
        if self.decision is None:
            raise AssertionError("local lifecycle fallback was not expected")
        return self.decision


class StaticModel:
    async def stream(self, request) -> AsyncIterator[ModelChunk]:
        del request
        yield ModelChunk("Έτοιμο.", ModelUsage(input_tokens=3, output_tokens=2))


def _candidate(index: int, text: str = "Remember that I prefer tea."):
    return LifecycleCandidate(index=index, text=text, language="en")


class DeterministicLifecycleGateTests(StoreCase, unittest.TestCase):
    def test_frozen_contract_and_bilingual_diagnostic_matrix(self) -> None:
        gate = DeterministicLifecycleGate()
        one = (_candidate(0),)
        cases = (
            (
                "What time is it?",
                "en",
                (),
                LifecycleDecision(LifecycleOperation.IGNORE),
            ),
            (
                "Remember that I prefer tea.",
                "en",
                (),
                LifecycleDecision(LifecycleOperation.CREATE),
            ),
            (
                "Correction: I prefer coffee instead.",
                "en",
                one,
                LifecycleDecision(LifecycleOperation.REPLACE, 0),
            ),
            (
                "Forget my drink preference.",
                "en",
                one,
                LifecycleDecision(LifecycleOperation.CANCEL, 0),
            ),
            (
                "Τι ώρα είναι;",
                "el",
                (),
                LifecycleDecision(LifecycleOperation.IGNORE),
            ),
            (
                "Να θυμάσαι ότι προτιμώ τσάι.",
                "el",
                (),
                LifecycleDecision(LifecycleOperation.CREATE),
            ),
            (
                "Διόρθωση: προτιμώ καφέ.",
                "el",
                one,
                LifecycleDecision(LifecycleOperation.REPLACE, 0),
            ),
            (
                "Ξέχασε την προτίμησή μου για ρόφημα.",
                "el",
                one,
                LifecycleDecision(LifecycleOperation.CANCEL, 0),
            ),
        )
        for text, language, active, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    gate.decide(LifecycleObservation(text, language), active),
                    expected,
                )
        self.assertEqual(gate.contract_sha256, EXPECTED_GATE_SHA256)
        self.assertEqual(
            DETERMINISTIC_LIFECYCLE_GATE_CONTRACT_SHA256,
            EXPECTED_GATE_SHA256,
        )

    def test_clear_commands_never_call_the_model_fallback(self) -> None:
        fallback = RejectingFallback()
        writer = DeterministicFirstLifecycleWriter(fallback)
        one = (_candidate(0),)
        decisions = asyncio.run(
            self._decide_all(
                writer,
                (
                    ("Remember that I prefer tea.", "en", ()),
                    ("Correction: I prefer coffee.", "en", one),
                    ("Ξέχνα την προτίμησή μου.", "el", one),
                    ("Tell me the weather.", "en", one),
                ),
            )
        )
        self.assertEqual(
            decisions,
            [
                LifecycleDecision(LifecycleOperation.CREATE),
                LifecycleDecision(LifecycleOperation.REPLACE, 0),
                LifecycleDecision(LifecycleOperation.CANCEL, 0),
                LifecycleDecision(LifecycleOperation.IGNORE),
            ],
        )
        self.assertEqual(fallback.calls, [])
        self.assertEqual(writer.contract_sha256, EXPECTED_DETERMINISTIC_FIRST_SHA256)
        self.assertEqual(
            DETERMINISTIC_FIRST_LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256,
            EXPECTED_DETERMINISTIC_FIRST_SHA256,
        )

    def test_ambiguous_target_or_incomplete_command_uses_fallback(self) -> None:
        fallback = RejectingFallback(
            LifecycleDecision(LifecycleOperation.REPLACE, 1)
        )
        writer = DeterministicFirstLifecycleWriter(fallback)
        active = (_candidate(0), _candidate(1, "Remember that I prefer juice."))
        decision = asyncio.run(
            writer.decide(
                LifecycleObservation("Correction: I prefer coffee.", "en"),
                active,
            )
        )
        self.assertEqual(decision, LifecycleDecision(LifecycleOperation.REPLACE, 1))
        self.assertEqual(len(fallback.calls), 1)

        incomplete = asyncio.run(
            writer.decide(LifecycleObservation("Remember that ...", "en"), ())
        )
        self.assertEqual(incomplete, LifecycleDecision(LifecycleOperation.REPLACE, 1))
        self.assertEqual(len(fallback.calls), 2)

        ambiguous_wording = asyncio.run(
            writer.decide(LifecycleObservation("Actually, coffee is better.", "en"), ())
        )
        self.assertEqual(
            ambiguous_wording,
            LifecycleDecision(LifecycleOperation.REPLACE, 1),
        )
        self.assertEqual(len(fallback.calls), 3)

    def test_keywords_inside_ordinary_text_do_not_mutate_memory(self) -> None:
        fallback = RejectingFallback()
        writer = DeterministicFirstLifecycleWriter(fallback)
        for text in (
            'Translate the phrase "remember that I prefer tea".',
            "I did not ask you to forget anything.",
            "The note says correction: coffee.",
        ):
            with self.subTest(text=text):
                decision = asyncio.run(
                    writer.decide(LifecycleObservation(text, "en"), (_candidate(0),))
                )
                self.assertEqual(decision.operation, LifecycleOperation.IGNORE)
        self.assertEqual(fallback.calls, [])

    def test_full_bilingual_create_replace_cancel_uses_zero_lifecycle_model_calls(
        self,
    ) -> None:
        sequences = (
            (
                "en",
                (
                    "Remember that I prefer tea.",
                    "Correction: I prefer coffee.",
                    "Forget my coffee preference.",
                ),
            ),
            (
                "el",
                (
                    "Να θυμάσαι ότι προτιμώ τσάι.",
                    "Διόρθωση: προτιμώ καφέ.",
                    "Ξέχασε την προτίμησή μου για καφέ.",
                ),
            ),
        )
        for language, texts in sequences:
            with self.subTest(language=language):
                self._assert_full_flow(language, texts)

    def _assert_full_flow(self, language: str, texts: tuple[str, str, str]) -> None:
        store = self.make_store()
        fallback = RejectingFallback()
        writer = DeterministicFirstLifecycleWriter(
            fallback,
            gate=DeterministicLifecycleGateV3(),
        )
        config = OrchestratorConfig.from_mapping(
            {
                "database_path": str(store.path),
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
                "memory_shadow_enabled": True,
            }
        )
        service = OrchestratorService(
            config,
            store,
            BudgetedModelPort(store, StaticModel()),
            memory=AnamnesisShadowAdapter(store, writer),
        )
        tasks = []
        for text in texts:
            task = store.admit_transcript(
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                payload={"text": text, "language": {"code": language}},
            ).task
            assert task is not None
            tasks.append(task)
            self.assertEqual(
                asyncio.run(service.execute_task(task.task_id)).state,
                TaskState.COMPLETED,
            )

        audits = [store.memory_shadow_audit(task.task_id) for task in tasks]
        self.assertEqual(
            [audit["payload"]["proposal"]["operation"] for audit in audits],
            ["upsert", "upsert", "cancel"],
        )
        first_key = audits[0]["payload"]["proposal"]["memory_key"]
        self.assertEqual(audits[1]["payload"]["proposal"]["memory_key"], first_key)
        self.assertEqual(audits[2]["payload"]["proposal"]["memory_key"], first_key)
        self.assertEqual(
            reconstruct_active_shadow_memories(
                store.memory_shadow_history(tasks[0].conversation_id)
            ),
            (),
        )
        self.assertEqual(fallback.calls, [])

    @staticmethod
    async def _decide_all(writer, cases):
        return [
            await writer.decide(LifecycleObservation(text, language), active)
            for text, language, active in cases
        ]


if __name__ == "__main__":
    unittest.main()
