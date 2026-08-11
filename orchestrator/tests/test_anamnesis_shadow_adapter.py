from __future__ import annotations

import asyncio
import copy
import hashlib
import unittest
import uuid
from collections.abc import AsyncIterator

import _bootstrap  # noqa: F401
from helpers import StoreCase
from jarvis_orchestrator.anamnesis_shadow import (
    ANAMNESIS_SHADOW_CONTRACT_SHA256,
    AnamnesisShadowAdapter,
    LifecycleCandidate,
    LifecycleDecision,
    LifecycleObservation,
    LifecycleOperation,
    reconstruct_active_shadow_memories,
)
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import (
    ModelChunk,
    ModelRequest,
    ModelUsage,
    TaskState,
)
from jarvis_orchestrator.memory import MemoryObservation
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService

EXPECTED_CONTRACT_SHA256 = (
    "792494544c50b16dddccf9250c428f59a576b9d5ee4a50f158922f20f6efcbf6"
)


class CapturingModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.requests.append(request)
        yield ModelChunk("Έτοιμο.", ModelUsage(input_tokens=4, output_tokens=2))


class SequencedWriter:
    contract_sha256 = ANAMNESIS_SHADOW_CONTRACT_SHA256

    def __init__(self, *decisions: LifecycleDecision) -> None:
        self._decisions = list(decisions)
        self.observations: list[LifecycleObservation] = []
        self.active_views: list[tuple[LifecycleCandidate, ...]] = []

    async def decide(self, observation, active_memories):
        self.observations.append(observation)
        self.active_views.append(active_memories)
        return self._decisions.pop(0)


def config(path):
    return OrchestratorConfig.from_mapping(
        {
            "database_path": str(path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "memory_shadow_enabled": True,
        }
    )


def admit(store, text: str):
    task = store.admit_transcript(
        event_id=str(uuid.uuid4()),
        turn_id=str(uuid.uuid4()),
        payload={"text": text, "language": {"code": "el"}},
    ).task
    assert task is not None
    return task


class AnamnesisShadowAdapterTests(StoreCase, unittest.TestCase):
    def test_model_facing_lifecycle_contract_is_frozen(self) -> None:
        self.assertEqual(
            ANAMNESIS_SHADOW_CONTRACT_SHA256,
            EXPECTED_CONTRACT_SHA256,
        )
        with self.assertRaises(ValueError):
            LifecycleDecision(LifecycleOperation.CREATE, 0)
        with self.assertRaises(ValueError):
            LifecycleDecision(LifecycleOperation.CANCEL)

    def test_create_replace_cancel_uses_system_owned_identity(self) -> None:
        store = self.make_store()
        model = CapturingModel()
        writer = SequencedWriter(
            LifecycleDecision(LifecycleOperation.CREATE),
            LifecycleDecision(LifecycleOperation.REPLACE, 0),
            LifecycleDecision(LifecycleOperation.CANCEL, 0),
        )
        service = OrchestratorService(
            config(store.path),
            store,
            BudgetedModelPort(store, model),
            memory=AnamnesisShadowAdapter(store, writer),
        )

        first = admit(store, "Να θυμάσαι ότι προτιμώ τσάι.")
        self.assertEqual(
            asyncio.run(service.execute_task(first.task_id)).state,
            TaskState.COMPLETED,
        )
        first_audit = store.memory_shadow_audit(first.task_id)
        assert first_audit is not None
        first_proposal = first_audit["payload"]["proposal"]
        self.assertEqual(first_proposal["operation"], "upsert")
        self.assertEqual(first_proposal["supersedes_source_event_ids"], [])
        self.assertEqual(
            first_proposal["writer_contract_sha256"],
            ANAMNESIS_SHADOW_CONTRACT_SHA256,
        )
        memory_key = first_proposal["memory_key"]

        second = admit(store, "Διόρθωση: προτιμώ καφέ.")
        self.assertEqual(
            asyncio.run(service.execute_task(second.task_id)).state,
            TaskState.COMPLETED,
        )
        second_audit = store.memory_shadow_audit(second.task_id)
        assert second_audit is not None
        second_proposal = second_audit["payload"]["proposal"]
        self.assertEqual(second_proposal["memory_key"], memory_key)
        self.assertEqual(
            second_proposal["supersedes_source_event_ids"],
            [store.task_input(first.task_id)["event_id"]],
        )

        third = admit(store, "Ξέχνα την προτίμησή μου για καφέ.")
        self.assertEqual(
            asyncio.run(service.execute_task(third.task_id)).state,
            TaskState.COMPLETED,
        )
        third_audit = store.memory_shadow_audit(third.task_id)
        assert third_audit is not None
        third_proposal = third_audit["payload"]["proposal"]
        self.assertEqual(third_proposal["operation"], "cancel")
        self.assertEqual(third_proposal["memory_key"], memory_key)
        self.assertEqual(
            third_proposal["supersedes_source_event_ids"],
            [store.task_input(second.task_id)["event_id"]],
        )
        self.assertEqual(
            reconstruct_active_shadow_memories(
                store.memory_shadow_history(first.conversation_id)
            ),
            (),
        )

        self.assertEqual(writer.active_views[0], ())
        self.assertEqual(len(writer.active_views[1]), 1)
        self.assertEqual(
            writer.active_views[1][0].text,
            "Να θυμάσαι ότι προτιμώ τσάι.",
        )
        self.assertEqual(
            writer.active_views[2][0].text,
            "Διόρθωση: προτιμώ καφέ.",
        )
        self.assertFalse(hasattr(writer.active_views[1][0], "memory_key"))
        self.assertFalse(hasattr(writer.active_views[1][0], "source_event_id"))
        self.assertFalse(hasattr(writer.observations[0], "task_id"))
        self.assertFalse(hasattr(writer.observations[0], "conversation_id"))
        self.assertNotIn(memory_key, str(model.requests))

    def test_create_key_is_deterministic_and_raw_text_is_not_in_audit(self) -> None:
        store = self.make_store()
        writer = SequencedWriter(LifecycleDecision(LifecycleOperation.CREATE))
        service = OrchestratorService(
            config(store.path),
            store,
            BudgetedModelPort(store, CapturingModel()),
            memory=AnamnesisShadowAdapter(store, writer),
        )
        text = "Remember that the Atlas project is private."
        task = admit(store, text)
        asyncio.run(service.execute_task(task.task_id))
        audit = store.memory_shadow_audit(task.task_id)
        assert audit is not None
        proposal = audit["payload"]["proposal"]
        self.assertRegex(proposal["memory_key"], r"^memory\.[0-9a-f]{32}$")
        self.assertNotIn(text, str(audit))
        expected_key = "memory." + hashlib.sha256(
            (task.conversation_id + "\0" + store.task_input(task.task_id)["event_id"]).encode(
                "utf-8"
            )
        ).hexdigest()[:32]
        self.assertEqual(proposal["memory_key"], expected_key)
        rebuilt = reconstruct_active_shadow_memories(
            store.memory_shadow_history(task.conversation_id)
        )
        self.assertEqual(rebuilt[0].memory_key, expected_key)

    def test_out_of_range_target_fails_closed_without_changing_response(self) -> None:
        store = self.make_store()
        model = CapturingModel()
        writer = SequencedWriter(LifecycleDecision(LifecycleOperation.REPLACE, 0))
        service = OrchestratorService(
            config(store.path),
            store,
            BudgetedModelPort(store, model),
            memory=AnamnesisShadowAdapter(store, writer),
        )
        task = admit(store, "Διόρθωσέ το.")
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        audit = store.memory_shadow_audit(task.task_id)
        assert audit is not None
        self.assertEqual(audit["event_type"], "memory.shadow.failed")
        self.assertEqual(audit["payload"]["error_code"], "service_error")
        self.assertEqual(
            store.canonical_conversation(task.conversation_id).turns[0].assistant_text,
            "Έτοιμο.",
        )

    def test_corrupt_audit_hash_is_rejected_before_writer_call(self) -> None:
        store = self.make_store()
        creator = SequencedWriter(LifecycleDecision(LifecycleOperation.CREATE))
        task = admit(store, "Remember that blue is preferred.")
        service = OrchestratorService(
            config(store.path),
            store,
            BudgetedModelPort(store, CapturingModel()),
            memory=AnamnesisShadowAdapter(store, creator),
        )
        asyncio.run(service.execute_task(task.task_id))
        corrupt = copy.deepcopy(store.memory_shadow_history(task.conversation_id))
        corrupt[0]["payload"]["proposal_sha256"] = "0" * 64

        class CorruptReader:
            def memory_shadow_history(self, conversation_id):
                del conversation_id
                return corrupt

        writer = SequencedWriter(LifecycleDecision(LifecycleOperation.IGNORE))
        observation = MemoryObservation(
            task_id=str(uuid.uuid4()),
            conversation_id=task.conversation_id,
            turn_id=str(uuid.uuid4()),
            source_event_id=str(uuid.uuid4()),
            text="new fact",
            language="en",
        )
        with self.assertRaisesRegex(ValueError, "proposal hash"):
            asyncio.run(
                AnamnesisShadowAdapter(CorruptReader(), writer).observe(observation)
            )
        self.assertEqual(writer.active_views, [])


if __name__ == "__main__":
    unittest.main()
