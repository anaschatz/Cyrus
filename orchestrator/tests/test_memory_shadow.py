from __future__ import annotations

import asyncio
import hashlib
import unittest
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import _bootstrap  # noqa: F401
from helpers import StoreCase
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import ModelChunk, ModelRequest, ModelUsage, TaskState
from jarvis_orchestrator.memory import (
    MemoryObservation,
    MemoryOperation,
    MemoryProposal,
)
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService

CONTRACT_SHA256 = "a" * 64


class CapturingModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.requests.append(request)
        yield ModelChunk("Έτοιμο.", ModelUsage(input_tokens=4, output_tokens=2))


class ProposingMemory:
    def __init__(self) -> None:
        self.observations: list[MemoryObservation] = []

    async def observe(self, observation: MemoryObservation) -> MemoryProposal:
        self.observations.append(observation)
        return MemoryProposal(
            operation=MemoryOperation.UPSERT,
            memory_key="profile.response_style",
            supersedes_source_event_ids=(),
            summary="Prefers concise responses",
            writer_contract_sha256=CONTRACT_SHA256,
        )


class FailingMemory:
    async def observe(self, observation: MemoryObservation) -> MemoryProposal:
        del observation
        raise RuntimeError("provider contained secret details")


class SlowMemory:
    async def observe(self, observation: MemoryObservation) -> MemoryProposal:
        del observation
        await asyncio.sleep(1)
        raise AssertionError("timeout should cancel this call")


class MalformedMemory:
    async def observe(self, observation: MemoryObservation):
        del observation
        return {"operation": "upsert"}


def config(
    path: Path, *, enabled: bool, timeout_seconds: float = 0.1
) -> OrchestratorConfig:
    return OrchestratorConfig.from_mapping(
        {
            "database_path": str(path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "memory_shadow_enabled": enabled,
            "memory_shadow_timeout_seconds": timeout_seconds,
        }
    )


class MemoryShadowTests(StoreCase, unittest.TestCase):
    @staticmethod
    def _admit_now(store):
        task = store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload={"text": "Γεια σου.", "language": {"code": "el"}},
        ).task
        assert task is not None
        return task

    def test_shadow_proposal_is_audited_after_response_without_prompt_injection(
        self,
    ) -> None:
        store = self.make_store()
        model = CapturingModel()
        memory = ProposingMemory()
        service = OrchestratorService(
            config(store.path, enabled=True),
            store,
            BudgetedModelPort(store, model),
            memory=memory,
        )
        text = "Να απαντάς σύντομα."
        task = store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload={"text": text, "language": {"code": "el"}},
        ).task
        assert task is not None
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual(len(memory.observations), 1)
        self.assertNotIn("Prefers concise responses", str(model.requests[0].messages))
        audit = store.memory_shadow_audit(task.task_id)
        assert audit is not None
        self.assertEqual(audit["event_type"], "memory.shadow.proposed")
        self.assertEqual(audit["payload"]["proposal"]["operation"], "upsert")
        self.assertEqual(
            audit["payload"]["text_sha256"],
            hashlib.sha256(text.encode()).hexdigest(),
        )
        self.assertNotIn(text, str(audit))
        self.assertFalse(audit["payload"]["authoritative_state_changed"])

    def test_shadow_provider_failure_never_changes_completed_response(self) -> None:
        store = self.make_store()
        model = CapturingModel()
        service = OrchestratorService(
            config(store.path, enabled=True),
            store,
            BudgetedModelPort(store, model),
            memory=FailingMemory(),
        )
        task = self._admit_now(store)
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        audit = store.memory_shadow_audit(task.task_id)
        assert audit is not None
        self.assertEqual(audit["event_type"], "memory.shadow.failed")
        self.assertEqual(audit["payload"]["error_code"], "service_error")
        self.assertNotIn("secret details", str(audit))
        history = store.canonical_conversation(task.conversation_id)
        self.assertEqual(history.turns[0].assistant_text, "Έτοιμο.")

    def test_disabled_shadow_mode_makes_no_memory_call_or_audit(self) -> None:
        store = self.make_store()
        model = CapturingModel()
        memory = ProposingMemory()
        service = OrchestratorService(
            config(store.path, enabled=False),
            store,
            BudgetedModelPort(store, model),
            memory=memory,
        )
        task = self._admit_now(store)
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual(memory.observations, [])
        self.assertIsNone(store.memory_shadow_audit(task.task_id))

    def test_shadow_timeout_is_audited_without_changing_response(self) -> None:
        store = self.make_store()
        model = CapturingModel()
        service = OrchestratorService(
            config(store.path, enabled=True, timeout_seconds=0.001),
            store,
            BudgetedModelPort(store, model),
            memory=SlowMemory(),
        )
        task = self._admit_now(store)
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        audit = store.memory_shadow_audit(task.task_id)
        assert audit is not None
        self.assertEqual(audit["event_type"], "memory.shadow.failed")
        self.assertEqual(audit["payload"]["error_code"], "timeout")
        self.assertFalse(audit["payload"]["authoritative_state_changed"])

    def test_malformed_adapter_result_is_only_an_audited_failure(self) -> None:
        store = self.make_store()
        model = CapturingModel()
        service = OrchestratorService(
            config(store.path, enabled=True),
            store,
            BudgetedModelPort(store, model),
            memory=MalformedMemory(),
        )
        task = self._admit_now(store)
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        audit = store.memory_shadow_audit(task.task_id)
        assert audit is not None
        self.assertEqual(audit["event_type"], "memory.shadow.failed")
        self.assertEqual(audit["payload"]["error_code"], "service_error")

    def test_shadow_audit_reservation_is_idempotent(self) -> None:
        store = self.make_store()
        task = self.admitted(
            store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())
        )
        executing = self.executing(store, task)
        store.start_response(executing.task_id, executing.epoch, "el-GR")
        completed = store.complete_response(
            executing.task_id,
            executing.epoch,
            "done",
            chunk_count=0,
            trailing_speech=[],
            last_unit_index=0,
        )
        first = store.begin_memory_shadow_audit(
            completed.task_id,
            completed.epoch,
            source_event_id="source-1",
            text_sha256="b" * 64,
        )
        second = store.begin_memory_shadow_audit(
            completed.task_id,
            completed.epoch,
            source_event_id="source-1",
            text_sha256="b" * 64,
        )
        self.assertTrue(first)
        self.assertFalse(second)

    def test_memory_proposal_contract_rejects_unsafe_shapes(self) -> None:
        with self.assertRaisesRegex(TypeError, "operation is invalid"):
            MemoryProposal(
                operation="upsert",  # type: ignore[arg-type]
                memory_key="profile.response_style",
                supersedes_source_event_ids=(),
                summary="invalid",
                writer_contract_sha256=CONTRACT_SHA256,
            )
        with self.assertRaisesRegex(ValueError, "ignored memory proposal"):
            MemoryProposal(
                operation=MemoryOperation.IGNORE,
                memory_key="profile.response_style",
                supersedes_source_event_ids=(),
                summary="ignored",
                writer_contract_sha256=CONTRACT_SHA256,
            )
        with self.assertRaisesRegex(ValueError, "one active source"):
            MemoryProposal(
                operation=MemoryOperation.CANCEL,
                memory_key="profile.response_style",
                supersedes_source_event_ids=(),
                summary="cancelled",
                writer_contract_sha256=CONTRACT_SHA256,
            )


if __name__ == "__main__":
    unittest.main()
