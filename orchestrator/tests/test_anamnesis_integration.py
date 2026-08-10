from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import _bootstrap  # noqa: F401
from helpers import StoreCase

from jarvis_orchestrator.anamnesis_adapter import (
    SUPPORTED_ANAMNESIS_CONTRACT_SHA256,
    AnamnesisMemoryAdapter,
)
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import ModelChunk, ModelRequest, ModelUsage, ReasonCode, TaskState
from jarvis_orchestrator.memory import MemoryObservation, MemoryRecall
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore


def config(path: Path, *, timeout: float = 0.1) -> OrchestratorConfig:
    return OrchestratorConfig.from_mapping(
        {
            "database_path": str(path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "memory_enabled": True,
            "memory_timeout_seconds": timeout,
            "memory_recall_limit": 5,
        }
    )


class CapturingModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.requests.append(request)
        yield ModelChunk("Έτοιμο.", ModelUsage(input_tokens=4, output_tokens=2))


def admit(store: OrchestratorStore, text: str):
    result = store.admit_transcript(
        event_id=str(uuid.uuid4()),
        turn_id=str(uuid.uuid4()),
        payload={"text": text, "language": {"code": "el"}},
    )
    assert result.task is not None
    return result.task


class SlowMemory:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def search(self, request):
        del request
        self.started.set()
        await self.release.wait()
        return MemoryRecall()

    async def remember(self, observation):
        del observation
        raise AssertionError("cancelled tasks cannot write memory")


class TimeoutMemory:
    async def search(self, request):
        del request
        await asyncio.sleep(1)
        return MemoryRecall()

    async def remember(self, observation):
        del observation
        await asyncio.sleep(1)


class FailingMemory:
    async def search(self, request):
        del request
        raise RuntimeError("private provider diagnostic")

    async def remember(self, observation):
        del observation
        raise RuntimeError("private provider diagnostic")


class AnamnesisIntegrationTests(StoreCase, unittest.TestCase):
    def test_schema_v1_is_migrated_transactionally_to_memory_schema_v2(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "orchestrator.sqlite3"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_meta VALUES('schema_version','1')")
        connection.commit()
        connection.close()
        store = OrchestratorStore(path)
        self.addCleanup(store.close)
        with store._lock:
            version = store._connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
            table = store._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_records'"
            ).fetchone()
        self.assertEqual(version, "2")
        self.assertIsNotNone(table)

    def test_adapter_is_pinned_to_real_anamnesis_runtime_contract(self) -> None:
        store = self.make_store()
        adapter = AnamnesisMemoryAdapter(store)
        self.assertEqual(
            adapter.qualification["contract_sha256"],
            SUPPORTED_ANAMNESIS_CONTRACT_SHA256,
        )
        self.assertEqual(adapter.qualification["persistence_owner"], "orchestrator-sqlite")
        self.assertFalse(adapter.qualification["authoritative"])
        self.assertFalse(adapter.qualification["network"])

    def test_completed_explicit_fact_is_recalled_via_memory_local(self) -> None:
        store = self.make_store()
        model = CapturingModel()
        adapter = AnamnesisMemoryAdapter(store)
        service = OrchestratorService(
            config(store.path),
            store,
            BudgetedModelPort(store, model),
            active_memory=adapter,
        )
        first = admit(store, "Να θυμάσαι ότι το αγαπημένο μου χρώμα είναι μπλε.")
        self.assertEqual(asyncio.run(service.execute_task(first.task_id)).state, TaskState.COMPLETED)
        self.assertEqual(store.memory_record_count(first.conversation_id), 1)

        second = admit(store, "Ποιο είναι το αγαπημένο μου χρώμα;")
        self.assertEqual(asyncio.run(service.execute_task(second.task_id)).state, TaskState.COMPLETED)
        prompt = model.requests[-1].messages[-1]["content"]
        self.assertIn("Retrospective memory JSON", prompt)
        self.assertIn("αγαπημένο μου χρώμα είναι μπλε", prompt)
        audit = store.memory_recall_audit(second.task_id)
        assert audit is not None
        self.assertEqual(audit["event_type"], "memory.recall.completed")
        self.assertEqual(audit["payload"]["result_count"], 1)
        with store._lock:
            route = store._connection.execute(
                "SELECT payload_json FROM task_events WHERE task_id=? AND event_type='LOADING_CONTEXT'",
                (second.task_id,),
            ).fetchone()[0]
        self.assertIn('"route":"memory_local"', route)

    def test_irrelevant_or_empty_recall_falls_back_to_local(self) -> None:
        store = self.make_store()
        model = CapturingModel()
        service = OrchestratorService(
            config(store.path),
            store,
            BudgetedModelPort(store, model),
            active_memory=AnamnesisMemoryAdapter(store),
        )
        task = admit(store, "Πες μου ένα αστείο.")
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertNotIn("Retrospective memory JSON", model.requests[-1].messages[-1]["content"])
        audit = store.memory_recall_audit(task.task_id)
        assert audit is not None
        self.assertEqual(audit["event_type"], "memory.recall.empty")
        self.assertEqual(store.memory_record_count(task.conversation_id), 0)

    def test_timeout_is_content_free_and_does_not_break_local_response(self) -> None:
        store = self.make_store()
        model = CapturingModel()
        service = OrchestratorService(
            config(store.path, timeout=0.001),
            store,
            BudgetedModelPort(store, model),
            active_memory=TimeoutMemory(),
        )
        task = admit(store, "Τι θυμάσαι;")
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        audit = store.memory_recall_audit(task.task_id)
        assert audit is not None
        self.assertEqual(audit["event_type"], "memory.recall.failed")
        self.assertEqual(audit["payload"]["error_code"], "timeout")
        self.assertNotIn("Retrospective memory JSON", model.requests[-1].messages[-1]["content"])

    def test_unavailable_provider_falls_back_without_leaking_exception_text(self) -> None:
        store = self.make_store()
        model = CapturingModel()
        service = OrchestratorService(
            config(store.path),
            store,
            BudgetedModelPort(store, model),
            active_memory=FailingMemory(),
        )
        task = admit(store, "Τι θυμάσαι;")
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        audit = store.memory_recall_audit(task.task_id)
        assert audit is not None
        self.assertEqual(audit["payload"]["error_code"], "service_error")
        self.assertNotIn("private provider diagnostic", str(audit))

    def test_cancelled_explicit_fact_is_never_stored(self) -> None:
        store = self.make_store()
        task = admit(store, "Να θυμάσαι ότι ο κωδικός είναι 1234.")
        store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED)
        self.assertEqual(store.memory_record_count(task.conversation_id), 0)

    def test_cancellation_during_recall_fails_epoch_cas_without_model_dispatch(self) -> None:
        async def exercise() -> tuple[TaskState, int]:
            store = self.make_store()
            model = CapturingModel()
            memory = SlowMemory()
            service = OrchestratorService(
                config(store.path),
                store,
                BudgetedModelPort(store, model),
                active_memory=memory,
            )
            task = admit(store, "Τι θυμάσαι για μένα;")
            execution = asyncio.create_task(service.execute_task(task.task_id))
            await memory.started.wait()
            store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED)
            memory.release.set()
            completed = await execution
            return completed.state, len(model.requests)

        state, requests = asyncio.run(exercise())
        self.assertEqual(state, TaskState.CANCELLED)
        self.assertEqual(requests, 0)

    def test_duplicate_write_restart_deletion_and_conversation_isolation(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "orchestrator.sqlite3"
        store = OrchestratorStore(path)
        model = CapturingModel()
        adapter = AnamnesisMemoryAdapter(store)
        service = OrchestratorService(
            config(path), store, BudgetedModelPort(store, model), active_memory=adapter
        )
        task = admit(store, "Remember that my bicycle is called Iris.")
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        observation = MemoryObservation(
            task_id=task.task_id,
            conversation_id=task.conversation_id,
            turn_id=task.turn_id,
            source_event_id=store.task_input(task.task_id)["event_id"],
            text="Remember that my bicycle is called Iris.",
            language=store.task_input(task.task_id)["language"],
            epoch=task.epoch,
        )
        self.assertTrue(asyncio.run(adapter.remember(observation)).stored)
        self.assertEqual(store.memory_record_count(task.conversation_id), 1)
        store.close()

        restarted = OrchestratorStore(path)
        self.addCleanup(restarted.close)
        restarted_adapter = AnamnesisMemoryAdapter(restarted)
        records = restarted.memory_records_for_scope(
            f"anamnesis::jarvis::{task.conversation_id}"
        )
        self.assertEqual([record["content"] for record in records], ["my bicycle is called Iris"])
        restarted_model = CapturingModel()
        restarted_service = OrchestratorService(
            config(path),
            restarted,
            BudgetedModelPort(restarted, restarted_model),
            active_memory=restarted_adapter,
        )
        recall_task = admit(restarted, "What is my bicycle called?")
        self.assertEqual(
            asyncio.run(restarted_service.execute_task(recall_task.task_id)).state,
            TaskState.COMPLETED,
        )
        self.assertIn("Iris", restarted_model.requests[-1].messages[-1]["content"])
        restarted.delete_conversation(
            task.conversation_id, restarted.active_conversation()["epoch"]
        )
        self.assertEqual(restarted.memory_record_count(task.conversation_id), 0)
        new_task = admit(restarted, "What is my bicycle called?")
        new_model = CapturingModel()
        new_service = OrchestratorService(
            config(path),
            restarted,
            BudgetedModelPort(restarted, new_model),
            active_memory=restarted_adapter,
        )
        self.assertEqual(asyncio.run(new_service.execute_task(new_task.task_id)).state, TaskState.COMPLETED)
        self.assertNotIn("Iris", new_model.requests[-1].messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
