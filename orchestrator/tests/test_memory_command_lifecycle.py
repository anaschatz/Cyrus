from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import tempfile
import unittest
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import Mock

import _bootstrap  # noqa: F401

from jarvis_orchestrator.anamnesis_adapter import AnamnesisMemoryAdapter
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import (
    ModelChunk,
    ModelRequest,
    OrchestratorError,
    TaskState,
)
from jarvis_orchestrator.memory import ExplicitMemoryCommandStatus
from jarvis_orchestrator.memory_crypto import (
    InMemoryMemoryEpochKeyProvider,
    MemoryEnvelopeCipherV1,
)
from jarvis_orchestrator.memory_runtime import InjectedMemoryCryptoRuntime
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore
from jarvis_orchestrator.tool_provenance import (
    ContentProvenanceV1,
    ContentSourceKind,
)


def _config(path: Path, *, memory_enabled: bool = True) -> OrchestratorConfig:
    return OrchestratorConfig.from_test_mapping(
        {
            "database_path": str(path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "memory_enabled": memory_enabled,
            "memory_timeout_seconds": 0.1,
            "memory_recall_limit": 5,
        }
    )


def _admit(store: OrchestratorStore, text: str, *, language: str = "en"):
    result = store.admit_transcript(
        event_id=str(uuid.uuid4()),
        turn_id=str(uuid.uuid4()),
        payload={"text": text, "language": {"code": language}},
    )
    assert result.task is not None
    return result.task


def _executing(store: OrchestratorStore, task):
    current = store.transition(
        task.task_id, task.epoch, TaskState.ADMITTED, TaskState.ROUTING
    )
    current = store.transition(
        current.task_id,
        current.epoch,
        TaskState.ROUTING,
        TaskState.LOADING_CONTEXT,
        payload={"route": "memory_command"},
    )
    return store.transition(
        current.task_id,
        current.epoch,
        TaskState.LOADING_CONTEXT,
        TaskState.EXECUTING,
    )


class NeverModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.requests.append(request)
        raise AssertionError("explicit memory commands must never dispatch a model")
        yield  # pragma: no cover


class MemoryCommandLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "orchestrator.sqlite3"
        self.provider = InMemoryMemoryEpochKeyProvider()
        self.runtime = InjectedMemoryCryptoRuntime(
            self.provider.install_id,
            MemoryEnvelopeCipherV1(self.provider),
        )
        self.store = OrchestratorStore(self.path, memory_crypto=self.runtime)
        self.store.prepare_memory_storage()
        self.addCleanup(self.store.close)

    def _service(self, *, enabled: bool = True):
        model = NeverModel()
        memory = AnamnesisMemoryAdapter(self.store) if enabled else None
        service = OrchestratorService(
            _config(self.path, memory_enabled=enabled),
            self.store,
            BudgetedModelPort(self.store, model),
            active_memory=memory,
        )
        return service, model

    def _reservation_count(self, task_id: str) -> int:
        with self.store._lock:
            return int(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM reservations WHERE task_id=?", (task_id,)
                ).fetchone()[0]
            )

    def _response_outbox(self, response_id: str) -> list[tuple[str, str]]:
        with self.store._lock:
            rows = self.store._connection.execute(
                """SELECT event_type,payload_json FROM outbox
                   WHERE response_id=? AND event_type LIKE 'response.%'
                   ORDER BY ordinal""",
                (response_id,),
            ).fetchall()
        return [(row["event_type"], row["payload_json"]) for row in rows]

    def test_remember_is_atomic_deterministic_and_deduplicated(self) -> None:
        service, model = self._service()
        fact = "my bicycle is called Iris"
        first = _admit(self.store, f"Remember that {fact}.")
        completed = asyncio.run(service.execute_task(first.task_id))

        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual(model.requests, [])
        self.assertEqual(self._reservation_count(first.task_id), 0)
        self.assertEqual(self.store.memory_record_count(first.conversation_id), 1)
        audit = self.store.memory_command_audit(first.task_id)
        assert audit is not None
        self.assertEqual(audit["event_type"], "memory.command.remember.stored")
        self.assertEqual(audit["payload"]["status"], "stored")
        self.assertTrue(audit["payload"]["changed"])
        self.assertNotIn(fact, json.dumps(audit, ensure_ascii=False))
        outbox = self._response_outbox(first.response_id)
        self.assertEqual(
            [event_type for event_type, _ in outbox],
            [
                "response.started",
                "response.text.delta",
                "response.text.done",
                "response.speech.commit",
                "response.speech.done",
            ],
        )
        self.assertNotIn(fact, str(outbox))
        canonical = self.store.canonical_conversation(first.conversation_id)
        self.assertEqual(canonical.turns[-1].assistant_text, "Stored in saved memory.")
        self.assertEqual(canonical.turns[-1].effects[-1].status, "stored")
        with self.store._lock:
            provenance = json.loads(
                self.store._connection.execute(
                    """SELECT provenance_json FROM content_provenance
                       WHERE entity_kind='canonical_turn' AND entity_id=?""",
                    (first.turn_id,),
                ).fetchone()[0]
            )
        self.assertEqual(
            {
                (
                    source["kind"],
                    source["source_id"],
                    source["content_sha256"],
                )
                for source in provenance["sources"]
            },
            {
                (
                    "USER_INPUT",
                    first.turn_id,
                    hashlib.sha256(f"Remember that {fact}.".encode()).hexdigest(),
                ),
                (
                    "CODE_TEMPLATE",
                    first.task_id,
                    hashlib.sha256(b"Stored in saved memory.").hexdigest(),
                ),
            },
        )

        duplicate = _admit(
            self.store, "Please remember that MY BICYCLE IS CALLED IRIS."
        )
        self.assertEqual(
            asyncio.run(service.execute_task(duplicate.task_id)).state,
            TaskState.COMPLETED,
        )
        self.assertEqual(model.requests, [])
        self.assertEqual(self._reservation_count(duplicate.task_id), 0)
        self.assertEqual(self.store.memory_record_count(first.conversation_id), 1)
        duplicate_audit = self.store.memory_command_audit(duplicate.task_id)
        assert duplicate_audit is not None
        self.assertEqual(
            duplicate_audit["event_type"],
            "memory.command.remember.already_exists",
        )
        self.assertFalse(duplicate_audit["payload"]["changed"])
        canonical = self.store.canonical_conversation(first.conversation_id)
        self.assertEqual(
            canonical.turns[-1].assistant_text,
            "It was already in saved memory.",
        )

    def test_forget_not_found_and_deleted_are_fail_closed(self) -> None:
        service, model = self._service()
        fact = "my bicycle is called Iris"
        empty = _admit(self.store, f"Forget that {fact}.")
        self.assertEqual(
            asyncio.run(service.execute_task(empty.task_id)).state,
            TaskState.COMPLETED,
        )
        empty_audit = self.store.memory_command_audit(empty.task_id)
        assert empty_audit is not None
        self.assertEqual(empty_audit["payload"]["status"], "not_found")
        self.assertFalse(empty_audit["payload"]["changed"])

        remembered = _admit(self.store, f"Remember that {fact}.")
        asyncio.run(service.execute_task(remembered.task_id))
        duplicate = _admit(self.store, f"Remember that {fact}.")
        self.assertEqual(
            asyncio.run(service.execute_task(duplicate.task_id)).state,
            TaskState.COMPLETED,
        )
        duplicate_audit = self.store.memory_command_audit(duplicate.task_id)
        assert duplicate_audit is not None
        self.assertEqual(duplicate_audit["payload"]["status"], "already_exists")
        self.assertEqual(self.store.memory_record_count(remembered.conversation_id), 1)

        deleted = _admit(self.store, f"Forget that {fact}.")
        self.assertEqual(
            asyncio.run(service.execute_task(deleted.task_id)).state,
            TaskState.COMPLETED,
        )
        deleted_audit = self.store.memory_command_audit(deleted.task_id)
        assert deleted_audit is not None
        self.assertEqual(deleted_audit["payload"]["status"], "deleted")
        self.assertTrue(deleted_audit["payload"]["changed"])
        self.assertEqual(self.store.memory_record_count(remembered.conversation_id), 0)
        canonical = self.store.canonical_conversation(remembered.conversation_id)
        self.assertEqual(
            canonical.turns[-1].assistant_text,
            "Removed from saved memory; chat history is unchanged.",
        )
        self.assertTrue(any(fact in turn.user_text for turn in canonical.turns))
        self.assertEqual(model.requests, [])

    def test_disabled_memory_returns_truthful_deterministic_outcome(self) -> None:
        service, model = self._service(enabled=False)
        fact = "my bicycle is called Iris"
        for command in (f"Remember that {fact}.", f"Forget that {fact}."):
            with self.subTest(command=command):
                task = _admit(self.store, command)
                completed = asyncio.run(service.execute_task(task.task_id))
                self.assertEqual(completed.state, TaskState.COMPLETED)
                self.assertEqual(self._reservation_count(task.task_id), 0)
                audit = self.store.memory_command_audit(task.task_id)
                assert audit is not None
                self.assertEqual(audit["payload"]["status"], "not_enabled")
                self.assertFalse(audit["payload"]["changed"])
                canonical = self.store.canonical_conversation(task.conversation_id)
                self.assertEqual(
                    canonical.turns[-1].assistant_text,
                    "Saved memory is not enabled; I made no change.",
                )
        self.assertEqual(model.requests, [])
        self.assertEqual(self.store.memory_record_count(task.conversation_id), 0)

    def test_unprepared_encrypted_storage_is_unavailable_without_model_or_provider(
        self,
    ) -> None:
        path = Path(self.temporary.name) / "unprepared.sqlite3"
        provider = InMemoryMemoryEpochKeyProvider()
        provider.bootstrap = Mock(wraps=provider.bootstrap)  # type: ignore[method-assign]
        provider.active_version = Mock(  # type: ignore[method-assign]
            wraps=provider.active_version
        )
        provider.load = Mock(wraps=provider.load)  # type: ignore[method-assign]
        runtime = InjectedMemoryCryptoRuntime(
            provider.install_id,
            MemoryEnvelopeCipherV1(provider),
        )
        store = OrchestratorStore(path, memory_crypto=runtime)
        self.addCleanup(store.close)
        model = NeverModel()
        service = OrchestratorService(
            _config(path),
            store,
            BudgetedModelPort(store, model),
            active_memory=AnamnesisMemoryAdapter(store),
        )
        task = _admit(store, "Remember that my bicycle is called Iris.")
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        audit = store.memory_command_audit(task.task_id)
        assert audit is not None
        self.assertEqual(audit["payload"]["status"], "unavailable")
        self.assertFalse(audit["payload"]["changed"])
        self.assertEqual(model.requests, [])
        self.assertEqual(provider.bootstrap.call_count, 0)
        self.assertEqual(provider.active_version.call_count, 0)
        self.assertEqual(provider.load.call_count, 0)

    def test_greek_command_prefixes_are_accent_insensitive_but_target_is_exact(
        self,
    ) -> None:
        service, model = self._service()
        fact = "η γάτα μου λέγεται Ίριδα"
        command_pairs = (
            (
                f"Να θυμάσαι ότι {fact}.",
                f"Ξέχασε ότι {fact}.",
            ),
            (
                f"Να θυμασαι οτι {fact}.",
                f"Ξεχασε οτι {fact}.",
            ),
        )
        for remember_command, forget_command in command_pairs:
            with self.subTest(prefix=remember_command.split(fact, 1)[0]):
                remembered = _admit(
                    self.store,
                    remember_command,
                    language="el-GR",
                )
                completed = asyncio.run(service.execute_task(remembered.task_id))
                self.assertEqual(completed.state, TaskState.COMPLETED)
                self.assertEqual(self._reservation_count(remembered.task_id), 0)
                remembered_audit = self.store.memory_command_audit(remembered.task_id)
                assert remembered_audit is not None
                self.assertEqual(remembered_audit["payload"]["status"], "stored")
                stored_content = self.store.memory_records_matching(
                    conversation_id=remembered.conversation_id,
                    target=fact,
                )[0]["content"]
                self.assertEqual(stored_content, fact)

                forgotten = _admit(
                    self.store,
                    forget_command,
                    language="el-GR",
                )
                completed = asyncio.run(service.execute_task(forgotten.task_id))
                self.assertEqual(completed.state, TaskState.COMPLETED)
                self.assertEqual(self._reservation_count(forgotten.task_id), 0)
                forgotten_audit = self.store.memory_command_audit(forgotten.task_id)
                assert forgotten_audit is not None
                self.assertEqual(forgotten_audit["payload"]["status"], "deleted")
                self.assertEqual(
                    self.store.memory_record_count(remembered.conversation_id),
                    0,
                )
        self.assertEqual(model.requests, [])

    def test_limit_quota_and_termless_commands_are_truthful_and_model_free(
        self,
    ) -> None:
        service, model = self._service()
        cases = (
            (
                f"Remember that {'x' * 4097}",
                ExplicitMemoryCommandStatus.LIMIT_EXCEEDED,
                "That item exceeds the saved-memory size limit; I made no change.",
            ),
            (
                "Remember that !!!",
                ExplicitMemoryCommandStatus.INVALID_CONTENT,
                "That item cannot be safely indexed; I made no change.",
            ),
            (
                "Remember that " + " ".join(f"term{index}" for index in range(129)),
                ExplicitMemoryCommandStatus.INVALID_CONTENT,
                "That item cannot be safely indexed; I made no change.",
            ),
        )
        for command, expected_status, expected_response in cases:
            with self.subTest(status=expected_status.value):
                task = _admit(self.store, command)
                completed = asyncio.run(service.execute_task(task.task_id))
                self.assertEqual(completed.state, TaskState.COMPLETED)
                audit = self.store.memory_command_audit(task.task_id)
                assert audit is not None
                self.assertEqual(audit["payload"]["status"], expected_status.value)
                self.assertFalse(audit["payload"]["changed"])
                self.assertEqual(self._reservation_count(task.task_id), 0)
                self.assertEqual(
                    self.store.canonical_conversation(task.conversation_id)
                    .turns[-1]
                    .assistant_text,
                    expected_response,
                )

        with self.store._transaction() as connection:
            connection.execute(
                "UPDATE memory_principal_usage "
                "SET active_record_count=2000 WHERE principal_id=("
                "SELECT principal_id FROM memory_principals "
                "WHERE owner_kind='local_owner')"
            )
        quota = _admit(self.store, "Remember that a new bounded fact")
        completed = asyncio.run(service.execute_task(quota.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        audit = self.store.memory_command_audit(quota.task_id)
        assert audit is not None
        self.assertEqual(audit["payload"]["status"], "quota_exceeded")
        self.assertFalse(audit["payload"]["changed"])
        self.assertEqual(self._reservation_count(quota.task_id), 0)
        self.assertEqual(
            self.store.canonical_conversation(quota.conversation_id)
            .turns[-1]
            .assistant_text,
            "Saved memory is full; I made no change.",
        )
        self.assertEqual(model.requests, [])
        self.assertEqual(self.store.memory_record_count(quota.conversation_id), 0)

    def test_committed_command_replays_without_duplicate_mutation_or_outbox(
        self,
    ) -> None:
        task = _executing(
            self.store,
            _admit(self.store, "Remember that my bicycle is called Iris."),
        )
        first_task, first_result = self.store.complete_explicit_memory_command(
            task.task_id,
            task.epoch,
            enabled=True,
            locale="en-US",
        )
        before_outbox = self._response_outbox(task.response_id)
        with self.store._lock:
            before_events = int(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM task_events WHERE task_id=?",
                    (task.task_id,),
                ).fetchone()[0]
            )

        self.store.close()
        restarted = OrchestratorStore(
            self.path,
            memory_crypto=self.runtime,
        )
        restarted.prepare_memory_storage()
        self.addCleanup(restarted.close)
        self.store = restarted

        second_task, second_result = self.store.complete_explicit_memory_command(
            task.task_id,
            task.epoch,
            enabled=True,
            locale="en-US",
        )
        self.assertEqual(second_task, first_task)
        self.assertEqual(second_result, first_result)
        self.assertEqual(second_result.status, ExplicitMemoryCommandStatus.STORED)
        self.assertEqual(self._response_outbox(task.response_id), before_outbox)
        self.assertEqual(self.store.memory_record_count(task.conversation_id), 1)
        with self.store._lock:
            after_events = int(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM task_events WHERE task_id=?",
                    (task.task_id,),
                ).fetchone()[0]
            )
            command_events = int(
                self.store._connection.execute(
                    """SELECT COUNT(*) FROM task_events
                       WHERE task_id=? AND event_type LIKE 'memory.command.%'""",
                    (task.task_id,),
                ).fetchone()[0]
            )
        self.assertEqual(after_events, before_events)
        self.assertEqual(command_events, 1)

    def test_tampered_command_event_type_cannot_replay(self) -> None:
        task = _executing(
            self.store,
            _admit(self.store, "Remember that my bicycle is called Iris."),
        )
        self.store.complete_explicit_memory_command(
            task.task_id,
            task.epoch,
            enabled=True,
            locale="en-US",
        )
        with self.store._transaction() as connection:
            connection.execute(
                """UPDATE task_events
                   SET event_type='memory.command.remember.already_exists'
                   WHERE task_id=? AND event_type='memory.command.remember.stored'""",
                (task.task_id,),
            )

        with self.assertRaises(OrchestratorError) as raised:
            self.store.complete_explicit_memory_command(
                task.task_id,
                task.epoch,
                enabled=True,
                locale="en-US",
            )
        self.assertEqual(raised.exception.code, "memory_command_replay_invalid")

    def test_tampered_response_journal_cannot_replay(self) -> None:
        task = _executing(
            self.store,
            _admit(self.store, "Remember that my bicycle is called Iris."),
        )
        self.store.complete_explicit_memory_command(
            task.task_id,
            task.epoch,
            enabled=True,
            locale="en-US",
        )
        with self.store._transaction() as connection:
            connection.execute(
                """UPDATE outbox SET payload_json=?
                   WHERE response_id=? AND event_type='response.text.delta'""",
                (
                    json.dumps(
                        {"chunk_index": 0, "text": "Tampered."},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    task.response_id,
                ),
            )

        with self.assertRaises(OrchestratorError) as raised:
            self.store.complete_explicit_memory_command(
                task.task_id,
                task.epoch,
                enabled=True,
                locale="en-US",
            )
        self.assertEqual(
            raised.exception.code,
            "memory_command_response_journal_invalid",
        )

    def test_forged_self_consistent_provenance_cannot_replay(self) -> None:
        user_text = "Remember that my bicycle is called Iris."
        task = _executing(self.store, _admit(self.store, user_text))
        completed, _ = self.store.complete_explicit_memory_command(
            task.task_id,
            task.epoch,
            enabled=True,
            locale="en-US",
        )
        canonical = self.store.canonical_conversation(task.conversation_id)
        response_text = canonical.turns[-1].assistant_text
        forged = ContentProvenanceV1.derive(
            (
                ContentProvenanceV1.source(
                    ContentSourceKind.USER_INPUT,
                    task.turn_id,
                    user_text,
                ),
                ContentProvenanceV1.source(
                    ContentSourceKind.CODE_TEMPLATE,
                    task.task_id,
                    response_text,
                ),
                ContentProvenanceV1.source(
                    ContentSourceKind.USER_INPUT,
                    "unrelated-forged-source",
                    "self-consistent but unrelated",
                ),
            )
        )
        with self.store._transaction() as connection:
            connection.execute(
                """UPDATE content_provenance
                   SET provenance_json=?,provenance_sha256=?,policy_label=?,
                       egress_disposition=?
                   WHERE entity_kind='canonical_turn' AND entity_id=?""",
                (
                    json.dumps(
                        forged.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    forged.sha256,
                    forged.policy_label.value,
                    forged.egress.value,
                    task.turn_id,
                ),
            )

        with self.assertRaises(OrchestratorError) as raised:
            self.store.complete_explicit_memory_command(
                completed.task_id,
                completed.epoch,
                enabled=True,
                locale="en-US",
            )
        self.assertEqual(raised.exception.code, "memory_command_replay_invalid")

    def test_transaction_failure_commits_neither_memory_nor_response(self) -> None:
        task = _executing(
            self.store,
            _admit(self.store, "Remember that my bicycle is called Iris."),
        )
        with self.store._lock:
            self.store._connection.executescript(
                """CREATE TRIGGER abort_memory_command
                   BEFORE INSERT ON task_events
                   WHEN NEW.event_type LIKE 'memory.command.%'
                   BEGIN
                     SELECT RAISE(ABORT, 'synthetic memory command failure');
                   END;"""
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.complete_explicit_memory_command(
                task.task_id,
                task.epoch,
                enabled=True,
                locale="en-US",
            )
        self.assertEqual(self.store.get_task(task.task_id).state, TaskState.EXECUTING)
        self.assertEqual(self.store.memory_record_count(task.conversation_id), 0)
        self.assertIsNone(self.store.memory_command_audit(task.task_id))
        self.assertEqual(self._response_outbox(task.response_id), [])
        with self.store._lock:
            status = self.store._connection.execute(
                "SELECT status,assistant_text FROM canonical_turns WHERE task_id=?",
                (task.task_id,),
            ).fetchone()
            self.store._connection.execute("DROP TRIGGER abort_memory_command")
        self.assertEqual((status["status"], status["assistant_text"]), ("PENDING", ""))

        completed, result = self.store.complete_explicit_memory_command(
            task.task_id,
            task.epoch,
            enabled=True,
            locale="en-US",
        )
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual(result.status, ExplicitMemoryCommandStatus.STORED)
        self.assertEqual(self.store.memory_record_count(task.conversation_id), 1)

    def test_service_transaction_failure_never_acknowledges_memory(self) -> None:
        service, model = self._service()
        task = _admit(self.store, "Remember that my bicycle is called Iris.")
        with self.store._lock:
            self.store._connection.executescript(
                """CREATE TRIGGER abort_service_memory_command
                   BEFORE INSERT ON task_events
                   WHEN NEW.event_type LIKE 'memory.command.%'
                   BEGIN
                     SELECT RAISE(ABORT, 'synthetic memory command failure');
                   END;"""
            )

        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.FAILED)
        self.assertEqual(model.requests, [])
        self.assertEqual(self._reservation_count(task.task_id), 0)
        self.assertEqual(self.store.memory_record_count(task.conversation_id), 0)
        self.assertIsNone(self.store.memory_command_audit(task.task_id))
        response_events = self._response_outbox(task.response_id)
        self.assertEqual(
            [event_type for event_type, _ in response_events],
            ["response.cancelled"],
        )
        self.assertNotIn("Iris", str(response_events))
        with self.store._lock:
            terminal_events = self.store._connection.execute(
                """SELECT event_type,payload_json FROM outbox
                   WHERE response_id=? ORDER BY ordinal""",
                (task.response_id,),
            ).fetchall()
        self.assertEqual(
            [row["event_type"] for row in terminal_events],
            ["server.ack", "server.error", "response.cancelled"],
        )
        self.assertNotIn("Iris", str([dict(row) for row in terminal_events]))


if __name__ == "__main__":
    unittest.main()
