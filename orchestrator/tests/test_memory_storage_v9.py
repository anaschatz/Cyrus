from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase

from jarvis_orchestrator.domain import (
    TaskState,
    canonical_json,
    canonical_sha256,
    utc_text,
)
from jarvis_orchestrator.memory import (
    MAX_MEMORY_PRINCIPAL_RECORDS,
    MAX_MEMORY_RECORD_TERMS,
    MEMORY_NORMALIZER_VERSION,
    MEMORY_RECORD_TTL_DAYS,
    ExplicitMemoryCommandStatus,
    MemoryRecallRequest,
    memory_normalized_sha256,
    memory_search_terms,
    normalize_explicit_memory_target,
)
from jarvis_orchestrator.memory_crypto import (
    InMemoryMemoryEpochKeyProvider,
    MemoryEnvelopeCipherV1,
)
from jarvis_orchestrator.memory_runtime import InjectedMemoryCryptoRuntime
from jarvis_orchestrator.store import MemoryStorageUnavailable, OrchestratorStore


def _admit(store: OrchestratorStore, text: str, *, now=NOW):
    result = store.admit_transcript(
        event_id=str(uuid.uuid4()),
        turn_id=str(uuid.uuid4()),
        payload={"text": text, "language": {"code": "en"}},
        now=now,
    )
    assert result.task is not None
    return result.task


def _executing(store: OrchestratorStore, task, *, now=NOW):
    task = store.transition(
        task.task_id, task.epoch, TaskState.ADMITTED, TaskState.ROUTING, now=now
    )
    task = store.transition(
        task.task_id,
        task.epoch,
        TaskState.ROUTING,
        TaskState.LOADING_CONTEXT,
        now=now,
    )
    return store.transition(
        task.task_id,
        task.epoch,
        TaskState.LOADING_CONTEXT,
        TaskState.EXECUTING,
        now=now,
    )


def _complete_command(store: OrchestratorStore, text: str, *, now=NOW):
    task = _executing(store, _admit(store, text, now=now), now=now)
    return store.complete_explicit_memory_command(
        task.task_id, task.epoch, enabled=True, locale="en-US", now=now
    )


def _downgrade_memory_to_v8(store: OrchestratorStore) -> None:
    """Create a faithful legacy memory boundary around the current core DB."""

    with store._lock:
        record_ids = [
            str(row[0])
            for row in store._connection.execute(
                "SELECT record_id FROM memory_records ORDER BY created_at,record_id"
            ).fetchall()
        ]
    records = [store.memory_record(record_id) for record_id in record_ids]
    assert all(record is not None for record in records)
    with store._transaction() as connection:
        connection.execute(
            """CREATE TABLE memory_records_v8_legacy (
               record_id TEXT PRIMARY KEY,
               conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
               task_id TEXT NOT NULL REFERENCES tasks(task_id),
               source_event_id TEXT NOT NULL,
               scoped_user_id TEXT NOT NULL,
               content TEXT NOT NULL,
               content_sha256 TEXT NOT NULL,
               metadata_json TEXT NOT NULL,
               created_at TEXT NOT NULL,
               UNIQUE(conversation_id,source_event_id)
               )"""
        )
        for record in records:
            assert record is not None
            connection.execute(
                """INSERT INTO memory_records_v8_legacy(
                   record_id,conversation_id,task_id,source_event_id,
                   scoped_user_id,content,content_sha256,metadata_json,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    record["record_id"],
                    record["conversation_id"],
                    record["task_id"],
                    record["source_event_id"],
                    f"anamnesis::jarvis::{record['conversation_id']}",
                    record["content"],
                    canonical_sha256({"content": record["content"]}),
                    record["metadata_json"],
                    record["created_at"],
                ),
            )
        connection.execute("DROP TABLE memory_terms")
        connection.execute("DROP TABLE memory_records")
        for table in (
            "memory_record_tombstones",
            "memory_record_audits",
            "memory_crypto_migration_audits",
            "memory_crypto_operations",
            "memory_crypto_state",
            "memory_key_epochs",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.execute(
            "ALTER TABLE memory_records_v8_legacy RENAME TO memory_records"
        )
        connection.execute(
            "CREATE INDEX memory_records_by_scope "
            "ON memory_records(scoped_user_id,created_at,record_id)"
        )
        connection.execute("DROP TABLE conversation_principals")
        connection.execute("DROP TABLE memory_principal_usage")
        connection.execute("DROP TABLE memory_principals")
        connection.execute("DROP TABLE memory_migration_audits")
        connection.execute("DELETE FROM schema_meta WHERE key='memory_last_sweep_at'")
        connection.execute(
            "UPDATE schema_meta SET value='8' WHERE key='schema_version'"
        )


class MemoryStorageV9Tests(StoreCase, unittest.TestCase):
    def make_store(self, **kwargs: object) -> OrchestratorStore:
        provider = InMemoryMemoryEpochKeyProvider()
        runtime = InjectedMemoryCryptoRuntime(
            provider.install_id,
            MemoryEnvelopeCipherV1(provider),
        )
        selected = super().make_store(memory_crypto=runtime, **kwargs)
        selected.prepare_memory_storage(now=NOW)
        selected._test_memory_runtime = runtime  # type: ignore[attr-defined]
        return selected

    def test_disabled_memory_rejects_legacy_schema_without_migration(self) -> None:
        store = self.make_store()
        _complete_command(store, "Remember that DISABLED_LEGACY_SECRET")
        _downgrade_memory_to_v8(store)
        path = store.path
        store.close()

        with self.assertRaises(MemoryStorageUnavailable) as caught:
            OrchestratorStore(path, memory_storage_enabled=False)
        self.assertEqual(caught.exception.code, "memory_disabled_legacy_schema")
        with sqlite3.connect(path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT value FROM schema_meta WHERE key='schema_version'"
                ).fetchone()[0],
                "8",
            )
            self.assertEqual(
                connection.execute("SELECT content FROM memory_records").fetchone()[0],
                "DISABLED_LEGACY_SECRET",
            )

    def test_disabled_existing_database_without_schema_identity_is_untouched(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "identity-missing.sqlite3"
        secret = "IDENTITY_MISSING_LEGACY_SECRET"
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE memory_records_v9_legacy(content TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO memory_records_v9_legacy(content) VALUES(?)",
                (secret,),
            )
        path.chmod(0o600)

        with self.assertRaises(MemoryStorageUnavailable) as caught:
            OrchestratorStore(path, memory_storage_enabled=False)
        self.assertEqual(caught.exception.code, "memory_disabled_legacy_schema")
        with sqlite3.connect(path) as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT name FROM sqlite_master WHERE name='schema_meta'"
                ).fetchone()
            )
            self.assertEqual(
                connection.execute(
                    "SELECT content FROM memory_records_v9_legacy"
                ).fetchone()[0],
                secret,
            )

    def test_disabled_schema_v10_never_reads_or_mutates_legacy_plaintext(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "disabled-v10.sqlite3"
        initial = OrchestratorStore(path)
        task = _admit(initial, "ordinary request")
        source_event_id = initial.task_input(task.task_id)["event_id"]
        secret = "DISABLED_V10_SECRET"
        fields = initial._memory_content_fields(secret)
        with initial._transaction() as connection:
            principal_id = initial._local_owner_principal_tx(connection)
            source_event_id = connection.execute(
                "SELECT event_id FROM inbox WHERE task_id=?", (task.task_id,)
            ).fetchone()[0]
            connection.execute(
                """INSERT INTO memory_records_v9_legacy(
                   record_id,principal_id,conversation_id,task_id,source_event_id,
                   content,content_sha256,normalized_sha256,content_bytes,
                   term_count,expires_at,normalizer_version,metadata_json,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    principal_id,
                    task.conversation_id,
                    task.task_id,
                    source_event_id,
                    secret,
                    fields["content_sha256"],
                    fields["normalized_sha256"],
                    fields["content_bytes"],
                    fields["term_count"],
                    utc_text(NOW + timedelta(days=30)),
                    MEMORY_NORMALIZER_VERSION,
                    "{}",
                    utc_text(NOW),
                ),
            )
        initial.close()

        disabled = OrchestratorStore(path, memory_storage_enabled=False)
        self.addCleanup(disabled.close)
        self.assertEqual(disabled.memory_storage_status()["storage_state"], "DISABLED")
        legacy_accesses: list[tuple[int, str | None]] = []

        def authorizer(
            action: int,
            arg1: str | None,
            _arg2: str | None,
            _db: str | None,
            _trigger: str | None,
        ) -> int:
            if arg1 == "memory_records_v9_legacy":
                legacy_accesses.append((action, arg1))
            return sqlite3.SQLITE_OK

        disabled._connection.set_authorizer(authorizer)
        _admit(disabled, "second ordinary request", now=NOW + timedelta(seconds=1))
        disabled._connection.set_authorizer(None)
        self.assertEqual(legacy_accesses, [])
        disabled.close()
        with sqlite3.connect(path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT content FROM memory_records_v9_legacy"
                ).fetchone()[0],
                secret,
            )

    def test_live_memory_readiness_fails_after_provider_key_loss(self) -> None:
        store = self.make_store()
        ready = store.memory_crypto_readiness()
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["code"], "memory_crypto_ready")
        runtime = store._test_memory_runtime  # type: ignore[attr-defined]
        provider = runtime.cipher.provider
        provider._keys.pop(provider.active_version())

        failed = store.memory_crypto_readiness()
        self.assertFalse(failed["ready"])
        self.assertEqual(failed["code"], "memory_key_not_initialized")
        self.assertNotEqual(failed["state_sha256"], ready["state_sha256"])

    def test_recent_canonical_turns_is_one_bounded_query_with_effects(self) -> None:
        store = self.make_store()
        completed_tasks = []
        for index in range(10):
            selected = NOW + timedelta(seconds=index)
            task = _executing(
                store,
                _admit(store, f"turn {index}", now=selected),
                now=selected,
            )
            response = f"answer {index}"
            store.start_response(task.task_id, task.epoch, "en-US", now=selected)
            store.append_response_event(
                task.task_id,
                task.epoch,
                "response.text.delta",
                {"chunk_index": 0, "text": response},
                now=selected,
            )
            completed_tasks.append(
                store.complete_response(
                    task.task_id,
                    task.epoch,
                    response,
                    chunk_count=1,
                    trailing_speech=[],
                    last_unit_index=0,
                    now=selected,
                )
            )

        latest_effect = {
            "kind": "memory",
            "name": "remember",
            "status": "stored",
            "arguments_sha256": "a" * 64,
            "result_sha256": "b" * 64,
        }
        with store._transaction() as connection:
            connection.execute(
                "UPDATE canonical_turns SET effects_json=? WHERE turn_id=?",
                (
                    canonical_json([latest_effect]),
                    completed_tasks[-1].turn_id,
                ),
            )

        statements: list[str] = []
        with store._lock:
            store._connection.set_trace_callback(statements.append)
        try:
            with patch.object(
                OrchestratorStore,
                "canonical_conversation",
                side_effect=AssertionError("full history must not be loaded"),
            ):
                turns = store.recent_canonical_turns(
                    completed_tasks[-1].conversation_id,
                    limit_pairs=8,
                )
        finally:
            with store._lock:
                store._connection.set_trace_callback(None)

        self.assertEqual(
            [turn.user_text for turn in turns], [f"turn {i}" for i in range(2, 10)]
        )
        self.assertEqual(turns[-1].assistant_text, "answer 9")
        self.assertEqual(turns[-1].effects[0].kind, "memory")
        selects = [
            " ".join(statement.split())
            for statement in statements
            if statement.lstrip().upper().startswith("SELECT")
        ]
        self.assertEqual(len(selects), 1)
        self.assertIn("WHERE conversation_id=", selects[0])
        self.assertIn("status='COMPLETED'", selects[0])
        self.assertIn(
            "ORDER BY created_at DESC,turn_id DESC LIMIT 8",
            selects[0],
        )
        for invalid in (False, 0, 9, 1.5):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "between 1 and 8"),
            ):
                store.recent_canonical_turns(
                    completed_tasks[-1].conversation_id,
                    limit_pairs=invalid,  # type: ignore[arg-type]
                )

    def test_record_and_query_boundaries_are_inclusive_and_term_bounded(
        self,
    ) -> None:
        fields = OrchestratorStore._memory_content_fields("x" * 4096)
        self.assertEqual(fields["content_bytes"], 4096)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            OrchestratorStore._memory_content_fields("x" * 4097)
        bounded_terms = " ".join(
            f"recordterm{index}" for index in range(MAX_MEMORY_RECORD_TERMS)
        )
        fields = OrchestratorStore._memory_content_fields(bounded_terms)
        self.assertEqual(fields["term_count"], MAX_MEMORY_RECORD_TERMS)
        with self.assertRaisesRegex(ValueError, "too many unique terms"):
            OrchestratorStore._memory_content_fields(
                bounded_terms + " recordtermoverflow"
            )
        store = self.make_store()
        completed, result = _complete_command(store, f"Remember that {bounded_terms}.")
        self.assertEqual(result.status, ExplicitMemoryCommandStatus.STORED)
        with store._lock:
            indexed_terms = store._connection.execute(
                """SELECT COUNT(*) FROM memory_terms mt
                   JOIN memory_records mr USING(record_id)
                   WHERE mr.task_id=?""",
                (completed.task_id,),
            ).fetchone()[0]
        self.assertEqual(indexed_terms, MAX_MEMORY_RECORD_TERMS)
        request = MemoryRecallRequest(
            task_id="task",
            conversation_id="conversation",
            epoch=1,
            query="x" * 4096,
            limit=5,
        )
        self.assertEqual(len(request.query.encode("utf-8")), 4096)
        with self.assertRaisesRegex(ValueError, "too large"):
            MemoryRecallRequest(
                task_id="task",
                conversation_id="conversation",
                epoch=1,
                query="x" * 4097,
                limit=5,
            )
        with self.assertRaisesRegex(ValueError, "too many unique terms"):
            MemoryRecallRequest(
                task_id="task",
                conversation_id="conversation",
                epoch=1,
                query=" ".join(f"term{index}" for index in range(33)),
                limit=5,
            )

    def test_local_owner_principal_is_db_stable_across_restart(self) -> None:
        store = self.make_store()
        task = _admit(store, "seed")
        with store._lock:
            principal_before = store._connection.execute(
                "SELECT principal_id FROM conversation_principals "
                "WHERE conversation_id=?",
                (task.conversation_id,),
            ).fetchone()[0]
        path = store.path
        store.close()
        reopened = OrchestratorStore(path)
        self.addCleanup(reopened.close)
        with reopened._lock:
            principal_after = reopened._connection.execute(
                "SELECT principal_id FROM conversation_principals "
                "WHERE conversation_id=?",
                (task.conversation_id,),
            ).fetchone()[0]
            owner_count = reopened._connection.execute(
                "SELECT COUNT(*) FROM memory_principals "
                "WHERE owner_kind='local_owner' AND owner_key='local-owner'"
            ).fetchone()[0]
        self.assertEqual(principal_after, principal_before)
        self.assertEqual(owner_count, 1)

    def test_v8_migration_collapses_oldest_active_duplicate_and_binds_owner(
        self,
    ) -> None:
        store = self.make_store()
        _first_task, first_result = _complete_command(
            store, "Remember that Η γάτα μου λέγεται Ίριδα.", now=NOW
        )
        self.assertEqual(first_result.status, ExplicitMemoryCommandStatus.STORED)
        duplicate_time = NOW + timedelta(minutes=1)
        duplicate_task, duplicate_result = _complete_command(
            store, "Remember that η γατα μου λεγεται ιριδα.", now=duplicate_time
        )
        self.assertEqual(
            duplicate_result.status, ExplicitMemoryCommandStatus.ALREADY_EXISTS
        )
        with store._lock:
            original = store._connection.execute(
                "SELECT * FROM memory_records"
            ).fetchone()
            duplicate_input = store._connection.execute(
                "SELECT event_id FROM inbox WHERE task_id=?",
                (duplicate_task.task_id,),
            ).fetchone()[0]
        _downgrade_memory_to_v8(store)
        duplicate_content = "η γατα μου λεγεται ιριδα"
        duplicate_metadata = {
            "conversation_id": duplicate_task.conversation_id,
            "task_id": duplicate_task.task_id,
            "task_epoch": str(duplicate_task.epoch),
            "turn_id": duplicate_task.turn_id,
            "source_event_id": duplicate_input,
            "language": "en",
            "_anamnesis_namespace": "jarvis",
            "_anamnesis_user_id": duplicate_task.conversation_id,
        }
        with store._transaction() as connection:
            connection.execute(
                """INSERT INTO memory_records(
                   record_id,conversation_id,task_id,source_event_id,scoped_user_id,
                   content,content_sha256,metadata_json,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    duplicate_task.conversation_id,
                    duplicate_task.task_id,
                    duplicate_input,
                    f"anamnesis::jarvis::{duplicate_task.conversation_id}",
                    duplicate_content,
                    canonical_sha256({"content": duplicate_content}),
                    canonical_json(duplicate_metadata),
                    utc_text(duplicate_time),
                ),
            )
        path = store.path
        store.close()
        migrated = OrchestratorStore(path)
        self.addCleanup(migrated.close)
        with migrated._lock:
            version = migrated._connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
            records = migrated._connection.execute(
                "SELECT * FROM memory_records_v9_legacy"
            ).fetchall()
            audit = dict(
                migrated._connection.execute(
                    "SELECT * FROM memory_migration_audits WHERE schema_version=9"
                ).fetchone()
            )
            principal_count = migrated._connection.execute(
                "SELECT COUNT(*) FROM memory_principals"
            ).fetchone()[0]
            binding_count = migrated._connection.execute(
                "SELECT COUNT(*) FROM conversation_principals"
            ).fetchone()[0]
        self.assertEqual(version, "10")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_id"], original["record_id"])
        self.assertEqual(records[0]["normalizer_version"], MEMORY_NORMALIZER_VERSION)
        self.assertEqual(audit["source_schema_version"], 8)
        self.assertEqual(audit["source_record_count"], 2)
        self.assertEqual(audit["migrated_record_count"], 1)
        self.assertEqual(audit["collapsed_duplicate_count"], 1)
        self.assertEqual(principal_count, 1)
        self.assertEqual(binding_count, 1)
        self.assertNotIn("content", json.dumps(audit))

    def test_malformed_v8_migration_rolls_back_without_partial_v9(self) -> None:
        store = self.make_store()
        _complete_command(store, "Remember that my bicycle is Iris.")
        _downgrade_memory_to_v8(store)
        with store._transaction() as connection:
            connection.execute(
                "UPDATE memory_records SET content_sha256=?", ("0" * 64,)
            )
        path = store.path
        store.close()
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            OrchestratorStore(path)
        raw = sqlite3.connect(path)
        try:
            version = raw.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
            columns = {
                row[1] for row in raw.execute("PRAGMA table_info(memory_records)")
            }
            v9_tables = raw.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
                "AND name='memory_principals'"
            ).fetchone()[0]
        finally:
            raw.close()
        self.assertEqual(version, "8")
        self.assertIn("scoped_user_id", columns)
        self.assertEqual(v9_tables, 0)

    def test_v8_migration_rejects_foreign_scope_and_rolls_back(self) -> None:
        store = self.make_store()
        _complete_command(store, "Remember that my bicycle is Iris.")
        _downgrade_memory_to_v8(store)
        with store._transaction() as connection:
            connection.execute(
                "UPDATE memory_records SET scoped_user_id='anamnesis::jarvis::foreign'"
            )
        path = store.path
        store.close()
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            OrchestratorStore(path)
        raw = sqlite3.connect(path)
        try:
            version = raw.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
            scope = raw.execute("SELECT scoped_user_id FROM memory_records").fetchone()[
                0
            ]
        finally:
            raw.close()
        self.assertEqual(version, "8")
        self.assertEqual(scope, "anamnesis::jarvis::foreign")

    def test_v8_migration_rejects_extra_metadata(self) -> None:
        store = self.make_store()
        _complete_command(store, "Remember that my bicycle is Iris.")
        _downgrade_memory_to_v8(store)
        with store._transaction() as connection:
            metadata = json.loads(
                connection.execute(
                    "SELECT metadata_json FROM memory_records"
                ).fetchone()[0]
            )
            metadata["unexpected_sensitive_note"] = "must-not-migrate"
            connection.execute(
                "UPDATE memory_records SET metadata_json=?",
                (canonical_json(metadata),),
            )
        path = store.path
        store.close()
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            OrchestratorStore(path)
        raw = sqlite3.connect(path)
        try:
            version = raw.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
        finally:
            raw.close()
        self.assertEqual(version, "8")

    def test_v8_migration_rejects_noncompleted_source_turn(self) -> None:
        store = self.make_store()
        completed, _ = _complete_command(store, "Remember that my bicycle is Iris.")
        _downgrade_memory_to_v8(store)
        with store._transaction() as connection:
            connection.execute(
                "UPDATE canonical_turns SET status='PENDING' WHERE task_id=?",
                (completed.task_id,),
            )
        path = store.path
        store.close()
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            OrchestratorStore(path)
        raw = sqlite3.connect(path)
        try:
            version = raw.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
        finally:
            raw.close()
        self.assertEqual(version, "8")

    def test_v8_migration_rejects_record_term_overflow_atomically(self) -> None:
        store = self.make_store()
        _complete_command(store, "Remember that my bicycle is Iris.")
        _downgrade_memory_to_v8(store)
        overflow = " ".join(
            f"legacyterm{index}" for index in range(MAX_MEMORY_RECORD_TERMS + 1)
        )
        with store._transaction() as connection:
            connection.execute(
                "UPDATE memory_records SET content=?,content_sha256=?",
                (overflow, canonical_sha256({"content": overflow})),
            )
        path = store.path
        store.close()
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            OrchestratorStore(path)
        raw = sqlite3.connect(path)
        try:
            version = raw.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
            count = raw.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
        finally:
            raw.close()
        self.assertEqual(version, "8")
        self.assertEqual(count, 1)

    def test_v1_failure_rolls_back_all_intermediate_schema_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.sqlite3"
            raw = sqlite3.connect(path)
            raw.execute(
                "CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
            raw.execute("INSERT INTO schema_meta VALUES('schema_version','1')")
            raw.execute("CREATE TABLE memory_records(record_id TEXT PRIMARY KEY)")
            raw.commit()
            raw.close()
            path.chmod(0o600)
            with self.assertRaises((RuntimeError, sqlite3.OperationalError)):
                OrchestratorStore(path)
            check = sqlite3.connect(path)
            try:
                version = check.execute(
                    "SELECT value FROM schema_meta WHERE key='schema_version'"
                ).fetchone()[0]
                has_research = check.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
                    "AND name='research_runs'"
                ).fetchone()[0]
            finally:
                check.close()
            self.assertEqual(version, "1")
            self.assertEqual(has_research, 0)

    def test_ttl_boundary_sweep_and_usage_are_exact(self) -> None:
        store = self.make_store()
        completed, result = _complete_command(
            store, "Remember that my bicycle is Iris.", now=NOW
        )
        self.assertEqual(result.status, ExplicitMemoryCommandStatus.STORED)
        expiry = NOW + timedelta(days=MEMORY_RECORD_TTL_DAYS)
        before = asyncio.run(
            store.search_memory_records(
                conversation_id=completed.conversation_id,
                query="bicycle Iris",
                limit=5,
                now=expiry - timedelta(milliseconds=1),
            )
        )
        at_boundary = asyncio.run(
            store.search_memory_records(
                conversation_id=completed.conversation_id,
                query="bicycle Iris",
                limit=5,
                now=expiry,
            )
        )
        self.assertEqual(len(before), 1)
        self.assertEqual(at_boundary, [])
        self.assertEqual(store.sweep_expired_memory(now=expiry), 1)
        self.assertEqual(
            store.memory_principal_usage(completed.conversation_id),
            {"active_record_count": 0, "active_content_bytes": 0},
        )

    def test_explicit_commands_exclude_records_at_exact_expiry(self) -> None:
        store = self.make_store()
        fact = "my bicycle is Iris"
        _, stored = _complete_command(store, f"Remember that {fact}.", now=NOW)
        self.assertEqual(stored.status, ExplicitMemoryCommandStatus.STORED)
        expiry = NOW + timedelta(days=MEMORY_RECORD_TTL_DAYS)
        _, forgotten = _complete_command(store, f"Forget that {fact}.", now=expiry)
        self.assertEqual(forgotten.status, ExplicitMemoryCommandStatus.NOT_FOUND)
        restored_task, restored = _complete_command(
            store, f"Remember that {fact}.", now=expiry
        )
        self.assertEqual(restored.status, ExplicitMemoryCommandStatus.STORED)
        self.assertEqual(store.memory_record_count(restored_task.conversation_id), 1)

    def test_recall_uses_term_index_and_has_no_recall_all_fallback(self) -> None:
        store = self.make_store()
        completed, _ = _complete_command(
            store, "Remember that my bicycle is called Iris."
        )
        terms = memory_search_terms("bicycle Iris")
        session = store._memory_session_or_unavailable()
        term_hashes = session.blind_term_tokens(terms=terms)
        with store._lock:
            principal_id = store._connection.execute(
                "SELECT principal_id FROM conversation_principals "
                "WHERE conversation_id=?",
                (completed.conversation_id,),
            ).fetchone()[0]
            placeholders = ",".join("?" for _ in term_hashes)
            plan = store._connection.execute(
                f"""EXPLAIN QUERY PLAN
                   SELECT r.record_id,COUNT(*) AS match_count
                   FROM memory_terms AS t
                   INDEXED BY memory_terms_by_principal_term
                   JOIN memory_records AS r
                     ON r.record_id=t.record_id AND r.principal_id=t.principal_id
                    AND r.key_epoch_id=t.key_epoch_id
                    AND r.key_version=t.key_version
                   WHERE t.principal_id=?
                     AND t.key_version=?
                     AND t.term_hmac IN ({placeholders})
                     AND r.expires_at>?
                   GROUP BY r.record_id
                   ORDER BY match_count DESC,r.created_at DESC,r.record_id
                   LIMIT ?""",
                (
                    principal_id,
                    session.key_version,
                    *term_hashes,
                    utc_text(NOW),
                    5,
                ),
            ).fetchall()
        detail = " ".join(str(row[3]) for row in plan)
        self.assertIn("memory_terms_by_principal_term", detail)
        self.assertNotIn("SCAN r", detail)
        no_match = asyncio.run(
            store.search_memory_records(
                conversation_id=completed.conversation_id,
                query="memory recall remember",
                limit=5,
                now=NOW,
            )
        )
        self.assertEqual(no_match, [])

    def test_read_only_recall_cancellation_does_not_interrupt_authority(self) -> None:
        store = self.make_store()
        completed, _ = _complete_command(
            store, "Remember that my bicycle is called Iris."
        )

        async def scenario() -> None:
            started = asyncio.Event()
            release = asyncio.Event()

            async def blocked_to_thread(function):
                started.set()
                await release.wait()
                return function()

            with patch(
                "jarvis_orchestrator.store.asyncio.to_thread",
                side_effect=blocked_to_thread,
            ):
                recall = asyncio.create_task(
                    store.search_memory_records(
                        conversation_id=completed.conversation_id,
                        query="bicycle",
                        limit=5,
                        now=NOW,
                    )
                )
                await started.wait()
                recall.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await recall
                release.set()
                await asyncio.sleep(0)

        asyncio.run(scenario())
        self.assertEqual(
            store.memory_principal_usage(completed.conversation_id)[
                "active_record_count"
            ],
            1,
        )
        self.assertEqual(store.get_task(completed.task_id).state, TaskState.COMPLETED)

    def test_concurrent_dedup_keeps_usage_exact(self) -> None:
        store = self.make_store()
        first = _executing(store, _admit(store, "Remember that first source."))
        first, _ = store.complete_explicit_memory_command(
            first.task_id,
            first.epoch,
            enabled=False,
            locale="en-US",
            now=NOW,
        )
        first_input = store.task_input(first.task_id)["event_id"]
        runtime = store._test_memory_runtime  # type: ignore[attr-defined]
        second_store = OrchestratorStore(store.path, memory_crypto=runtime)
        second_store.prepare_memory_storage(now=NOW)
        self.addCleanup(second_store.close)
        barrier = threading.Barrier(2)

        def insert(
            selected_store: OrchestratorStore,
            task,
            event_id: str,
        ) -> str:
            metadata = {
                "conversation_id": task.conversation_id,
                "task_id": task.task_id,
                "task_epoch": str(task.epoch),
                "turn_id": task.turn_id,
                "source_event_id": event_id,
                "language": "en",
                "_anamnesis_namespace": "jarvis",
                "_anamnesis_user_id": task.conversation_id,
            }
            barrier.wait()
            return str(
                selected_store.add_memory_record(
                    conversation_id=task.conversation_id,
                    task_id=task.task_id,
                    epoch=task.epoch,
                    source_event_id=event_id,
                    scoped_user_id=(f"anamnesis::jarvis::{task.conversation_id}"),
                    content="Η γάτα μου λέγεται Ίριδα",
                    metadata=metadata,
                    now=NOW,
                )["record_id"]
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(
                executor.map(
                    lambda arguments: insert(*arguments),
                    (
                        (store, first, first_input),
                        (second_store, first, first_input),
                    ),
                )
            )
        self.assertEqual(len(set(outcomes)), 1)
        usage = store.memory_principal_usage(first.conversation_id)
        self.assertEqual(usage["active_record_count"], 1)
        self.assertEqual(
            usage["active_content_bytes"],
            len("Η γάτα μου λέγεται Ίριδα".encode()),
        )

    def test_cross_conversation_recall_and_source_deletion_consistency(self) -> None:
        store = self.make_store()
        first, _ = _complete_command(store, "Remember that my bicycle is called Iris.")
        second_conversation = store.create_conversation(first.epoch, now=NOW)[
            "conversation_id"
        ]
        recalled = asyncio.run(
            store.search_memory_records(
                conversation_id=second_conversation,
                query="bicycle Iris",
                limit=5,
                now=NOW,
            )
        )
        self.assertEqual(len(recalled), 1)
        with store._lock:
            principals = {
                row[0]
                for row in store._connection.execute(
                    "SELECT principal_id FROM conversation_principals"
                ).fetchall()
            }
        self.assertEqual(len(principals), 1)
        store.delete_conversation(first.conversation_id, first.epoch, now=NOW)
        self.assertEqual(
            asyncio.run(
                store.search_memory_records(
                    conversation_id=second_conversation,
                    query="bicycle Iris",
                    limit=5,
                    now=NOW,
                )
            ),
            [],
        )
        self.assertEqual(
            store.memory_principal_usage(second_conversation)["active_record_count"],
            0,
        )
        with store._lock:
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM conversation_principals "
                    "WHERE conversation_id=?",
                    (first.conversation_id,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM memory_principals"
                ).fetchone()[0],
                1,
            )

    def test_greek_normalization_golden_is_accent_insensitive(self) -> None:
        accented = "  Η Γάτα μου λέγεται Ίριδα. "
        plain = "η γατα μου λεγεται ιριδα"
        self.assertEqual(
            normalize_explicit_memory_target(accented),
            "η γατα μου λεγεται ιριδα",
        )
        self.assertEqual(
            memory_normalized_sha256(accented), memory_normalized_sha256(plain)
        )
        self.assertEqual(
            memory_search_terms(accented),
            ("γατα", "η", "ιριδα", "λεγεται", "μου"),
        )

    def test_migration_rejects_principal_record_quota_overflow(self) -> None:
        store = self.make_store()
        seed = _admit(store, "seed")
        _downgrade_memory_to_v8(store)
        stamp = utc_text(NOW)
        absolute = utc_text(NOW + timedelta(minutes=5))
        with store._transaction() as connection:
            for index in range(MAX_MEMORY_PRINCIPAL_RECORDS + 1):
                task_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"task-{index}"))
                turn_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"turn-{index}"))
                response_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"response-{index}"))
                event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"event-{index}"))
                content = f"unique fact {index}"
                connection.execute(
                    """INSERT INTO tasks(
                       task_id,conversation_id,turn_id,response_id,epoch,state,
                       admitted_at,absolute_expires_at,created_at,updated_at
                       ) VALUES(?,?,?,?,?,'COMPLETED',?,?,?,?)""",
                    (
                        task_id,
                        seed.conversation_id,
                        turn_id,
                        response_id,
                        seed.epoch,
                        stamp,
                        absolute,
                        stamp,
                        stamp,
                    ),
                )
                metadata = canonical_json(
                    {
                        "conversation_id": seed.conversation_id,
                        "task_id": task_id,
                        "task_epoch": str(seed.epoch),
                        "turn_id": turn_id,
                        "source_event_id": event_id,
                        "language": "en",
                        "_anamnesis_namespace": "jarvis",
                        "_anamnesis_user_id": seed.conversation_id,
                    }
                )
                connection.execute(
                    """INSERT INTO canonical_turns(
                       turn_id,conversation_id,task_id,user_text,user_language,
                       assistant_text,status,created_at,completed_at
                       ) VALUES(?,?,?,?,?,'','COMPLETED',?,?)""",
                    (
                        turn_id,
                        seed.conversation_id,
                        task_id,
                        "seed",
                        "en",
                        stamp,
                        stamp,
                    ),
                )
                connection.execute(
                    "INSERT INTO inbox("
                    "event_id,semantic_sha256,turn_id,task_id,created_at) "
                    "VALUES(?,?,?,?,?)",
                    (event_id, "0" * 64, turn_id, task_id, stamp),
                )
                connection.execute(
                    """INSERT INTO memory_records(
                       record_id,conversation_id,task_id,source_event_id,
                       scoped_user_id,content,content_sha256,metadata_json,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid5(uuid.NAMESPACE_URL, f"memory-{index}")),
                        seed.conversation_id,
                        task_id,
                        event_id,
                        f"anamnesis::jarvis::{seed.conversation_id}",
                        content,
                        canonical_sha256({"content": content}),
                        metadata,
                        stamp,
                    ),
                )
        path = store.path
        store.close()
        with self.assertRaisesRegex(RuntimeError, "quota"):
            OrchestratorStore(path)
        raw = sqlite3.connect(path)
        try:
            version = raw.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
        finally:
            raw.close()
        self.assertEqual(version, "8")


if __name__ == "__main__":
    unittest.main()
