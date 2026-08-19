from __future__ import annotations

import unittest
import uuid
from datetime import timedelta

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.domain import (
    OrchestratorError,
    TaskState,
    canonical_json,
    utc_text,
)
from jarvis_orchestrator.memory import MEMORY_RECORD_TTL_DAYS
from jarvis_orchestrator.memory_crypto import (
    InMemoryMemoryEpochKeyProvider,
    MemoryEnvelopeCipherV1,
    MemoryEnvelopeInvalid,
    MemoryKeyUnavailable,
)
from jarvis_orchestrator.memory_runtime import (
    InjectedMemoryCryptoRuntime,
    ProviderBoundMemoryCryptoRuntime,
)
from jarvis_orchestrator.store import (
    MemoryStorageUnavailable,
    OrchestratorStore,
    _memory_term_sha256,
)


def _admit(store: OrchestratorStore, text: str, *, now=NOW):
    admitted = store.admit_transcript(
        event_id=str(uuid.uuid4()),
        turn_id=str(uuid.uuid4()),
        payload={"text": text, "language": {"code": "en"}},
        now=now,
    )
    assert admitted.task is not None
    return admitted.task


def _executing(store: OrchestratorStore, task, *, now=NOW):
    current = store.transition(
        task.task_id,
        task.epoch,
        TaskState.ADMITTED,
        TaskState.ROUTING,
        now=now,
    )
    current = store.transition(
        current.task_id,
        current.epoch,
        TaskState.ROUTING,
        TaskState.LOADING_CONTEXT,
        now=now,
    )
    return store.transition(
        current.task_id,
        current.epoch,
        TaskState.LOADING_CONTEXT,
        TaskState.EXECUTING,
        now=now,
    )


def _complete_command(store: OrchestratorStore, text: str, *, now=NOW):
    task = _executing(store, _admit(store, text, now=now), now=now)
    return store.complete_explicit_memory_command(
        task.task_id,
        task.epoch,
        enabled=True,
        locale="en-US",
        now=now,
    )


def _complete_disabled_source(store: OrchestratorStore, *, now=NOW):
    task = _executing(
        store,
        _admit(store, "Remember that migration source.", now=now),
        now=now,
    )
    completed, _ = store.complete_explicit_memory_command(
        task.task_id,
        task.epoch,
        enabled=False,
        locale="en-US",
        now=now,
    )
    return completed


def _insert_legacy_record(
    store: OrchestratorStore,
    task,
    *,
    content: str,
    created_at,
) -> dict[str, object]:
    fields = store._memory_content_fields(content)
    with store._lock:
        event_id = str(
            store._connection.execute(
                "SELECT event_id FROM inbox WHERE task_id=?",
                (task.task_id,),
            ).fetchone()[0]
        )
        principal_id = str(
            store._connection.execute(
                "SELECT principal_id FROM conversation_principals "
                "WHERE conversation_id=?",
                (task.conversation_id,),
            ).fetchone()[0]
        )
    record_id = str(uuid.uuid4())
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
    expires_at = created_at + timedelta(days=MEMORY_RECORD_TTL_DAYS)
    with store._transaction() as connection:
        connection.execute(
            """INSERT INTO memory_records_v9_legacy(
               record_id,principal_id,conversation_id,task_id,source_event_id,
               content,content_sha256,normalized_sha256,content_bytes,
               term_count,expires_at,normalizer_version,metadata_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record_id,
                principal_id,
                task.conversation_id,
                task.task_id,
                event_id,
                content,
                fields["content_sha256"],
                fields["normalized_sha256"],
                fields["content_bytes"],
                fields["term_count"],
                utc_text(expires_at),
                fields["normalizer_version"],
                canonical_json(metadata),
                utc_text(created_at),
            ),
        )
        for term in fields["terms"]:
            connection.execute(
                "INSERT INTO memory_terms_v9_legacy("
                "principal_id,term_sha256,record_id) VALUES(?,?,?)",
                (principal_id, _memory_term_sha256(term), record_id),
            )
        connection.execute(
            """UPDATE memory_principal_usage
               SET active_record_count=active_record_count+1,
                   active_content_bytes=active_content_bytes+?
               WHERE principal_id=?""",
            (fields["content_bytes"], principal_id),
        )
    return {
        "record_id": record_id,
        "principal_id": principal_id,
        "content": content,
        "content_sha256": fields["content_sha256"],
        "normalized_sha256": fields["normalized_sha256"],
        "term_sha256s": tuple(_memory_term_sha256(term) for term in fields["terms"]),
        "expires_at": utc_text(expires_at),
    }


class TransactionSpyProvider(InMemoryMemoryEpochKeyProvider):
    def __init__(self) -> None:
        super().__init__()
        self.store: OrchestratorStore | None = None
        self.calls: list[str] = []
        self.transaction_violations: list[str] = []

    def _observe(self, name: str) -> None:
        self.calls.append(name)
        if self.store is not None and self.store._connection.in_transaction:
            self.transaction_violations.append(name)

    def bootstrap(self):
        self._observe("bootstrap")
        return super().bootstrap()

    def active_version(self):
        self._observe("active_version")
        return super().active_version()

    def load(self, version: int):
        self._observe("load")
        return super().load(version)

    def exists(self, version: int):
        self._observe("exists")
        return super().exists(version)


class FreshInstallProvider(InMemoryMemoryEpochKeyProvider):
    """Test a provider whose installation identity appears only at bootstrap."""

    @property
    def install_id(self) -> str:
        with self._lock:
            if self._active is None:
                raise MemoryKeyUnavailable("memory_key_not_initialized")
            return self._install_id


class MemoryStorageV10Tests(StoreCase, unittest.TestCase):
    def memory_store(
        self,
        *,
        prepared: bool,
        provider: InMemoryMemoryEpochKeyProvider | None = None,
    ):
        selected_provider = provider or InMemoryMemoryEpochKeyProvider()
        runtime = InjectedMemoryCryptoRuntime(
            selected_provider.install_id,
            MemoryEnvelopeCipherV1(selected_provider),
        )
        store = self.make_store(memory_crypto=runtime)
        if isinstance(selected_provider, TransactionSpyProvider):
            selected_provider.store = store
        if prepared:
            store.prepare_memory_storage(now=NOW)
        return store, selected_provider, runtime

    def test_constructor_installs_v10_without_provider_or_native_calls(self) -> None:
        provider = TransactionSpyProvider()
        store, _, _ = self.memory_store(prepared=False, provider=provider)
        self.assertEqual(provider.calls, [])
        with store._lock:
            version = store._connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
            state = store._connection.execute(
                "SELECT storage_state FROM memory_crypto_state"
            ).fetchone()[0]
            columns = {
                row[1]
                for row in store._connection.execute(
                    "PRAGMA table_info(memory_records)"
                ).fetchall()
            }
        self.assertEqual(version, "10")
        self.assertEqual(state, "LEGACY_PENDING")
        self.assertNotIn("content", columns)
        self.assertNotIn("content_sha256", columns)
        self.assertNotIn("normalized_sha256", columns)
        self.assertIn("payload_ciphertext", columns)
        self.assertIn("normalized_hmac", columns)

    def test_fresh_provider_identity_is_bound_only_after_journaled_bootstrap(
        self,
    ) -> None:
        provider = FreshInstallProvider()
        runtime = ProviderBoundMemoryCryptoRuntime(MemoryEnvelopeCipherV1(provider))
        store = self.make_store(memory_crypto=runtime)
        with store._lock:
            initial = store._connection.execute(
                "SELECT storage_state,provider_install_id FROM memory_crypto_state"
            ).fetchone()
        self.assertEqual(tuple(initial), ("LEGACY_PENDING", None))
        status = store.prepare_memory_storage(now=NOW)
        self.assertTrue(status["ready"])
        with store._lock:
            bound = store._connection.execute(
                "SELECT provider_install_id FROM memory_crypto_state"
            ).fetchone()[0]
        self.assertEqual(bound, provider.install_id)

    def test_provider_is_never_called_under_transaction_after_prepare(self) -> None:
        provider = TransactionSpyProvider()
        store, _, _ = self.memory_store(prepared=True, provider=provider)
        self.assertTrue(provider.calls)
        self.assertEqual(provider.transaction_violations, [])
        provider.calls.clear()

        first, _ = _complete_command(store, "Remember that my bicycle is Iris.")
        _complete_command(store, "Remember that MY BICYCLE IS IRIS.")
        page = store.memory_record_page(
            conversation_id=first.conversation_id,
            limit=1,
            now=NOW,
        )
        record_id = page["items"][0]["record_id"]
        store.memory_record(record_id)
        store.delete_memory_record_by_id(
            record_id,
            expected_revision=page["memory_revision"],
            request_id=str(uuid.uuid4()),
            now=NOW,
        )
        self.assertEqual(provider.calls, [])
        self.assertEqual(provider.transaction_violations, [])

    def test_cutover_removes_plaintext_and_precomplete_source_digest(self) -> None:
        store, _, _ = self.memory_store(prepared=False)
        source = _complete_disabled_source(store)
        legacy = _insert_legacy_record(
            store,
            source,
            content="private violet bicycle",
            created_at=NOW,
        )

        def stop_after_operation(phase: str) -> None:
            if phase == "operation_prepared":
                raise RuntimeError("crash")

        with self.assertRaisesRegex(RuntimeError, "crash"):
            store.prepare_memory_storage(now=NOW, _crash_hook=stop_after_operation)
        with store._lock:
            source_digest = str(
                store._connection.execute(
                    "SELECT request_sha256 FROM memory_crypto_operations"
                ).fetchone()[0]
            )
        store.prepare_memory_storage(now=NOW)

        with store._lock:
            tables = {
                str(row[0])
                for row in store._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            operation_hash = str(
                store._connection.execute(
                    "SELECT request_sha256 FROM memory_crypto_operations"
                ).fetchone()[0]
            )
            values: list[str] = []
            for table in sorted(tables - {"sqlite_sequence"}):
                quoted = table.replace('"', '""')
                for row in store._connection.execute(
                    f'SELECT * FROM "{quoted}"'
                ).fetchall():
                    values.extend(
                        value.hex() if isinstance(value, bytes) else str(value)
                        for value in row
                        if value is not None
                    )
        joined = "\n".join(values)
        self.assertNotIn("memory_records_v9_legacy", tables)
        self.assertNotIn("memory_terms_v9_legacy", tables)
        self.assertNotEqual(operation_hash, source_digest)
        for forbidden in (
            legacy["content"],
            legacy["content_sha256"],
            legacy["normalized_sha256"],
            source_digest,
            *legacy["term_sha256s"],
        ):
            self.assertNotIn(str(forbidden), joined)
        self.assertEqual(
            store.memory_record(legacy["record_id"])["content"], legacy["content"]
        )

    def test_crash_resume_uses_frozen_cutoff_then_normal_sweep_expires(self) -> None:
        store, _, _ = self.memory_store(prepared=False)
        source = _complete_disabled_source(store)
        legacy = _insert_legacy_record(
            store,
            source,
            content="short lived migration fact",
            created_at=NOW
            - timedelta(days=MEMORY_RECORD_TTL_DAYS)
            + timedelta(seconds=1),
        )

        def crash_after_copy(phase: str) -> None:
            if phase == "record_copied":
                raise RuntimeError("crash-after-copy")

        with self.assertRaisesRegex(RuntimeError, "crash-after-copy"):
            store.prepare_memory_storage(now=NOW, _crash_hook=crash_after_copy)
        retry_time = NOW + timedelta(seconds=2)
        store.prepare_memory_storage(now=retry_time)
        with store._lock:
            encrypted_at_cutover = store._connection.execute(
                "SELECT encrypted_record_count FROM memory_crypto_migration_audits"
            ).fetchone()[0]
            ttl_audits = store._connection.execute(
                "SELECT COUNT(*) FROM memory_record_audits "
                "WHERE record_id=? AND action='TTL_EXPIRED'",
                (legacy["record_id"],),
            ).fetchone()[0]
        self.assertEqual(encrypted_at_cutover, 1)
        self.assertEqual(ttl_audits, 1)
        self.assertIsNone(store.memory_record(legacy["record_id"]))
        self.assertEqual(store.sweep_expired_memory(now=retry_time), 0)

    def test_resume_rejects_self_consistent_tamper_of_already_copied_source(
        self,
    ) -> None:
        store, _, _ = self.memory_store(prepared=False)
        source = _complete_disabled_source(store)
        legacy = _insert_legacy_record(
            store,
            source,
            content="original migrated fact",
            created_at=NOW,
        )

        def crash_after_copy(phase: str) -> None:
            if phase == "record_copied":
                raise RuntimeError("crash-after-copy")

        with self.assertRaisesRegex(RuntimeError, "crash-after-copy"):
            store.prepare_memory_storage(now=NOW, _crash_hook=crash_after_copy)
        replacement = "forged migrated fact"
        fields = store._memory_content_fields(replacement)
        with store._transaction() as connection:
            connection.execute(
                """UPDATE memory_records_v9_legacy
                   SET content=?,content_sha256=?,normalized_sha256=?,
                       content_bytes=?,term_count=? WHERE record_id=?""",
                (
                    replacement,
                    fields["content_sha256"],
                    fields["normalized_sha256"],
                    fields["content_bytes"],
                    fields["term_count"],
                    legacy["record_id"],
                ),
            )
            connection.execute(
                "DELETE FROM memory_terms_v9_legacy WHERE record_id=?",
                (legacy["record_id"],),
            )
            for term in fields["terms"]:
                connection.execute(
                    "INSERT INTO memory_terms_v9_legacy("
                    "principal_id,term_sha256,record_id) VALUES(?,?,?)",
                    (
                        legacy["principal_id"],
                        _memory_term_sha256(term),
                        legacy["record_id"],
                    ),
                )
        with self.assertRaisesRegex(
            MemoryStorageUnavailable,
            "memory_migration_source_changed",
        ):
            store.prepare_memory_storage(now=NOW + timedelta(seconds=1))
        with store._lock:
            state = store._connection.execute(
                "SELECT storage_state FROM memory_crypto_state"
            ).fetchone()[0]
            legacy_count = store._connection.execute(
                "SELECT COUNT(*) FROM memory_records_v9_legacy"
            ).fetchone()[0]
        self.assertEqual(state, "MIGRATING")
        self.assertEqual(legacy_count, 1)

    def test_expired_migration_over_sweep_bound_has_one_tombstone_and_audit(
        self,
    ) -> None:
        store, _, _ = self.memory_store(prepared=False)
        seed = _complete_disabled_source(store)
        with store._lock:
            principal_id = str(
                store._connection.execute(
                    "SELECT principal_id FROM conversation_principals "
                    "WHERE conversation_id=?",
                    (seed.conversation_id,),
                ).fetchone()[0]
            )
        record_created = NOW - timedelta(days=MEMORY_RECORD_TTL_DAYS + 1)
        stamp = utc_text(NOW)
        absolute = utc_text(NOW + timedelta(minutes=15))
        record_ids: list[str] = []
        total_bytes = 0
        with store._transaction() as connection:
            for index in range(257):
                task_id = str(uuid.uuid4())
                turn_id = str(uuid.uuid4())
                response_id = str(uuid.uuid4())
                event_id = str(uuid.uuid4())
                record_id = str(uuid.uuid4())
                content = f"expired migration fact {index}"
                fields = store._memory_content_fields(content)
                metadata = {
                    "conversation_id": seed.conversation_id,
                    "task_id": task_id,
                    "task_epoch": str(seed.epoch),
                    "turn_id": turn_id,
                    "source_event_id": event_id,
                    "language": "en",
                    "_anamnesis_namespace": "jarvis",
                    "_anamnesis_user_id": seed.conversation_id,
                }
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
                connection.execute(
                    """INSERT INTO canonical_turns(
                       turn_id,conversation_id,task_id,user_text,user_language,
                       assistant_text,status,created_at,completed_at
                       ) VALUES(?,?,?,?,?,'','COMPLETED',?,?)""",
                    (
                        turn_id,
                        seed.conversation_id,
                        task_id,
                        "migration source",
                        "en",
                        stamp,
                        stamp,
                    ),
                )
                connection.execute(
                    "INSERT INTO inbox(event_id,semantic_sha256,turn_id,task_id,created_at) "
                    "VALUES(?,?,?,?,?)",
                    (event_id, "0" * 64, turn_id, task_id, stamp),
                )
                connection.execute(
                    """INSERT INTO memory_records_v9_legacy(
                       record_id,principal_id,conversation_id,task_id,
                       source_event_id,content,content_sha256,normalized_sha256,
                       content_bytes,term_count,expires_at,normalizer_version,
                       metadata_json,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        record_id,
                        principal_id,
                        seed.conversation_id,
                        task_id,
                        event_id,
                        content,
                        fields["content_sha256"],
                        fields["normalized_sha256"],
                        fields["content_bytes"],
                        fields["term_count"],
                        utc_text(
                            record_created + timedelta(days=MEMORY_RECORD_TTL_DAYS)
                        ),
                        fields["normalizer_version"],
                        canonical_json(metadata),
                        utc_text(record_created),
                    ),
                )
                for term in fields["terms"]:
                    connection.execute(
                        "INSERT INTO memory_terms_v9_legacy("
                        "principal_id,term_sha256,record_id) VALUES(?,?,?)",
                        (principal_id, _memory_term_sha256(term), record_id),
                    )
                record_ids.append(record_id)
                total_bytes += int(fields["content_bytes"])
            connection.execute(
                "UPDATE memory_principal_usage SET active_record_count=?,"
                "active_content_bytes=? WHERE principal_id=?",
                (257, total_bytes, principal_id),
            )
        store.prepare_memory_storage(now=NOW)
        placeholders = ",".join("?" for _ in record_ids)
        with store._lock:
            tombstones = store._connection.execute(
                f"SELECT COUNT(*) FROM memory_record_tombstones "
                f"WHERE record_id IN ({placeholders})",
                tuple(record_ids),
            ).fetchone()[0]
            audits = store._connection.execute(
                f"SELECT COUNT(*) FROM memory_record_audits "
                f"WHERE action='TTL_EXPIRED' AND record_id IN ({placeholders})",
                tuple(record_ids),
            ).fetchone()[0]
            encrypted = store._connection.execute(
                "SELECT COUNT(*) FROM memory_records"
            ).fetchone()[0]
        self.assertEqual(tombstones, 257)
        self.assertEqual(audits, 257)
        self.assertEqual(encrypted, 0)

    def test_envelope_and_authenticated_binding_tamper_fail_closed(self) -> None:
        store, _, _ = self.memory_store(prepared=True)
        task, _ = _complete_command(store, "Remember that my bicycle is Iris.")
        page = store.memory_record_page(
            conversation_id=task.conversation_id,
            limit=1,
            now=NOW,
        )
        record_id = page["items"][0]["record_id"]
        with store._transaction() as connection:
            value = bytes(
                connection.execute(
                    "SELECT payload_ciphertext FROM memory_records WHERE record_id=?",
                    (record_id,),
                ).fetchone()[0]
            )
            connection.execute(
                "UPDATE memory_records SET payload_ciphertext=? WHERE record_id=?",
                (bytes([value[0] ^ 1]) + value[1:], record_id),
            )
        with self.assertRaises(MemoryEnvelopeInvalid):
            store.memory_record(record_id)

    def test_logical_crud_delete_is_cas_idempotent_and_audited_once(self) -> None:
        store, _, _ = self.memory_store(prepared=True)
        task, _ = _complete_command(store, "Remember that my bicycle is Iris.")
        _complete_command(store, "Remember that my helmet is violet.")
        page = store.memory_record_page(
            conversation_id=task.conversation_id,
            limit=1,
            now=NOW,
        )
        self.assertIsNotNone(page["next_cursor"])
        second_page = store.memory_record_page(
            conversation_id=task.conversation_id,
            limit=1,
            after=page["next_cursor"],
            now=NOW,
        )
        self.assertEqual(len(second_page["items"]), 1)
        record_id = page["items"][0]["record_id"]
        other_record_id = second_page["items"][0]["record_id"]
        request_id = str(uuid.uuid4())
        first = store.delete_memory_record_by_id(
            record_id,
            expected_revision=page["memory_revision"],
            request_id=request_id,
            now=NOW,
        )
        replay = store.delete_memory_record_by_id(
            record_id,
            expected_revision=page["memory_revision"],
            request_id=request_id,
            now=NOW,
        )
        self.assertEqual(replay, first)
        self.assertFalse(first["chat_history_changed"])
        self.assertFalse(first["physical_erasure_claimed"])
        self.assertIsNone(store.memory_record(record_id))
        with self.assertRaises(OrchestratorError) as conflict:
            store.delete_memory_record_by_id(
                record_id,
                expected_revision=first["memory_revision"],
                request_id=str(uuid.uuid4()),
                now=NOW,
            )
        self.assertEqual(conflict.exception.code, "idempotency_conflict")
        with self.assertRaises(OrchestratorError) as reused:
            store.delete_memory_record_by_id(
                other_record_id,
                expected_revision=first["memory_revision"],
                request_id=request_id,
                now=NOW,
            )
        self.assertEqual(reused.exception.code, "idempotency_conflict")
        self.assertIsNotNone(store.memory_record(other_record_id))
        with store._lock:
            tombstones = store._connection.execute(
                "SELECT COUNT(*) FROM memory_record_tombstones WHERE record_id=?",
                (record_id,),
            ).fetchone()[0]
            audit_rows = store._connection.execute(
                "SELECT * FROM memory_record_audits "
                "WHERE record_id=? AND action='CONTROL_DELETED'",
                (record_id,),
            ).fetchall()
        self.assertEqual(tombstones, 1)
        self.assertEqual(len(audit_rows), 1)
        store._require_memory_audit_row(dict(audit_rows[0]))

    def test_source_conversation_delete_tombstones_and_audits_before_purge(
        self,
    ) -> None:
        store, _, _ = self.memory_store(prepared=True)
        source, _ = _complete_command(store, "Remember that my bicycle is Iris.")
        record_id = store.memory_record_page(
            conversation_id=source.conversation_id,
            limit=1,
            now=NOW,
        )["items"][0]["record_id"]
        successor = store.create_conversation(source.epoch, now=NOW)["conversation_id"]
        store.delete_conversation(source.conversation_id, source.epoch, now=NOW)
        self.assertIsNone(store.memory_record(record_id))
        self.assertEqual(
            store.memory_record_page(
                conversation_id=successor,
                limit=1,
                now=NOW,
            )["items"],
            [],
        )
        with store._lock:
            tombstone = dict(
                store._connection.execute(
                    "SELECT * FROM memory_record_tombstones WHERE record_id=?",
                    (record_id,),
                ).fetchone()
            )
            audits = store._connection.execute(
                "SELECT * FROM memory_record_audits WHERE record_id=? "
                "AND action='SOURCE_CONVERSATION_DELETED'",
                (record_id,),
            ).fetchall()
        self.assertEqual(tombstone["reason"], "source_conversation_delete")
        self.assertEqual(len(audits), 1)
        store._require_memory_audit_row(dict(audits[0]))


if __name__ == "__main__":
    unittest.main()
