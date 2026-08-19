from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.jcs import jcs_bytes
from jarvis_orchestrator.memory_backup import (
    MemoryBackupManifestV1,
)
from jarvis_orchestrator.memory_crypto import (
    InMemoryMemoryEpochKeyProvider,
    MemoryEnvelopeCipherV1,
    MemoryEpochProviderAttestationV1,
)
from jarvis_orchestrator.memory_runtime import InjectedMemoryCryptoRuntime
from jarvis_orchestrator.operations import (
    OperationalError,
    backup_database,
    database_diagnostics,
    memory_backup_manifest_path,
    restore_database,
)
from jarvis_orchestrator.store import OrchestratorStore


class MutableAttestor:
    def __init__(
        self,
        attestation: MemoryEpochProviderAttestationV1,
        *,
        present: set[int],
        no_open_connections: list[int] | None = None,
    ) -> None:
        self.attestation = attestation
        self.present = set(present)
        self.no_open_connections = no_open_connections
        self.calls: list[tuple[str, int | None]] = []

    def _guard(self) -> None:
        if self.no_open_connections is not None and self.no_open_connections[0] != 0:
            raise AssertionError("provider called with SQLite connection open")

    def attest(self) -> MemoryEpochProviderAttestationV1:
        self._guard()
        self.calls.append(("attest", None))
        return self.attestation

    def exists(self, version: int) -> bool:
        self._guard()
        self.calls.append(("exists", version))
        return version in self.present


class MemoryBackupOperationsTests(unittest.TestCase):
    def assert_code(self, code: str, function, /, *args, **kwargs) -> None:
        with self.assertRaises(OperationalError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)

    def encrypted_store(
        self,
        path: Path,
    ) -> tuple[OrchestratorStore, InMemoryMemoryEpochKeyProvider]:
        provider = InMemoryMemoryEpochKeyProvider()
        runtime = InjectedMemoryCryptoRuntime(
            install_id=provider.install_id,
            cipher=MemoryEnvelopeCipherV1(provider),
        )
        store = OrchestratorStore(path, memory_crypto=runtime)
        prepared = store.prepare_memory_storage(runtime)
        self.assertTrue(prepared["ready"])
        store.checkpoint()
        return store, provider

    def test_encrypted_backup_sidecar_is_canonical_private_and_restorable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            restored = root / "restored.sqlite3"
            store, provider = self.encrypted_store(database)

            report = backup_database(
                database,
                backup,
                memory_provider_attestor=provider,
            )
            self.assertTrue(report["memory_manifest"])
            sidecar = memory_backup_manifest_path(backup)
            payload = sidecar.read_bytes()
            manifest = MemoryBackupManifestV1.from_jcs_bytes(payload)
            document = json.loads(payload)
            self.assertEqual(payload, jcs_bytes(document))
            self.assertEqual(stat.S_IMODE(sidecar.stat().st_mode), 0o600)
            self.assertEqual(manifest.backup_sha256, report["backup_sha256"])
            self.assertTrue(document["same_install_only"])
            self.assertFalse(document["portable"])
            self.assertFalse(document["transcript_encrypted"])
            self.assertFalse(document["physical_erasure_claimed"])
            self.assertNotIn("content", document)
            self.assertNotIn("hmac", payload.decode("utf-8").lower())
            self.assertNotIn("token", payload.decode("utf-8").lower())

            store.close()
            restore = restore_database(
                backup,
                restored,
                service_lock=root / "restore.lock",
                memory_provider_attestor=provider,
            )
            self.assertTrue(restore["memory_manifest"])
            self.assertTrue(database_diagnostics(restored).ready)

    def test_encrypted_backup_requires_matching_provider_and_every_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            store, provider = self.encrypted_store(database)
            attestation = provider.attest()

            self.assert_code(
                "memory_backup_provider_required",
                backup_database,
                database,
                backup,
            )
            missing = MutableAttestor(attestation, present=set())
            self.assert_code(
                "memory_backup_provider_key_missing",
                backup_database,
                database,
                backup,
                memory_provider_attestor=missing,
            )
            wrong_install = MutableAttestor(
                MemoryEpochProviderAttestationV1(
                    install_id="f" * 64,
                    active_version=attestation.active_version,
                    high_water=attestation.high_water,
                ),
                present={attestation.active_version},
            )
            self.assert_code(
                "memory_backup_provider_install_mismatch",
                backup_database,
                database,
                backup,
                memory_provider_attestor=wrong_install,
            )
            store.close()

    def test_pending_crypto_operation_and_migrating_state_refuse_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            store, provider = self.encrypted_store(database)
            with store._transaction() as connection:
                connection.execute(
                    "UPDATE memory_crypto_operations SET phase='UNCERTAIN',"
                    "completed_at=NULL"
                )
            store.checkpoint()
            self.assert_code(
                "memory_backup_crypto_operation_pending",
                backup_database,
                database,
                backup,
                memory_provider_attestor=provider,
            )
            with store._transaction() as connection:
                connection.execute(
                    "UPDATE memory_crypto_operations SET phase='COMPLETE',"
                    "completed_at=updated_at"
                )
                operation_id = connection.execute(
                    "SELECT operation_id FROM memory_crypto_operations LIMIT 1"
                ).fetchone()[0]
                connection.execute(
                    "UPDATE memory_crypto_state SET active_operation_id=?",
                    (operation_id,),
                )
            store.checkpoint()
            self.assert_code(
                "memory_backup_state_invalid",
                backup_database,
                database,
                backup,
                memory_provider_attestor=provider,
            )
            with store._transaction() as connection:
                connection.execute(
                    "UPDATE memory_crypto_state SET active_operation_id=NULL,"
                    "storage_state='MIGRATING'"
                )
            store.checkpoint()
            self.assert_code(
                "memory_backup_migration_incomplete",
                backup_database,
                database,
                backup,
                memory_provider_attestor=provider,
            )
            store.close()

    def test_legacy_pending_plaintext_is_refused_but_empty_is_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty_database = root / "empty.sqlite3"
            empty_backup = root / "empty-backup.sqlite3"
            empty_store = OrchestratorStore(empty_database)
            empty_report = backup_database(empty_database, empty_backup)
            self.assertFalse(empty_report["memory_manifest"])
            self.assertFalse(memory_backup_manifest_path(empty_backup).exists())
            empty_store.close()

            database = root / "plaintext.sqlite3"
            backup = root / "plaintext-backup.sqlite3"
            store = OrchestratorStore(database)
            admitted = store.admit_transcript(
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                payload={
                    "asr_score": {
                        "calibrated": False,
                        "method": "test",
                        "value": 1.0,
                    },
                    "language": {"code": "en", "score": 1.0},
                    "text": "plaintext legacy memory",
                },
            )
            task = admitted.task
            self.assertIsNotNone(task)
            assert task is not None
            with store._transaction() as connection:
                principal_id = connection.execute(
                    "SELECT principal_id FROM memory_principals"
                ).fetchone()[0]
                source_event_id = connection.execute(
                    "SELECT event_id FROM inbox WHERE task_id=?",
                    (task.task_id,),
                ).fetchone()[0]
                content = "private saved fact"
                connection.execute(
                    """INSERT INTO memory_records_v9_legacy(
                       record_id,principal_id,conversation_id,task_id,
                       source_event_id,content,content_sha256,normalized_sha256,
                       content_bytes,term_count,expires_at,normalizer_version,
                       metadata_json,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid4()),
                        principal_id,
                        task.conversation_id,
                        task.task_id,
                        source_event_id,
                        content,
                        hashlib.sha256(content.encode()).hexdigest(),
                        hashlib.sha256(content.lower().encode()).hexdigest(),
                        len(content.encode()),
                        1,
                        "2030-01-01T00:00:00.000Z",
                        "jarvis.memory-normalizer/1",
                        "{}",
                        "2026-08-19T00:00:00.000Z",
                    ),
                )
            store.checkpoint()
            self.assert_code(
                "memory_backup_legacy_plaintext_refused",
                backup_database,
                database,
                backup,
            )
            self.assertFalse(backup.exists())
            store.close()

    def test_restore_rejects_missing_tampered_or_database_mismatched_sidecar(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            restored = root / "restored.sqlite3"
            store, provider = self.encrypted_store(database)
            backup_database(
                database,
                backup,
                memory_provider_attestor=provider,
            )
            sidecar = memory_backup_manifest_path(backup)
            original = sidecar.read_bytes()
            wal = Path(f"{backup}-wal")
            wal.write_bytes(b"stale")
            os.chmod(wal, 0o600)
            self.assert_code(
                "memory_backup_sqlite_sidecar_present",
                restore_database,
                backup,
                restored,
                service_lock=root / "restore.lock",
                memory_provider_attestor=provider,
            )
            for suffix in ("-wal", "-shm", "-journal"):
                Path(f"{backup}{suffix}").unlink(missing_ok=True)
            sidecar.unlink()
            self.assert_code(
                "memory_backup_manifest_missing",
                restore_database,
                backup,
                restored,
                service_lock=root / "restore.lock",
                memory_provider_attestor=provider,
            )

            sidecar.write_bytes(original)
            os.chmod(sidecar, 0o600)
            tampered = json.loads(original)
            tampered["portable"] = True
            sidecar.write_bytes(jcs_bytes(tampered))
            self.assert_code(
                "memory_backup_manifest_invalid",
                restore_database,
                backup,
                restored,
                service_lock=root / "restore.lock",
                memory_provider_attestor=provider,
            )

            sidecar.write_bytes(original)
            with sqlite3.connect(backup) as connection:
                connection.execute(
                    "UPDATE memory_crypto_state SET memory_revision=memory_revision+1"
                )
                connection.commit()
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            for suffix in ("-wal", "-shm", "-journal"):
                Path(f"{backup}{suffix}").unlink(missing_ok=True)
            self.assert_code(
                "memory_backup_manifest_mismatch",
                restore_database,
                backup,
                restored,
                service_lock=root / "restore.lock",
                memory_provider_attestor=provider,
            )
            store.close()

    def test_restore_missing_key_never_mutates_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            destination = root / "destination.sqlite3"
            store, provider = self.encrypted_store(database)
            backup_database(
                database,
                backup,
                memory_provider_attestor=provider,
            )
            destination_store = OrchestratorStore(destination)
            destination_store.close()
            before = hashlib.sha256(destination.read_bytes()).hexdigest()
            attestation = provider.attest()
            missing = MutableAttestor(attestation, present=set())
            self.assert_code(
                "memory_backup_provider_key_missing",
                restore_database,
                backup,
                destination,
                service_lock=root / "restore.lock",
                memory_provider_attestor=missing,
            )
            after = hashlib.sha256(destination.read_bytes()).hexdigest()
            self.assertEqual(after, before)
            store.close()

    def test_sidecar_publish_failure_is_never_reported_as_committed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            store, provider = self.encrypted_store(database)
            real_replace = os.replace
            call_count = 0

            def fail_second_replace(source, destination):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("simulated manifest publish crash")
                return real_replace(source, destination)

            with patch(
                "jarvis_orchestrator.operations.os.replace",
                side_effect=fail_second_replace,
            ):
                self.assert_code(
                    "backup_commit_uncertain",
                    backup_database,
                    database,
                    backup,
                    memory_provider_attestor=provider,
                )
            self.assertTrue(backup.exists())
            self.assertFalse(memory_backup_manifest_path(backup).exists())
            self.assert_code(
                "memory_backup_manifest_missing",
                restore_database,
                backup,
                root / "restored.sqlite3",
                service_lock=root / "restore.lock",
                memory_provider_attestor=provider,
            )
            store.close()

    def test_restore_publish_error_after_replace_is_commit_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            destination = root / "destination.sqlite3"
            store, provider = self.encrypted_store(database)
            backup_database(
                database,
                backup,
                memory_provider_attestor=provider,
            )
            destination_store = OrchestratorStore(destination)
            destination_store.close()
            real_replace = os.replace

            def replace_then_error(source, target):
                real_replace(source, target)
                raise OSError("simulated post-commit durability failure")

            with patch(
                "jarvis_orchestrator.operations.os.replace",
                side_effect=replace_then_error,
            ):
                self.assert_code(
                    "restore_commit_uncertain",
                    restore_database,
                    backup,
                    destination,
                    service_lock=root / "restore.lock",
                    memory_provider_attestor=provider,
                )
            self.assertEqual(
                hashlib.sha256(destination.read_bytes()).hexdigest(),
                hashlib.sha256(backup.read_bytes()).hexdigest(),
            )
            store.close()

    def test_provider_attestation_runs_only_after_operations_sqlite_closes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            restored = root / "restored.sqlite3"
            store, provider = self.encrypted_store(database)
            counter = [0]
            real_connect = sqlite3.connect

            class TrackingConnection(sqlite3.Connection):
                def __init__(self, *args, **kwargs) -> None:
                    super().__init__(*args, **kwargs)
                    self._counted_open = True
                    counter[0] += 1

                def close(self) -> None:
                    if self._counted_open:
                        self._counted_open = False
                        counter[0] -= 1
                    super().close()

            def tracking_connect(*args, **kwargs):
                kwargs["factory"] = TrackingConnection
                return real_connect(*args, **kwargs)

            attestation = provider.attest()
            guarded = MutableAttestor(
                attestation,
                present={attestation.active_version},
                no_open_connections=counter,
            )
            with patch(
                "jarvis_orchestrator.operations.sqlite3.connect",
                side_effect=tracking_connect,
            ):
                backup_database(
                    database,
                    backup,
                    memory_provider_attestor=guarded,
                )
                restore_database(
                    backup,
                    restored,
                    service_lock=root / "restore.lock",
                    memory_provider_attestor=guarded,
                )
            self.assertEqual(counter[0], 0)
            self.assertGreaterEqual(
                sum(call[0] == "attest" for call in guarded.calls),
                3,
            )
            store.close()


if __name__ == "__main__":
    unittest.main()
