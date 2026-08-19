from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from _bootstrap import ROOT  # noqa: F401

from jarvis_orchestrator.operations import (
    ContentFreeOperationalLog,
    OperationalError,
    ProcessLock,
    backup_database,
    restore_database,
)
from jarvis_orchestrator.private_state import (
    PrivateStateError,
    audit_orchestrator_state,
    create_or_replace_private_file,
    harden_orchestrator_state,
)
from jarvis_orchestrator.store import OrchestratorStore


@contextmanager
def _umask(value: int):
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


@unittest.skipUnless(os.name == "posix", "POSIX owner/mode contract")
class PrivateRuntimeStateTests(unittest.TestCase):
    def test_new_database_log_lock_and_signal_are_exact_owner_only_under_open_umask(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _umask(0):
            root = Path(directory)
            state = root / "private-state"
            database = state / "orchestrator.sqlite3"
            store = OrchestratorStore(database)
            operations = state / "operations"
            log_path = operations / "orchestrator.operations.jsonl"
            lock_path = operations / "orchestrator.lock"
            stop_path = operations / "orchestrator.stop-request"
            logger = ContentFreeOperationalLog(log_path, maximum_bytes=128, backup_count=2)
            for index in range(20):
                logger.emit("mode_probe", count=index)
            logger.close()
            with ProcessLock(lock_path):
                self.assertTrue(lock_path.is_file())
            create_or_replace_private_file(stop_path, b"stop\n")
            store.close()

            self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(operations.stat().st_mode), 0o700)
            for path in (database, lock_path, stop_path, *operations.glob("*.jsonl*")):
                self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o600)
                self.assertEqual(path.lstat().st_uid, os.getuid())

    def test_store_rejects_exposed_parent_database_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir(mode=0o700)
            database = state / "db.sqlite3"
            database.write_bytes(b"")
            database.chmod(0o600)

            state.chmod(0o750)
            with self.assertRaisesRegex(
                PrivateStateError,
                "private_state_directory_permissions_invalid",
            ):
                OrchestratorStore(database)
            state.chmod(0o700)

            database.chmod(0o640)
            with self.assertRaisesRegex(
                PrivateStateError,
                "private_state_file_permissions_invalid",
            ):
                OrchestratorStore(database)
            database.chmod(0o600)

            sidecar = Path(f"{database}-wal")
            sidecar.write_bytes(b"")
            sidecar.chmod(0o644)
            with self.assertRaisesRegex(
                PrivateStateError,
                "private_state_file_permissions_invalid",
            ):
                OrchestratorStore(database)

    def test_database_and_operational_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir(mode=0o700)
            target = root / "target"
            target.write_bytes(b"")
            target.chmod(0o600)
            linked_database = state / "db.sqlite3"
            linked_database.symlink_to(target)
            with self.assertRaisesRegex(PrivateStateError, "symlink_forbidden"):
                OrchestratorStore(linked_database)

            operations = state / "operations"
            operations.mkdir(mode=0o700)
            log = operations / "orchestrator.operations.jsonl"
            log.symlink_to(target)
            with self.assertRaisesRegex(OperationalError, "symlink_forbidden"):
                ContentFreeOperationalLog(log)
            log.unlink()
            rotation = operations / "orchestrator.operations.jsonl.1"
            rotation.symlink_to(target)
            with self.assertRaisesRegex(OperationalError, "symlink_forbidden"):
                ContentFreeOperationalLog(log)
            rotation.unlink()
            lock = operations / "orchestrator.lock"
            lock.symlink_to(target)
            with self.assertRaisesRegex(OperationalError, "symlink_forbidden"):
                ProcessLock(lock).acquire()

    def test_audit_and_harden_are_content_free_and_never_follow_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "private-state"
            operations = state / "operations"
            state.mkdir(mode=0o700)
            operations.mkdir(mode=0o700)
            database = state / "db.sqlite3"
            database.write_bytes(b"fixture")
            log = operations / "orchestrator.operations.jsonl"
            log.write_bytes(b"{}\n")
            outside = root / "outside"
            outside.write_bytes(b"outside")
            state.chmod(0o755)
            operations.chmod(0o750)
            database.chmod(0o644)
            log.chmod(0o640)
            outside.chmod(0o644)
            rotation = operations / "orchestrator.operations.jsonl.1"
            rotation.symlink_to(outside)

            audit = audit_orchestrator_state(database)
            self.assertFalse(audit.ready)
            hardened = harden_orchestrator_state(database)
            self.assertFalse(hardened.ready)
            self.assertEqual(hardened.hardened_directory_count, 2)
            self.assertEqual(hardened.hardened_file_count, 2)
            self.assertEqual(hardened.violation_counts, {"file_symlink": 1})
            self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(operations.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(log.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(outside.stat().st_mode), 0o644)
            encoded = json.dumps(hardened.to_dict(), sort_keys=True)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn("outside", encoded)

            rotation.unlink()
            self.assertTrue(audit_orchestrator_state(database).ready)

    def test_hardener_does_not_chmod_when_current_uid_does_not_own_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            state.mkdir(mode=0o700)
            database = state / "db.sqlite3"
            database.write_bytes(b"fixture")
            state.chmod(0o755)
            original_mode = stat.S_IMODE(state.stat().st_mode)
            with patch(
                "jarvis_orchestrator.private_state.os.getuid",
                return_value=os.getuid() + 1,
            ):
                report = harden_orchestrator_state(database)
            self.assertFalse(report.ready)
            self.assertEqual(report.violation_counts, {"directory_owner": 1})
            self.assertEqual(report.hardened_directory_count, 0)
            self.assertEqual(stat.S_IMODE(state.stat().st_mode), original_mode)

    def test_backup_and_restore_outputs_are_private_and_reject_symlink_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _umask(0):
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            restored = root / "restored.sqlite3"
            lock = root / "restore.lock"
            store = OrchestratorStore(database)
            backup_database(database, backup)
            store.close()
            restore_database(backup, restored, service_lock=lock)
            self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(restored.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(lock.stat().st_mode), 0o600)

            linked = root / "linked-backup.sqlite3"
            linked.symlink_to(backup)
            with self.assertRaisesRegex(OperationalError, "symlink_forbidden"):
                backup_database(database, linked)


if __name__ == "__main__":
    unittest.main()
