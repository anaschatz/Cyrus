from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator import operations, private_state
from jarvis_orchestrator.memory_backup import MemoryBackupError, MemoryBackupManifestV1
from jarvis_orchestrator.store import OrchestratorStore


class OperationalRecoveryBoundaryTests(unittest.TestCase):
    def test_malformed_manifest_keeps_typed_error(self):
        for raw in (b'{"x":' + b'[' * 1500 + b'0' + b']' * 1500 + b'}',
                    b'{"x":' + b'9' * 5000 + b'}'):
            with self.subTest(prefix=raw[:10]), self.assertRaises(MemoryBackupError):
                MemoryBackupManifestV1.from_jcs_bytes(raw)

    @unittest.skipUnless(os.name == 'posix', 'POSIX file race')
    def test_control_writer_does_not_truncate_replaced_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'control'
            previous = path.with_suffix('.previous')
            path.write_bytes(b'original')
            path.chmod(0o600)
            real_open = os.open
            def replace_before_open(name, flags, *args, **kwargs):
                if Path(name) == path:
                    path.rename(previous)
                    path.write_bytes(b'replacement must survive')
                    path.chmod(0o600)
                return real_open(name, flags, *args, **kwargs)
            with patch.object(private_state.os, 'open', side_effect=replace_before_open):
                with self.assertRaises(private_state.PrivateStateError):
                    private_state.create_or_replace_private_file(path, b'new')
            self.assertEqual(path.read_bytes(), b'replacement must survive')
            self.assertEqual(previous.read_bytes(), b'original')

    @unittest.skipUnless(os.name == 'posix', 'POSIX FIFO')
    def test_private_read_write_and_harden_reject_fifo_swaps_promptly(self):
        script = '''import os, sys
from pathlib import Path
from unittest.mock import patch
from jarvis_orchestrator import operations as op, private_state as ps
p=Path(sys.argv[1]); mode=sys.argv[2]
p.write_bytes(b"fixture"); p.chmod(0o600)
real_open=os.open
def swapped(name, flags, *args, **kwargs):
    if Path(name)==p:
        p.unlink(); os.mkfifo(p, 0o600)
    return real_open(name, flags, *args, **kwargs)
with patch.object(os, "open", side_effect=swapped):
    try:
        if mode=="write": ps.create_or_replace_private_file(p, b"new")
        elif mode=="harden":
            assert ps._chmod_owned_no_follow(p, "file", 0o600) is False
            sys.exit(0)
        elif mode=="hash": op._sha256_file(p)
        else: op._read_private_bounded(p, maximum=100)
    except (ps.PrivateStateError, op.OperationalError): sys.exit(0)
sys.exit(1)
'''
        for mode in ('write', 'harden', 'hash', 'manifest'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, '-c', script, str(Path(directory) / 'input'), mode],
                                        capture_output=True, timeout=2)
                self.assertEqual(result.returncode, 0, result.stderr.decode())

    def database(self, path):
        store = OrchestratorStore(path)
        store.close()

    @unittest.skipUnless(os.name == 'posix', 'POSIX fchmod')
    def test_temporary_setup_failure_closes_descriptor_and_removes_file(self):
        for mode in ('backup', 'restore'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, backup, destination = (root / name for name in ('source.db', 'backup.db', 'destination.db'))
                self.database(source)
                operations.backup_database(source, backup)
                lock = root / 'restore.lock'
                with operations.ProcessLock(lock):
                    pass
                created = []
                real_mkstemp = tempfile.mkstemp
                def temporary(*args, **kwargs):
                    result = real_mkstemp(*args, **kwargs)
                    created.append(result)
                    return result
                try:
                    with patch.object(operations.tempfile, 'mkstemp', side_effect=temporary), \
                         patch.object(operations.os, 'fchmod', side_effect=OSError('synthetic failure')):
                        with self.assertRaises((OSError, operations.OperationalError)):
                            if mode == 'backup':
                                operations.backup_database(source, destination)
                            else:
                                operations.restore_database(backup, destination, service_lock=lock)
                    self.assertEqual(len(created), 1)
                    with self.assertRaises(OSError):
                        os.fstat(created[0][0])
                    self.assertFalse(Path(created[0][1]).exists())
                    self.assertFalse(destination.exists())
                finally:
                    for descriptor, name in created:
                        try: os.close(descriptor)
                        except OSError: pass
                        Path(name).unlink(missing_ok=True)

    def test_target_connection_failure_closes_source_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.db'
            self.database(source)
            opened = []
            real_connect = sqlite3.connect
            def connect(database, *args, **kwargs):
                if str(database).endswith('.tmp'):
                    raise sqlite3.OperationalError('synthetic target failure')
                connection = real_connect(database, *args, **kwargs)
                opened.append(connection)
                return connection
            try:
                with patch.object(operations.sqlite3, 'connect', side_effect=connect):
                    with self.assertRaises(operations.OperationalError):
                        operations.backup_database(source, Path(directory) / 'backup.db')
                for connection in opened:
                    with self.assertRaises(sqlite3.ProgrammingError):
                        connection.execute('SELECT 1')
            finally:
                for connection in opened: connection.close()

    def test_restore_rechecks_destination_wal_after_lock_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, backup, destination = (root / name for name in ('source.db', 'backup.db', 'destination.db'))
            self.database(source)
            operations.backup_database(source, backup)
            destination.write_bytes(b'original destination')
            destination.chmod(0o600)
            wal = Path(str(destination) + '-wal')
            acquire = operations.ProcessLock.acquire
            def late_wal(lock):
                acquire(lock)
                wal.write_bytes(b'uncheckpointed state')
                wal.chmod(0o600)
            with patch.object(operations.ProcessLock, 'acquire', new=late_wal):
                with self.assertRaisesRegex(operations.OperationalError, 'sqlite_sidecar_present'):
                    operations.restore_database(backup, destination, service_lock=root / 'restore.lock')
            self.assertEqual(destination.read_bytes(), b'original destination')
            self.assertEqual(wal.read_bytes(), b'uncheckpointed state')

    def test_backup_rechecks_destination_wal_before_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / 'source.db', root / 'backup.db'
            self.database(source)
            destination.write_bytes(b'original backup')
            destination.chmod(0o600)
            wal = Path(str(destination) + '-wal')
            digest = operations._sha256_file
            def late_wal(path):
                result = digest(path)
                wal.write_bytes(b'uncheckpointed state')
                wal.chmod(0o600)
                return result
            with patch.object(operations, '_sha256_file', side_effect=late_wal):
                with self.assertRaisesRegex(operations.OperationalError, 'sqlite_sidecar_present'):
                    operations.backup_database(source, destination)
            self.assertEqual(destination.read_bytes(), b'original backup')


if __name__ == '__main__':
    unittest.main()
