from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
import test_macos_beta_lease as fixtures
from jarvis_orchestrator import macos_beta_lease as lease, macos_beta_session as session


class LeaseBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MacOSBetaLeaseTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.active = self.fixture.activate()
        self.path = self.active.path

    def test_reader_rejects_deep_json_and_permission_change(self):
        valid = self.path.read_bytes()
        self.path.write_bytes(b"[" * 2000 + b"0" + b"]" * 2000)
        self.assertIsNone(lease._read_signed(self.path, self.active.key))
        self.path.write_bytes(valid)
        read = os.read

        def change_mode(fd, size):
            raw = read(fd, size)
            self.path.chmod(0o644)
            return raw

        with patch.object(os, "read", side_effect=change_mode):
            self.assertIsNone(lease._read_signed(self.path, self.active.key))

    @unittest.skipUnless(os.name == "posix", "POSIX FIFO boundary")
    def test_fifo_substituted_at_open_does_not_block(self):
        code = """
import os, sys
from pathlib import Path
from unittest.mock import patch
from jarvis_orchestrator.macos_beta_lease import _read_signed
path = Path(sys.argv[1]); opened = os.open
def substitute(value, flags, *args, **kwargs):
    if os.fspath(value) == str(path):
        path.unlink(); os.mkfifo(path, 0o600)
    return opened(value, flags, *args, **kwargs)
with patch.object(os, "open", side_effect=substitute):
    assert _read_signed(path, b"k" * 32) is None
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", code, str(self.path)],
                capture_output=True,
                timeout=2,
            )
        except subprocess.TimeoutExpired:
            self.fail("lease FIFO read blocked")
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_writer_preserves_replacement_and_does_not_chmod_it(self):
        replace = os.replace

        def swapped(source, destination):
            replace(source, destination)
            Path(destination).unlink()
            Path(destination).write_bytes(b"foreign lease")
            Path(destination).chmod(0o644)

        with patch.object(os, "replace", side_effect=swapped):
            with self.assertRaises(RuntimeError):
                lease._write_private(self.path, {"new": True})
        self.assertEqual(self.path.read_bytes(), b"foreign lease")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o644)

    def test_writer_failure_does_not_close_a_reused_descriptor(self):
        foreign = self.fixture.root / "foreign"
        foreign.write_bytes(b"keep")
        descriptors = []

        def failed_replace(*args):
            descriptors.append(os.open(foreign, os.O_RDONLY))
            raise OSError("publication failure")

        try:
            with (
                patch.object(os, "replace", side_effect=failed_replace),
                self.assertRaises(OSError),
            ):
                lease._write_private(self.path, {"new": True})
            self.assertEqual(os.fstat(descriptors[0]).st_size, 4)
        finally:
            for fd in descriptors:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def test_close_preserves_a_new_session_published_after_read(self):
        read = lease._read_signed

        def replaced(path, key):
            value = read(path, key)
            lease._write_private(
                path,
                lease._signed({**self.active.core, "session_nonce": "e" * 64}, key),
            )
            return value

        with patch.object(lease, "_read_signed", side_effect=replaced):
            self.active.close()
        self.assertTrue(self.path.exists())
        self.assertEqual(read(self.path, self.active.key)["session_nonce"], "e" * 64)

    def test_heartbeat_samples_time_after_probes_and_validation_rechecks_age(self):
        self.fixture.now[0] += 500_000_000

        def slow(port):
            self.fixture.now[0] += 2_000_000_000
            return self.fixture.listeners[port]

        self.active.listener_pids = slow
        self.active.heartbeat()
        self.assertEqual(self.active.core["heartbeat_unix_ns"], self.fixture.now[0])
        self.assertTrue(self.fixture.valid())
        self.assertFalse(
            lease.validate_live_session_lease(
                self.fixture.stack,
                inspect_process=self.fixture.inspect,
                listener_pids=slow,
                now_ns=lambda: self.fixture.now[0],
                platform="darwin",
            )
        )

    def test_process_reuse_during_listener_inspection_fails_closed(self):
        def replaced(port):
            old = self.fixture.snapshots[4101]
            self.fixture.snapshots[4101] = lease.ProcessSnapshot(
                old.pid, old.parent_pid, old.uid, "e" * 64
            )
            return self.fixture.listeners[port]

        self.assertFalse(
            lease.validate_live_session_lease(
                self.fixture.stack,
                inspect_process=self.fixture.inspect,
                listener_pids=replaced,
                now_ns=lambda: self.fixture.now[0],
                platform="darwin",
            )
        )

    def test_process_snapshot_reuses_kernel_microseconds_and_parent(self):
        class Kernel:
            microseconds = 1
            status = 2

            def proc_pidinfo(self, pid, kind, flags, pointer, size):
                info = ctypes.cast(
                    pointer, ctypes.POINTER(session._ProcBsdInfo)
                ).contents
                info.pbi_pid = pid
                info.pbi_ppid = 99
                info.pbi_uid = os.getuid()
                info.pbi_status = self.status
                info.pbi_start_tvsec = 123
                info.pbi_start_tvusec = self.microseconds
                return size

        kernel = Kernel()
        with (
            patch.object(sys, "platform", "darwin"),
            patch.object(session, "_libproc", return_value=kernel),
        ):
            first = lease._inspect_process(4101)
            kernel.microseconds = 2
            second = lease._inspect_process(4101)
            kernel.status = 5
            self.assertIsNone(lease._inspect_process(4101))
        self.assertIsNotNone(first)
        self.assertEqual(first.parent_pid, 99)
        self.assertNotEqual(first.start_token, second.start_token)

    @unittest.skipUnless(sys.platform == "darwin", "macOS read-only process identity")
    def test_current_process_native_snapshot(self):
        current = lease._inspect_process(os.getpid())
        self.assertIsNotNone(current)
        self.assertEqual(current.pid, os.getpid())
        self.assertEqual(current.parent_pid, os.getppid())
        self.assertEqual(current.uid, os.getuid())


if __name__ == "__main__":
    unittest.main()
