from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.scheduler_crypto import SchedulerKeyUnavailable, _WindowsDpapiKeyProvider


class SchedulerKeyFileTests(unittest.TestCase):
    def test_dpapi_reads_only_bounded_regular_key_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = _WindowsDpapiKeyProvider.__new__(_WindowsDpapiKeyProvider)
            provider.root, provider.namespace = Path(temporary), "synthetic-keys"
            provider._unprotect = Mock(return_value=b"a" * 32)
            path = provider._path(1)
            for content in (b"", b"x" * (16 * 1024 + 1), b"wrapped synthetic key"):
                with self.subTest(size=len(content)):
                    path.write_bytes(content)
                    path.chmod(0o600)
                    provider._unprotect.reset_mock()
                    if content == b"wrapped synthetic key":
                        self.assertEqual(provider.load(1), b"a" * 32)
                        provider._unprotect.assert_called_once_with(content)
                    else:
                        with self.assertRaises(SchedulerKeyUnavailable):
                            provider.load(1)
                        provider._unprotect.assert_not_called()
            if os.name == "posix":
                target = provider.root / "target"
                path.rename(target)
                path.symlink_to(target)
                provider._unprotect.reset_mock()
                with self.assertRaises(SchedulerKeyUnavailable):
                    provider.load(1)
                provider._unprotect.assert_not_called()
                self.assertEqual(target.read_bytes(), b"wrapped synthetic key")

    def test_failed_creation_closes_file_and_only_removes_its_own_path(self):
        for replacement in ((False, True) if os.name == "posix" else (False,)):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as temporary:
                provider = _WindowsDpapiKeyProvider.__new__(_WindowsDpapiKeyProvider)
                provider.root, provider.namespace = Path(temporary), "synthetic-keys"
                provider._protect = Mock(return_value=b"synthetic wrapped key")
                destination = provider._path(1)

                def fail_sync(descriptor):
                    if replacement:
                        destination.unlink()
                        destination.write_bytes(b"replacement file")
                    raise OSError("synthetic sync failure")

                with patch("jarvis_orchestrator.scheduler_crypto.os.fsync", side_effect=fail_sync) as sync:
                    with self.assertRaisesRegex(OSError, "synthetic sync failure"):
                        provider.create_next()
                sync.assert_called_once()
                with self.assertRaises(OSError):
                    os.fstat(sync.call_args.args[0])
                if replacement:
                    self.assertEqual(destination.read_bytes(), b"replacement file")
                else:
                    self.assertFalse(destination.exists())
                    self.assertEqual(provider.create_next()[0], 1)
                    self.assertEqual(destination.read_bytes(), b"synthetic wrapped key")

    def test_key_creation_uses_one_version_snapshot_and_never_replaces_a_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = _WindowsDpapiKeyProvider.__new__(_WindowsDpapiKeyProvider)
            provider.root, provider.namespace = Path(temporary), "synthetic-keys"
            provider._protect = Mock(return_value=b"synthetic wrapped key")
            first = provider._path(1)
            first.write_bytes(b"old wrapped key")
            original_versions = provider._versions

            def changing_versions():
                versions = original_versions()
                first.unlink(missing_ok=True)
                return versions

            with patch.object(provider, "_versions", side_effect=changing_versions):
                version, key = provider.create_next()
            self.assertEqual(version, 2)
            self.assertEqual(len(key), 32)
            second = provider._path(2)
            self.assertEqual(second.read_bytes(), b"synthetic wrapped key")
            with patch.object(provider, "_versions", return_value=[1]):
                with self.assertRaises(FileExistsError):
                    provider.create_next()
            self.assertEqual(second.read_bytes(), b"synthetic wrapped key")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO fault injection")
    def test_key_fifo_is_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.mkfifo(root / "synthetic-keys.v1.dpapi", 0o600)
            script = """
from pathlib import Path
import sys
from jarvis_orchestrator.scheduler_crypto import SchedulerKeyUnavailable, _WindowsDpapiKeyProvider
provider = _WindowsDpapiKeyProvider.__new__(_WindowsDpapiKeyProvider)
provider.root, provider.namespace = Path(sys.argv[1]), 'synthetic-keys'
try:
    provider.load(1)
except SchedulerKeyUnavailable:
    raise SystemExit(0)
raise SystemExit(1)
"""
            try:
                result = subprocess.run([sys.executable, "-c", script, str(root)],
                                        capture_output=True, timeout=1, check=False)
            except subprocess.TimeoutExpired:
                self.fail("Key-file read blocked on a FIFO")
            self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
