from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

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
