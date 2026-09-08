from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import test_stack_migration as fixtures
from jarvis_orchestrator import stack_migration as migration


class StackMigrationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StackMigrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_json_and_option_failures_remain_typed(self):
        for raw in (b"[" * 2000 + b"0" + b"]" * 2000, b'{"number":1e10000}'):
            with self.subTest(size=len(raw)):
                self.fixture.legacy.write_bytes(raw)
                with self.assertRaises(migration.StackMigrationError):
                    migration._strict_private_json(self.fixture.legacy, "unavailable")
        self.fixture.legacy.write_bytes(fixtures.canonical(self.fixture.document()))
        for key in ({}, []):
            with (
                self.subTest(key=key),
                self.assertRaises(migration.StackMigrationError),
            ):
                migration.prepare_auth_v2_stack(
                    self.fixture.legacy, self.fixture.output, macos_ptt_key=key
                )
        self.assertFalse(self.fixture.output.exists())

    def test_private_reader_rejects_mid_read_mutation(self):
        path = self.fixture.legacy
        real_read = os.read
        changed = False

        def mutate(fd, size):
            nonlocal changed
            value = real_read(fd, size)
            if value and not changed:
                changed = True
                path.write_bytes(b"changed")
            return value

        with patch.object(os, "read", mutate):
            with self.assertRaises(migration.StackMigrationError):
                migration._read_private_bytes(path, "unavailable")

    @unittest.skipUnless(os.name == "posix", "POSIX ownership")
    def test_foreign_owner_is_rejected_before_migration(self):
        with patch.object(
            os, "geteuid", return_value=self.fixture.legacy.stat().st_uid + 1
        ):
            with self.assertRaises(migration.StackMigrationError):
                migration._private_regular_file(self.fixture.legacy, "unavailable")

    @unittest.skipUnless(os.name == "posix", "POSIX FIFO")
    def test_private_reader_rejects_fifo_swap_without_blocking(self):
        script = """import os,sys
from pathlib import Path
from unittest.mock import patch
from jarvis_orchestrator import stack_migration as m
p=Path(sys.argv[1]);real_open=os.open
def swap(path,*args,**kwargs):
 if Path(path)==p:p.unlink();os.mkfifo(p,0o600)
 return real_open(path,*args,**kwargs)
with patch.object(os,'open',swap):
 try:m._read_private_bytes(p,'unavailable')
 except m.StackMigrationError:sys.exit(0)
sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.fixture.legacy)],
            capture_output=True,
            timeout=2,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    @unittest.skipUnless(os.name == "posix", "POSIX directory identity")
    def test_rollback_never_traverses_a_replacement_root(self):
        root = self.fixture.output
        root.mkdir(mode=0o700)
        (root / migration.VOICE_CONFIG_NAME).write_bytes(b"owned")
        info = root.stat()
        identity = (info.st_dev, info.st_ino)
        moved = self.fixture.root / "original-root"
        real_open = os.open
        swapped = False

        def swap_before_open(path, *args, **kwargs):
            nonlocal swapped
            if Path(path) == root and not swapped:
                swapped = True
                root.rename(moved)
                root.mkdir(mode=0o700)
                (root / migration.VOICE_CONFIG_NAME).write_bytes(b"foreign data")
            return real_open(path, *args, **kwargs)

        with patch.object(os, "open", swap_before_open):
            with self.assertRaises(migration.StackMigrationError):
                migration._rollback(root, identity)
        self.assertEqual(
            (root / migration.VOICE_CONFIG_NAME).read_bytes(), b"foreign data"
        )
        self.assertEqual((moved / migration.VOICE_CONFIG_NAME).read_bytes(), b"owned")

    def test_private_writer_failure_preserves_a_replacement_file(self):
        path = self.fixture.root / "created.json"
        original = self.fixture.root / "moved-created.json"

        def replace_before_write(fd, payload):
            path.rename(original)
            path.write_bytes(b"foreign file")
            raise OSError("synthetic write failure")

        with patch.object(os, "write", replace_before_write):
            with self.assertRaises(migration.StackMigrationError):
                migration._create_private_json(path, {"key": "value"})
        self.assertEqual(path.read_bytes(), b"foreign file")
        self.assertTrue(original.exists())

    def test_creation_rechecks_directory_identity_before_returning(self):
        root = self.fixture.output
        original = self.fixture.root / "moved-created-root"
        real_lstat = Path.lstat
        swapped = False

        def swap_after_stat(path, *args, **kwargs):
            nonlocal swapped
            before = real_lstat(path, *args, **kwargs)
            if path == root and not swapped:
                swapped = True
                root.rename(original)
                root.mkdir(mode=0o700)
                (root / "keep").write_bytes(b"foreign directory")
            return before

        with patch.object(Path, "lstat", swap_after_stat):
            with self.assertRaises(migration.StackMigrationError):
                migration._create_output_directory(root)
        self.assertEqual((root / "keep").read_bytes(), b"foreign directory")
        self.assertTrue(original.exists())

    def test_windows_cleanup_holds_a_read_handle_without_delete_sharing(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        root = self.fixture.output
        root.mkdir(mode=0o700)
        identity = (root.stat().st_dev, root.stat().st_ino)
        constants = SimpleNamespace(
            GENERIC_READ=0x80000000,
            FILE_SHARE_READ=1,
            FILE_SHARE_WRITE=2,
            OPEN_EXISTING=3,
            FILE_FLAG_BACKUP_SEMANTICS=0x02000000,
            FILE_FLAG_OPEN_REPARSE_POINT=0x00200000,
        )
        for fail in (False, True):
            with self.subTest(fail=fail):
                handle = Mock()
                create = Mock(return_value=handle)
                with patch.dict(
                    sys.modules,
                    {
                        "win32con": constants,
                        "win32file": SimpleNamespace(CreateFile=create),
                    },
                ):
                    with patch.object(
                        Path, "unlink", side_effect=OSError if fail else None
                    ):
                        if fail:
                            with self.assertRaises(OSError):
                                migration._remove_windows_output_files(root, identity)
                        else:
                            migration._remove_windows_output_files(root, identity)
                args = create.call_args.args
                self.assertEqual(args[1:3], (constants.GENERIC_READ, 3))
                self.assertEqual(
                    args[5],
                    constants.FILE_FLAG_BACKUP_SEMANTICS
                    | constants.FILE_FLAG_OPEN_REPARSE_POINT,
                )
                handle.close.assert_called_once()

    @unittest.skipUnless(os.name == "nt", "native Windows directory sharing")
    def test_native_windows_root_cannot_be_renamed_during_file_cleanup(self):
        root = self.fixture.output
        root.mkdir()
        identity = (root.stat().st_dev, root.stat().st_ino)
        original_unlink = Path.unlink

        def verify_lock(path, *args, **kwargs):
            with self.assertRaises(OSError):
                root.rename(self.fixture.root / "unexpected-renamed-root")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", verify_lock):
            migration._remove_windows_output_files(root, identity)
        root.rename(self.fixture.root / "released-root")

    def test_rollback_leaves_unexpected_subdirectories_untouched(self):
        root = self.fixture.output
        root.mkdir(mode=0o700)
        unexpected = root / "unexpected"
        unexpected.mkdir()
        (unexpected / "keep").write_bytes(b"foreign subtree")
        info = root.stat()
        with self.assertRaises(migration.StackMigrationError):
            migration._rollback(root, (info.st_dev, info.st_ino))
        self.assertEqual((unexpected / "keep").read_bytes(), b"foreign subtree")


if __name__ == "__main__":
    unittest.main()
