from __future__ import annotations

import os
import plistlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.background import BackgroundServiceManager
from jarvis_orchestrator.operations import OperationalError


class BackgroundBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = self.root = Path(temporary.name)
        config = root / "config.json"
        config.write_text("{}")

        class Manager(BackgroundServiceManager):
            @property
            def macos_plist(self):
                return root / "LaunchAgents" / "service.plist"

        self.manager = Manager(config_path=config, database_path=root / "state.sqlite3")
        self.manager.macos_plist.parent.mkdir(mode=0o755)

    @unittest.skipUnless(os.name == "posix", "POSIX link publication boundary")
    def test_plist_publication_does_not_overwrite_link_target(self):
        target = self.root / "foreign.txt"
        for link in ("symlink", "hardlink"):
            with self.subTest(link=link):
                target.write_bytes(b"preserve me")
                target.chmod(0o644)
                path = self.manager.macos_plist
                if link == "symlink":
                    path.symlink_to(target)
                else:
                    os.link(target, path)
                try:
                    self.manager._write_macos_plist()
                    self.assertEqual(target.read_bytes(), b"preserve me")
                    self.assertEqual(target.stat().st_mode & 0o777, 0o644)
                    self.assertFalse(path.is_symlink())
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                    self.assertTrue(plistlib.loads(path.read_bytes())["RunAtLoad"])
                finally:
                    path.unlink(missing_ok=True)

    def test_failed_plist_publication_preserves_previous_file(self):
        path = self.manager.macos_plist
        path.write_bytes(b"previous plist")
        with patch("os.replace", side_effect=OSError("publication failed")):
            with self.assertRaises((OSError, OperationalError)):
                self.manager._write_macos_plist()
        self.assertEqual(path.read_bytes(), b"previous plist")
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_command_start_or_timeout_failure_has_stable_code(self):
        for allowed_fail in (False, True):
            for failure in (
                FileNotFoundError("private executable"),
                subprocess.TimeoutExpired(["private command"], 15),
            ):
                with (
                    self.subTest(allowed_fail=allowed_fail, failure=type(failure)),
                    patch("subprocess.run", side_effect=failure),
                ):
                    with self.assertRaises(OperationalError) as observed:
                        self.manager._run(
                            ["mock-service-manager"], allowed_fail=allowed_fail
                        )
                    self.assertEqual(
                        str(observed.exception), "background_manager_command_failed"
                    )


if __name__ == "__main__":
    unittest.main()
