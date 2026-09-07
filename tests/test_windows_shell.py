from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.app_paths import ApplicationPaths
from jarvis_voice.errors import BackendError
from jarvis_voice.tray_app import VoiceTrayApplication
from jarvis_voice.windows.hardening import LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, apply_frozen_dll_policy
from jarvis_voice.windows.single_instance import ERROR_ALREADY_EXISTS, MUTEX_NAME, SingleInstanceGuard
from jarvis_voice.windows.tray import TrayCommand, TrayCoordinator


class FakeMutexApi:
    def __init__(self, error: int = 0) -> None:
        self.error = error
        self.created: list[str] = []
        self.closed: list[int] = []

    def create(self, name: str) -> tuple[int, int]:
        self.created.append(name)
        return 77, self.error

    def close(self, handle: int) -> None:
        self.closed.append(handle)


class FakeDllApi:
    def __init__(self) -> None:
        self.flags: list[int] = []
        self.paths: list[str] = []

    def set_default_directories(self, flags: int) -> None:
        self.flags.append(flags)

    def add_directory(self, path: str) -> object:
        self.paths.append(path)
        return object()


class FakeTrayShell:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def add(self, tooltip: str) -> bool:
        self.calls.append(("add", tooltip)); return True

    def set_version(self) -> bool:
        self.calls.append(("set_version", None)); return True

    def modify(self, tooltip: str) -> bool:
        self.calls.append(("modify", tooltip)); return True

    def delete(self) -> None:
        self.calls.append(("delete", None))


class WindowsShellTests(unittest.TestCase):
    def test_single_instance_closes_handles_and_uses_content_free_local_name(self) -> None:
        api = FakeMutexApi(); guard = SingleInstanceGuard(api=api)
        guard.acquire(); guard.acquire(); guard.release(); guard.release()
        self.assertEqual(api.created, [MUTEX_NAME])
        self.assertEqual(api.closed, [77])
        self.assertTrue(MUTEX_NAME.startswith("Local\\JARVISVoice-"))
        duplicate = FakeMutexApi(ERROR_ALREADY_EXISTS)
        with self.assertRaisesRegex(BackendError, "application.already_running"):
            SingleInstanceGuard(api=duplicate).acquire()
        self.assertEqual(duplicate.closed, [77])

    def test_tray_shutdown_interruption_does_not_skip_owned_cleanup(self) -> None:
        for entry in ("run", "exit"):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as temp:
                api = FakeMutexApi()
                guard = SingleInstanceGuard(api=api)
                paths = ApplicationPaths.from_environment({"LOCALAPPDATA": temp})
                tray = mock.Mock()
                lifecycle = mock.Mock()
                lifecycle.shutdown.side_effect = KeyboardInterrupt()
                app = VoiceTrayApplication(paths=paths, single_instance=guard,
                                           tray_factory=lambda **kwargs: tray)
                try:
                    with mock.patch("jarvis_voice.tray_app.VoiceLifecycleController", return_value=lifecycle):
                        with self.assertRaises(KeyboardInterrupt):
                            if entry == "run":
                                app.run()
                            else:
                                app._lifecycle, app._tray = lifecycle, tray
                                app._shutdown_and_close()
                    if entry == "run":
                        self.assertEqual(api.created, [MUTEX_NAME])
                        self.assertEqual(api.closed, [77])
                    else:
                        tray.request_close.assert_called_once()
                finally:
                    guard.release()

    def test_dll_hardening_uses_default_dirs_and_does_not_mutate_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            before = os.environ.get("PATH")
            api = FakeDllApi()
            result = apply_frozen_dll_policy(Path(temporary), api=api)
            self.assertTrue(result.applied)
            self.assertEqual(api.flags, [LOAD_LIBRARY_SEARCH_DEFAULT_DIRS])
            self.assertEqual(api.paths, [str(Path(temporary).resolve())])
            self.assertEqual(os.environ.get("PATH"), before)

    def test_tray_readds_after_explorer_restart_and_routes_known_commands(self) -> None:
        shell = FakeTrayShell(); commands: list[TrayCommand] = []
        coordinator = TrayCoordinator(shell, command_handler=commands.append, status_provider=lambda: "JARVIS Voice — Running")
        coordinator.start(); coordinator.refresh(); coordinator.taskbar_created()
        coordinator.dispatch(TrayCommand.STOP); coordinator.dispatch(9999); coordinator.close(); coordinator.close()
        self.assertEqual(commands, [TrayCommand.STOP])
        self.assertEqual([call[0] for call in shell.calls], ["add", "set_version", "modify", "add", "set_version", "delete"])


if __name__ == "__main__":
    unittest.main()
