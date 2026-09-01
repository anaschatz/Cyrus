from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from jarvis_inference.private_temp import (
    _BUILTIN_ADMINISTRATORS_SID,
    _FILE_ALL_ACCESS,
    _PRIVATE_ACE_FLAGS,
    _SYSTEM_SID,
    PrivateLaunchDirectory,
    PrivateRuntimeDirectory,
    PrivateTempFailure,
    _verify_windows_private_directory,
    _WindowsAce,
    _WindowsAclApi,
    _WindowsAclSnapshot,
    private_temp_environment,
    trusted_windows_directory,
)
from jarvis_inference.windows_acl import create_private_directory
from jarvis_inference.windows_process import WindowsJobProcess


class FakeJobApi:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.active = 1
        self.code: int | None = None
        self.wait_result = True

    def create_job(self) -> int:
        self.calls.append("create_job")
        return 10

    def create_suspended(self, command, environment, cwd):
        self.calls.append(("create_suspended", list(command), dict(environment), cwd))
        return 20, 30, 43210

    def assign(self, job: int, process: int) -> None:
        self.calls.append(("assign", job, process))

    def resume(self, thread: int) -> None:
        self.calls.append(("resume", thread))

    def close_handle(self, handle: int) -> None:
        self.calls.append(("close", handle))

    def terminate_process(self, process: int) -> None:
        self.calls.append(("terminate_process", process))
        self.code = 1

    def exit_code(self, process: int) -> int | None:
        self.calls.append(("exit_code", process))
        return self.code

    def wait_process(self, process: int, timeout: float | None) -> bool:
        self.calls.append(("wait", process, timeout))
        if self.wait_result and self.code is None:
            self.code = 0
            self.active = 0
        return self.wait_result

    def active_processes(self, job: int) -> int:
        self.calls.append(("active", job))
        return self.active

    def terminate_job(self, job: int) -> None:
        self.calls.append(("terminate_job", job))
        self.active = 0
        self.code = 1


class FakeAclApi:
    current_sid = "S-1-5-21-1000"

    def __init__(self) -> None:
        self.applied: tuple[str, ...] | None = None
        self.snapshot: _WindowsAclSnapshot | None = None
        self.failure: Exception | None = None

    def current_user_sid(self) -> str:
        return self.current_sid

    def apply_private_dacl(
        self, path: Path, principals: tuple[str, ...]
    ) -> None:
        del path
        self.applied = principals
        if self.failure is not None:
            raise self.failure
        if self.snapshot is None:
            self.snapshot = _WindowsAclSnapshot(
                owner_sid=self.current_sid,
                dacl_present=True,
                dacl_defaulted=False,
                protected=True,
                entries=tuple(
                    _WindowsAce(
                        sid=sid,
                        mask=_FILE_ALL_ACCESS,
                        ace_type=0,
                        flags=_PRIVATE_ACE_FLAGS,
                    )
                    for sid in principals
                ),
            )

    def read_private_dacl(self, path: Path) -> _WindowsAclSnapshot:
        del path
        if self.failure is not None:
            raise self.failure
        assert self.snapshot is not None
        return self.snapshot


class FakeCreateAclApi(FakeAclApi):
    def __init__(self) -> None:
        super().__init__()
        self.create_calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def create_private_directory(
        self,
        path: Path,
        current_user_sid: str,
        principals: tuple[str, ...],
    ) -> None:
        self.create_calls.append((path, current_user_sid, principals))
        if len(self.create_calls) == 1:
            raise FileExistsError
        path.mkdir()
        self.snapshot = _WindowsAclSnapshot(
            owner_sid=current_user_sid,
            dacl_present=True,
            dacl_defaulted=False,
            protected=True,
            entries=tuple(
                _WindowsAce(
                    sid=sid,
                    mask=_FILE_ALL_ACCESS,
                    ace_type=0,
                    flags=_PRIVATE_ACE_FLAGS,
                )
                for sid in principals
            ),
        )


class WindowsJobProcessTests(unittest.TestCase):
    def process(self, api: FakeJobApi) -> WindowsJobProcess:
        return WindowsJobProcess(
            [r"C:\runtime\runner.exe", "--fixed"],
            {"TEMP": r"C:\runtime\private"},
            Path(r"C:\runtime"),
            api=api,
        )

    def test_process_is_suspended_assigned_then_resumed(self) -> None:
        api = FakeJobApi()
        process = self.process(api)
        self.assertEqual(process.pid, 43210)
        self.assertEqual(api.calls[0], "create_job")
        self.assertEqual(api.calls[1][0], "create_suspended")
        self.assertEqual(api.calls[2], ("assign", 10, 20))
        self.assertEqual(api.calls[3], ("resume", 30))
        self.assertEqual(api.calls[4], ("close", 30))
        process.close_tree()

    def test_close_tree_kills_residual_descendants_before_job_handle_closes(self) -> None:
        api = FakeJobApi()
        process = self.process(api)
        api.code = 0  # Root exited, but a descendant remains in the Job.
        process.close_tree()
        terminate_index = api.calls.index(("terminate_job", 10))
        process_close_index = api.calls.index(("close", 20))
        job_close_index = api.calls.index(("close", 10))
        self.assertLess(terminate_index, process_close_index)
        self.assertLess(process_close_index, job_close_index)
        self.assertEqual(api.active, 0)

    def test_wait_timeout_is_popen_compatible_and_tree_can_be_drained(self) -> None:
        api = FakeJobApi()
        api.wait_result = False
        process = self.process(api)
        with self.assertRaises(subprocess.TimeoutExpired):
            process.wait(timeout=0.01)
        process.terminate()
        process.close_tree()
        self.assertIn(("terminate_job", 10), api.calls)

    def test_failed_drain_retains_handles_for_a_later_proof(self) -> None:
        api = FakeJobApi()
        process = self.process(api)
        api.terminate_job = lambda job: api.calls.append(("terminate_job", job))
        with self.assertRaisesRegex(RuntimeError, "did not drain"):
            process.close_tree(timeout=0)
        self.assertNotIn(("close", 20), api.calls)
        self.assertNotIn(("close", 10), api.calls)
        api.active = 0
        api.code = 1
        process.close_tree()
        self.assertIn(("close", 20), api.calls)
        self.assertIn(("close", 10), api.calls)

    def test_last_resort_finalizer_closes_job_handle_last(self) -> None:
        api = FakeJobApi()
        process = self.process(api)
        process.__del__()
        del process
        gc.collect()
        self.assertEqual(api.calls[-2:], [("close", 20), ("close", 10)])

    def test_native_spawn_restricts_inheritance_to_explicit_standard_handles(self) -> None:
        source = (
            Path(__file__).parents[1]
            / "src"
            / "jarvis_inference"
            / "windows_process.py"
        ).read_text(encoding="utf-8")
        self.assertIn("PROC_THREAD_ATTRIBUTE_HANDLE_LIST", source)
        self.assertIn("EXTENDED_STARTUPINFO_PRESENT", source)
        self.assertIn("UpdateProcThreadAttribute", source)
        self.assertIn("(wintypes.HANDLE * 3)", source)


class PrivateRuntimeDirectoryTests(unittest.TestCase):
    def test_portable_acl_boundary_verifies_exact_principals(self) -> None:
        api = FakeAclApi()
        principals = (
            api.current_sid,
            _SYSTEM_SID,
            _BUILTIN_ADMINISTRATORS_SID,
        )
        api.snapshot = _WindowsAclSnapshot(
            owner_sid=api.current_sid,
            dacl_present=True,
            dacl_defaulted=False,
            protected=True,
            entries=tuple(
                _WindowsAce(
                    sid=sid,
                    mask=_FILE_ALL_ACCESS,
                    ace_type=0,
                    flags=_PRIVATE_ACE_FLAGS,
                )
                for sid in principals
            ),
        )
        _verify_windows_private_directory(Path("ignored"), api=api)
        self.assertIsNone(api.applied)

    def test_portable_acl_boundary_rejects_permissive_or_inherited_acl(self) -> None:
        expected = (
            FakeAclApi.current_sid,
            _SYSTEM_SID,
            _BUILTIN_ADMINISTRATORS_SID,
        )
        cases = (
            _WindowsAclSnapshot(
                owner_sid=FakeAclApi.current_sid,
                dacl_present=True,
                dacl_defaulted=False,
                protected=False,
                entries=(),
            ),
            _WindowsAclSnapshot(
                owner_sid=FakeAclApi.current_sid,
                dacl_present=True,
                dacl_defaulted=False,
                protected=True,
                entries=tuple(
                    _WindowsAce(
                        sid=sid,
                        mask=_FILE_ALL_ACCESS,
                        ace_type=0,
                        flags=_PRIVATE_ACE_FLAGS | 0x10,
                    )
                    for sid in expected
                ),
            ),
            _WindowsAclSnapshot(
                owner_sid=FakeAclApi.current_sid,
                dacl_present=True,
                dacl_defaulted=False,
                protected=True,
                entries=tuple(
                    _WindowsAce(
                        sid=sid,
                        mask=_FILE_ALL_ACCESS,
                        ace_type=0,
                        flags=_PRIVATE_ACE_FLAGS,
                    )
                    for sid in (*expected, "S-1-1-0")
                ),
            ),
        )
        for snapshot in cases:
            with self.subTest(snapshot=snapshot):
                api = FakeAclApi()
                api.snapshot = snapshot
                with self.assertRaises(PrivateTempFailure):
                    _verify_windows_private_directory(Path("ignored"), api=api)

    def test_portable_acl_boundary_fails_closed_when_native_api_is_unverifiable(
        self,
    ) -> None:
        api = FakeAclApi()
        api.failure = OSError("unavailable")
        with self.assertRaisesRegex(
            PrivateTempFailure, "ACL could not be verified"
        ):
            _verify_windows_private_directory(Path("ignored"), api=api)

    def test_portable_acl_verification_never_repairs_existing_directory(self) -> None:
        api = FakeAclApi()
        api.snapshot = _WindowsAclSnapshot(
            owner_sid=api.current_sid,
            dacl_present=True,
            dacl_defaulted=False,
            protected=False,
            entries=(),
        )
        with self.assertRaises(PrivateTempFailure):
            _verify_windows_private_directory(Path("ignored"), api=api)
        self.assertIsNone(api.applied)

    def test_windows_creation_source_uses_atomic_security_attributes(self) -> None:
        source = (
            Path(__file__).parents[1]
            / "src"
            / "jarvis_inference"
            / "windows_acl.py"
        ).read_text(encoding="utf-8")
        self.assertIn("CreateDirectoryW", source)
        self.assertIn("_SecurityAttributes", source)
        self.assertIn("_normalise_sid(current_user_sid)}D:P{entries}", source)

    def test_atomic_creation_retries_collision_and_verifies_exact_acl(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            api = FakeCreateAclApi()
            created = create_private_directory(
                Path(parent), prefix="cyrus-native-", api=api
            )
            self.assertTrue(created.is_dir())
            self.assertEqual(len(api.create_calls), 2)
            self.assertEqual(api.create_calls[-1][0], created)
            self.assertEqual(api.create_calls[-1][1], api.current_sid)
            self.assertEqual(
                api.create_calls[-1][2],
                (
                    api.current_sid,
                    _SYSTEM_SID,
                    _BUILTIN_ADMINISTRATORS_SID,
                ),
            )

    def test_environment_is_pinned_and_directory_is_removed_after_close(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as parent:
            owner = PrivateRuntimeDirectory(Path(parent), prefix="cyrus-test-")
            path = owner.open()
            environment = private_temp_environment(path)
            self.assertEqual(
                environment,
                {"TEMP": str(path), "TMP": str(path), "TMPDIR": str(path)},
            )
            self.assertTrue(path.is_dir())
            owner.close()
            self.assertFalse(path.exists())

    def test_symlink_cannot_be_used_as_private_temp(self) -> None:
        if os.name == "nt":
            self.skipTest("symlink creation is privilege-dependent on Windows")
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as parent:
            root = Path(parent)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(PrivateTempFailure):
                private_temp_environment(link)
            nested = target / "nested"
            nested.mkdir()
            with self.assertRaises(PrivateTempFailure):
                private_temp_environment(link / "nested")

    def test_trusted_windows_root_uses_reader_and_rejects_reparse_chain(self) -> None:
        if os.name == "nt":
            self.skipTest("portable injected-path contract runs off Windows")
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as parent:
            root = Path(parent)
            windows = root / "Windows"
            windows.mkdir()
            self.assertEqual(
                trusted_windows_directory(
                    windows_directory_reader=lambda: windows
                ),
                windows,
            )
            linked = root / "linked"
            linked.symlink_to(windows, target_is_directory=True)
            with self.assertRaises(PrivateTempFailure):
                trusted_windows_directory(
                    windows_directory_reader=lambda: linked
                )

    def test_launch_temp_is_retained_until_tree_drain_is_proven(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as parent:
            with PrivateLaunchDirectory(
                Path(parent), prefix="cyrus-launch-test-"
            ) as launch:
                retained = launch.path
                launch.mark_launch_started()
            self.assertIsNotNone(retained)
            assert retained is not None
            self.assertTrue(retained.is_dir())
            shutil.rmtree(retained)

            with PrivateLaunchDirectory(
                Path(parent), prefix="cyrus-launch-test-"
            ) as launch:
                removed = launch.path
                launch.mark_launch_started()
                launch.mark_tree_drained()
            self.assertIsNotNone(removed)
            assert removed is not None
            self.assertFalse(removed.exists())


def _native_process_active(pid: int) -> bool:  # pragma: no cover - Windows smoke
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.OpenProcess(0x00100000, False, pid)
    if not handle:
        return False
    try:
        return int(kernel32.WaitForSingleObject(handle, 0)) == 258
    finally:
        kernel32.CloseHandle(handle)


@unittest.skipUnless(os.name == "nt", "native Windows Job Object smoke")
class NativeWindowsJobSmokeTests(unittest.TestCase):
    def _environment(self, temporary: Path) -> dict[str, str]:
        result = {
            key: os.environ[key]
            for key in ("SYSTEMROOT", "WINDIR")
            if key in os.environ
        }
        result.update(private_temp_environment(temporary))
        return result

    def test_timeout_kills_descendant_before_return(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "descendant.pid"
            source = (
                "import pathlib,subprocess,sys,time;"
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid));"
                "time.sleep(60)"
            )
            process = WindowsJobProcess(
                [sys.executable, "-c", source],
                self._environment(root),
                root,
            )
            stop = time.monotonic() + 5
            while not pid_file.is_file() and time.monotonic() < stop:
                time.sleep(0.02)
            self.assertTrue(pid_file.is_file())
            descendant = int(pid_file.read_text(encoding="ascii"))
            with self.assertRaises(subprocess.TimeoutExpired):
                process.wait(timeout=0.01)
            process.terminate()
            process.close_tree()
            self.assertFalse(_native_process_active(descendant))

    def test_parent_crash_closes_job_and_kills_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "managed-child.pid"
            environment = self._environment(root)
            environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
            source = (
                "import os,pathlib,sys;"
                "from jarvis_inference.windows_process import WindowsJobProcess;"
                "env={k:os.environ[k] for k in ('SYSTEMROOT','WINDIR','TEMP','TMP','TMPDIR') if k in os.environ};"
                "p=WindowsJobProcess([sys.executable,'-c','import time;time.sleep(60)'],env,pathlib.Path.cwd());"
                f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid));"
                "os._exit(7)"
            )
            completed = subprocess.run(
                [sys.executable, "-c", source],
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            self.assertEqual(completed.returncode, 7)
            child = int(pid_file.read_text(encoding="ascii"))
            stop = time.monotonic() + 5
            while _native_process_active(child) and time.monotonic() < stop:
                time.sleep(0.02)
            self.assertFalse(_native_process_active(child))


@unittest.skipUnless(os.name == "nt", "native Windows ACL smoke")
class NativeWindowsPrivateAclSmokeTests(unittest.TestCase):
    def test_created_directory_has_exact_protected_owner_acl(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            owner = PrivateRuntimeDirectory(Path(parent), prefix="cyrus-acl-")
            path = owner.open()
            api = _WindowsAclApi()
            current_sid = api.current_user_sid()
            snapshot = api.read_private_dacl(path)
            self.assertEqual(snapshot.owner_sid, current_sid)
            self.assertTrue(snapshot.dacl_present)
            self.assertFalse(snapshot.dacl_defaulted)
            self.assertTrue(snapshot.protected)
            self.assertEqual(
                {entry.sid for entry in snapshot.entries},
                {
                    current_sid,
                    _SYSTEM_SID,
                    _BUILTIN_ADMINISTRATORS_SID,
                },
            )
            self.assertTrue(
                all(
                    entry.mask == _FILE_ALL_ACCESS
                    and entry.ace_type == 0
                    and entry.flags == _PRIVATE_ACE_FLAGS
                    for entry in snapshot.entries
                )
            )
            owner.close()

    def test_permissive_acl_is_rejected_before_temp_environment_is_returned(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as parent:
            owner = PrivateRuntimeDirectory(Path(parent), prefix="cyrus-acl-")
            path = owner.open()
            api = _WindowsAclApi()
            current_sid = api.current_user_sid()
            principals = (
                current_sid,
                _SYSTEM_SID,
                _BUILTIN_ADMINISTRATORS_SID,
            )
            expected = "".join(
                f"(A;OICI;FA;;;{sid})" for sid in principals
            )
            api._set_dacl_from_sddl(
                path,
                f"D:P{expected}(A;OICI;GR;;;WD)",
                protected=True,
            )
            self.assertIn(
                "S-1-1-0",
                {entry.sid for entry in api.read_private_dacl(path).entries},
            )
            with self.assertRaises(PrivateTempFailure):
                private_temp_environment(path)
            owner.close()

    def test_unverifiable_acl_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            missing = Path(parent) / "missing"
            with self.assertRaises(PrivateTempFailure):
                _verify_windows_private_directory(missing)


if __name__ == "__main__":
    unittest.main()
