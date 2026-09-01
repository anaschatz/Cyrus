from __future__ import annotations

import ctypes
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from ctypes import wintypes
from pathlib import Path
from typing import Any

CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
CREATE_NO_WINDOW = 0x08000000
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
STARTF_USESTDHANDLES = 0x00000100
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
STILL_ACTIVE = 259
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
INFINITE = 0xFFFFFFFF
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class WindowsProcessFailure(RuntimeError):
    """A contained Windows child could not be created or drained safely."""


class _SecurityAttributes(ctypes.Structure):
    _fields_ = (
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    )


class _StartupInfo(ctypes.Structure):
    _fields_ = (
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    )


class _ProcessInformation(ctypes.Structure):
    _fields_ = (
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    )


class _StartupInfoEx(ctypes.Structure):
    _fields_ = (
        ("StartupInfo", _StartupInfo),
        ("lpAttributeList", wintypes.LPVOID),
    )


class _IoCounters(ctypes.Structure):
    _fields_ = (
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    )


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    )


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


class _BasicAccountingInformation(ctypes.Structure):
    _fields_ = (
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    )


def _last_error(message: str) -> WindowsProcessFailure:
    code = ctypes.get_last_error()
    return WindowsProcessFailure(f"{message} (Windows error {code})")


def _handle_value(handle: Any) -> int:
    value = getattr(handle, "value", handle)
    if not isinstance(value, int) or value <= 0:
        raise WindowsProcessFailure("Windows returned an invalid handle")
    return value


def _milliseconds(timeout: float | None) -> int:
    if timeout is None:
        return INFINITE
    if timeout < 0:
        return 0
    return min(0xFFFFFFFE, max(0, int(timeout * 1000 + 0.999)))


def _environment_block(environment: Mapping[str, str]) -> ctypes.Array[Any]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\x00" in key
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise WindowsProcessFailure("child environment is invalid")
        folded = key.casefold()
        if folded in seen:
            raise WindowsProcessFailure("child environment has duplicate keys")
        seen.add(folded)
        entries.append((key, value))
    entries.sort(key=lambda pair: pair[0].casefold())
    block = "\x00".join(f"{key}={value}" for key, value in entries) + "\x00\x00"
    return ctypes.create_unicode_buffer(block)


class _Kernel32JobApi:
    def __init__(self) -> None:  # pragma: no cover - native Windows only
        if os.name != "nt":
            raise WindowsProcessFailure("Windows job objects require Windows")
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:  # pragma: no cover - native Windows
        kernel32 = self.kernel32
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_SecurityAttributes),
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPCWSTR,
            wintypes.LPVOID,
            ctypes.POINTER(_ProcessInformation),
        ]
        kernel32.CreateProcessW.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel32.ResumeThread.restype = wintypes.DWORD
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.InitializeProcThreadAttributeList.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
        kernel32.UpdateProcThreadAttribute.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.c_size_t,
            wintypes.LPVOID,
            ctypes.c_size_t,
            wintypes.LPVOID,
            wintypes.LPVOID,
        ]
        kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
        kernel32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
        kernel32.DeleteProcThreadAttributeList.restype = None

    def create_job(self) -> int:  # pragma: no cover - native Windows
        handle = self.kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise _last_error("CreateJobObjectW failed")
        information = _ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            self.close_handle(handle)
            raise _last_error("SetInformationJobObject failed")
        return _handle_value(handle)

    def _nul_handle(self, access: int) -> int:  # pragma: no cover - native Windows
        security = _SecurityAttributes(
            ctypes.sizeof(_SecurityAttributes), None, True
        )
        handle = self.kernel32.CreateFileW(
            "NUL",
            access,
            0x00000001 | 0x00000002,
            ctypes.byref(security),
            3,
            0x00000080,
            None,
        )
        value = getattr(handle, "value", handle)
        if not handle or value == INVALID_HANDLE_VALUE:
            raise _last_error("opening NUL failed")
        return _handle_value(handle)

    def create_suspended(
        self,
        command: Sequence[str],
        environment: Mapping[str, str],
        cwd: Path,
    ) -> tuple[int, int, int]:  # pragma: no cover - native Windows
        if not command or not Path(command[0]).is_absolute():
            raise WindowsProcessFailure("child executable must be absolute")
        if not cwd.is_absolute():
            raise WindowsProcessFailure("child working directory must be absolute")
        command_line = ctypes.create_unicode_buffer(
            subprocess.list2cmdline([str(part) for part in command])
        )
        environment_buffer = _environment_block(environment)
        stdin_handle = self._nul_handle(0x80000000)
        stdout_handle: int | None = None
        stderr_handle: int | None = None
        attribute_list: ctypes.Array[Any] | None = None
        attributes_initialized = False
        process = _ProcessInformation()
        try:
            stdout_handle = self._nul_handle(0x40000000)
            stderr_handle = self._nul_handle(0x40000000)
            startup = _StartupInfoEx()
            startup.StartupInfo.cb = ctypes.sizeof(_StartupInfoEx)
            startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
            startup.StartupInfo.hStdInput = stdin_handle
            startup.StartupInfo.hStdOutput = stdout_handle
            startup.StartupInfo.hStdError = stderr_handle

            attribute_size = ctypes.c_size_t()
            self.kernel32.InitializeProcThreadAttributeList(
                None, 1, 0, ctypes.byref(attribute_size)
            )
            if attribute_size.value <= 0:
                raise _last_error("sizing process attribute list failed")
            attribute_list = ctypes.create_string_buffer(attribute_size.value)
            attribute_pointer = ctypes.cast(attribute_list, wintypes.LPVOID)
            if not self.kernel32.InitializeProcThreadAttributeList(
                attribute_pointer, 1, 0, ctypes.byref(attribute_size)
            ):
                raise _last_error("InitializeProcThreadAttributeList failed")
            attributes_initialized = True
            startup.lpAttributeList = attribute_pointer
            allowed_handles = (wintypes.HANDLE * 3)(
                stdin_handle, stdout_handle, stderr_handle
            )
            if not self.kernel32.UpdateProcThreadAttribute(
                attribute_pointer,
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                ctypes.cast(allowed_handles, wintypes.LPVOID),
                ctypes.sizeof(allowed_handles),
                None,
                None,
            ):
                raise _last_error("restricting inherited handles failed")
            created = self.kernel32.CreateProcessW(
                str(command[0]),
                command_line,
                None,
                None,
                True,
                CREATE_SUSPENDED
                | CREATE_UNICODE_ENVIRONMENT
                | CREATE_NO_WINDOW
                | EXTENDED_STARTUPINFO_PRESENT,
                ctypes.cast(environment_buffer, wintypes.LPVOID),
                str(cwd),
                ctypes.byref(startup),
                ctypes.byref(process),
            )
            if not created:
                raise _last_error("CreateProcessW failed")
            return (
                _handle_value(process.hProcess),
                _handle_value(process.hThread),
                int(process.dwProcessId),
            )
        finally:
            if attribute_list is not None and attributes_initialized:
                self.kernel32.DeleteProcThreadAttributeList(
                    ctypes.cast(attribute_list, wintypes.LPVOID)
                )
            self.close_handle(stdin_handle)
            if stdout_handle is not None:
                self.close_handle(stdout_handle)
            if stderr_handle is not None:
                self.close_handle(stderr_handle)

    def assign(self, job: int, process: int) -> None:  # pragma: no cover
        if not self.kernel32.AssignProcessToJobObject(job, process):
            raise _last_error("AssignProcessToJobObject failed")

    def resume(self, thread: int) -> None:  # pragma: no cover
        if self.kernel32.ResumeThread(thread) == 0xFFFFFFFF:
            raise _last_error("ResumeThread failed")

    def exit_code(self, process: int) -> int | None:  # pragma: no cover
        value = wintypes.DWORD()
        if not self.kernel32.GetExitCodeProcess(process, ctypes.byref(value)):
            raise _last_error("GetExitCodeProcess failed")
        return None if value.value == STILL_ACTIVE else int(value.value)

    def wait_process(self, process: int, timeout: float | None) -> bool:  # pragma: no cover
        result = int(self.kernel32.WaitForSingleObject(process, _milliseconds(timeout)))
        if result == WAIT_OBJECT_0:
            return True
        if result == WAIT_TIMEOUT:
            return False
        raise _last_error("WaitForSingleObject failed")

    def terminate_job(self, job: int) -> None:  # pragma: no cover
        if not self.kernel32.TerminateJobObject(job, 1):
            raise _last_error("TerminateJobObject failed")

    def terminate_process(self, process: int) -> None:  # pragma: no cover
        self.kernel32.TerminateProcess(process, 1)

    def active_processes(self, job: int) -> int:  # pragma: no cover
        information = _BasicAccountingInformation()
        if not self.kernel32.QueryInformationJobObject(
            job,
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
            None,
        ):
            raise _last_error("QueryInformationJobObject failed")
        return int(information.ActiveProcesses)

    def close_handle(self, handle: int | None) -> None:  # pragma: no cover
        if handle:
            self.kernel32.CloseHandle(handle)


class WindowsJobProcess:
    """Popen-like owner for one suspended-before-assignment Windows tree."""

    def __init__(
        self,
        command: Sequence[str],
        environment: Mapping[str, str],
        cwd: Path,
        *,
        api: Any | None = None,
    ) -> None:
        self.args = [str(part) for part in command]
        self._api = api or _Kernel32JobApi()
        self._job: int | None = None
        self._process: int | None = None
        self._returncode: int | None = None
        self.pid = 0
        job = self._api.create_job()
        self._job = job
        thread: int | None = None
        try:
            process, thread, pid = self._api.create_suspended(
                self.args, dict(environment), Path(cwd)
            )
            self._process = process
            self.pid = pid
            self._api.assign(job, process)
            self._api.resume(thread)
        except BaseException:
            if self._process is not None:
                try:
                    self._api.terminate_process(self._process)
                except BaseException:  # noqa: BLE001, S110 - emergency cleanup
                    pass
            try:
                self._api.terminate_job(job)
            except BaseException:  # noqa: BLE001, S110 - Job close is fallback
                pass
            self._release_handles()
            raise
        finally:
            if thread is not None:
                self._api.close_handle(thread)

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def poll(self) -> int | None:
        if self._process is None:
            return self._returncode
        value = self._api.exit_code(self._process)
        if value is not None:
            self._returncode = value
        return value

    def wait(self, timeout: float | None = None) -> int:
        if self._process is None:
            if self._returncode is None:
                raise WindowsProcessFailure("process handle is closed")
            return self._returncode
        if not self._api.wait_process(self._process, timeout):
            raise subprocess.TimeoutExpired(self.args, timeout)
        value = self._api.exit_code(self._process)
        if value is None:
            raise WindowsProcessFailure("process wait completed without an exit code")
        self._returncode = value
        return value

    def terminate(self) -> None:
        if self._job is not None and self._api.active_processes(self._job) > 0:
            self._api.terminate_job(self._job)

    def kill(self) -> None:
        self.terminate()

    def close_tree(self, timeout: float = 10.0) -> None:
        """Kill residual descendants, prove the job empty, then close handles."""

        job = self._job
        if job is None:
            return
        if self._api.active_processes(job) > 0:
            self._api.terminate_job(job)
        deadline = time.monotonic() + max(0.0, timeout)
        while self._api.active_processes(job) > 0:
            if time.monotonic() >= deadline:
                # Retain both handles so a later close can retry the proof.
                # The finalizer still closes the Job as a last-resort kill.
                raise WindowsProcessFailure("Windows process tree did not drain")
            time.sleep(0.01)
        if self._process is not None and self._returncode is None:
            self._returncode = self._api.exit_code(self._process)
        self._release_handles()

    def _release_handles(self) -> None:
        process, job = self._process, self._job
        self._process = None
        self._job = None
        if process is not None:
            self._api.close_handle(process)
        if job is not None:
            self._api.close_handle(job)

    def __del__(self) -> None:  # pragma: no cover - last-resort process safety
        try:
            self._release_handles()
        except BaseException:  # noqa: BLE001, S110 - destructor cannot escape
            pass


def run_windows_job(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path,
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    """Run a native child and return only after its entire Job is empty."""

    process = WindowsJobProcess(command, environment, cwd)
    timed_out = False
    try:
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                pass
            raise
    finally:
        process.close_tree(timeout=10.0 if timed_out else 5.0)
    return subprocess.CompletedProcess([str(part) for part in command], returncode)
