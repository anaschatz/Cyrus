"""Per-session single-instance guard backed by a non-inheritable mutex."""

from __future__ import annotations

import sys
from typing import Protocol

from ..errors import BackendError

ERROR_ALREADY_EXISTS = 183
MUTEX_NAME = "Local\\JARVISVoice-4C73147A-45A9-4FEA-9EE1-4D2645298F54"


class MutexApi(Protocol):
    def create(self, name: str) -> tuple[int, int]: ...
    def close(self, handle: int) -> None: ...


class Win32MutexApi:
    def create(self, name: str) -> tuple[int, int]:  # pragma: no cover - native Windows only
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        ctypes.set_last_error(0)
        handle = kernel32.CreateMutexW(None, False, name)
        error = ctypes.get_last_error()
        if not handle:
            raise ctypes.WinError(error)
        return int(handle), int(error)

    def close(self, handle: int) -> None:  # pragma: no cover - native Windows only
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        if not kernel32.CloseHandle(handle):
            raise ctypes.WinError()


class SingleInstanceGuard:
    def __init__(self, *, api: MutexApi | None = None, name: str = MUTEX_NAME) -> None:
        if api is None and sys.platform != "win32":
            raise BackendError("application.single_instance_windows_only", "Single-instance guard requires Windows", subsystem="application")
        self.api = api or Win32MutexApi()
        self.name = name
        self._handle: int | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            return
        try:
            handle, error = self.api.create(self.name)
        except Exception as exc:
            raise BackendError("application.single_instance_failed", "Single-instance guard could not be created", subsystem="application") from exc
        if error == ERROR_ALREADY_EXISTS:
            try:
                self.api.close(handle)
            finally:
                raise BackendError("application.already_running", "JARVIS Voice is already running in this session", subsystem="application")
        self._handle = handle

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                self.api.close(handle)
            except Exception as exc:
                raise BackendError("application.single_instance_release_failed", "Single-instance guard could not be released", subsystem="application") from exc

    def __enter__(self) -> "SingleInstanceGuard":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
