"""Frozen-process DLL search policy without PATH mutation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..errors import BackendError

LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
_DIRECTORY_HANDLES: list[Any] = []
_APPLIED_DIRECTORY: Path | None = None


class DllDirectoryApi(Protocol):
    def set_default_directories(self, flags: int) -> None: ...
    def add_directory(self, path: str) -> Any: ...


class Win32DllDirectoryApi:
    def set_default_directories(self, flags: int) -> None:  # pragma: no cover - native Windows only
        import ctypes
        from ctypes import wintypes

        function = ctypes.WinDLL("kernel32", use_last_error=True).SetDefaultDllDirectories
        function.argtypes = [wintypes.DWORD]
        function.restype = wintypes.BOOL
        if not function(flags):
            raise ctypes.WinError()

    def add_directory(self, path: str) -> Any:  # pragma: no cover - native Windows only
        return os.add_dll_directory(path)


@dataclass(frozen=True, slots=True)
class DllHardeningResult:
    applied: bool
    code: str
    directories_added: int


def apply_frozen_dll_policy(library_directory: str | Path, *, api: DllDirectoryApi | None = None) -> DllHardeningResult:
    global _APPLIED_DIRECTORY
    if api is None and sys.platform != "win32":
        raise BackendError("hardening.windows_only", "DLL hardening requires Windows", subsystem="hardening")
    directory = Path(library_directory).resolve(strict=False)
    if not directory.is_dir():
        raise BackendError("hardening.library_directory_missing", "Frozen library directory is unavailable", subsystem="hardening")
    if api is None and _APPLIED_DIRECTORY == directory:
        return DllHardeningResult(True, "hardening.dll_policy_already_applied", 0)
    selected = api or Win32DllDirectoryApi()
    try:
        selected.set_default_directories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
        handle = selected.add_directory(str(directory))
    except Exception as exc:
        raise BackendError("hardening.dll_policy_failed", "DLL search hardening failed", subsystem="hardening") from exc
    _DIRECTORY_HANDLES.append(handle)
    if api is None:
        _APPLIED_DIRECTORY = directory
    return DllHardeningResult(True, "hardening.dll_policy_applied", 1)


def dll_policy_applied() -> bool:
    return _APPLIED_DIRECTORY is not None
