from __future__ import annotations

import ctypes
import os
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .strict_json import canonical_bytes
from .strict_json import parse_body
from .errors import ApplianceError
from .limits import MANIFEST_BYTES


class ExclusiveFileLock(AbstractContextManager["ExclusiveFileLock"]):
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self) -> "ExclusiveFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.handle.seek(0, os.SEEK_END) == 0:
                    self.handle.write(b"\0")
                    self.handle.flush()
                    os.fsync(self.handle.fileno())
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is None:
            return
        if os.name == "nt":
            import msvcrt
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


def durable_replace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".qualification-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        payload = canonical_bytes(value) + b"\n"
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            if sys_platform_is_macos():
                _full_fsync(handle.fileno())
            else:
                os.fsync(handle.fileno())
        if os.name == "nt":
            _windows_replace(temporary, path)
        else:
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def sys_platform_is_macos() -> bool:
    import sys
    return sys.platform == "darwin"


def _full_fsync(descriptor: int) -> None:
    try:
        import fcntl
        command = getattr(fcntl, "F_FULLFSYNC")
        fcntl.fcntl(descriptor, command)
    except (AttributeError, OSError):
        os.fsync(descriptor)


def _windows_replace(source: Path, destination: Path) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if destination.exists():
        replace = kernel32.ReplaceFileW
        replace.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        replace.restype = ctypes.c_int
        ok = replace(str(destination), str(source), None, 0, None, None)
    else:
        move = kernel32.MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move.restype = ctypes.c_int
        ok = move(str(source), str(destination), 0x1 | 0x8)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())


def read_report(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
        if len(raw) > MANIFEST_BYTES:
            return None
        return parse_body(raw)
    except (OSError, ApplianceError):
        return None
