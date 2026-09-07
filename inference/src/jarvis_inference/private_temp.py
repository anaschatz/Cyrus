from __future__ import annotations

import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from typing import Any, Self

from . import windows_acl as _windows_acl
from .windows_acl import (
    WindowsAclFailure,
    validate_private_directory_acl,
)

_BUILTIN_ADMINISTRATORS_SID = _windows_acl._BUILTIN_ADMINISTRATORS_SID
_FILE_ALL_ACCESS = _windows_acl._FILE_ALL_ACCESS
_PRIVATE_ACE_FLAGS = _windows_acl._PRIVATE_ACE_FLAGS
_SYSTEM_SID = _windows_acl._SYSTEM_SID
_WindowsAce = _windows_acl.WindowsAce
_WindowsAclApi = _windows_acl.WindowsAclApi
_WindowsAclSnapshot = _windows_acl.WindowsAclSnapshot
_create_windows_private_directory = _windows_acl.create_private_directory

_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_LOCAL_DRIVE_TYPES = {2, 3, 5, 6}


class PrivateTempFailure(RuntimeError):
    """A private native-runtime temporary directory could not be trusted."""


def _verify_windows_private_directory(
    path: Path,
    *,
    api: Any | None = None,
) -> None:
    try:
        validate_private_directory_acl(Path(path), api=api)
    except WindowsAclFailure as exc:
        raise PrivateTempFailure(
            "private temporary directory ACL could not be verified"
        ) from exc


def _native_drive_type(root: str) -> int:  # pragma: no cover - native Windows
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        function = kernel32.GetDriveTypeW
        function.argtypes = [ctypes.c_wchar_p]
        function.restype = ctypes.c_uint
        return int(function(root))
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def validate_private_temp_directory(
    path: Path,
    *,
    drive_type_reader: Callable[[str], int] | None = None,
) -> Path:
    """Require an existing absolute directory with no Windows reparse hop."""

    candidate = Path(path)
    if not candidate.is_absolute() or any(
        part in {".", ".."} for part in candidate.parts
    ):
        raise PrivateTempFailure("private temporary directory path is unsafe")
    if os.name == "nt":  # pragma: no branch - platform boundary
        windows = PureWindowsPath(str(candidate))
        if (
            not windows.is_absolute()
            or len(windows.drive) != 2
            or not windows.drive[0].isalpha()
            or windows.drive[1] != ":"
        ):
            raise PrivateTempFailure("private temporary directory is not local")
        reader = drive_type_reader or _native_drive_type
        try:
            drive_type = reader(windows.anchor)
        except (OSError, TypeError, ValueError) as exc:
            raise PrivateTempFailure(
                "private temporary drive could not be verified"
            ) from exc
        if drive_type not in _LOCAL_DRIVE_TYPES:
            raise PrivateTempFailure("private temporary directory is not local")

    current = Path(candidate.anchor)
    components = candidate.parts[1:] if candidate.anchor else candidate.parts
    for component in components:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise PrivateTempFailure(
                "private temporary directory is unavailable"
            ) from exc
        if current.is_symlink() or _is_reparse(metadata):
            raise PrivateTempFailure(
                "private temporary directory uses a reparse point"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise PrivateTempFailure(
                "private temporary directory path is not a directory"
            )
    return candidate


def _native_windows_directory() -> Path:  # pragma: no cover - native Windows
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        function = kernel32.GetWindowsDirectoryW
        function.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
        function.restype = ctypes.c_uint
        buffer = ctypes.create_unicode_buffer(32_768)
        length = int(function(buffer, len(buffer)))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise PrivateTempFailure("Windows directory could not be derived") from exc
    if length <= 0 or length >= len(buffer):
        raise PrivateTempFailure("Windows directory could not be derived")
    return Path(buffer.value)


def trusted_windows_directory(
    *, windows_directory_reader: Callable[[], Path] | None = None
) -> Path:
    """Return the OS-derived, local, non-reparse Windows installation root."""

    reader = windows_directory_reader or _native_windows_directory
    try:
        candidate = Path(reader())
    except (OSError, TypeError, ValueError) as exc:
        raise PrivateTempFailure("Windows directory could not be derived") from exc
    return validate_private_temp_directory(candidate)


def private_temp_environment(path: Path) -> dict[str, str]:
    """Return caller-pinned temp variables after native ACL verification."""

    trusted = validate_private_temp_directory(path)
    if os.name == "nt":  # pragma: no cover - native Windows
        _verify_windows_private_directory(trusted)
        trusted = validate_private_temp_directory(trusted)
    value = str(trusted)
    return {"TEMP": value, "TMP": value, "TMPDIR": value}


class PrivateRuntimeDirectory:
    """Own a private runtime directory until its whole process tree exits."""

    def __init__(self, parent: Path, *, prefix: str) -> None:
        self.parent = Path(parent)
        self.prefix = prefix
        self.path: Path | None = None

    def open(self) -> Path:
        if self.path is not None:
            raise PrivateTempFailure(
                "private temporary directory is already open"
            )
        validate_private_temp_directory(self.parent)
        try:
            if os.name == "nt":  # pragma: no cover - native Windows
                try:
                    created = _create_windows_private_directory(
                        self.parent,
                        prefix=self.prefix,
                    )
                except WindowsAclFailure as exc:
                    raise PrivateTempFailure(
                        "private temporary directory creation failed"
                    ) from exc
                validate_private_temp_directory(created)
                _verify_windows_private_directory(created)
            else:
                created = Path(
                    tempfile.mkdtemp(prefix=self.prefix, dir=self.parent)
                )
                os.chmod(created, 0o700)
                validate_private_temp_directory(created)
        except (OSError, PrivateTempFailure):
            if "created" in locals():
                try:
                    shutil.rmtree(created)
                except OSError:
                    pass
            raise
        self.path = created
        return created

    def close(self) -> None:
        path = self.path
        if path is None:
            return
        validate_private_temp_directory(path)
        try:
            shutil.rmtree(path)
        except OSError as exc:
            raise PrivateTempFailure(
                "private temporary directory cleanup failed"
            ) from exc
        self.path = None

    def __enter__(self) -> Path:
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class PrivateLaunchDirectory:
    """Cleanup gate that refuses removal after an unproven child-tree exit."""

    def __init__(self, parent: Path, *, prefix: str) -> None:
        self._owner = PrivateRuntimeDirectory(parent, prefix=prefix)
        self.path: Path | None = None
        self._launch_started = False
        self._tree_drained = False

    def __enter__(self) -> Self:
        self.path = self._owner.open()
        return self

    def mark_launch_started(self) -> None:
        if self.path is None or self._launch_started:
            raise PrivateTempFailure("native launch lifecycle is invalid")
        self._launch_started = True

    def mark_tree_drained(self) -> None:
        if not self._launch_started:
            raise PrivateTempFailure("native launch did not start")
        self._tree_drained = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._launch_started or self._tree_drained:
            self._owner.close()
