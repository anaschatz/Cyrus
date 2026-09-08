"""POSIX owner-only filesystem boundary for Orchestrator runtime state.

The checks in this module intentionally return or raise stable, content-free
codes.  Paths are never included in reports.  Windows keeps its existing ACL
boundary; the POSIX uid/mode contract is enforced on macOS and Linux.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
DATABASE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
DEFAULT_OPERATIONAL_FILE_NAMES = (
    "orchestrator.lock",
    "orchestrator.stop-request",
    "orchestrator.operations.jsonl",
)
DEFAULT_OPERATIONAL_LOG_PREFIX = "orchestrator.operations.jsonl."


class PrivateStateError(RuntimeError):
    """A content-free private-state boundary violation."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class PrivateStateReportV1:
    version: str
    ready: bool
    platform_enforced: bool
    checked_directory_count: int
    checked_file_count: int
    missing_directory_count: int
    missing_file_count: int
    hardened_directory_count: int
    hardened_file_count: int
    violation_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_directory_count": self.checked_directory_count,
            "checked_file_count": self.checked_file_count,
            "hardened_directory_count": self.hardened_directory_count,
            "hardened_file_count": self.hardened_file_count,
            "missing_directory_count": self.missing_directory_count,
            "missing_file_count": self.missing_file_count,
            "platform_enforced": self.platform_enforced,
            "ready": self.ready,
            "version": self.version,
            "violation_counts": dict(sorted(self.violation_counts.items())),
        }


def absolute_path_no_resolve(path: str | Path) -> Path:
    """Return an absolute lexical path without following any symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def database_family(path: str | Path) -> tuple[Path, ...]:
    selected = absolute_path_no_resolve(path)
    return (selected,) + tuple(Path(f"{selected}{suffix}") for suffix in DATABASE_SIDECAR_SUFFIXES)


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PrivateStateError("private_state_inspection_failed") from exc


def _require_posix_directory(path: Path) -> os.stat_result:
    metadata = _lstat(path)
    if metadata is None:
        raise PrivateStateError("private_state_directory_missing")
    if stat.S_ISLNK(metadata.st_mode):
        raise PrivateStateError("private_state_symlink_forbidden")
    if not stat.S_ISDIR(metadata.st_mode):
        raise PrivateStateError("private_state_directory_type_invalid")
    if metadata.st_uid != os.getuid():
        raise PrivateStateError("private_state_owner_invalid")
    if stat.S_IMODE(metadata.st_mode) != PRIVATE_DIRECTORY_MODE:
        raise PrivateStateError("private_state_directory_permissions_invalid")
    return metadata


def ensure_private_directory(path: str | Path, *, create: bool = False) -> Path:
    """Validate one state directory, creating it as exact 0700 when requested."""

    selected = absolute_path_no_resolve(path)
    if os.name != "posix":
        if create:
            selected.mkdir(parents=True, exist_ok=True)
        elif not selected.is_dir():
            raise PrivateStateError("private_state_directory_missing")
        return selected

    metadata = _lstat(selected)
    if metadata is None and create:
        missing: list[Path] = []
        cursor = selected
        while _lstat(cursor) is None:
            missing.append(cursor)
            if cursor.parent == cursor:
                raise PrivateStateError("private_state_directory_create_failed")
            cursor = cursor.parent
        # Only the requested directory is part of the private-state boundary.
        # Missing ancestors are nevertheless created with the same safe mode.
        for directory in reversed(missing):
            try:
                os.mkdir(directory, PRIVATE_DIRECTORY_MODE)
                os.chmod(directory, PRIVATE_DIRECTORY_MODE, follow_symlinks=False)
            except OSError as exc:
                raise PrivateStateError("private_state_directory_create_failed") from exc
        metadata = _lstat(selected)
    if metadata is None:
        raise PrivateStateError("private_state_directory_missing")
    _require_posix_directory(selected)
    return selected


def _validate_regular_metadata(metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise PrivateStateError("private_state_symlink_forbidden")
    if not stat.S_ISREG(metadata.st_mode):
        raise PrivateStateError("private_state_file_type_invalid")
    if os.name == "posix" and metadata.st_uid != os.getuid():
        raise PrivateStateError("private_state_owner_invalid")
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PrivateStateError("private_state_file_permissions_invalid")


def assert_private_regular_file(path: str | Path, *, required: bool = True) -> bool:
    """Validate an existing owner-only regular file without following links."""

    selected = absolute_path_no_resolve(path)
    metadata = _lstat(selected)
    if metadata is None:
        if required:
            raise PrivateStateError("private_state_file_missing")
        return False
    _validate_regular_metadata(metadata)
    return True


def _open_private_fd(path: Path, flags: int, *, create: bool) -> int:
    """Open one private regular file and bind the fd to the lstat identity."""

    selected = absolute_path_no_resolve(path)
    ensure_private_directory(selected.parent, create=create)
    before = _lstat(selected)
    if before is not None:
        _validate_regular_metadata(before)
    elif not create:
        raise PrivateStateError("private_state_file_missing")
    # Truncation must wait until descriptor and pathname identity are verified.
    open_flags = (flags & ~os.O_TRUNC) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    if create and before is None:
        open_flags |= os.O_CREAT | os.O_EXCL
    if os.name == "posix":
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(selected, open_flags, PRIVATE_FILE_MODE)
    except OSError as exc:
        raise PrivateStateError("private_state_file_open_failed") from exc
    try:
        opened = os.fstat(descriptor)
        _validate_regular_metadata(opened)
        if before is not None and (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise PrivateStateError("private_state_identity_changed")
        current = _lstat(selected)
        if current is None or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise PrivateStateError("private_state_identity_changed")
        _validate_regular_metadata(current)
        if before is None and os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
            opened = os.fstat(descriptor)
            if stat.S_IMODE(opened.st_mode) != PRIVATE_FILE_MODE:
                raise PrivateStateError("private_state_file_permissions_invalid")
        if flags & os.O_TRUNC:
            os.ftruncate(descriptor, 0)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def prepare_private_database(path: str | Path) -> Path:
    """Prepare a DB path and validate every existing SQLite sidecar."""

    selected = absolute_path_no_resolve(path)
    ensure_private_directory(selected.parent, create=True)
    for candidate in database_family(selected):
        assert_private_regular_file(candidate, required=False)
    if not assert_private_regular_file(selected, required=False):
        descriptor = _open_private_fd(selected, os.O_RDWR | os.O_EXCL, create=True)
        os.close(descriptor)
    return selected


def assert_private_database(path: str | Path, *, required: bool = True) -> Path:
    selected = absolute_path_no_resolve(path)
    ensure_private_directory(selected.parent)
    for index, candidate in enumerate(database_family(selected)):
        assert_private_regular_file(candidate, required=required if index == 0 else False)
    return selected


def open_private_append(path: str | Path) -> int:
    return _open_private_fd(
        absolute_path_no_resolve(path),
        os.O_WRONLY | os.O_APPEND,
        create=True,
    )


def open_private_lock(path: str | Path) -> int:
    return _open_private_fd(
        absolute_path_no_resolve(path),
        os.O_RDWR | os.O_APPEND,
        create=True,
    )


def assert_private_log_family(path: str | Path) -> Path:
    selected = absolute_path_no_resolve(path)
    ensure_private_directory(selected.parent, create=True)
    try:
        entries = tuple(os.scandir(selected.parent))
    except OSError as exc:
        raise PrivateStateError("private_state_inspection_failed") from exc
    prefix = selected.name + "."
    for entry in entries:
        if entry.name == selected.name or entry.name.startswith(prefix):
            assert_private_regular_file(Path(entry.path))
    return selected


def create_or_replace_private_file(path: str | Path, payload: bytes) -> Path:
    """Write a small private control file without following an existing link."""

    selected = absolute_path_no_resolve(path)
    descriptor = _open_private_fd(selected, os.O_WRONLY | os.O_TRUNC, create=True)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        raise PrivateStateError("private_state_file_write_failed") from exc
    finally:
        os.close(descriptor)
    assert_private_regular_file(selected)
    return selected


def unlink_private_file(path: str | Path, *, missing_ok: bool = True) -> bool:
    """Unlink only an inspected regular private file; symlinks fail closed."""

    selected = absolute_path_no_resolve(path)
    ensure_private_directory(selected.parent)
    metadata = _lstat(selected)
    if metadata is None:
        if missing_ok:
            return False
        raise PrivateStateError("private_state_file_missing")
    _validate_regular_metadata(metadata)
    try:
        selected.unlink()
    except OSError as exc:
        raise PrivateStateError("private_state_file_unlink_failed") from exc
    return True


def _chmod_owned_no_follow(path: Path, expected: str, mode: int) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if expected == "directory":
        flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # An owner may legitimately need to recover an accidentally mode-000
        # entry.  follow_symlinks=False preserves the no-follow contract even
        # when no usable descriptor can be opened first.
        try:
            before = path.lstat()
            valid_type = (
                stat.S_ISDIR(before.st_mode)
                if expected == "directory"
                else stat.S_ISREG(before.st_mode)
            )
            if stat.S_ISLNK(before.st_mode) or not valid_type or before.st_uid != os.getuid():
                return False
            os.chmod(path, mode, follow_symlinks=False)
            after = path.lstat()
            return (
                (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
                and after.st_uid == os.getuid()
                and stat.S_IMODE(after.st_mode) == mode
            )
        except (NotImplementedError, OSError):
            return False
    try:
        metadata = os.fstat(descriptor)
        valid_type = (
            stat.S_ISDIR(metadata.st_mode)
            if expected == "directory"
            else stat.S_ISREG(metadata.st_mode)
        )
        if not valid_type or metadata.st_uid != os.getuid():
            return False
        os.fchmod(descriptor, mode)
        return stat.S_IMODE(os.fstat(descriptor).st_mode) == mode
    except OSError:
        return False
    finally:
        os.close(descriptor)


def _record(violations: dict[str, int], code: str) -> None:
    violations[code] = violations.get(code, 0) + 1


def inspect_or_harden_orchestrator_state(
    database_path: str | Path,
    *,
    harden: bool = False,
) -> PrivateStateReportV1:
    """Audit or safely chmod an existing Orchestrator state tree.

    Hardening changes only current-user-owned regular files and directories.
    Every chmod is descriptor-based with ``O_NOFOLLOW``; symlinks, wrong-owner
    entries and special files remain violations and are never mutated.
    """

    if os.name != "posix":
        return PrivateStateReportV1(
            version="orchestrator-private-state/1",
            ready=True,
            platform_enforced=False,
            checked_directory_count=0,
            checked_file_count=0,
            missing_directory_count=0,
            missing_file_count=0,
            hardened_directory_count=0,
            hardened_file_count=0,
            violation_counts={},
        )

    database = absolute_path_no_resolve(database_path)
    state = database.parent
    operations = state / "operations"
    violations: dict[str, int] = {}
    checked_directories = 0
    checked_files = 0
    missing_directories = 0
    missing_files = 0
    hardened_directories = 0
    hardened_files = 0

    def inspect_directory(path: Path, *, optional: bool) -> bool:
        nonlocal checked_directories, missing_directories, hardened_directories
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            missing_directories += 1
            if not optional:
                _record(violations, "directory_missing")
            return False
        except OSError:
            _record(violations, "inspection_failed")
            return False
        checked_directories += 1
        if stat.S_ISLNK(metadata.st_mode):
            _record(violations, "directory_symlink")
            return False
        if not stat.S_ISDIR(metadata.st_mode):
            _record(violations, "directory_type")
            return False
        if metadata.st_uid != os.getuid():
            _record(violations, "directory_owner")
            return False
        if stat.S_IMODE(metadata.st_mode) != PRIVATE_DIRECTORY_MODE:
            if harden and _chmod_owned_no_follow(path, "directory", PRIVATE_DIRECTORY_MODE):
                hardened_directories += 1
            else:
                _record(violations, "directory_permissions")
                return False
        return True

    def inspect_file(path: Path, *, required: bool = False) -> None:
        nonlocal checked_files, missing_files, hardened_files
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            missing_files += 1
            if required:
                _record(violations, "file_missing")
            return
        except OSError:
            _record(violations, "inspection_failed")
            return
        checked_files += 1
        if stat.S_ISLNK(metadata.st_mode):
            _record(violations, "file_symlink")
            return
        if not stat.S_ISREG(metadata.st_mode):
            _record(violations, "file_type")
            return
        if metadata.st_uid != os.getuid():
            _record(violations, "file_owner")
            return
        if stat.S_IMODE(metadata.st_mode) != PRIVATE_FILE_MODE:
            if harden and _chmod_owned_no_follow(path, "file", PRIVATE_FILE_MODE):
                hardened_files += 1
            else:
                _record(violations, "file_permissions")

    state_ready = inspect_directory(state, optional=False)
    if state_ready:
        for index, candidate in enumerate(database_family(database)):
            inspect_file(candidate, required=index == 0)
        operations_ready = inspect_directory(operations, optional=True)
        if operations_ready:
            try:
                entries = tuple(os.scandir(operations))
            except OSError:
                _record(violations, "inspection_failed")
            else:
                selected_names = set(DEFAULT_OPERATIONAL_FILE_NAMES)
                for entry in entries:
                    if entry.name in selected_names or entry.name.startswith(
                        DEFAULT_OPERATIONAL_LOG_PREFIX
                    ):
                        inspect_file(Path(entry.path))

    return PrivateStateReportV1(
        version="orchestrator-private-state/1",
        ready=not violations,
        platform_enforced=True,
        checked_directory_count=checked_directories,
        checked_file_count=checked_files,
        missing_directory_count=missing_directories,
        missing_file_count=missing_files,
        hardened_directory_count=hardened_directories,
        hardened_file_count=hardened_files,
        violation_counts=violations,
    )


def audit_orchestrator_state(database_path: str | Path) -> PrivateStateReportV1:
    return inspect_or_harden_orchestrator_state(database_path, harden=False)


def harden_orchestrator_state(database_path: str | Path) -> PrivateStateReportV1:
    return inspect_or_harden_orchestrator_state(database_path, harden=True)


__all__ = [
    "DATABASE_SIDECAR_SUFFIXES",
    "PRIVATE_DIRECTORY_MODE",
    "PRIVATE_FILE_MODE",
    "PrivateStateError",
    "PrivateStateReportV1",
    "absolute_path_no_resolve",
    "assert_private_database",
    "assert_private_log_family",
    "assert_private_regular_file",
    "audit_orchestrator_state",
    "create_or_replace_private_file",
    "database_family",
    "ensure_private_directory",
    "harden_orchestrator_state",
    "inspect_or_harden_orchestrator_state",
    "open_private_append",
    "open_private_lock",
    "prepare_private_database",
    "unlink_private_file",
]
