"""Owner-only filesystem boundary for Voice SQLite journals.

The Voice runtime opens three persistent SQLite journals even when cloud and
tool capabilities are disabled.  This module validates or hardens the complete
journal family *before* SQLite is allowed to follow a configured path.  Reports
contain counts and fixed codes only; they never contain a filesystem path.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import BackendError


PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
VOICE_PRIVATE_STATE_VERSION = "jarvis-voice-private-state/1"


class VoicePrivateStateError(BackendError):
    """Content-free Voice state boundary failure."""

    def __init__(self, code: str = "voice.private_state_not_ready") -> None:
        super().__init__(
            code,
            "Voice private state is unavailable",
            subsystem="private_state",
        )


@dataclass(frozen=True, slots=True)
class VoicePrivateStateReportV1:
    version: str
    ready: bool
    platform_enforced: bool
    checked_directory_count: int
    checked_file_count: int
    missing_directory_count: int
    missing_file_count: int
    created_directory_count: int
    created_file_count: int
    hardened_directory_count: int
    hardened_file_count: int
    violation_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_directory_count": self.checked_directory_count,
            "checked_file_count": self.checked_file_count,
            "created_directory_count": self.created_directory_count,
            "created_file_count": self.created_file_count,
            "hardened_directory_count": self.hardened_directory_count,
            "hardened_file_count": self.hardened_file_count,
            "missing_directory_count": self.missing_directory_count,
            "missing_file_count": self.missing_file_count,
            "platform_enforced": self.platform_enforced,
            "ready": self.ready,
            "version": self.version,
            "violation_counts": dict(sorted(self.violation_counts.items())),
        }


def _absolute_no_resolve(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def sqlite_journal_family(path: str | Path) -> tuple[Path, ...]:
    selected = _absolute_no_resolve(path)
    return (selected,) + tuple(
        Path(f"{selected}{suffix}") for suffix in SQLITE_SIDECAR_SUFFIXES
    )


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise VoicePrivateStateError("voice.private_state_inspection_failed") from exc


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    """Reject POSIX links and Windows reparse points without following them."""

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & reparse_flag
    )


def _record(violations: dict[str, int], code: str) -> None:
    violations[code] = violations.get(code, 0) + 1


def _chmod_owned_no_follow(path: Path, *, directory: bool, mode: int) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        before = path.lstat()
        before_type = (
            stat.S_ISDIR(before.st_mode)
            if directory
            else stat.S_ISREG(before.st_mode)
        )
        if _is_link_or_reparse(before) or not before_type:
            return False
        if before.st_uid != os.getuid():
            return False
        descriptor = os.open(path, flags)
    except OSError:
        # Permit recovery from an owner-created mode-000 entry without ever
        # following a final symlink.
        try:
            before = path.lstat()
            expected_type = (
                stat.S_ISDIR(before.st_mode)
                if directory
                else stat.S_ISREG(before.st_mode)
            )
            if _is_link_or_reparse(before) or not expected_type:
                return False
            if before.st_uid != os.getuid():
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
        opened = os.fstat(descriptor)
        expected_type = (
            stat.S_ISDIR(opened.st_mode)
            if directory
            else stat.S_ISREG(opened.st_mode)
        )
        if (
            not expected_type
            or opened.st_uid != os.getuid()
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            return False
        os.fchmod(descriptor, mode)
        changed = os.fstat(descriptor)
        after = path.lstat()
        return (
            stat.S_IMODE(changed.st_mode) == mode
            and (changed.st_dev, changed.st_ino) == (after.st_dev, after.st_ino)
            and after.st_uid == os.getuid()
            and stat.S_IMODE(after.st_mode) == mode
        )
    except OSError:
        return False
    finally:
        os.close(descriptor)


def default_voice_state_boundary() -> Path:
    """Return the explicit trust anchor used by default Voice state paths."""

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if not local_app_data:
            raise VoicePrivateStateError("voice.private_state_boundary_missing")
        if not Path(local_app_data).expanduser().is_absolute():
            raise VoicePrivateStateError("voice.private_state_boundary_invalid")
        selected = _absolute_no_resolve(local_app_data)
        return selected
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    if sys.platform != "darwin" and xdg:
        if not Path(xdg).expanduser().is_absolute():
            raise VoicePrivateStateError("voice.private_state_boundary_invalid")
        selected = _absolute_no_resolve(xdg)
        home = _absolute_no_resolve(Path.home())
        if selected == home or home in selected.parents:
            return home
        return selected
    return _absolute_no_resolve(Path.home())


def _trusted_boundary_for_paths(paths: tuple[Path, ...]) -> Path:
    """Infer only well-known OS/user anchors; arbitrary roots stay explicit."""

    candidates: list[Path] = []
    try:
        candidates.append(default_voice_state_boundary())
    except VoicePrivateStateError:
        pass
    # Tests and supported caller-provided ephemeral journals live below the
    # process temp root.  The root itself is the explicit OS-provided anchor;
    # notably on macOS this avoids treating the conventional /var alias as a
    # journal-controlled ancestor.
    candidates.append(_absolute_no_resolve(tempfile.gettempdir()))
    for candidate in dict.fromkeys(candidates):
        if all(path == candidate or candidate in path.parents for path in paths):
            return candidate
    raise VoicePrivateStateError("voice.private_state_boundary_required")


def _directory_chain(boundary: Path, directory: Path) -> tuple[Path, ...]:
    try:
        relative = directory.relative_to(boundary)
    except ValueError as exc:
        raise VoicePrivateStateError(
            "voice.private_state_boundary_invalid"
        ) from exc
    chain = [boundary]
    cursor = boundary
    for component in relative.parts:
        cursor = cursor / component
        chain.append(cursor)
    return tuple(chain)


def _create_private_regular_file(path: Path) -> bool:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
    except FileExistsError:
        return False
    except OSError as exc:
        raise VoicePrivateStateError("voice.private_state_file_create_failed") from exc
    try:
        if os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise VoicePrivateStateError("voice.private_state_file_invalid")
        if os.name == "posix" and (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != PRIVATE_FILE_MODE
        ):
            raise VoicePrivateStateError("voice.private_state_file_permissions_invalid")
        after = path.lstat()
        if (
            _is_link_or_reparse(after)
            or not stat.S_ISREG(after.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise VoicePrivateStateError("voice.private_state_identity_changed")
    except VoicePrivateStateError:
        raise
    except OSError as exc:
        raise VoicePrivateStateError(
            "voice.private_state_file_create_failed"
        ) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    return True


def inspect_or_harden_voice_journal_state(
    journal_paths: Iterable[str | Path],
    *,
    harden: bool = False,
    create: bool = False,
    trusted_boundary: str | Path | None = None,
) -> VoicePrivateStateReportV1:
    """Audit or safely migrate a bounded set of Voice SQLite journals.

    On POSIX, every journal directory must be current-UID 0700 and every
    existing database/sidecar must be current-UID regular 0600.  Hardening only
    changes entries already owned by the current UID.  Symlinks, wrong owners,
    special files, and identity races remain violations.

    Windows retains its existing per-user LocalAppData/ACL boundary: this code
    creates missing regular files and rejects explicit final symlinks or special
    files, but does not pretend POSIX mode bits describe Windows ACLs.
    """

    if create and not harden:
        raise ValueError("create requires harden")
    selected = tuple(_absolute_no_resolve(path) for path in journal_paths)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("journal paths must be a non-empty unique set")
    if any(not path.is_absolute() for path in selected):
        raise ValueError("journal paths must be absolute")
    boundary = (
        _trusted_boundary_for_paths(selected)
        if trusted_boundary is None
        else _absolute_no_resolve(trusted_boundary)
    )
    if any(path != boundary and boundary not in path.parents for path in selected):
        raise VoicePrivateStateError("voice.private_state_boundary_invalid")

    posix = os.name == "posix"
    violations: dict[str, int] = {}
    checked_directories = 0
    checked_files = 0
    missing_directories = 0
    missing_files = 0
    created_directories = 0
    created_files = 0
    hardened_directories = 0
    hardened_files = 0

    directories = tuple(dict.fromkeys(path.parent for path in selected))
    directory_chains = {
        directory: _directory_chain(boundary, directory)
        for directory in directories
    }
    all_directories = tuple(
        dict.fromkeys(
            component
            for chain in directory_chains.values()
            for component in chain
        )
    )
    directory_ready: dict[Path, bool] = {}
    component_ready: dict[Path, bool] = {}
    for directory in all_directories:
        parent_ready = (
            True
            if directory == boundary
            else component_ready.get(directory.parent, False)
        )
        if not parent_ready:
            component_ready[directory] = False
            continue
        metadata = _lstat(directory)
        if metadata is None:
            missing_directories += 1
            if directory == boundary:
                _record(violations, "boundary_missing")
            elif harden and create:
                try:
                    os.mkdir(directory, PRIVATE_DIRECTORY_MODE)
                    if posix:
                        os.chmod(
                            directory,
                            PRIVATE_DIRECTORY_MODE,
                            follow_symlinks=False,
                        )
                    created_directories += 1
                    metadata = _lstat(directory)
                except OSError:
                    _record(violations, "directory_create_failed")
            if metadata is None:
                component_ready[directory] = False
                continue
        checked_directories += 1
        if _is_link_or_reparse(metadata):
            _record(
                violations,
                (
                    "directory_symlink"
                    if directory in directories
                    else "ancestor_symlink"
                ),
            )
            component_ready[directory] = False
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            _record(
                violations,
                "directory_type" if directory in directories else "ancestor_type",
            )
            component_ready[directory] = False
            continue
        if posix and metadata.st_uid != os.getuid():
            _record(
                violations,
                "directory_owner" if directory in directories else "ancestor_owner",
            )
            component_ready[directory] = False
            continue
        if (
            posix
            and directory in directories
            and stat.S_IMODE(metadata.st_mode) != PRIVATE_DIRECTORY_MODE
        ):
            if harden and _chmod_owned_no_follow(
                directory, directory=True, mode=PRIVATE_DIRECTORY_MODE
            ):
                hardened_directories += 1
            else:
                _record(violations, "directory_permissions")
                component_ready[directory] = False
                continue
        component_ready[directory] = True

    for directory, chain in directory_chains.items():
        directory_ready[directory] = all(
            component_ready.get(component, False) for component in chain
        )

    for journal in selected:
        if not directory_ready.get(journal.parent, False):
            # Never inspect a child through an untrusted or missing ancestor.
            continue
        if harden and create and directory_ready.get(journal.parent, False):
            try:
                if _create_private_regular_file(journal):
                    created_files += 1
            except VoicePrivateStateError as exc:
                _record(violations, exc.code.rsplit(".", 1)[-1])
        for candidate in sqlite_journal_family(journal):
            metadata = _lstat(candidate)
            if metadata is None:
                missing_files += 1
                continue
            checked_files += 1
            if _is_link_or_reparse(metadata):
                _record(violations, "file_symlink")
                continue
            if not stat.S_ISREG(metadata.st_mode):
                _record(violations, "file_type")
                continue
            if posix and metadata.st_uid != os.getuid():
                _record(violations, "file_owner")
                continue
            if posix and stat.S_IMODE(metadata.st_mode) != PRIVATE_FILE_MODE:
                if harden and _chmod_owned_no_follow(
                    candidate, directory=False, mode=PRIVATE_FILE_MODE
                ):
                    hardened_files += 1
                else:
                    _record(violations, "file_permissions")

    return VoicePrivateStateReportV1(
        version=VOICE_PRIVATE_STATE_VERSION,
        ready=not violations,
        platform_enforced=posix,
        checked_directory_count=checked_directories,
        checked_file_count=checked_files,
        missing_directory_count=missing_directories,
        missing_file_count=missing_files,
        created_directory_count=created_directories,
        created_file_count=created_files,
        hardened_directory_count=hardened_directories,
        hardened_file_count=hardened_files,
        violation_counts=violations,
    )


def _default_journal_paths() -> tuple[Path, Path, Path]:
    # Lazy imports avoid a cycle because each journal class uses the preparation
    # helpers in this module before calling sqlite3.connect().
    from .action_approval import default_action_approval_journal_path
    from .cloud_approval import default_cloud_approval_journal_path
    from .delivery_journal import default_delivery_journal_path

    return (
        default_delivery_journal_path(),
        default_cloud_approval_journal_path(),
        default_action_approval_journal_path(),
    )


def audit_voice_journal_state(
    journal_paths: Iterable[str | Path] | None = None,
    *,
    trusted_boundary: str | Path | None = None,
) -> VoicePrivateStateReportV1:
    return inspect_or_harden_voice_journal_state(
        _default_journal_paths() if journal_paths is None else journal_paths,
        trusted_boundary=trusted_boundary,
    )


def harden_voice_journal_state(
    journal_paths: Iterable[str | Path] | None = None,
    *,
    trusted_boundary: str | Path | None = None,
) -> VoicePrivateStateReportV1:
    return inspect_or_harden_voice_journal_state(
        _default_journal_paths() if journal_paths is None else journal_paths,
        harden=True,
        create=True,
        trusted_boundary=trusted_boundary,
    )


def prepare_voice_journal_state(
    journal_paths: Iterable[str | Path] | None = None,
    *,
    trusted_boundary: str | Path | None = None,
) -> VoicePrivateStateReportV1:
    report = harden_voice_journal_state(
        journal_paths,
        trusted_boundary=trusted_boundary,
    )
    if not report.ready:
        raise VoicePrivateStateError()
    return report


def prepare_voice_journal_path(
    path: str | Path,
    *,
    trusted_boundary: str | Path | None = None,
) -> Path:
    selected = _absolute_no_resolve(path)
    prepare_voice_journal_state(
        (selected,),
        trusted_boundary=trusted_boundary,
    )
    return selected


__all__ = [
    "PRIVATE_DIRECTORY_MODE",
    "PRIVATE_FILE_MODE",
    "SQLITE_SIDECAR_SUFFIXES",
    "VOICE_PRIVATE_STATE_VERSION",
    "VoicePrivateStateError",
    "VoicePrivateStateReportV1",
    "audit_voice_journal_state",
    "default_voice_state_boundary",
    "harden_voice_journal_state",
    "inspect_or_harden_voice_journal_state",
    "prepare_voice_journal_path",
    "prepare_voice_journal_state",
    "sqlite_journal_family",
]
