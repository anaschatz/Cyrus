"""Content-free, owner-only state snapshots for the native desktop UI."""

from __future__ import annotations

import json
import os
import secrets
import stat
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Callable

from .private_state import (
    PRIVATE_FILE_MODE,
    VoicePrivateStateError,
    default_voice_state_boundary,
    prepare_voice_journal_path,
)


UI_STATE_VERSION = "jarvis-voice-ui-state/2"
MAXIMUM_UI_STATE_BYTES = 512


class VoiceUiState(str, Enum):
    STARTING = "starting"
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"
    OFFLINE = "offline"


def default_voice_ui_state_path() -> Path:
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA", "").strip()
        if not root:
            raise VoicePrivateStateError("voice.ui_state_root_missing")
        return Path(root) / "JARVIS Voice" / "state" / "ui-state.json"
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "JARVIS Voice"
            / "state"
            / "ui-state.json"
        )
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "jarvis-voice" / "ui-state.json"


class VoiceUiStatePublisher:
    """Publish a tiny enum snapshot without transcript or response content.

    The snapshot is replaced atomically inside an already validated owner-only
    directory. A one-second heartbeat lets a separate UI distinguish an idle
    live runtime from a stale file left after an unclean exit.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        trusted_boundary: str | Path | None = None,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        heartbeat_interval_seconds: float = 1.0,
        active_interval_seconds: float = 0.1,
    ) -> None:
        if heartbeat_interval_seconds <= 0 or not 0 < active_interval_seconds <= heartbeat_interval_seconds:
            raise ValueError("snapshot intervals are invalid")
        self.path = prepare_voice_journal_path(
            default_voice_ui_state_path() if path is None else path,
            trusted_boundary=(
                default_voice_state_boundary()
                if trusted_boundary is None
                else trusted_boundary
            ),
        )
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._active_interval_seconds = active_interval_seconds
        self._state = VoiceUiState.STARTING
        self._activity_milli = 0
        self._state_started_at_unix_ms = int(self._wall_clock() * 1000)
        self._generation = 0
        self._last_write_monotonic = float("-inf")
        self._activity_lock = threading.Lock()
        self.publish(VoiceUiState.STARTING)

    @property
    def state(self) -> VoiceUiState:
        return self._state

    def publish(self, state: VoiceUiState) -> None:
        if not isinstance(state, VoiceUiState):
            raise TypeError("state must be VoiceUiState")
        if state != self._state:
            self._state_started_at_unix_ms = int(self._wall_clock() * 1000)
        self._state = state
        if state != VoiceUiState.LISTENING:
            with self._activity_lock:
                self._activity_milli = 0
        self._write_snapshot()

    def observe_input_activity(self, activity_milli: int) -> None:
        """Accept only a bounded scalar; raw audio is neither retained nor written."""

        if (
            not isinstance(activity_milli, int)
            or isinstance(activity_milli, bool)
            or not 0 <= activity_milli <= 1000
        ):
            raise ValueError("activity_milli must be an integer from 0 through 1000")
        if self._state == VoiceUiState.LISTENING:
            with self._activity_lock:
                self._activity_milli = activity_milli

    def heartbeat(self) -> None:
        now = self._monotonic_clock()
        interval = (
            self._active_interval_seconds
            if self._state == VoiceUiState.LISTENING
            else self._heartbeat_interval_seconds
        )
        if now - self._last_write_monotonic >= interval:
            self._write_snapshot(monotonic_now=now)

    def _write_snapshot(self, *, monotonic_now: float | None = None) -> None:
        self._generation += 1
        with self._activity_lock:
            activity_milli = self._activity_milli
        payload = json.dumps(
            {
                "activity_milli": activity_milli,
                "generation": self._generation,
                "state": self._state.value,
                "state_started_at_unix_ms": self._state_started_at_unix_ms,
                "updated_at_unix_ms": int(self._wall_clock() * 1000),
                "version": UI_STATE_VERSION,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        if len(payload) > MAXIMUM_UI_STATE_BYTES:
            raise VoicePrivateStateError("voice.ui_state_snapshot_too_large")

        parent_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            parent_fd = os.open(self.path.parent, parent_flags)
        except OSError as exc:
            raise VoicePrivateStateError("voice.ui_state_directory_invalid") from exc
        temporary_name = f".ui-state-{secrets.token_hex(8)}.tmp"
        temporary_fd: int | None = None
        try:
            parent_metadata = os.fstat(parent_fd)
            if not stat.S_ISDIR(parent_metadata.st_mode):
                raise VoicePrivateStateError("voice.ui_state_directory_invalid")
            if os.name == "posix" and (
                parent_metadata.st_uid != os.getuid()
                or stat.S_IMODE(parent_metadata.st_mode) != 0o700
            ):
                raise VoicePrivateStateError("voice.ui_state_directory_permissions_invalid")

            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            temporary_fd = os.open(
                temporary_name,
                flags,
                PRIVATE_FILE_MODE,
                dir_fd=parent_fd,
            )
            if os.name == "posix":
                os.fchmod(temporary_fd, PRIVATE_FILE_MODE)
            written = 0
            while written < len(payload):
                count = os.write(temporary_fd, payload[written:])
                if count <= 0:
                    raise VoicePrivateStateError("voice.ui_state_write_failed")
                written += count
            os.fsync(temporary_fd)
            metadata = os.fstat(temporary_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise VoicePrivateStateError("voice.ui_state_file_invalid")
            if os.name == "posix" and (
                metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != PRIVATE_FILE_MODE
            ):
                raise VoicePrivateStateError("voice.ui_state_file_permissions_invalid")
            os.close(temporary_fd)
            temporary_fd = None
            os.replace(
                temporary_name,
                self.path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
            final = self.path.lstat()
            if (
                stat.S_ISLNK(final.st_mode)
                or not stat.S_ISREG(final.st_mode)
                or (os.name == "posix" and (
                    final.st_uid != os.getuid()
                    or stat.S_IMODE(final.st_mode) != PRIVATE_FILE_MODE
                ))
            ):
                raise VoicePrivateStateError("voice.ui_state_file_invalid")
            self._last_write_monotonic = (
                self._monotonic_clock()
                if monotonic_now is None
                else monotonic_now
            )
        except VoicePrivateStateError:
            raise
        except OSError as exc:
            raise VoicePrivateStateError("voice.ui_state_write_failed") from exc
        finally:
            if temporary_fd is not None:
                try:
                    os.close(temporary_fd)
                except OSError:
                    pass
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            os.close(parent_fd)


__all__ = [
    "MAXIMUM_UI_STATE_BYTES",
    "UI_STATE_VERSION",
    "VoiceUiState",
    "VoiceUiStatePublisher",
    "default_voice_ui_state_path",
]
