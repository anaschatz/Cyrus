"""Small content-free local lifecycle log with bounded rotation."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from .errors import BackendError

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")


class SafeEventLogger:
    """Writes only enumerated event/status/error tokens, never free-form content."""

    def __init__(self, path: str | Path, *, maximum_bytes: int = 1_048_576, backup_count: int = 3) -> None:
        if maximum_bytes < 1 or backup_count < 0:
            raise ValueError("Invalid log rotation bounds")
        self.path = Path(path)
        self.maximum_bytes = maximum_bytes
        self.backup_count = backup_count

    @staticmethod
    def _token(value: str, field: str) -> str:
        if not _SAFE_TOKEN.fullmatch(value):
            raise BackendError(
                "logging.unsafe_token",
                "Local log values must be content-free tokens",
                subsystem="logging",
                safe_context={"field": field},
            )
        return value

    def write(self, event: str, *, status: str | None = None, code: str | None = None) -> None:
        record = {"timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"), "event": self._token(event, "event")}
        if status is not None:
            record["status"] = self._token(status, "status")
        if code is not None:
            record["code"] = self._token(code, "code")
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed(len(encoded))
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "ab") as stream:
                stream.write(encoded)
        except OSError as exc:
            raise BackendError("logging.write_failed", "The local event log could not be written", subsystem="logging") from exc

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        try:
            current_size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if current_size + incoming_bytes <= self.maximum_bytes:
            return
        if self.backup_count == 0:
            self.path.unlink(missing_ok=True)
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backup_count}")
        oldest.unlink(missing_ok=True)
        for index in range(self.backup_count - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))
