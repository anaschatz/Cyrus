"""Typed, content-safe errors for the voice runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class VoiceError(Exception):
    code: str
    message: str
    subsystem: str
    retryable: bool = False
    safe_context: Mapping[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_log_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": "voice.error",
            "code": self.code,
            "subsystem": self.subsystem,
            "retryable": self.retryable,
        }
        if self.safe_context:
            record["context"] = dict(self.safe_context)
        return record


class ProtocolError(VoiceError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, safe_context: Mapping[str, Any] | None = None) -> None:
        super().__init__(code, message, "protocol", retryable, safe_context)


class StateTransitionError(VoiceError):
    def __init__(self, code: str, message: str, *, safe_context: Mapping[str, Any] | None = None) -> None:
        super().__init__(code, message, "state", False, safe_context)


class BackendError(VoiceError):
    def __init__(self, code: str, message: str, *, subsystem: str, retryable: bool = False, safe_context: Mapping[str, Any] | None = None) -> None:
        super().__init__(code, message, subsystem, retryable, safe_context)

