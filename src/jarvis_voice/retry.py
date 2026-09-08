"""Deterministic acknowledgement retry scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

RETRY_OFFSETS = (0.0, 0.5, 1.5, 3.5)
ACK_DEADLINE_SECONDS = 4.0
RECONNECT_TTL_SECONDS = 30.0


@dataclass(slots=True)
class AckRetryScheduler:
    event_id: str
    clock: Callable[[], float]
    started_at: float | None = None
    acknowledged: bool = False
    _emitted_attempts: set[int] = field(default_factory=set)

    def start(self) -> None:
        if self.started_at is not None:
            raise RuntimeError("Retry scheduler has already started")
        self.started_at = self.clock()
        self._emitted_attempts.add(0)  # The caller performs the initial send.

    def due_attempts(self) -> list[int]:
        if self.started_at is None:
            raise RuntimeError("Retry scheduler has not started")
        if self.acknowledged:
            return []
        elapsed = self.clock() - self.started_at
        due: list[int] = []
        for attempt, offset in enumerate(RETRY_OFFSETS):
            if elapsed >= offset and attempt not in self._emitted_attempts:
                self._emitted_attempts.add(attempt)
                due.append(attempt)
        return due

    def acknowledge(self, acked_event_id: str) -> bool:
        if acked_event_id != self.event_id:
            return False
        self.acknowledged = True
        return True

    @property
    def connected_timed_out(self) -> bool:
        return bool(self.started_at is not None and not self.acknowledged and self.clock() - self.started_at >= ACK_DEADLINE_SECONDS)

    @property
    def reconnect_ttl_expired(self) -> bool:
        return bool(self.started_at is not None and not self.acknowledged and self.clock() - self.started_at >= RECONNECT_TTL_SECONDS)
