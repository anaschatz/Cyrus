"""Deterministic reliable submission lifecycle for transcript ACKs."""

from __future__ import annotations

from collections.abc import Callable

from .core import VoiceCore
from .protocol import Envelope
from .retry import AckRetryScheduler


class ReliableSubmissionManager:
    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.original: Envelope | None = None
        self.scheduler: AckRetryScheduler | None = None
        self._timeout_consumed = False

    def begin(self, event: Envelope) -> None:
        if event.type != "input.transcript.final":
            raise ValueError("Only final transcripts require reliable submission")
        self.original = event
        self.scheduler = AckRetryScheduler(event.event_id, self.clock)
        self.scheduler.start()
        self._timeout_consumed = False

    def cancel(self) -> None:
        self.original = None
        self.scheduler = None

    def acknowledge(self, event: Envelope) -> bool:
        if event.type != "server.ack" or self.scheduler is None:
            return False
        accepted = self.scheduler.acknowledge(event.payload["acked_event_id"])
        if accepted:
            self.original = None
        return accepted

    def due_retransmissions(self, core: VoiceCore) -> list[Envelope]:
        if self.scheduler is None or self.original is None:
            return []
        return [core.retransmit(self.original) for _attempt in self.scheduler.due_attempts()]

    def consume_connected_timeout(self) -> bool:
        if self.scheduler is None or self._timeout_consumed or not self.scheduler.connected_timed_out:
            return False
        self._timeout_consumed = True
        return True

    @property
    def can_reconnect(self) -> bool:
        return bool(self.scheduler is not None and self.original is not None and not self.scheduler.reconnect_ttl_expired)

    def reconnect_event(self, core: VoiceCore) -> Envelope:
        if not self.can_reconnect or self.original is None:
            raise RuntimeError("No reconnectable submission")
        return core.retransmit(self.original)
