"""Content-free health supervision for the native macOS voice runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MacRecoveryRequest:
    kind: str


class MacRecoverySupervisor:
    """Detect dead CoreAudio streams and wake-sized event-loop gaps.

    The runtime loop is intentionally non-blocking. A multi-second heartbeat
    gap therefore acts as a conservative wake signal and triggers a complete
    CoreAudio/event-tap refresh. Failed reopens use bounded exponential retry.
    """

    def __init__(
        self,
        *,
        sleep_gap_seconds: float = 5.0,
        initial_retry_seconds: float = 0.25,
        maximum_retry_seconds: float = 5.0,
    ) -> None:
        self.sleep_gap_seconds = sleep_gap_seconds
        self.initial_retry_seconds = initial_retry_seconds
        self.maximum_retry_seconds = maximum_retry_seconds
        self._last_monotonic: float | None = None
        self._last_wall: float | None = None
        self._retry_not_before = 0.0
        self._retry_delay = initial_retry_seconds
        self._pending_kind: str | None = None

    def poll(
        self,
        *,
        audio_healthy: bool,
        monotonic_now: float,
        wall_now: float,
    ) -> MacRecoveryRequest | None:
        previous_monotonic = self._last_monotonic
        previous_wall = self._last_wall
        self._last_monotonic = monotonic_now
        self._last_wall = wall_now
        if monotonic_now < self._retry_not_before:
            return None
        if self._pending_kind is not None:
            return MacRecoveryRequest(self._pending_kind)
        woke = False
        if previous_monotonic is not None and previous_wall is not None:
            monotonic_gap = max(0.0, monotonic_now - previous_monotonic)
            wall_gap = max(0.0, wall_now - previous_wall)
            woke = max(monotonic_gap, wall_gap) >= self.sleep_gap_seconds
        if woke:
            return MacRecoveryRequest("sleep_resume")
        if not audio_healthy:
            return MacRecoveryRequest("microphone_disconnect")
        return None

    def succeeded(self) -> None:
        self._pending_kind = None
        self._retry_not_before = 0.0
        self._retry_delay = self.initial_retry_seconds

    def failed(self, kind: str, *, monotonic_now: float) -> None:
        self._pending_kind = kind
        self._retry_not_before = monotonic_now + self._retry_delay
        self._retry_delay = min(self.maximum_retry_seconds, self._retry_delay * 2.0)
