"""Isolated scheduler speech queue sharing only the platform TTS device."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from .backends import SpeechUnit, TtsBackend
from .scheduler_protocol import SchedulerEnvelope, SchedulerEventFactory


@dataclass(frozen=True, slots=True)
class SchedulerPlaybackRecord:
    notification_id: str
    unit_index: int
    locale: str
    speakable_text: str


class SchedulerPlaybackJournal(Protocol):
    def next_staged_playback(self) -> SchedulerPlaybackRecord | None: ...
    def set_playback_state(self, notification_id: str, unit_index: int, state: str) -> bool: ...
    def recover_interrupted_playback(self) -> Sequence[tuple[str, int]]: ...
    def cancel_notification_playback(self, notification_id: str) -> Sequence[int]: ...


@dataclass(slots=True)
class _ActivePlayback:
    record: SchedulerPlaybackRecord
    unit: SpeechUnit
    started_receipt_sent: bool = False


class SchedulerPlaybackController:
    """Serialize scheduler units without mutating conversational response state."""

    RESPONSE_PREFIX = "scheduler-playback:"

    def __init__(
        self,
        *,
        journal: SchedulerPlaybackJournal,
        tts: TtsBackend,
        factory: SchedulerEventFactory,
        reserve_epoch: Callable[[], int],
        utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.journal = journal
        self.tts = tts
        self.factory = factory
        self.reserve_epoch = reserve_epoch
        self.utc_now = utc_now
        self.active: _ActivePlayback | None = None

    def _receipt(self, record: SchedulerPlaybackRecord, state: str, reason_code: str) -> SchedulerEnvelope:
        return self.factory.create(
            "scheduler.playback.receipt",
            {
                "notification_id": record.notification_id,
                "unit_index": record.unit_index,
                "state": state,
                "reason_code": reason_code,
                "observed_at_utc": self.utc_now().isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            },
        )

    def startup_recovery(self) -> list[SchedulerEnvelope]:
        return [
            self._receipt(
                SchedulerPlaybackRecord(notification_id, unit_index, "und", ""),
                "CANCELLED",
                "restart_interrupted_playback",
            )
            for notification_id, unit_index in self.journal.recover_interrupted_playback()
        ]

    def pump(self, *, allowed: bool) -> list[SchedulerEnvelope]:
        """Start at most one durable unit when conversation and TTS are idle."""

        if not allowed or self.active is not None or not self.tts.is_idle:
            return []
        record = self.journal.next_staged_playback()
        if record is None:
            return []
        if not self.journal.set_playback_state(record.notification_id, record.unit_index, "PLAYING"):
            return []
        unit = SpeechUnit(
            response_id=f"{self.RESPONSE_PREFIX}{uuid.uuid4()}",
            epoch=self.reserve_epoch(),
            unit_index=record.unit_index,
            locale=record.locale,
            text=record.speakable_text,
        )
        self.active = _ActivePlayback(record, unit)
        try:
            self.tts.enqueue(unit)
        except Exception:
            self.active = None
            self.journal.set_playback_state(record.notification_id, record.unit_index, "SKIPPED")
            return [self._receipt(record, "SKIPPED", "tts_enqueue_failed")]
        return []

    @classmethod
    def owns(cls, unit: SpeechUnit) -> bool:
        return unit.response_id.startswith(cls.RESPONSE_PREFIX)

    def on_started(self, unit: SpeechUnit) -> tuple[bool, list[SchedulerEnvelope]]:
        if not self.owns(unit):
            return False, []
        active = self.active
        if active is None or active.unit != unit or active.started_receipt_sent:
            return True, []
        active.started_receipt_sent = True
        return True, [self._receipt(active.record, "PLAYING", "tts_started")]

    def on_completed(self, unit: SpeechUnit) -> tuple[bool, list[SchedulerEnvelope]]:
        if not self.owns(unit):
            return False, []
        active = self.active
        if active is None or active.unit != unit:
            return True, []
        self.active = None
        if not self.journal.set_playback_state(
            active.record.notification_id, active.record.unit_index, "COMPLETED"
        ):
            return True, []
        return True, [self._receipt(active.record, "COMPLETED", "tts_completed")]

    def cancel_active(self, *, cancellation_epoch: int, reason_code: str) -> list[SchedulerEnvelope]:
        active = self.active
        if active is None:
            return []
        self.active = None
        self.tts.cancel(cancellation_epoch, reason_code)
        cancelled = self.journal.cancel_notification_playback(active.record.notification_id)
        return [
            self._receipt(
                SchedulerPlaybackRecord(
                    active.record.notification_id,
                    unit_index,
                    active.record.locale,
                    "",
                ),
                "CANCELLED",
                reason_code,
            )
            for unit_index in cancelled
        ]

    def cancel_tombstoned(self, notification_id: str, *, cancellation_epoch: int) -> list[SchedulerEnvelope]:
        active = self.active
        if active is None or active.record.notification_id != notification_id:
            return []
        self.active = None
        self.tts.cancel(cancellation_epoch, "notification_tombstoned")
        # The journal tombstone is persisted before this side effect and has
        # already moved the row to CANCELLED.
        return [self._receipt(active.record, "CANCELLED", "notification_tombstoned")]
