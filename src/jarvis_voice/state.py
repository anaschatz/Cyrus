"""Pure input, response, and playback state machines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .backends import SpeechUnit
from .errors import ProtocolError, StateTransitionError


class InputState(str, Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    TRANSCRIBING = "transcribing"
    SUBMITTING = "submitting"


class ApplyResult(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE = "stale"


@dataclass(slots=True)
class ResponseSession:
    response_id: str
    epoch: int
    units: dict[int, SpeechUnit] = field(default_factory=dict)
    pending: deque[int] = field(default_factory=deque)
    active: set[int] = field(default_factory=set)
    completed: set[int] = field(default_factory=set)
    speech_done_last: int | None = None
    tombstoned: bool = False
    cancelled: bool = False
    playback_started_emitted: bool = False
    playback_completed_emitted: bool = False
    playback_interrupted_emitted: bool = False

    def add_commit(self, unit: SpeechUnit) -> ApplyResult:
        if self.tombstoned or self.cancelled:
            return ApplyResult.STALE
        if self.speech_done_last is not None:
            raise ProtocolError(
                "protocol.commit_after_speech_done",
                "Speech commit arrived after response.speech.done",
                safe_context={"unit_index": unit.unit_index},
            )
        existing = self.units.get(unit.unit_index)
        if existing is not None:
            if existing.locale == unit.locale and existing.text == unit.text:
                return ApplyResult.DUPLICATE
            raise ProtocolError(
                "protocol.conflicting_speech_commit",
                "Speech unit index was reused with different content",
                safe_context={"unit_index": unit.unit_index},
            )
        expected = len(self.units) + 1
        if unit.unit_index != expected:
            raise ProtocolError(
                "protocol.speech_unit_gap",
                "Speech unit index is not the next expected value",
                safe_context={"expected_unit_index": expected, "received_unit_index": unit.unit_index},
            )
        self.units[unit.unit_index] = unit
        self.pending.append(unit.unit_index)
        return ApplyResult.ACCEPTED

    def mark_speech_done(self, last_unit_index: int) -> ApplyResult:
        if self.tombstoned or self.cancelled:
            return ApplyResult.STALE
        if self.speech_done_last is not None:
            if self.speech_done_last == last_unit_index:
                return ApplyResult.DUPLICATE
            raise ProtocolError(
                "protocol.conflicting_speech_done",
                "response.speech.done changed last_unit_index",
                safe_context={
                    "existing_last_unit_index": self.speech_done_last,
                    "received_last_unit_index": last_unit_index,
                },
            )
        if last_unit_index != len(self.units):
            raise ProtocolError(
                "protocol.speech_done_gap",
                "response.speech.done does not match received speech units",
                safe_context={
                    "received_units": len(self.units),
                    "last_unit_index": last_unit_index,
                },
            )
        self.speech_done_last = last_unit_index
        return ApplyResult.ACCEPTED

    def mark_unit_started(self, unit_index: int) -> ApplyResult:
        if self.tombstoned or self.cancelled:
            return ApplyResult.STALE
        if unit_index in self.active or unit_index in self.completed:
            return ApplyResult.DUPLICATE
        if not self.pending or self.pending[0] != unit_index:
            raise StateTransitionError(
                "state.tts_start_out_of_order",
                "TTS unit did not start in queue order",
                safe_context={"unit_index": unit_index},
            )
        self.pending.popleft()
        self.active.add(unit_index)
        return ApplyResult.ACCEPTED

    def mark_unit_completed(self, unit_index: int) -> ApplyResult:
        if self.tombstoned or self.cancelled:
            return ApplyResult.STALE
        if unit_index in self.completed:
            return ApplyResult.DUPLICATE
        if unit_index not in self.active:
            raise StateTransitionError(
                "state.tts_complete_without_start",
                "TTS unit completed before it started",
                safe_context={"unit_index": unit_index},
            )
        self.active.remove(unit_index)
        self.completed.add(unit_index)
        return ApplyResult.ACCEPTED

    def tombstone(self, *, cancelled: bool = False) -> bool:
        if self.tombstoned:
            return False
        self.tombstoned = True
        self.cancelled = self.cancelled or cancelled
        self.pending.clear()
        self.active.clear()
        if self.playback_completed_emitted or self.playback_interrupted_emitted:
            return False
        self.playback_interrupted_emitted = True
        return True

    def can_complete_playback(self, *, backend_idle: bool) -> bool:
        if (
            self.tombstoned
            or self.cancelled
            or self.playback_completed_emitted
            or self.speech_done_last is None
            or not backend_idle
            or self.pending
            or self.active
        ):
            return False
        expected = set(range(1, self.speech_done_last + 1))
        return self.completed == expected


@dataclass(frozen=True, slots=True)
class PttDownResult:
    epoch: int
    turn_id: str
    cancel_response_id: str | None
    emit_playback_interrupted: bool
    ignored: bool = False


class VoiceStateMachine:
    def __init__(self) -> None:
        self.input_state = InputState.IDLE
        self.turn_id: str | None = None
        self.output_epoch = 0
        self.response: ResponseSession | None = None

    def reserve_output_epoch(self) -> int:
        """Reserve the shared backend cancellation fence for isolated output."""

        self.output_epoch += 1
        return self.output_epoch

    def ptt_down(self, turn_id: str) -> PttDownResult:
        if self.input_state == InputState.CAPTURING:
            return PttDownResult(
                epoch=self.output_epoch,
                turn_id=self.turn_id or turn_id,
                cancel_response_id=None,
                emit_playback_interrupted=False,
                ignored=True,
            )
        # The epoch changes before any old output is touched. Platform adapters can
        # use the returned epoch as their atomic mute/cancellation fence.
        self.output_epoch += 1
        cancel_response_id: str | None = None
        emit_interrupted = False
        if self.response is not None and not self.response.playback_completed_emitted:
            cancel_response_id = self.response.response_id
            emit_interrupted = self.response.tombstone(cancelled=True)
        self.turn_id = turn_id
        self.input_state = InputState.CAPTURING
        return PttDownResult(
            epoch=self.output_epoch,
            turn_id=turn_id,
            cancel_response_id=cancel_response_id,
            emit_playback_interrupted=emit_interrupted,
        )

    def ptt_up(self) -> None:
        if self.input_state != InputState.CAPTURING:
            raise StateTransitionError(
                "state.ptt_up_without_capture", "PTT_UP requires an active capture"
            )
        self.input_state = InputState.TRANSCRIBING

    def transcription_complete(self) -> None:
        if self.input_state != InputState.TRANSCRIBING:
            raise StateTransitionError(
                "state.transcription_not_active",
                "Transcription completion requires TRANSCRIBING state",
            )
        self.input_state = InputState.SUBMITTING

    def no_speech_complete(self) -> None:
        if self.input_state != InputState.TRANSCRIBING:
            raise StateTransitionError(
                "state.no_speech_not_transcribing",
                "No-speech completion requires TRANSCRIBING state",
            )
        self.input_state = InputState.IDLE

    def abort_input(self) -> None:
        self.input_state = InputState.IDLE
        self.turn_id = None

    def submission_complete(self) -> None:
        if self.input_state != InputState.SUBMITTING:
            raise StateTransitionError(
                "state.submission_not_active",
                "Submission completion requires SUBMITTING state",
            )
        self.input_state = InputState.IDLE

    def start_response(self, response_id: str) -> ResponseSession:
        if self.response is not None and not (
            self.response.tombstoned or self.response.playback_completed_emitted
        ):
            self.response.tombstone(cancelled=True)
        self.output_epoch += 1
        self.response = ResponseSession(response_id=response_id, epoch=self.output_epoch)
        return self.response

    def _matching_response(self, response_id: str, epoch: int | None = None) -> ResponseSession | None:
        if self.response is None or self.response.response_id != response_id:
            return None
        if epoch is not None and self.response.epoch != epoch:
            return None
        return self.response

    def add_speech_commit(
        self, response_id: str, unit_index: int, locale: str, text: str
    ) -> tuple[ApplyResult, SpeechUnit | None]:
        response = self._matching_response(response_id)
        if response is None:
            return ApplyResult.STALE, None
        unit = SpeechUnit(
            response_id=response_id,
            epoch=response.epoch,
            unit_index=unit_index,
            locale=locale,
            text=text,
        )
        result = response.add_commit(unit)
        return result, unit if result == ApplyResult.ACCEPTED else None

    def mark_speech_done(self, response_id: str, last_unit_index: int) -> ApplyResult:
        response = self._matching_response(response_id)
        if response is None:
            return ApplyResult.STALE
        return response.mark_speech_done(last_unit_index)

    def mark_unit_started(self, unit: SpeechUnit) -> ApplyResult:
        response = self._matching_response(unit.response_id, unit.epoch)
        if response is None:
            return ApplyResult.STALE
        return response.mark_unit_started(unit.unit_index)

    def mark_unit_completed(self, unit: SpeechUnit) -> ApplyResult:
        response = self._matching_response(unit.response_id, unit.epoch)
        if response is None:
            return ApplyResult.STALE
        return response.mark_unit_completed(unit.unit_index)

    def cancel_response(self, response_id: str) -> bool:
        response = self._matching_response(response_id)
        if response is None:
            return False
        return response.tombstone(cancelled=True)

    def maybe_complete_playback(self, *, backend_idle: bool) -> bool:
        if self.response is None or not self.response.can_complete_playback(
            backend_idle=backend_idle
        ):
            return False
        self.response.playback_completed_emitted = True
        return True
