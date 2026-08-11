"""Small orchestration core connecting protocol events to pure state machines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .backends import AsrBackend, AudioChunk, SpeechUnit, TranscriptResult, TtsBackend, VadBackend
from .delivery_journal import DeliveryDedupJournal
from .errors import BackendError, ProtocolError, StateTransitionError
from .protocol import Envelope, EventFactory, SERVER_EVENT_TYPES, new_uuid4
from .sequencing import SequenceResult, SequenceTracker
from .state import ApplyResult, InputState, VoiceStateMachine


@dataclass(frozen=True, slots=True)
class PreparedTranscription:
    turn_id: str
    audio: AudioChunk


class VoiceCore:
    def __init__(
        self,
        *,
        asr: AsrBackend,
        tts: TtsBackend,
        session_id: str | None = None,
        delivery_journal: DeliveryDedupJournal | None = None,
    ) -> None:
        self.state = VoiceStateMachine()
        self.asr = asr
        self.tts = tts
        self.client_events = EventFactory(session_id=session_id)
        self.server_sequence = SequenceTracker()
        self.pending_submission_event_id: str | None = None
        self.delivery_journal = delivery_journal

    @property
    def session_id(self) -> str:
        return self.client_events.session_id

    def begin_new_connection(self) -> None:
        """Reset per-connection sender and receiver sequence numbers."""

        self.client_events = EventFactory(session_id=self.session_id)
        self.server_sequence = SequenceTracker()

    def retransmit(self, event: Envelope) -> Envelope:
        """Create a new frame with a new seq and the original event identity."""

        return self.client_events.create(
            event.type,
            event.payload,
            event_id=event.event_id,
            turn_id=event.turn_id,
            response_id=event.response_id,
        )

    def client_hello(self) -> Envelope:
        return self.client_events.create(
            "client.hello",
            {
                "protocol": "jarvis.voice/1",
                "capabilities": [
                    "speech_commits",
                    "speech_done",
                    "output_epochs",
                    "jarvis.cloud-approval/1",
                ],
            },
        )

    def ptt_down(self, turn_id: str | None = None) -> list[Envelope]:
        turn_id = turn_id or new_uuid4()
        result = self.state.ptt_down(turn_id)
        if result.ignored:
            return []
        self.tts.cancel(result.epoch, "ptt_down")
        events: list[Envelope] = []
        if result.cancel_response_id is not None:
            events.append(
                self.client_events.create(
                    "response.cancel",
                    {"reason": "ptt_down", "epoch": result.epoch},
                    turn_id=turn_id,
                    response_id=result.cancel_response_id,
                )
            )
            if result.emit_playback_interrupted:
                events.append(
                    self.client_events.create(
                        "playback.interrupted",
                        {"reason": "ptt_down", "epoch": result.epoch},
                        turn_id=turn_id,
                        response_id=result.cancel_response_id,
                    )
                )
        events.append(
            self.client_events.create(
                "input.capture.started",
                {"epoch": result.epoch},
                turn_id=turn_id,
            )
        )
        return events

    def ptt_up_and_transcribe(self, audio: object) -> Envelope:
        self.state.ptt_up()
        result = self.asr.transcribe(audio)
        self.state.transcription_complete()
        event = self.client_events.create(
            "input.transcript.final",
            result.to_payload(),
            turn_id=self.state.turn_id,
        )
        self.pending_submission_event_id = event.event_id
        return event

    def ptt_up_and_process(self, audio: AudioChunk, vad: VadBackend) -> Envelope:
        """Finish capture and emit either no-speech or a final transcript."""

        prepared = self.prepare_transcription(audio, vad)
        if isinstance(prepared, Envelope):
            return prepared
        try:
            result = self.asr.transcribe(prepared.audio)
        except BackendError as exc:
            if exc.code != "asr.empty_transcript":
                raise
            event = self.complete_no_speech(prepared.turn_id, "asr_empty_transcript")
            if event is None:
                raise StateTransitionError("state.stale_transcription", "Transcription result became stale")
            return event
        event = self.complete_transcription(prepared.turn_id, result)
        if event is None:
            raise StateTransitionError("state.stale_transcription", "Transcription result became stale")
        return event

    def prepare_transcription(self, audio: AudioChunk, vad: VadBackend) -> PreparedTranscription | Envelope:
        """End capture and VAD without blocking the event loop on ASR."""

        self.state.ptt_up()
        turn_id = self.state.turn_id
        if turn_id is None:
            raise AssertionError("An active input turn must have a turn_id")
        vad_result = vad.trim(audio)
        if not vad_result.speech_detected:
            self.state.no_speech_complete()
            return self.client_events.create(
                "input.no_speech",
                {"reason": "vad_no_speech"},
                turn_id=turn_id,
            )
        return PreparedTranscription(turn_id, vad_result.audio)

    def complete_transcription(self, turn_id: str, result: TranscriptResult) -> Envelope | None:
        """Publish a result only if its turn is still the active transcription."""

        if self.state.input_state != InputState.TRANSCRIBING or self.state.turn_id != turn_id:
            return None
        self.state.transcription_complete()
        event = self.client_events.create(
            "input.transcript.final",
            result.to_payload(),
            turn_id=turn_id,
        )
        self.pending_submission_event_id = event.event_id
        return event

    def complete_no_speech(self, turn_id: str, reason: str) -> Envelope | None:
        if self.state.input_state != InputState.TRANSCRIBING or self.state.turn_id != turn_id:
            return None
        self.state.no_speech_complete()
        return self.client_events.create("input.no_speech", {"reason": reason}, turn_id=turn_id)

    def transcription_failed(self, turn_id: str, code: str) -> Envelope | None:
        if self.state.input_state != InputState.TRANSCRIBING or self.state.turn_id != turn_id:
            return None
        return self.input_failed(code)

    def input_failed(self, code: str) -> Envelope:
        turn_id = self.state.turn_id
        self.state.abort_input()
        return self.client_events.create(
            "client.error",
            {"code": code, "retryable": False},
            turn_id=turn_id,
        )

    def handle_server_event(self, event: Envelope) -> list[Envelope]:
        if event.type not in SERVER_EVENT_TYPES:
            raise ProtocolError(
                "protocol.wrong_direction",
                "Client event received on the server event path",
                safe_context={"event_type": event.type},
            )
        sequence_result = self.server_sequence.observe(event)
        if sequence_result == SequenceResult.DUPLICATE:
            return []
        if self.delivery_journal is not None and self.delivery_journal.observe(event):
            self._synchronize_suppressed_replay(event)
            return []

        if event.type == "server.ack":
            if (
                self.state.input_state == InputState.SUBMITTING
                and event.payload["acked_event_id"]
                == self.pending_submission_event_id
            ):
                self.state.submission_complete()
                self.pending_submission_event_id = None
            return []
        if event.type == "response.started":
            if event.response_id is None:
                raise AssertionError("Envelope validation should require response_id")
            self.state.start_response(event.response_id)
            return []
        if event.type == "response.speech.commit":
            if event.response_id is None:
                raise AssertionError("Envelope validation should require response_id")
            result, unit = self.state.add_speech_commit(
                event.response_id,
                event.payload["unit_index"],
                event.payload["locale"],
                event.payload["text"],
            )
            if result == ApplyResult.ACCEPTED and unit is not None:
                self.tts.enqueue(unit)
            return []
        if event.type == "response.speech.done":
            if event.response_id is None:
                raise AssertionError("Envelope validation should require response_id")
            self.state.mark_speech_done(
                event.response_id, event.payload["last_unit_index"]
            )
            return self._playback_completed_if_ready()
        if event.type == "response.cancelled":
            if event.response_id is not None and self.state.cancel_response(event.response_id):
                self.tts.cancel(self.state.output_epoch, "response_cancelled")
            return []
        # Text events and server status/error events do not affect speech completion.
        return []

    def _synchronize_suppressed_replay(self, event: Envelope) -> None:
        """Rebuild volatile response state without repeating user-visible effects.

        The journal is persisted before effects. If the process dies in that
        narrow window, replay must still establish enough state for later,
        previously unseen speech units. No TTS enqueue, text application or
        playback telemetry is produced here.
        """

        if event.type == "response.started" and event.response_id is not None:
            try:
                self.state.start_response(event.response_id)
            except StateTransitionError:
                pass
        elif event.type == "response.speech.commit" and event.response_id is not None:
            try:
                self.state.add_speech_commit(
                    event.response_id,
                    event.payload["unit_index"],
                    event.payload["locale"],
                    event.payload["text"],
                )
            except StateTransitionError:
                pass
        elif event.type == "response.speech.done" and event.response_id is not None:
            try:
                self.state.mark_speech_done(event.response_id, event.payload["last_unit_index"])
            except StateTransitionError:
                pass
        elif event.type == "response.cancelled" and event.response_id is not None:
            self.state.cancel_response(event.response_id)

    def on_tts_unit_started(self, unit: SpeechUnit) -> list[Envelope]:
        result = self.state.mark_unit_started(unit)
        if result != ApplyResult.ACCEPTED or self.state.response is None:
            return []
        if self.state.response.playback_started_emitted:
            return []
        self.state.response.playback_started_emitted = True
        return [
            self.client_events.create(
                "playback.started",
                {"first_unit_index": unit.unit_index, "epoch": unit.epoch},
                response_id=unit.response_id,
                turn_id=self.state.turn_id,
            )
        ]

    def on_tts_unit_completed(self, unit: SpeechUnit) -> list[Envelope]:
        result = self.state.mark_unit_completed(unit)
        if result != ApplyResult.ACCEPTED:
            return []
        return self._playback_completed_if_ready()

    def _playback_completed_if_ready(self) -> list[Envelope]:
        if not self.state.maybe_complete_playback(backend_idle=self.tts.is_idle):
            return []
        response = self.state.response
        if response is None:
            return []
        return [
            self.client_events.create(
                "playback.completed",
                {
                    "last_unit_index": response.speech_done_last,
                    "epoch": response.epoch,
                },
                response_id=response.response_id,
                turn_id=self.state.turn_id,
            )
        ]

    def handle_server_events(self, events: Iterable[Envelope]) -> list[Envelope]:
        outbound: list[Envelope] = []
        for event in events:
            outbound.extend(self.handle_server_event(event))
        return outbound
