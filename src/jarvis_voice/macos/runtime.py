"""macOS event pump wired to the shared protocol/state core."""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..audio import resample_audio
from ..backends import SpeechUnit
from ..config import VoiceConfig
from ..core import VoiceCore
from ..cloud_approval import (
    CloudApprovalController,
    CloudApprovalEnvelope,
    CloudApprovalJournal,
    default_cloud_approval_journal_path,
)
from ..action_approval import (
    ActionApprovalController,
    ActionApprovalEnvelope,
    ActionApprovalJournal,
    default_action_approval_journal_path,
)
from ..delivery_journal import DeliveryDedupJournal, default_delivery_journal_path
from ..errors import BackendError
from ..protocol import Envelope
from ..ptt import ApprovalKeyPublisher, PttEvent, PttEventPublisher, PttTransition
from ..reliable import ReliableSubmissionManager
from ..transcription import AsyncTranscriptionCoordinator
from ..transport import LoopbackWebSocketTransport
from ..state import InputState
from ..vad import SileroOnnxVad
from .audio_capture import CoreAudioCapture
from .asr_selection import create_macos_asr_backend
from .ptt import MacEventTapPttSource, mac_ptt_keycode, mac_ptt_vkey
from .system_tts import MacSystemTtsBackend


@dataclass(frozen=True, slots=True)
class BackendCallback:
    kind: str
    unit: SpeechUnit


class MacVoiceRuntime:
    def __init__(
        self,
        config: VoiceConfig,
        *,
        audio_frame_observer: Callable[[bytes, int, float], None] | None = None,
    ) -> None:
        if sys.platform != "darwin":
            raise BackendError("runtime.macos_only", "The macOS runtime requires native macOS", subsystem="runtime")
        artifact = config.vad.artifact
        if not artifact.path or artifact.size is None or not artifact.sha256:
            raise BackendError("vad.artifact_manifest_required", "Silero artifact manifest is required", subsystem="vad")
        mac_ptt_keycode(config.ptt_key)
        if any(not config.macos.system_voices.get(locale) for locale in ("el-GR", "en-US")):
            raise BackendError("tts.macos_voice_mapping_required", "Explicit Greek and English macOS voices are required", subsystem="tts")
        self.config = config
        self.callbacks: queue.Queue[BackendCallback] = queue.Queue()
        self.tts = MacSystemTtsBackend(
            voice_mapping=config.macos.system_voices,
            on_started=lambda unit: self.callbacks.put_nowait(BackendCallback("started", unit)),
            on_completed=lambda unit: self.callbacks.put_nowait(BackendCallback("completed", unit)),
        )
        self.asr = create_macos_asr_backend(config)
        self.delivery_journal = DeliveryDedupJournal(default_delivery_journal_path())
        self.core = VoiceCore(asr=self.asr, tts=self.tts, delivery_journal=self.delivery_journal)
        self.cloud_journal = CloudApprovalJournal(default_cloud_approval_journal_path())
        self.cloud_approval = CloudApprovalController(session_id=self.core.session_id, journal=self.cloud_journal)
        self.action_journal = ActionApprovalJournal(default_action_approval_journal_path())
        self.action_approval = ActionApprovalController(
            session_id=self.core.session_id,
            journal=self.action_journal,
            approval_arbiter=self.cloud_approval.approval_arbiter,
        )
        self.transcription = AsyncTranscriptionCoordinator(self.asr)
        self.vad = SileroOnnxVad(
            artifact_path=artifact.path,
            size=artifact.size,
            sha256=artifact.sha256,
            threshold=config.vad.threshold,
            minimum_speech_ms=config.vad.minimum_speech_ms,
            padding_ms=config.vad.padding_ms,
        )
        self.transport = LoopbackWebSocketTransport(config.orchestrator_url)
        self.publisher = PttEventPublisher(
            capacity=config.ptt_queue_capacity,
            maximum_hold_seconds=config.maximum_hold_seconds,
            ptt_vkey=mac_ptt_vkey(config.ptt_key),
        )
        self.approval_keys = ApprovalKeyPublisher()
        self.ptt = MacEventTapPttSource(
            self.publisher,
            key=config.ptt_key,
            approval_publisher=self.approval_keys,
        )
        self.audio = CoreAudioCapture(
            queue_capacity=config.audio_queue_capacity,
            maximum_utterance_seconds=config.maximum_utterance_seconds,
            device=config.audio_input_device,
            frame_observer=audio_frame_observer,
        )
        self._running = False
        self._stop_requested = threading.Event()
        self._timings_lock = threading.Lock()
        self.timings: list[dict[str, float | str | int | bool]] = []
        self._ptt_cycle = 0
        self._turn_cycles: dict[str, int] = {}
        self._response_tokens: dict[str, int] = {}
        self._next_response_token = 1
        self.reliable = ReliableSubmissionManager(time.monotonic)

    def _response_token(self, response_id: str | None) -> int:
        if response_id is None:
            return 0
        token = self._response_tokens.get(response_id)
        if token is None:
            token = self._next_response_token
            self._next_response_token += 1
            self._response_tokens[response_id] = token
        return token

    def _record_timing(self, event: str, timestamp: float | None = None, **details: object) -> None:
        entry: dict[str, float | str | int | bool] = {
            "event": event,
            "timestamp": time.perf_counter() if timestamp is None else timestamp,
        }
        for key, value in details.items():
            if isinstance(value, (str, int, float, bool)):
                entry[key] = value
        with self._timings_lock:
            self.timings.append(entry)

    def timing_snapshot(self) -> tuple[dict[str, float | str | int | bool], ...]:
        """Return content-free native telemetry without exposing mutable state."""

        with self._timings_lock:
            return tuple(dict(item) for item in self.timings)

    def _send_all(self, events: list[Envelope]) -> None:
        for event in events:
            if event.type == "input.transcript.final":
                self.reliable.begin(event)
                language = event.payload.get("language", {})
                code = language.get("code", "unknown") if isinstance(language, Mapping) else "unknown"
                self._record_timing(
                    "transcript_final_created",
                    language=str(code),
                    ptt_cycle=self._turn_cycles.get(event.turn_id or "", self._ptt_cycle),
                )
            elif event.type == "input.no_speech":
                self._record_timing("input_no_speech_created")
            elif event.type == "response.cancel":
                self._record_timing("response_cancel_created")
            elif event.type == "playback.interrupted":
                self._record_timing("playback_interrupted_created")
            elif event.type == "playback.completed":
                self._record_timing(
                    "playback_completed_created",
                    response_token=self._response_token(event.response_id),
                )
            self.transport.send(event)

    def _reconnect_pending(self) -> None:
        if not self.reliable.can_reconnect:
            raise BackendError("transport.reconnect_ttl_expired", "Transcript reconnect TTL expired", subsystem="transport")
        self.transport.close()
        self.delivery_journal.close()
        self.core.begin_new_connection()
        self.cloud_approval.begin_new_connection()
        self.action_approval.begin_new_connection()
        self.transport.send(self.core.client_hello())
        self.transport.send(self.reliable.reconnect_event(self.core))

    def _handle_ptt(self, event: PttEvent) -> None:
        if event.transition == PttTransition.DOWN:
            self._ptt_cycle += 1
            self.cloud_approval.disarm_for_ptt()
            output_active = not self.tts.is_idle
            self._record_timing(
                "ptt_down_received",
                event.received_at,
                output_active=output_active,
                ptt_cycle=self._ptt_cycle,
            )
            outbound = self.core.ptt_down()
            if any(item.type == "response.cancel" for item in outbound):
                self._record_timing("tts_cancel_dispatched")
            self.transcription.cancel_current()
            self.audio.begin_turn()
            self._send_all(outbound)
            return
        try:
            self._record_timing(
                "ptt_up_received",
                event.received_at,
                ptt_cycle=self._ptt_cycle,
                forced=event.transition == PttTransition.FORCED_UP,
                reason=event.reason,
            )
            captured = self.audio.end_turn()
            normalized = resample_audio(captured)
            prepared = self.core.prepare_transcription(normalized, self.vad)
            if prepared.turn_id is not None:
                self._turn_cycles[prepared.turn_id] = self._ptt_cycle
            if isinstance(prepared, Envelope):
                self._send_all([prepared])
            else:
                self.transcription.submit(prepared.turn_id, prepared.audio)
        except BackendError as exc:
            self.audio.abort_turn()
            self._send_all([self.core.input_failed(exc.code)])

    def _poll_transcription(self) -> None:
        outcome = self.transcription.poll()
        if outcome is None:
            return
        if outcome.error is None and outcome.result is not None:
            event = self.core.complete_transcription(outcome.turn_id, outcome.result)
        elif outcome.error is not None and outcome.error.code == "asr.empty_transcript":
            event = self.core.complete_no_speech(outcome.turn_id, "asr_empty_transcript")
        else:
            code = outcome.error.code if outcome.error is not None else "asr.backend_failed"
            event = self.core.transcription_failed(outcome.turn_id, code)
        if event is not None:
            self._send_all([event])

    def run(self) -> None:  # pragma: no cover - native hardware/TCC loop
        if self._stop_requested.is_set():
            self.close()
            return
        self._running = True
        try:
            self.audio.start()
            self._record_timing("audio_stream_started")
            self.ptt.start()
            self._record_timing("ptt_source_started")
            self.transport.send(self.core.client_hello())
            while self._running and not self._stop_requested.is_set():
                now = time.perf_counter()
                self.publisher.watchdog(now)
                try:
                    self._handle_ptt(self.publisher.events.get(timeout=0.01))
                except queue.Empty:
                    pass
                forced = self.publisher.take_forced_release()
                if forced is not None:
                    self._handle_ptt(forced)
                while True:
                    try:
                        key_event = self.approval_keys.events.get_nowait()
                    except queue.Empty:
                        break
                    if key_event.is_down:
                        decision = self.action_approval.key_down(
                            key_event.key,
                            key_counter=key_event.counter,
                            capture_idle=self.core.state.input_state == InputState.IDLE,
                            autorepeat=key_event.autorepeat,
                        ) or self.cloud_approval.key_down(
                            key_event.key,
                            key_counter=key_event.counter,
                            capture_idle=self.core.state.input_state == InputState.IDLE,
                            autorepeat=key_event.autorepeat,
                        )
                        if decision is not None:
                            self.transport.send(decision)
                            if isinstance(decision, ActionApprovalEnvelope):
                                self.action_journal.mark_sent(decision.event_id, now=now)
                            else:
                                self.cloud_approval.mark_decision_sent(decision.event_id, now=now)
                self._poll_transcription()
                while True:
                    try:
                        callback = self.callbacks.get_nowait()
                    except queue.Empty:
                        break
                    self._record_timing(
                        "tts_unit_started" if callback.kind == "started" else "tts_unit_completed",
                        unit_index=callback.unit.unit_index,
                        locale=callback.unit.locale,
                        response_token=self._response_token(callback.unit.response_id),
                    )
                    events = (
                        self.core.on_tts_unit_started(callback.unit)
                        if callback.kind == "started"
                        else self.core.on_tts_unit_completed(callback.unit)
                    )
                    self._send_all(events)
                for retry in self.reliable.due_retransmissions(self.core):
                    self.transport.send(retry)
                for retry in self.cloud_approval.due_decision_retransmissions(now=now):
                    self.transport.send(retry)
                for retry in self.action_approval.due_decision_retransmissions(now=now):
                    self.transport.send(retry)
                    self.action_journal.mark_sent(retry.event_id, now=now)
                if self.reliable.consume_connected_timeout():
                    self._reconnect_pending()
                while True:
                    server_event = self.transport.receive()
                    if server_event is None:
                        break
                    if isinstance(server_event, CloudApprovalEnvelope):
                        for reply in self.cloud_approval.handle_server_event(
                            server_event,
                            key_counter=self.approval_keys.counter,
                            both_keys_up=self.approval_keys.both_keys_up,
                        ):
                            self.transport.send(reply)
                        continue
                    if isinstance(server_event, ActionApprovalEnvelope):
                        for reply in self.action_approval.handle_server_event(
                            server_event,
                            key_counter=self.approval_keys.counter,
                            both_keys_up=self.approval_keys.both_keys_up,
                        ):
                            self.transport.send(reply)
                        continue
                    if server_event.type == "response.speech.commit":
                        self._record_timing(
                            "speech_commit_received",
                            unit_index=server_event.payload["unit_index"],
                            locale=server_event.payload["locale"],
                            response_token=self._response_token(server_event.response_id),
                        )
                    elif server_event.type == "response.speech.done":
                        self._record_timing(
                            "speech_done_received",
                            last_unit_index=server_event.payload["last_unit_index"],
                            response_token=self._response_token(server_event.response_id),
                        )
                    if server_event.type == "server.ack":
                        self.reliable.acknowledge(server_event)
                    self._send_all(self.core.handle_server_event(server_event))
        finally:
            self.close()

    def request_stop(self) -> None:
        self._stop_requested.set()
        self._running = False
        self.audio.abort_turn()
        self.transcription.cancel_current()
        self.tts.cancel(2**63 - 1, "application_stop")

    def close(self) -> None:
        self._running = False
        self._stop_requested.set()
        self.ptt.stop()
        self.audio.abort_turn()
        self.audio.stop()
        self.transcription.close()
        self.tts.close()
        self.transport.close()
        self.cloud_journal.close()
