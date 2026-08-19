"""Source-run Windows event pump connecting the guarded adapters."""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass

from .asr_factory import AsrBackendFactory
from .audio import resample_audio
from .backends import SpeechUnit
from .config import VoiceConfig, configured_voice_auth_key
from .core import VoiceCore
from .cloud_approval import (
    CloudApprovalController,
    CloudApprovalEnvelope,
    CloudApprovalJournal,
    default_cloud_approval_journal_path,
)
from .action_approval import (
    ActionApprovalController,
    ActionApprovalEnvelope,
    ActionApprovalJournal,
    default_action_approval_journal_path,
)
from .delivery_journal import DeliveryDedupJournal, default_delivery_journal_path
from .errors import BackendError
from .ptt import ApprovalKeyPublisher, PttEvent, PttEventPublisher, PttTransition, function_key_vkey
from .private_state import default_voice_state_boundary, prepare_voice_journal_state
from .reliable import ReliableSubmissionManager
from .transport import LoopbackWebSocketTransport
from .transcription import AsyncTranscriptionCoordinator
from .state import InputState
from .vad import SileroOnnxVad
from .windows.audio_capture import WasapiAudioCapture
from .windows.raw_input import RawInputPttSource
from .windows.sapi import SapiTtsBackend
from .scheduler_protocol import SchedulerEnvelope
from .scheduler_runtime import SchedulerNativeRuntime


@dataclass(frozen=True, slots=True)
class BackendCallback:
    kind: str
    unit: SpeechUnit


class WindowsVoiceRuntime:
    def __init__(self, config: VoiceConfig, *, asr_factory: AsrBackendFactory | None = None) -> None:
        if sys.platform != "win32":
            raise BackendError("runtime.windows_only", "The source runtime requires native Windows", subsystem="runtime")
        voice_auth_key = configured_voice_auth_key(config)
        artifact = config.vad.artifact
        if not artifact.path or artifact.size is None or not artifact.sha256:
            raise BackendError("vad.artifact_manifest_required", "Silero artifact manifest is required", subsystem="vad")
        if not config.whisper_model_directory:
            raise BackendError("asr.model_directory_required", "Whisper model directory is required", subsystem="asr")
        journal_paths = (
            default_delivery_journal_path(),
            default_cloud_approval_journal_path(),
            default_action_approval_journal_path(),
        )
        state_boundary = default_voice_state_boundary()
        # Validate/harden the entire state family before any journal connects
        # or an expensive native backend is initialized.
        prepare_voice_journal_state(
            journal_paths,
            trusted_boundary=state_boundary,
        )
        self.config = config
        self.callbacks: queue.Queue[BackendCallback] = queue.Queue()
        self.tts = SapiTtsBackend(
            voice_mapping=config.sapi_voices,
            on_started=lambda unit: self.callbacks.put_nowait(BackendCallback("started", unit)),
            on_completed=lambda unit: self.callbacks.put_nowait(BackendCallback("completed", unit)),
        )
        self.asr = (asr_factory or AsrBackendFactory()).create(config)
        self.delivery_journal = DeliveryDedupJournal(
            journal_paths[0],
            trusted_boundary=state_boundary,
        )
        self.core = VoiceCore(asr=self.asr, tts=self.tts, delivery_journal=self.delivery_journal)
        self.cloud_journal = CloudApprovalJournal(
            journal_paths[1],
            trusted_boundary=state_boundary,
        )
        self.cloud_approval = CloudApprovalController(session_id=self.core.session_id, journal=self.cloud_journal)
        self.action_journal = ActionApprovalJournal(
            journal_paths[2],
            trusted_boundary=state_boundary,
        )
        self.action_approval = ActionApprovalController(
            session_id=self.core.session_id,
            journal=self.action_journal,
            approval_arbiter=self.cloud_approval.approval_arbiter,
        )
        self.scheduler_runtime: SchedulerNativeRuntime | None = None
        if config.scheduler.enabled:
            self.scheduler_runtime = SchedulerNativeRuntime.from_config(
                config.scheduler,
                session_id=self.core.session_id,
                approval_arbiter=self.cloud_approval.approval_arbiter,
                tts=self.tts,
                reserve_epoch=self.core.state.reserve_output_epoch,
            )
            self.core.configure_scheduler_extension(
                self.scheduler_runtime.capabilities,
                self.scheduler_runtime.manifest_sha256,
                self.scheduler_runtime.callback_qualification_sha256,
            )
        self.transcription = AsyncTranscriptionCoordinator(self.asr) if config.asr.execution_profile == "cuda" else None
        self.vad = SileroOnnxVad(
            artifact_path=artifact.path,
            size=artifact.size,
            sha256=artifact.sha256,
            threshold=config.vad.threshold,
            minimum_speech_ms=config.vad.minimum_speech_ms,
            padding_ms=config.vad.padding_ms,
        )
        self.transport = LoopbackWebSocketTransport(
            config.orchestrator_url,
            auth_key=voice_auth_key,
        )
        self.publisher = PttEventPublisher(
            capacity=config.ptt_queue_capacity,
            maximum_hold_seconds=config.maximum_hold_seconds,
            ptt_vkey=function_key_vkey(config.ptt_key),
        )
        self.approval_keys = ApprovalKeyPublisher()
        self.ptt = RawInputPttSource(self.publisher, self.approval_keys)
        self.audio = WasapiAudioCapture(queue_capacity=config.audio_queue_capacity, maximum_utterance_seconds=config.maximum_utterance_seconds, device=config.audio_input_device)
        self._running = False
        self._stop_requested = threading.Event()
        self.timings: list[dict[str, float | str]] = []
        self.reliable = ReliableSubmissionManager(time.monotonic)

    def _send_all(self, events: list) -> None:
        for event in events:
            if event.type == "input.transcript.final":
                self.reliable.begin(event)
                self.timings.append({"event": "transcript_final_created", "timestamp": time.perf_counter()})
            elif event.type == "playback.completed":
                self.timings.append({"event": "playback_completed_created", "timestamp": time.perf_counter()})
            self.transport.send(event)

    def _reconnect_pending(self) -> None:
        if not self.reliable.can_reconnect:
            raise BackendError("transport.reconnect_ttl_expired", "Transcript reconnect TTL expired", subsystem="transport")
        self.transport.close()
        self.delivery_journal.close()
        self.core.begin_new_connection()
        self.cloud_approval.begin_new_connection()
        self.action_approval.begin_new_connection()
        if self.scheduler_runtime is not None:
            self.scheduler_runtime.begin_new_connection()
        self.transport.send(self.core.client_hello())
        self.transport.send(self.reliable.reconnect_event(self.core))

    def _handle_ptt(self, event: PttEvent) -> None:
        if event.transition == PttTransition.DOWN:
            self.cloud_approval.disarm_for_ptt()
            if self.scheduler_runtime is not None:
                self.scheduler_runtime.disarm_for_ptt()
                for receipt in self.scheduler_runtime.cancel_playback(
                    cancellation_epoch=self.core.state.output_epoch + 1,
                    reason_code="ptt_down",
                ):
                    self.transport.send(receipt)
            self.timings.append({"event": "ptt_down_received", "timestamp": event.received_at})
            outbound = self.core.ptt_down()
            if self.transcription is not None:
                self.transcription.cancel_current()
            self.audio.begin_turn()
            self._send_all(outbound)
            return
        try:
            self.timings.append({"event": "ptt_up_received", "timestamp": event.received_at})
            captured = self.audio.end_turn()
            normalized = resample_audio(captured)
            if self.transcription is None:
                self._send_all([self.core.ptt_up_and_process(normalized, self.vad)])
                return
            prepared = self.core.prepare_transcription(normalized, self.vad)
            from .protocol import Envelope
            if isinstance(prepared, Envelope):
                self._send_all([prepared])
            else:
                self.transcription.submit(prepared.turn_id, prepared.audio)
        except BackendError as exc:
            self.audio.abort_turn()
            self._send_all([self.core.input_failed(exc.code)])

    def _poll_transcription(self) -> None:
        if self.transcription is None:
            return
        outcome = self.transcription.poll()
        if outcome is None:
            return
        if outcome.error is None and outcome.result is not None:
            event = self.core.complete_transcription(outcome.turn_id, outcome.result)
        elif outcome.error is not None and outcome.error.code == "asr.empty_transcript":
            event = self.core.complete_no_speech(outcome.turn_id, "asr_empty_transcript")
        else:
            event = self.core.transcription_failed(outcome.turn_id, outcome.error.code if outcome.error is not None else "asr.backend_failed")
        if event is not None:  # A cancelled turn's late result is deliberately stale.
            self._send_all([event])

    def run(self) -> None:  # pragma: no cover - native Windows hardware loop
        if self._stop_requested.is_set():
            self.close()
            return
        self._running = True
        try:
            self.audio.start()
            self.ptt.start()
            self.transport.send(self.core.client_hello())
            while self._running and not self._stop_requested.is_set():
                now = time.perf_counter()
                self.publisher.watchdog(now)
                try:
                    self._handle_ptt(self.publisher.events.get(timeout=0.01))
                except queue.Empty:
                    pass
                # A lost UP is delivered out-of-band only after the queued DOWN,
                # preserving hold-to-talk transition order under overflow.
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
                        ) or (
                            self.scheduler_runtime.key_down(
                                key_event.key,
                                key_counter=key_event.counter,
                                capture_idle=self.core.state.input_state == InputState.IDLE,
                                autorepeat=key_event.autorepeat,
                            )
                            if self.scheduler_runtime is not None else None
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
                    if self.scheduler_runtime is not None:
                        handled, scheduler_receipts = self.scheduler_runtime.handle_tts_callback(
                            callback.kind, callback.unit
                        )
                        for receipt in scheduler_receipts:
                            self.transport.send(receipt)
                        if handled:
                            continue
                    events = self.core.on_tts_unit_started(callback.unit) if callback.kind == "started" else self.core.on_tts_unit_completed(callback.unit)
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
                    if isinstance(server_event, SchedulerEnvelope):
                        if self.scheduler_runtime is None:
                            raise BackendError(
                                "scheduler.extension_not_negotiated",
                                "Scheduler event received without a qualified runtime",
                                subsystem="scheduler",
                            )
                        self.scheduler_runtime.handle_server_event(
                            server_event,
                            key_counter=self.approval_keys.counter,
                            both_keys_up=self.approval_keys.both_keys_up,
                            send=self.transport.send,
                        )
                        continue
                    if server_event.type == "response.started" and self.scheduler_runtime is not None:
                        for receipt in self.scheduler_runtime.cancel_playback(
                            cancellation_epoch=self.core.state.output_epoch,
                            reason_code="voice_response_started",
                        ):
                            self.transport.send(receipt)
                    if server_event.type == "response.speech.commit":
                        self.timings.append({"event": "speech_commit_received", "timestamp": time.perf_counter()})
                    if server_event.type == "server.ack":
                        self.reliable.acknowledge(server_event)
                    self._send_all(self.core.handle_server_event(server_event))
                if self.scheduler_runtime is not None:
                    for action in self.scheduler_runtime.pump_native_callbacks():
                        self.transport.send(action)
                    response = self.core.state.response
                    scheduler_allowed = (
                        self.core.state.input_state == InputState.IDLE
                        and (
                            response is None
                            or response.tombstoned
                            or response.playback_completed_emitted
                        )
                    )
                    for receipt in self.scheduler_runtime.pump_playback(
                        allowed=scheduler_allowed
                    ):
                        self.transport.send(receipt)
        finally:
            self.close()

    def request_stop(self) -> None:
        """Thread-safe, immediate voice cancellation requested by the tray."""

        self._stop_requested.set()
        self._running = False
        self.audio.abort_turn()
        self.tts.cancel(2**63 - 1, "application_stop")
        if self.transcription is not None:
            self.transcription.cancel_current()

    def close(self) -> None:
        self._running = False
        self._stop_requested.set()
        self.ptt.stop()
        self.audio.abort_turn()
        self.audio.stop()
        if self.transcription is not None:
            self.transcription.cancel_current()
            self.transcription.close()
        else:
            close_asr = getattr(self.asr, "close", None)
            if close_asr is not None:
                close_asr()
        self.tts.cancel(2**63 - 1, "application_close")
        self.tts.close()
        self.transport.close()
        self.cloud_journal.close()
        self.action_journal.close()
        if self.scheduler_runtime is not None:
            self.scheduler_runtime.close()
