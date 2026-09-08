"""Voice composition for qualified scheduler transport and native delivery."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .approval_arbiter import GlobalApprovalController
from .backends import SpeechUnit, TtsBackend
from .private_state import default_voice_state_boundary, prepare_voice_journal_path
from .scheduler_client import SchedulerClientController
from .scheduler_callback import NativeCallbackRejected, NativeSchedulerCallback
from .scheduler_journal import SchedulerClientJournal, default_scheduler_journal_path
from .scheduler_native import QualifiedNativeSchedulerDelivery, load_qualified_native_delivery
from .scheduler_playback import SchedulerPlaybackController
from .scheduler_protocol import SchedulerEnvelope
from .scheduler_tts_qualification import load_scheduler_tts_qualification


class SchedulerNativeRuntime:
    def __init__(
        self,
        *,
        controller: SchedulerClientController,
        delivery: QualifiedNativeSchedulerDelivery,
        capabilities: tuple[str, ...],
        manifest_sha256: str,
        playback: SchedulerPlaybackController | None = None,
        callback_qualification_sha256: str | None = None,
    ) -> None:
        if ("scheduler_notification_speech" in capabilities) != (playback is not None):
            raise ValueError("scheduler notification speech runtime qualification mismatch")
        self.controller = controller
        self.delivery = delivery
        self.capabilities = capabilities
        self.manifest_sha256 = manifest_sha256
        self.playback = playback
        self.callback_qualification_sha256 = callback_qualification_sha256
        self.controller.journal.reset_emitted_native_callbacks()

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        session_id: str,
        approval_arbiter: GlobalApprovalController,
        tts: TtsBackend | None = None,
        reserve_epoch: Callable[[], int] | None = None,
        runtime_profile_sha256: str | None = None,
        platform_name: str | None = None,
    ) -> "SchedulerNativeRuntime":
        compatibility, _qualification, delivery, callback_qualification = load_qualified_native_delivery(
            config, platform_name=platform_name
        )
        with ExitStack() as initialization:
            adapter = getattr(delivery, "adapter", delivery)
            close_adapter = getattr(adapter, "close", None)
            if close_adapter is not None:
                initialization.callback(close_adapter)
            from jarvis_orchestrator.scheduler_crypto import (
                SchedulerContentCipher,
                platform_key_provider,
            )

            state_boundary = default_voice_state_boundary()
            journal_path = prepare_voice_journal_path(
                default_scheduler_journal_path(), trusted_boundary=state_boundary
            )
            journal = SchedulerClientJournal(
                journal_path,
                cipher=SchedulerContentCipher(platform_key_provider(
                    store_role="voice", state_root=journal_path.parent
                )),
                trusted_boundary=state_boundary,
            )
            initialization.callback(journal.close)
            controller = SchedulerClientController(
                session_id=session_id,
                journal=journal,
                approval_arbiter=approval_arbiter,
            )
            playback: SchedulerPlaybackController | None = None
            if "scheduler_notification_speech" in compatibility.capabilities:
                if (
                    tts is None
                    or reserve_epoch is None
                    or config.tts_qualification_path is None
                    or config.tts_qualification_sha256 is None
                ):
                    raise ValueError("scheduler TTS qualification is required")
                tts_qualification = load_scheduler_tts_qualification(
                    Path(config.tts_qualification_path),
                    config.tts_qualification_sha256,
                    platform_name=platform_name or sys.platform,
                    compatibility=compatibility,
                )
                if tts_qualification.runtime_profile_sha256 != runtime_profile_sha256:
                    raise ValueError("scheduler TTS runtime profile mismatch")
                playback = SchedulerPlaybackController(
                    journal=journal,
                    tts=tts,
                    factory=controller.factory,
                    reserve_epoch=reserve_epoch,
                )
            runtime = cls(
                controller=controller,
                delivery=delivery,
                capabilities=compatibility.capabilities,
                manifest_sha256=compatibility.sha256,
                playback=playback,
                callback_qualification_sha256=(
                    callback_qualification.sha256 if callback_qualification is not None else None
                ),
            )
            initialization.pop_all()
            return runtime

    def begin_new_connection(self) -> None:
        self.controller.begin_new_connection()
        self.controller.journal.reset_emitted_native_callbacks()
        if self.playback is not None:
            self.playback.factory = self.controller.factory

    def disarm_for_ptt(self) -> None:
        self.controller.disarm_for_ptt()

    def key_down(self, key: str, *, key_counter: int, capture_idle: bool,
                 autorepeat: bool) -> SchedulerEnvelope | None:
        return self.controller.key_down(
            key,
            key_counter=key_counter,
            capture_idle=capture_idle,
            autorepeat=autorepeat,
        )

    def _dispatch(
        self, payload: dict[str, Any], send: Callable[[SchedulerEnvelope], None]
    ) -> None:
        notification_id = payload["notification_id"]
        if datetime.fromisoformat(payload["expires_at_utc"]) <= datetime.now(UTC):
            return
        if not self.controller.native_dispatch_needed(notification_id):
            return
        attempt_id, dispatching = self.controller.begin_native_dispatch(notification_id)
        # The FULL-sync local journal fence exists before this receipt and the OS call.
        send(dispatching)
        result = self.delivery.post(payload)
        send(self.controller.finish_native_dispatch(
            notification_id=notification_id,
            attempt_id=attempt_id,
            accepted=result.accepted,
            ambiguous=result.ambiguous,
            reason_code=result.reason_code,
        ))

    def handle_server_event(
        self,
        event: SchedulerEnvelope,
        *,
        key_counter: int,
        both_keys_up: bool,
        send: Callable[[SchedulerEnvelope], None],
    ) -> None:
        replies = self.controller.handle_server_event(
            event, key_counter=key_counter, both_keys_up=both_keys_up
        )
        for reply in replies:
            send(reply)
        applied = any(
            reply.type == "scheduler.event_ack" and reply.payload["status"] == "applied"
            for reply in replies
        )
        if event.type == "scheduler.ready":
            for recovery in self.controller.startup_recovery_frames():
                send(recovery)
            for payload in self.controller.journal.pending_notification_payloads():
                self._dispatch(payload, send)
            if self.playback is not None:
                for recovery in self.playback.startup_recovery():
                    send(recovery)
        elif event.type == "scheduler.notification.requested":
            self._dispatch(dict(event.payload), send)
        elif event.type == "scheduler.notification.tombstone":
            self.delivery.remove(event.payload["notification_id"])
            if applied and self.playback is not None:
                for receipt in self.playback.cancel_tombstoned(
                    event.payload["notification_id"],
                    cancellation_epoch=self.playback.active.unit.epoch if self.playback.active else 0,
                ):
                    send(receipt)
        elif event.type == "scheduler.notification.action_result":
            self.controller.journal.close_native_callback(event.payload["action_event_id"])

    def pump_playback(self, *, allowed: bool) -> list[SchedulerEnvelope]:
        if self.playback is None:
            return []
        return self.playback.pump(allowed=allowed)

    def cancel_playback(self, *, cancellation_epoch: int, reason_code: str) -> list[SchedulerEnvelope]:
        if self.playback is None:
            return []
        return self.playback.cancel_active(
            cancellation_epoch=cancellation_epoch,
            reason_code=reason_code,
        )

    def handle_tts_callback(
        self, kind: str, unit: SpeechUnit
    ) -> tuple[bool, list[SchedulerEnvelope]]:
        if self.playback is None:
            return False, []
        return (
            self.playback.on_started(unit)
            if kind == "started"
            else self.playback.on_completed(unit)
        )

    def ingest_native_callback(self, raw: bytes | str, *, key: bytes) -> str:
        callback = NativeSchedulerCallback.verify(raw, key=key)
        if self.controller.journal.notification_payload(callback.notification_id) is None:
            raise NativeCallbackRejected("callback.notification_not_active")
        payload = callback.payload()
        import hashlib
        import json
        semantic = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        return self.controller.journal.stage_native_callback(
            callback_id=callback.callback_id,
            semantic_sha256=semantic,
            payload=payload,
        )

    def pump_native_callbacks(self) -> list[SchedulerEnvelope]:
        adapter = getattr(self.delivery, "adapter", self.delivery)
        drain = getattr(adapter, "drain_callbacks", None)
        callback_key = getattr(adapter, "callback_key", None)
        if drain is not None and isinstance(callback_key, bytes):
            for raw in drain():
                try:
                    self.ingest_native_callback(raw, key=callback_key)
                except NativeCallbackRejected:
                    continue
        outbound: list[SchedulerEnvelope] = []
        for callback in self.controller.journal.pending_native_callbacks():
            notification = self.controller.journal.notification_payload(callback["notification_id"])
            if notification is None:
                continue
            event_id = self.controller.journal.native_callback_event_id(callback["callback_id"])
            event = self.controller.factory.create("scheduler.notification.action", {
                "action": callback["action"],
                "job_id": notification["job_id"],
                "job_revision": notification["job_revision"],
                "occurrence_id": notification["occurrence_id"],
                "snooze_generation": notification["delivery_generation"],
                "requested_due_at_utc": callback["requested_due_at_utc"],
            }, event_id=event_id)
            if self.controller.journal.mark_native_callback_emitted(
                callback["callback_id"], event.event_id
            ):
                outbound.append(event)
        return outbound

    def close(self) -> None:
        adapter = getattr(self.delivery, "adapter", self.delivery)
        close_adapter = getattr(adapter, "close", None)
        try:
            if close_adapter is not None:
                close_adapter()
        finally:
            self.controller.journal.close()
