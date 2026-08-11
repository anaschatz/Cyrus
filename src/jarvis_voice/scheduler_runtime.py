"""Voice composition for qualified scheduler transport and native delivery."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .approval_arbiter import GlobalApprovalController
from .backends import SpeechUnit, TtsBackend
from .scheduler_client import SchedulerClientController
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
    ) -> None:
        if ("scheduler_notification_speech" in capabilities) != (playback is not None):
            raise ValueError("scheduler notification speech runtime qualification mismatch")
        self.controller = controller
        self.delivery = delivery
        self.capabilities = capabilities
        self.manifest_sha256 = manifest_sha256
        self.playback = playback

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
        compatibility, _qualification, delivery = load_qualified_native_delivery(
            config, platform_name=platform_name
        )
        from jarvis_orchestrator.scheduler_crypto import (
            SchedulerContentCipher,
            platform_key_provider,
        )

        journal_path = default_scheduler_journal_path()
        journal = SchedulerClientJournal(
            journal_path,
            cipher=SchedulerContentCipher(platform_key_provider(
                store_role="voice", state_root=journal_path.parent
            )),
        )
        controller = SchedulerClientController(
            session_id=session_id,
            journal=journal,
            approval_arbiter=approval_arbiter,
        )
        try:
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
            return cls(
                controller=controller,
                delivery=delivery,
                capabilities=compatibility.capabilities,
                manifest_sha256=compatibility.sha256,
                playback=playback,
            )
        except BaseException:
            journal.close()
            raise

    def begin_new_connection(self) -> None:
        self.controller.begin_new_connection()
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

    def close(self) -> None:
        self.controller.journal.close()
