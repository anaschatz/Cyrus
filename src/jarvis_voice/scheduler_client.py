"""Voice-side scheduler extension controller; no native adapter is implied."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .approval_arbiter import GlobalApprovalController
from .errors import ProtocolError
from .scheduler_journal import SchedulerClientJournal
from .scheduler_protocol import (
    CLIENT_SCHEDULER_EVENTS,
    SERVER_SCHEDULER_EVENTS,
    SCHEDULER_PROTOCOL,
    SchedulerEnvelope,
    SchedulerEventFactory,
)


@dataclass(frozen=True, slots=True)
class SchedulerApproval:
    approval_id: str
    proposal_id: str
    proposal_sha256: str
    presentation_id: str
    requested_event_id: str
    decision_nonce: str


def select_notification_presentation(
    payload: dict[str, Any], *, lock_state: str, qualified_full_body_opt_in: bool = False
) -> dict[str, Any]:
    """Apply the frozen privacy downgrade before any native adapter call."""

    if lock_state not in {"locked", "unlocked", "unknown"}:
        raise ValueError("invalid lock state")
    visibility = payload.get("visibility")
    if visibility not in {"private", "title_only", "full"}:
        raise ValueError("invalid notification visibility")
    if lock_state == "unknown" or visibility == "private":
        effective = "private"
    elif lock_state == "locked" and not (
        visibility == "full"
        and qualified_full_body_opt_in
        and payload.get("allow_full_body_when_locked") is True
    ):
        effective = "title_only"
    else:
        effective = visibility
    if effective == "private":
        title, body = payload["private_title"], payload["private_body"]
    elif effective == "title_only":
        title, body = payload["title"], payload["private_body"]
    else:
        title, body = payload["title"], payload["body"]
    return {
        "body": body,
        "effective_visibility": effective,
        "title": title,
        "tts_allowed": lock_state == "unlocked",
    }


class SchedulerClientController:
    def __init__(self, *, session_id: str, journal: SchedulerClientJournal,
                 approval_arbiter: GlobalApprovalController) -> None:
        self.factory = SchedulerEventFactory(session_id)
        self.journal = journal
        self.approval_arbiter = approval_arbiter
        self.approval: SchedulerApproval | None = None
        self._next_server_seq = 1
        self._seen: dict[int, str] = {}

    def begin_new_connection(self) -> None:
        self.factory = SchedulerEventFactory(self.factory.session_id)
        self._next_server_seq = 1
        self._seen.clear()

    def startup_recovery_frames(self) -> list[SchedulerEnvelope]:
        return [self.delivery_receipt(
            notification_id=row["notification_id"],
            attempt_id=row["attempt_id"],
            stage="AMBIGUOUS_OS_DISPATCH",
            reason_code="restart_without_native_receipt",
        ) for row in self.journal.recover_ambiguous_dispatches()]

    def handle_server_event(self, event: SchedulerEnvelope, *, key_counter: int,
                            both_keys_up: bool) -> list[SchedulerEnvelope]:
        if event.type not in SERVER_SCHEDULER_EVENTS:
            raise ProtocolError("protocol.wrong_direction", "scheduler event has wrong direction")
        fingerprint = event.to_json()
        if event.seq == self._next_server_seq:
            self._seen[event.seq] = fingerprint
            self._next_server_seq += 1
        elif event.seq < self._next_server_seq and self._seen.get(event.seq) == fingerprint:
            return []
        else:
            raise ProtocolError("protocol.sequence_gap", "scheduler sequence is invalid")
        if event.type == "scheduler.ready":
            return []
        status = self.journal.stage(
            event_id=event.event_id,
            semantic_sha256=event.semantic_sha256,
            event_type=event.type,
            payload=event.payload,
        )
        outbound = [self.factory.create(
            "scheduler.event_ack", {"acked_event_id": event.event_id, "status": status}
        )]
        if event.type == "scheduler.approval.requested" and status == "applied":
            self.approval_arbiter.stage(
                protocol=SCHEDULER_PROTOCOL,
                approval_id=event.payload["approval_id"],
                presentation_id=event.payload["presentation_id"],
                requested_event_id=event.event_id,
                key_counter=key_counter,
                both_keys_up=both_keys_up,
            )
            self.approval = SchedulerApproval(
                event.payload["approval_id"], event.payload["proposal_id"],
                event.payload["proposal_sha256"], event.payload["presentation_id"],
                event.event_id, event.payload["decision_nonce"],
            )
        elif event.type == "scheduler.approval.decision_result":
            self.approval_arbiter.close(
                protocol=SCHEDULER_PROTOCOL, approval_id=event.payload["approval_id"]
            )
            self.approval = None
        elif event.type == "scheduler.notification.tombstone":
            self.journal.tombstone(event.payload["notification_id"], event.payload["reason_code"])
        elif event.type == "scheduler.notification.speech":
            self.journal.stage_playback(
                notification_id=event.payload["notification_id"],
                unit_index=event.payload["unit_index"],
                frame_event_id=event.event_id,
            )
        return outbound

    def disarm_for_ptt(self) -> None:
        self.approval_arbiter.disarm_for_capture()

    def rearm_after_capture(self, *, key_counter: int, both_keys_up: bool) -> bool:
        return self.approval_arbiter.capture_ended(
            key_counter=key_counter, both_keys_up=both_keys_up
        )

    def key_down(self, key: str, *, key_counter: int, capture_idle: bool,
                 autorepeat: bool) -> SchedulerEnvelope | None:
        approval = self.approval
        if (approval is None or key not in {"F10", "F11"}
                or not self.approval_arbiter.begin_decision(
                    protocol=SCHEDULER_PROTOCOL, key_counter=key_counter,
                    capture_idle=capture_idle, autorepeat=autorepeat)):
            return None
        self.approval = None
        return self.factory.create("scheduler.approval.decision", {
            "approval_id": approval.approval_id,
            "proposal_id": approval.proposal_id,
            "proposal_sha256": approval.proposal_sha256,
            "presentation_id": approval.presentation_id,
            "requested_event_id": approval.requested_event_id,
            "decision_nonce": approval.decision_nonce,
            "decision": "approve" if key == "F10" else "deny",
        })

    def delivery_receipt(self, *, notification_id: str, attempt_id: str | None,
                         stage: str, reason_code: str) -> SchedulerEnvelope:
        if stage not in {"STAGED", "DISPATCHING", "OS_ACCEPTED", "OS_REJECTED", "AMBIGUOUS_OS_DISPATCH"}:
            raise ValueError("invalid delivery stage")
        return self.factory.create("scheduler.notification.receipt", {
            "notification_id": notification_id,
            "attempt_id": attempt_id or str(uuid.uuid4()),
            "stage": stage,
            "reason_code": reason_code,
            "observed_at_utc": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        })

    def begin_native_dispatch(self, notification_id: str) -> tuple[str, SchedulerEnvelope]:
        attempt_id = self.journal.begin_native_dispatch(notification_id)
        return attempt_id, self.delivery_receipt(
            notification_id=notification_id, attempt_id=attempt_id,
            stage="DISPATCHING", reason_code="native_call_starting",
        )

    def finish_native_dispatch(self, *, notification_id: str, attempt_id: str,
                               accepted: bool, reason_code: str) -> SchedulerEnvelope:
        stage = "OS_ACCEPTED" if accepted else "OS_REJECTED"
        if not self.journal.finish_native_dispatch(
            attempt_id, state=stage, reason_code=reason_code
        ):
            raise ValueError("native attempt is not dispatching")
        return self.delivery_receipt(
            notification_id=notification_id, attempt_id=attempt_id,
            stage=stage, reason_code=reason_code,
        )
