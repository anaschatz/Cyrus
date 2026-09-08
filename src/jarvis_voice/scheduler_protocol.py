"""Frozen ``jarvis.scheduler/1`` transport contract.

Sequence numbers are scoped to (protocol, direction, websocket connection).
The base voice protocol and every negotiated extension therefore own independent
factories and inbound trackers even when they share one WebSocket.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping

from .errors import ProtocolError


SCHEDULER_PROTOCOL = "jarvis.scheduler/1"
APPROVAL_ARBITER_PROTOCOL = "jarvis.approval-arbiter/1"
SCHEDULER_SPEECH_CAPABILITY = "scheduler_notification_speech"
MAX_SCHEDULER_FRAME_BYTES = 16 * 1024

CLIENT_SCHEDULER_EVENTS = frozenset(
    {
        "scheduler.event_ack",
        "scheduler.approval.decision",
        "scheduler.notification.receipt",
        "scheduler.notification.action",
        "scheduler.playback.receipt",
    }
)
SERVER_SCHEDULER_EVENTS = frozenset(
    {
        "scheduler.ready",
        "scheduler.approval.requested",
        "scheduler.approval.decision_result",
        "scheduler.notification.requested",
        "scheduler.notification.speech",
        "scheduler.notification.tombstone",
        "scheduler.notification.action_result",
    }
)
SCHEDULER_EVENTS = CLIENT_SCHEDULER_EVENTS | SERVER_SCHEDULER_EVENTS

_FIELDS: dict[str, set[str]] = {
    "scheduler.ready": {"protocol", "maximum_frame_bytes", "sequence_namespace"},
    "scheduler.event_ack": {"acked_event_id", "status"},
    "scheduler.approval.requested": {
        "approval_id", "proposal_id", "proposal_sha256", "operation", "job_id",
        "expected_job_revision", "occurrence_id", "preview", "expires_at_utc",
        "decision_nonce", "presentation_id", "approve_key", "deny_key",
    },
    "scheduler.approval.decision": {
        "approval_id", "proposal_id", "proposal_sha256", "presentation_id",
        "requested_event_id", "decision_nonce", "decision",
    },
    "scheduler.approval.decision_result": {
        "decision_event_id", "approval_id", "proposal_id", "proposal_sha256", "result", "reason_code",
        "job_id", "job_revision", "occurrence_id", "effective_due_at_utc",
    },
    "scheduler.notification.requested": {
        "notification_id", "occurrence_id", "job_id", "job_revision",
        "delivery_generation", "scheduled_at_utc", "effective_due_at_utc",
        "expires_at_utc", "content_sha256", "title", "body", "private_title",
        "private_body", "locale", "visibility", "allow_full_body_when_locked", "silent",
    },
    "scheduler.notification.receipt": {
        "notification_id", "attempt_id", "stage", "reason_code", "observed_at_utc",
    },
    "scheduler.notification.action": {
        "action", "job_id", "job_revision", "occurrence_id", "snooze_generation",
        "requested_due_at_utc",
    },
    "scheduler.notification.action_result": {
        "action_event_id", "result", "reason_code",
    },
    "scheduler.notification.speech": {
        "occurrence_id", "notification_id", "unit_index", "locale", "speakable_text",
    },
    "scheduler.playback.receipt": {
        "notification_id", "unit_index", "state", "reason_code", "observed_at_utc",
    },
    "scheduler.notification.tombstone": {
        "notification_id", "occurrence_id", "reason_code", "created_at_utc",
    },
}


def _uuid4(value: Any, name: str) -> str:
    try:
        parsed = uuid.UUID(value) if isinstance(value, str) else None
    except ValueError as exc:
        raise ProtocolError("protocol.invalid_field", f"{name} must be UUID-v4") from exc
    if parsed is None or parsed.version != 4 or str(parsed) != value:
        raise ProtocolError("protocol.invalid_field", f"{name} must be canonical UUID-v4")
    return value


def _timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProtocolError("protocol.invalid_field", f"{name} must be UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProtocolError("protocol.invalid_field", f"{name} must be UTC RFC3339") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ProtocolError("protocol.invalid_field", f"{name} must be UTC RFC3339")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ProtocolError("protocol.invalid_field", f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class SchedulerEnvelope:
    protocol: str
    type: str
    event_id: str
    session_id: str
    seq: int
    sent_at: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.protocol != SCHEDULER_PROTOCOL or not isinstance(self.type, str) or self.type not in SCHEDULER_EVENTS:
            raise ProtocolError("protocol.unknown_event", "unknown scheduler protocol/event")
        object.__setattr__(self, "event_id", _uuid4(self.event_id, "event_id"))
        object.__setattr__(self, "session_id", _uuid4(self.session_id, "session_id"))
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 1:
            raise ProtocolError("protocol.invalid_field", "seq must be positive integer")
        object.__setattr__(self, "sent_at", _timestamp(self.sent_at, "sent_at"))
        if not isinstance(self.payload, Mapping) or set(self.payload) != _FIELDS[self.type]:
            raise ProtocolError("protocol.forbidden_field", f"{self.type} payload fields are not exact")
        object.__setattr__(self, "payload", dict(self.payload))
        try:
            self._validate_payload()
        except (TypeError, ValueError, RecursionError) as exc:
            raise ProtocolError("protocol.invalid_field", "invalid scheduler payload") from exc
        self.to_json()

    def _validate_payload(self) -> None:
        p = self.payload
        if self.type == "scheduler.ready":
            if p != {
                "protocol": SCHEDULER_PROTOCOL,
                "maximum_frame_bytes": MAX_SCHEDULER_FRAME_BYTES,
                "sequence_namespace": "protocol_direction_connection",
            }:
                raise ProtocolError("protocol.invalid_field", "invalid scheduler.ready")
        for name in ("approval_id", "proposal_id", "presentation_id", "requested_event_id", "decision_event_id",
                     "decision_nonce", "job_id", "attempt_id"):
            if name in p:
                if name == "job_id" and p[name] is None and self.type in {
                    "scheduler.approval.requested", "scheduler.approval.decision_result",
                }:
                    continue
                _uuid4(p[name], name)
        for name in ("job_revision", "expected_job_revision", "delivery_generation", "snooze_generation"):
            if name not in p:
                continue
            value = p[name]
            if value is None and self.type in {
                "scheduler.approval.requested", "scheduler.approval.decision_result",
            }:
                continue
            minimum = 0 if name.endswith("generation") else 1
            if type(value) is not int or not minimum <= value <= 2**63 - 1:
                raise ProtocolError("protocol.invalid_field", f"{name} must be a bounded integer")
        for name in ("proposal_sha256", "content_sha256", "notification_id", "occurrence_id"):
            if name in p:
                if name == "occurrence_id" and p[name] is None and self.type in {
                    "scheduler.approval.requested", "scheduler.approval.decision_result",
                }:
                    continue
                _sha256(p[name], name)
        for name in ("expires_at_utc", "scheduled_at_utc", "effective_due_at_utc",
                     "observed_at_utc", "requested_due_at_utc", "created_at_utc"):
            if name in p and p[name] is not None:
                _timestamp(p[name], name)
        if self.type == "scheduler.event_ack":
            _uuid4(p["acked_event_id"], "acked_event_id")
            if p["status"] not in {"applied", "duplicate"}:
                raise ProtocolError("protocol.invalid_field", "invalid ACK status")
        elif self.type == "scheduler.approval.requested":
            if p["approve_key"] != "F10" or p["deny_key"] != "F11":
                raise ProtocolError("protocol.invalid_field", "approval keys are frozen")
            if not isinstance(p["preview"], Mapping):
                raise ProtocolError("protocol.invalid_field", "preview must be object")
        elif self.type == "scheduler.approval.decision" and p["decision"] not in {"approve", "deny"}:
            raise ProtocolError("protocol.invalid_field", "invalid decision")
        elif self.type == "scheduler.notification.receipt" and p["stage"] not in {
            "STAGED", "DISPATCHING", "OS_ACCEPTED", "OS_REJECTED", "AMBIGUOUS_OS_DISPATCH"
        }:
            raise ProtocolError("protocol.invalid_field", "invalid delivery stage")
        elif self.type == "scheduler.notification.action":
            if (p["action"] not in {"SNOOZE", "CANCEL"}
                    or (p["requested_due_at_utc"] is None) != (p["action"] == "CANCEL")):
                raise ProtocolError("protocol.invalid_field", "invalid notification action")
        elif self.type == "scheduler.notification.action_result":
            _uuid4(p["action_event_id"], "action_event_id")
            if (
                p["result"] not in {"APPLIED", "DUPLICATE"}
                or not isinstance(p["reason_code"], str)
                or re.fullmatch(r"[a-z0-9_]{1,64}", p["reason_code"]) is None
            ):
                raise ProtocolError("protocol.invalid_field", "invalid notification action result")
        elif self.type == "scheduler.notification.speech":
            if (
                not isinstance(p["unit_index"], int)
                or isinstance(p["unit_index"], bool)
                or p["unit_index"] < 1
                or not isinstance(p["locale"], str)
                or re.fullmatch(r"[a-z]{2,3}(?:-[A-Z]{2})?", p["locale"]) is None
                or not isinstance(p["speakable_text"], str)
                or not p["speakable_text"]
                or len(p["speakable_text"].encode("utf-8")) > 4096
                or unicodedata.normalize("NFC", p["speakable_text"]) != p["speakable_text"]
            ):
                raise ProtocolError("protocol.invalid_field", "invalid scheduler speech unit")
        elif self.type == "scheduler.playback.receipt":
            if (
                not isinstance(p["unit_index"], int)
                or isinstance(p["unit_index"], bool)
                or p["unit_index"] < 1
                or p["state"] not in {"PLAYING", "COMPLETED", "CANCELLED", "SKIPPED"}
                or not isinstance(p["reason_code"], str)
                or re.fullmatch(r"[a-z0-9_]{1,64}", p["reason_code"]) is None
            ):
                raise ProtocolError("protocol.invalid_field", "invalid scheduler playback receipt")

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "type": self.type, "event_id": self.event_id,
                "session_id": self.session_id, "seq": self.seq, "sent_at": self.sent_at,
                "payload": dict(self.payload)}

    def to_json(self) -> str:
        try:
            raw = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            size = len(raw.encode("utf-8"))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ProtocolError("protocol.invalid_json", "scheduler frame must be finite UTF-8 JSON") from exc
        if size > MAX_SCHEDULER_FRAME_BYTES:
            raise ProtocolError("protocol.frame_too_large", "scheduler frame exceeds 16 KiB")
        return raw

    @property
    def semantic_sha256(self) -> str:
        value = {k: v for k, v in self.to_dict().items() if k != "seq"}
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    @classmethod
    def from_json(cls, raw: str | bytes) -> "SchedulerEnvelope":
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ProtocolError("protocol.invalid_json", "scheduler frame is not UTF-8") from exc
        try:
            size = len(raw.encode("utf-8"))
        except UnicodeError as exc:
            raise ProtocolError("protocol.invalid_json", "scheduler frame is not UTF-8") from exc
        if size > MAX_SCHEDULER_FRAME_BYTES or raw.startswith("\ufeff"):
            raise ProtocolError("protocol.frame_too_large", "invalid scheduler frame")
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ProtocolError("protocol.invalid_json", "duplicate JSON key")
                result[key] = value
            return result
        try:
            value = json.loads(raw, object_pairs_hook=pairs,
                               parse_constant=lambda value: (_ for _ in ()).throw(
                                   ProtocolError("protocol.invalid_json", f"invalid number {value}")))
        except ProtocolError:
            raise
        except (ValueError, TypeError, RecursionError) as exc:
            raise ProtocolError("protocol.invalid_json", "invalid scheduler JSON") from exc
        if not isinstance(value, Mapping) or set(value) != {
            "protocol", "type", "event_id", "session_id", "seq", "sent_at", "payload"
        }:
            raise ProtocolError("protocol.forbidden_field", "scheduler envelope fields are not exact")
        return cls(**dict(value))


class SchedulerEventFactory:
    def __init__(self, session_id: str) -> None:
        self.session_id = _uuid4(session_id, "session_id")
        self._sequence = 0

    @property
    def next_seq(self) -> int:
        return self._sequence + 1

    def consume_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def create(self, event_type: str, payload: Mapping[str, Any], *, event_id: str | None = None,
               sent_at: str | None = None) -> SchedulerEnvelope:
        event = SchedulerEnvelope(
            protocol=SCHEDULER_PROTOCOL,
            type=event_type,
            event_id=event_id or str(uuid.uuid4()),
            session_id=self.session_id,
            seq=self.next_seq,
            sent_at=sent_at or datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            payload=dict(payload),
        )
        self._sequence += 1
        return event
