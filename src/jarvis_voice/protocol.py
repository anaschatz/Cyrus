"""Wire models and validation for ``jarvis.voice/1``."""

from __future__ import annotations

import json
import hashlib
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .errors import ProtocolError

PROTOCOL_VERSION = 1

CLIENT_EVENT_TYPES = frozenset(
    {
        "client.hello",
        "input.capture.started",
        "input.transcript.final",
        "input.no_speech",
        "response.cancel",
        "playback.started",
        "playback.completed",
        "playback.interrupted",
        "client.error",
        "client.goodbye",
    }
)
SERVER_EVENT_TYPES = frozenset(
    {
        "server.ready",
        "server.ack",
        "response.started",
        "response.text.delta",
        "response.text.done",
        "response.speech.commit",
        "response.speech.done",
        "response.cancelled",
        "server.error",
    }
)
EVENT_TYPES = CLIENT_EVENT_TYPES | SERVER_EVENT_TYPES


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def new_uuid4() -> str:
    return str(uuid.uuid4())


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _require_uuid4(value: Any, field_name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise ProtocolError(
            "protocol.invalid_field",
            f"{field_name} must be a UUID string",
            safe_context={"field": field_name},
        )
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError(
            "protocol.invalid_field",
            f"{field_name} must be a valid UUID",
            safe_context={"field": field_name},
        ) from exc
    if parsed.version != 4 or str(parsed) != value.lower():
        raise ProtocolError(
            "protocol.invalid_field",
            f"{field_name} must be a canonical UUID-v4",
            safe_context={"field": field_name},
        )
    return value.lower()


def _require_timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise ProtocolError(
            "protocol.invalid_field",
            "sent_at must be an ISO-8601 UTC timestamp",
            safe_context={"field": "sent_at"},
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolError(
            "protocol.invalid_field",
            "sent_at must be an ISO-8601 UTC timestamp",
            safe_context={"field": "sent_at"},
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ProtocolError(
            "protocol.invalid_field",
            "sent_at must use UTC",
            safe_context={"field": "sent_at"},
        )
    return value


def _require_response_id(response_id: str | None, event_type: str) -> None:
    if response_id is None:
        raise ProtocolError(
            "protocol.missing_field",
            f"{event_type} requires response_id",
            safe_context={"field": "response_id", "event_type": event_type},
        )


def _require_payload_fields(payload: Mapping[str, Any], event_type: str, *names: str) -> None:
    for name in names:
        if name not in payload:
            raise ProtocolError(
                "protocol.missing_field",
                f"{event_type} payload requires {name}",
                safe_context={"field": name, "event_type": event_type},
            )


def _require_exact_payload(payload: Mapping[str, Any], event_type: str, *names: str) -> None:
    _require_payload_fields(payload, event_type, *names)
    if set(payload) != set(names):
        raise ProtocolError(
            "protocol.forbidden_field",
            f"{event_type} payload has unknown fields",
            safe_context={"event_type": event_type},
        )


@dataclass(frozen=True, slots=True)
class Envelope:
    """Validated protocol envelope.

    Unknown top-level fields are deliberately discarded by :meth:`from_dict`.
    """

    type: str
    event_id: str
    session_id: str
    seq: int
    sent_at: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    turn_id: str | None = None
    response_id: str | None = None
    v: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if type(self.v) is not int or self.v != PROTOCOL_VERSION:
            raise ProtocolError(
                "protocol.version_mismatch",
                "Unsupported protocol version",
                safe_context={"field": "v"},
            )
        if not isinstance(self.type, str) or self.type not in EVENT_TYPES:
            raise ProtocolError(
                "protocol.unknown_event",
                "Unknown protocol event type",
                safe_context={"field": "type"},
            )
        object.__setattr__(self, "event_id", _require_uuid4(self.event_id, "event_id"))
        object.__setattr__(
            self, "session_id", _require_uuid4(self.session_id, "session_id")
        )
        object.__setattr__(
            self, "turn_id", _require_uuid4(self.turn_id, "turn_id", nullable=True)
        )
        object.__setattr__(
            self,
            "response_id",
            _require_uuid4(self.response_id, "response_id", nullable=True),
        )
        if not isinstance(self.seq, int) or isinstance(self.seq, bool) or self.seq < 1:
            raise ProtocolError(
                "protocol.invalid_field",
                "seq must be a positive integer",
                safe_context={"field": "seq"},
            )
        object.__setattr__(self, "sent_at", _require_timestamp(self.sent_at))
        if not isinstance(self.payload, Mapping):
            raise ProtocolError(
                "protocol.invalid_field",
                "payload must be an object",
                safe_context={"field": "payload"},
            )
        object.__setattr__(self, "payload", dict(self.payload))
        self._validate_event_payload()
        try:
            json.dumps(self.payload, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ProtocolError("protocol.invalid_json", "Payload must contain finite UTF-8 JSON") from exc

    def _validate_event_payload(self) -> None:
        payload = self.payload
        if self.type.startswith("response.") or self.type.startswith("playback."):
            _require_response_id(self.response_id, self.type)
        if self.type == "server.ack":
            _require_exact_payload(payload, self.type, "acked_event_id", "status")
            acked_id = _require_uuid4(payload["acked_event_id"], "payload.acked_event_id")
            object.__setattr__(self, "payload", {**payload, "acked_event_id": acked_id})
            if not isinstance(payload["status"], str) or payload["status"] not in {"accepted", "duplicate"}:
                raise ProtocolError(
                    "protocol.invalid_field",
                    "server.ack status must be accepted or duplicate",
                    safe_context={"field": "payload.status"},
                )
        elif self.type == "server.ready":
            _require_exact_payload(payload, self.type, "protocol")
            if payload["protocol"] != "jarvis.voice/1":
                raise ProtocolError("protocol.invalid_field", "server.ready protocol is invalid")
        elif self.type == "response.started":
            _require_exact_payload(payload, self.type, "locale")
            if not isinstance(payload["locale"], str) or not payload["locale"].strip() or nfc(payload["locale"]) != payload["locale"]:
                raise ProtocolError("protocol.invalid_field", "response locale is invalid")
        elif self.type == "response.text.delta":
            _require_exact_payload(payload, self.type, "chunk_index", "text")
            index = payload["chunk_index"]
            text = payload["text"]
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ProtocolError("protocol.invalid_field", "text chunk_index must be non-negative")
            if not isinstance(text, str) or not text or nfc(text) != text:
                raise ProtocolError("protocol.text_not_nfc", "Text delta must be non-empty NFC text")
        elif self.type == "response.text.done":
            _require_exact_payload(payload, self.type, "chunk_count")
            count = payload["chunk_count"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ProtocolError("protocol.invalid_field", "chunk_count must be non-negative")
        elif self.type == "response.speech.commit":
            _require_exact_payload(payload, self.type, "unit_index", "locale", "text")
            index = payload["unit_index"]
            if not isinstance(index, int) or isinstance(index, bool) or index < 1:
                raise ProtocolError(
                    "protocol.invalid_field",
                    "speech unit_index must start at 1",
                    safe_context={"field": "payload.unit_index"},
                )
            locale = payload["locale"]
            text = payload["text"]
            if not isinstance(locale, str) or not locale.strip():
                raise ProtocolError(
                    "protocol.invalid_field",
                    "speech locale must be a non-empty string",
                    safe_context={"field": "payload.locale"},
                )
            if not isinstance(text, str) or not text:
                raise ProtocolError(
                    "protocol.invalid_field",
                    "speech text must be a non-empty string",
                    safe_context={"field": "payload.text"},
                )
            if nfc(locale) != locale or nfc(text) != text:
                raise ProtocolError(
                    "protocol.text_not_nfc",
                    "Speech locale and text must be NFC-normalized",
                    safe_context={"event_type": self.type},
                )
        elif self.type == "response.speech.done":
            _require_exact_payload(payload, self.type, "last_unit_index")
            index = payload["last_unit_index"]
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ProtocolError(
                    "protocol.invalid_field",
                    "last_unit_index must be a non-negative integer",
                    safe_context={"field": "payload.last_unit_index"},
                )
        elif self.type == "response.cancelled":
            _require_exact_payload(payload, self.type, "reason_code", "partial_output")
            if not isinstance(payload["reason_code"], str) or payload["reason_code"] not in {
                "user_cancelled", "superseded", "client_disconnected", "approval_expired",
                "approval_reconnect_timeout", "service_interrupted", "active_budget_exhausted",
                "absolute_lifetime_exhausted", "backend_failure", "internal_failure",
                "continuation_incompatible", "conversation_deleted",
            }:
                raise ProtocolError("protocol.invalid_field", "response cancellation reason is invalid")
            if payload["partial_output"] is not False:
                raise ProtocolError("protocol.invalid_field", "partial_output must be false")
        elif self.type == "server.error":
            _require_exact_payload(payload, self.type, "code", "retryable", "terminal")
            if not isinstance(payload["code"], str) or not payload["code"]:
                raise ProtocolError("protocol.invalid_field", "server error code is invalid")
            if payload["retryable"] is not False or payload["terminal"] is not True:
                raise ProtocolError("protocol.invalid_field", "server error flags are invalid")
        elif self.type == "input.transcript.final":
            _require_payload_fields(payload, self.type, "text", "language", "asr_score")
            if "confidence" in payload:
                raise ProtocolError(
                    "protocol.forbidden_field",
                    "Use asr_score instead of confidence",
                    safe_context={"field": "payload.confidence"},
                )
            if not isinstance(payload["text"], str):
                raise ProtocolError(
                    "protocol.invalid_field",
                    "Transcript text must be a string",
                    safe_context={"field": "payload.text"},
                )
            score = payload["asr_score"]
            if not isinstance(score, Mapping):
                raise ProtocolError(
                    "protocol.invalid_field",
                    "asr_score must be an object",
                    safe_context={"field": "payload.asr_score"},
                )
            _require_payload_fields(score, self.type, "value", "method", "calibrated")
            language = payload["language"]
            if not isinstance(language, Mapping):
                raise ProtocolError(
                    "protocol.invalid_field",
                    "language must be an object",
                    safe_context={"field": "payload.language"},
                )
            _require_payload_fields(language, self.type, "code", "score")
            if not isinstance(language["code"], str) or not language["code"]:
                raise ProtocolError(
                    "protocol.invalid_field",
                    "language code must be a non-empty string",
                    safe_context={"field": "payload.language.code"},
                )
            if not isinstance(language["score"], (int, float)) or isinstance(
                language["score"], bool
            ):
                raise ProtocolError(
                    "protocol.invalid_field",
                    "language score must be numeric",
                    safe_context={"field": "payload.language.score"},
                )
            if not isinstance(score["value"], (int, float)) or isinstance(
                score["value"], bool
            ):
                raise ProtocolError(
                    "protocol.invalid_field",
                    "asr_score value must be numeric",
                    safe_context={"field": "payload.asr_score.value"},
                )
            if not isinstance(score["method"], str) or not score["method"]:
                raise ProtocolError(
                    "protocol.invalid_field",
                    "asr_score method must be a non-empty string",
                    safe_context={"field": "payload.asr_score.method"},
                )
            if score["calibrated"] is not False:
                raise ProtocolError(
                    "protocol.invalid_field",
                    "asr_score must explicitly be uncalibrated",
                    safe_context={"field": "payload.asr_score.calibrated"},
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Envelope":
        if not isinstance(data, Mapping):
            raise ProtocolError("protocol.invalid_envelope", "Envelope must be an object")
        required = ("v", "type", "event_id", "session_id", "seq", "sent_at", "payload")
        for name in required:
            if name not in data:
                raise ProtocolError(
                    "protocol.missing_field",
                    f"Envelope requires {name}",
                    safe_context={"field": name},
                )
        return cls(
            v=data["v"],
            type=data["type"],
            event_id=data["event_id"],
            session_id=data["session_id"],
            seq=data["seq"],
            sent_at=data["sent_at"],
            turn_id=data.get("turn_id"),
            response_id=data.get("response_id"),
            payload=data["payload"],
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Envelope":
        try:
            data = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise ProtocolError(
                "protocol.invalid_json", "Frame must contain valid UTF-8 JSON"
            ) from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "type": self.type,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "seq": self.seq,
            "sent_at": self.sent_at,
            "turn_id": self.turn_id,
            "response_id": self.response_id,
            "payload": dict(self.payload),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


class EventFactory:
    """Creates correctly sequenced envelopes for one sender/connection."""

    def __init__(self, session_id: str | None = None, *, initial_seq: int = 1) -> None:
        self.session_id = _require_uuid4(session_id or new_uuid4(), "session_id")
        if initial_seq < 1:
            raise ValueError("initial_seq must be positive")
        self._next_seq = initial_seq

    @property
    def next_seq(self) -> int:
        return self._next_seq

    def consume_sequence(self) -> int:
        """Advance after an externally reconstructed envelope uses ``next_seq``."""

        current = self._next_seq
        self._next_seq += 1
        return current

    def create(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        event_id: str | None = None,
        turn_id: str | None = None,
        response_id: str | None = None,
        sent_at: str | None = None,
    ) -> Envelope:
        event = Envelope(
            type=event_type,
            event_id=event_id or new_uuid4(),
            session_id=self.session_id,
            seq=self._next_seq,
            sent_at=sent_at or utc_now_text(),
            turn_id=turn_id,
            response_id=response_id,
            payload=payload or {},
        )
        self._next_seq += 1
        return event
