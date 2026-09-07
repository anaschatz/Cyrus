"""Sensitive-content-free ``jarvis.action-approval/1`` voice extension."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import ProtocolError
from .private_state import prepare_voice_journal_path
from .approval_arbiter import GlobalApprovalController


ACTION_APPROVAL_PROTOCOL = "jarvis.action-approval/1"
MAX_ACTION_APPROVAL_FRAME_BYTES = 16 * 1024
CLIENT_EVENTS = frozenset({"action.approval.event_ack", "action.approval.decision"})
SERVER_EVENTS = frozenset(
    {
        "action.approval.ready",
        "action.approval.requested",
        "action.approval.decision_ack",
        "action.approval.decision_result",
        "action.approval.invalidated",
    }
)
EVENTS = CLIENT_EVENTS | SERVER_EVENTS
DECISION_REASONS = frozenset(
    {
        "approved",
        "denied",
        "approval_expired",
        "task_cancelled",
        "stale_epoch",
        "binding_mismatch",
        "presentation_superseded",
        "not_owner",
        "invalid_state",
        "capture_active",
        "approval_busy",
    }
)
INVALIDATION_REASONS = frozenset(
    {
        "approval_expired",
        "task_cancelled",
        "superseded",
        "binding_changed",
        "readiness_changed",
        "policy_changed",
        "session_superseded",
    }
)


def default_action_approval_journal_path() -> Path:
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA", "").strip()
        if not root:
            raise ProtocolError("action_journal.state_root_missing", "LOCALAPPDATA is required")
        return Path(root) / "JARVIS Voice" / "state" / "action-approval.sqlite3"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "JARVIS Voice" / "state" / "action-approval.sqlite3"
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "jarvis-voice" / "action-approval.sqlite3"


def _uuid4(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ProtocolError("protocol.invalid_field", f"{name} must be UUID-v4")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ProtocolError("protocol.invalid_field", f"{name} must be UUID-v4") from exc
    if parsed.version != 4 or str(parsed) != value.lower():
        raise ProtocolError("protocol.invalid_field", f"{name} must be canonical UUID-v4")
    return value.lower()


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError("protocol.invalid_field", f"{name} must be lowercase SHA-256")
    return value


def _timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise ProtocolError("protocol.invalid_field", "timestamp must be UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolError("protocol.invalid_field", "timestamp must be UTC RFC3339") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ProtocolError("protocol.invalid_field", "timestamp must be UTC RFC3339")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], event_type: str) -> None:
    if set(value) != fields:
        raise ProtocolError("protocol.forbidden_field", f"{event_type} payload fields are not exact")


REQUESTED_FIELDS = {
    "approval_id",
    "presentation_id",
    "task_id",
    "tool_call_id",
    "epoch",
    "tool_id",
    "tool_version",
    "approval_binding_sha256",
    "preview_sha256",
    "arguments_sha256",
    "user_intent_binding_sha256",
    "registry_sha256",
    "tool_schema_sha256",
    "policy_sha256",
    "expires_at",
    "decision_nonce",
    "preview_fetch_nonce",
    "approve_key",
    "deny_key",
}


@dataclass(frozen=True, slots=True)
class ActionApprovalEnvelope:
    protocol: str
    type: str
    event_id: str
    session_id: str
    seq: int
    sent_at: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.protocol != ACTION_APPROVAL_PROTOCOL or self.type not in EVENTS:
            raise ProtocolError("protocol.unknown_event", "unknown action approval event")
        object.__setattr__(self, "event_id", _uuid4(self.event_id, "event_id"))
        object.__setattr__(self, "session_id", _uuid4(self.session_id, "session_id"))
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 1:
            raise ProtocolError("protocol.invalid_field", "seq must be positive integer")
        object.__setattr__(self, "sent_at", _timestamp(self.sent_at))
        if not isinstance(self.payload, Mapping):
            raise ProtocolError("protocol.invalid_field", "payload must be object")
        object.__setattr__(self, "payload", dict(self.payload))
        self._validate_payload()

    def _validate_payload(self) -> None:
        payload = self.payload
        if self.type == "action.approval.ready":
            _exact(payload, {"protocol"}, self.type)
            if payload["protocol"] != ACTION_APPROVAL_PROTOCOL:
                raise ProtocolError("protocol.invalid_field", "invalid action approval protocol")
        elif self.type == "action.approval.requested":
            _exact(payload, REQUESTED_FIELDS, self.type)
            for name in ("approval_id", "presentation_id", "task_id", "tool_call_id", "decision_nonce", "preview_fetch_nonce"):
                _uuid4(payload[name], name)
            for name in ("approval_binding_sha256", "preview_sha256", "arguments_sha256", "user_intent_binding_sha256", "registry_sha256", "tool_schema_sha256", "policy_sha256"):
                _sha(payload[name], name)
            for name in ("epoch", "tool_version"):
                if isinstance(payload[name], bool) or not isinstance(payload[name], int) or payload[name] < (1 if name == "tool_version" else 0):
                    raise ProtocolError("protocol.invalid_field", f"{name} is invalid")
            if not isinstance(payload["tool_id"], str) or not payload["tool_id"].isascii() or not payload["tool_id"]:
                raise ProtocolError("protocol.invalid_field", "tool_id is invalid")
            _timestamp(payload["expires_at"])
            if payload["approve_key"] != "F10" or payload["deny_key"] != "F11":
                raise ProtocolError("protocol.invalid_field", "approval keys are frozen")
        elif self.type == "action.approval.event_ack":
            _exact(payload, {"acked_event_id", "status"}, self.type)
            _uuid4(payload["acked_event_id"], "acked_event_id")
            if payload["status"] not in {"applied", "duplicate"}:
                raise ProtocolError("protocol.invalid_field", "event ACK status is invalid")
        elif self.type == "action.approval.decision":
            _exact(payload, {"approval_id", "presentation_id", "requested_event_id", "approval_binding_sha256", "decision_nonce", "decision"}, self.type)
            for name in ("approval_id", "presentation_id", "requested_event_id", "decision_nonce"):
                _uuid4(payload[name], name)
            _sha(payload["approval_binding_sha256"], "approval_binding_sha256")
            if payload["decision"] not in {"approve", "deny"}:
                raise ProtocolError("protocol.invalid_field", "decision is invalid")
        elif self.type == "action.approval.decision_ack":
            _exact(payload, {"decision_event_id", "status"}, self.type)
            _uuid4(payload["decision_event_id"], "decision_event_id")
            if payload["status"] not in {"accepted", "duplicate"}:
                raise ProtocolError("protocol.invalid_field", "decision ACK is invalid")
        elif self.type == "action.approval.decision_result":
            _exact(payload, {"decision_event_id", "approval_id", "presentation_id", "approval_binding_sha256", "result", "reason_code"}, self.type)
            for name in ("decision_event_id", "approval_id", "presentation_id"):
                _uuid4(payload[name], name)
            _sha(payload["approval_binding_sha256"], "approval_binding_sha256")
            if payload["result"] not in {"applied", "rejected"} or payload["reason_code"] not in DECISION_REASONS:
                raise ProtocolError("protocol.invalid_field", "decision result is invalid")
        elif self.type == "action.approval.invalidated":
            _exact(payload, {"approval_id", "presentation_id", "approval_binding_sha256", "reason_code"}, self.type)
            _uuid4(payload["approval_id"], "approval_id")
            _uuid4(payload["presentation_id"], "presentation_id")
            _sha(payload["approval_binding_sha256"], "approval_binding_sha256")
            if payload["reason_code"] not in INVALIDATION_REASONS:
                raise ProtocolError("protocol.invalid_field", "invalidation reason is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "type": self.type, "event_id": self.event_id, "session_id": self.session_id, "seq": self.seq, "sent_at": self.sent_at, "payload": dict(self.payload)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    @property
    def semantic_sha256(self) -> str:
        semantic = {name: value for name, value in self.to_dict().items() if name not in {"session_id", "seq"}}
        return hashlib.sha256(json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, raw: str | bytes) -> "ActionApprovalEnvelope":
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ProtocolError("protocol.invalid_json", "frame is not UTF-8") from exc
        if raw.startswith("\ufeff") or len(raw.encode("utf-8")) > MAX_ACTION_APPROVAL_FRAME_BYTES:
            raise ProtocolError("protocol.frame_too_large", "action approval frame is invalid")

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ProtocolError("protocol.invalid_json", "duplicate JSON key")
                result[key] = value
            return result

        try:
            value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError("protocol.invalid_json", value)))
        except ProtocolError:
            raise
        except json.JSONDecodeError as exc:
            raise ProtocolError("protocol.invalid_json", "invalid JSON") from exc
        if not isinstance(value, Mapping) or set(value) != {"protocol", "type", "event_id", "session_id", "seq", "sent_at", "payload"}:
            raise ProtocolError("protocol.forbidden_field", "action approval envelope fields are not exact")
        return cls(**dict(value))


class ActionApprovalEventFactory:
    def __init__(self, session_id: str) -> None:
        self.session_id = _uuid4(session_id, "session_id")
        self._sequence = 0

    @property
    def next_seq(self) -> int:
        return self._sequence + 1

    def consume_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def create(self, event_type: str, payload: Mapping[str, Any], *, event_id: str | None = None, sent_at: str | None = None) -> ActionApprovalEnvelope:
        self.consume_sequence()
        return ActionApprovalEnvelope(
            ACTION_APPROVAL_PROTOCOL,
            event_type,
            event_id or str(uuid.uuid4()),
            self.session_id,
            self._sequence,
            sent_at or datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            dict(payload),
        )


class ActionApprovalJournal:
    """Content-free persist-before-send journal for action decisions."""

    def __init__(
        self,
        path: Path,
        *,
        trusted_boundary: str | Path | None = None,
    ) -> None:
        self.path = prepare_voice_journal_path(
            path,
            trusted_boundary=trusted_boundary,
        )
        self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        try:
            self._connection.executescript(
                """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS action_events(
                event_id TEXT PRIMARY KEY, semantic_sha256 TEXT NOT NULL,
                event_type TEXT NOT NULL, approval_id TEXT, presentation_id TEXT,
                binding_sha256 TEXT, observed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS action_decisions(
                decision_event_id TEXT PRIMARY KEY, approval_id TEXT NOT NULL,
                presentation_id TEXT NOT NULL, requested_event_id TEXT NOT NULL,
                binding_sha256 TEXT NOT NULL, decision_nonce TEXT NOT NULL,
                decision TEXT NOT NULL CHECK(decision IN ('approve','deny')),
                state TEXT NOT NULL CHECK(state IN ('PENDING','ACKED','RESOLVED','INVALIDATED')),
                created_at REAL NOT NULL, last_sent_at REAL, result_code TEXT, sent_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_action_decision
            ON action_decisions(approval_id,presentation_id);
            """
            )
            self._connection.execute("BEGIN IMMEDIATE")
            if "sent_at" not in {row["name"] for row in self._connection.execute("PRAGMA table_info(action_decisions)")}:
                self._connection.execute("ALTER TABLE action_decisions ADD COLUMN sent_at TEXT")
            self._connection.execute("UPDATE action_decisions SET last_sent_at=NULL WHERE state='PENDING'")
            self._connection.execute("COMMIT")
            prepare_voice_journal_path(
                self.path,
                trusted_boundary=trusted_boundary,
            )
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def observe(self, event: ActionApprovalEnvelope) -> bool:
        payload = event.payload
        approval_id = payload.get("approval_id")
        presentation_id = payload.get("presentation_id")
        binding = payload.get("approval_binding_sha256")
        with self._lock:
            existing = self._connection.execute("SELECT semantic_sha256 FROM action_events WHERE event_id=?", (event.event_id,)).fetchone()
            if existing is not None:
                if existing["semantic_sha256"] != event.semantic_sha256:
                    raise ProtocolError("protocol.conflicting_duplicate", "conflicting action event")
                return False
            self._connection.execute(
                "INSERT INTO action_events VALUES(?,?,?,?,?,?,?)",
                (event.event_id, event.semantic_sha256, event.type, approval_id, presentation_id, binding, time.time()),
            )
            return True

    def persist_decision(self, requested: ActionApprovalEnvelope, decision: str, *, factory: ActionApprovalEventFactory | None = None) -> ActionApprovalEnvelope:
        if requested.type != "action.approval.requested" or decision not in {"approve", "deny"}:
            raise ProtocolError("protocol.invalid_field", "invalid action decision")
        payload = requested.payload
        payload_value = {
                "approval_id": payload["approval_id"],
                "presentation_id": payload["presentation_id"],
                "requested_event_id": requested.event_id,
                "approval_binding_sha256": payload["approval_binding_sha256"],
                "decision_nonce": payload["decision_nonce"],
                "decision": decision,
            }
        event = (
            factory.create("action.approval.decision", payload_value)
            if factory is not None
            else ActionApprovalEventFactory(requested.session_id).create(
                "action.approval.decision", payload_value
            )
        )
        with self._lock:
            self._connection.execute(
                """INSERT INTO action_decisions VALUES(?,?,?,?,?,?,?,'PENDING',?,?,NULL,?)
                   ON CONFLICT(approval_id,presentation_id) DO NOTHING""",
                (event.event_id, payload["approval_id"], payload["presentation_id"], requested.event_id, payload["approval_binding_sha256"], payload["decision_nonce"], decision, time.time(), None, event.sent_at),
            )
        return event

    def mark_ack(self, decision_event_id: str) -> None:
        with self._lock:
            self._connection.execute("UPDATE action_decisions SET state='ACKED' WHERE decision_event_id=? AND state='PENDING'", (decision_event_id,))

    def mark_result(self, decision_event_id: str, reason_code: str) -> None:
        with self._lock:
            self._connection.execute("UPDATE action_decisions SET state='RESOLVED',result_code=? WHERE decision_event_id=?", (reason_code, decision_event_id))

    def mark_invalidated(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self._connection.execute(
                """UPDATE action_decisions SET state='INVALIDATED',result_code=?
                   WHERE approval_id=? AND presentation_id=? AND binding_sha256=? AND state IN ('PENDING','ACKED')""",
                (payload["reason_code"], payload["approval_id"], payload["presentation_id"], payload["approval_binding_sha256"]),
            )

    def mark_sent(self, decision_event_id: str, *, now: float | None = None) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE action_decisions SET last_sent_at=? WHERE decision_event_id=? AND state='PENDING'",
                (time.monotonic() if now is None else now, decision_event_id),
            )

    def pending_decisions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM action_decisions WHERE state='PENDING' ORDER BY created_at,decision_event_id"
            ).fetchall()
        return [dict(row) for row in rows]


class ActionApprovalController:
    """Action extension sequencing and F10/F11 arbitration, content-free."""

    def __init__(self, *, session_id: str, journal: ActionApprovalJournal, approval_arbiter: GlobalApprovalController) -> None:
        self.factory = ActionApprovalEventFactory(session_id)
        self.journal = journal
        self.approval_arbiter = approval_arbiter
        self.requested: ActionApprovalEnvelope | None = None
        self._next_server_seq = 1
        self._seen: dict[int, str] = {}

    def begin_new_connection(self) -> None:
        self.factory = ActionApprovalEventFactory(self.factory.session_id)
        self._next_server_seq = 1
        self._seen.clear()

    def handle_server_event(self, event: ActionApprovalEnvelope, *, key_counter: int, both_keys_up: bool) -> list[ActionApprovalEnvelope]:
        if event.type not in SERVER_EVENTS:
            raise ProtocolError("protocol.wrong_direction", "server action event required")
        fingerprint = event.to_json()
        if event.seq == self._next_server_seq:
            self._seen[event.seq] = fingerprint
            self._next_server_seq += 1
        elif event.seq < self._next_server_seq and self._seen.get(event.seq) == fingerprint:
            return []
        else:
            raise ProtocolError("protocol.sequence_gap", "action approval sequence is invalid")
        if event.type == "action.approval.ready":
            return []
        if event.type == "action.approval.decision_ack":
            self.journal.mark_ack(event.payload["decision_event_id"])
            return []
        duplicate = not self.journal.observe(event)
        replies = [self.factory.create("action.approval.event_ack", {"acked_event_id": event.event_id, "status": "duplicate" if duplicate else "applied"})]
        if duplicate:
            return replies
        if event.type == "action.approval.requested":
            self.approval_arbiter.stage(
                protocol=ACTION_APPROVAL_PROTOCOL,
                approval_id=event.payload["approval_id"],
                presentation_id=event.payload["presentation_id"],
                requested_event_id=event.event_id,
                key_counter=key_counter,
                both_keys_up=both_keys_up,
            )
            self.requested = event
        elif event.type in {"action.approval.decision_result", "action.approval.invalidated"}:
            approval_id = event.payload["approval_id"]
            if event.type == "action.approval.decision_result":
                self.journal.mark_result(event.payload["decision_event_id"], event.payload["reason_code"])
            else:
                self.journal.mark_invalidated(event.payload)
            self.approval_arbiter.close(protocol=ACTION_APPROVAL_PROTOCOL, approval_id=approval_id)
            if self.requested is not None and self.requested.payload["approval_id"] == approval_id:
                self.requested = None
        return replies

    def key_down(self, key: str, *, key_counter: int, capture_idle: bool, autorepeat: bool) -> ActionApprovalEnvelope | None:
        if key not in {"F10", "F11"} or self.requested is None:
            return None
        if not self.approval_arbiter.begin_decision(
            protocol=ACTION_APPROVAL_PROTOCOL,
            key_counter=key_counter,
            capture_idle=capture_idle,
            autorepeat=autorepeat,
        ):
            return None
        return self.journal.persist_decision(
            self.requested,
            "approve" if key == "F10" else "deny",
            factory=self.factory,
        )

    def due_decision_retransmissions(self, *, now: float | None = None) -> list[ActionApprovalEnvelope]:
        selected = time.monotonic() if now is None else now
        result: list[ActionApprovalEnvelope] = []
        for row in self.journal.pending_decisions():
            if row["sent_at"] is None:
                raise ProtocolError("action_journal.replay_identity_missing",
                                    "pending action decision has no recorded timestamp; explicit recovery is required")
            last = row["last_sent_at"]
            if last is not None and selected - float(last) < 1.0:
                continue
            result.append(
                self.factory.create(
                    "action.approval.decision",
                    {
                        "approval_id": row["approval_id"],
                        "presentation_id": row["presentation_id"],
                        "requested_event_id": row["requested_event_id"],
                        "approval_binding_sha256": row["binding_sha256"],
                        "decision_nonce": row["decision_nonce"],
                        "decision": row["decision"],
                    },
                    event_id=row["decision_event_id"], sent_at=row["sent_at"],
                )
            )
        return result
