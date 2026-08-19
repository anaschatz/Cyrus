"""Negotiated ``jarvis.cloud-approval/1`` transport extension.

This module is transport reliability only.  It never sees a cloud prompt or
decides whether cloud egress is allowed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .errors import ProtocolError
from .private_state import prepare_voice_journal_path
from .approval_arbiter import GlobalApprovalController


CLOUD_APPROVAL_PROTOCOL = "jarvis.cloud-approval/1"
MAX_EXTENSION_FRAME_BYTES = 16 * 1024
CLIENT_EXTENSION_EVENTS = frozenset({"cloud.approval.event_ack", "cloud.approval.decision"})
SERVER_EXTENSION_EVENTS = frozenset(
    {
        "cloud.approval.ready",
        "cloud.approval.requested",
        "cloud.approval.decision_ack",
        "cloud.approval.decision_result",
        "cloud.approval.invalidated",
    }
)
EXTENSION_EVENTS = CLIENT_EXTENSION_EVENTS | SERVER_EXTENSION_EVENTS

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
    }
)
INVALIDATION_REASONS = frozenset(
    {
        "approval_expired",
        "task_cancelled",
        "superseded",
        "binding_changed",
        "endpoint_unhealthy",
        "budget_changed",
        "revalidation_unavailable",
        "session_superseded",
    }
)


def default_cloud_approval_journal_path() -> Path:
    import sys

    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA", "").strip()
        if not root:
            raise ProtocolError("cloud_journal.state_root_missing", "LOCALAPPDATA is required")
        return Path(root) / "JARVIS Voice" / "state" / "cloud-approval.sqlite3"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "JARVIS Voice" / "state" / "cloud-approval.sqlite3"
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "jarvis-voice" / "cloud-approval.sqlite3"


def _uuid4(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ProtocolError("protocol.invalid_field", f"{field_name} must be UUID-v4")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("protocol.invalid_field", f"{field_name} must be UUID-v4") from exc
    if parsed.version != 4 or str(parsed) != value.lower():
        raise ProtocolError("protocol.invalid_field", f"{field_name} must be canonical UUID-v4")
    return value.lower()


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError("protocol.invalid_field", f"{field_name} must be lowercase SHA-256")
    return value


def _timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise ProtocolError("protocol.invalid_field", "sent_at must be UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolError("protocol.invalid_field", "sent_at must be UTC RFC3339") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ProtocolError("protocol.invalid_field", "sent_at must be UTC RFC3339")
    return value


def _exact(payload: Mapping[str, Any], names: set[str], event_type: str) -> None:
    if set(payload) != names:
        raise ProtocolError("protocol.forbidden_field", f"{event_type} payload fields are not exact")


_REQUESTED_FIELDS = {
    "approval_id",
    "presentation_id",
    "task_id",
    "epoch",
    "approval_binding_sha256",
    "egress_sha256",
    "route_plan_sha256",
    "model_canonical_id",
    "provider_endpoint_tag",
    "quantization",
    "billable_prompt_tokens",
    "input_token_limit",
    "output_token_limit",
    "max_reserved_microusd",
    "stable_catalog_sha256",
    "qualification_sha256",
    "included_turn_ids",
    "expires_at",
    "decision_nonce",
    "approve_key",
    "deny_key",
}


@dataclass(frozen=True, slots=True)
class CloudApprovalEnvelope:
    protocol: str
    type: str
    event_id: str
    session_id: str
    seq: int
    sent_at: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.protocol != CLOUD_APPROVAL_PROTOCOL or self.type not in EXTENSION_EVENTS:
            raise ProtocolError("protocol.unknown_event", "unknown cloud approval protocol/event")
        object.__setattr__(self, "event_id", _uuid4(self.event_id, "event_id"))
        object.__setattr__(self, "session_id", _uuid4(self.session_id, "session_id"))
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 1:
            raise ProtocolError("protocol.invalid_field", "seq must be a positive integer")
        object.__setattr__(self, "sent_at", _timestamp(self.sent_at))
        if not isinstance(self.payload, Mapping):
            raise ProtocolError("protocol.invalid_field", "payload must be an object")
        object.__setattr__(self, "payload", dict(self.payload))
        self._validate_payload()

    def _validate_payload(self) -> None:
        payload = self.payload
        if self.type == "cloud.approval.ready":
            _exact(payload, {"protocol"}, self.type)
            if payload["protocol"] != CLOUD_APPROVAL_PROTOCOL:
                raise ProtocolError("protocol.invalid_field", "invalid ready protocol")
        elif self.type == "cloud.approval.requested":
            _exact(payload, _REQUESTED_FIELDS, self.type)
            for name in ("approval_id", "presentation_id", "task_id", "decision_nonce"):
                _uuid4(payload[name], name)
            for name in (
                "approval_binding_sha256",
                "egress_sha256",
                "route_plan_sha256",
                "stable_catalog_sha256",
                "qualification_sha256",
            ):
                _sha256(payload[name], name)
            for name in ("epoch", "billable_prompt_tokens", "input_token_limit", "output_token_limit", "max_reserved_microusd"):
                if isinstance(payload[name], bool) or not isinstance(payload[name], int) or payload[name] < 0:
                    raise ProtocolError("protocol.invalid_field", f"{name} must be non-negative integer")
            if payload["approve_key"] != "F10" or payload["deny_key"] != "F11":
                raise ProtocolError("protocol.invalid_field", "approval keys are frozen")
            if not isinstance(payload["included_turn_ids"], list) or len(payload["included_turn_ids"]) > 8:
                raise ProtocolError("protocol.invalid_field", "included_turn_ids is invalid")
            for turn_id in payload["included_turn_ids"]:
                _uuid4(turn_id, "included_turn_ids")
            _timestamp(payload["expires_at"])
            for name in ("model_canonical_id", "provider_endpoint_tag", "quantization"):
                if not isinstance(payload[name], str) or not payload[name] or unicodedata.normalize("NFC", payload[name]) != payload[name]:
                    raise ProtocolError("protocol.invalid_field", f"{name} is invalid")
        elif self.type == "cloud.approval.event_ack":
            _exact(payload, {"acked_event_id", "status"}, self.type)
            _uuid4(payload["acked_event_id"], "acked_event_id")
            if payload["status"] not in {"applied", "duplicate"}:
                raise ProtocolError("protocol.invalid_field", "event ACK status is invalid")
        elif self.type == "cloud.approval.decision":
            _exact(
                payload,
                {"approval_id", "presentation_id", "requested_event_id", "approval_binding_sha256", "decision_nonce", "decision"},
                self.type,
            )
            for name in ("approval_id", "presentation_id", "requested_event_id", "decision_nonce"):
                _uuid4(payload[name], name)
            _sha256(payload["approval_binding_sha256"], "approval_binding_sha256")
            if payload["decision"] not in {"approve", "deny"}:
                raise ProtocolError("protocol.invalid_field", "decision is invalid")
        elif self.type == "cloud.approval.decision_ack":
            _exact(payload, {"decision_event_id", "status"}, self.type)
            _uuid4(payload["decision_event_id"], "decision_event_id")
            if payload["status"] not in {"accepted", "duplicate"}:
                raise ProtocolError("protocol.invalid_field", "decision ACK status is invalid")
        elif self.type == "cloud.approval.decision_result":
            _exact(
                payload,
                {"decision_event_id", "approval_id", "presentation_id", "approval_binding_sha256", "result", "reason_code"},
                self.type,
            )
            for name in ("decision_event_id", "approval_id", "presentation_id"):
                _uuid4(payload[name], name)
            _sha256(payload["approval_binding_sha256"], "approval_binding_sha256")
            if payload["result"] not in {"applied", "rejected"} or payload["reason_code"] not in DECISION_REASONS:
                raise ProtocolError("protocol.invalid_field", "decision result is invalid")
        elif self.type == "cloud.approval.invalidated":
            _exact(payload, {"approval_id", "presentation_id", "approval_binding_sha256", "reason_code"}, self.type)
            _uuid4(payload["approval_id"], "approval_id")
            _uuid4(payload["presentation_id"], "presentation_id")
            _sha256(payload["approval_binding_sha256"], "approval_binding_sha256")
            if payload["reason_code"] not in INVALIDATION_REASONS:
                raise ProtocolError("protocol.invalid_field", "invalidation reason is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "type": self.type,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "seq": self.seq,
            "sent_at": self.sent_at,
            "payload": dict(self.payload),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    @property
    def semantic_sha256(self) -> str:
        semantic = {
            "event_id": self.event_id,
            "payload": dict(self.payload),
            "protocol": self.protocol,
            "sent_at": self.sent_at,
            "type": self.type,
        }
        return hashlib.sha256(
            json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CloudApprovalEnvelope":
        exact = {"protocol", "type", "event_id", "session_id", "seq", "sent_at", "payload"}
        if set(raw) != exact:
            raise ProtocolError("protocol.forbidden_field", "extension envelope fields are not exact")
        return cls(**dict(raw))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "CloudApprovalEnvelope":
        if isinstance(raw, bytes):
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ProtocolError("protocol.invalid_json", "extension frame is not UTF-8") from exc
        else:
            text = raw
        if text.startswith("\ufeff"):
            raise ProtocolError("protocol.invalid_json", "BOM is forbidden")
        if len(text.encode("utf-8")) > MAX_EXTENSION_FRAME_BYTES:
            raise ProtocolError("protocol.frame_too_large", "extension frame exceeds 16 KiB")

        def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ProtocolError("protocol.invalid_json", "duplicate JSON key")
                result[key] = value
            return result

        def constant(value: str) -> None:
            raise ProtocolError("protocol.invalid_json", f"non-finite number {value} is forbidden")

        try:
            value = json.loads(text, object_pairs_hook=object_pairs, parse_constant=constant)
        except ProtocolError:
            raise
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ProtocolError("protocol.invalid_json", "invalid extension JSON") from exc
        if not isinstance(value, Mapping):
            raise ProtocolError("protocol.invalid_json", "extension envelope must be an object")
        return cls.from_dict(value)


class CloudApprovalEventFactory:
    def __init__(self, session_id: str) -> None:
        self.session_id = _uuid4(session_id, "session_id")
        self._sequence = 0

    def reset(self, session_id: str | None = None) -> None:
        if session_id is not None:
            self.session_id = _uuid4(session_id, "session_id")
        self._sequence = 0

    @property
    def next_seq(self) -> int:
        return self._sequence + 1

    def consume_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def create(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        event_id: str | None = None,
        sent_at: str | None = None,
    ) -> CloudApprovalEnvelope:
        self._sequence += 1
        return CloudApprovalEnvelope(
            protocol=CLOUD_APPROVAL_PROTOCOL,
            type=event_type,
            event_id=event_id or str(uuid.uuid4()),
            session_id=self.session_id,
            seq=self._sequence,
            sent_at=sent_at or datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            payload=dict(payload),
        )

    def reconstruct(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        event_id: str,
        sent_at: str,
    ) -> CloudApprovalEnvelope:
        event = CloudApprovalEnvelope(
            protocol=CLOUD_APPROVAL_PROTOCOL,
            type=event_type,
            event_id=event_id,
            session_id=self.session_id,
            seq=self.next_seq,
            sent_at=sent_at,
            payload=dict(payload),
        )
        self.consume_sequence()
        return event


class CloudApprovalJournal:
    """Bounded persisted decision/presentation journal with no prompt content."""

    def __init__(
        self,
        path: str | Path,
        *,
        hard_cap: int = 64,
        trusted_boundary: str | Path | None = None,
    ) -> None:
        self.path = prepare_voice_journal_path(
            path,
            trusted_boundary=trusted_boundary,
        )
        self.hard_cap = hard_cap
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._closed = False
        try:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.executescript(
                """
            CREATE TABLE IF NOT EXISTS cloud_events (
                event_id TEXT PRIMARY KEY,
                semantic_sha256 TEXT NOT NULL,
                event_kind TEXT NOT NULL CHECK(event_kind IN ('requested','decision')),
                approval_id TEXT NOT NULL,
                presentation_id TEXT NOT NULL,
                requested_event_id TEXT NOT NULL,
                approval_binding_sha256 TEXT NOT NULL,
                decision_nonce TEXT NOT NULL,
                decision TEXT,
                status TEXT NOT NULL,
                sent_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cloud_seen (
                event_id TEXT PRIMARY KEY,
                semantic_sha256 TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS cloud_events_expiry ON cloud_events(expires_at);
            CREATE TABLE IF NOT EXISTS approval_arbiter_local (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                protocol TEXT NOT NULL,
                approval_id TEXT NOT NULL,
                presentation_id TEXT NOT NULL,
                requested_event_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN (
                    'STAGED','ARMED','DISARMED_CAPTURE','DECISION_PENDING','CLOSED'
                )),
                arm_counter INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT OR IGNORE INTO approval_arbiter_local(
                singleton,protocol,approval_id,presentation_id,requested_event_id,state,arm_counter,updated_at
            )
            SELECT 1,'jarvis.cloud-approval/1',approval_id,presentation_id,requested_event_id,
                   'STAGED',0,sent_at
            FROM cloud_events WHERE event_kind='requested' AND status='ARMED'
            ORDER BY rowid DESC LIMIT 1;
            """
            )
            prepare_voice_journal_path(
                self.path,
                trusted_boundary=trusted_boundary,
            )
        except BaseException:
            self._connection.close()
            raise
        self.prune()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def prune(self, now_text: str | None = None) -> int:
        stamp = now_text or datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._lock:
            removed = self._connection.execute("DELETE FROM cloud_events WHERE expires_at<=?", (stamp,)).rowcount
            removed += self._connection.execute("DELETE FROM cloud_seen WHERE expires_at<=?", (stamp,)).rowcount
            return removed

    def observe_server_event(self, envelope: CloudApprovalEnvelope, *, expires_at: str) -> bool:
        """Persist before UI/application and return True for an exact replay."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                self._connection.execute("DELETE FROM cloud_seen WHERE expires_at<=?", (stamp,))
                self._connection.execute("DELETE FROM cloud_events WHERE expires_at<=?", (stamp,))
                existing = self._connection.execute("SELECT semantic_sha256 FROM cloud_seen WHERE event_id=?", (envelope.event_id,)).fetchone()
                if existing is not None:
                    if existing["semantic_sha256"] != envelope.semantic_sha256:
                        raise ProtocolError("protocol.conflicting_duplicate", "server extension identity conflict")
                    self._connection.execute("COMMIT")
                    return True
                count = self._connection.execute(
                    "SELECT COUNT(*) FROM (SELECT event_id FROM cloud_seen UNION SELECT event_id FROM cloud_events)"
                ).fetchone()[0]
                already_counted = self._connection.execute(
                    "SELECT 1 FROM cloud_seen WHERE event_id=?", (envelope.event_id,)
                ).fetchone() is not None
                if count >= self.hard_cap and not already_counted:
                    raise ProtocolError("protocol.journal_full", "cloud approval journal is full")
                self._connection.execute(
                    "INSERT INTO cloud_seen(event_id,semantic_sha256,expires_at) VALUES(?,?,?)",
                    (envelope.event_id, envelope.semantic_sha256, expires_at),
                )
                self._connection.execute("COMMIT")
                return False
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def record(
        self,
        envelope: CloudApprovalEnvelope,
        *,
        event_kind: str,
        status: str,
        decision: str | None = None,
        expires_at: str,
    ) -> bool:
        if event_kind not in {"requested", "decision"}:
            raise ValueError("invalid journal event kind")
        payload = envelope.payload
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute("DELETE FROM cloud_events WHERE expires_at<=?", (datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),))
                existing = self._connection.execute("SELECT semantic_sha256 FROM cloud_events WHERE event_id=?", (envelope.event_id,)).fetchone()
                if existing is not None:
                    if existing["semantic_sha256"] != envelope.semantic_sha256:
                        raise ProtocolError("protocol.conflicting_duplicate", "journal event identity conflict")
                    self._connection.execute("COMMIT")
                    return False
                count = self._connection.execute(
                    "SELECT COUNT(*) FROM (SELECT event_id FROM cloud_seen UNION SELECT event_id FROM cloud_events)"
                ).fetchone()[0]
                already_counted = self._connection.execute(
                    "SELECT 1 FROM cloud_seen WHERE event_id=?", (envelope.event_id,)
                ).fetchone() is not None
                if count >= self.hard_cap and not already_counted:
                    raise ProtocolError("protocol.journal_full", "cloud approval journal is full")
                self._connection.execute(
                    """INSERT INTO cloud_events(
                       event_id,semantic_sha256,event_kind,approval_id,presentation_id,
                       requested_event_id,approval_binding_sha256,decision_nonce,decision,status,sent_at,expires_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        envelope.event_id,
                        envelope.semantic_sha256,
                        event_kind,
                        payload["approval_id"],
                        payload["presentation_id"],
                        payload.get("requested_event_id", envelope.event_id),
                        payload["approval_binding_sha256"],
                        payload["decision_nonce"],
                        decision,
                        status,
                        envelope.sent_at,
                        expires_at,
                    ),
                )
                self._connection.execute("COMMIT")
                return True
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def mark_decision_acknowledged(self, decision_event_id: str) -> bool:
        with self._lock:
            return self._connection.execute(
                "UPDATE cloud_events SET status='ACKED' WHERE event_id=? AND event_kind='decision'",
                (decision_event_id,),
            ).rowcount == 1

    def pending_decisions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._connection.execute(
                "SELECT * FROM cloud_events WHERE event_kind='decision' AND status='PENDING' ORDER BY rowid"
            ).fetchall()]

    def pending_presentation(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM cloud_events WHERE event_kind='requested' AND status='ARMED'
                   ORDER BY rowid DESC LIMIT 1"""
            ).fetchone()
            return dict(row) if row is not None else None

    def close_presentation(self, approval_id: str) -> None:
        with self._lock:
            self._connection.execute(
                """UPDATE cloud_events SET status='CLOSED'
                   WHERE event_kind='requested' AND approval_id=? AND status='ARMED'""",
                (approval_id,),
            )

    def load_arbiter_presentation(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM approval_arbiter_local WHERE singleton=1 AND state!='CLOSED'"
            ).fetchone()
            return dict(row) if row is not None else None

    def store_arbiter_presentation(self, *, protocol: str, approval_id: str,
                                   presentation_id: str, requested_event_id: str,
                                   state: str, arm_counter: int) -> None:
        with self._lock:
            self._connection.execute(
                """INSERT INTO approval_arbiter_local(
                   singleton,protocol,approval_id,presentation_id,requested_event_id,state,arm_counter,updated_at
                   ) VALUES(1,?,?,?,?,?,?,?) ON CONFLICT(singleton) DO UPDATE SET
                   protocol=excluded.protocol,approval_id=excluded.approval_id,
                   presentation_id=excluded.presentation_id,requested_event_id=excluded.requested_event_id,
                   state=excluded.state,arm_counter=excluded.arm_counter,updated_at=excluded.updated_at""",
                (protocol, approval_id, presentation_id, requested_event_id, state, arm_counter,
                 datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")),
            )

    def close_arbiter_presentation(self) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE approval_arbiter_local SET state='CLOSED',updated_at=? WHERE singleton=1",
                (datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),),
            )


@dataclass(frozen=True, slots=True)
class ArmedPresentation:
    approval_id: str
    presentation_id: str
    requested_event_id: str
    approval_binding_sha256: str
    decision_nonce: str
    expires_at: str
    arm_counter: int


class CloudApprovalController:
    """Voice-side F10/F11 arming and durable decision retransmission."""

    def __init__(
        self,
        *,
        session_id: str,
        journal: CloudApprovalJournal,
        approval_arbiter: GlobalApprovalController | None = None,
    ) -> None:
        self.factory = CloudApprovalEventFactory(session_id)
        self.journal = journal
        self.approval_arbiter = approval_arbiter or GlobalApprovalController(journal)
        pending_decisions = self.journal.pending_decisions()
        pending = self.journal.pending_presentation() if not pending_decisions else None
        self.presentation: ArmedPresentation | None = (
            ArmedPresentation(
                approval_id=pending["approval_id"],
                presentation_id=pending["presentation_id"],
                requested_event_id=pending["requested_event_id"],
                approval_binding_sha256=pending["approval_binding_sha256"],
                decision_nonce=pending["decision_nonce"],
                expires_at=pending["expires_at"],
                arm_counter=(self.approval_arbiter.presentation.arm_counter
                             if self.approval_arbiter.presentation is not None else 0),
            ) if pending is not None else None
        )
        self._next_server_seq = 1
        self._server_seen: dict[int, str] = {}
        self._decision_retry: dict[str, tuple[int, float]] = {
            row["event_id"]: (0, 0.0) for row in pending_decisions
        }

    def begin_new_connection(self) -> None:
        self.factory.reset()
        self._next_server_seq = 1
        self._server_seen.clear()
        self._decision_retry = {
            row["event_id"]: (0, 0.0) for row in self.journal.pending_decisions()
        }

    def handle_server_event(
        self,
        event: CloudApprovalEnvelope,
        *,
        key_counter: int,
        both_keys_up: bool,
    ) -> list[CloudApprovalEnvelope]:
        if event.type not in SERVER_EXTENSION_EVENTS:
            raise ProtocolError("protocol.wrong_direction", "client extension event received from server")
        fingerprint = event.to_json()
        if event.seq == self._next_server_seq:
            self._server_seen[event.seq] = fingerprint
            self._next_server_seq += 1
        elif event.seq < self._next_server_seq and self._server_seen.get(event.seq) == fingerprint:
            return []
        else:
            raise ProtocolError("protocol.sequence_gap", "cloud approval sequence is invalid")
        if event.type == "cloud.approval.ready":
            return []
        # The decision ACK is itself the transport acknowledgement for a
        # client decision.  Acknowledging that ACK would create an
        # unnecessary ACK-of-ACK exchange which is not part of the frozen
        # extension reliability contract.
        if event.type == "cloud.approval.decision_ack":
            self.journal.mark_decision_acknowledged(event.payload["decision_event_id"])
            self._decision_retry.pop(event.payload["decision_event_id"], None)
            return []
        expires = event.payload.get("expires_at")
        if not isinstance(expires, str):
            expires = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        duplicate = self.journal.observe_server_event(event, expires_at=expires)
        outbound = [
            self.factory.create(
                "cloud.approval.event_ack",
                {"acked_event_id": event.event_id, "status": "duplicate" if duplicate else "applied"},
            )
        ]
        if event.type == "cloud.approval.requested" and not duplicate:
            self.journal.record(event, event_kind="requested", status="ARMED", expires_at=event.payload["expires_at"])
            armed = self.approval_arbiter.stage(
                protocol=CLOUD_APPROVAL_PROTOCOL,
                approval_id=event.payload["approval_id"],
                presentation_id=event.payload["presentation_id"],
                requested_event_id=event.event_id,
                key_counter=key_counter,
                both_keys_up=both_keys_up,
            )
            self.presentation = (
                ArmedPresentation(
                    approval_id=event.payload["approval_id"],
                    presentation_id=event.payload["presentation_id"],
                    requested_event_id=event.event_id,
                    approval_binding_sha256=event.payload["approval_binding_sha256"],
                    decision_nonce=event.payload["decision_nonce"],
                    expires_at=event.payload["expires_at"],
                    arm_counter=key_counter,
                )
                if armed
                else None
            )
        elif event.type in {"cloud.approval.decision_result", "cloud.approval.invalidated"}:
            self.journal.close_presentation(event.payload["approval_id"])
            self.approval_arbiter.close(
                protocol=CLOUD_APPROVAL_PROTOCOL, approval_id=event.payload["approval_id"]
            )
            if self.presentation is not None and self.presentation.approval_id == event.payload["approval_id"]:
                self.presentation = None
        return outbound

    def disarm_for_ptt(self) -> None:
        self.approval_arbiter.disarm_for_capture()

    def rearm_after_capture(self, *, key_counter: int, both_keys_up: bool) -> bool:
        if not self.approval_arbiter.capture_ended(
            key_counter=key_counter, both_keys_up=both_keys_up
        ) or self.presentation is None:
            return False
        value = self.presentation
        self.presentation = ArmedPresentation(
            approval_id=value.approval_id,
            presentation_id=value.presentation_id,
            requested_event_id=value.requested_event_id,
            approval_binding_sha256=value.approval_binding_sha256,
            decision_nonce=value.decision_nonce,
            expires_at=value.expires_at,
            arm_counter=key_counter,
        )
        return True

    def key_down(
        self,
        key: str,
        *,
        key_counter: int,
        capture_idle: bool,
        autorepeat: bool,
    ) -> CloudApprovalEnvelope | None:
        presentation = self.presentation
        if (
            presentation is None
            or key not in {"F10", "F11"}
            or autorepeat
            or not capture_idle
            or key_counter <= presentation.arm_counter
            or not self.approval_arbiter.begin_decision(
                protocol=CLOUD_APPROVAL_PROTOCOL,
                key_counter=key_counter,
                capture_idle=capture_idle,
                autorepeat=autorepeat,
            )
        ):
            return None
        decision = "approve" if key == "F10" else "deny"
        event = self.factory.create(
            "cloud.approval.decision",
            {
                "approval_binding_sha256": presentation.approval_binding_sha256,
                "approval_id": presentation.approval_id,
                "decision": decision,
                "decision_nonce": presentation.decision_nonce,
                "presentation_id": presentation.presentation_id,
                "requested_event_id": presentation.requested_event_id,
            },
        )
        # Persist before returning it to the transport.
        self.journal.record(
            event,
            event_kind="decision",
            status="PENDING",
            decision=decision,
            expires_at=presentation.expires_at,
        )
        self.presentation = None
        self._decision_retry[event.event_id] = (0, 0.0)
        return event

    def pending_decision_frames(self) -> list[CloudApprovalEnvelope]:
        return [self._decision_from_row(row) for row in self.journal.pending_decisions()]

    def _decision_from_row(self, row: Mapping[str, Any]) -> CloudApprovalEnvelope:
        return self.factory.create(
            "cloud.approval.decision",
            {
                "approval_binding_sha256": row["approval_binding_sha256"],
                "approval_id": row["approval_id"],
                "decision": row["decision"],
                "decision_nonce": row["decision_nonce"],
                "presentation_id": row["presentation_id"],
                "requested_event_id": row["requested_event_id"],
            },
            event_id=row["event_id"],
            sent_at=row["sent_at"],
        )

    def mark_decision_sent(self, event_id: str, *, now: float | None = None) -> None:
        attempt, _next = self._decision_retry.get(event_id, (0, 0.0))
        delays = (0.25, 0.5, 1.0, 2.0)
        delay = delays[attempt] if attempt < len(delays) else 5.0
        self._decision_retry[event_id] = (attempt + 1, (time.monotonic() if now is None else now) + delay)

    def due_decision_retransmissions(self, *, now: float | None = None) -> list[CloudApprovalEnvelope]:
        selected = time.monotonic() if now is None else now
        rows = {row["event_id"]: row for row in self.journal.pending_decisions()}
        due: list[CloudApprovalEnvelope] = []
        for event_id, (attempt, next_at) in tuple(self._decision_retry.items()):
            row = rows.get(event_id)
            if row is None:
                self._decision_retry.pop(event_id, None)
            elif next_at <= selected:
                event = self._decision_from_row(row)
                due.append(event)
                delays = (0.25, 0.5, 1.0, 2.0)
                delay = delays[attempt] if attempt < len(delays) else 5.0
                self._decision_retry[event_id] = (attempt + 1, selected + delay)
        return due
