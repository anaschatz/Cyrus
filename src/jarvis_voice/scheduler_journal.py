"""Encrypted durable scheduler inbox and isolated playback journal for Voice."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .errors import ProtocolError
from .scheduler_playback import SchedulerPlaybackRecord


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
INSERT OR IGNORE INTO metadata(key,value) VALUES('schema_version','1');
CREATE TABLE IF NOT EXISTS encrypted_frames(
  event_id TEXT PRIMARY KEY,
  semantic_sha256 TEXT NOT NULL,
  record_type TEXT NOT NULL,
  record_id TEXT NOT NULL UNIQUE,
  key_version INTEGER NOT NULL,
  nonce BLOB NOT NULL CHECK(length(nonce)=12),
  ciphertext_and_tag BLOB NOT NULL,
  content_sha256 TEXT NOT NULL,
  aad_sha256 TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('STAGED','APPLIED','CLOSED','QUARANTINED')),
  created_at_utc TEXT NOT NULL,
  UNIQUE(key_version,nonce)
);
CREATE TABLE IF NOT EXISTS scheduler_playback(
  notification_id TEXT NOT NULL,
  unit_index INTEGER NOT NULL,
  frame_event_id TEXT NOT NULL REFERENCES encrypted_frames(event_id),
  state TEXT NOT NULL CHECK(state IN ('STAGED','PLAYING','COMPLETED','CANCELLED','SKIPPED')),
  updated_at_utc TEXT NOT NULL,
  PRIMARY KEY(notification_id,unit_index)
);
CREATE TABLE IF NOT EXISTS scheduler_tombstones(
  notification_id TEXT PRIMARY KEY,
  reason_code TEXT NOT NULL,
  created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scheduler_delivery_attempts(
  attempt_id TEXT PRIMARY KEY,
  notification_id TEXT NOT NULL,
  attempt_ordinal INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('DISPATCHING','OS_ACCEPTED','OS_REJECTED','AMBIGUOUS_OS_DISPATCH')),
  reason_code TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  observed_at_utc TEXT,
  UNIQUE(notification_id,attempt_ordinal)
);
CREATE TABLE IF NOT EXISTS scheduler_native_callbacks(
  callback_id TEXT PRIMARY KEY,
  notification_id TEXT NOT NULL,
  frame_event_id TEXT NOT NULL UNIQUE REFERENCES encrypted_frames(event_id),
  semantic_sha256 TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('PENDING','EMITTED','CLOSED')),
  scheduler_event_id TEXT,
  created_at_utc TEXT NOT NULL,
  emitted_at_utc TEXT
);
"""


def default_scheduler_journal_path() -> Path:
    import sys

    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA", "").strip()
        if not root:
            raise ValueError("scheduler journal state root missing")
        return Path(root) / "JARVIS Voice" / "state" / "scheduler.sqlite3"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "JARVIS Voice" / "state" / "scheduler.sqlite3"
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "jarvis-voice" / "scheduler.sqlite3"


class SchedulerClientJournal:
    """Cipher is injected so the Voice executable can use its own key namespace."""

    def __init__(self, path: Path, *, cipher: Any) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = cipher
        self._db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._db.executescript(SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    @staticmethod
    def _binding(record_type: str, record_id: str) -> Any:
        # Import only when a scheduler-capable distribution includes the shared
        # crypto contract. A client without it must not advertise the extension.
        from jarvis_orchestrator.scheduler_crypto import ContentBinding
        return ContentBinding("voice", record_type, record_id, None, 0)

    def stage(self, *, event_id: str, semantic_sha256: str, event_type: str,
              payload: Mapping[str, Any]) -> str:
        plaintext = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode()
        record_id = event_id
        encrypted = self._cipher.encrypt(plaintext, self._binding(event_type, record_id))
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                prior = self._db.execute(
                    "SELECT semantic_sha256,state FROM encrypted_frames WHERE event_id=?", (event_id,)
                ).fetchone()
                if prior is not None:
                    if prior["semantic_sha256"] != semantic_sha256:
                        self._db.execute(
                            "UPDATE encrypted_frames SET state='QUARANTINED' WHERE event_id=?", (event_id,)
                        )
                        self._db.execute("COMMIT")
                        raise ValueError("protocol.conflicting_duplicate")
                    self._db.execute("COMMIT")
                    return "duplicate"
                self._db.execute(
                    """INSERT INTO encrypted_frames(
                       event_id,semantic_sha256,record_type,record_id,key_version,nonce,
                       ciphertext_and_tag,content_sha256,aad_sha256,state,created_at_utc
                       ) VALUES(?,?,?,?,?,?,?,?,?,'STAGED',?)""",
                    (event_id, semantic_sha256, event_type, record_id, encrypted.key_version,
                     encrypted.nonce, encrypted.ciphertext_and_tag, encrypted.content_sha256,
                     encrypted.aad_sha256, self._now()),
                )
                self._db.execute("COMMIT")
                return "applied"
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    def read(self, event_id: str) -> dict[str, Any]:
        from jarvis_orchestrator.scheduler_crypto import EncryptedContent
        with self._lock:
            row = self._db.execute("SELECT * FROM encrypted_frames WHERE event_id=?", (event_id,)).fetchone()
        if row is None or row["state"] == "QUARANTINED":
            raise KeyError(event_id)
        raw = self._cipher.decrypt(
            EncryptedContent(row["key_version"], bytes(row["nonce"]), bytes(row["ciphertext_and_tag"]),
                             row["content_sha256"], row["aad_sha256"]),
            self._binding(row["record_type"], row["record_id"]),
        )
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("scheduler journal payload invalid")
        return value

    def stage_playback(self, *, notification_id: str, unit_index: int, frame_event_id: str) -> bool:
        with self._lock:
            if self._db.execute(
                "SELECT 1 FROM scheduler_tombstones WHERE notification_id=?", (notification_id,)
            ).fetchone():
                return False
            existing = self._db.execute(
                """SELECT frame_event_id FROM scheduler_playback
                   WHERE notification_id=? AND unit_index=?""",
                (notification_id, unit_index),
            ).fetchone()
            if existing is not None:
                previous = self.read(existing["frame_event_id"])
                current = self.read(frame_event_id)
                fields = ("occurrence_id", "notification_id", "unit_index", "locale", "speakable_text")
                if all(previous.get(name) == current.get(name) for name in fields):
                    return False
                raise ProtocolError(
                    "protocol.conflicting_scheduler_speech",
                    "Scheduler speech unit index was reused with different content",
                )
            maximum = self._db.execute(
                "SELECT COALESCE(MAX(unit_index),0) FROM scheduler_playback WHERE notification_id=?",
                (notification_id,),
            ).fetchone()[0]
            if unit_index != int(maximum) + 1:
                raise ProtocolError(
                    "protocol.scheduler_speech_gap",
                    "Scheduler speech unit index is not contiguous",
                )
            self._db.execute(
                """INSERT INTO scheduler_playback(
                   notification_id,unit_index,frame_event_id,state,updated_at_utc
                   ) VALUES(?,?,?,'STAGED',?)""",
                (notification_id, unit_index, frame_event_id, self._now()),
            )
            return True

    def next_staged_playback(self) -> SchedulerPlaybackRecord | None:
        with self._lock:
            row = self._db.execute(
                """SELECT notification_id,unit_index,frame_event_id
                   FROM scheduler_playback WHERE state='STAGED'
                   ORDER BY updated_at_utc,notification_id,unit_index LIMIT 1"""
            ).fetchone()
        if row is None:
            return None
        payload = self.read(row["frame_event_id"])
        return SchedulerPlaybackRecord(
            notification_id=row["notification_id"],
            unit_index=int(row["unit_index"]),
            locale=payload["locale"],
            speakable_text=payload["speakable_text"],
        )

    def recover_interrupted_playback(self) -> list[tuple[str, int]]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                rows = self._db.execute(
                    """SELECT notification_id,unit_index FROM scheduler_playback
                       WHERE state='PLAYING' ORDER BY notification_id,unit_index"""
                ).fetchall()
                self._db.execute(
                    """UPDATE scheduler_playback SET state='CANCELLED',updated_at_utc=?
                       WHERE state='PLAYING'""",
                    (self._now(),),
                )
                self._db.execute("COMMIT")
                return [(row["notification_id"], int(row["unit_index"])) for row in rows]
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def cancel_notification_playback(self, notification_id: str) -> list[int]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                rows = self._db.execute(
                    """SELECT unit_index FROM scheduler_playback
                       WHERE notification_id=? AND state IN ('STAGED','PLAYING')
                       ORDER BY unit_index""",
                    (notification_id,),
                ).fetchall()
                self._db.execute(
                    """UPDATE scheduler_playback SET state='CANCELLED',updated_at_utc=?
                       WHERE notification_id=? AND state IN ('STAGED','PLAYING')""",
                    (self._now(), notification_id),
                )
                self._db.execute("COMMIT")
                return [int(row["unit_index"]) for row in rows]
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def begin_native_dispatch(self, notification_id: str) -> str:
        import uuid
        attempt_id = str(uuid.uuid4())
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                ordinal = self._db.execute(
                    "SELECT COUNT(*)+1 FROM scheduler_delivery_attempts WHERE notification_id=?",
                    (notification_id,),
                ).fetchone()[0]
                self._db.execute(
                    """INSERT INTO scheduler_delivery_attempts(
                       attempt_id,notification_id,attempt_ordinal,state,reason_code,created_at_utc
                       ) VALUES(?,?,?,'DISPATCHING','native_call_not_started',?)""",
                    (attempt_id, notification_id, ordinal, self._now()),
                )
                self._db.execute("COMMIT")
                return attempt_id
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def finish_native_dispatch(self, attempt_id: str, *, state: str, reason_code: str) -> bool:
        if state not in {"OS_ACCEPTED", "OS_REJECTED", "AMBIGUOUS_OS_DISPATCH"}:
            raise ValueError("invalid native dispatch terminal state")
        with self._lock:
            return self._db.execute(
                """UPDATE scheduler_delivery_attempts SET state=?,reason_code=?,observed_at_utc=?
                   WHERE attempt_id=? AND state='DISPATCHING'""",
                (state, reason_code, self._now(), attempt_id),
            ).rowcount == 1

    def recover_ambiguous_dispatches(self) -> list[dict[str, Any]]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                rows = self._db.execute(
                    """SELECT attempt_id,notification_id,attempt_ordinal
                       FROM scheduler_delivery_attempts WHERE state='DISPATCHING'
                       ORDER BY notification_id,attempt_ordinal"""
                ).fetchall()
                self._db.execute(
                    """UPDATE scheduler_delivery_attempts SET state='AMBIGUOUS_OS_DISPATCH',
                       reason_code='restart_without_native_receipt',observed_at_utc=?
                       WHERE state='DISPATCHING'""",
                    (self._now(),),
                )
                self._db.execute("COMMIT")
                return [dict(row) for row in rows]
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def notification_dispatch_needed(self, notification_id: str) -> bool:
        with self._lock:
            if self._db.execute(
                "SELECT 1 FROM scheduler_tombstones WHERE notification_id=?", (notification_id,)
            ).fetchone():
                return False
            accepted = self._db.execute(
                """SELECT 1 FROM scheduler_delivery_attempts
                   WHERE notification_id=? AND state='OS_ACCEPTED'""",
                (notification_id,),
            ).fetchone()
            dispatching = self._db.execute(
                """SELECT 1 FROM scheduler_delivery_attempts
                   WHERE notification_id=? AND state='DISPATCHING'""",
                (notification_id,),
            ).fetchone()
            return accepted is None and dispatching is None

    def pending_notification_payloads(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """SELECT event_id FROM encrypted_frames f
                   WHERE f.record_type='scheduler.notification.requested'
                   AND f.state<>'QUARANTINED'
                   ORDER BY f.created_at_utc,f.event_id"""
            ).fetchall()
        payloads: list[dict[str, Any]] = []
        for row in rows:
            payload = self.read(row["event_id"])
            notification_id = payload.get("notification_id")
            if isinstance(notification_id, str) and self.notification_dispatch_needed(notification_id):
                payloads.append(payload)
        return payloads

    def notification_payload(self, notification_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """SELECT event_id FROM encrypted_frames
                   WHERE record_type='scheduler.notification.requested'
                   AND state<>'QUARANTINED' ORDER BY created_at_utc DESC"""
            ).fetchall()
            tombstoned = self._db.execute(
                "SELECT 1 FROM scheduler_tombstones WHERE notification_id=?", (notification_id,)
            ).fetchone() is not None
        if tombstoned:
            return None
        for item in row:
            payload = self.read(item["event_id"])
            if payload.get("notification_id") == notification_id:
                return payload
        return None

    def stage_native_callback(
        self, *, callback_id: str, semantic_sha256: str, payload: Mapping[str, Any]
    ) -> str:
        notification_id = payload.get("notification_id")
        if not isinstance(notification_id, str):
            raise ValueError("callback notification missing")
        status = self.stage(
            event_id=callback_id,
            semantic_sha256=semantic_sha256,
            event_type="scheduler.native.callback",
            payload=payload,
        )
        with self._lock:
            if self._db.execute(
                "SELECT 1 FROM scheduler_tombstones WHERE notification_id=?", (notification_id,)
            ).fetchone() is not None:
                raise ValueError("callback.notification_not_active")
            prior = self._db.execute(
                "SELECT semantic_sha256 FROM scheduler_native_callbacks WHERE callback_id=?",
                (callback_id,),
            ).fetchone()
            if prior is not None:
                if prior["semantic_sha256"] != semantic_sha256:
                    raise ValueError("protocol.conflicting_duplicate")
                return "duplicate"
            self._db.execute(
                """INSERT INTO scheduler_native_callbacks(
                   callback_id,notification_id,frame_event_id,semantic_sha256,state,created_at_utc
                   ) VALUES(?,?,?,?, 'PENDING',?)""",
                (callback_id, notification_id, callback_id, semantic_sha256, self._now()),
            )
        return status

    def pending_native_callbacks(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """SELECT callback_id FROM scheduler_native_callbacks
                   WHERE state='PENDING' ORDER BY created_at_utc,callback_id"""
            ).fetchall()
        return [self.read(row["callback_id"]) for row in rows]

    def mark_native_callback_emitted(self, callback_id: str, scheduler_event_id: str) -> bool:
        with self._lock:
            return self._db.execute(
                """UPDATE scheduler_native_callbacks SET state='EMITTED',scheduler_event_id=?,emitted_at_utc=?
                   WHERE callback_id=? AND state='PENDING'""",
                (scheduler_event_id, self._now(), callback_id),
            ).rowcount == 1

    def reset_emitted_native_callbacks(self) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE scheduler_native_callbacks SET state='PENDING' WHERE state='EMITTED'"
            )

    def native_callback_event_id(self, callback_id: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT scheduler_event_id FROM scheduler_native_callbacks WHERE callback_id=?",
                (callback_id,),
            ).fetchone()
        return None if row is None else row["scheduler_event_id"]

    def close_native_callback(self, scheduler_event_id: str) -> bool:
        with self._lock:
            return self._db.execute(
                """UPDATE scheduler_native_callbacks SET state='CLOSED'
                   WHERE scheduler_event_id=? AND state='EMITTED'""",
                (scheduler_event_id,),
            ).rowcount == 1

    def tombstone(self, notification_id: str, reason_code: str) -> None:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    "INSERT OR IGNORE INTO scheduler_tombstones VALUES(?,?,?)",
                    (notification_id, reason_code, self._now()),
                )
                self._db.execute(
                    """UPDATE scheduler_playback SET state='CANCELLED',updated_at_utc=?
                       WHERE notification_id=? AND state IN ('STAGED','PLAYING')""",
                    (self._now(), notification_id),
                )
                self._db.execute(
                    """UPDATE scheduler_native_callbacks SET state='CLOSED'
                       WHERE notification_id=? AND state IN ('PENDING','EMITTED')""",
                    (notification_id,),
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def set_playback_state(self, notification_id: str, unit_index: int, state: str) -> bool:
        if state not in {"PLAYING", "COMPLETED", "CANCELLED", "SKIPPED"}:
            raise ValueError("invalid scheduler playback state")
        with self._lock:
            if state in {"PLAYING", "COMPLETED"} and self._db.execute(
                "SELECT 1 FROM scheduler_tombstones WHERE notification_id=?", (notification_id,)
            ).fetchone():
                return False
            allowed_previous = {
                "PLAYING": ("STAGED",),
                "COMPLETED": ("PLAYING",),
                "CANCELLED": ("STAGED", "PLAYING"),
                "SKIPPED": ("STAGED", "PLAYING"),
            }[state]
            placeholders = ",".join("?" for _item in allowed_previous)
            return self._db.execute(
                """UPDATE scheduler_playback SET state=?,updated_at_utc=?
                   WHERE notification_id=? AND unit_index=?
                   AND state IN (""" + placeholders + ")",
                (state, self._now(), notification_id, unit_index, *allowed_previous),
            ).rowcount == 1

    def close(self) -> None:
        with self._lock:
            self._db.close()
