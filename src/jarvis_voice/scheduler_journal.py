"""Encrypted durable scheduler inbox and isolated playback journal for Voice."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


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
"""


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
            self._db.execute(
                """INSERT OR IGNORE INTO scheduler_playback(
                   notification_id,unit_index,frame_event_id,state,updated_at_utc
                   ) VALUES(?,?,?,'STAGED',?)""",
                (notification_id, unit_index, frame_event_id, self._now()),
            )
            return True

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
        if state not in {"OS_ACCEPTED", "OS_REJECTED"}:
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
            return self._db.execute(
                """UPDATE scheduler_playback SET state=?,updated_at_utc=?
                   WHERE notification_id=? AND unit_index=?""",
                (state, self._now(), notification_id, unit_index),
            ).rowcount == 1

    def close(self) -> None:
        with self._lock:
            self._db.close()
