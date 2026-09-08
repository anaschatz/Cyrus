"""Bounded persisted deduplication for replayed server events."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

from .errors import ProtocolError
from .private_state import prepare_voice_journal_path
from .protocol import Envelope


def default_delivery_journal_path() -> Path:
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA", "").strip()
        if not root:
            raise ProtocolError("delivery_journal.state_root_missing", "LOCALAPPDATA is required")
        return Path(root) / "JARVIS Voice" / "state" / "delivery-dedup.sqlite3"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "JARVIS Voice" / "state" / "delivery-dedup.sqlite3"
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "jarvis-voice" / "delivery-dedup.sqlite3"


def semantic_fingerprint(event: Envelope) -> str:
    value = {
        "v": event.v,
        "type": event.type,
        "event_id": event.event_id,
        "sent_at": event.sent_at,
        "turn_id": event.turn_id,
        "response_id": event.response_id,
        "payload": dict(event.payload),
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DeliveryDedupJournal:
    def __init__(
        self,
        path: str | Path,
        *,
        ttl_seconds: float = 60.0,
        maximum_rows: int = 4096,
        clock=time.time,
        trusted_boundary: str | Path | None = None,
    ) -> None:
        if ttl_seconds <= 0 or maximum_rows < 1:
            raise ValueError("journal limits must be positive")
        self.path = prepare_voice_journal_path(
            path,
            trusted_boundary=trusted_boundary,
        )
        self.ttl_seconds = ttl_seconds
        self.maximum_rows = maximum_rows
        self.clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        try:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS delivery_dedup(
                    event_id TEXT PRIMARY KEY,
                    semantic_sha256 TEXT NOT NULL,
                    response_id TEXT,
                    expires_at REAL NOT NULL
                )"""
            )
            # SQLite may have materialized WAL/SHM after the first preparation.
            # Re-audit them before the journal can receive an event.
            prepare_voice_journal_path(
                self.path,
                trusted_boundary=trusted_boundary,
            )
            self.prune()
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def prune(self) -> int:
        with self._lock:
            return self._connection.execute("DELETE FROM delivery_dedup WHERE expires_at<=?", (self.clock(),)).rowcount

    def observe(self, event: Envelope) -> bool:
        """Persist before side effects; return True for an exact replay."""

        now = self.clock()
        digest = semantic_fingerprint(event)
        expires = now + self.ttl_seconds
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute("DELETE FROM delivery_dedup WHERE expires_at<=?", (now,))
                existing = self._connection.execute("SELECT * FROM delivery_dedup WHERE event_id=?", (event.event_id,)).fetchone()
                if existing is not None:
                    if existing[1] != digest:
                        raise ProtocolError("protocol.conflicting_duplicate", "Server event identity changed during replay")
                    self._connection.execute("UPDATE delivery_dedup SET expires_at=? WHERE event_id=?", (expires, event.event_id))
                    self._connection.execute("COMMIT")
                    return True
                count = self._connection.execute("SELECT COUNT(*) FROM delivery_dedup").fetchone()[0]
                if count >= self.maximum_rows:
                    raise ProtocolError("delivery_journal.capacity_exceeded", "The live delivery journal is full")
                self._connection.execute(
                    "INSERT INTO delivery_dedup(event_id,semantic_sha256,response_id,expires_at) VALUES(?,?,?,?)",
                    (event.event_id, digest, event.response_id, expires),
                )
                self._connection.execute("COMMIT")
                return False
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
