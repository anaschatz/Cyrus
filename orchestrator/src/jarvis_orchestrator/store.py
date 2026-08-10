"""SQLite WAL persistence, transactional epoch fences, budgets and outbox."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .domain import (
    ACTIVE_STATES,
    DEFAULT_ABSOLUTE_LIFETIME,
    DEFAULT_APPROVAL_TTL,
    DEFAULT_DELIVERY_TTL,
    DEFAULT_DISCONNECT_TTL,
    DEFAULT_MODEL_REQUESTS,
    DEFAULT_OUTPUT_TOKENS,
    TERMINAL_STATES,
    AdmissionResult,
    ApprovalStatus,
    BudgetExhaustedError,
    CanonicalConversationV1,
    CanonicalEffectV1,
    CanonicalTurnV1,
    ContinuationArtifactV1,
    InvalidStateError,
    LogicalEvent,
    OrchestratorError,
    ReasonCode,
    StaleEpochError,
    TaskSnapshot,
    TaskState,
    canonical_json,
    canonical_sha256,
    new_uuid4,
    nfc,
    parse_utc,
    utc_now,
    utc_text,
)
from .reducer import require_transition

SCHEMA_VERSION = 2


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    state TEXT NOT NULL CHECK(state IN ('ACTIVE','CLOSED','DELETED')),
    current_epoch INTEGER NOT NULL CHECK(current_epoch >= 0),
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_conversation
ON conversations(state) WHERE state='ACTIVE';
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    turn_id TEXT NOT NULL UNIQUE,
    response_id TEXT NOT NULL UNIQUE,
    epoch INTEGER NOT NULL CHECK(epoch >= 0),
    state TEXT NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0 CHECK(tombstoned IN (0,1)),
    admitted_at TEXT NOT NULL,
    absolute_expires_at TEXT NOT NULL,
    active_elapsed_seconds REAL NOT NULL DEFAULT 0 CHECK(active_elapsed_seconds >= 0),
    active_started_at TEXT,
    disconnect_expires_at TEXT,
    delivery_expires_at TEXT,
    terminal_reason TEXT,
    partial_output INTEGER NOT NULL DEFAULT 0 CHECK(partial_output IN (0,1)),
    model_requests_used INTEGER NOT NULL DEFAULT 0 CHECK(model_requests_used >= 0),
    output_tokens_used INTEGER NOT NULL DEFAULT 0 CHECK(output_tokens_used >= 0),
    corrections_used INTEGER NOT NULL DEFAULT 0 CHECK(corrections_used >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS task_by_conversation ON tasks(conversation_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS one_live_task
ON tasks(conversation_id) WHERE tombstoned=0 AND state NOT IN ('COMPLETED','CANCELLED','INTERRUPTED','BUDGET_EXHAUSTED','FAILED');
CREATE TABLE IF NOT EXISTS task_events (
    event_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_memory_shadow_audit_per_task
ON task_events(task_id) WHERE event_type IN (
    'memory.shadow.pending','memory.shadow.proposed','memory.shadow.failed'
);
CREATE TABLE IF NOT EXISTS canonical_turns (
    turn_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
    user_text TEXT NOT NULL,
    user_language TEXT NOT NULL,
    assistant_text TEXT NOT NULL DEFAULT '',
    effects_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL CHECK(status IN ('PENDING','COMPLETED','AUDIT_ONLY')),
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS inbox (
    event_id TEXT PRIMARY KEY,
    semantic_sha256 TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    event_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    task_id TEXT,
    task_epoch INTEGER,
    guard_epoch INTEGER NOT NULL,
    turn_id TEXT,
    response_id TEXT,
    ordinal INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    original_sent_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('REPLAYABLE','CLAIMED','NON_REPLAYABLE','STALE')),
    claimed_by TEXT,
    claim_expires_at TEXT,
    delivery_expires_at TEXT,
    acked_event_id TEXT,
    ack_status TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS response_event_order
ON outbox(response_id, ordinal) WHERE response_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS one_response_started
ON outbox(response_id) WHERE event_type='response.started';
CREATE UNIQUE INDEX IF NOT EXISTS one_response_text_done
ON outbox(response_id) WHERE event_type='response.text.done';
CREATE UNIQUE INDEX IF NOT EXISTS one_response_speech_done
ON outbox(response_id) WHERE event_type='response.speech.done';
CREATE UNIQUE INDEX IF NOT EXISTS one_response_cancelled
ON outbox(response_id) WHERE event_type='response.cancelled';
CREATE UNIQUE INDEX IF NOT EXISTS stable_ack_identity
ON outbox(acked_event_id, ack_status) WHERE acked_event_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS replay_order ON outbox(conversation_id, state, ordinal);
CREATE TABLE IF NOT EXISTS outbox_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL REFERENCES outbox(event_id),
    connection_id TEXT NOT NULL,
    completed INTEGER NOT NULL CHECK(completed IN (0,1)),
    finished_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reservations (
    reservation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    kind TEXT NOT NULL,
    purpose TEXT NOT NULL,
    request_slots INTEGER NOT NULL,
    output_allowance INTEGER NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('RESERVED','DISPATCHED','RECONCILED','CANCELLED')),
    actual_output_tokens INTEGER,
    created_at TEXT NOT NULL,
    dispatched_at TEXT,
    reconciled_at TEXT
);
CREATE INDEX IF NOT EXISTS reservations_by_task ON reservations(task_id, epoch, state);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('research','cloud_model','tool')),
    name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    arguments_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS pending_approvals ON approvals(status, created_at);
CREATE TABLE IF NOT EXISTS continuation_artifacts (
    artifact_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    artifact_json TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_executions (
    execution_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    result_sha256 TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(task_id, epoch, tool_call_id)
);
CREATE TABLE IF NOT EXISTS capability_reservations (
    reservation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('tool','research','cloud_model')),
    request_key TEXT NOT NULL,
    arguments_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('RESERVED','COMPLETED','CANCELLED')),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(task_id,epoch,kind,request_key)
);
CREATE TABLE IF NOT EXISTS idempotency_commands (
    idempotency_key TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    canonical_path TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    expected_epoch INTEGER NOT NULL,
    response_status INTEGER NOT NULL,
    response_json TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
"""


MIGRATION_2_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS one_memory_recall_audit_per_task
ON task_events(task_id) WHERE event_type IN (
    'memory.recall.pending','memory.recall.completed','memory.recall.empty',
    'memory.recall.failed'
);
CREATE TABLE IF NOT EXISTS memory_records (
    record_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    source_event_id TEXT NOT NULL,
    scoped_user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(conversation_id, source_event_id)
);
CREATE INDEX IF NOT EXISTS memory_records_by_scope
ON memory_records(scoped_user_id, created_at, record_id);
"""


class OrchestratorStore:
    def __init__(
        self,
        path: str | Path,
        *,
        model_request_limit: int = DEFAULT_MODEL_REQUESTS,
        output_token_limit: int = DEFAULT_OUTPUT_TOKENS,
        active_budget_seconds: float = 120.0,
        approval_ttl: timedelta = DEFAULT_APPROVAL_TTL,
        disconnect_ttl: timedelta = DEFAULT_DISCONNECT_TTL,
        delivery_ttl: timedelta = DEFAULT_DELIVERY_TTL,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.model_request_limit = model_request_limit
        self.output_token_limit = output_token_limit
        self.active_budget_seconds = active_budget_seconds
        self.approval_ttl = approval_ttl
        self.disconnect_ttl = disconnect_ttl
        self.delivery_ttl = delivery_ttl
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA busy_timeout=2000")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
            existing_version = self._connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()
            if existing_version is not None and existing_version["value"] not in {"1", "2"}:
                raise RuntimeError("unsupported orchestrator database schema version")
            self._connection.executescript(SCHEMA_SQL)
            if existing_version is None:
                self._connection.execute(
                    "INSERT INTO schema_meta(key,value) VALUES('schema_version','1')"
                )
                current_version = 1
            else:
                current_version = int(existing_version["value"])
            if current_version == 1:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + MIGRATION_2_SQL
                    + "\nUPDATE schema_meta SET value='2' WHERE key='schema_version' AND value='1';\n"
                    + "COMMIT;"
                )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=row["task_id"],
            conversation_id=row["conversation_id"],
            turn_id=row["turn_id"],
            response_id=row["response_id"],
            epoch=row["epoch"],
            state=TaskState(row["state"]),
            admitted_at=row["admitted_at"],
            absolute_expires_at=row["absolute_expires_at"],
            active_elapsed_seconds=row["active_elapsed_seconds"],
            active_started_at=row["active_started_at"],
            disconnect_expires_at=row["disconnect_expires_at"],
            delivery_expires_at=row["delivery_expires_at"],
            tombstoned=bool(row["tombstoned"]),
            partial_output=bool(row["partial_output"]),
        )

    def get_task(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            row = self._connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise OrchestratorError("not_found", status=404)
        return self._task_from_row(row)

    def get_active_task(self, conversation_id: str) -> TaskSnapshot | None:
        terminals = tuple(state.value for state in TERMINAL_STATES)
        placeholders = ",".join("?" for _ in terminals)
        with self._lock:
            row = self._connection.execute(
                f"SELECT * FROM tasks WHERE conversation_id=? AND tombstoned=0 AND state NOT IN ({placeholders}) ORDER BY created_at DESC LIMIT 1",
                (conversation_id, *terminals),
            ).fetchone()
        return self._task_from_row(row) if row is not None else None

    def task_input(self, task_id: str) -> dict[str, str]:
        with self._lock:
            row = self._connection.execute(
                """SELECT t.user_text,t.user_language,i.event_id
                   FROM canonical_turns t JOIN inbox i USING(task_id)
                   WHERE t.task_id=?""",
                (task_id,),
            ).fetchone()
        if row is None:
            raise OrchestratorError("not_found", status=404)
        return {
            "text": row["user_text"],
            "language": row["user_language"],
            "event_id": row["event_id"],
        }

    def begin_memory_shadow_audit(
        self,
        task_id: str,
        epoch: int,
        *,
        source_event_id: str,
        text_sha256: str,
        now: datetime | None = None,
    ) -> bool:
        selected = now or utc_now()
        payload = {
            "status": "pending",
            "source_event_id": source_event_id,
            "text_sha256": text_sha256,
        }
        with self._transaction() as connection:
            task = connection.execute(
                "SELECT epoch,state FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if task is None:
                raise OrchestratorError("not_found", status=404)
            if task["epoch"] != epoch or task["state"] != TaskState.COMPLETED.value:
                raise StaleEpochError()
            try:
                self._append_task_event_tx(
                    connection,
                    task_id,
                    epoch,
                    "memory.shadow.pending",
                    payload,
                    selected,
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def finish_memory_shadow_audit(
        self,
        task_id: str,
        epoch: int,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        now: datetime | None = None,
    ) -> None:
        if event_type not in {"memory.shadow.proposed", "memory.shadow.failed"}:
            raise ValueError("memory shadow audit event type is invalid")
        selected = now or utc_now()
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE task_events SET event_type=?,payload_json=?,created_at=?
                   WHERE task_id=? AND epoch=? AND event_type='memory.shadow.pending'""",
                (
                    event_type,
                    canonical_json(payload),
                    utc_text(selected),
                    task_id,
                    epoch,
                ),
            )
            if changed.rowcount != 1:
                raise OrchestratorError("memory_shadow_audit_missing", status=409)

    def memory_shadow_audit(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT epoch,event_type,payload_json,created_at
                   FROM task_events WHERE task_id=? AND event_type IN (
                       'memory.shadow.pending','memory.shadow.proposed',
                       'memory.shadow.failed'
                   )""",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "epoch": row["epoch"],
            "event_type": row["event_type"],
            "payload": json.loads(row["payload_json"]),
            "created_at": row["created_at"],
        }

    def memory_shadow_history(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return proposed shadow events with their canonical source text.

        This is a read-only projection used to reconstruct experimental memory
        state. It excludes pending/failed attempts and does not duplicate source
        text inside the shadow audit payload.
        """

        with self._lock:
            rows = self._connection.execute(
                """SELECT e.event_sequence,e.epoch,e.payload_json,e.created_at,
                          t.user_text,t.user_language,i.event_id
                   FROM task_events e
                   JOIN tasks k USING(task_id)
                   JOIN canonical_turns t USING(task_id)
                   JOIN inbox i USING(task_id)
                   WHERE k.conversation_id=?
                     AND e.event_type='memory.shadow.proposed'
                   ORDER BY e.event_sequence""",
                (conversation_id,),
            ).fetchall()
        return [
            {
                "event_sequence": row["event_sequence"],
                "epoch": row["epoch"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
                "source_text": row["user_text"],
                "source_language": row["user_language"],
                "canonical_source_event_id": row["event_id"],
            }
            for row in rows
        ]

    def begin_memory_recall(
        self,
        task_id: str,
        epoch: int,
        *,
        query_sha256: str,
        now: datetime | None = None,
    ) -> bool:
        """Reserve the one recall attempt for a routing task."""

        selected = now or utc_now()
        payload = {"status": "pending", "query_sha256": query_sha256}
        with self._transaction() as connection:
            self._guarded_task_tx(connection, task_id, epoch, (TaskState.ROUTING,))
            existing = connection.execute(
                """SELECT payload_json FROM task_events WHERE task_id=? AND event_type IN (
                   'memory.recall.pending','memory.recall.completed',
                   'memory.recall.empty','memory.recall.failed')""",
                (task_id,),
            ).fetchone()
            if existing is not None:
                previous = json.loads(existing["payload_json"])
                if previous.get("query_sha256") != query_sha256:
                    raise OrchestratorError("idempotency_conflict", status=409)
                return False
            self._append_task_event_tx(
                connection,
                task_id,
                epoch,
                "memory.recall.pending",
                payload,
                selected,
            )
            return True

    def finish_memory_recall(
        self,
        task_id: str,
        epoch: int,
        *,
        status: str,
        query_sha256: str,
        result_sha256: str = "",
        result_count: int = 0,
        error_code: str = "",
        now: datetime | None = None,
    ) -> None:
        event_types = {
            "completed": "memory.recall.completed",
            "empty": "memory.recall.empty",
            "failed": "memory.recall.failed",
        }
        if status not in event_types:
            raise ValueError("memory recall status is invalid")
        selected = now or utc_now()
        payload = {
            "status": status,
            "query_sha256": query_sha256,
            "result_sha256": result_sha256,
            "result_count": result_count,
            "error_code": error_code,
            "authoritative_state_changed": False,
        }
        with self._transaction() as connection:
            self._guarded_task_tx(connection, task_id, epoch, (TaskState.ROUTING,))
            changed = connection.execute(
                """UPDATE task_events SET event_type=?,payload_json=?,created_at=?
                   WHERE task_id=? AND epoch=? AND event_type='memory.recall.pending'""",
                (
                    event_types[status],
                    canonical_json(payload),
                    utc_text(selected),
                    task_id,
                    epoch,
                ),
            )
            if changed.rowcount != 1:
                raise OrchestratorError("memory_recall_audit_missing", status=409)

    def memory_recall_audit(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT epoch,event_type,payload_json,created_at
                   FROM task_events WHERE task_id=? AND event_type IN (
                       'memory.recall.pending','memory.recall.completed',
                       'memory.recall.empty','memory.recall.failed')""",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "epoch": row["epoch"],
            "event_type": row["event_type"],
            "payload": json.loads(row["payload_json"]),
            "created_at": row["created_at"],
        }

    def add_memory_record(
        self,
        *,
        conversation_id: str,
        task_id: str,
        epoch: int,
        source_event_id: str,
        scoped_user_id: str,
        content: str,
        metadata: Mapping[str, Any],
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Persist one completed-turn memory with task/epoch CAS and stable identity."""

        selected = now or utc_now()
        content_hash = canonical_sha256({"content": content})
        metadata_json = canonical_json(metadata)
        with self._transaction() as connection:
            task = connection.execute(
                """SELECT t.*,c.current_epoch,c.state AS conversation_state,
                          ct.status AS turn_status
                   FROM tasks t JOIN conversations c USING(conversation_id)
                   JOIN canonical_turns ct USING(task_id)
                   WHERE t.task_id=? AND t.conversation_id=?""",
                (task_id, conversation_id),
            ).fetchone()
            if (
                task is None
                or task["epoch"] != epoch
                or task["current_epoch"] != epoch
                or task["state"] != TaskState.COMPLETED.value
                or bool(task["tombstoned"])
                or task["conversation_state"] != "ACTIVE"
                or task["turn_status"] != "COMPLETED"
            ):
                raise StaleEpochError()
            existing = connection.execute(
                """SELECT * FROM memory_records
                   WHERE conversation_id=? AND source_event_id=?""",
                (conversation_id, source_event_id),
            ).fetchone()
            if existing is not None:
                if (
                    existing["content_sha256"] != content_hash
                    or existing["scoped_user_id"] != scoped_user_id
                    or existing["metadata_json"] != metadata_json
                ):
                    raise OrchestratorError("idempotency_conflict", status=409)
                return dict(existing)
            record_id = new_uuid4()
            connection.execute(
                """INSERT INTO memory_records(
                   record_id,conversation_id,task_id,source_event_id,scoped_user_id,
                   content,content_sha256,metadata_json,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    record_id,
                    conversation_id,
                    task_id,
                    source_event_id,
                    scoped_user_id,
                    content,
                    content_hash,
                    metadata_json,
                    utc_text(selected),
                ),
            )
            row = connection.execute(
                "SELECT * FROM memory_records WHERE record_id=?", (record_id,)
            ).fetchone()
            return dict(row)

    def memory_record(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM memory_records WHERE record_id=?", (record_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def memory_records_for_scope(self, scoped_user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM memory_records WHERE scoped_user_id=?
                   ORDER BY created_at DESC,record_id""",
                (scoped_user_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def delete_memory_record(self, record_id: str) -> bool:
        with self._transaction() as connection:
            return connection.execute(
                "DELETE FROM memory_records WHERE record_id=?", (record_id,)
            ).rowcount == 1

    def memory_record_count(self, conversation_id: str) -> int:
        with self._lock:
            return int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM memory_records WHERE conversation_id=?",
                    (conversation_id,),
                ).fetchone()[0]
            )

    def active_conversation(self) -> dict[str, Any] | None:
        with self._lock:
            conversation = self._connection.execute("SELECT * FROM conversations WHERE state='ACTIVE'").fetchone()
            if conversation is None:
                return None
            task = self.get_active_task(conversation["conversation_id"])
        return {
            "conversation_id": conversation["conversation_id"],
            "epoch": conversation["current_epoch"],
            "state": "ACTIVE",
            "active_task_id": task.task_id if task else None,
            "active_task_state": task.state.value if task else None,
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
        }

    def create_conversation(self, expected_epoch: int, *, now: datetime | None = None) -> dict[str, Any]:
        selected = now or utc_now()
        stamp = utc_text(selected)
        conversation_id = new_uuid4()
        with self._transaction() as connection:
            current = connection.execute("SELECT * FROM conversations WHERE state='ACTIVE'").fetchone()
            if current is None:
                if expected_epoch != 0:
                    raise StaleEpochError()
            else:
                if current["current_epoch"] != expected_epoch:
                    raise StaleEpochError()
                if self._active_task_tx(connection, current["conversation_id"]) is not None:
                    raise OrchestratorError("conversation_busy", status=409, retryable=True)
                connection.execute(
                    "UPDATE conversations SET state='CLOSED',updated_at=? WHERE conversation_id=? AND current_epoch=?",
                    (stamp, current["conversation_id"], expected_epoch),
                )
                connection.execute(
                    """UPDATE outbox SET state='NON_REPLAYABLE',claimed_by=NULL,claim_expires_at=NULL
                       WHERE conversation_id=? AND state IN ('REPLAYABLE','CLAIMED')""",
                    (current["conversation_id"],),
                )
            connection.execute(
                "INSERT INTO conversations(conversation_id,state,current_epoch,revision,created_at,updated_at) VALUES(?, 'ACTIVE', 0, 0, ?, ?)",
                (conversation_id, stamp, stamp),
            )
        return {"conversation_id": conversation_id, "epoch": 0, "state": "ACTIVE"}

    @staticmethod
    def _active_task_tx(connection: sqlite3.Connection, conversation_id: str) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM tasks WHERE conversation_id=? AND tombstoned=0 AND state NOT IN ('COMPLETED','CANCELLED','INTERRUPTED','BUDGET_EXHAUSTED','FAILED') ORDER BY created_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()

    @staticmethod
    def transcript_digest(*, event_type: str, turn_id: str, response_id: str | None, payload: Mapping[str, Any]) -> str:
        return canonical_sha256(
            {"type": event_type, "turn_id": turn_id, "response_id": response_id, "payload": dict(payload)}
        )

    @staticmethod
    def _next_ordinal(connection: sqlite3.Connection, conversation_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(ordinal),0)+1 AS next FROM outbox WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        return int(row["next"])

    def _append_outbox_tx(
        self,
        connection: sqlite3.Connection,
        *,
        conversation_id: str,
        guard_epoch: int,
        event_type: str,
        payload: Mapping[str, Any],
        task_id: str | None = None,
        task_epoch: int | None = None,
        turn_id: str | None = None,
        response_id: str | None = None,
        event_id: str | None = None,
        acked_event_id: str | None = None,
        ack_status: str | None = None,
        delivery_expires_at: str | None = None,
        now: datetime | None = None,
    ) -> str:
        logical_id = event_id or new_uuid4()
        connection.execute(
            """INSERT INTO outbox(
                event_id,conversation_id,task_id,task_epoch,guard_epoch,turn_id,response_id,ordinal,
                event_type,payload_json,original_sent_at,state,delivery_expires_at,acked_event_id,ack_status
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,'REPLAYABLE',?,?,?)""",
            (
                logical_id,
                conversation_id,
                task_id,
                task_epoch,
                guard_epoch,
                turn_id,
                response_id,
                self._next_ordinal(connection, conversation_id),
                event_type,
                canonical_json(payload),
                utc_text(now),
                delivery_expires_at,
                acked_event_id,
                ack_status,
            ),
        )
        return logical_id

    def _append_task_event_tx(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        epoch: int,
        event_type: str,
        payload: Mapping[str, Any],
        now: datetime,
    ) -> None:
        connection.execute(
            "INSERT INTO task_events(task_id,epoch,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
            (task_id, epoch, event_type, canonical_json(payload), utc_text(now)),
        )

    def _cancel_task_tx(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        guard_epoch: int,
        state: TaskState,
        reason: ReasonCode,
        now: datetime,
    ) -> None:
        if TaskState(row["state"]) in TERMINAL_STATES:
            return
        elapsed = float(row["active_elapsed_seconds"])
        if row["active_started_at"] is not None and TaskState(row["state"]) in ACTIVE_STATES:
            elapsed += max(0.0, (now - parse_utc(row["active_started_at"])).total_seconds())
        result = connection.execute(
            """UPDATE tasks SET state=?,tombstoned=1,terminal_reason=?,active_elapsed_seconds=?,
               active_started_at=NULL,disconnect_expires_at=NULL,delivery_expires_at=?,updated_at=?
               WHERE task_id=? AND epoch=? AND tombstoned=0""",
            (
                state.value,
                reason.value,
                elapsed,
                utc_text(now + self.delivery_ttl),
                utc_text(now),
                row["task_id"],
                row["epoch"],
            ),
        )
        if result.rowcount != 1:
            raise StaleEpochError()
        connection.execute(
            "UPDATE approvals SET status='CANCELLED',decided_at=? WHERE task_id=? AND epoch=? AND status='PENDING'",
            (utc_text(now), row["task_id"], row["epoch"]),
        )
        connection.execute(
            "UPDATE reservations SET state='CANCELLED',reconciled_at=? WHERE task_id=? AND epoch=? AND state='RESERVED'",
            (utc_text(now), row["task_id"], row["epoch"]),
        )
        connection.execute(
            "UPDATE capability_reservations SET status='CANCELLED',completed_at=? WHERE task_id=? AND epoch=? AND status='RESERVED'",
            (utc_text(now), row["task_id"], row["epoch"]),
        )
        connection.execute(
            "UPDATE outbox SET state='STALE',claimed_by=NULL,claim_expires_at=NULL WHERE response_id=? AND event_type NOT IN ('server.error','response.cancelled','response.speech.done')",
            (row["response_id"],),
        )
        connection.execute(
            "UPDATE canonical_turns SET status='AUDIT_ONLY' WHERE task_id=? AND status='PENDING'",
            (row["task_id"],),
        )
        self._append_task_event_tx(connection, row["task_id"], row["epoch"], state.value, {"reason_code": reason.value}, now)
        self._append_outbox_tx(
            connection,
            conversation_id=row["conversation_id"],
            task_id=row["task_id"],
            task_epoch=row["epoch"],
            guard_epoch=guard_epoch,
            turn_id=row["turn_id"],
            response_id=row["response_id"],
            event_type="response.cancelled",
            payload={"reason_code": reason.value, "partial_output": False},
            delivery_expires_at=utc_text(now + self.delivery_ttl),
            now=now,
        )

    def admit_transcript(
        self,
        *,
        event_id: str,
        turn_id: str,
        payload: Mapping[str, Any],
        now: datetime | None = None,
        absolute_lifetime: timedelta = DEFAULT_ABSOLUTE_LIFETIME,
    ) -> AdmissionResult:
        selected = now or utc_now()
        digest = self.transcript_digest(event_type="input.transcript.final", turn_id=turn_id, response_id=None, payload=payload)
        with self._transaction() as connection:
            existing = connection.execute("SELECT * FROM inbox WHERE event_id=?", (event_id,)).fetchone()
            conversation = connection.execute("SELECT * FROM conversations WHERE state='ACTIVE'").fetchone()
            if conversation is None:
                conversation_id = new_uuid4()
                stamp = utc_text(selected)
                connection.execute(
                    "INSERT INTO conversations(conversation_id,state,current_epoch,revision,created_at,updated_at) VALUES(?,'ACTIVE',0,0,?,?)",
                    (conversation_id, stamp, stamp),
                )
                conversation = connection.execute("SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
            if existing is not None:
                if existing["semantic_sha256"] != digest:
                    raise OrchestratorError("protocol.conflicting_duplicate", status=409)
                ack = connection.execute(
                    "SELECT event_id FROM outbox WHERE acked_event_id=? AND ack_status='duplicate'",
                    (event_id,),
                ).fetchone()
                if ack is None:
                    ack_id = self._append_outbox_tx(
                        connection,
                        conversation_id=conversation["conversation_id"],
                        guard_epoch=conversation["current_epoch"],
                        task_id=existing["task_id"],
                        turn_id=existing["turn_id"],
                        event_type="server.ack",
                        payload={"acked_event_id": event_id, "status": "duplicate"},
                        acked_event_id=event_id,
                        ack_status="duplicate",
                        now=selected,
                    )
                else:
                    ack_id = ack["event_id"]
                return AdmissionResult("duplicate", ack_id, None)

            old = self._active_task_tx(connection, conversation["conversation_id"])
            new_epoch = int(conversation["current_epoch"]) + 1
            updated = connection.execute(
                "UPDATE conversations SET current_epoch=?,updated_at=? WHERE conversation_id=? AND current_epoch=? AND state='ACTIVE'",
                (new_epoch, utc_text(selected), conversation["conversation_id"], conversation["current_epoch"]),
            )
            if updated.rowcount != 1:
                raise StaleEpochError()
            connection.execute(
                "UPDATE outbox SET state='NON_REPLAYABLE',claimed_by=NULL,claim_expires_at=NULL WHERE conversation_id=? AND guard_epoch<? AND state IN ('REPLAYABLE','CLAIMED')",
                (conversation["conversation_id"], new_epoch),
            )
            if old is not None:
                self._cancel_task_tx(
                    connection,
                    old,
                    guard_epoch=new_epoch,
                    state=TaskState.CANCELLED,
                    reason=ReasonCode.SUPERSEDED,
                    now=selected,
                )
            task_id = new_uuid4()
            response_id = new_uuid4()
            stamp = utc_text(selected)
            absolute = utc_text(selected + absolute_lifetime)
            connection.execute(
                """INSERT INTO tasks(
                   task_id,conversation_id,turn_id,response_id,epoch,state,admitted_at,absolute_expires_at,created_at,updated_at
                ) VALUES(?,?,?,?,?,'ADMITTED',?,?,?,?)""",
                (task_id, conversation["conversation_id"], turn_id, response_id, new_epoch, stamp, absolute, stamp, stamp),
            )
            language = payload.get("language", {})
            language_code = language.get("code", "unknown") if isinstance(language, Mapping) else "unknown"
            connection.execute(
                "INSERT INTO canonical_turns(turn_id,conversation_id,task_id,user_text,user_language,status,created_at) VALUES(?,?,?,?,?,'PENDING',?)",
                (turn_id, conversation["conversation_id"], task_id, str(payload.get("text", "")), str(language_code), stamp),
            )
            connection.execute(
                "INSERT INTO inbox(event_id,semantic_sha256,turn_id,task_id,created_at) VALUES(?,?,?,?,?)",
                (event_id, digest, turn_id, task_id, stamp),
            )
            self._append_task_event_tx(connection, task_id, new_epoch, "ADMITTED", {"input_event_id": event_id}, selected)
            ack_id = self._append_outbox_tx(
                connection,
                conversation_id=conversation["conversation_id"],
                task_id=task_id,
                task_epoch=new_epoch,
                guard_epoch=new_epoch,
                turn_id=turn_id,
                event_type="server.ack",
                payload={"acked_event_id": event_id, "status": "accepted"},
                acked_event_id=event_id,
                ack_status="accepted",
                now=selected,
            )
            task_row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return AdmissionResult("accepted", ack_id, self._task_from_row(task_row))

    def transition(
        self,
        task_id: str,
        expected_epoch: int,
        expected_state: TaskState,
        target: TaskState,
        *,
        now: datetime | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> TaskSnapshot:
        require_transition(expected_state, target)
        if target in TERMINAL_STATES:
            raise InvalidStateError()
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT t.*,c.current_epoch FROM tasks t JOIN conversations c USING(conversation_id)
                   WHERE t.task_id=? AND t.epoch=? AND t.state=? AND t.tombstoned=0 AND c.current_epoch=? AND c.state='ACTIVE'""",
                (task_id, expected_epoch, expected_state.value, expected_epoch),
            ).fetchone()
            if row is None:
                raise StaleEpochError()
            elapsed = float(row["active_elapsed_seconds"])
            started_at = row["active_started_at"]
            if expected_state in ACTIVE_STATES and started_at is not None and target not in ACTIVE_STATES:
                elapsed += max(0.0, (selected - parse_utc(started_at)).total_seconds())
                started_at = None
            elif expected_state not in ACTIVE_STATES and target in ACTIVE_STATES:
                started_at = utc_text(selected)
            connection.execute(
                "UPDATE tasks SET state=?,active_elapsed_seconds=?,active_started_at=?,updated_at=? WHERE task_id=? AND epoch=? AND state=?",
                (target.value, elapsed, started_at, utc_text(selected), task_id, expected_epoch, expected_state.value),
            )
            self._append_task_event_tx(connection, task_id, expected_epoch, target.value, payload or {}, selected)
            return self._task_from_row(connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone())

    def _guarded_task_tx(
        self, connection: sqlite3.Connection, task_id: str, epoch: int, states: Sequence[TaskState]
    ) -> sqlite3.Row:
        placeholders = ",".join("?" for _ in states)
        row = connection.execute(
            f"""SELECT t.*,c.current_epoch FROM tasks t JOIN conversations c USING(conversation_id)
                WHERE t.task_id=? AND t.epoch=? AND t.tombstoned=0 AND c.current_epoch=? AND c.state='ACTIVE'
                AND t.state IN ({placeholders})""",
            (task_id, epoch, epoch, *(state.value for state in states)),
        ).fetchone()
        if row is None:
            raise StaleEpochError()
        return row

    def start_response(self, task_id: str, epoch: int, locale: str, *, now: datetime | None = None) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            row = self._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            connection.execute(
                "UPDATE tasks SET state='RESPONDING',updated_at=? WHERE task_id=? AND epoch=? AND state='EXECUTING'",
                (utc_text(selected), task_id, epoch),
            )
            self._append_task_event_tx(connection, task_id, epoch, "RESPONDING", {}, selected)
            self._append_outbox_tx(
                connection,
                conversation_id=row["conversation_id"],
                task_id=task_id,
                task_epoch=epoch,
                guard_epoch=epoch,
                turn_id=row["turn_id"],
                response_id=row["response_id"],
                event_type="response.started",
                payload={"locale": locale},
                now=selected,
            )

    def append_response_event(
        self,
        task_id: str,
        epoch: int,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> str:
        if event_type not in {"response.text.delta", "response.speech.commit"}:
            raise ValueError("only nonterminal response events are accepted")
        selected = now or utc_now()
        with self._transaction() as connection:
            row = self._guarded_task_tx(connection, task_id, epoch, (TaskState.RESPONDING,))
            existing = connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE response_id=? AND event_type=?",
                (row["response_id"], event_type),
            ).fetchone()[0]
            if event_type == "response.text.delta":
                if (
                    not isinstance(payload.get("chunk_index"), int)
                    or isinstance(payload.get("chunk_index"), bool)
                    or payload.get("chunk_index") != existing
                    or not isinstance(payload.get("text"), str)
                    or not payload["text"]
                    or nfc(payload["text"]) != payload["text"]
                ):
                    raise InvalidStateError()
            else:
                if (
                    not isinstance(payload.get("unit_index"), int)
                    or isinstance(payload.get("unit_index"), bool)
                    or payload.get("unit_index") != existing + 1
                    or not isinstance(payload.get("locale"), str)
                    or not payload["locale"]
                    or not isinstance(payload.get("text"), str)
                    or not payload["text"]
                    or nfc(payload["locale"]) != payload["locale"]
                    or nfc(payload["text"]) != payload["text"]
                ):
                    raise InvalidStateError()
            connection.execute("UPDATE tasks SET partial_output=1,updated_at=? WHERE task_id=?", (utc_text(selected), task_id))
            return self._append_outbox_tx(
                connection,
                conversation_id=row["conversation_id"],
                task_id=task_id,
                task_epoch=epoch,
                guard_epoch=epoch,
                turn_id=row["turn_id"],
                response_id=row["response_id"],
                event_type=event_type,
                payload=payload,
                now=selected,
            )

    def complete_response(
        self,
        task_id: str,
        epoch: int,
        assistant_text: str,
        *,
        chunk_count: int,
        trailing_speech: Sequence[tuple[int, str, str]],
        last_unit_index: int,
        now: datetime | None = None,
        delivery_ttl: timedelta | None = None,
    ) -> TaskSnapshot:
        selected = now or utc_now()
        if not isinstance(assistant_text, str) or nfc(assistant_text) != assistant_text:
            raise InvalidStateError()
        delivery = utc_text(selected + (delivery_ttl or self.delivery_ttl))
        with self._transaction() as connection:
            row = self._guarded_task_tx(connection, task_id, epoch, (TaskState.RESPONDING,))
            existing_text = connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE response_id=? AND event_type='response.text.delta'",
                (row["response_id"],),
            ).fetchone()[0]
            existing_speech = connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE response_id=? AND event_type='response.speech.commit'",
                (row["response_id"],),
            ).fetchone()[0]
            if (
                not isinstance(chunk_count, int)
                or isinstance(chunk_count, bool)
                or not isinstance(last_unit_index, int)
                or isinstance(last_unit_index, bool)
                or chunk_count != existing_text
                or last_unit_index != existing_speech + len(trailing_speech)
            ):
                raise InvalidStateError()
            for offset, (unit_index, locale, text) in enumerate(trailing_speech, start=1):
                if (
                    unit_index != existing_speech + offset
                    or not locale
                    or not text
                    or nfc(locale) != locale
                    or nfc(text) != text
                ):
                    raise InvalidStateError()
            self._append_task_event_tx(connection, task_id, epoch, "COMMITTING", {}, selected)
            self._append_outbox_tx(
                connection,
                conversation_id=row["conversation_id"], task_id=task_id, task_epoch=epoch, guard_epoch=epoch,
                turn_id=row["turn_id"], response_id=row["response_id"], event_type="response.text.done",
                payload={"chunk_count": chunk_count}, delivery_expires_at=delivery, now=selected,
            )
            for unit_index, locale, text in trailing_speech:
                self._append_outbox_tx(
                    connection,
                    conversation_id=row["conversation_id"], task_id=task_id, task_epoch=epoch, guard_epoch=epoch,
                    turn_id=row["turn_id"], response_id=row["response_id"], event_type="response.speech.commit",
                    payload={"unit_index": unit_index, "locale": locale, "text": text}, delivery_expires_at=delivery, now=selected,
                )
            self._append_outbox_tx(
                connection,
                conversation_id=row["conversation_id"], task_id=task_id, task_epoch=epoch, guard_epoch=epoch,
                turn_id=row["turn_id"], response_id=row["response_id"], event_type="response.speech.done",
                payload={"last_unit_index": last_unit_index}, delivery_expires_at=delivery, now=selected,
            )
            elapsed = float(row["active_elapsed_seconds"])
            if row["active_started_at"] is not None:
                elapsed += max(0.0, (selected - parse_utc(row["active_started_at"])).total_seconds())
            updated = connection.execute(
                """UPDATE tasks SET state='COMPLETED',active_elapsed_seconds=?,active_started_at=NULL,
                   delivery_expires_at=?,disconnect_expires_at=NULL,updated_at=?
                   WHERE task_id=? AND epoch=? AND state='RESPONDING' AND tombstoned=0""",
                (elapsed, delivery, utc_text(selected), task_id, epoch),
            )
            if updated.rowcount != 1:
                raise StaleEpochError()
            connection.execute(
                "UPDATE outbox SET delivery_expires_at=? WHERE task_id=? AND state='REPLAYABLE'",
                (delivery, task_id),
            )
            connection.execute(
                "UPDATE canonical_turns SET assistant_text=?,status='COMPLETED',completed_at=? WHERE task_id=? AND status='PENDING'",
                (assistant_text, utc_text(selected), task_id),
            )
            connection.execute(
                "UPDATE conversations SET revision=revision+1,updated_at=? WHERE conversation_id=? AND current_epoch=?",
                (utc_text(selected), row["conversation_id"], epoch),
            )
            self._append_task_event_tx(connection, task_id, epoch, "COMPLETED", {}, selected)
            return self._task_from_row(connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone())

    def cancel_task(
        self,
        task_id: str,
        expected_epoch: int,
        reason: ReasonCode,
        *,
        terminal_state: TaskState = TaskState.CANCELLED,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT t.*,c.current_epoch FROM tasks t JOIN conversations c USING(conversation_id)
                   WHERE task_id=? AND t.epoch=? AND c.current_epoch=? AND c.state='ACTIVE'""",
                (task_id, expected_epoch, expected_epoch),
            ).fetchone()
            if row is None:
                raise StaleEpochError()
            if TaskState(row["state"]) in TERMINAL_STATES:
                raise InvalidStateError()
            new_epoch = expected_epoch + 1
            changed = connection.execute(
                "UPDATE conversations SET current_epoch=?,updated_at=? WHERE conversation_id=? AND current_epoch=?",
                (new_epoch, utc_text(selected), row["conversation_id"], expected_epoch),
            )
            if changed.rowcount != 1:
                raise StaleEpochError()
            self._cancel_task_tx(connection, row, guard_epoch=new_epoch, state=terminal_state, reason=reason, now=selected)
            return {"task_id": task_id, "previous_epoch": expected_epoch, "epoch": new_epoch, "state": terminal_state.value}

    def terminate_failure(
        self,
        task_id: str,
        expected_epoch: int,
        *,
        state: TaskState,
        reason: ReasonCode,
        error_code: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if state not in {TaskState.BUDGET_EXHAUSTED, TaskState.FAILED}:
            raise ValueError("failure state is invalid")
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT t.*,c.current_epoch FROM tasks t JOIN conversations c USING(conversation_id)
                   WHERE task_id=? AND t.epoch=? AND c.current_epoch=? AND c.state='ACTIVE'""",
                (task_id, expected_epoch, expected_epoch),
            ).fetchone()
            if row is None:
                raise StaleEpochError()
            if TaskState(row["state"]) in TERMINAL_STATES:
                raise InvalidStateError()
            new_epoch = expected_epoch + 1
            changed = connection.execute(
                "UPDATE conversations SET current_epoch=?,updated_at=? WHERE conversation_id=? AND current_epoch=?",
                (new_epoch, utc_text(selected), row["conversation_id"], expected_epoch),
            )
            if changed.rowcount != 1:
                raise StaleEpochError()
            if not bool(row["partial_output"]):
                self._append_outbox_tx(
                    connection,
                    conversation_id=row["conversation_id"],
                    task_id=task_id,
                    task_epoch=expected_epoch,
                    guard_epoch=new_epoch,
                    turn_id=row["turn_id"],
                    response_id=row["response_id"],
                    event_type="server.error",
                    payload={"code": error_code, "retryable": False, "terminal": True},
                    delivery_expires_at=utc_text(selected + self.delivery_ttl),
                    now=selected,
                )
            self._cancel_task_tx(connection, row, guard_epoch=new_epoch, state=state, reason=reason, now=selected)
            return {"task_id": task_id, "previous_epoch": expected_epoch, "epoch": new_epoch, "state": state.value}

    def reserve_model(
        self,
        task_id: str,
        epoch: int,
        *,
        purpose: str,
        output_allowance: int,
        now: datetime | None = None,
    ) -> str:
        if output_allowance < 1:
            raise ValueError("output allowance must be positive")
        selected = now or utc_now()
        with self._transaction() as connection:
            row = self._guarded_task_tx(
                connection, task_id, epoch, (TaskState.ROUTING, TaskState.EXECUTING)
            )
            if parse_utc(row["absolute_expires_at"]) <= selected:
                raise BudgetExhaustedError("absolute_lifetime_exhausted")
            active_elapsed = float(row["active_elapsed_seconds"])
            if row["active_started_at"] is not None:
                active_elapsed += max(
                    0.0,
                    (selected - parse_utc(row["active_started_at"])).total_seconds(),
                )
            if active_elapsed >= self.active_budget_seconds:
                raise BudgetExhaustedError("active_budget_exhausted")
            if int(row["model_requests_used"]) >= self.model_request_limit:
                raise BudgetExhaustedError()
            outstanding = connection.execute(
                "SELECT COALESCE(SUM(output_allowance),0) AS total FROM reservations WHERE task_id=? AND epoch=? AND state IN ('RESERVED','DISPATCHED')",
                (task_id, epoch),
            ).fetchone()["total"]
            if int(row["output_tokens_used"]) + int(outstanding) + output_allowance > self.output_token_limit:
                raise BudgetExhaustedError()
            reservation_id = new_uuid4()
            connection.execute(
                """INSERT INTO reservations(reservation_id,task_id,epoch,kind,purpose,request_slots,output_allowance,state,created_at)
                   VALUES(?,?,?,'model',?,1,?,'RESERVED',?)""",
                (reservation_id, task_id, epoch, purpose, output_allowance, utc_text(selected)),
            )
            connection.execute(
                "UPDATE tasks SET model_requests_used=model_requests_used+1,updated_at=? WHERE task_id=? AND epoch=?",
                (utc_text(selected), task_id, epoch),
            )
            self._append_task_event_tx(connection, task_id, epoch, "MODEL_RESERVED", {"reservation_id": reservation_id, "purpose": purpose, "output_allowance": output_allowance}, selected)
            return reservation_id

    def remaining_model_budget(self, task_id: str, epoch: int) -> dict[str, int]:
        with self._lock:
            row = self._connection.execute(
                "SELECT model_requests_used,output_tokens_used FROM tasks WHERE task_id=? AND epoch=?",
                (task_id, epoch),
            ).fetchone()
            if row is None:
                raise OrchestratorError("not_found", status=404)
            outstanding = self._connection.execute(
                "SELECT COALESCE(SUM(output_allowance),0) AS total FROM reservations WHERE task_id=? AND epoch=? AND state IN ('RESERVED','DISPATCHED')",
                (task_id, epoch),
            ).fetchone()["total"]
        return {
            "requests": max(0, self.model_request_limit - int(row["model_requests_used"])),
            "output_tokens": max(0, self.output_token_limit - int(row["output_tokens_used"]) - int(outstanding)),
        }

    def mark_correction(self, task_id: str, epoch: int, *, now: datetime | None = None) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            row = self._guarded_task_tx(
                connection, task_id, epoch, (TaskState.ROUTING, TaskState.EXECUTING)
            )
            if int(row["corrections_used"]) >= 1:
                raise BudgetExhaustedError()
            connection.execute(
                "UPDATE tasks SET corrections_used=corrections_used+1,updated_at=? WHERE task_id=? AND epoch=?",
                (utc_text(selected), task_id, epoch),
            )
            self._append_task_event_tx(
                connection, task_id, epoch, "CORRECTION_REQUIRED", {}, selected
            )

    def mark_model_dispatched(self, reservation_id: str, task_id: str, epoch: int, *, now: datetime | None = None) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            self._guarded_task_tx(
                connection, task_id, epoch, (TaskState.ROUTING, TaskState.EXECUTING)
            )
            result = connection.execute(
                "UPDATE reservations SET state='DISPATCHED',dispatched_at=? WHERE reservation_id=? AND task_id=? AND epoch=? AND state='RESERVED'",
                (utc_text(selected), reservation_id, task_id, epoch),
            )
            if result.rowcount != 1:
                raise StaleEpochError()

    def reconcile_model(
        self,
        reservation_id: str,
        task_id: str,
        epoch: int,
        actual_output_tokens: int | None,
        *,
        now: datetime | None = None,
    ) -> int:
        selected = now or utc_now()
        with self._transaction() as connection:
            reservation = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id=? AND task_id=? AND epoch=?",
                (reservation_id, task_id, epoch),
            ).fetchone()
            if reservation is None or reservation["state"] not in {"DISPATCHED", "RESERVED"}:
                raise StaleEpochError()
            consumed = int(reservation["output_allowance"]) if actual_output_tokens is None else max(0, min(int(actual_output_tokens), int(reservation["output_allowance"])))
            connection.execute(
                "UPDATE reservations SET state='RECONCILED',actual_output_tokens=?,reconciled_at=? WHERE reservation_id=?",
                (consumed, utc_text(selected), reservation_id),
            )
            live = connection.execute(
                """SELECT 1 FROM tasks t JOIN conversations c USING(conversation_id)
                   WHERE t.task_id=? AND t.epoch=? AND t.tombstoned=0
                   AND c.current_epoch=? AND c.state='ACTIVE'""",
                (task_id, epoch, epoch),
            ).fetchone()
            if live is not None:
                connection.execute(
                    "UPDATE tasks SET output_tokens_used=output_tokens_used+?,updated_at=? WHERE task_id=?",
                    (consumed, utc_text(selected), task_id),
                )
                self._append_task_event_tx(connection, task_id, epoch, "MODEL_RECONCILED", {"reservation_id": reservation_id, "output_tokens": consumed}, selected)
            return consumed

    def create_approval(
        self,
        task_id: str,
        epoch: int,
        *,
        kind: str,
        name: str,
        arguments: Mapping[str, Any],
        now: datetime | None = None,
        ttl: timedelta | None = None,
    ) -> str:
        if kind not in {"research", "cloud_model", "tool"}:
            raise ValueError("invalid approval kind")
        selected = now or utc_now()
        with self._transaction() as connection:
            row = self._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            expires = min(selected + (ttl or self.approval_ttl), parse_utc(row["absolute_expires_at"]))
            approval_id = new_uuid4()
            args_json = canonical_json(arguments)
            connection.execute(
                """INSERT INTO approvals(approval_id,conversation_id,task_id,epoch,kind,name,arguments_json,arguments_sha256,status,created_at,expires_at)
                   VALUES(?,?,?,?,?,?,?,?, 'PENDING',?,?)""",
                (approval_id, row["conversation_id"], task_id, epoch, kind, name, args_json, canonical_sha256(arguments), utc_text(selected), utc_text(expires)),
            )
            elapsed = float(row["active_elapsed_seconds"])
            if row["active_started_at"] is not None:
                elapsed += max(0.0, (selected - parse_utc(row["active_started_at"])).total_seconds())
            connection.execute(
                "UPDATE tasks SET state='AWAITING_APPROVAL',active_elapsed_seconds=?,active_started_at=NULL,disconnect_expires_at=NULL,updated_at=? WHERE task_id=? AND epoch=? AND state='EXECUTING'",
                (elapsed, utc_text(selected), task_id, epoch),
            )
            self._append_task_event_tx(connection, task_id, epoch, "AWAITING_APPROVAL", {"approval_id": approval_id}, selected)
            return approval_id

    def list_pending_approvals(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        selected = now or utc_now()
        self.expire_approvals(now=selected)
        with self._lock:
            rows = self._connection.execute("SELECT * FROM approvals WHERE status='PENDING' ORDER BY created_at,approval_id").fetchall()
        return [
            {
                "approval_id": row["approval_id"],
                "conversation_id": row["conversation_id"],
                "task_id": row["task_id"],
                "epoch": row["epoch"],
                "kind": row["kind"],
                "name": row["name"],
                "arguments": json.loads(row["arguments_json"]),
                "arguments_sha256": row["arguments_sha256"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
                "status": "PENDING",
            }
            for row in rows
        ]

    def decide_approval(
        self,
        approval_id: str,
        expected_epoch: int,
        decision: str,
        *,
        client_connected: bool,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if decision not in {"approve", "deny"}:
            raise OrchestratorError("invalid_request", status=400)
        selected = now or utc_now()
        with self._lock:
            before = self._connection.execute(
                "SELECT epoch,expires_at,status FROM approvals WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
        if before is not None and before["status"] == "PENDING" and parse_utc(before["expires_at"]) <= selected:
            self.expire_approval(approval_id, now=selected)
            raise OrchestratorError("approval_expired", status=410)
        with self._transaction() as connection:
            approval = connection.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
            if approval is None:
                raise OrchestratorError("not_found", status=404)
            if approval["epoch"] != expected_epoch:
                raise StaleEpochError()
            if approval["status"] != "PENDING":
                raise InvalidStateError()
            task = self._guarded_task_tx(connection, approval["task_id"], expected_epoch, (TaskState.AWAITING_APPROVAL,))
            approval_status = ApprovalStatus.APPROVED if decision == "approve" else ApprovalStatus.DENIED
            target = TaskState.EXECUTING if client_connected else TaskState.DECISION_WAITING_RECONNECT
            reconnect_deadline = None if client_connected else utc_text(selected + self.disconnect_ttl)
            connection.execute(
                "UPDATE approvals SET status=?,decided_at=? WHERE approval_id=? AND status='PENDING'",
                (approval_status.value, utc_text(selected), approval_id),
            )
            connection.execute(
                "UPDATE tasks SET state=?,active_started_at=?,disconnect_expires_at=?,updated_at=? WHERE task_id=? AND epoch=? AND state='AWAITING_APPROVAL'",
                (target.value, utc_text(selected) if client_connected else None, reconnect_deadline, utc_text(selected), task["task_id"], expected_epoch),
            )
            turn = connection.execute("SELECT effects_json FROM canonical_turns WHERE task_id=?", (task["task_id"],)).fetchone()
            effects = json.loads(turn["effects_json"])
            effects.append({
                "kind": "ToolDenied" if approval_status == ApprovalStatus.DENIED else approval["kind"],
                "name": approval["name"],
                "status": approval_status.value,
                "arguments_sha256": approval["arguments_sha256"],
                "result_sha256": "",
            })
            connection.execute("UPDATE canonical_turns SET effects_json=? WHERE task_id=?", (canonical_json(effects), task["task_id"]))
            self._append_task_event_tx(connection, task["task_id"], expected_epoch, target.value, {"approval_id": approval_id, "decision": decision}, selected)
            return {"approval_id": approval_id, "task_id": task["task_id"], "epoch": expected_epoch, "approval_status": approval_status.value, "task_state": target.value}

    def expire_approval(self, approval_id: str, *, now: datetime | None = None) -> bool:
        selected = now or utc_now()
        with self._transaction() as connection:
            approval = connection.execute(
                "SELECT * FROM approvals WHERE approval_id=? AND status='PENDING' AND expires_at<=?",
                (approval_id, utc_text(selected)),
            ).fetchone()
            if approval is None:
                return False
            task = connection.execute(
                """SELECT t.*,c.current_epoch FROM tasks t JOIN conversations c USING(conversation_id)
                   WHERE t.task_id=? AND t.epoch=? AND t.tombstoned=0
                   AND c.current_epoch=? AND c.state='ACTIVE'""",
                (approval["task_id"], approval["epoch"], approval["epoch"]),
            ).fetchone()
            if task is None:
                connection.execute(
                    "UPDATE approvals SET status='EXPIRED',decided_at=? WHERE approval_id=?",
                    (utc_text(selected), approval_id),
                )
                return True
            new_epoch = int(approval["epoch"]) + 1
            changed = connection.execute(
                "UPDATE conversations SET current_epoch=?,updated_at=? WHERE conversation_id=? AND current_epoch=?",
                (new_epoch, utc_text(selected), task["conversation_id"], approval["epoch"]),
            )
            if changed.rowcount != 1:
                raise StaleEpochError()
            connection.execute(
                "UPDATE approvals SET status='EXPIRED',decided_at=? WHERE approval_id=?",
                (utc_text(selected), approval_id),
            )
            self._cancel_task_tx(
                connection,
                task,
                guard_epoch=new_epoch,
                state=TaskState.CANCELLED,
                reason=ReasonCode.APPROVAL_EXPIRED,
                now=selected,
            )
            return True

    def expire_approvals(self, *, now: datetime | None = None) -> int:
        selected = now or utc_now()
        with self._lock:
            rows = self._connection.execute("SELECT approval_id FROM approvals WHERE status='PENDING' AND expires_at<=?", (utc_text(selected),)).fetchall()
        count = 0
        for row in rows:
            try:
                if self.expire_approval(row["approval_id"], now=selected):
                    count += 1
            except OrchestratorError:
                continue
        return count

    def set_disconnected(self, task_id: str, epoch: int, *, now: datetime | None = None, ttl: timedelta | None = None) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            row = self._guarded_task_tx(connection, task_id, epoch, tuple(ACTIVE_STATES | {TaskState.AWAITING_APPROVAL, TaskState.DECISION_WAITING_RECONNECT}))
            if TaskState(row["state"]) == TaskState.AWAITING_APPROVAL:
                return
            connection.execute("UPDATE tasks SET disconnect_expires_at=?,updated_at=? WHERE task_id=? AND epoch=?", (utc_text(selected + (ttl or self.disconnect_ttl)), utc_text(selected), task_id, epoch))

    def set_reconnected(self, task_id: str, epoch: int, *, now: datetime | None = None) -> TaskSnapshot:
        selected = now or utc_now()
        with self._transaction() as connection:
            row = self._guarded_task_tx(connection, task_id, epoch, tuple(ACTIVE_STATES | {TaskState.AWAITING_APPROVAL, TaskState.DECISION_WAITING_RECONNECT}))
            state = TaskState(row["state"])
            target = TaskState.EXECUTING if state == TaskState.DECISION_WAITING_RECONNECT else state
            active_started = utc_text(selected) if target == TaskState.EXECUTING and state != TaskState.EXECUTING else row["active_started_at"]
            connection.execute("UPDATE tasks SET state=?,active_started_at=?,disconnect_expires_at=NULL,updated_at=? WHERE task_id=? AND epoch=?", (target.value, active_started, utc_text(selected), task_id, epoch))
            if target != state:
                self._append_task_event_tx(connection, task_id, epoch, target.value, {"reconnected": True}, selected)
            return self._task_from_row(connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone())

    def expire_deadlines(self, *, now: datetime | None = None) -> int:
        selected = now or utc_now()
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM tasks WHERE tombstoned=0
                   AND state NOT IN ('COMPLETED','CANCELLED','INTERRUPTED','BUDGET_EXHAUSTED','FAILED')""",
            ).fetchall()
        count = 0
        for row in rows:
            absolute = parse_utc(row["absolute_expires_at"]) <= selected
            state = TaskState(row["state"])
            disconnected = row["disconnect_expires_at"] is not None and parse_utc(
                row["disconnect_expires_at"]
            ) <= selected
            active_elapsed = float(row["active_elapsed_seconds"])
            if row["active_started_at"] is not None and state in ACTIVE_STATES:
                active_elapsed += max(
                    0.0,
                    (selected - parse_utc(row["active_started_at"])).total_seconds(),
                )
            active_exhausted = active_elapsed >= self.active_budget_seconds
            if not (absolute or disconnected or active_exhausted):
                continue
            if absolute:
                reason = ReasonCode.ABSOLUTE_LIFETIME_EXHAUSTED
            elif active_exhausted:
                reason = ReasonCode.ACTIVE_BUDGET_EXHAUSTED
            else:
                reason = (
                    ReasonCode.APPROVAL_RECONNECT_TIMEOUT
                    if state == TaskState.DECISION_WAITING_RECONNECT
                    else ReasonCode.CLIENT_DISCONNECTED
                )
            terminal = (
                TaskState.BUDGET_EXHAUSTED
                if absolute or active_exhausted
                else TaskState.CANCELLED
            )
            try:
                self.cancel_task(row["task_id"], row["epoch"], reason, terminal_state=terminal, now=selected)
                count += 1
            except OrchestratorError:
                continue
        self.expire_delivery(now=selected)
        self.expire_commands(now=selected)
        return count

    def claim_outbox(
        self,
        conversation_id: str,
        connection_id: str,
        *,
        now: datetime | None = None,
        lease: timedelta = timedelta(seconds=1),
        exclude_event_ids: Sequence[str] = (),
        limit: int = 128,
    ) -> list[LogicalEvent]:
        selected = now or utc_now()
        with self._transaction() as connection:
            connection.execute(
                "UPDATE outbox SET state='REPLAYABLE',claimed_by=NULL,claim_expires_at=NULL WHERE state='CLAIMED' AND claim_expires_at<=?",
                (utc_text(selected),),
            )
            excluded = ""
            parameters: list[Any] = [conversation_id, utc_text(selected)]
            if exclude_event_ids:
                excluded = f" AND event_id NOT IN ({','.join('?' for _ in exclude_event_ids)})"
                parameters.extend(exclude_event_ids)
            parameters.append(limit)
            rows = connection.execute(
                f"""SELECT o.* FROM outbox o JOIN conversations c USING(conversation_id)
                    WHERE o.conversation_id=? AND o.state='REPLAYABLE'
                    AND (o.delivery_expires_at IS NULL OR o.delivery_expires_at>?)
                    AND o.guard_epoch=c.current_epoch AND c.state='ACTIVE'{excluded}
                    ORDER BY o.ordinal LIMIT ?""",
                tuple(parameters),
            ).fetchall()
            claimed: list[sqlite3.Row] = []
            for row in rows:
                result = connection.execute(
                    "UPDATE outbox SET state='CLAIMED',claimed_by=?,claim_expires_at=? WHERE event_id=? AND state='REPLAYABLE' AND guard_epoch=(SELECT current_epoch FROM conversations WHERE conversation_id=?)",
                    (connection_id, utc_text(selected + lease), row["event_id"], conversation_id),
                )
                if result.rowcount == 1:
                    claimed.append(row)
        return [
            LogicalEvent(
                event_id=row["event_id"], conversation_id=row["conversation_id"], task_id=row["task_id"], task_epoch=row["task_epoch"], guard_epoch=row["guard_epoch"],
                turn_id=row["turn_id"], response_id=row["response_id"], ordinal=row["ordinal"], event_type=row["event_type"], payload=json.loads(row["payload_json"]),
                original_sent_at=row["original_sent_at"], delivery_expires_at=row["delivery_expires_at"],
            )
            for row in claimed
        ]

    def finish_outbox_attempt(self, event_id: str, connection_id: str, *, complete: bool) -> None:
        stale = False
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM outbox WHERE event_id=?", (event_id,)).fetchone() is None:
                raise StaleEpochError()
            connection.execute(
                "INSERT INTO outbox_attempts(event_id,connection_id,completed,finished_at) VALUES(?,?,?,?)",
                (event_id, connection_id, int(complete), utc_text()),
            )
            target = "REPLAYABLE"
            result = connection.execute(
                "UPDATE outbox SET state=?,claimed_by=NULL,claim_expires_at=NULL WHERE event_id=? AND state='CLAIMED' AND claimed_by=?",
                (target, event_id, connection_id),
            )
            if result.rowcount != 1:
                stale = True
        if stale:
            raise StaleEpochError()

    def outbox_claim_is_current(
        self,
        event_id: str,
        connection_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        selected = now or utc_now()
        with self._lock:
            row = self._connection.execute(
                """SELECT 1 FROM outbox o JOIN conversations c USING(conversation_id)
                   WHERE o.event_id=? AND o.state='CLAIMED' AND o.claimed_by=?
                   AND o.guard_epoch=c.current_epoch AND c.state='ACTIVE'
                   AND (o.delivery_expires_at IS NULL OR o.delivery_expires_at>?)""",
                (event_id, connection_id, utc_text(selected)),
            ).fetchone()
        return row is not None

    def retire_response_delivery(self, response_id: str) -> None:
        with self._transaction() as connection:
            connection.execute("UPDATE outbox SET state='NON_REPLAYABLE',claimed_by=NULL,claim_expires_at=NULL WHERE response_id=? AND state IN ('REPLAYABLE','CLAIMED')", (response_id,))

    def expire_delivery(self, *, now: datetime | None = None) -> int:
        selected = now or utc_now()
        with self._transaction() as connection:
            result = connection.execute(
                "UPDATE outbox SET state='NON_REPLAYABLE',claimed_by=NULL,claim_expires_at=NULL WHERE delivery_expires_at IS NOT NULL AND delivery_expires_at<=? AND state IN ('REPLAYABLE','CLAIMED')",
                (utc_text(selected),),
            )
            return result.rowcount

    def canonical_conversation(self, conversation_id: str) -> CanonicalConversationV1:
        with self._lock:
            conversation = self._connection.execute("SELECT * FROM conversations WHERE conversation_id=? AND state!='DELETED'", (conversation_id,)).fetchone()
            if conversation is None:
                raise OrchestratorError("not_found", status=404)
            rows = self._connection.execute("SELECT * FROM canonical_turns WHERE conversation_id=? AND status='COMPLETED' ORDER BY created_at,turn_id", (conversation_id,)).fetchall()
        turns: list[CanonicalTurnV1] = []
        for row in rows:
            effects = tuple(CanonicalEffectV1(**item) for item in json.loads(row["effects_json"]))
            turns.append(CanonicalTurnV1(row["turn_id"], row["user_text"], row["user_language"], row["assistant_text"], effects))
        return CanonicalConversationV1(conversation_id, conversation["revision"], tuple(turns))

    def save_continuation(self, artifact: ContinuationArtifactV1) -> None:
        if artifact.artifact_version != 1:
            raise OrchestratorError("continuation_incompatible", status=409)
        document = asdict(artifact)
        selected = parse_utc(artifact.created_at)
        with self._transaction() as connection:
            task = self._guarded_task_tx(
                connection,
                artifact.task_id,
                artifact.epoch,
                (TaskState.AWAITING_APPROVAL,),
            )
            if task["conversation_id"] != artifact.conversation_id:
                raise OrchestratorError("continuation_incompatible", status=409)
            canonical = self.canonical_conversation(artifact.conversation_id)
            canonical_document = asdict(canonical)
            if canonical.revision != artifact.canonical_revision or canonical_sha256(canonical_document) != artifact.canonical_sha256:
                raise OrchestratorError("continuation_incompatible", status=409)
            connection.execute(
                """INSERT INTO continuation_artifacts(
                   artifact_id,task_id,epoch,artifact_json,artifact_sha256,created_at,expires_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (
                    artifact.artifact_id,
                    artifact.task_id,
                    artifact.epoch,
                    canonical_json(document),
                    canonical_sha256(document),
                    utc_text(selected),
                    artifact.expires_at,
                ),
            )

    def load_continuation(
        self,
        artifact_id: str,
        *,
        task_id: str,
        epoch: int,
        model_digest: str,
        profile_digest: str,
        tool_schema_digest: str,
        principal_sha256: str,
        policy_version: str,
        arguments_sha256: str,
        now: datetime | None = None,
    ) -> ContinuationArtifactV1:
        selected = now or utc_now()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM continuation_artifacts WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
            task = self._connection.execute(
                """SELECT t.*,c.current_epoch,c.state AS conversation_state
                   FROM tasks t JOIN conversations c USING(conversation_id)
                   WHERE t.task_id=?""",
                (task_id,),
            ).fetchone()
        incompatible = row is None or task is None
        if not incompatible:
            try:
                document = json.loads(row["artifact_json"])
                artifact = ContinuationArtifactV1(**document)
            except (TypeError, ValueError, json.JSONDecodeError):
                incompatible = True
            else:
                canonical = self.canonical_conversation(artifact.conversation_id)
                incompatible = any(
                    (
                        row["artifact_sha256"] != canonical_sha256(document),
                        artifact.artifact_version != 1,
                        artifact.task_id != task_id,
                        artifact.epoch != epoch,
                        task["epoch"] != epoch,
                        task["current_epoch"] != epoch,
                        bool(task["tombstoned"]),
                        task["conversation_state"] != "ACTIVE",
                        parse_utc(artifact.expires_at) <= selected,
                        artifact.canonical_revision != canonical.revision,
                        artifact.canonical_sha256 != canonical_sha256(asdict(canonical)),
                        artifact.model_digest != model_digest,
                        artifact.profile_digest != profile_digest,
                        artifact.tool_schema_digest != tool_schema_digest,
                        artifact.principal_sha256 != principal_sha256,
                        artifact.policy_version != policy_version,
                        artifact.arguments_sha256 != arguments_sha256,
                    )
                )
        if incompatible:
            raise OrchestratorError("continuation_incompatible", status=409)
        return artifact

    def reserve_capability(
        self,
        task_id: str,
        epoch: int,
        *,
        kind: str,
        request_key: str,
        arguments: Mapping[str, Any],
        now: datetime | None = None,
    ) -> str:
        limits = {"tool": 4, "research": 2, "cloud_model": 0}
        if kind not in limits:
            raise ValueError("invalid capability kind")
        selected = now or utc_now()
        with self._transaction() as connection:
            self._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            existing = connection.execute(
                "SELECT * FROM capability_reservations WHERE task_id=? AND epoch=? AND kind=? AND request_key=?",
                (task_id, epoch, kind, request_key),
            ).fetchone()
            digest = canonical_sha256(arguments)
            if existing is not None:
                if existing["arguments_sha256"] != digest:
                    raise OrchestratorError("idempotency_conflict", status=409)
                return existing["reservation_id"]
            used = connection.execute(
                "SELECT COUNT(*) FROM capability_reservations WHERE task_id=? AND epoch=? AND kind=? AND status!='CANCELLED'",
                (task_id, epoch, kind),
            ).fetchone()[0]
            if used >= limits[kind]:
                raise BudgetExhaustedError()
            reservation_id = new_uuid4()
            connection.execute(
                """INSERT INTO capability_reservations(
                   reservation_id,task_id,epoch,kind,request_key,arguments_sha256,status,created_at
                   ) VALUES(?,?,?,?,?,?,'RESERVED',?)""",
                (reservation_id, task_id, epoch, kind, request_key, digest, utc_text(selected)),
            )
            return reservation_id

    def reconcile_capability(
        self,
        reservation_id: str,
        task_id: str,
        epoch: int,
        *,
        effect: CanonicalEffectV1,
        now: datetime | None = None,
    ) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            self._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            reservation = connection.execute(
                "SELECT * FROM capability_reservations WHERE reservation_id=? AND task_id=? AND epoch=?",
                (reservation_id, task_id, epoch),
            ).fetchone()
            if reservation is None or reservation["status"] != "RESERVED":
                raise StaleEpochError()
            turn = connection.execute(
                "SELECT effects_json FROM canonical_turns WHERE task_id=? AND status='PENDING'",
                (task_id,),
            ).fetchone()
            if turn is None:
                raise StaleEpochError()
            effects = json.loads(turn["effects_json"])
            effects.append(asdict(effect))
            connection.execute(
                "UPDATE canonical_turns SET effects_json=? WHERE task_id=?",
                (canonical_json(effects), task_id),
            )
            connection.execute(
                "UPDATE capability_reservations SET status='COMPLETED',completed_at=? WHERE reservation_id=?",
                (utc_text(selected), reservation_id),
            )

    def recover_after_restart(self, *, now: datetime | None = None) -> int:
        selected = now or utc_now()
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM tasks WHERE tombstoned=0 AND state IN ('ROUTING','LOADING_CONTEXT','EXECUTING','RESPONDING','COMMITTING')"
            ).fetchall()
        count = 0
        for row in rows:
            try:
                self.cancel_task(row["task_id"], row["epoch"], ReasonCode.SERVICE_INTERRUPTED, terminal_state=TaskState.INTERRUPTED, now=selected)
                count += 1
            except OrchestratorError:
                continue
        return count

    def lookup_command(self, key: str, method: str, path: str, body: Any, expected_epoch: int, *, now: datetime | None = None) -> tuple[int, str | None] | None:
        selected = now or utc_now()
        digest = canonical_sha256(body)
        with self._lock:
            row = self._connection.execute("SELECT * FROM idempotency_commands WHERE idempotency_key=? AND expires_at>?", (key, utc_text(selected))).fetchone()
        if row is None:
            return None
        if row["method"] != method or row["canonical_path"] != path or row["request_sha256"] != digest or row["expected_epoch"] != expected_epoch:
            raise OrchestratorError("idempotency_conflict", status=409)
        return row["response_status"], row["response_json"]

    def store_command(self, key: str, method: str, path: str, body: Any, expected_epoch: int, status: int, response_json: str | None, *, now: datetime | None = None) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO idempotency_commands(idempotency_key,method,canonical_path,request_sha256,expected_epoch,response_status,response_json,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (key, method, path, canonical_sha256(body), expected_epoch, status, response_json, utc_text(selected), utc_text(selected + timedelta(hours=24))),
            )

    def expire_commands(self, *, now: datetime | None = None) -> int:
        with self._transaction() as connection:
            return connection.execute("DELETE FROM idempotency_commands WHERE expires_at<=?", (utc_text(now or utc_now()),)).rowcount

    def delete_conversation(self, conversation_id: str, expected_epoch: int, *, now: datetime | None = None) -> None:
        del now
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM conversations WHERE conversation_id=? AND state!='DELETED'", (conversation_id,)).fetchone()
            if row is None:
                raise OrchestratorError("not_found", status=404)
            if row["current_epoch"] != expected_epoch:
                raise StaleEpochError()
            if self._active_task_tx(connection, conversation_id) is not None:
                raise OrchestratorError("conversation_busy", status=409, retryable=True)
            task_rows = connection.execute(
                "SELECT task_id FROM tasks WHERE conversation_id=?", (conversation_id,)
            ).fetchall()
            task_ids = [item["task_id"] for item in task_rows]
            if task_ids:
                placeholders = ",".join("?" for _ in task_ids)
                for table in (
                    "continuation_artifacts",
                    "tool_executions",
                    "capability_reservations",
                    "approvals",
                    "reservations",
                    "task_events",
                    "inbox",
                ):
                    connection.execute(
                        f"DELETE FROM {table} WHERE task_id IN ({placeholders})",
                        tuple(task_ids),
                    )
            connection.execute(
                "DELETE FROM outbox_attempts WHERE event_id IN (SELECT event_id FROM outbox WHERE conversation_id=?)",
                (conversation_id,),
            )
            connection.execute(
                "DELETE FROM memory_records WHERE conversation_id=?", (conversation_id,)
            )
            connection.execute("DELETE FROM outbox WHERE conversation_id=?", (conversation_id,))
            connection.execute("DELETE FROM canonical_turns WHERE conversation_id=?", (conversation_id,))
            connection.execute("DELETE FROM tasks WHERE conversation_id=?", (conversation_id,))
            connection.execute(
                "DELETE FROM conversations WHERE conversation_id=? AND current_epoch=?",
                (conversation_id, expected_epoch),
            )

    def checkpoint(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
