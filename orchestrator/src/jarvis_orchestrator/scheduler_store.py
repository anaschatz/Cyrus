"""Schema-v5 scheduler repository and deterministic state transitions."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from .domain import OrchestratorError, TaskState, new_uuid4, parse_utc, require_uuid4, utc_now, utc_text
from .jcs import domain_sha256, jcs_text
from .scheduler_contract import (
    APPROVAL_TTL_SECONDS,
    CONTENT_RETENTION_MS,
    MAXIMUM_ACTIVE_JOBS,
    MAXIMUM_PAGE_SIZE,
    MAXIMUM_QUERY_HORIZON_DAYS,
    MAXIMUM_SNOOZE_GENERATIONS,
    MINIMUM_LEAD_MS,
    MISSED_GRACE_MS,
    CompiledProposal,
    PackagedTimezoneProvider,
    ScheduleContractError,
    ensure_minimum_lead,
    next_calendar_local,
    notification_id,
    occurrence_id,
    strict_utc,
)
from .scheduler_crypto import (ContentBinding, EncryptedContent, SchedulerContentCipher,
                               SchedulerCryptoError)


SCHEDULER_PROTOCOL = "jarvis.scheduler/1"
APPROVAL_ARBITER_PROTOCOL = "jarvis.approval-arbiter/1"
MAXIMUM_APPROVAL_QUEUE = 64


MIGRATION_5_SQL = """
CREATE TABLE IF NOT EXISTS scheduler_content (
    record_id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL,
    job_id TEXT,
    job_revision INTEGER NOT NULL CHECK(job_revision >= 0),
    key_version INTEGER NOT NULL CHECK(key_version >= 1),
    nonce BLOB NOT NULL CHECK(length(nonce)=12),
    ciphertext_and_tag BLOB NOT NULL,
    content_sha256 TEXT NOT NULL,
    aad_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    purge_after TEXT,
    purged_at TEXT,
    UNIQUE(key_version,nonce)
);
CREATE TABLE IF NOT EXISTS approval_arbiter_queue (
    arbiter_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    approval_id TEXT NOT NULL UNIQUE,
    producer_protocol TEXT NOT NULL,
    binding_sha256 TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN (
        'WAITING_FOR_CHANNEL','PRESENTING','PRESENTED','DECIDING','WAITING_FOR_OWNER',
        'RESOLVED','EXPIRED','INVALIDATED'
    )),
    owner_session_id TEXT,
    presentation_id TEXT,
    requested_event_id TEXT UNIQUE,
    decision_nonce TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_global_approval_channel
ON approval_arbiter_queue((1)) WHERE state IN (
    'PRESENTING','PRESENTED','DECIDING','WAITING_FOR_OWNER'
);
CREATE INDEX IF NOT EXISTS approval_arbiter_fifo
ON approval_arbiter_queue(state,arbiter_seq,approval_id);
CREATE TABLE IF NOT EXISTS scheduler_proposals (
    proposal_id TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL UNIQUE,
    proposal_sha256 TEXT NOT NULL UNIQUE,
    operation TEXT NOT NULL CHECK(operation IN ('CREATE_JOB','CANCEL_JOB','SNOOZE_OCCURRENCE')),
    state TEXT NOT NULL CHECK(state IN ('PENDING','APPROVED','DENIED','EXPIRED','SUPERSEDED')),
    first_due_at_utc TEXT,
    target_job_id TEXT,
    target_job_revision INTEGER,
    target_occurrence_id TEXT,
    content_record_id TEXT NOT NULL REFERENCES scheduler_content(record_id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS scheduler_proposals_by_state
ON scheduler_proposals(state,created_at,proposal_id);
CREATE TABLE IF NOT EXISTS scheduler_jobs (
    job_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL UNIQUE REFERENCES scheduler_proposals(proposal_id),
    job_revision INTEGER NOT NULL CHECK(job_revision >= 1),
    state TEXT NOT NULL CHECK(state IN ('ACTIVE','DRAINING','COMPLETED','CANCELLED')),
    schedule_json TEXT NOT NULL,
    schedule_sha256 TEXT NOT NULL,
    content_record_id TEXT NOT NULL REFERENCES scheduler_content(record_id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    terminal_at TEXT,
    content_purge_after TEXT,
    UNIQUE(job_id,job_revision)
);
CREATE INDEX IF NOT EXISTS scheduler_jobs_order
ON scheduler_jobs(created_at DESC,job_id DESC);
CREATE TABLE IF NOT EXISTS scheduler_recurrence_cursors (
    job_id TEXT PRIMARY KEY REFERENCES scheduler_jobs(job_id),
    job_revision INTEGER NOT NULL,
    next_sequence INTEGER NOT NULL CHECK(next_sequence >= 0),
    next_scheduled_at_utc TEXT,
    next_local_slot TEXT,
    exhausted INTEGER NOT NULL CHECK(exhausted IN (0,1)),
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scheduler_occurrences (
    occurrence_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES scheduler_jobs(job_id),
    job_revision INTEGER NOT NULL,
    sequence INTEGER NOT NULL CHECK(sequence >= 0),
    scheduled_at_utc TEXT NOT NULL,
    effective_due_at_utc TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN (
        'SCHEDULED','FIRED','SETTLED','MISSED_COALESCED','MISSED_EXPIRED','CANCELLED'
    )),
    snooze_generation INTEGER NOT NULL DEFAULT 0 CHECK(snooze_generation BETWEEN 0 AND 8),
    action_deadline_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(job_id,job_revision,sequence),
    UNIQUE(job_id,job_revision,scheduled_at_utc)
);
CREATE INDEX IF NOT EXISTS scheduler_occurrences_due
ON scheduler_occurrences(state,effective_due_at_utc,job_id,occurrence_id);
CREATE TABLE IF NOT EXISTS scheduler_deferrals (
    occurrence_id TEXT NOT NULL REFERENCES scheduler_occurrences(occurrence_id),
    generation INTEGER NOT NULL CHECK(generation BETWEEN 1 AND 8),
    prior_effective_due_at_utc TEXT NOT NULL,
    snooze_until_at_utc TEXT NOT NULL,
    cause_event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(occurrence_id,generation)
);
CREATE TABLE IF NOT EXISTS scheduler_deliveries (
    notification_id TEXT PRIMARY KEY,
    occurrence_id TEXT NOT NULL REFERENCES scheduler_occurrences(occurrence_id),
    delivery_generation INTEGER NOT NULL CHECK(delivery_generation BETWEEN 0 AND 8),
    state TEXT NOT NULL CHECK(state IN (
        'SCHEDULED','OUTBOXED','STAGED','DISPATCHING','RETRY_WAIT','BLOCKED_CAPABILITY',
        'BLOCKED_KEY_UNAVAILABLE','OS_ACCEPTED','EXPIRED_UNDELIVERED','REVOKED'
    )),
    effective_due_at_utc TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    content_record_id TEXT REFERENCES scheduler_content(record_id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(occurrence_id,delivery_generation)
);
CREATE INDEX IF NOT EXISTS scheduler_deliveries_due
ON scheduler_deliveries(state,effective_due_at_utc,notification_id);
CREATE TABLE IF NOT EXISTS scheduler_delivery_attempts (
    attempt_id TEXT PRIMARY KEY,
    notification_id TEXT NOT NULL REFERENCES scheduler_deliveries(notification_id),
    attempt_ordinal INTEGER NOT NULL CHECK(attempt_ordinal >= 1),
    state TEXT NOT NULL CHECK(state IN (
        'CREATED','DISPATCHING','OS_ACCEPTED','OS_REJECTED','AMBIGUOUS_OS_DISPATCH'
    )),
    reason_code TEXT NOT NULL,
    created_at TEXT NOT NULL,
    observed_at TEXT,
    UNIQUE(notification_id,attempt_ordinal)
);
CREATE TABLE IF NOT EXISTS scheduler_inbox (
    event_id TEXT PRIMARY KEY,
    semantic_sha256 TEXT NOT NULL,
    protocol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    content_record_id TEXT REFERENCES scheduler_content(record_id),
    state TEXT NOT NULL CHECK(state IN ('RECEIVED','APPLIED','QUARANTINED')),
    created_at TEXT NOT NULL,
    applied_at TEXT
);
CREATE TABLE IF NOT EXISTS scheduler_outbox (
    event_id TEXT PRIMARY KEY,
    protocol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entity_id TEXT,
    semantic_sha256 TEXT NOT NULL,
    content_record_id TEXT NOT NULL REFERENCES scheduler_content(record_id),
    ordinal INTEGER NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('PENDING','LEASED','ACKED','REVOKED','EXPIRED')),
    original_sent_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    lease_owner TEXT,
    lease_expires_at TEXT,
    ack_status TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS scheduler_outbox_replay
ON scheduler_outbox(state,ordinal,event_id);
CREATE TABLE IF NOT EXISTS scheduler_ledger (
    ledger_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    cause_event_id TEXT,
    attempt_id TEXT,
    semantic_sha256 TEXT NOT NULL,
    occurred_at_utc TEXT NOT NULL,
    clock_observed_at_utc TEXT NOT NULL,
    worker_boot_id TEXT,
    monotonic_lease_deadline_ns INTEGER,
    clock_jump_detected INTEGER NOT NULL DEFAULT 0 CHECK(clock_jump_detected IN (0,1)),
    metadata_json TEXT NOT NULL,
    previous_event_hash TEXT,
    event_hash TEXT NOT NULL UNIQUE
);
INSERT OR IGNORE INTO approval_arbiter_queue(
    approval_id,producer_protocol,binding_sha256,state,owner_session_id,
    presentation_id,requested_event_id,decision_nonce,created_at,expires_at,updated_at
)
SELECT b.approval_id,'jarvis.cloud-approval/1',b.binding_sha256,'WAITING_FOR_CHANNEL',
       p.owner_session_id,p.presentation_id,p.requested_event_id,p.decision_nonce,
       b.created_at,b.expires_at,b.created_at
FROM cloud_approval_bindings b
JOIN approvals a USING(approval_id)
LEFT JOIN cloud_presentations p ON p.approval_id=b.approval_id AND p.state='PENDING'
WHERE a.status='PENDING';
INSERT INTO schema_meta(key,value) VALUES('scheduler_schema_version','5')
ON CONFLICT(key) DO UPDATE SET value='5';
"""


_TERMINAL_JOB_STATES = frozenset({"COMPLETED", "CANCELLED"})
_TERMINAL_DELIVERY_STATES = frozenset({"OS_ACCEPTED", "EXPIRED_UNDELIVERED", "REVOKED"})
_ACTIVE_ARBITER_STATES = ("PRESENTING", "PRESENTED", "DECIDING", "WAITING_FOR_OWNER")


class SchedulerStore:
    """Scheduler repository sharing the Orchestrator's SQLite transaction lock."""

    def __init__(
        self,
        orchestrator_store: Any,
        *,
        cipher: SchedulerContentCipher,
        zones: PackagedTimezoneProvider,
    ) -> None:
        self._store = orchestrator_store
        self._connection = orchestrator_store._connection
        self._lock = orchestrator_store._lock
        self._transaction = orchestrator_store._transaction
        self.cipher = cipher
        self.zones = zones

    def recover_after_restart(self, *, now: datetime | None = None) -> dict[str, int]:
        selected = now or utc_now()
        with self._transaction() as connection:
            attempts = connection.execute(
                "SELECT attempt_id,notification_id FROM scheduler_delivery_attempts WHERE state='DISPATCHING'"
            ).fetchall()
            for attempt in attempts:
                connection.execute(
                    """UPDATE scheduler_delivery_attempts SET state='AMBIGUOUS_OS_DISPATCH',
                       reason_code='restart_without_native_receipt',observed_at=? WHERE attempt_id=?""",
                    (utc_text(selected), attempt["attempt_id"]),
                )
                connection.execute(
                    "UPDATE scheduler_deliveries SET state='RETRY_WAIT',updated_at=? WHERE notification_id=? AND state='DISPATCHING'",
                    (utc_text(selected), attempt["notification_id"]),
                )
                self._ledger_tx(
                    connection,
                    entity_type="delivery",
                    entity_id=attempt["notification_id"],
                    event_type="scheduler.delivery.ambiguous_os_dispatch",
                    from_state="DISPATCHING",
                    to_state="RETRY_WAIT",
                    now=selected,
                    attempt_id=attempt["attempt_id"],
                    metadata={"reason_code": "restart_without_native_receipt"},
                )
            outbox = connection.execute(
                """UPDATE scheduler_outbox SET state='PENDING',lease_owner=NULL,lease_expires_at=NULL
                   WHERE state='LEASED'"""
            ).rowcount
            approvals = connection.execute(
                """UPDATE approval_arbiter_queue SET state='WAITING_FOR_OWNER',updated_at=?
                   WHERE state IN ('PRESENTING','PRESENTED')""",
                (utc_text(selected),),
            ).rowcount
            return {"ambiguous_attempt_count": len(attempts),
                    "replayed_outbox_count": outbox, "waiting_owner_approval_count": approvals}

    @staticmethod
    def _sha(value: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise OrchestratorError("invalid_request", status=400)

    def _put_content_tx(
        self,
        connection: sqlite3.Connection,
        *,
        record_type: str,
        record_id: str,
        job_id: str | None,
        job_revision: int,
        value: Mapping[str, Any],
        now: datetime,
        purge_after: datetime | None = None,
    ) -> str:
        plaintext = jcs_text(value).encode("utf-8")
        encrypted = self.cipher.encrypt(
            plaintext,
            ContentBinding(
                store_role="orchestrator",
                record_type=record_type,
                record_id=record_id,
                job_id=job_id,
                job_revision=job_revision,
            ),
        )
        try:
            connection.execute(
                """INSERT INTO scheduler_content(
                   record_id,record_type,job_id,job_revision,key_version,nonce,
                   ciphertext_and_tag,content_sha256,aad_sha256,created_at,purge_after
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record_id,
                    record_type,
                    job_id,
                    job_revision,
                    encrypted.key_version,
                    encrypted.nonce,
                    encrypted.ciphertext_and_tag,
                    encrypted.content_sha256,
                    encrypted.aad_sha256,
                    utc_text(now),
                    utc_text(purge_after) if purge_after else None,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise OrchestratorError("scheduler_content_conflict", status=409) from exc
        return record_id

    def _get_content_tx(self, connection: sqlite3.Connection, record_id: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT * FROM scheduler_content WHERE record_id=? AND purged_at IS NULL",
            (record_id,),
        ).fetchone()
        if row is None:
            raise OrchestratorError("scheduler_content_unavailable", status=503)
        encrypted = EncryptedContent(
            key_version=row["key_version"],
            nonce=bytes(row["nonce"]),
            ciphertext_and_tag=bytes(row["ciphertext_and_tag"]),
            content_sha256=row["content_sha256"],
            aad_sha256=row["aad_sha256"],
        )
        plaintext = self.cipher.decrypt(
            encrypted,
            ContentBinding(
                store_role="orchestrator",
                record_type=row["record_type"],
                record_id=row["record_id"],
                job_id=row["job_id"],
                job_revision=row["job_revision"],
            ),
        )
        value = json.loads(plaintext)
        if not isinstance(value, dict):
            raise OrchestratorError("scheduler_content_invalid", status=503)
        return value

    @staticmethod
    def _ledger_tx(
        connection: sqlite3.Connection,
        *,
        entity_type: str,
        entity_id: str,
        event_type: str,
        from_state: str | None,
        to_state: str | None,
        now: datetime,
        cause_event_id: str | None = None,
        attempt_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        worker_boot_id: str | None = None,
        monotonic_lease_deadline_ns: int | None = None,
        clock_jump_detected: bool = False,
    ) -> int:
        previous = connection.execute(
            "SELECT event_hash FROM scheduler_ledger ORDER BY ledger_seq DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous["event_hash"] if previous else None
        metadata_value = dict(metadata or {})
        semantic = {
            "attempt_id": attempt_id,
            "cause_event_id": cause_event_id,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "event_type": event_type,
            "from_state": from_state,
            "metadata": metadata_value,
            "occurred_at_utc": utc_text(now),
            "previous_event_hash": previous_hash,
            "to_state": to_state,
        }
        semantic_sha256 = domain_sha256("scheduler-ledger-event/1", semantic)
        event_hash = domain_sha256(
            "scheduler-ledger-chain/1",
            {"event": semantic, "previous_event_hash": previous_hash},
        )
        cursor = connection.execute(
            """INSERT INTO scheduler_ledger(
               entity_type,entity_id,event_type,from_state,to_state,cause_event_id,
               attempt_id,semantic_sha256,occurred_at_utc,clock_observed_at_utc,
               worker_boot_id,monotonic_lease_deadline_ns,clock_jump_detected,
               metadata_json,previous_event_hash,event_hash
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                entity_type,
                entity_id,
                event_type,
                from_state,
                to_state,
                cause_event_id,
                attempt_id,
                semantic_sha256,
                utc_text(now),
                utc_text(now),
                worker_boot_id,
                monotonic_lease_deadline_ns,
                int(clock_jump_detected),
                jcs_text(metadata_value),
                previous_hash,
                event_hash,
            ),
        )
        return int(cursor.lastrowid)

    def _append_outbox_tx(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        expires_at: datetime,
        now: datetime,
        event_id: str | None = None,
        entity_id: str | None = None,
    ) -> str:
        selected_event_id = event_id or new_uuid4()
        record_id = new_uuid4()
        self._put_content_tx(
            connection,
            record_type="SERVER_OUTBOX_PAYLOAD",
            record_id=record_id,
            job_id=payload.get("job_id") if isinstance(payload.get("job_id"), str) else None,
            job_revision=int(payload.get("job_revision", 0) or 0),
            value=dict(payload),
            now=now,
            purge_after=expires_at + timedelta(milliseconds=CONTENT_RETENTION_MS),
        )
        ordinal = connection.execute(
            "SELECT COALESCE(MAX(ordinal),0)+1 FROM scheduler_outbox"
        ).fetchone()[0]
        semantic = domain_sha256(
            "scheduler-wire-semantic/1",
            {
                "event_id": selected_event_id,
                "payload": dict(payload),
                "protocol": SCHEDULER_PROTOCOL,
                "sent_at": utc_text(now),
                "type": event_type,
            },
        )
        connection.execute(
            """INSERT INTO scheduler_outbox(
               event_id,protocol,event_type,entity_id,semantic_sha256,content_record_id,ordinal,
               state,original_sent_at,expires_at,created_at
               ) VALUES(?,?,?,?,?,?,?,'PENDING',?,?,?)""",
            (
                selected_event_id,
                SCHEDULER_PROTOCOL,
                event_type,
                entity_id,
                semantic,
                record_id,
                ordinal,
                utc_text(now),
                utc_text(expires_at),
                utc_text(now),
            ),
        )
        return selected_event_id

    def _api_result_tx(
        self, connection: sqlite3.Connection, event_id: str, semantic_sha256: str
    ) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT semantic_sha256,content_record_id,state FROM scheduler_inbox WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        if row["semantic_sha256"] != semantic_sha256:
            raise OrchestratorError("protocol.conflicting_duplicate", status=409)
        if row["state"] == "QUARANTINED" or row["content_record_id"] is None:
            raise OrchestratorError("protocol.conflicting_duplicate", status=409)
        return self._get_content_tx(connection, row["content_record_id"])

    def _existing_inbox(
        self, event_id: str, semantic_sha256: str
    ) -> tuple[bool, dict[str, Any] | None]:
        with self._lock:
            row = self._connection.execute(
                "SELECT semantic_sha256,content_record_id,state FROM scheduler_inbox WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if row is None:
                return False, None
            conflict = row["semantic_sha256"] != semantic_sha256 or row["state"] == "QUARANTINED"
            content_record_id = row["content_record_id"]
        if conflict:
            with self._transaction() as connection:
                connection.execute(
                    "UPDATE scheduler_inbox SET state='QUARANTINED' WHERE event_id=?", (event_id,)
                )
            raise OrchestratorError("protocol.conflicting_duplicate", status=409)
        if content_record_id is None:
            return True, None
        with self._lock:
            return True, self._get_content_tx(self._connection, content_record_id)

    def lookup_api_result(self, event_id: str, semantic_sha256: str) -> dict[str, Any] | None:
        require_uuid4(event_id)
        self._sha(semantic_sha256)
        exists, result = self._existing_inbox(event_id, semantic_sha256)
        return result if exists else None

    def _record_api_result_tx(
        self,
        connection: sqlite3.Connection,
        *,
        event_id: str,
        semantic_sha256: str,
        event_type: str,
        status: int,
        result: Mapping[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        prior = self._api_result_tx(connection, event_id, semantic_sha256)
        if prior is not None:
            return prior
        record_id = new_uuid4()
        wrapper = {"result": dict(result), "status": status}
        self._put_content_tx(
            connection,
            record_type="CONTROL_API_RESULT",
            record_id=record_id,
            job_id=None,
            job_revision=0,
            value=wrapper,
            now=now,
            purge_after=now + timedelta(milliseconds=CONTENT_RETENTION_MS),
        )
        connection.execute(
            """INSERT INTO scheduler_inbox(
               event_id,semantic_sha256,protocol,event_type,content_record_id,state,created_at,applied_at
               ) VALUES(?,?,?,?,?,'APPLIED',?,?)""",
            (
                event_id,
                semantic_sha256,
                SCHEDULER_PROTOCOL,
                event_type,
                record_id,
                utc_text(now),
                utc_text(now),
            ),
        )
        return wrapper

    def store_api_result(
        self,
        *,
        event_id: str,
        semantic_sha256: str,
        event_type: str,
        status: int,
        result: Mapping[str, Any],
        now: datetime | None = None,
    ) -> dict[str, Any]:
        require_uuid4(event_id)
        self._sha(semantic_sha256)
        replay_exists, replay = self._existing_inbox(event_id, semantic_sha256)
        if replay_exists:
            if replay is None:
                raise OrchestratorError("scheduler_content_unavailable", status=503)
            return replay
        selected = now or utc_now()
        with self._transaction() as connection:
            return self._record_api_result_tx(
                connection,
                event_id=event_id,
                semantic_sha256=semantic_sha256,
                event_type=event_type,
                status=status,
                result=result,
                now=selected,
            )

    def create_proposal(
        self,
        compiled: CompiledProposal,
        *,
        task_id: str | None = None,
        epoch: int | None = None,
        now: datetime | None = None,
        api_event_id: str | None = None,
        api_semantic_sha256: str | None = None,
    ) -> dict[str, Any]:
        if (task_id is None) != (epoch is None):
            raise ValueError("task id and epoch must be supplied together")
        if (api_event_id is None) != (api_semantic_sha256 is None):
            raise ValueError("api event id and semantic hash must be supplied together")
        if api_event_id is not None:
            require_uuid4(api_event_id)
            self._sha(str(api_semantic_sha256))
            replay_exists, replay = self._existing_inbox(
                api_event_id, str(api_semantic_sha256)
            )
            if replay_exists:
                if replay is None:
                    raise OrchestratorError("scheduler_content_unavailable", status=503)
                return dict(replay["result"])
        selected = now or utc_now()
        operation = str(compiled.proposal["operation"])
        if operation not in {"CREATE_JOB", "CANCEL_JOB", "SNOOZE_OCCURRENCE"}:
            raise OrchestratorError("invalid_request", status=400)
        proposal_id = new_uuid4()
        approval_id = new_uuid4()
        expires = selected + timedelta(seconds=APPROVAL_TTL_SECONDS)
        with self._transaction() as connection:
            task = None
            if task_id is not None:
                task = self._store._guarded_task_tx(
                    connection, task_id, int(epoch), (TaskState.EXECUTING,)
                )
            if api_event_id is not None:
                prior = self._api_result_tx(connection, api_event_id, str(api_semantic_sha256))
                if prior is not None:
                    return dict(prior["result"])
            queue_count = connection.execute(
                """SELECT COUNT(*) FROM approval_arbiter_queue
                   WHERE state IN ('WAITING_FOR_CHANNEL','PRESENTING','PRESENTED','DECIDING','WAITING_FOR_OWNER')"""
            ).fetchone()[0]
            if queue_count >= MAXIMUM_APPROVAL_QUEUE:
                raise OrchestratorError("APPROVAL_QUEUE_FULL", status=409)
            if operation == "CREATE_JOB":
                active_count = connection.execute(
                    "SELECT COUNT(*) FROM scheduler_jobs WHERE state IN ('ACTIVE','DRAINING')"
                ).fetchone()[0]
                if active_count >= MAXIMUM_ACTIVE_JOBS:
                    raise OrchestratorError("maximum_active_jobs", status=409)
            record_id = new_uuid4()
            self._put_content_tx(
                connection,
                record_type="PROPOSAL_PREVIEW",
                record_id=record_id,
                job_id=compiled.proposal.get("target_job_id"),
                job_revision=int(compiled.proposal.get("target_job_revision") or 0),
                value=dict(compiled.proposal),
                now=selected,
                purge_after=expires + timedelta(milliseconds=CONTENT_RETENTION_MS),
            )
            connection.execute(
                """INSERT INTO scheduler_proposals(
                   proposal_id,approval_id,proposal_sha256,operation,state,first_due_at_utc,
                   target_job_id,target_job_revision,target_occurrence_id,content_record_id,
                   created_at,expires_at
                   ) VALUES(?,?,?,?, 'PENDING',?,?,?,?,?,?,?)""",
                (
                    proposal_id,
                    approval_id,
                    compiled.proposal_sha256,
                    operation,
                    compiled.first_due_at_utc,
                    compiled.proposal.get("target_job_id"),
                    compiled.proposal.get("target_job_revision"),
                    compiled.proposal.get("target_occurrence_id"),
                    record_id,
                    utc_text(selected),
                    utc_text(expires),
                ),
            )
            if task is not None:
                connection.execute(
                    """INSERT INTO scheduler_task_links(
                       proposal_id,task_id,epoch,state,result_json,created_at,resolved_at
                       ) VALUES(?,?,?,'PENDING',NULL,?,NULL)""",
                    (proposal_id, task_id, int(epoch), utc_text(selected)),
                )
                elapsed = float(task["active_elapsed_seconds"])
                if task["active_started_at"] is not None:
                    elapsed += max(
                        0.0,
                        (selected - parse_utc(task["active_started_at"])).total_seconds(),
                    )
                changed = connection.execute(
                    """UPDATE tasks SET state='AWAITING_APPROVAL',active_elapsed_seconds=?,
                       active_started_at=NULL,disconnect_expires_at=NULL,updated_at=?
                       WHERE task_id=? AND epoch=? AND state='EXECUTING'""",
                    (elapsed, utc_text(selected), task_id, int(epoch)),
                )
                if changed.rowcount != 1:
                    raise OrchestratorError("stale_epoch", status=409)
                self._store._append_task_event_tx(
                    connection,
                    task_id,
                    int(epoch),
                    "scheduler.awaiting_approval",
                    {"approval_id": approval_id, "proposal_id": proposal_id},
                    selected,
                )
            connection.execute(
                """INSERT INTO approval_arbiter_queue(
                   approval_id,producer_protocol,binding_sha256,state,created_at,expires_at,updated_at
                   ) VALUES(?,?,?,'WAITING_FOR_CHANNEL',?,?,?)""",
                (
                    approval_id,
                    SCHEDULER_PROTOCOL,
                    compiled.proposal_sha256,
                    utc_text(selected),
                    utc_text(expires),
                    utc_text(selected),
                ),
            )
            self._ledger_tx(
                connection,
                entity_type="proposal",
                entity_id=proposal_id,
                event_type="scheduler.proposal.created",
                from_state=None,
                to_state="PENDING",
                now=selected,
                metadata={"approval_id": approval_id, "proposal_sha256": compiled.proposal_sha256},
            )
            result = {
                "approval_id": approval_id,
                "approval_state": "WAITING_FOR_CHANNEL",
                "expires_at_utc": utc_text(expires),
                "proposal_id": proposal_id,
                "proposal_sha256": compiled.proposal_sha256,
            }
            if api_event_id is not None:
                self._record_api_result_tx(
                    connection,
                    event_id=api_event_id,
                    semantic_sha256=str(api_semantic_sha256),
                    event_type="scheduler.control.proposal_created",
                    status=201,
                    result=result,
                    now=selected,
                )
            return result

    def task_run(self, task_id: str, epoch: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT l.*,p.approval_id,p.operation,p.proposal_sha256,p.state AS proposal_state
                   FROM scheduler_task_links l JOIN scheduler_proposals p USING(proposal_id)
                   WHERE l.task_id=? AND l.epoch=?""",
                (task_id, epoch),
            ).fetchone()
        if row is None:
            return None
        return {
            "approval_id": row["approval_id"],
            "operation": row["operation"],
            "proposal_id": row["proposal_id"],
            "proposal_sha256": row["proposal_sha256"],
            "proposal_state": row["proposal_state"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "state": row["state"],
        }

    def task_id_for_proposal(self, proposal_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT task_id FROM scheduler_task_links WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
        return str(row["task_id"]) if row is not None else None

    def _expire_waiting_tx(self, connection: sqlite3.Connection, now: datetime) -> None:
        rows = connection.execute(
            """SELECT q.approval_id,p.proposal_id,p.first_due_at_utc
               FROM approval_arbiter_queue q LEFT JOIN scheduler_proposals p USING(approval_id)
               WHERE q.producer_protocol=?
               AND q.state IN ('WAITING_FOR_CHANNEL','PRESENTING','PRESENTED','WAITING_FOR_OWNER')
               AND (q.expires_at<=? OR (p.first_due_at_utc IS NOT NULL AND p.first_due_at_utc<?))""",
            (SCHEDULER_PROTOCOL, utc_text(now), utc_text(now + timedelta(milliseconds=MINIMUM_LEAD_MS))),
        ).fetchall()
        for row in rows:
            connection.execute(
                "UPDATE approval_arbiter_queue SET state='EXPIRED',updated_at=? WHERE approval_id=?",
                (utc_text(now), row["approval_id"]),
            )
            if row["proposal_id"]:
                connection.execute(
                    "UPDATE scheduler_proposals SET state='EXPIRED',decided_at=? WHERE proposal_id=? AND state='PENDING'",
                    (utc_text(now), row["proposal_id"]),
                )
                self._ledger_tx(
                    connection,
                    entity_type="proposal",
                    entity_id=row["proposal_id"],
                    event_type="scheduler.proposal.expired",
                    from_state="PENDING",
                    to_state="EXPIRED",
                    now=now,
                    metadata={"reason_code": "approval_expired"},
                )

    def expire_approvals(self, *, now: datetime | None = None) -> int:
        selected = now or utc_now()
        with self._transaction() as connection:
            before = connection.execute(
                """SELECT COUNT(*) FROM approval_arbiter_queue
                   WHERE producer_protocol=? AND state='EXPIRED'""", (SCHEDULER_PROTOCOL,)
            ).fetchone()[0]
            self._expire_waiting_tx(connection, selected)
            after = connection.execute(
                """SELECT COUNT(*) FROM approval_arbiter_queue
                   WHERE producer_protocol=? AND state='EXPIRED'""", (SCHEDULER_PROTOCOL,)
            ).fetchone()[0]
            return int(after - before)

    def promote_next_approval(
        self,
        owner_session_id: str,
        *,
        supported_protocols: Sequence[str],
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        require_uuid4(owner_session_id)
        selected = now or utc_now()
        with self._transaction() as connection:
            self._expire_waiting_tx(connection, selected)
            active = connection.execute(
                """SELECT approval_id FROM approval_arbiter_queue
                   WHERE state IN ('PRESENTING','PRESENTED','DECIDING') LIMIT 1""",
            ).fetchone()
            if active is not None:
                return None
            row = connection.execute(
                """SELECT q.*,p.proposal_id,p.first_due_at_utc,p.content_record_id,p.operation
                   FROM approval_arbiter_queue q
                   LEFT JOIN scheduler_proposals p USING(approval_id)
                   WHERE q.state IN ('WAITING_FOR_OWNER','WAITING_FOR_CHANNEL')
                   ORDER BY CASE q.state WHEN 'WAITING_FOR_OWNER' THEN 0 ELSE 1 END,
                            q.arbiter_seq,q.approval_id LIMIT 1"""
            ).fetchone()
            if row is None or row["producer_protocol"] not in supported_protocols:
                return None
            if row["producer_protocol"] != SCHEDULER_PROTOCOL:
                return None
            if row["first_due_at_utc"] is not None:
                try:
                    ensure_minimum_lead(row["first_due_at_utc"], selected)
                except ScheduleContractError:
                    connection.execute(
                        "UPDATE approval_arbiter_queue SET state='EXPIRED',updated_at=? WHERE approval_id=?",
                        (utc_text(selected), row["approval_id"]),
                    )
                    connection.execute(
                        "UPDATE scheduler_proposals SET state='EXPIRED',decided_at=? WHERE proposal_id=?",
                        (utc_text(selected), row["proposal_id"]),
                    )
                    return None
            proposal = self._get_content_tx(connection, row["content_record_id"])
            if row["state"] == "WAITING_FOR_OWNER" and row["requested_event_id"]:
                connection.execute(
                    """UPDATE scheduler_outbox SET state='REVOKED'
                       WHERE event_id=? AND state IN ('PENDING','LEASED')""",
                    (row["requested_event_id"],),
                )
            presentation_id = new_uuid4()
            decision_nonce = new_uuid4()
            requested_event_id = new_uuid4()
            preview = {
                "first_due_at_utc": row["first_due_at_utc"],
                "notification": proposal.get("content"),
                "operation": row["operation"],
                "schedule": proposal.get("schedule"),
            }
            payload = {
                "approval_id": row["approval_id"],
                "approve_key": "F10",
                "decision_nonce": decision_nonce,
                "deny_key": "F11",
                "expected_job_revision": proposal.get("target_job_revision"),
                "expires_at_utc": row["expires_at"],
                "job_id": proposal.get("target_job_id"),
                "occurrence_id": proposal.get("target_occurrence_id"),
                "operation": row["operation"],
                "presentation_id": presentation_id,
                "preview": preview,
                "proposal_id": row["proposal_id"],
                "proposal_sha256": row["binding_sha256"],
            }
            self._append_outbox_tx(
                connection,
                event_type="scheduler.approval.requested",
                payload=payload,
                expires_at=parse_utc(row["expires_at"]),
                now=selected,
                event_id=requested_event_id,
            )
            connection.execute(
                """UPDATE approval_arbiter_queue SET state='PRESENTING',owner_session_id=?,
                   presentation_id=?,requested_event_id=?,decision_nonce=?,updated_at=?
                   WHERE approval_id=? AND state IN ('WAITING_FOR_CHANNEL','WAITING_FOR_OWNER')""",
                (
                    owner_session_id,
                    presentation_id,
                    requested_event_id,
                    decision_nonce,
                    utc_text(selected),
                    row["approval_id"],
                ),
            )
            return {"event_id": requested_event_id, "payload": payload}

    def acknowledge_scheduler_event(
        self, event_id: str, *, status: str, speech_supported: bool = False,
        now: datetime | None = None,
    ) -> bool:
        require_uuid4(event_id)
        if status not in {"applied", "duplicate"}:
            raise OrchestratorError("invalid_request", status=400)
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT event_type,state,content_record_id,expires_at FROM scheduler_outbox WHERE event_id=?", (event_id,)
            ).fetchone()
            if row is None:
                raise OrchestratorError("not_found", status=404)
            if row["state"] == "ACKED":
                return False
            connection.execute(
                "UPDATE scheduler_outbox SET state='ACKED',ack_status=?,lease_owner=NULL,lease_expires_at=NULL WHERE event_id=?",
                (status, event_id),
            )
            if row["event_type"] == "scheduler.approval.requested":
                connection.execute(
                    """UPDATE approval_arbiter_queue SET state='PRESENTED',updated_at=?
                       WHERE requested_event_id=? AND state='PRESENTING'""",
                    (utc_text(selected), event_id),
                )
            elif row["event_type"] == "scheduler.notification.requested":
                notification = self._get_content_tx(connection, row["content_record_id"])
                delivery = connection.execute(
                    "SELECT state FROM scheduler_deliveries WHERE notification_id=?",
                    (notification["notification_id"],),
                ).fetchone()
                if delivery is not None and delivery["state"] == "OUTBOXED":
                    connection.execute(
                        "UPDATE scheduler_deliveries SET state='STAGED',updated_at=? WHERE notification_id=?",
                        (utc_text(selected), notification["notification_id"]),
                    )
                    self._ledger_tx(
                        connection,
                        entity_type="delivery",
                        entity_id=notification["notification_id"],
                        event_type="scheduler.delivery.staged",
                        from_state="OUTBOXED",
                        to_state="STAGED",
                        now=selected,
                        cause_event_id=event_id,
                        metadata={"ack_status": status},
                    )
                job = connection.execute(
                    "SELECT content_record_id FROM scheduler_jobs WHERE job_id=? AND job_revision=?",
                    (notification["job_id"], notification["job_revision"]),
                ).fetchone()
                content = self._get_content_tx(connection, job["content_record_id"]) if job else {}
                speakable = content.get("speakable_text")
                if speech_supported and speakable:
                    self._append_outbox_tx(
                        connection,
                        event_type="scheduler.notification.speech",
                        entity_id=notification["notification_id"],
                        payload={
                            "occurrence_id": notification["occurrence_id"],
                            "notification_id": notification["notification_id"],
                            "unit_index": 0,
                            "locale": notification["locale"],
                            "speakable_text": speakable,
                        },
                        expires_at=parse_utc(row["expires_at"]),
                        now=selected,
                    )
                elif speakable:
                    self._ledger_tx(
                        connection,
                        entity_type="delivery",
                        entity_id=notification["notification_id"],
                        event_type="scheduler.speech.skipped",
                        from_state=None,
                        to_state="SKIPPED_UNSUPPORTED",
                        now=selected,
                        metadata={"reason_code": "SKIPPED_UNSUPPORTED"},
                    )
            return True

    def _proposal_row_tx(self, connection: sqlite3.Connection, approval_id: str) -> sqlite3.Row:
        row = connection.execute(
            """SELECT p.*,q.state AS approval_state,q.binding_sha256,q.owner_session_id,
                      q.presentation_id,q.requested_event_id,q.decision_nonce
               FROM scheduler_proposals p JOIN approval_arbiter_queue q USING(approval_id)
               WHERE p.approval_id=?""",
            (approval_id,),
        ).fetchone()
        if row is None:
            raise OrchestratorError("not_found", status=404)
        return row

    def decide_approval(
        self,
        *,
        approval_id: str,
        proposal_id: str,
        proposal_sha256: str,
        decision: str,
        cause_event_id: str,
        owner_session_id: str | None = None,
        presentation_id: str | None = None,
        requested_event_id: str | None = None,
        decision_nonce: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        require_uuid4(approval_id)
        require_uuid4(proposal_id)
        require_uuid4(cause_event_id)
        self._sha(proposal_sha256)
        if decision not in {"approve", "deny"}:
            raise OrchestratorError("invalid_request", status=400)
        selected = parse_utc(utc_text(now or utc_now()))
        command_semantic = domain_sha256(
            "scheduler-approval-api-decision/1",
            {"approval_id": approval_id, "proposal_id": proposal_id,
             "proposal_sha256": proposal_sha256, "decision": decision},
        )
        replay_exists, replay = self._existing_inbox(cause_event_id, command_semantic)
        if replay_exists:
            if replay is None:
                raise OrchestratorError("scheduler_content_unavailable", status=503)
            return replay
        with self._lock:
            lead_row = self._connection.execute(
                """SELECT first_due_at_utc,state,expires_at FROM scheduler_proposals
                   WHERE proposal_id=? AND approval_id=?""",
                (proposal_id, approval_id),
            ).fetchone()
        if lead_row is not None and lead_row["state"] == "PENDING" and parse_utc(lead_row["expires_at"]) <= selected:
            with self._transaction() as connection:
                changed = connection.execute(
                    "UPDATE scheduler_proposals SET state='EXPIRED',decided_at=? WHERE proposal_id=? AND state='PENDING'",
                    (utc_text(selected), proposal_id),
                )
                if changed.rowcount:
                    connection.execute(
                        "UPDATE approval_arbiter_queue SET state='EXPIRED',updated_at=? WHERE approval_id=?",
                        (utc_text(selected), approval_id),
                    )
                    self._ledger_tx(
                        connection, entity_type="proposal", entity_id=proposal_id,
                        event_type="scheduler.proposal.expired", from_state="PENDING", to_state="EXPIRED",
                        now=selected, metadata={"reason_code": "approval_expired"},
                    )
            raise OrchestratorError("approval_expired", status=410)
        if decision == "approve":
            if lead_row is not None and lead_row["state"] == "PENDING" and lead_row["first_due_at_utc"]:
                try:
                    ensure_minimum_lead(lead_row["first_due_at_utc"], selected)
                except ScheduleContractError as exc:
                    with self._transaction() as connection:
                        changed = connection.execute(
                            """UPDATE scheduler_proposals SET state='EXPIRED',decided_at=?
                               WHERE proposal_id=? AND state='PENDING'""",
                            (utc_text(selected), proposal_id),
                        )
                        if changed.rowcount:
                            connection.execute(
                                "UPDATE approval_arbiter_queue SET state='EXPIRED',updated_at=? WHERE approval_id=?",
                                (utc_text(selected), approval_id),
                            )
                            self._ledger_tx(
                                connection, entity_type="proposal", entity_id=proposal_id,
                                event_type="scheduler.proposal.expired", from_state="PENDING",
                                to_state="EXPIRED", now=selected,
                                metadata={"reason_code": "minimum_lead_violated"},
                            )
                    raise OrchestratorError("minimum_lead_violated", status=410) from exc
        with self._transaction() as connection:
            prior = connection.execute(
                "SELECT semantic_sha256,content_record_id FROM scheduler_inbox WHERE event_id=?",
                (cause_event_id,),
            ).fetchone()
            if prior is not None:
                if prior["semantic_sha256"] != command_semantic:
                    raise OrchestratorError("protocol.conflicting_duplicate", status=409)
                return self._get_content_tx(connection, prior["content_record_id"])
            row = self._proposal_row_tx(connection, approval_id)
            if owner_session_id is not None and (
                row["owner_session_id"] != owner_session_id
                or row["presentation_id"] != presentation_id
                or row["requested_event_id"] != requested_event_id
                or row["decision_nonce"] != decision_nonce
                or row["approval_state"] not in {"PRESENTING", "PRESENTED"}
            ):
                raise OrchestratorError("not_owner", status=409)
            if row["proposal_id"] != proposal_id or row["proposal_sha256"] != proposal_sha256:
                raise OrchestratorError("binding_mismatch", status=409)
            if row["state"] != "PENDING" or row["approval_state"] not in {
                "WAITING_FOR_CHANNEL", "PRESENTING", "PRESENTED", "WAITING_FOR_OWNER"
            }:
                raise OrchestratorError("invalid_state", status=409)
            if parse_utc(row["expires_at"]) <= selected:
                connection.execute(
                    "UPDATE scheduler_proposals SET state='EXPIRED',decided_at=? WHERE proposal_id=?",
                    (utc_text(selected), proposal_id),
                )
                connection.execute(
                    "UPDATE approval_arbiter_queue SET state='EXPIRED',updated_at=? WHERE approval_id=?",
                    (utc_text(selected), approval_id),
                )
                raise OrchestratorError("approval_expired", status=410)
            if decision == "approve" and row["first_due_at_utc"]:
                ensure_minimum_lead(row["first_due_at_utc"], selected)
            connection.execute(
                "UPDATE approval_arbiter_queue SET state='DECIDING',updated_at=? WHERE approval_id=?",
                (utc_text(selected), approval_id),
            )
            job_id: str | None = None
            occurrence: str | None = None
            job_revision: int | None = None
            if decision == "deny":
                connection.execute(
                    "UPDATE scheduler_proposals SET state='DENIED',decided_at=? WHERE proposal_id=?",
                    (utc_text(selected), proposal_id),
                )
                result_reason = "denied"
            else:
                proposal = self._get_content_tx(connection, row["content_record_id"])
                operation = row["operation"]
                if operation == "CREATE_JOB":
                    job_id, job_revision, occurrence = self._activate_job_tx(
                        connection, row=row, proposal=proposal, now=selected, cause_event_id=cause_event_id
                    )
                    result_reason = "approved"
                elif operation == "CANCEL_JOB":
                    job_id = str(row["target_job_id"])
                    job_revision = int(row["target_job_revision"])
                    self._cancel_job_tx(connection, job_id, job_revision, selected, cause_event_id)
                    result_reason = "cancelled"
                else:
                    job_id = str(row["target_job_id"])
                    job_revision = int(row["target_job_revision"])
                    occurrence = str(row["target_occurrence_id"])
                    self._snooze_tx(
                        connection,
                        job_id=job_id,
                        job_revision=job_revision,
                        occurrence=occurrence,
                        snooze_until_at_utc=proposal["schedule"]["snooze_until_at_utc"],
                        cause_event_id=cause_event_id,
                        now=selected,
                        expected_generation=proposal["target_snooze_generation"],
                    )
                    result_reason = "snoozed"
                connection.execute(
                    "UPDATE scheduler_proposals SET state='APPROVED',decided_at=? WHERE proposal_id=?",
                    (utc_text(selected), proposal_id),
                )
            connection.execute(
                "UPDATE approval_arbiter_queue SET state='RESOLVED',updated_at=? WHERE approval_id=?",
                (utc_text(selected), approval_id),
            )
            result_payload = {
                "approval_id": approval_id,
                "decision_event_id": cause_event_id,
                "effective_due_at_utc": row["first_due_at_utc"],
                "job_id": job_id,
                "job_revision": job_revision,
                "occurrence_id": occurrence,
                "proposal_id": proposal_id,
                "proposal_sha256": proposal_sha256,
                "reason_code": result_reason,
                "result": "applied",
            }
            task_link = connection.execute(
                "SELECT task_id,epoch,state FROM scheduler_task_links WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
            if task_link is not None:
                task = self._store._guarded_task_tx(
                    connection,
                    task_link["task_id"],
                    int(task_link["epoch"]),
                    (TaskState.AWAITING_APPROVAL,),
                )
                link_state = "APPROVED" if decision == "approve" else "DENIED"
                connection.execute(
                    """UPDATE scheduler_task_links SET state=?,result_json=?,resolved_at=?
                       WHERE proposal_id=? AND state='PENDING'""",
                    (link_state, jcs_text(result_payload), utc_text(selected), proposal_id),
                )
                connection.execute(
                    """UPDATE tasks SET state='EXECUTING',active_started_at=?,updated_at=?
                       WHERE task_id=? AND epoch=? AND state='AWAITING_APPROVAL'""",
                    (
                        utc_text(selected),
                        utc_text(selected),
                        task_link["task_id"],
                        int(task_link["epoch"]),
                    ),
                )
                turn = connection.execute(
                    "SELECT effects_json FROM canonical_turns WHERE task_id=? AND status='PENDING'",
                    (task_link["task_id"],),
                ).fetchone()
                if turn is None:
                    raise OrchestratorError("stale_epoch", status=409)
                effects = json.loads(turn["effects_json"])
                effects.append(
                    {
                        "kind": "scheduler" if decision == "approve" else "ToolDenied",
                        "name": operation,
                        "status": link_state,
                        "arguments_sha256": proposal_sha256,
                        "result_sha256": domain_sha256(
                            "jarvis.scheduler-task-result/1", result_payload
                        ),
                    }
                )
                connection.execute(
                    "UPDATE canonical_turns SET effects_json=? WHERE task_id=?",
                    (jcs_text(effects), task_link["task_id"]),
                )
                self._store._append_task_event_tx(
                    connection,
                    task_link["task_id"],
                    int(task_link["epoch"]),
                    "scheduler.approval.decided",
                    {
                        "decision": decision,
                        "proposal_id": proposal_id,
                        "result_sha256": domain_sha256(
                            "jarvis.scheduler-task-result/1", result_payload
                        ),
                    },
                    selected,
                )
            self._append_outbox_tx(
                connection,
                event_type="scheduler.approval.decision_result",
                payload=result_payload,
                expires_at=selected + timedelta(seconds=APPROVAL_TTL_SECONDS),
                now=selected,
            )
            result_record_id = new_uuid4()
            self._put_content_tx(
                connection,
                record_type="INBOX_RESULT",
                record_id=result_record_id,
                job_id=job_id,
                job_revision=job_revision or 0,
                value=result_payload,
                now=selected,
                purge_after=selected + timedelta(milliseconds=CONTENT_RETENTION_MS),
            )
            connection.execute(
                """INSERT INTO scheduler_inbox(
                   event_id,semantic_sha256,protocol,event_type,content_record_id,state,created_at,applied_at
                   ) VALUES(?,?,?,'scheduler.approval.api_decision',?,'APPLIED',?,?)""",
                (cause_event_id, command_semantic, SCHEDULER_PROTOCOL, result_record_id,
                 utc_text(selected), utc_text(selected)),
            )
            self._ledger_tx(
                connection,
                entity_type="proposal",
                entity_id=proposal_id,
                event_type="scheduler.proposal.decided",
                from_state="PENDING",
                to_state="APPROVED" if decision == "approve" else "DENIED",
                now=selected,
                cause_event_id=cause_event_id,
                metadata={"reason_code": result_reason},
            )
            return result_payload

    def _activate_job_tx(
        self,
        connection: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        proposal: Mapping[str, Any],
        now: datetime,
        cause_event_id: str,
    ) -> tuple[str, int, str]:
        active_count = connection.execute(
            "SELECT COUNT(*) FROM scheduler_jobs WHERE state IN ('ACTIVE','DRAINING')"
        ).fetchone()[0]
        if active_count >= MAXIMUM_ACTIVE_JOBS:
            raise OrchestratorError("maximum_active_jobs", status=409)
        job_id = new_uuid4()
        revision = 1
        schedule = dict(proposal["schedule"])
        scheduled_at = str(row["first_due_at_utc"])
        occurrence = occurrence_id(job_id, revision, scheduled_at)
        original_content = connection.execute(
            "SELECT * FROM scheduler_content WHERE record_id=?", (row["content_record_id"],)
        ).fetchone()
        if original_content is None:
            raise OrchestratorError("scheduler_content_unavailable", status=503)
        # Re-encrypt under the final job/revision AAD; ciphertext is never copied.
        job_content_id = new_uuid4()
        self._put_content_tx(
            connection,
            record_type="APPROVED_CONTENT",
            record_id=job_content_id,
            job_id=job_id,
            job_revision=revision,
            value=dict(proposal["content"]),
            now=now,
        )
        count = schedule.get("count")
        one_shot = schedule["kind"] == "ONE_SHOT"
        exhausted = one_shot or count == 1
        state = "DRAINING" if exhausted else "ACTIVE"
        connection.execute(
            """INSERT INTO scheduler_jobs(
               job_id,proposal_id,job_revision,state,schedule_json,schedule_sha256,
               content_record_id,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                job_id,
                row["proposal_id"],
                revision,
                state,
                jcs_text(schedule),
                domain_sha256("schedule-contract/1", schedule),
                job_content_id,
                utc_text(now),
                utc_text(now),
            ),
        )
        connection.execute(
            """INSERT INTO scheduler_occurrences(
               occurrence_id,job_id,job_revision,sequence,scheduled_at_utc,effective_due_at_utc,
               state,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,'SCHEDULED',?,?)""",
            (occurrence, job_id, revision, 0, scheduled_at, scheduled_at, utc_text(now), utc_text(now)),
        )
        connection.execute(
            """INSERT INTO scheduler_recurrence_cursors(
               job_id,job_revision,next_sequence,next_scheduled_at_utc,next_local_slot,exhausted,updated_at
               ) VALUES(?,?,?,?,?,?,?)""",
            (
                job_id,
                revision,
                1,
                None,
                None,
                int(exhausted),
                utc_text(now),
            ),
        )
        self._ledger_tx(
            connection,
            entity_type="job",
            entity_id=job_id,
            event_type="scheduler.job.activated",
            from_state=None,
            to_state=state,
            now=now,
            cause_event_id=cause_event_id,
            metadata={"job_revision": revision, "proposal_id": row["proposal_id"]},
        )
        self._ledger_tx(
            connection,
            entity_type="occurrence",
            entity_id=occurrence,
            event_type="scheduler.occurrence.materialized",
            from_state=None,
            to_state="SCHEDULED",
            now=now,
            cause_event_id=cause_event_id,
            metadata={"scheduled_at_utc": scheduled_at, "sequence": 0},
        )
        return job_id, revision, occurrence

    def _materialize_next_tx(self, connection: sqlite3.Connection, job_id: str, now: datetime) -> str | None:
        job = connection.execute("SELECT * FROM scheduler_jobs WHERE job_id=?", (job_id,)).fetchone()
        cursor = connection.execute(
            "SELECT * FROM scheduler_recurrence_cursors WHERE job_id=?", (job_id,)
        ).fetchone()
        if job is None or cursor is None or cursor["exhausted"]:
            return None
        schedule = json.loads(job["schedule_json"])
        previous = connection.execute(
            """SELECT * FROM scheduler_occurrences WHERE job_id=? AND job_revision=?
               ORDER BY sequence DESC LIMIT 1""",
            (job_id, job["job_revision"]),
        ).fetchone()
        if previous is None:
            raise OrchestratorError("scheduler_cursor_invalid", status=500)
        sequence = int(cursor["next_sequence"])
        count = schedule.get("count")
        if count is not None and sequence >= int(count):
            connection.execute(
                "UPDATE scheduler_recurrence_cursors SET exhausted=1,next_scheduled_at_utc=NULL,updated_at=? WHERE job_id=?",
                (utc_text(now), job_id),
            )
            connection.execute(
                "UPDATE scheduler_jobs SET state='DRAINING',updated_at=? WHERE job_id=? AND state='ACTIVE'",
                (utc_text(now), job_id),
            )
            return None
        if schedule["kind"] == "FIXED_INTERVAL":
            next_due = parse_utc(previous["scheduled_at_utc"]) + timedelta(milliseconds=int(schedule["interval_ms"]))
            local_slot = None
        elif schedule["kind"].startswith("CALENDAR_"):
            zone, _digest = self.zones.load(schedule["timezone"])
            current_local = parse_utc(previous["scheduled_at_utc"]).astimezone(zone).replace(tzinfo=None)
            while True:
                next_local = next_calendar_local(schedule, current_local)
                candidates = []
                for fold in (0, 1):
                    aware = next_local.replace(tzinfo=zone, fold=fold)
                    if aware.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == next_local:
                        utc = aware.astimezone(UTC)
                        if utc not in candidates:
                            candidates.append(utc)
                if candidates:
                    next_due = sorted(candidates)[0]
                    local_slot = next_local.isoformat(timespec="milliseconds")
                    break
                current_local = next_local  # gap: skip without consuming count
        else:
            return None
        until = schedule.get("until_at_utc")
        if until is not None and next_due > parse_utc(until):
            connection.execute(
                "UPDATE scheduler_recurrence_cursors SET exhausted=1,next_scheduled_at_utc=NULL,updated_at=? WHERE job_id=?",
                (utc_text(now), job_id),
            )
            connection.execute(
                "UPDATE scheduler_jobs SET state='DRAINING',updated_at=? WHERE job_id=? AND state='ACTIVE'",
                (utc_text(now), job_id),
            )
            return None
        scheduled = utc_text(next_due)
        identifier = occurrence_id(job_id, job["job_revision"], scheduled)
        inserted = connection.execute(
            """INSERT OR IGNORE INTO scheduler_occurrences(
               occurrence_id,job_id,job_revision,sequence,scheduled_at_utc,effective_due_at_utc,
               state,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,'SCHEDULED',?,?)""",
            (
                identifier,
                job_id,
                job["job_revision"],
                sequence,
                scheduled,
                scheduled,
                utc_text(now),
                utc_text(now),
            ),
        )
        if inserted.rowcount:
            self._ledger_tx(
                connection,
                entity_type="occurrence",
                entity_id=identifier,
                event_type="scheduler.occurrence.materialized",
                from_state=None,
                to_state="SCHEDULED",
                now=now,
                metadata={"scheduled_at_utc": scheduled, "sequence": sequence},
            )
        next_exhausted = bool(count is not None and sequence + 1 >= int(count))
        connection.execute(
            """UPDATE scheduler_recurrence_cursors SET next_sequence=?,next_scheduled_at_utc=?,
               next_local_slot=?,exhausted=?,updated_at=? WHERE job_id=?""",
            (sequence + 1, scheduled, local_slot, int(next_exhausted), utc_text(now), job_id),
        )
        if next_exhausted:
            connection.execute(
                "UPDATE scheduler_jobs SET state='DRAINING',updated_at=? WHERE job_id=? AND state='ACTIVE'",
                (utc_text(now), job_id),
            )
        return identifier

    def _delivery_payload_tx(
        self, connection: sqlite3.Connection, occurrence_row: sqlite3.Row, generation: int
    ) -> tuple[dict[str, Any], sqlite3.Row]:
        job = connection.execute("SELECT * FROM scheduler_jobs WHERE job_id=?", (occurrence_row["job_id"],)).fetchone()
        if job is None:
            raise OrchestratorError("not_found", status=404)
        content = self._get_content_tx(connection, job["content_record_id"])
        identifier = notification_id(occurrence_row["occurrence_id"], generation)
        payload = {
            "allow_full_body_when_locked": False,
            "body": content["body"],
            "content_sha256": domain_sha256("scheduler-notification-content/1", content),
            "delivery_generation": generation,
            "effective_due_at_utc": occurrence_row["effective_due_at_utc"],
            "expires_at_utc": utc_text(parse_utc(occurrence_row["effective_due_at_utc"]) + timedelta(milliseconds=MISSED_GRACE_MS)),
            "job_id": job["job_id"],
            "job_revision": job["job_revision"],
            "locale": content["locale"],
            "notification_id": identifier,
            "occurrence_id": occurrence_row["occurrence_id"],
            "private_body": "Έχεις μια υπενθύμιση." if content["locale"] == "el-GR" else "You have a reminder.",
            "private_title": "CYRUS",
            "scheduled_at_utc": occurrence_row["scheduled_at_utc"],
            "silent": False,
            "title": content["title"],
            "visibility": content["visibility"],
        }
        return payload, job

    def fire_due(
        self, *, now: datetime | None = None, batch_size: int = 256,
        worker_boot_id: str | None = None,
        monotonic_lease_deadline_ns: int | None = None,
        clock_jump_detected: bool = False,
    ) -> list[str]:
        selected = now or utc_now()
        if not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be from 1 to 256")
        fired: list[str] = []
        with self._transaction() as connection:
            incomplete_jobs: set[str] = set()
            jobs = connection.execute(
                """SELECT j.job_id FROM scheduler_jobs j JOIN scheduler_recurrence_cursors c USING(job_id)
                   WHERE j.state IN ('ACTIVE','DRAINING') AND c.exhausted=0"""
            ).fetchall()
            for job in jobs:
                materialized = 0
                while materialized < batch_size:
                    latest = connection.execute(
                        """SELECT effective_due_at_utc FROM scheduler_occurrences
                           WHERE job_id=? AND state='SCHEDULED' ORDER BY sequence DESC LIMIT 1""",
                        (job["job_id"],),
                    ).fetchone()
                    if latest is None or parse_utc(latest["effective_due_at_utc"]) > selected:
                        break
                    created = self._materialize_next_tx(connection, job["job_id"], selected)
                    if created is None:
                        break
                    materialized += 1
                cursor = connection.execute(
                    "SELECT exhausted FROM scheduler_recurrence_cursors WHERE job_id=?",
                    (job["job_id"],),
                ).fetchone()
                latest = connection.execute(
                    """SELECT effective_due_at_utc FROM scheduler_occurrences
                       WHERE job_id=? AND state='SCHEDULED' ORDER BY sequence DESC LIMIT 1""",
                    (job["job_id"],),
                ).fetchone()
                if (materialized == batch_size and cursor is not None and not cursor["exhausted"]
                        and latest is not None and parse_utc(latest["effective_due_at_utc"]) <= selected):
                    incomplete_jobs.add(job["job_id"])
            rows = connection.execute(
                """SELECT * FROM scheduler_occurrences WHERE state='SCHEDULED'
                   AND effective_due_at_utc<=? ORDER BY effective_due_at_utc,job_id,occurrence_id LIMIT ?""",
                (utc_text(selected), batch_size),
            ).fetchall()
            rows = [row for row in rows if row["job_id"] not in incomplete_jobs]
            latest_by_job: dict[str, sqlite3.Row] = {}
            for job_id in {row["job_id"] for row in rows}:
                latest = connection.execute(
                    """SELECT * FROM scheduler_occurrences WHERE job_id=? AND state='SCHEDULED'
                       AND effective_due_at_utc<=? AND effective_due_at_utc>=?
                       ORDER BY effective_due_at_utc DESC,occurrence_id DESC LIMIT 1""",
                    (job_id, utc_text(selected),
                     utc_text(selected - timedelta(milliseconds=MISSED_GRACE_MS))),
                ).fetchone()
                if latest is not None:
                    latest_by_job[job_id] = latest
            for row in rows:
                due = parse_utc(row["effective_due_at_utc"])
                if due < selected - timedelta(milliseconds=MISSED_GRACE_MS):
                    target = "MISSED_EXPIRED"
                elif latest_by_job[row["job_id"]]["occurrence_id"] != row["occurrence_id"]:
                    target = "MISSED_COALESCED"
                else:
                    target = "FIRED"
                if target != "FIRED":
                    connection.execute(
                        "UPDATE scheduler_occurrences SET state=?,updated_at=? WHERE occurrence_id=? AND state='SCHEDULED'",
                        (target, utc_text(selected), row["occurrence_id"]),
                    )
                    self._ledger_tx(
                        connection,
                        entity_type="occurrence",
                        entity_id=row["occurrence_id"],
                        event_type="scheduler.occurrence.missed",
                        from_state="SCHEDULED",
                        to_state=target,
                        now=selected,
                        metadata={"scheduled_at_utc": row["scheduled_at_utc"]},
                        worker_boot_id=worker_boot_id,
                        monotonic_lease_deadline_ns=monotonic_lease_deadline_ns,
                        clock_jump_detected=clock_jump_detected,
                    )
                else:
                    try:
                        payload, _job = self._delivery_payload_tx(connection, row, row["snooze_generation"])
                    except SchedulerCryptoError:
                        blocked_notification = notification_id(row["occurrence_id"], row["snooze_generation"])
                        expires = utc_text(parse_utc(row["effective_due_at_utc"]) + timedelta(milliseconds=MISSED_GRACE_MS))
                        existing = connection.execute(
                            "SELECT 1 FROM scheduler_deliveries WHERE notification_id=?",
                            (blocked_notification,),
                        ).fetchone()
                        if existing is None:
                            connection.execute(
                                """INSERT INTO scheduler_deliveries(
                                   notification_id,occurrence_id,delivery_generation,state,effective_due_at_utc,
                                   expires_at,content_record_id,created_at,updated_at
                                   ) VALUES(?,?,?,'BLOCKED_KEY_UNAVAILABLE',?,?,NULL,?,?)""",
                                (blocked_notification, row["occurrence_id"], row["snooze_generation"],
                                 row["effective_due_at_utc"], expires, utc_text(selected), utc_text(selected)),
                            )
                        else:
                            connection.execute(
                                """UPDATE scheduler_deliveries SET state='BLOCKED_KEY_UNAVAILABLE',updated_at=?
                                   WHERE notification_id=? AND state='SCHEDULED'""",
                                (utc_text(selected), blocked_notification),
                            )
                        connection.execute(
                            """UPDATE scheduler_occurrences SET state='FIRED',action_deadline_at=?,updated_at=?
                               WHERE occurrence_id=? AND state='SCHEDULED'""",
                            (expires, utc_text(selected), row["occurrence_id"]),
                        )
                        self._ledger_tx(
                            connection, entity_type="delivery", entity_id=blocked_notification,
                            event_type="scheduler.delivery.blocked_key_unavailable", from_state=None,
                            to_state="BLOCKED_KEY_UNAVAILABLE", now=selected,
                            metadata={"reason_code": "scheduler_key_unavailable"},
                        )
                        fired.append(row["occurrence_id"])
                        continue
                    record_id = new_uuid4()
                    self._put_content_tx(
                        connection,
                        record_type="DELIVERY_PAYLOAD",
                        record_id=record_id,
                        job_id=row["job_id"],
                        job_revision=row["job_revision"],
                        value=payload,
                        now=selected,
                        purge_after=parse_utc(payload["expires_at_utc"]) + timedelta(milliseconds=CONTENT_RETENTION_MS),
                    )
                    existing_delivery = connection.execute(
                        "SELECT state FROM scheduler_deliveries WHERE notification_id=?",
                        (payload["notification_id"],),
                    ).fetchone()
                    if existing_delivery is None:
                        connection.execute(
                            """INSERT INTO scheduler_deliveries(
                           notification_id,occurrence_id,delivery_generation,state,effective_due_at_utc,
                           expires_at,content_record_id,created_at,updated_at
                           ) VALUES(?,?,?,'OUTBOXED',?,?,?,?,?)""",
                            (
                                payload["notification_id"], row["occurrence_id"], row["snooze_generation"],
                                row["effective_due_at_utc"], payload["expires_at_utc"], record_id,
                                utc_text(selected), utc_text(selected),
                            ),
                        )
                    elif existing_delivery["state"] == "SCHEDULED":
                        connection.execute(
                            """UPDATE scheduler_deliveries SET state='OUTBOXED',content_record_id=?,
                               expires_at=?,updated_at=? WHERE notification_id=? AND state='SCHEDULED'""",
                            (record_id, payload["expires_at_utc"], utc_text(selected), payload["notification_id"]),
                        )
                    else:
                        raise OrchestratorError("delivery_generation_conflict", status=409)
                    self._append_outbox_tx(
                        connection,
                        event_type="scheduler.notification.requested",
                        entity_id=payload["notification_id"],
                        payload=payload,
                        expires_at=parse_utc(payload["expires_at_utc"]),
                        now=selected,
                    )
                    connection.execute(
                        """UPDATE scheduler_occurrences SET state='FIRED',action_deadline_at=?,updated_at=?
                           WHERE occurrence_id=? AND state='SCHEDULED'""",
                        (payload["expires_at_utc"], utc_text(selected), row["occurrence_id"]),
                    )
                    self._ledger_tx(
                        connection,
                        entity_type="occurrence",
                        entity_id=row["occurrence_id"],
                        event_type="scheduler.occurrence.fired",
                        from_state="SCHEDULED",
                        to_state="FIRED",
                        now=selected,
                        metadata={"notification_id": payload["notification_id"]},
                        worker_boot_id=worker_boot_id,
                        monotonic_lease_deadline_ns=monotonic_lease_deadline_ns,
                        clock_jump_detected=clock_jump_detected,
                    )
                    fired.append(row["occurrence_id"])
            self._settle_jobs_tx(connection, selected)
        return fired

    def _snooze_tx(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        job_revision: int,
        occurrence: str,
        snooze_until_at_utc: str,
        cause_event_id: str,
        now: datetime,
        expected_generation: int | None = None,
    ) -> int:
        ensure_minimum_lead(snooze_until_at_utc, now)
        row = connection.execute(
            "SELECT * FROM scheduler_occurrences WHERE occurrence_id=? AND job_id=? AND job_revision=?",
            (occurrence, job_id, job_revision),
        ).fetchone()
        if row is None:
            raise OrchestratorError("not_found", status=404)
        if row["state"] not in {"SCHEDULED", "FIRED"}:
            raise OrchestratorError("invalid_state", status=409)
        if expected_generation is not None and int(row["snooze_generation"]) != expected_generation:
            raise OrchestratorError("target_binding_mismatch", status=409)
        generation = int(row["snooze_generation"]) + 1
        if generation > MAXIMUM_SNOOZE_GENERATIONS:
            raise OrchestratorError("snooze_limit_reached", status=409)
        if row["action_deadline_at"] and parse_utc(row["action_deadline_at"]) < now:
            raise OrchestratorError("snooze_window_expired", status=410)
        prior_deliveries = connection.execute(
            """SELECT notification_id,state FROM scheduler_deliveries
               WHERE occurrence_id=? AND state NOT IN ('EXPIRED_UNDELIVERED','REVOKED')""",
            (occurrence,),
        ).fetchall()
        for delivery in prior_deliveries:
            if delivery["state"] != "SCHEDULED":
                self._revoke_and_tombstone_tx(
                    connection,
                    notification=delivery["notification_id"],
                    occurrence=occurrence,
                    reason_code="occurrence_snoozed",
                    now=now,
                )
        connection.execute(
            """UPDATE scheduler_deliveries SET state='REVOKED',updated_at=?
               WHERE occurrence_id=? AND state NOT IN ('OS_ACCEPTED','EXPIRED_UNDELIVERED','REVOKED')""",
            (utc_text(now), occurrence),
        )
        connection.execute(
            """INSERT INTO scheduler_deferrals(
               occurrence_id,generation,prior_effective_due_at_utc,snooze_until_at_utc,cause_event_id,created_at
               ) VALUES(?,?,?,?,?,?)""",
            (
                occurrence,
                generation,
                row["effective_due_at_utc"],
                snooze_until_at_utc,
                cause_event_id,
                utc_text(now),
            ),
        )
        connection.execute(
            """UPDATE scheduler_occurrences SET state='SCHEDULED',effective_due_at_utc=?,
               snooze_generation=?,action_deadline_at=NULL,updated_at=? WHERE occurrence_id=?""",
            (snooze_until_at_utc, generation, utc_text(now), occurrence),
        )
        deferred_notification = notification_id(occurrence, generation)
        connection.execute(
            """INSERT INTO scheduler_deliveries(
               notification_id,occurrence_id,delivery_generation,state,effective_due_at_utc,
               expires_at,content_record_id,created_at,updated_at
               ) VALUES(?,?,?,'SCHEDULED',?,?,NULL,?,?)""",
            (deferred_notification, occurrence, generation, snooze_until_at_utc,
             utc_text(parse_utc(snooze_until_at_utc) + timedelta(milliseconds=MISSED_GRACE_MS)),
             utc_text(now), utc_text(now)),
        )
        self._ledger_tx(
            connection,
            entity_type="occurrence",
            entity_id=occurrence,
            event_type="scheduler.occurrence.snoozed",
            from_state=row["state"],
            to_state="SCHEDULED",
            now=now,
            cause_event_id=cause_event_id,
            metadata={"generation": generation, "snooze_until_at_utc": snooze_until_at_utc},
        )
        return generation

    def snooze_occurrence(
        self,
        *,
        job_id: str,
        job_revision: int,
        occurrence: str,
        snooze_until_at_utc: str,
        cause_event_id: str,
        expected_generation: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        selected = now or utc_now()
        with self._transaction() as connection:
            generation = self._snooze_tx(
                connection,
                job_id=job_id,
                job_revision=job_revision,
                occurrence=occurrence,
                snooze_until_at_utc=snooze_until_at_utc,
                cause_event_id=cause_event_id,
                now=selected,
                expected_generation=expected_generation,
            )
        return {"delivery_generation": generation, "occurrence_id": occurrence, "snooze_until_at_utc": snooze_until_at_utc}

    def _revoke_and_tombstone_tx(
        self,
        connection: sqlite3.Connection,
        *,
        notification: str,
        occurrence: str,
        reason_code: str,
        now: datetime,
    ) -> None:
        connection.execute(
            """UPDATE scheduler_outbox SET state='REVOKED',lease_owner=NULL,lease_expires_at=NULL
               WHERE entity_id=? AND event_type IN (
                 'scheduler.notification.requested','scheduler.notification.speech'
               ) AND state IN ('PENDING','LEASED')""",
            (notification,),
        )
        try:
            self._append_outbox_tx(
                connection,
                event_type="scheduler.notification.tombstone",
                entity_id=notification,
                payload={
                    "notification_id": notification,
                    "occurrence_id": occurrence,
                    "reason_code": reason_code,
                    "created_at_utc": utc_text(now),
                },
                expires_at=now + timedelta(milliseconds=MISSED_GRACE_MS),
                now=now,
            )
        except SchedulerCryptoError:
            self._ledger_tx(
                connection,
                entity_type="delivery",
                entity_id=notification,
                event_type="scheduler.notification.tombstone_blocked",
                from_state=None,
                to_state=None,
                now=now,
                metadata={"reason_code": "scheduler_key_unavailable"},
            )

    def _cancel_job_tx(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        job_revision: int,
        now: datetime,
        cause_event_id: str,
    ) -> None:
        row = connection.execute(
            "SELECT state FROM scheduler_jobs WHERE job_id=? AND job_revision=?", (job_id, job_revision)
        ).fetchone()
        if row is None:
            raise OrchestratorError("not_found", status=404)
        if row["state"] in _TERMINAL_JOB_STATES:
            raise OrchestratorError("invalid_state", status=409)
        deliveries = connection.execute(
            """SELECT d.notification_id,d.occurrence_id,d.state
               FROM scheduler_deliveries d JOIN scheduler_occurrences o USING(occurrence_id)
               WHERE o.job_id=? AND d.state NOT IN ('EXPIRED_UNDELIVERED','REVOKED')""",
            (job_id,),
        ).fetchall()
        for delivery in deliveries:
            if delivery["state"] != "SCHEDULED":
                self._revoke_and_tombstone_tx(
                    connection,
                    notification=delivery["notification_id"],
                    occurrence=delivery["occurrence_id"],
                    reason_code="job_cancelled",
                    now=now,
                )
        connection.execute(
            """UPDATE scheduler_jobs SET state='CANCELLED',updated_at=?,terminal_at=?,content_purge_after=?
               WHERE job_id=? AND job_revision=?""",
            (
                utc_text(now),
                utc_text(now),
                utc_text(now + timedelta(milliseconds=CONTENT_RETENTION_MS)),
                job_id,
                job_revision,
            ),
        )
        connection.execute(
            """UPDATE scheduler_content SET purge_after=? WHERE job_id=? AND purged_at IS NULL
               AND (purge_after IS NULL OR purge_after>?)""",
            (utc_text(now + timedelta(milliseconds=CONTENT_RETENTION_MS)), job_id,
             utc_text(now + timedelta(milliseconds=CONTENT_RETENTION_MS))),
        )
        connection.execute(
            "UPDATE scheduler_occurrences SET state='CANCELLED',updated_at=? WHERE job_id=? AND state='SCHEDULED'",
            (utc_text(now), job_id),
        )
        connection.execute(
            """UPDATE scheduler_deliveries SET state='REVOKED',updated_at=? WHERE occurrence_id IN (
               SELECT occurrence_id FROM scheduler_occurrences WHERE job_id=?)
               AND state NOT IN ('OS_ACCEPTED','EXPIRED_UNDELIVERED','REVOKED')""",
            (utc_text(now), job_id),
        )
        self._ledger_tx(
            connection,
            entity_type="job",
            entity_id=job_id,
            event_type="scheduler.job.cancelled",
            from_state=row["state"],
            to_state="CANCELLED",
            now=now,
            cause_event_id=cause_event_id,
            metadata={"job_revision": job_revision},
        )

    def cancel_job(
        self, job_id: str, job_revision: int, *, cause_event_id: str, now: datetime | None = None
    ) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            self._cancel_job_tx(connection, job_id, job_revision, selected, cause_event_id)

    def apply_notification_action(
        self,
        *,
        event_id: str,
        semantic_sha256: str,
        action: str,
        job_id: str,
        job_revision: int,
        occurrence: str,
        expected_generation: int,
        requested_due_at_utc: str | None,
        now: datetime | None = None,
    ) -> bool:
        """Durably deduplicate and apply a native notification action atomically."""

        require_uuid4(event_id)
        self._sha(semantic_sha256)
        selected = now or utc_now()
        replay_exists, _replay = self._existing_inbox(event_id, semantic_sha256)
        if replay_exists:
            with self._transaction() as connection:
                self._append_outbox_tx(
                    connection,
                    event_type="scheduler.notification.action_result",
                    payload={
                        "action_event_id": event_id,
                        "result": "DUPLICATE",
                        "reason_code": "duplicate_action",
                    },
                    expires_at=selected + timedelta(seconds=30),
                    now=selected,
                    entity_id=event_id,
                )
            return False
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT semantic_sha256 FROM scheduler_inbox WHERE event_id=?", (event_id,)
            ).fetchone()
            if existing is not None:
                if existing["semantic_sha256"] != semantic_sha256:
                    raise OrchestratorError("protocol.conflicting_duplicate", status=409)
                return False
            if action == "CANCEL":
                if requested_due_at_utc is not None:
                    raise OrchestratorError("invalid_request", status=400)
                self._cancel_job_tx(connection, job_id, job_revision, selected, event_id)
            elif action == "SNOOZE" and requested_due_at_utc is not None:
                self._snooze_tx(
                    connection,
                    job_id=job_id,
                    job_revision=job_revision,
                    occurrence=occurrence,
                    snooze_until_at_utc=requested_due_at_utc,
                    cause_event_id=event_id,
                    now=selected,
                    expected_generation=expected_generation,
                )
            else:
                raise OrchestratorError("invalid_request", status=400)
            connection.execute(
                """INSERT INTO scheduler_inbox(
                   event_id,semantic_sha256,protocol,event_type,state,created_at,applied_at
                   ) VALUES(?,?,?,'scheduler.notification.action','APPLIED',?,?)""",
                (event_id, semantic_sha256, SCHEDULER_PROTOCOL, utc_text(selected), utc_text(selected)),
            )
            self._append_outbox_tx(
                connection,
                event_type="scheduler.notification.action_result",
                payload={
                    "action_event_id": event_id,
                    "result": "APPLIED",
                    "reason_code": "action_applied",
                },
                expires_at=selected + timedelta(seconds=30),
                now=selected,
                entity_id=event_id,
            )
        return True

    def record_delivery_receipt(
        self,
        *,
        event_id: str,
        semantic_sha256: str,
        notification: str,
        attempt_id: str,
        stage: str,
        reason_code: str,
        observed_at_utc: str,
    ) -> bool:
        require_uuid4(event_id)
        require_uuid4(attempt_id)
        self._sha(semantic_sha256)
        strict_utc(observed_at_utc)
        allowed = {"STAGED", "DISPATCHING", "OS_ACCEPTED", "OS_REJECTED", "AMBIGUOUS_OS_DISPATCH"}
        if stage not in allowed:
            raise OrchestratorError("invalid_request", status=400)
        selected = parse_utc(observed_at_utc)
        replay_exists, _replay = self._existing_inbox(event_id, semantic_sha256)
        if replay_exists:
            return False
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT semantic_sha256 FROM scheduler_inbox WHERE event_id=?", (event_id,)
            ).fetchone()
            if existing:
                if existing["semantic_sha256"] != semantic_sha256:
                    raise OrchestratorError("protocol.conflicting_duplicate", status=409)
                return False
            delivery = connection.execute(
                "SELECT * FROM scheduler_deliveries WHERE notification_id=?", (notification,)
            ).fetchone()
            if delivery is None:
                raise OrchestratorError("not_found", status=404)
            connection.execute(
                """INSERT INTO scheduler_inbox(
                   event_id,semantic_sha256,protocol,event_type,state,created_at,applied_at
                   ) VALUES(?,?,?,'scheduler.notification.receipt','APPLIED',?,?)""",
                (event_id, semantic_sha256, SCHEDULER_PROTOCOL, utc_text(selected), utc_text(selected)),
            )
            if delivery["state"] in _TERMINAL_DELIVERY_STATES:
                self._ledger_tx(
                    connection,
                    entity_type="delivery",
                    entity_id=notification,
                    event_type="scheduler.delivery.late_receipt_ignored",
                    from_state=delivery["state"],
                    to_state=delivery["state"],
                    now=selected,
                    cause_event_id=event_id,
                    attempt_id=attempt_id,
                    metadata={"reason_code": "terminal_delivery_precedence", "receipt_stage": stage},
                )
                return False
            attempt = connection.execute(
                "SELECT * FROM scheduler_delivery_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                ordinal = connection.execute(
                    "SELECT COUNT(*)+1 FROM scheduler_delivery_attempts WHERE notification_id=?",
                    (notification,),
                ).fetchone()[0]
                initial = "DISPATCHING" if stage == "DISPATCHING" else "CREATED"
                connection.execute(
                    """INSERT INTO scheduler_delivery_attempts(
                       attempt_id,notification_id,attempt_ordinal,state,reason_code,created_at
                       ) VALUES(?,?,?,?,?,?)""",
                    (attempt_id, notification, ordinal, initial, reason_code, utc_text(selected)),
                )
            if stage == "STAGED":
                delivery_state = (
                    "RETRY_WAIT"
                    if delivery["state"] == "BLOCKED_CAPABILITY" and reason_code == "capability_restored"
                    else "STAGED"
                )
            elif stage == "DISPATCHING":
                delivery_state = "DISPATCHING"
            elif stage == "OS_ACCEPTED":
                delivery_state = "OS_ACCEPTED"
                connection.execute(
                    """UPDATE scheduler_occurrences SET action_deadline_at=?,updated_at=?
                       WHERE occurrence_id=?""",
                    (
                        utc_text(selected + timedelta(milliseconds=MISSED_GRACE_MS)),
                        utc_text(selected),
                        delivery["occurrence_id"],
                    ),
                )
            elif stage == "OS_REJECTED" and reason_code in {"permission_denied", "adapter_unavailable", "unsupported"}:
                delivery_state = "BLOCKED_CAPABILITY"
            else:
                delivery_state = "RETRY_WAIT"
            connection.execute(
                "UPDATE scheduler_deliveries SET state=?,updated_at=? WHERE notification_id=?",
                (delivery_state, utc_text(selected), notification),
            )
            if stage in {"OS_ACCEPTED", "OS_REJECTED", "AMBIGUOUS_OS_DISPATCH"}:
                connection.execute(
                    """UPDATE scheduler_delivery_attempts SET state=?,reason_code=?,observed_at=?
                       WHERE attempt_id=?""",
                    (stage, reason_code, utc_text(selected), attempt_id),
                )
            self._ledger_tx(
                connection,
                entity_type="delivery",
                entity_id=notification,
                event_type=f"scheduler.delivery.{stage.lower()}",
                from_state=delivery["state"],
                to_state=delivery_state,
                now=selected,
                cause_event_id=event_id,
                attempt_id=attempt_id,
                metadata={"reason_code": reason_code},
            )
            self._settle_jobs_tx(connection, selected)
            return True

    def _settle_jobs_tx(self, connection: sqlite3.Connection, now: datetime) -> None:
        connection.execute(
            """UPDATE scheduler_occurrences SET state='SETTLED',updated_at=?
               WHERE state='FIRED' AND action_deadline_at IS NOT NULL AND action_deadline_at<=?
               AND NOT EXISTS (
                 SELECT 1 FROM scheduler_deliveries d WHERE d.occurrence_id=scheduler_occurrences.occurrence_id
                 AND d.state NOT IN ('OS_ACCEPTED','EXPIRED_UNDELIVERED','REVOKED')
               )""",
            (utc_text(now), utc_text(now)),
        )
        jobs = connection.execute("SELECT * FROM scheduler_jobs WHERE state='DRAINING'").fetchall()
        for job in jobs:
            pending = connection.execute(
                """SELECT COUNT(*) FROM scheduler_occurrences o WHERE o.job_id=? AND o.state NOT IN (
                   'SETTLED','MISSED_COALESCED','MISSED_EXPIRED','CANCELLED')""",
                (job["job_id"],),
            ).fetchone()[0]
            if pending == 0:
                connection.execute(
                    """UPDATE scheduler_jobs SET state='COMPLETED',updated_at=?,terminal_at=?,content_purge_after=?
                       WHERE job_id=? AND state='DRAINING'""",
                    (
                        utc_text(now),
                        utc_text(now),
                        utc_text(now + timedelta(milliseconds=CONTENT_RETENTION_MS)),
                        job["job_id"],
                    ),
                )
                purge_at = utc_text(now + timedelta(milliseconds=CONTENT_RETENTION_MS))
                connection.execute(
                    """UPDATE scheduler_content SET purge_after=? WHERE job_id=? AND purged_at IS NULL
                       AND (purge_after IS NULL OR purge_after>?)""",
                    (purge_at, job["job_id"], purge_at),
                )
                self._ledger_tx(
                    connection,
                    entity_type="job",
                    entity_id=job["job_id"],
                    event_type="scheduler.job.completed",
                    from_state="DRAINING",
                    to_state="COMPLETED",
                    now=now,
                )

    def expire_deliveries(self, *, now: datetime | None = None) -> int:
        selected = now or utc_now()
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT * FROM scheduler_deliveries WHERE expires_at<=? AND state NOT IN (
                   'OS_ACCEPTED','EXPIRED_UNDELIVERED','REVOKED')""",
                (utc_text(selected),),
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE scheduler_deliveries SET state='EXPIRED_UNDELIVERED',updated_at=? WHERE notification_id=?",
                    (utc_text(selected), row["notification_id"]),
                )
                self._ledger_tx(
                    connection,
                    entity_type="delivery",
                    entity_id=row["notification_id"],
                    event_type="scheduler.delivery.expired",
                    from_state=row["state"],
                    to_state="EXPIRED_UNDELIVERED",
                    now=selected,
                    metadata={"reason_code": "permission_or_capability_unavailable_until_expiry"},
                )
            self._settle_jobs_tx(connection, selected)
            return len(rows)

    def retry_key_blocked_deliveries(self, *, now: datetime | None = None, limit: int = 100) -> int:
        selected = now or utc_now()
        if not 1 <= limit <= MAXIMUM_PAGE_SIZE:
            raise ValueError("limit out of range")
        restored = 0
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT d.*,o.job_id,o.job_revision,o.scheduled_at_utc,o.effective_due_at_utc
                   FROM scheduler_deliveries d JOIN scheduler_occurrences o USING(occurrence_id)
                   WHERE d.state='BLOCKED_KEY_UNAVAILABLE' AND d.expires_at>?
                   ORDER BY d.effective_due_at_utc,d.notification_id LIMIT ?""",
                (utc_text(selected), limit),
            ).fetchall()
            for row in rows:
                try:
                    payload, _job = self._delivery_payload_tx(
                        connection, row, row["delivery_generation"]
                    )
                except SchedulerCryptoError:
                    continue
                record_id = new_uuid4()
                self._put_content_tx(
                    connection, record_type="DELIVERY_PAYLOAD", record_id=record_id,
                    job_id=row["job_id"], job_revision=row["job_revision"], value=payload,
                    now=selected,
                    purge_after=parse_utc(payload["expires_at_utc"]) + timedelta(milliseconds=CONTENT_RETENTION_MS),
                )
                connection.execute(
                    """UPDATE scheduler_deliveries SET state='OUTBOXED',content_record_id=?,updated_at=?
                       WHERE notification_id=? AND state='BLOCKED_KEY_UNAVAILABLE'""",
                    (record_id, utc_text(selected), row["notification_id"]),
                )
                self._append_outbox_tx(
                    connection, event_type="scheduler.notification.requested", payload=payload,
                    expires_at=parse_utc(payload["expires_at_utc"]), now=selected,
                    entity_id=payload["notification_id"],
                )
                self._ledger_tx(
                    connection, entity_type="delivery", entity_id=row["notification_id"],
                    event_type="scheduler.delivery.key_restored",
                    from_state="BLOCKED_KEY_UNAVAILABLE", to_state="OUTBOXED", now=selected,
                    metadata={"reason_code": "scheduler_key_restored"},
                )
                restored += 1
        return restored

    def claim_outbox(
        self,
        owner: str,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        selected = now or utc_now()
        if not 1 <= limit <= MAXIMUM_PAGE_SIZE:
            raise ValueError("limit out of range")
        with self._transaction() as connection:
            connection.execute(
                """UPDATE scheduler_outbox SET state='PENDING',lease_owner=NULL,lease_expires_at=NULL
                   WHERE state='LEASED' AND lease_expires_at<=?""",
                (utc_text(selected),),
            )
            connection.execute(
                "UPDATE scheduler_outbox SET state='EXPIRED' WHERE state IN ('PENDING','LEASED') AND expires_at<=?",
                (utc_text(selected),),
            )
            rows = connection.execute(
                """SELECT * FROM scheduler_outbox WHERE state='PENDING'
                   ORDER BY ordinal,event_id LIMIT ?""",
                (limit,),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                connection.execute(
                    """UPDATE scheduler_outbox SET state='LEASED',lease_owner=?,lease_expires_at=?
                       WHERE event_id=? AND state='PENDING'""",
                    (owner, utc_text(selected + timedelta(seconds=5)), row["event_id"]),
                )
                result.append(
                    {
                        "event_id": row["event_id"],
                        "event_type": row["event_type"],
                        "original_sent_at": row["original_sent_at"],
                        "payload": self._get_content_tx(connection, row["content_record_id"]),
                        "semantic_sha256": row["semantic_sha256"],
                    }
                )
            return result

    def mark_owner_disconnected(self, owner_session_id: str, *, now: datetime | None = None) -> bool:
        selected = now or utc_now()
        with self._transaction() as connection:
            changed = connection.execute(
                """UPDATE approval_arbiter_queue SET state='WAITING_FOR_OWNER',updated_at=?
                   WHERE owner_session_id=? AND state IN ('PRESENTING','PRESENTED')""",
                (utc_text(selected), owner_session_id),
            )
            return changed.rowcount == 1

    def list_jobs(self, *, limit: int = 50, cursor: tuple[str, str] | None = None) -> list[dict[str, Any]]:
        if not 1 <= limit <= MAXIMUM_PAGE_SIZE:
            raise OrchestratorError("invalid_request", status=400)
        query = "SELECT * FROM scheduler_jobs"
        params: list[Any] = []
        if cursor:
            query += " WHERE (created_at<? OR (created_at=? AND job_id<?))"
            params.extend([cursor[0], cursor[0], cursor[1]])
        query += " ORDER BY created_at DESC,job_id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [
            {
                "created_at_utc": row["created_at"],
                "job_id": row["job_id"],
                "job_revision": row["job_revision"],
                "state": row["state"],
                "updated_at_utc": row["updated_at"],
            }
            for row in rows
        ]

    def list_occurrences(
        self,
        job_id: str,
        *,
        start_at_utc: str,
        end_at_utc: str,
        limit: int = 50,
        cursor: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        start, end = strict_utc(start_at_utc), strict_utc(end_at_utc)
        if end < start or end - start > timedelta(days=MAXIMUM_QUERY_HORIZON_DAYS) or not 1 <= limit <= MAXIMUM_PAGE_SIZE:
            raise OrchestratorError("invalid_request", status=400)
        query = """SELECT * FROM scheduler_occurrences WHERE job_id=?
                   AND scheduled_at_utc>=? AND scheduled_at_utc<=?"""
        params: list[Any] = [job_id, start_at_utc, end_at_utc]
        if cursor:
            query += " AND (scheduled_at_utc>? OR (scheduled_at_utc=? AND occurrence_id>?))"
            params.extend([cursor[0], cursor[0], cursor[1]])
        query += " ORDER BY scheduled_at_utc,occurrence_id LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def ledger(self, *, after_seq: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        if after_seq < 0 or not 1 <= limit <= MAXIMUM_PAGE_SIZE:
            raise OrchestratorError("invalid_request", status=400)
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM scheduler_ledger WHERE ledger_seq>?
                   ORDER BY ledger_seq LIMIT ?""",
                (after_seq, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def purge_content(self, *, now: datetime | None = None) -> int:
        selected = now or utc_now()
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT record_id FROM scheduler_content WHERE purged_at IS NULL AND purge_after IS NOT NULL AND purge_after<=?",
                (utc_text(selected),),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """UPDATE scheduler_content SET nonce=?,ciphertext_and_tag=x'',purged_at=?
                       WHERE record_id=? AND purged_at IS NULL""",
                    (os.urandom(12), utc_text(selected), row["record_id"]),
                )
            return len(rows)

    def rotate_content_key(self, *, batch_size: int = 100, start_new_version: bool = True) -> dict[str, int]:
        if not 1 <= batch_size <= MAXIMUM_PAGE_SIZE:
            raise OrchestratorError("invalid_request", status=400)
        target_version = (
            self.cipher.rotate_key() if start_new_version else self.cipher.provider.active_version()
        )
        rotated = 0
        while True:
            with self._transaction() as connection:
                rows = connection.execute(
                    """SELECT * FROM scheduler_content WHERE purged_at IS NULL AND key_version<>?
                       ORDER BY created_at,record_id LIMIT ?""",
                    (target_version, batch_size),
                ).fetchall()
                for row in rows:
                    binding = ContentBinding(
                        store_role="orchestrator", record_type=row["record_type"],
                        record_id=row["record_id"], job_id=row["job_id"],
                        job_revision=row["job_revision"],
                    )
                    plaintext = self.cipher.decrypt(
                        EncryptedContent(row["key_version"], bytes(row["nonce"]),
                                         bytes(row["ciphertext_and_tag"]), row["content_sha256"],
                                         row["aad_sha256"]),
                        binding,
                    )
                    encrypted = self.cipher.encrypt(plaintext, binding)
                    if encrypted.key_version != target_version:
                        raise OrchestratorError("scheduler_key_rotation_race", status=503)
                    connection.execute(
                        """UPDATE scheduler_content SET key_version=?,nonce=?,ciphertext_and_tag=?,
                           content_sha256=?,aad_sha256=? WHERE record_id=? AND key_version=?""",
                        (encrypted.key_version, encrypted.nonce, encrypted.ciphertext_and_tag,
                         encrypted.content_sha256, encrypted.aad_sha256, row["record_id"],
                         row["key_version"]),
                    )
                rotated += len(rows)
            if len(rows) < batch_size:
                break
        return {"key_version": target_version, "rotated_record_count": rotated}

    def readiness(self) -> dict[str, Any]:
        with self._lock:
            version = self._connection.execute(
                "SELECT value FROM schema_meta WHERE key='scheduler_schema_version'"
            ).fetchone()
            active_jobs = self._connection.execute(
                "SELECT COUNT(*) FROM scheduler_jobs WHERE state IN ('ACTIVE','DRAINING')"
            ).fetchone()[0]
        state = "READY"
        reason = "ready"
        if not version or version["value"] != "5":
            state, reason = "BLOCKED", "scheduler_schema_mismatch"
        else:
            try:
                from .scheduler_contract import validate_runtime
                validate_runtime()
                self.zones.load("Europe/Athens")
                self.cipher.probe()
            except Exception:
                state, reason = "BLOCKED", "scheduler_key_timezone_or_runtime_unavailable"
        return {
            "active_job_count": active_jobs,
            "reason_code": reason,
            "state": state,
        }
