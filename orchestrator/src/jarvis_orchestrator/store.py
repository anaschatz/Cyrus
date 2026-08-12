"""SQLite WAL persistence, transactional epoch fences, budgets and outbox."""

from __future__ import annotations

import hashlib
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
from .cloud_contract import (
    MONTHLY_HARD_CAP_MICROUSD,
    CloudApprovalBindingV1,
    QualifiedEndpointV1,
    build_dispatch_body,
)
from .cloud_routing import RoutePlanV1, RoutingFactsV1, TaskProfileV1
from .jcs import domain_sha256, jcs_text
from .memory_release import shadow_safety_fields, validate_shadow_audit_payload
from .research import (
    MAX_RUN_DOCUMENT_BYTES,
    EvidenceBundleV1,
    NormalizedDocumentV1,
    SearchPlanV1,
    reconstruct,
    validate_syntactic_citation_coverage,
)
from .scheduler_store import MIGRATION_5_SQL
from .tool_store import MIGRATION_6_SQL
from .tool_provenance import (
    ContentProvenanceV1,
    ContentSourceKind,
    EgressDisposition,
    PolicyLabel,
    ProvenanceError,
)

SCHEMA_VERSION = 7


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


MIGRATION_3_SQL = """
CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    plan_json TEXT NOT NULL,
    plan_sha256 TEXT NOT NULL,
    approval_id TEXT UNIQUE REFERENCES approvals(approval_id),
    stage TEXT NOT NULL CHECK(stage IN (
        'AWAITING_APPROVAL','APPROVED','DENIED','RUNNING','COMPLETED','DISCARDED'
    )),
    status TEXT CHECK(status IS NULL OR status IN (
        'SUFFICIENT','PARTIAL','INSUFFICIENT','FAILED'
    )),
    bundle_json TEXT,
    bundle_sha256 TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK((bundle_json IS NULL) = (bundle_sha256 IS NULL))
);
CREATE INDEX IF NOT EXISTS research_runs_by_conversation
ON research_runs(conversation_id, created_at, run_id);
CREATE TABLE IF NOT EXISTS research_documents (
    run_id TEXT NOT NULL REFERENCES research_runs(run_id),
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    source_id TEXT NOT NULL,
    source_json TEXT NOT NULL,
    document_json TEXT NOT NULL,
    document_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(run_id, source_id)
);
CREATE INDEX IF NOT EXISTS research_documents_by_conversation
ON research_documents(conversation_id, run_id, source_id);
"""


MIGRATION_4_SQL = """
CREATE TABLE IF NOT EXISTS task_profiles (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    profile_json TEXT NOT NULL,
    profile_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routing_facts (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    facts_json TEXT NOT NULL,
    facts_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS route_plans (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    plan_json TEXT NOT NULL,
    plan_sha256 TEXT NOT NULL,
    target TEXT NOT NULL CHECK(target IN ('local','deepseek','terra','capability_failure')),
    reason_code TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS endpoint_qualifications (
    qualification_sha256 TEXT PRIMARY KEY,
    model_canonical_id TEXT NOT NULL,
    provider_endpoint_tag TEXT NOT NULL,
    quantization TEXT NOT NULL,
    parameter_profile_sha256 TEXT NOT NULL,
    qualification_version TEXT NOT NULL,
    stable_catalog_sha256 TEXT NOT NULL,
    artifact_json TEXT NOT NULL,
    enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
    created_at TEXT NOT NULL,
    UNIQUE(model_canonical_id,provider_endpoint_tag,quantization,parameter_profile_sha256,qualification_version)
);
CREATE TABLE IF NOT EXISTS endpoint_health_snapshots (
    health_sha256 TEXT PRIMARY KEY,
    qualification_sha256 TEXT NOT NULL REFERENCES endpoint_qualifications(qualification_sha256),
    snapshot_json TEXT NOT NULL,
    observed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS endpoint_health_by_qualification
ON endpoint_health_snapshots(qualification_sha256,observed_at DESC);
CREATE TABLE IF NOT EXISTS cloud_approval_bindings (
    approval_id TEXT PRIMARY KEY REFERENCES approvals(approval_id),
    task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    epoch INTEGER NOT NULL,
    binding_json TEXT NOT NULL,
    binding_sha256 TEXT NOT NULL UNIQUE,
    dispatch_body_json TEXT NOT NULL,
    dispatch_body_sha256 TEXT NOT NULL,
    egress_sha256 TEXT NOT NULL,
    route_plan_sha256 TEXT NOT NULL,
    stable_catalog_sha256 TEXT NOT NULL,
    qualification_sha256 TEXT NOT NULL,
    invalidation_reason TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cloud_presentations (
    presentation_id TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL REFERENCES cloud_approval_bindings(approval_id),
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    owner_session_id TEXT NOT NULL,
    decision_nonce TEXT NOT NULL,
    requested_event_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK(state IN ('PENDING','DECIDED','INVALIDATED','SUPERSEDED')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(approval_id,owner_session_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_cloud_presentation
ON cloud_presentations(approval_id) WHERE state='PENDING';
CREATE TABLE IF NOT EXISTS cloud_extension_inbox (
    event_id TEXT PRIMARY KEY,
    semantic_sha256 TEXT NOT NULL,
    approval_id TEXT,
    presentation_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cloud_extension_outbox (
    event_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    task_id TEXT REFERENCES tasks(task_id),
    task_epoch INTEGER,
    guard_epoch INTEGER NOT NULL,
    approval_id TEXT,
    presentation_id TEXT,
    ordinal INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    original_sent_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('REPLAYABLE','ACKED','NON_REPLAYABLE','STALE')),
    next_attempt_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    expires_at TEXT NOT NULL,
    acked_event_id TEXT,
    ack_status TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS stable_cloud_decision_ack
ON cloud_extension_outbox(acked_event_id,event_type) WHERE acked_event_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS cloud_extension_replay_order
ON cloud_extension_outbox(conversation_id,state,ordinal);
CREATE TABLE IF NOT EXISTS cloud_cost_reservations (
    reservation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    approval_id TEXT NOT NULL REFERENCES cloud_approval_bindings(approval_id),
    utc_month TEXT NOT NULL,
    model_canonical_id TEXT NOT NULL,
    provider_endpoint_tag TEXT NOT NULL,
    pricing_sha256 TEXT NOT NULL,
    reserved_microusd INTEGER NOT NULL CHECK(reserved_microusd >= 0),
    actual_microusd INTEGER,
    state TEXT NOT NULL CHECK(state IN ('RESERVED','DISPATCHED','RECONCILED','AMBIGUOUS','CANCELLED')),
    created_at TEXT NOT NULL,
    dispatched_at TEXT,
    reconciled_at TEXT
);
CREATE INDEX IF NOT EXISTS cloud_cost_by_month
ON cloud_cost_reservations(utc_month,state);
CREATE TABLE IF NOT EXISTS cloud_dispatch_permits (
    permit_id TEXT PRIMARY KEY,
    reservation_id TEXT NOT NULL UNIQUE REFERENCES cloud_cost_reservations(reservation_id),
    task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    dispatch_body_sha256 TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('ISSUED','CONSUMED','CANCELLED')),
    created_at TEXT NOT NULL,
    consumed_at TEXT
);
CREATE TABLE IF NOT EXISTS cloud_provider_receipts (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    generation_id TEXT,
    receipt_json TEXT,
    receipt_sha256 TEXT,
    provider_identity_verified INTEGER NOT NULL CHECK(provider_identity_verified IN (0,1)),
    created_at TEXT NOT NULL
);
"""


MIGRATION_7_SQL = """
CREATE TABLE IF NOT EXISTS planner_routes (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    route_json TEXT NOT NULL,
    route_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scheduler_task_links (
    proposal_id TEXT PRIMARY KEY REFERENCES scheduler_proposals(proposal_id),
    task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('PENDING','APPROVED','DENIED','EXPIRED')),
    result_json TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS scheduler_task_links_by_task
ON scheduler_task_links(task_id,epoch,state);
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
        tool_authority_enabled: bool = False,
    ) -> None:
        if not isinstance(tool_authority_enabled, bool):
            raise TypeError("tool_authority_enabled must be boolean")
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
        self.tool_authority_enabled = tool_authority_enabled
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
            if existing_version is not None and existing_version["value"] not in {"1", "2", "3", "4", "5", "6", "7"}:
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
                current_version = 2
            if current_version == 2:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + MIGRATION_3_SQL
                    + "\nUPDATE schema_meta SET value='3' WHERE key='schema_version' AND value='2';\n"
                    + "COMMIT;"
                )
                current_version = 3
            if current_version == 3:
                task_columns = {
                    row["name"]
                    for row in self._connection.execute("PRAGMA table_info(tasks)").fetchall()
                }
                add_dispatch_count = (
                    "ALTER TABLE tasks ADD COLUMN cloud_dispatch_count INTEGER NOT NULL DEFAULT 0 "
                    "CHECK(cloud_dispatch_count >= 0 AND cloud_dispatch_count <= 1);\n"
                    if "cloud_dispatch_count" not in task_columns
                    else ""
                )
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + add_dispatch_count
                    + MIGRATION_4_SQL
                    + "\nUPDATE schema_meta SET value='4' WHERE key='schema_version' AND value='3';\n"
                    + "COMMIT;"
                )
                current_version = 4
            if current_version == 4:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + MIGRATION_5_SQL
                    + "\nUPDATE schema_meta SET value='5' WHERE key='schema_version' AND value='4';\n"
                    + "COMMIT;"
                )
                current_version = 5
            if current_version == 5:
                tool_columns = {
                    row["name"]
                    for row in self._connection.execute(
                        "PRAGMA table_info(tool_executions)"
                    ).fetchall()
                }
                migration_6_lines: list[str] = []
                for statement_line in MIGRATION_6_SQL.splitlines():
                    stripped = statement_line.strip()
                    if stripped.startswith("ALTER TABLE tool_executions ADD COLUMN "):
                        column_name = stripped.split()[5]
                        if column_name in tool_columns:
                            continue
                    migration_6_lines.append(statement_line)
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + "\n".join(migration_6_lines)
                    + "\nUPDATE schema_meta SET value='6' WHERE key='schema_version' AND value='5';\n"
                    + "COMMIT;"
                )
                current_version = 6
            if current_version == 6:
                self._connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + MIGRATION_7_SQL
                    + "\nUPDATE schema_meta SET value='7' WHERE key='schema_version' AND value='6';\n"
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

    def persist_planner_route(
        self,
        task_id: str,
        epoch: int,
        route: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Persist the validated planner decision before any capability work."""

        selected = now or utc_now()
        document = dict(route)
        encoded = canonical_json(document)
        digest = canonical_sha256(document)
        with self._transaction() as connection:
            self._guarded_task_tx(connection, task_id, epoch, (TaskState.ROUTING,))
            existing = connection.execute(
                "SELECT route_json,route_sha256 FROM planner_routes WHERE task_id=? AND epoch=?",
                (task_id, epoch),
            ).fetchone()
            if existing is not None:
                if existing["route_sha256"] != digest or existing["route_json"] != encoded:
                    raise OrchestratorError("route_validation_failed", status=409)
                return json.loads(existing["route_json"])
            connection.execute(
                "INSERT INTO planner_routes(task_id,epoch,route_json,route_sha256,created_at) VALUES(?,?,?,?,?)",
                (task_id, epoch, encoded, digest, utc_text(selected)),
            )
            self._append_task_event_tx(
                connection,
                task_id,
                epoch,
                "planner.route.selected",
                {"route": document.get("route", ""), "route_sha256": digest},
                selected,
            )
        return document

    def planner_route(self, task_id: str, epoch: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT route_json FROM planner_routes WHERE task_id=? AND epoch=?",
                (task_id, epoch),
            ).fetchone()
        return json.loads(row["route_json"]) if row is not None else None

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
            **shadow_safety_fields(),
            "status": "pending",
            "source_event_id": source_event_id,
            "text_sha256": text_sha256,
        }
        with self._transaction() as connection:
            task = connection.execute(
                """SELECT k.epoch,k.state,t.user_text,i.event_id
                   FROM tasks k JOIN canonical_turns t USING(task_id)
                   JOIN inbox i USING(task_id)
                   WHERE k.task_id=?""",
                (task_id,),
            ).fetchone()
            if task is None:
                raise OrchestratorError("not_found", status=404)
            if task["epoch"] != epoch or task["state"] != TaskState.COMPLETED.value:
                raise StaleEpochError()
            if (
                source_event_id != task["event_id"]
                or text_sha256
                != hashlib.sha256(task["user_text"].encode("utf-8")).hexdigest()
            ):
                raise ValueError("memory shadow source binding is invalid")
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
        validate_shadow_audit_payload(event_type, payload)
        selected = now or utc_now()
        with self._transaction() as connection:
            pending_row = connection.execute(
                """SELECT e.event_sequence,e.payload_json,t.user_text,i.event_id
                   FROM task_events e JOIN canonical_turns t USING(task_id)
                   JOIN inbox i USING(task_id)
                   WHERE e.task_id=? AND e.epoch=?
                     AND e.event_type='memory.shadow.pending'""",
                (task_id, epoch),
            ).fetchone()
            if pending_row is None:
                raise OrchestratorError("memory_shadow_audit_missing", status=409)
            pending = json.loads(pending_row["payload_json"])
            expected_pending = {
                **shadow_safety_fields(),
                "status": "pending",
                "source_event_id": pending_row["event_id"],
                "text_sha256": hashlib.sha256(
                    pending_row["user_text"].encode("utf-8")
                ).hexdigest(),
            }
            if (
                pending != expected_pending
                or payload.get("source_event_id") != pending["source_event_id"]
                or payload.get("text_sha256") != pending["text_sha256"]
            ):
                raise OrchestratorError(
                    "memory_shadow_audit_binding_invalid", status=409
                )
            changed = connection.execute(
                """UPDATE task_events SET event_type=?,payload_json=?,created_at=?
                   WHERE event_sequence=? AND event_type='memory.shadow.pending'""",
                (
                    event_type,
                    canonical_json(payload),
                    utc_text(selected),
                    pending_row["event_sequence"],
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

    def shadow_authority_fingerprint(self, task_id: str) -> str:
        """Hash task-scoped authority that a shadow writer must never change.

        Task scope avoids false positives if the next turn is admitted while
        this completed turn's bounded shadow observation is still running.
        """

        with self._lock:
            projection = {
                "memory_records": [
                    dict(row)
                    for row in self._connection.execute(
                        """SELECT record_id,task_id,source_event_id,scoped_user_id,
                                  content_sha256,metadata_json,created_at
                           FROM memory_records WHERE task_id=?
                           ORDER BY record_id""",
                        (task_id,),
                    ).fetchall()
                ],
                "approvals": [
                    dict(row)
                    for row in self._connection.execute(
                        """SELECT approval_id,task_id,epoch,kind,name,
                                  arguments_sha256,status,created_at,expires_at,decided_at
                           FROM approvals WHERE task_id=?
                           ORDER BY approval_id""",
                        (task_id,),
                    ).fetchall()
                ],
                "capability_reservations": [
                    dict(row)
                    for row in self._connection.execute(
                        """SELECT r.reservation_id,r.task_id,r.epoch,r.kind,
                                  r.request_key,r.arguments_sha256,r.status,
                                  r.created_at,r.completed_at
                           FROM capability_reservations r
                           WHERE r.task_id=?
                           ORDER BY r.reservation_id""",
                        (task_id,),
                    ).fetchall()
                ],
                "tool_executions": [
                    dict(row)
                    for row in self._connection.execute(
                        """SELECT x.execution_id,x.task_id,x.epoch,x.tool_call_id,
                                  x.tool_name,x.arguments_sha256,x.status,
                                  x.result_sha256,x.created_at,x.completed_at
                           FROM tool_executions x
                           WHERE x.task_id=?
                           ORDER BY x.execution_id""",
                        (task_id,),
                    ).fetchall()
                ],
            }
        return canonical_sha256(projection)

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
            """SELECT COALESCE(MAX(ordinal),0)+1 AS next FROM (
                   SELECT ordinal FROM outbox WHERE conversation_id=?
                   UNION ALL
                   SELECT ordinal FROM cloud_extension_outbox WHERE conversation_id=?
               )""",
            (conversation_id, conversation_id),
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

    def _append_cloud_outbox_tx(
        self,
        connection: sqlite3.Connection,
        *,
        conversation_id: str,
        guard_epoch: int,
        event_type: str,
        payload: Mapping[str, Any],
        expires_at: str,
        task_id: str | None = None,
        task_epoch: int | None = None,
        approval_id: str | None = None,
        presentation_id: str | None = None,
        event_id: str | None = None,
        acked_event_id: str | None = None,
        ack_status: str | None = None,
        now: datetime | None = None,
    ) -> str:
        logical_id = event_id or new_uuid4()
        stamp = utc_text(now)
        connection.execute(
            """INSERT INTO cloud_extension_outbox(
               event_id,conversation_id,task_id,task_epoch,guard_epoch,approval_id,
               presentation_id,ordinal,event_type,payload_json,original_sent_at,state,
               next_attempt_at,expires_at,acked_event_id,ack_status
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,'REPLAYABLE',?,?,?,?)""",
            (
                logical_id,
                conversation_id,
                task_id,
                task_epoch,
                guard_epoch,
                approval_id,
                presentation_id,
                self._next_ordinal(connection, conversation_id),
                event_type,
                jcs_text(payload),
                stamp,
                stamp,
                expires_at,
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
        cloud_invalidation_reason = (
            "superseded"
            if reason == ReasonCode.SUPERSEDED
            else "approval_expired"
            if reason == ReasonCode.APPROVAL_EXPIRED
            else "task_cancelled"
        )
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
        pending_presentations = connection.execute(
            """SELECT p.presentation_id,p.approval_id,b.binding_sha256,b.expires_at
               FROM cloud_presentations p
               JOIN cloud_approval_bindings b USING(approval_id)
               WHERE p.task_id=? AND p.epoch=? AND p.state='PENDING'""",
            (row["task_id"], row["epoch"]),
        ).fetchall()
        connection.execute(
            """UPDATE cloud_approval_bindings SET invalidation_reason=?
               WHERE task_id=? AND epoch=? AND invalidation_reason IS NULL""",
            (
                cloud_invalidation_reason,
                row["task_id"],
                row["epoch"],
            ),
        )
        connection.execute(
            """UPDATE cloud_presentations SET state='INVALIDATED',updated_at=?
               WHERE task_id=? AND epoch=? AND state='PENDING'""",
            (utc_text(now), row["task_id"], row["epoch"]),
        )
        connection.execute(
            """UPDATE approval_arbiter_queue SET state='INVALIDATED',updated_at=?
               WHERE approval_id IN (
                 SELECT approval_id FROM cloud_approval_bindings WHERE task_id=? AND epoch=?
               ) AND state IN ('WAITING_FOR_CHANNEL','PRESENTING','PRESENTED','WAITING_FOR_OWNER')""",
            (utc_text(now), row["task_id"], row["epoch"]),
        )
        connection.execute(
            """UPDATE cloud_extension_outbox SET state='STALE'
               WHERE task_id=? AND task_epoch=? AND state='REPLAYABLE'""",
            (row["task_id"], row["epoch"]),
        )
        for presentation in pending_presentations:
            self._append_cloud_outbox_tx(
                connection,
                conversation_id=row["conversation_id"],
                task_id=row["task_id"],
                task_epoch=row["epoch"],
                guard_epoch=guard_epoch,
                approval_id=presentation["approval_id"],
                presentation_id=presentation["presentation_id"],
                event_type="cloud.approval.invalidated",
                payload={
                    "approval_binding_sha256": presentation["binding_sha256"],
                    "approval_id": presentation["approval_id"],
                    "presentation_id": presentation["presentation_id"],
                    "reason_code": cloud_invalidation_reason,
                },
                expires_at=utc_text(now + self.delivery_ttl),
                now=now,
            )
        connection.execute(
            """UPDATE cloud_cost_reservations SET state='CANCELLED',reconciled_at=?
               WHERE task_id=? AND epoch=? AND state='RESERVED'""",
            (utc_text(now), row["task_id"], row["epoch"]),
        )
        connection.execute(
            """UPDATE cloud_cost_reservations
               SET state='AMBIGUOUS',actual_microusd=reserved_microusd,reconciled_at=?
               WHERE task_id=? AND epoch=? AND state='DISPATCHED'""",
            (utc_text(now), row["task_id"], row["epoch"]),
        )
        connection.execute(
            """UPDATE cloud_dispatch_permits SET state='CANCELLED'
               WHERE task_id=? AND epoch=? AND state='ISSUED'""",
            (row["task_id"], row["epoch"]),
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
            """UPDATE tool_executions SET status='CANCELLED',reason_code='tool.cancelled_pre_dispatch',
               completed_at=?,updated_at=? WHERE task_id=? AND epoch=?
               AND status IN ('PLANNED','AWAITING_APPROVAL','APPROVED')""",
            (utc_text(now), utc_text(now), row["task_id"], row["epoch"]),
        )
        connection.execute(
            """UPDATE tool_executions SET status='AMBIGUOUS',reason_code='tool.ambiguous_after_dispatch',
               completed_at=?,updated_at=? WHERE task_id=? AND epoch=?
               AND status IN ('DISPATCH_COMMITTED','DISPATCHED','VERIFYING')""",
            (utc_text(now), utc_text(now), row["task_id"], row["epoch"]),
        )
        connection.execute(
            """UPDATE approval_arbiter_queue SET state='INVALIDATED',updated_at=?
               WHERE approval_id IN (
                 SELECT approval_id FROM approvals WHERE task_id=? AND epoch=?
               ) AND state IN ('WAITING_FOR_CHANNEL','PRESENTING','PRESENTED','DECIDING','WAITING_FOR_OWNER')""",
            (utc_text(now), row["task_id"], row["epoch"]),
        )
        scheduler_links = connection.execute(
            """SELECT l.proposal_id,p.approval_id FROM scheduler_task_links l
               JOIN scheduler_proposals p USING(proposal_id)
               WHERE l.task_id=? AND l.epoch=? AND l.state='PENDING'""",
            (row["task_id"], row["epoch"]),
        ).fetchall()
        for link in scheduler_links:
            connection.execute(
                "UPDATE scheduler_task_links SET state='EXPIRED',resolved_at=? WHERE proposal_id=?",
                (utc_text(now), link["proposal_id"]),
            )
            connection.execute(
                "UPDATE scheduler_proposals SET state='EXPIRED',decided_at=? WHERE proposal_id=? AND state='PENDING'",
                (utc_text(now), link["proposal_id"]),
            )
            connection.execute(
                "UPDATE approval_arbiter_queue SET state='INVALIDATED',updated_at=? WHERE approval_id=?",
                (utc_text(now), link["approval_id"]),
            )
        connection.execute(
            """UPDATE research_runs SET stage='DISCARDED',updated_at=?
               WHERE task_id=? AND epoch=? AND stage!='COMPLETED'""",
            (utc_text(now), row["task_id"], row["epoch"]),
        )
        connection.execute(
            "UPDATE outbox SET guard_epoch=? WHERE response_id=? AND event_type='server.ack'",
            (guard_epoch, row["response_id"]),
        )
        connection.execute(
            "UPDATE outbox SET state='STALE',claimed_by=NULL,claim_expires_at=NULL WHERE response_id=? AND event_type NOT IN ('server.ack','server.error','response.cancelled','response.speech.done')",
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
                    task_binding = connection.execute(
                        "SELECT response_id,epoch FROM tasks WHERE task_id=?",
                        (existing["task_id"],),
                    ).fetchone()
                    ack_id = self._append_outbox_tx(
                        connection,
                        conversation_id=conversation["conversation_id"],
                        guard_epoch=conversation["current_epoch"],
                        task_id=existing["task_id"],
                        task_epoch=(
                            int(task_binding["epoch"])
                            if task_binding is not None
                            else None
                        ),
                        turn_id=existing["turn_id"],
                        response_id=(
                            task_binding["response_id"]
                            if task_binding is not None
                            else None
                        ),
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
                response_id=response_id,
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
            turn = connection.execute(
                "SELECT turn_id,user_text FROM canonical_turns WHERE task_id=? AND status='PENDING'",
                (task_id,),
            ).fetchone()
            if turn is None:
                raise InvalidStateError()
            provenance = self._derive_response_provenance_tx(
                connection,
                task_id=task_id,
                conversation_id=row["conversation_id"],
                turn_id=turn["turn_id"],
                user_text=turn["user_text"],
                assistant_text=assistant_text,
            )
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
                """INSERT INTO content_provenance(
                   entity_kind,entity_id,provenance_json,provenance_sha256,
                   policy_label,egress_disposition,created_at
                   ) VALUES('canonical_turn',?,?,?,?,?,?)""",
                (
                    turn["turn_id"],
                    jcs_text(provenance.to_dict()),
                    provenance.sha256,
                    provenance.policy_label.value,
                    provenance.egress.value,
                    utc_text(selected),
                ),
            )
            connection.execute(
                "UPDATE conversations SET revision=revision+1,updated_at=? WHERE conversation_id=? AND current_epoch=?",
                (utc_text(selected), row["conversation_id"], epoch),
            )
            self._append_task_event_tx(connection, task_id, epoch, "COMPLETED", {}, selected)
            return self._task_from_row(connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone())

    def _derive_response_provenance_tx(
        self,
        connection: sqlite3.Connection,
        *,
        task_id: str,
        conversation_id: str,
        turn_id: str,
        user_text: str,
        assistant_text: str,
    ) -> ContentProvenanceV1:
        """Derive output labels in code; callers and models cannot downgrade them."""

        inputs: list[ContentProvenanceV1] = [
            ContentProvenanceV1.source(ContentSourceKind.USER_INPUT, turn_id, user_text)
        ]
        prior = connection.execute(
            """SELECT p.provenance_json,p.policy_label,p.egress_disposition,ct.turn_id,ct.assistant_text
               FROM canonical_turns ct
               LEFT JOIN content_provenance p
                 ON p.entity_kind='canonical_turn' AND p.entity_id=ct.turn_id
               WHERE ct.conversation_id=? AND ct.status='COMPLETED'
               ORDER BY ct.created_at,ct.turn_id""",
            (conversation_id,),
        ).fetchall()
        for item in prior:
            try:
                if item["provenance_json"] is None:
                    raise ProvenanceError("missing provenance")
                inputs.append(ContentProvenanceV1.from_dict(json.loads(item["provenance_json"])))
            except (json.JSONDecodeError, ProvenanceError):
                inputs.append(
                    ContentProvenanceV1.source(
                        ContentSourceKind.LEGACY_UNVERIFIED,
                        item["turn_id"],
                        item["assistant_text"],
                    )
                )

        event_types = {
            event["event_type"]
            for event in connection.execute(
                "SELECT event_type FROM task_events WHERE task_id=?", (task_id,)
            ).fetchall()
        }
        if any(value.startswith("tool.") for value in event_types):
            # This rule covers model text, adapter summaries and deterministic
            # templates alike: every canonical output of a tool task is private.
            return ContentProvenanceV1.tool_task_output(
                task_id=task_id,
                content=assistant_text,
                inputs=inputs,
                deterministic=True,
            )
        if any(value.startswith("memory.recall.") for value in event_types):
            inputs.append(
                ContentProvenanceV1.source(
                    ContentSourceKind.MEMORY_RECALL, task_id, assistant_text
                )
            )
        if any(value.startswith("research.") for value in event_types):
            inputs.append(
                ContentProvenanceV1.source(
                    ContentSourceKind.RESEARCH_EVIDENCE, task_id, assistant_text
                )
            )
        inputs.append(
            ContentProvenanceV1.source(
                ContentSourceKind.LOCAL_MODEL, task_id, assistant_text
            )
        )
        return ContentProvenanceV1.derive(inputs)

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

    def record_internal_diagnostic(
        self, task_id: str, expected_epoch: int, code: str, *, now: datetime | None = None
    ) -> None:
        """Persist one closed, content-free failure classification."""

        allowed = {
            "attribute_error",
            "key_error",
            "lookup_error",
            "os_error",
            "runtime_error",
            "type_error",
            "value_error",
            "other_error",
        }
        if code not in allowed:
            raise ValueError("diagnostic code is invalid")
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT t.state FROM tasks t JOIN conversations c USING(conversation_id)
                   WHERE t.task_id=? AND t.epoch=? AND t.tombstoned=0
                   AND c.current_epoch=? AND c.state='ACTIVE'""",
                (task_id, expected_epoch, expected_epoch),
            ).fetchone()
            if row is None or TaskState(row["state"]) in TERMINAL_STATES:
                raise StaleEpochError()
            self._append_task_event_tx(
                connection,
                task_id,
                expected_epoch,
                "diagnostic.internal_failure",
                {"code": code},
                selected,
            )

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
        if kind == "tool" and not self.tool_authority_enabled:
            raise OrchestratorError("forbidden", status=403)
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
            if approval["kind"] == "research":
                connection.execute(
                    """UPDATE research_runs SET stage=?,updated_at=?
                       WHERE approval_id=? AND stage='AWAITING_APPROVAL'""",
                    (
                        "APPROVED" if approval_status == ApprovalStatus.APPROVED else "DENIED",
                        utc_text(selected),
                        approval_id,
                    ),
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

    def cloud_safe_history(self, conversation_id: str, *, limit_pairs: int = 4) -> tuple[CanonicalTurnV1, ...]:
        """Completed pairs with no memory, research or tool-derived content."""

        with self._lock:
            rows = self._connection.execute(
                """SELECT ct.* FROM canonical_turns ct
                   JOIN content_provenance p
                     ON p.entity_kind='canonical_turn' AND p.entity_id=ct.turn_id
                   WHERE ct.conversation_id=? AND ct.status='COMPLETED'
                   AND p.policy_label='STANDARD'
                   AND p.egress_disposition='REVIEWABLE'
                   AND length(p.provenance_sha256)=64
                   AND ct.effects_json='[]'
                   AND NOT EXISTS (
                       SELECT 1 FROM task_events e WHERE e.task_id=ct.task_id AND (
                         e.event_type LIKE 'memory.recall.%'
                         OR e.event_type LIKE 'research.%'
                         OR e.event_type LIKE 'tool.%'
                       )
                   )
                   AND NOT EXISTS (SELECT 1 FROM research_runs r WHERE r.task_id=ct.task_id)
                   ORDER BY ct.created_at DESC,ct.turn_id DESC LIMIT ?""",
                (conversation_id, limit_pairs),
            ).fetchall()
        return tuple(
            CanonicalTurnV1(
                row["turn_id"],
                row["user_text"],
                row["user_language"],
                row["assistant_text"],
                (),
            )
            for row in reversed(rows)
        )

    def cloud_run(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT b.*,a.status AS approval_status,a.decided_at
                   FROM cloud_approval_bindings b JOIN approvals a USING(approval_id)
                   WHERE b.task_id=?""",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["binding"] = json.loads(result.pop("binding_json"))
        result["dispatch_body"] = json.loads(result.pop("dispatch_body_json"))
        return result

    def approval_kind(self, approval_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT kind FROM approvals WHERE approval_id=?", (approval_id,)
            ).fetchone()
        return None if row is None else str(row["kind"])

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
        limits = {"tool": 4, "research": 1, "cloud_model": 0}
        if kind not in limits:
            raise ValueError("invalid capability kind")
        if kind == "tool" and not self.tool_authority_enabled:
            raise OrchestratorError("forbidden", status=403)
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

    def create_research_approval(
        self,
        task_id: str,
        epoch: int,
        plan: SearchPlanV1,
        *,
        now: datetime | None = None,
        ttl: timedelta | None = None,
    ) -> dict[str, str]:
        """Persist the immutable plan and approval in one transaction.

        No research adapter may be invoked before this row exists and its
        linked approval has become APPROVED.
        """

        selected = now or utc_now()
        plan_json = canonical_json(asdict(plan))
        with self._transaction() as connection:
            row = self._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            existing = connection.execute(
                "SELECT run_id,plan_sha256,approval_id FROM research_runs WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if existing is not None:
                if existing["plan_sha256"] != plan.plan_hash:
                    raise OrchestratorError("research_plan_changed", status=409)
                return {
                    "run_id": existing["run_id"],
                    "approval_id": existing["approval_id"],
                    "plan_hash": existing["plan_sha256"],
                }
            run_id = new_uuid4()
            approval_id = new_uuid4()
            arguments = plan.approval_arguments
            arguments_json = canonical_json(arguments)
            expires = min(selected + (ttl or self.approval_ttl), parse_utc(row["absolute_expires_at"]))
            connection.execute(
                """INSERT INTO approvals(
                   approval_id,conversation_id,task_id,epoch,kind,name,
                   arguments_json,arguments_sha256,status,created_at,expires_at
                   ) VALUES(?,?,?,?,?,'web-research',?,?,'PENDING',?,?)""",
                (
                    approval_id,
                    row["conversation_id"],
                    task_id,
                    epoch,
                    "research",
                    arguments_json,
                    canonical_sha256(arguments),
                    utc_text(selected),
                    utc_text(expires),
                ),
            )
            connection.execute(
                """INSERT INTO research_runs(
                   run_id,conversation_id,task_id,epoch,plan_json,plan_sha256,
                   approval_id,stage,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,'AWAITING_APPROVAL',?,?)""",
                (
                    run_id,
                    row["conversation_id"],
                    task_id,
                    epoch,
                    plan_json,
                    plan.plan_hash,
                    approval_id,
                    utc_text(selected),
                    utc_text(selected),
                ),
            )
            elapsed = float(row["active_elapsed_seconds"])
            if row["active_started_at"] is not None:
                elapsed += max(0.0, (selected - parse_utc(row["active_started_at"])).total_seconds())
            changed = connection.execute(
                """UPDATE tasks SET state='AWAITING_APPROVAL',active_elapsed_seconds=?,
                   active_started_at=NULL,disconnect_expires_at=NULL,updated_at=?
                   WHERE task_id=? AND epoch=? AND state='EXECUTING'""",
                (elapsed, utc_text(selected), task_id, epoch),
            )
            if changed.rowcount != 1:
                raise StaleEpochError()
            self._append_task_event_tx(
                connection,
                task_id,
                epoch,
                "research.plan.awaiting_approval",
                {"approval_id": approval_id, "plan_sha256": plan.plan_hash},
                selected,
            )
            return {"run_id": run_id, "approval_id": approval_id, "plan_hash": plan.plan_hash}

    def research_run(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM research_runs WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            approval = self._connection.execute(
                "SELECT status FROM approvals WHERE approval_id=?", (row["approval_id"],)
            ).fetchone()
        return {
            "run_id": row["run_id"],
            "conversation_id": row["conversation_id"],
            "task_id": row["task_id"],
            "epoch": row["epoch"],
            "plan_json": row["plan_json"],
            "plan_hash": row["plan_sha256"],
            "approval_id": row["approval_id"],
            "approval_status": approval["status"] if approval is not None else "CANCELLED",
            "stage": row["stage"],
            "status": row["status"],
            "bundle_json": row["bundle_json"],
            "bundle_hash": row["bundle_sha256"],
        }

    def begin_research_run(
        self, task_id: str, epoch: int, *, request_key: str, now: datetime | None = None
    ) -> str:
        selected = now or utc_now()
        with self._transaction() as connection:
            self._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            run = connection.execute(
                """SELECT r.*,a.status AS approval_status FROM research_runs r
                   JOIN approvals a ON a.approval_id=r.approval_id WHERE r.task_id=?""",
                (task_id,),
            ).fetchone()
            if run is None or run["approval_status"] != "APPROVED":
                raise OrchestratorError("research_not_approved", status=409)
            existing = connection.execute(
                """SELECT * FROM capability_reservations
                   WHERE task_id=? AND epoch=? AND kind='research' AND request_key=?""",
                (task_id, epoch, request_key),
            ).fetchone()
            digest = canonical_sha256({"plan_hash": run["plan_sha256"]})
            if existing is not None:
                if existing["arguments_sha256"] != digest:
                    raise OrchestratorError("idempotency_conflict", status=409)
                return existing["reservation_id"]
            used = connection.execute(
                """SELECT COUNT(*) FROM capability_reservations
                   WHERE task_id=? AND epoch=? AND kind='research' AND status!='CANCELLED'""",
                (task_id, epoch),
            ).fetchone()[0]
            if used >= 1:
                raise BudgetExhaustedError()
            reservation_id = new_uuid4()
            connection.execute(
                """INSERT INTO capability_reservations(
                   reservation_id,task_id,epoch,kind,request_key,arguments_sha256,status,created_at
                   ) VALUES(?,?,?,'research',?,?,'RESERVED',?)""",
                (reservation_id, task_id, epoch, request_key, digest, utc_text(selected)),
            )
            connection.execute(
                "UPDATE research_runs SET stage='RUNNING',updated_at=? WHERE run_id=?",
                (utc_text(selected), run["run_id"]),
            )
            return reservation_id

    def commit_research_result(
        self,
        reservation_id: str,
        task_id: str,
        epoch: int,
        *,
        bundle: EvidenceBundleV1,
        cited_documents: Mapping[str, NormalizedDocumentV1],
        now: datetime | None = None,
    ) -> None:
        """Atomically persist cited evidence and reconcile the whole pipeline."""

        selected = now or utc_now()
        source_by_id = {source.source_id: source for source in bundle.sources}
        if set(cited_documents) != set(source_by_id):
            raise ValueError("only and all cited documents must be committed")
        if sum(document.byte_length for document in cited_documents.values()) > MAX_RUN_DOCUMENT_BYTES:
            raise ValueError("research document budget exceeded")
        for source_id, document in cited_documents.items():
            if source_by_id[source_id].normalized_document_hash != document.sha256:
                raise ValueError("source and document hash mismatch")
        for claim in bundle.claims:
            anchor_document = cited_documents[claim.anchor.source_id]
            if reconstruct(claim.anchor, anchor_document) != claim.statement:
                raise ValueError("claim is not the exact anchor span")
            if claim.corroborating is not None:
                corroborating_text = reconstruct(
                    claim.corroborating,
                    cited_documents[claim.corroborating.source_id],
                )
                if not 40 <= len(corroborating_text) <= 400:
                    raise ValueError("corroborating span is invalid")
        validate_syntactic_citation_coverage(bundle, cited_documents)
        with self._transaction() as connection:
            task = self._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            reservation = connection.execute(
                """SELECT * FROM capability_reservations
                   WHERE reservation_id=? AND task_id=? AND epoch=? AND kind='research'""",
                (reservation_id, task_id, epoch),
            ).fetchone()
            run = connection.execute(
                "SELECT * FROM research_runs WHERE task_id=? AND epoch=?",
                (task_id, epoch),
            ).fetchone()
            if reservation is None or reservation["status"] != "RESERVED" or run is None or run["stage"] != "RUNNING":
                raise StaleEpochError()
            if run["plan_sha256"] != bundle.plan_hash:
                raise OrchestratorError("research_plan_changed", status=409)
            for source_id, document in cited_documents.items():
                connection.execute(
                    """INSERT INTO research_documents(
                       run_id,conversation_id,task_id,source_id,source_json,
                       document_json,document_sha256,created_at
                       ) VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        run["run_id"],
                        task["conversation_id"],
                        task_id,
                        source_id,
                        canonical_json(asdict(source_by_id[source_id])),
                        canonical_json(asdict(document)),
                        document.sha256,
                        utc_text(selected),
                    ),
                )
            connection.execute(
                """UPDATE research_runs SET stage='COMPLETED',status=?,bundle_json=?,
                   bundle_sha256=?,updated_at=? WHERE run_id=? AND stage='RUNNING'""",
                (bundle.status.value, bundle.to_json(), bundle.sha256, utc_text(selected), run["run_id"]),
            )
            connection.execute(
                """UPDATE capability_reservations SET status='COMPLETED',completed_at=?
                   WHERE reservation_id=? AND status='RESERVED'""",
                (utc_text(selected), reservation_id),
            )
            turn = connection.execute(
                "SELECT effects_json FROM canonical_turns WHERE task_id=? AND status='PENDING'",
                (task_id,),
            ).fetchone()
            if turn is None:
                raise StaleEpochError()
            effects = json.loads(turn["effects_json"])
            effects.append(
                asdict(
                    CanonicalEffectV1(
                        kind="research",
                        name="web-research",
                        status=bundle.status.value,
                        arguments_sha256=run["plan_sha256"],
                        result_sha256=bundle.sha256,
                    )
                )
            )
            connection.execute(
                "UPDATE canonical_turns SET effects_json=? WHERE task_id=?",
                (canonical_json(effects), task_id),
            )
            self._append_task_event_tx(
                connection,
                task_id,
                epoch,
                "research.completed",
                {"status": bundle.status.value, "bundle_sha256": bundle.sha256},
                selected,
            )

    def research_documents(self, task_id: str) -> dict[str, NormalizedDocumentV1]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT source_id,document_json FROM research_documents WHERE task_id=? ORDER BY source_id",
                (task_id,),
            ).fetchall()
        return {row["source_id"]: NormalizedDocumentV1(**json.loads(row["document_json"])) for row in rows}

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

    def capability_reservation(
        self,
        task_id: str,
        epoch: int,
        *,
        kind: str,
        request_key: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM capability_reservations
                   WHERE task_id=? AND epoch=? AND kind=? AND request_key=?""",
                (task_id, epoch, kind, request_key),
            ).fetchone()
        return dict(row) if row is not None else None

    def recover_after_restart(self, *, now: datetime | None = None) -> int:
        selected = now or utc_now()
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM tasks WHERE tombstoned=0 AND state IN ('ROUTING','LOADING_CONTEXT','EXECUTING','RESPONDING','COMMITTING')"
            ).fetchall()
        count = 0
        for row in rows:
            try:
                with self._transaction() as connection:
                    resumable = connection.execute(
                        """SELECT 1 FROM tasks t JOIN conversations c USING(conversation_id)
                           WHERE t.task_id=? AND t.epoch=? AND t.state='EXECUTING'
                           AND t.tombstoned=0 AND c.current_epoch=t.epoch
                           AND (
                             EXISTS(
                               SELECT 1 FROM cloud_approval_bindings b
                               JOIN approvals a USING(approval_id)
                               WHERE b.task_id=t.task_id AND b.epoch=t.epoch
                                 AND a.status='APPROVED' AND t.cloud_dispatch_count=0
                             )
                             OR EXISTS(
                               SELECT 1 FROM scheduler_task_links s
                               WHERE s.task_id=t.task_id AND s.epoch=t.epoch
                                 AND s.state IN ('APPROVED','DENIED')
                             )
                             OR EXISTS(
                               SELECT 1 FROM tool_executions x
                               WHERE x.task_id=t.task_id AND x.epoch=t.epoch
                                 AND x.status IN ('APPROVED','DENIED','VERIFIED','ACCEPTED')
                             )
                           )""",
                        (row["task_id"], row["epoch"]),
                    ).fetchone()
                    if resumable is not None:
                        elapsed = float(row["active_elapsed_seconds"])
                        if row["active_started_at"] is not None:
                            elapsed += max(
                                0.0,
                                (selected - parse_utc(row["active_started_at"])).total_seconds(),
                            )
                        connection.execute(
                            """UPDATE tasks SET state='DECISION_WAITING_RECONNECT',
                               active_elapsed_seconds=?,active_started_at=NULL,
                               disconnect_expires_at=?,updated_at=?
                               WHERE task_id=? AND epoch=? AND state='EXECUTING'""",
                            (
                                elapsed,
                                utc_text(selected + self.disconnect_ttl),
                                utc_text(selected),
                                row["task_id"],
                                row["epoch"],
                            ),
                        )
                        self._append_task_event_tx(
                            connection,
                            row["task_id"],
                            row["epoch"],
                            "DECISION_WAITING_RECONNECT",
                            {"restart_recovery": True},
                            selected,
                        )
                        continue
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

    def persist_route_artifacts(
        self,
        task_id: str,
        epoch: int,
        profile: TaskProfileV1,
        facts: RoutingFactsV1,
        plan: RoutePlanV1,
        *,
        now: datetime | None = None,
    ) -> None:
        """Persist the complete deterministic routing evidence under epoch CAS."""

        selected = now or utc_now()
        if facts.profile_sha256 != profile.sha256 or plan.profile_sha256 != profile.sha256 or plan.routing_facts_sha256 != facts.sha256:
            raise OrchestratorError("route_artifact_mismatch", status=409)
        documents = (
            ("task_profiles", "profile_json", "profile_sha256", jcs_text(profile.canonical()), profile.sha256),
            ("routing_facts", "facts_json", "facts_sha256", jcs_text(facts.canonical()), facts.sha256),
            ("route_plans", "plan_json", "plan_sha256", plan.canonical_json(), plan.sha256),
        )
        with self._transaction() as connection:
            self._guarded_task_tx(connection, task_id, epoch, (TaskState.ROUTING,))
            existing = connection.execute(
                "SELECT plan_sha256 FROM route_plans WHERE task_id=?", (task_id,)
            ).fetchone()
            if existing is not None:
                if existing["plan_sha256"] != plan.sha256:
                    raise OrchestratorError("route_plan_changed", status=409)
                return
            for table, json_field, digest_field, document, digest in documents[:2]:
                connection.execute(
                    f"INSERT INTO {table}(task_id,epoch,{json_field},{digest_field},created_at) VALUES(?,?,?,?,?)",
                    (task_id, epoch, document, digest, utc_text(selected)),
                )
            connection.execute(
                """INSERT INTO route_plans(
                   task_id,epoch,plan_json,plan_sha256,target,reason_code,created_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (task_id, epoch, plan.canonical_json(), plan.sha256, plan.target.value, plan.reason.value, utc_text(selected)),
            )
            self._append_task_event_tx(
                connection,
                task_id,
                epoch,
                "ROUTE_PLANNED",
                {
                    "profile_sha256": profile.sha256,
                    "routing_facts_sha256": facts.sha256,
                    "route_plan_sha256": plan.sha256,
                    "target": plan.target.value,
                    "reason_code": plan.reason.value,
                },
                selected,
            )

    def route_artifacts(self, task_id: str, epoch: int) -> dict[str, Any]:
        """Load the immutable routing snapshot; never recompute denial policy."""

        with self._lock:
            row = self._connection.execute(
                """SELECT p.profile_json,p.profile_sha256,
                          f.facts_json,f.facts_sha256,
                          r.plan_json,r.plan_sha256
                   FROM task_profiles p
                   JOIN routing_facts f USING(task_id,epoch)
                   JOIN route_plans r USING(task_id,epoch)
                   WHERE p.task_id=? AND p.epoch=?""",
                (task_id, epoch),
            ).fetchone()
        if row is None:
            raise OrchestratorError("route_artifacts_missing", status=409)
        profile = json.loads(row["profile_json"])
        facts = json.loads(row["facts_json"])
        plan = json.loads(row["plan_json"])
        if (
            domain_sha256("task-profile/1", profile) != row["profile_sha256"]
            or domain_sha256("routing-facts/1", facts) != row["facts_sha256"]
            or domain_sha256("route-plan/1", plan) != row["plan_sha256"]
        ):
            raise OrchestratorError("route_artifact_mismatch", status=409)
        return {"profile": profile, "facts": facts, "plan": plan}

    def register_endpoint_qualification(
        self,
        endpoint: QualifiedEndpointV1,
        *,
        enabled: bool,
        now: datetime | None = None,
    ) -> None:
        selected = now or utc_now()
        endpoint.validate()
        artifact = {
            "completion_rate_per_m": endpoint.completion_rate_per_m,
            "context_limit": endpoint.context_limit,
            "model_canonical_id": endpoint.model_canonical_id,
            "output_limit": endpoint.output_limit,
            "prompt_rate_per_m": endpoint.prompt_rate_per_m,
            "provider_endpoint_tag": endpoint.provider_endpoint_tag,
            "qualification_sha256": endpoint.qualification_sha256,
            "qualification_version": endpoint.qualification_version,
            "quantization": endpoint.quantization,
            "request_rate": endpoint.request_rate,
            "runtime_provider_name": endpoint.runtime_provider_name,
            "supported_parameters": list(endpoint.supported_parameters),
        }
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO endpoint_qualifications(
                   qualification_sha256,model_canonical_id,provider_endpoint_tag,quantization,
                   parameter_profile_sha256,qualification_version,stable_catalog_sha256,
                   artifact_json,enabled,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(qualification_sha256) DO UPDATE SET enabled=excluded.enabled""",
                (
                    endpoint.qualification_sha256,
                    endpoint.model_canonical_id,
                    endpoint.provider_endpoint_tag,
                    endpoint.quantization,
                    endpoint.parameter_profile_sha256,
                    endpoint.qualification_version,
                    endpoint.stable_catalog_sha256,
                    jcs_text(artifact),
                    int(enabled),
                    utc_text(selected),
                ),
            )

    def store_endpoint_health(
        self,
        qualification_sha256: str,
        health_sha256: str,
        snapshot: Mapping[str, Any],
        *,
        observed_at: str,
    ) -> None:
        with self._transaction() as connection:
            qualification = connection.execute(
                "SELECT 1 FROM endpoint_qualifications WHERE qualification_sha256=? AND enabled=1",
                (qualification_sha256,),
            ).fetchone()
            if qualification is None:
                raise OrchestratorError("cloud_unqualified", status=409)
            connection.execute(
                """INSERT OR IGNORE INTO endpoint_health_snapshots(
                   health_sha256,qualification_sha256,snapshot_json,observed_at
                   ) VALUES(?,?,?,?)""",
                (health_sha256, qualification_sha256, jcs_text(snapshot), observed_at),
            )

    def quarantine_endpoint(self, qualification_sha256: str) -> None:
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE endpoint_qualifications SET enabled=0 WHERE qualification_sha256=?",
                (qualification_sha256,),
            )
            if changed.rowcount != 1:
                raise OrchestratorError("cloud_unqualified", status=409)

    def disable_all_endpoint_qualifications(self) -> int:
        """Fail-closed startup fence before release qualification is rechecked."""

        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE endpoint_qualifications SET enabled=0 WHERE enabled=1"
            )
            return changed.rowcount

    def store_cloud_provider_receipt(
        self,
        task_id: str,
        epoch: int,
        *,
        generation_id: str | None,
        receipt: Mapping[str, Any] | None,
        verified: bool,
        now: datetime | None = None,
    ) -> None:
        selected = now or utc_now()
        receipt_json = jcs_text(receipt) if receipt is not None else None
        receipt_sha256 = domain_sha256("cloud-provider-receipt/1", receipt) if receipt is not None else None
        with self._transaction() as connection:
            task = connection.execute(
                """SELECT t.epoch,c.current_epoch FROM tasks t JOIN conversations c USING(conversation_id)
                   WHERE t.task_id=?""",
                (task_id,),
            ).fetchone()
            if task is None or task["epoch"] != epoch or task["current_epoch"] != epoch:
                raise StaleEpochError()
            connection.execute(
                """INSERT INTO cloud_provider_receipts(
                   task_id,epoch,generation_id,receipt_json,receipt_sha256,
                   provider_identity_verified,created_at
                   ) VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(task_id) DO UPDATE SET
                   generation_id=excluded.generation_id,receipt_json=excluded.receipt_json,
                   receipt_sha256=excluded.receipt_sha256,
                   provider_identity_verified=excluded.provider_identity_verified""",
                (
                    task_id,
                    epoch,
                    generation_id,
                    receipt_json,
                    receipt_sha256,
                    int(verified),
                    utc_text(selected),
                ),
            )

    def create_cloud_approval(
        self,
        task_id: str,
        epoch: int,
        binding: CloudApprovalBindingV1,
        dispatch_body: Mapping[str, Any],
        *,
        owner_session_id: str,
        now: datetime | None = None,
    ) -> dict[str, str]:
        selected = now or utc_now()
        if binding.task_id != task_id or binding.epoch != epoch:
            raise OrchestratorError("binding_mismatch", status=409)
        dispatch_digest = domain_sha256("cloud-dispatch-body/1", dispatch_body)
        if dispatch_digest != binding.dispatch_body_sha256:
            raise OrchestratorError("binding_mismatch", status=409)
        expires = parse_utc(binding.expires_at)
        if expires <= selected:
            raise OrchestratorError("approval_expired", status=410)
        with self._transaction() as connection:
            task = self._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            if task["conversation_id"] != binding.conversation_id:
                raise OrchestratorError("binding_mismatch", status=409)
            qualified = connection.execute(
                """SELECT artifact_json,stable_catalog_sha256,enabled
                   FROM endpoint_qualifications WHERE qualification_sha256=?""",
                (binding.qualification_sha256,),
            ).fetchone()
            if (
                qualified is None
                or not bool(qualified["enabled"])
                or qualified["stable_catalog_sha256"] != binding.stable_catalog_sha256
            ):
                raise OrchestratorError("cloud_unqualified", status=409)
            artifact = json.loads(qualified["artifact_json"])
            endpoint = QualifiedEndpointV1(
                model_canonical_id=artifact["model_canonical_id"],
                provider_endpoint_tag=artifact["provider_endpoint_tag"],
                quantization=artifact["quantization"],
                supported_parameters=tuple(artifact["supported_parameters"]),
                qualification_version=artifact["qualification_version"],
                qualification_sha256=artifact["qualification_sha256"],
                context_limit=artifact["context_limit"],
                output_limit=artifact["output_limit"],
                prompt_rate_per_m=artifact["prompt_rate_per_m"],
                completion_rate_per_m=artifact["completion_rate_per_m"],
                request_rate=artifact["request_rate"],
                runtime_provider_name=artifact["runtime_provider_name"],
            ).validate()
            if endpoint.stable_catalog_sha256 != qualified["stable_catalog_sha256"]:
                raise OrchestratorError("route_artifact_mismatch", status=409)
            raw_messages = dispatch_body.get("messages")
            if not isinstance(raw_messages, list):
                raise OrchestratorError("binding_mismatch", status=409)
            try:
                expected_body = build_dispatch_body(
                    endpoint,
                    messages=tuple(raw_messages),
                    output_token_limit=binding.output_token_limit,
                )
            except (TypeError, ValueError) as exc:
                raise OrchestratorError("binding_mismatch", status=409) from exc
            if jcs_text(expected_body) != jcs_text(dispatch_body):
                raise OrchestratorError("binding_mismatch", status=409)
            plan = connection.execute(
                "SELECT * FROM route_plans WHERE task_id=? AND epoch=?",
                (task_id, epoch),
            ).fetchone()
            if plan is None or plan["plan_sha256"] != binding.route_plan_sha256:
                raise OrchestratorError("binding_mismatch", status=409)
            existing = connection.execute(
                "SELECT approval_id,binding_sha256 FROM cloud_approval_bindings WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if existing is not None:
                if existing["binding_sha256"] != binding.sha256:
                    raise OrchestratorError("binding_changed", status=409)
                presentation = connection.execute(
                    "SELECT presentation_id,requested_event_id FROM cloud_presentations WHERE approval_id=? AND state='PENDING'",
                    (existing["approval_id"],),
                ).fetchone()
                return {
                    "approval_id": existing["approval_id"],
                    "presentation_id": presentation["presentation_id"],
                    "requested_event_id": presentation["requested_event_id"],
                }
            approval_id = new_uuid4()
            presentation_id = new_uuid4()
            requested_event_id = new_uuid4()
            decision_nonce = new_uuid4()
            arguments = {
                "approval_binding_sha256": binding.sha256,
                "egress_sha256": binding.egress_sha256,
                "max_reserved_microusd": binding.max_reserved_microusd,
                "model_canonical_id": binding.model_canonical_id,
                "provider_endpoint_tag": binding.provider_endpoint_tag,
                "route_plan_sha256": binding.route_plan_sha256,
            }
            connection.execute(
                """INSERT INTO approvals(
                   approval_id,conversation_id,task_id,epoch,kind,name,arguments_json,
                   arguments_sha256,status,created_at,expires_at
                   ) VALUES(?,?,?,?, 'cloud_model',?,?,?, 'PENDING',?,?)""",
                (
                    approval_id,
                    task["conversation_id"],
                    task_id,
                    epoch,
                    binding.model_canonical_id,
                    jcs_text(arguments),
                    domain_sha256("cloud-approval-arguments/1", arguments),
                    utc_text(selected),
                    binding.expires_at,
                ),
            )
            connection.execute(
                """INSERT INTO cloud_approval_bindings(
                   approval_id,task_id,conversation_id,epoch,binding_json,binding_sha256,
                   dispatch_body_json,dispatch_body_sha256,egress_sha256,route_plan_sha256,
                   stable_catalog_sha256,qualification_sha256,created_at,expires_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    approval_id,
                    task_id,
                    task["conversation_id"],
                    epoch,
                    binding.canonical_json(),
                    binding.sha256,
                    jcs_text(dispatch_body),
                    binding.dispatch_body_sha256,
                    binding.egress_sha256,
                    binding.route_plan_sha256,
                    binding.stable_catalog_sha256,
                    binding.qualification_sha256,
                    utc_text(selected),
                    binding.expires_at,
                ),
            )
            queue_count = connection.execute(
                """SELECT COUNT(*) FROM approval_arbiter_queue WHERE state IN (
                   'WAITING_FOR_CHANNEL','PRESENTING','PRESENTED','DECIDING','WAITING_FOR_OWNER')"""
            ).fetchone()[0]
            if queue_count >= 64:
                raise OrchestratorError("APPROVAL_QUEUE_FULL", status=409)
            connection.execute(
                """INSERT INTO approval_arbiter_queue(
                   approval_id,producer_protocol,binding_sha256,state,created_at,expires_at,updated_at
                   ) VALUES(?, 'jarvis.cloud-approval/1',?,'WAITING_FOR_CHANNEL',?,?,?)""",
                (approval_id, binding.sha256, utc_text(selected), binding.expires_at, utc_text(selected)),
            )
            active_channel = connection.execute(
                """SELECT approval_id FROM approval_arbiter_queue WHERE state IN (
                   'PRESENTING','PRESENTED','DECIDING','WAITING_FOR_OWNER') LIMIT 1"""
            ).fetchone()
            queue_head = connection.execute(
                """SELECT approval_id FROM approval_arbiter_queue WHERE state='WAITING_FOR_CHANNEL'
                   ORDER BY arbiter_seq,approval_id LIMIT 1"""
            ).fetchone()
            present_now = active_channel is None and queue_head is not None and queue_head["approval_id"] == approval_id
            connection.execute(
                """INSERT INTO cloud_presentations(
                   presentation_id,approval_id,task_id,epoch,owner_session_id,decision_nonce,
                   requested_event_id,state,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,'PENDING',?,?)""",
                (
                    presentation_id,
                    approval_id,
                    task_id,
                    epoch,
                    owner_session_id,
                    decision_nonce,
                    requested_event_id,
                    utc_text(selected),
                    utc_text(selected),
                ),
            )
            requested_payload = {
                "approval_binding_sha256": binding.sha256,
                "approval_id": approval_id,
                "approve_key": "F10",
                "billable_prompt_tokens": binding.billable_prompt_tokens,
                "decision_nonce": decision_nonce,
                "deny_key": "F11",
                "egress_sha256": binding.egress_sha256,
                "epoch": epoch,
                "expires_at": binding.expires_at,
                "included_turn_ids": list(binding.included_turn_ids),
                "input_token_limit": binding.input_token_limit,
                "max_reserved_microusd": binding.max_reserved_microusd,
                "model_canonical_id": binding.model_canonical_id,
                "output_token_limit": binding.output_token_limit,
                "presentation_id": presentation_id,
                "provider_endpoint_tag": binding.provider_endpoint_tag,
                "qualification_sha256": binding.qualification_sha256,
                "quantization": binding.quantization,
                "route_plan_sha256": binding.route_plan_sha256,
                "stable_catalog_sha256": binding.stable_catalog_sha256,
                "task_id": task_id,
            }
            self._append_cloud_outbox_tx(
                connection,
                conversation_id=task["conversation_id"],
                task_id=task_id,
                task_epoch=epoch,
                guard_epoch=epoch,
                approval_id=approval_id,
                presentation_id=presentation_id,
                event_type="cloud.approval.requested",
                payload=requested_payload,
                expires_at=binding.expires_at,
                event_id=requested_event_id,
                now=selected,
            )
            if present_now:
                connection.execute(
                    """UPDATE approval_arbiter_queue SET state='PRESENTING',owner_session_id=?,
                       presentation_id=?,requested_event_id=?,decision_nonce=?,updated_at=?
                       WHERE approval_id=? AND state='WAITING_FOR_CHANNEL'""",
                    (owner_session_id, presentation_id, requested_event_id, decision_nonce,
                     utc_text(selected), approval_id),
                )
            else:
                connection.execute(
                    "UPDATE cloud_extension_outbox SET state='NON_REPLAYABLE' WHERE event_id=?",
                    (requested_event_id,),
                )
            elapsed = float(task["active_elapsed_seconds"])
            if task["active_started_at"] is not None:
                elapsed += max(0.0, (selected - parse_utc(task["active_started_at"])).total_seconds())
            changed = connection.execute(
                """UPDATE tasks SET state='AWAITING_APPROVAL',active_elapsed_seconds=?,
                   active_started_at=NULL,disconnect_expires_at=NULL,updated_at=?
                   WHERE task_id=? AND epoch=? AND state='EXECUTING' AND tombstoned=0""",
                (elapsed, utc_text(selected), task_id, epoch),
            )
            if changed.rowcount != 1:
                raise StaleEpochError()
            self._append_task_event_tx(
                connection,
                task_id,
                epoch,
                "AWAITING_APPROVAL",
                {"approval_id": approval_id, "approval_binding_sha256": binding.sha256},
                selected,
            )
            return {
                "approval_id": approval_id,
                "presentation_id": presentation_id,
                "requested_event_id": requested_event_id,
            }

    def claim_cloud_presentation_owner(
        self,
        session_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, str] | None:
        """CAS ownership to the newest negotiated connection."""

        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT p.*,b.binding_json,b.binding_sha256,b.conversation_id,
                          b.expires_at,c.current_epoch,q.state AS arbiter_state
                   FROM cloud_presentations p
                   JOIN cloud_approval_bindings b USING(approval_id)
                   JOIN approval_arbiter_queue q USING(approval_id)
                   JOIN approvals a USING(approval_id)
                   JOIN conversations c USING(conversation_id)
                   JOIN tasks t USING(task_id)
                   WHERE p.state='PENDING' AND a.status='PENDING'
                   AND t.state='AWAITING_APPROVAL' AND t.tombstoned=0
                   AND t.epoch=c.current_epoch AND b.expires_at>?
                   AND q.producer_protocol='jarvis.cloud-approval/1'
                   AND (
                     q.state IN ('PRESENTING','PRESENTED','WAITING_FOR_OWNER')
                     OR (
                       q.state='WAITING_FOR_CHANNEL'
                       AND NOT EXISTS(SELECT 1 FROM approval_arbiter_queue active WHERE active.state IN (
                         'PRESENTING','PRESENTED','DECIDING','WAITING_FOR_OWNER'))
                       AND q.arbiter_seq=(SELECT MIN(head.arbiter_seq) FROM approval_arbiter_queue head
                                         WHERE head.state='WAITING_FOR_CHANNEL')
                     )
                   )
                   ORDER BY q.arbiter_seq,q.approval_id LIMIT 1""",
                (utc_text(selected),),
            ).fetchone()
            if row is None:
                return None
            if row["owner_session_id"] == session_id and row["arbiter_state"] != "WAITING_FOR_OWNER":
                connection.execute(
                    """UPDATE approval_arbiter_queue SET state='PRESENTING',owner_session_id=?,
                       presentation_id=?,requested_event_id=?,decision_nonce=?,updated_at=?
                       WHERE approval_id=?""",
                    (session_id, row["presentation_id"], row["requested_event_id"],
                     row["decision_nonce"], utc_text(selected), row["approval_id"]),
                )
                connection.execute(
                    """UPDATE cloud_extension_outbox SET state='REPLAYABLE',next_attempt_at=?,attempt_count=0
                       WHERE event_id=? AND state='NON_REPLAYABLE'""",
                    (utc_text(selected), row["requested_event_id"]),
                )
                return {
                    "approval_id": row["approval_id"],
                    "presentation_id": row["presentation_id"],
                    "requested_event_id": row["requested_event_id"],
                    "old_session_id": "",
                }
            old_session_id = row["owner_session_id"]
            connection.execute(
                "UPDATE cloud_presentations SET state='SUPERSEDED',updated_at=? WHERE presentation_id=? AND state='PENDING'",
                (utc_text(selected), row["presentation_id"]),
            )
            connection.execute(
                "UPDATE cloud_extension_outbox SET state='STALE' WHERE event_id=? AND state='REPLAYABLE'",
                (row["requested_event_id"],),
            )
            self._append_cloud_outbox_tx(
                connection,
                conversation_id=row["conversation_id"],
                task_id=row["task_id"],
                task_epoch=row["epoch"],
                guard_epoch=row["current_epoch"],
                approval_id=row["approval_id"],
                presentation_id=row["presentation_id"],
                event_type="cloud.approval.invalidated",
                payload={
                    "approval_binding_sha256": row["binding_sha256"],
                    "approval_id": row["approval_id"],
                    "presentation_id": row["presentation_id"],
                    "reason_code": "session_superseded",
                },
                expires_at=row["expires_at"],
                now=selected,
            )
            binding = json.loads(row["binding_json"])
            presentation_id = new_uuid4()
            requested_event_id = new_uuid4()
            decision_nonce = new_uuid4()
            connection.execute(
                """INSERT INTO cloud_presentations(
                   presentation_id,approval_id,task_id,epoch,owner_session_id,
                   decision_nonce,requested_event_id,state,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,'PENDING',?,?)""",
                (
                    presentation_id,
                    row["approval_id"],
                    row["task_id"],
                    row["epoch"],
                    session_id,
                    decision_nonce,
                    requested_event_id,
                    utc_text(selected),
                    utc_text(selected),
                ),
            )
            self._append_cloud_outbox_tx(
                connection,
                conversation_id=row["conversation_id"],
                task_id=row["task_id"],
                task_epoch=row["epoch"],
                guard_epoch=row["current_epoch"],
                approval_id=row["approval_id"],
                presentation_id=presentation_id,
                event_type="cloud.approval.requested",
                payload={
                    "approval_binding_sha256": row["binding_sha256"],
                    "approval_id": row["approval_id"],
                    "approve_key": "F10",
                    "billable_prompt_tokens": binding["billable_prompt_tokens"],
                    "decision_nonce": decision_nonce,
                    "deny_key": "F11",
                    "egress_sha256": binding["egress_sha256"],
                    "epoch": row["epoch"],
                    "expires_at": row["expires_at"],
                    "included_turn_ids": binding["included_turn_ids"],
                    "input_token_limit": binding["input_token_limit"],
                    "max_reserved_microusd": binding["max_reserved_microusd"],
                    "model_canonical_id": binding["model_canonical_id"],
                    "output_token_limit": binding["output_token_limit"],
                    "presentation_id": presentation_id,
                    "provider_endpoint_tag": binding["provider_endpoint_tag"],
                    "qualification_sha256": binding["qualification_sha256"],
                    "quantization": binding["quantization"],
                    "route_plan_sha256": binding["route_plan_sha256"],
                    "stable_catalog_sha256": binding["stable_catalog_sha256"],
                    "task_id": row["task_id"],
                },
                expires_at=row["expires_at"],
                event_id=requested_event_id,
                now=selected,
            )
            connection.execute(
                """UPDATE approval_arbiter_queue SET state='PRESENTING',owner_session_id=?,
                   presentation_id=?,requested_event_id=?,decision_nonce=?,updated_at=?
                   WHERE approval_id=?""",
                (session_id, presentation_id, requested_event_id, decision_nonce,
                 utc_text(selected), row["approval_id"]),
            )
            return {
                "approval_id": row["approval_id"],
                "presentation_id": presentation_id,
                "requested_event_id": requested_event_id,
                "old_session_id": old_session_id,
            }

    def claim_cloud_extension_events(
        self,
        conversation_id: str,
        *,
        now: datetime | None = None,
        limit: int = 64,
    ) -> list[dict[str, Any]]:
        selected = now or utc_now()
        stamp = utc_text(selected)
        with self._transaction() as connection:
            connection.execute(
                """UPDATE cloud_extension_outbox SET state='NON_REPLAYABLE'
                   WHERE state='REPLAYABLE' AND expires_at<=?""",
                (stamp,),
            )
            rows = connection.execute(
                """SELECT e.* FROM cloud_extension_outbox e
                   LEFT JOIN conversations c USING(conversation_id)
                   WHERE e.conversation_id=? AND e.state='REPLAYABLE'
                   AND e.next_attempt_at<=? AND e.expires_at>?
                   AND c.state='ACTIVE' AND c.current_epoch=e.guard_epoch
                   ORDER BY e.ordinal LIMIT ?""",
                (conversation_id, stamp, stamp, limit),
            ).fetchall()
            result: list[dict[str, Any]] = []
            schedule_ms = (250, 500, 1000, 2000)
            for row in rows:
                attempts = int(row["attempt_count"])
                delay_ms = schedule_ms[attempts] if attempts < len(schedule_ms) else 5000
                next_attempt = selected + timedelta(milliseconds=delay_ms)
                connection.execute(
                    """UPDATE cloud_extension_outbox
                       SET attempt_count=attempt_count+1,next_attempt_at=?
                       WHERE event_id=? AND state='REPLAYABLE'""",
                    (utc_text(next_attempt), row["event_id"]),
                )
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json"))
                result.append(item)
            return result

    def reset_cloud_extension_replay(self, conversation_id: str, *, now: datetime | None = None) -> None:
        with self._transaction() as connection:
            connection.execute(
                """UPDATE cloud_extension_outbox SET next_attempt_at=?,attempt_count=0
                   WHERE conversation_id=? AND state='REPLAYABLE' AND expires_at>?""",
                (utc_text(now or utc_now()), conversation_id, utc_text(now or utc_now())),
            )

    def acknowledge_cloud_extension(self, event_id: str, *, now: datetime | None = None) -> bool:
        selected = now or utc_now()
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE cloud_extension_outbox SET state='ACKED' WHERE event_id=? AND state='REPLAYABLE'",
                (event_id,),
            ).rowcount == 1
            if changed:
                connection.execute(
                    """UPDATE approval_arbiter_queue SET state='PRESENTED',updated_at=?
                       WHERE requested_event_id=? AND state='PRESENTING'""",
                    (utc_text(selected), event_id),
                )
            return changed

    def cloud_extension_is_current(self, event_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                """SELECT 1 FROM cloud_extension_outbox e
                   JOIN conversations c USING(conversation_id)
                   WHERE e.event_id=? AND e.state='REPLAYABLE' AND e.expires_at>?
                   AND c.state='ACTIVE' AND c.current_epoch=e.guard_epoch""",
                (event_id, utc_text()),
            ).fetchone()
        return row is not None

    def finish_cloud_extension_write(self, event_id: str, *, complete: bool) -> None:
        if not complete:
            return
        with self._transaction() as connection:
            connection.execute(
                """UPDATE cloud_extension_outbox SET state='ACKED'
                   WHERE event_id=? AND event_type='cloud.approval.decision_ack'
                   AND state='REPLAYABLE'""",
                (event_id,),
            )

    def decide_cloud_approval(
        self,
        *,
        decision_event_id: str,
        semantic_sha256: str,
        approval_id: str,
        presentation_id: str,
        requested_event_id: str,
        approval_binding_sha256: str,
        decision_nonce: str,
        decision: str,
        session_id: str,
        capture_idle: bool,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Transport-ACK every valid decision; report business rejection separately."""

        if decision not in {"approve", "deny"}:
            raise OrchestratorError("invalid_request", status=400)
        selected = now or utc_now()
        with self._transaction() as connection:
            duplicate = connection.execute(
                "SELECT semantic_sha256 FROM cloud_extension_inbox WHERE event_id=?",
                (decision_event_id,),
            ).fetchone()
            if duplicate is not None:
                if duplicate["semantic_sha256"] != semantic_sha256:
                    raise OrchestratorError("protocol.conflicting_duplicate", status=409)
                ack = connection.execute(
                    "SELECT event_id,payload_json FROM cloud_extension_outbox WHERE acked_event_id=? AND event_type='cloud.approval.decision_ack'",
                    (decision_event_id,),
                ).fetchone()
                result = connection.execute(
                    "SELECT event_id,payload_json FROM cloud_extension_outbox WHERE acked_event_id=? AND event_type='cloud.approval.decision_result'",
                    (decision_event_id,),
                ).fetchone()
                connection.execute(
                    """UPDATE cloud_extension_outbox
                       SET state='REPLAYABLE',next_attempt_at=?,attempt_count=0
                       WHERE event_id IN (?,?) AND state IN ('ACKED','NON_REPLAYABLE')
                       AND expires_at>?""",
                    (utc_text(selected), ack["event_id"], result["event_id"], utc_text(selected)),
                )
                return {
                    "duplicate": True,
                    "ack_event_id": ack["event_id"],
                    "result_event_id": result["event_id"],
                    "result": json.loads(result["payload_json"]),
                }

            binding = connection.execute(
                """SELECT b.*,a.status AS approval_status,t.state AS task_state,
                          t.tombstoned,c.current_epoch,p.owner_session_id,p.decision_nonce,
                          p.requested_event_id,p.state AS presentation_state,
                          q.state AS arbiter_state,q.owner_session_id AS arbiter_owner_session_id
                   FROM cloud_approval_bindings b
                   JOIN approvals a USING(approval_id)
                   JOIN approval_arbiter_queue q USING(approval_id)
                   JOIN tasks t USING(task_id)
                   JOIN conversations c USING(conversation_id)
                   LEFT JOIN cloud_presentations p
                     ON p.approval_id=b.approval_id AND p.presentation_id=?
                   WHERE b.approval_id=?""",
                (presentation_id, approval_id),
            ).fetchone()
            reason = "invalid_state"
            applied = False
            if binding is not None:
                if parse_utc(binding["expires_at"]) <= selected:
                    reason = "approval_expired"
                elif bool(binding["tombstoned"]):
                    reason = "task_cancelled"
                elif binding["epoch"] != binding["current_epoch"]:
                    reason = "stale_epoch"
                elif binding["binding_sha256"] != approval_binding_sha256:
                    reason = "binding_mismatch"
                elif binding["presentation_state"] == "SUPERSEDED":
                    reason = "presentation_superseded"
                elif binding["owner_session_id"] != session_id:
                    reason = "not_owner"
                elif binding["arbiter_owner_session_id"] != session_id or binding["arbiter_state"] not in {"PRESENTING", "PRESENTED"}:
                    reason = "not_owner"
                elif not capture_idle:
                    reason = "capture_active"
                elif binding["decision_nonce"] != decision_nonce or binding["requested_event_id"] != requested_event_id:
                    reason = "binding_mismatch"
                elif binding["approval_status"] != "PENDING" or binding["presentation_state"] != "PENDING" or binding["task_state"] != TaskState.AWAITING_APPROVAL.value:
                    reason = "invalid_state"
                else:
                    reason = "approved" if decision == "approve" else "denied"
                    applied = True

            if applied:
                connection.execute(
                    "UPDATE approval_arbiter_queue SET state='DECIDING',updated_at=? WHERE approval_id=?",
                    (utc_text(selected), approval_id),
                )

            connection.execute(
                """INSERT INTO cloud_extension_inbox(
                   event_id,semantic_sha256,approval_id,presentation_id,created_at
                   ) VALUES(?,?,?,?,?)""",
                (decision_event_id, semantic_sha256, approval_id, presentation_id, utc_text(selected)),
            )
            conversation_id = binding["conversation_id"] if binding is not None else connection.execute(
                "SELECT conversation_id FROM conversations WHERE state='ACTIVE'"
            ).fetchone()["conversation_id"]
            guard_epoch = binding["current_epoch"] if binding is not None else connection.execute(
                "SELECT current_epoch FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone()["current_epoch"]
            expires_at = binding["expires_at"] if binding is not None else utc_text(selected + self.delivery_ttl)
            ack_event_id = self._append_cloud_outbox_tx(
                connection,
                conversation_id=conversation_id,
                task_id=binding["task_id"] if binding is not None else None,
                task_epoch=binding["epoch"] if binding is not None else None,
                guard_epoch=guard_epoch,
                approval_id=approval_id,
                presentation_id=presentation_id,
                event_type="cloud.approval.decision_ack",
                payload={"decision_event_id": decision_event_id, "status": "accepted"},
                expires_at=expires_at,
                acked_event_id=decision_event_id,
                ack_status="accepted",
                now=selected,
            )
            result_payload = {
                "approval_binding_sha256": approval_binding_sha256,
                "approval_id": approval_id,
                "decision_event_id": decision_event_id,
                "presentation_id": presentation_id,
                "reason_code": reason,
                "result": "applied" if applied else "rejected",
            }
            result_event_id = self._append_cloud_outbox_tx(
                connection,
                conversation_id=conversation_id,
                task_id=binding["task_id"] if binding is not None else None,
                task_epoch=binding["epoch"] if binding is not None else None,
                guard_epoch=guard_epoch,
                approval_id=approval_id,
                presentation_id=presentation_id,
                event_type="cloud.approval.decision_result",
                payload=result_payload,
                expires_at=expires_at,
                acked_event_id=decision_event_id,
                ack_status="result",
                now=selected,
            )
            if applied and binding is not None:
                status = "APPROVED" if decision == "approve" else "DENIED"
                connection.execute(
                    "UPDATE approvals SET status=?,decided_at=? WHERE approval_id=? AND status='PENDING'",
                    (status, utc_text(selected), approval_id),
                )
                connection.execute(
                    "UPDATE cloud_presentations SET state='DECIDED',updated_at=? WHERE presentation_id=? AND state='PENDING'",
                    (utc_text(selected), presentation_id),
                )
                turn = connection.execute(
                    "SELECT effects_json FROM canonical_turns WHERE task_id=? AND status='PENDING'",
                    (binding["task_id"],),
                ).fetchone()
                if turn is not None:
                    effects = json.loads(turn["effects_json"])
                    effects.append(
                        {
                            "kind": "ToolDenied" if decision == "deny" else "cloud_model",
                            "name": binding["binding_json"] and json.loads(binding["binding_json"])["model_canonical_id"],
                            "status": status,
                            "arguments_sha256": binding["binding_sha256"],
                            "result_sha256": "",
                        }
                    )
                    connection.execute(
                        "UPDATE canonical_turns SET effects_json=? WHERE task_id=?",
                        (canonical_json(effects), binding["task_id"]),
                    )
                connection.execute(
                    """UPDATE tasks SET state='DECISION_WAITING_RECONNECT',active_started_at=NULL,
                       disconnect_expires_at=?,updated_at=?
                       WHERE task_id=? AND epoch=? AND state='AWAITING_APPROVAL'""",
                    (utc_text(selected + self.disconnect_ttl), utc_text(selected), binding["task_id"], binding["epoch"]),
                )
                self._append_task_event_tx(
                    connection,
                    binding["task_id"],
                    binding["epoch"],
                    "DECISION_WAITING_RECONNECT",
                    {"approval_id": approval_id, "decision": decision},
                    selected,
                )
                connection.execute(
                    "UPDATE approval_arbiter_queue SET state='RESOLVED',updated_at=? WHERE approval_id=?",
                    (utc_text(selected), approval_id),
                )
            return {
                "duplicate": False,
                "ack_event_id": ack_event_id,
                "result_event_id": result_event_id,
                "result": result_payload,
            }

    def resume_cloud_decision(self, approval_id: str, *, now: datetime | None = None) -> TaskSnapshot:
        """Resume only after the decision result has entered the publication stream."""

        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT a.status,t.* FROM approvals a JOIN tasks t USING(task_id)
                   JOIN conversations c USING(conversation_id)
                   WHERE a.approval_id=? AND a.epoch=t.epoch AND c.current_epoch=t.epoch
                   AND t.state='DECISION_WAITING_RECONNECT' AND t.tombstoned=0""",
                (approval_id,),
            ).fetchone()
            if row is None:
                raise StaleEpochError()
            connection.execute(
                """UPDATE tasks SET state='EXECUTING',active_started_at=?,disconnect_expires_at=NULL,updated_at=?
                   WHERE task_id=? AND epoch=? AND state='DECISION_WAITING_RECONNECT'""",
                (utc_text(selected), utc_text(selected), row["task_id"], row["epoch"]),
            )
            self._append_task_event_tx(connection, row["task_id"], row["epoch"], "EXECUTING", {"approval_id": approval_id}, selected)
            return self._task_from_row(connection.execute("SELECT * FROM tasks WHERE task_id=?", (row["task_id"],)).fetchone())

    def issue_cloud_dispatch_permit(
        self,
        task_id: str,
        epoch: int,
        *,
        approval_id: str,
        approval_binding_sha256: str,
        stable_catalog_sha256: str,
        qualification_sha256: str,
        reserved_microusd: int,
        key_remaining_microusd: int,
        utc_month: str,
        now: datetime | None = None,
    ) -> dict[str, str]:
        selected = now or utc_now()
        if (
            utc_month != selected.strftime("%Y-%m")
            or reserved_microusd < 0
            or key_remaining_microusd < reserved_microusd
        ):
            raise BudgetExhaustedError("cloud_budget_exhausted")
        with self._transaction() as connection:
            task = self._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            binding = connection.execute(
                """SELECT b.*,a.status AS approval_status,q.enabled,q.stable_catalog_sha256 AS qualified_stable
                   FROM cloud_approval_bindings b JOIN approvals a USING(approval_id)
                   JOIN endpoint_qualifications q USING(qualification_sha256)
                   WHERE b.approval_id=? AND b.task_id=? AND b.epoch=?""",
                (approval_id, task_id, epoch),
            ).fetchone()
            if (
                binding is None
                or binding["approval_status"] != "APPROVED"
                or binding["binding_sha256"] != approval_binding_sha256
                or binding["stable_catalog_sha256"] != stable_catalog_sha256
                or binding["qualification_sha256"] != qualification_sha256
                or binding["qualified_stable"] != stable_catalog_sha256
                or not bool(binding["enabled"])
                or binding["invalidation_reason"] is not None
                or parse_utc(binding["expires_at"]) <= selected
                or int(task["cloud_dispatch_count"]) != 0
                or reserved_microusd != int(json.loads(binding["binding_json"])["max_reserved_microusd"])
            ):
                raise OrchestratorError("binding_mismatch", status=409)
            used = connection.execute(
                """SELECT COALESCE(SUM(CASE
                       WHEN state='RECONCILED' THEN actual_microusd
                       WHEN state IN ('RESERVED','DISPATCHED','AMBIGUOUS') THEN reserved_microusd
                       ELSE 0 END),0) AS used
                   FROM cloud_cost_reservations WHERE utc_month=?""",
                (utc_month,),
            ).fetchone()["used"]
            if int(used) + reserved_microusd > MONTHLY_HARD_CAP_MICROUSD:
                raise BudgetExhaustedError("cloud_budget_exhausted")
            reservation_id = new_uuid4()
            permit_id = new_uuid4()
            connection.execute(
                """INSERT INTO cloud_cost_reservations(
                   reservation_id,task_id,epoch,approval_id,utc_month,model_canonical_id,
                   provider_endpoint_tag,pricing_sha256,reserved_microusd,state,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,'RESERVED',?)""",
                (
                    reservation_id,
                    task_id,
                    epoch,
                    approval_id,
                    utc_month,
                    binding["binding_json"] and json.loads(binding["binding_json"])["model_canonical_id"],
                    json.loads(binding["binding_json"])["provider_endpoint_tag"],
                    stable_catalog_sha256,
                    reserved_microusd,
                    utc_text(selected),
                ),
            )
            connection.execute(
                """INSERT INTO cloud_dispatch_permits(
                   permit_id,reservation_id,task_id,epoch,dispatch_body_sha256,state,created_at
                   ) VALUES(?,?,?,?,?,'ISSUED',?)""",
                (permit_id, reservation_id, task_id, epoch, binding["dispatch_body_sha256"], utc_text(selected)),
            )
            changed = connection.execute(
                """UPDATE tasks SET cloud_dispatch_count=1,updated_at=?
                   WHERE task_id=? AND epoch=? AND cloud_dispatch_count=0 AND state='EXECUTING' AND tombstoned=0""",
                (utc_text(selected), task_id, epoch),
            )
            if changed.rowcount != 1:
                raise StaleEpochError()
            return {"permit_id": permit_id, "reservation_id": reservation_id, "dispatch_body_json": binding["dispatch_body_json"]}

    def consume_cloud_dispatch_permit(
        self,
        permit_id: str,
        task_id: str,
        epoch: int,
        dispatch_body_sha256: str,
        *,
        now: datetime | None = None,
    ) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            self._guarded_task_tx(connection, task_id, epoch, (TaskState.EXECUTING,))
            permit = connection.execute(
                """SELECT * FROM cloud_dispatch_permits
                   WHERE permit_id=? AND task_id=? AND epoch=? AND state='ISSUED'""",
                (permit_id, task_id, epoch),
            ).fetchone()
            if permit is None or permit["dispatch_body_sha256"] != dispatch_body_sha256:
                raise StaleEpochError()
            connection.execute(
                "UPDATE cloud_dispatch_permits SET state='CONSUMED',consumed_at=? WHERE permit_id=? AND state='ISSUED'",
                (utc_text(selected), permit_id),
            )
            connection.execute(
                "UPDATE cloud_cost_reservations SET state='DISPATCHED',dispatched_at=? WHERE reservation_id=? AND state='RESERVED'",
                (utc_text(selected), permit["reservation_id"]),
            )

    def reconcile_cloud_dispatch(
        self,
        reservation_id: str,
        *,
        actual_microusd: int | None,
        ambiguous: bool = False,
        now: datetime | None = None,
    ) -> int:
        selected = now or utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM cloud_cost_reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise OrchestratorError("not_found", status=404)
            if row["state"] in {"RECONCILED", "AMBIGUOUS"}:
                return int(row["actual_microusd"])
            if row["state"] != "DISPATCHED":
                raise InvalidStateError()
            consumed = int(row["reserved_microusd"]) if actual_microusd is None else max(0, int(actual_microusd))
            state = "AMBIGUOUS" if ambiguous or actual_microusd is None else "RECONCILED"
            connection.execute(
                """UPDATE cloud_cost_reservations SET state=?,actual_microusd=?,reconciled_at=?
                   WHERE reservation_id=? AND state='DISPATCHED'""",
                (state, consumed, utc_text(selected), reservation_id),
            )
            return consumed

    def cloud_monthly_remaining(self, utc_month: str) -> int:
        with self._lock:
            used = self._connection.execute(
                """SELECT COALESCE(SUM(CASE
                       WHEN state='RECONCILED' THEN actual_microusd
                       WHEN state IN ('RESERVED','DISPATCHED','AMBIGUOUS') THEN reserved_microusd
                       ELSE 0 END),0)
                   FROM cloud_cost_reservations WHERE utc_month=?""",
                (utc_month,),
            ).fetchone()[0]
        return max(0, MONTHLY_HARD_CAP_MICROUSD - int(used))

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
                connection.execute(
                    f"""DELETE FROM cloud_extension_inbox WHERE approval_id IN (
                       SELECT approval_id FROM approvals WHERE task_id IN ({placeholders})
                       )""",
                    tuple(task_ids),
                )
                connection.execute(
                    f"DELETE FROM research_documents WHERE task_id IN ({placeholders})",
                    tuple(task_ids),
                )
                connection.execute(
                    f"DELETE FROM research_runs WHERE task_id IN ({placeholders})",
                    tuple(task_ids),
                )
                for table in (
                    "cloud_provider_receipts",
                    "cloud_dispatch_permits",
                    "cloud_cost_reservations",
                    "cloud_presentations",
                    "cloud_approval_bindings",
                    "route_plans",
                    "routing_facts",
                    "task_profiles",
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
                "DELETE FROM cloud_extension_outbox WHERE conversation_id=?",
                (conversation_id,),
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
        self.checkpoint()

    def checkpoint(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
