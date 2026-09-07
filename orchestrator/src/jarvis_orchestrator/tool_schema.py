"""Existing schema-v6 Tool/Action migration, shared by store initialization."""

MIGRATION_6_SQL = """
ALTER TABLE tool_executions ADD COLUMN approval_id TEXT REFERENCES approvals(approval_id);
ALTER TABLE tool_executions ADD COLUMN tool_version INTEGER NOT NULL DEFAULT 1;
ALTER TABLE tool_executions ADD COLUMN verification TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_executions ADD COLUMN reason_code TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_executions ADD COLUMN idempotency_key TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_executions ADD COLUMN registry_sha256 TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_executions ADD COLUMN tool_schema_sha256 TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_executions ADD COLUMN policy_sha256 TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_executions ADD COLUMN scheduler_contract_sha256 TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_executions ADD COLUMN user_intent_binding_sha256 TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_executions ADD COLUMN preview_sha256 TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_executions ADD COLUMN arguments_payload_id TEXT;
ALTER TABLE tool_executions ADD COLUMN preview_payload_id TEXT;
ALTER TABLE tool_executions ADD COLUMN result_payload_id TEXT;
ALTER TABLE tool_executions ADD COLUMN permit_id TEXT;
ALTER TABLE tool_executions ADD COLUMN dispatch_stage TEXT NOT NULL DEFAULT '';
ALTER TABLE tool_executions ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1));
ALTER TABLE tool_executions ADD COLUMN updated_at TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS tool_execution_idempotency
ON tool_executions(idempotency_key) WHERE idempotency_key<>'';
CREATE UNIQUE INDEX IF NOT EXISTS tool_execution_approval
ON tool_executions(approval_id) WHERE approval_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS tool_sensitive_payloads(
    payload_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL REFERENCES tool_executions(execution_id),
    record_kind TEXT NOT NULL CHECK(record_kind IN ('ARGUMENTS','PREVIEW','RESULT','PRIVATE_NOTE')),
    key_version INTEGER NOT NULL CHECK(key_version>=1),
    nonce BLOB NOT NULL CHECK(length(nonce)=12),
    ciphertext_and_tag BLOB NOT NULL,
    content_sha256 TEXT NOT NULL,
    aad_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    purge_after TEXT,
    purged_at TEXT,
    UNIQUE(key_version,nonce),
    UNIQUE(execution_id,record_kind)
);
CREATE TABLE IF NOT EXISTS tool_execution_receipts(
    execution_id TEXT NOT NULL REFERENCES tool_executions(execution_id),
    stage TEXT NOT NULL CHECK(stage IN ('HOST_RECEIVED','OS_CALL_ENTERING','OS_CALL_RETURNED','VERIFICATION_COMPLETE')),
    semantic_sha256 TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY(execution_id,stage)
);
CREATE TABLE IF NOT EXISTS tool_late_receipts(
    late_receipt_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL REFERENCES tool_executions(execution_id),
    stage TEXT NOT NULL,
    semantic_sha256 TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS action_approval_presentations(
    presentation_id TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL REFERENCES approvals(approval_id),
    execution_id TEXT NOT NULL REFERENCES tool_executions(execution_id),
    owner_session_id TEXT,
    requested_event_id TEXT UNIQUE,
    decision_nonce TEXT NOT NULL,
    preview_fetch_nonce TEXT NOT NULL,
    preview_fetch_nonce_sha256 TEXT NOT NULL,
    preview_consumed_at TEXT,
    state TEXT NOT NULL CHECK(state IN ('PENDING','PRESENTED','DECIDING','RESOLVED','INVALIDATED')),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS action_approval_outbox(
    event_id TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL REFERENCES approvals(approval_id),
    presentation_id TEXT NOT NULL REFERENCES action_approval_presentations(presentation_id),
    event_type TEXT NOT NULL,
    semantic_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('PENDING','SENT','ACKED','INVALIDATED')),
    created_at TEXT NOT NULL,
    last_sent_at TEXT
);
CREATE TABLE IF NOT EXISTS action_approval_inbox(
    event_id TEXT PRIMARY KEY,
    semantic_sha256 TEXT NOT NULL,
    event_type TEXT NOT NULL,
    approval_id TEXT NOT NULL,
    presentation_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS content_provenance(
    entity_kind TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    provenance_sha256 TEXT NOT NULL,
    policy_label TEXT NOT NULL CHECK(policy_label IN ('STANDARD','LOCAL_PRIVATE')),
    egress_disposition TEXT NOT NULL CHECK(egress_disposition IN ('REVIEWABLE','NO_EGRESS')),
    created_at TEXT NOT NULL,
    PRIMARY KEY(entity_kind,entity_id)
);
CREATE TABLE IF NOT EXISTS model_request_provenance(
    reservation_id TEXT PRIMARY KEY REFERENCES reservations(reservation_id),
    consumed_entity_ids_json TEXT NOT NULL,
    aggregate_provenance_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_private_notes(
    note_ref TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL UNIQUE REFERENCES tool_executions(execution_id),
    title TEXT NOT NULL,
    body_payload_id TEXT NOT NULL UNIQUE REFERENCES tool_sensitive_payloads(payload_id),
    body_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    deleted_at TEXT
);
CREATE TABLE IF NOT EXISTS action_audit(
    audit_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    arguments_sha256 TEXT NOT NULL,
    result_sha256 TEXT,
    occurred_at TEXT NOT NULL,
    retain_until TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS action_audit_retention ON action_audit(retain_until,audit_seq);

-- Pre-v6 assistant text has no trustworthy provenance reconstruction.  Mark it
-- private during migration so it can never be selected for later cloud egress.
INSERT OR IGNORE INTO content_provenance(
    entity_kind,entity_id,provenance_json,provenance_sha256,
    policy_label,egress_disposition,created_at
)
SELECT 'canonical_turn',turn_id,'{}','',
       'LOCAL_PRIVATE','NO_EGRESS',COALESCE(completed_at,created_at)
FROM canonical_turns WHERE status='COMPLETED';
"""
