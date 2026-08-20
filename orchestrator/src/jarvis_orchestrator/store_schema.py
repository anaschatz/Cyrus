"""Versioned SQLite schema and migration scripts for OrchestratorStore."""

from __future__ import annotations

from .memory import (
    MAX_MEMORY_PRINCIPAL_BYTES,
    MAX_MEMORY_PRINCIPAL_RECORDS,
    MAX_MEMORY_RECORD_BYTES,
    MAX_MEMORY_RECORD_TERMS,
    MEMORY_NORMALIZER_VERSION,
)

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
    source_event_id TEXT NOT NULL REFERENCES inbox(event_id),
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


MIGRATION_8_SQL = """
CREATE TABLE IF NOT EXISTS voice_playback_receipts (
    event_id TEXT PRIMARY KEY,
    semantic_sha256 TEXT NOT NULL,
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    epoch INTEGER NOT NULL,
    response_id TEXT NOT NULL UNIQUE,
    terminal_type TEXT NOT NULL CHECK(terminal_type IN (
        'playback.completed','playback.interrupted'
    )),
    client_sent_at TEXT NOT NULL,
    received_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS voice_playback_receipts_by_task
ON voice_playback_receipts(task_id,received_at);
"""


MIGRATION_9_SQL = f"""
CREATE TABLE IF NOT EXISTS memory_principals (
    principal_id TEXT PRIMARY KEY,
    owner_kind TEXT NOT NULL CHECK(owner_kind='local_owner'),
    owner_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    UNIQUE(owner_kind)
);
CREATE TABLE IF NOT EXISTS conversation_principals (
    conversation_id TEXT PRIMARY KEY
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    principal_id TEXT NOT NULL REFERENCES memory_principals(principal_id),
    bound_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversation_principals_by_principal
ON conversation_principals(principal_id,conversation_id);
CREATE TABLE IF NOT EXISTS memory_principal_usage (
    principal_id TEXT PRIMARY KEY REFERENCES memory_principals(principal_id),
    active_record_count INTEGER NOT NULL DEFAULT 0
        CHECK(active_record_count BETWEEN 0 AND {MAX_MEMORY_PRINCIPAL_RECORDS}),
    active_content_bytes INTEGER NOT NULL DEFAULT 0
        CHECK(active_content_bytes BETWEEN 0 AND {MAX_MEMORY_PRINCIPAL_BYTES}),
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_records_v9 (
    record_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES memory_principals(principal_id),
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    source_event_id TEXT NOT NULL REFERENCES inbox(event_id),
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL CHECK(length(content_sha256)=64),
    normalized_sha256 TEXT NOT NULL CHECK(length(normalized_sha256)=64),
    content_bytes INTEGER NOT NULL
        CHECK(content_bytes BETWEEN 1 AND {MAX_MEMORY_RECORD_BYTES}
              AND content_bytes=length(CAST(content AS BLOB))),
    term_count INTEGER NOT NULL
        CHECK(term_count BETWEEN 1 AND {MAX_MEMORY_RECORD_TERMS}),
    expires_at TEXT NOT NULL,
    normalizer_version TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(conversation_id,source_event_id),
    UNIQUE(principal_id,normalized_sha256)
);
CREATE TABLE IF NOT EXISTS memory_terms (
    principal_id TEXT NOT NULL REFERENCES memory_principals(principal_id),
    term_sha256 TEXT NOT NULL CHECK(length(term_sha256)=64),
    record_id TEXT NOT NULL REFERENCES memory_records_v9(record_id) ON DELETE CASCADE,
    PRIMARY KEY(principal_id,term_sha256,record_id)
);
CREATE INDEX IF NOT EXISTS memory_terms_by_principal_term
ON memory_terms(principal_id,term_sha256,record_id);
CREATE TABLE IF NOT EXISTS memory_migration_audits (
    schema_version INTEGER PRIMARY KEY CHECK(schema_version=9),
    source_schema_version INTEGER NOT NULL,
    source_record_count INTEGER NOT NULL CHECK(source_record_count >= 0),
    migrated_record_count INTEGER NOT NULL CHECK(migrated_record_count >= 0),
    collapsed_duplicate_count INTEGER NOT NULL
        CHECK(collapsed_duplicate_count >= 0),
    expired_record_count INTEGER NOT NULL CHECK(expired_record_count >= 0),
    conversation_binding_count INTEGER NOT NULL
        CHECK(conversation_binding_count >= 0),
    created_at TEXT NOT NULL
);
"""


MEMORY_V9_FINALIZE_SQL = f"""
CREATE INDEX memory_records_by_principal_expiry
ON memory_records(principal_id,expires_at,record_id);
CREATE INDEX memory_records_by_expiry
ON memory_records(expires_at,record_id);
CREATE INDEX memory_records_by_source_task
ON memory_records(task_id,record_id);
CREATE TRIGGER memory_records_quota_guard
BEFORE INSERT ON memory_records
BEGIN
    SELECT CASE WHEN NOT EXISTS(
        SELECT 1 FROM memory_principal_usage WHERE principal_id=NEW.principal_id
    ) THEN RAISE(ABORT,'memory principal usage missing') END;
    SELECT CASE WHEN (
        SELECT active_record_count FROM memory_principal_usage
        WHERE principal_id=NEW.principal_id
    ) + 1 > {MAX_MEMORY_PRINCIPAL_RECORDS}
    OR (
        SELECT active_content_bytes FROM memory_principal_usage
        WHERE principal_id=NEW.principal_id
    ) + NEW.content_bytes > {MAX_MEMORY_PRINCIPAL_BYTES}
    THEN RAISE(ABORT,'memory principal quota exceeded') END;
END;
CREATE TRIGGER memory_records_usage_insert
AFTER INSERT ON memory_records
BEGIN
    UPDATE memory_principal_usage
    SET active_record_count=active_record_count+1,
        active_content_bytes=active_content_bytes+NEW.content_bytes,
        updated_at=NEW.created_at
    WHERE principal_id=NEW.principal_id;
END;
CREATE TRIGGER memory_records_usage_delete
AFTER DELETE ON memory_records
BEGIN
    UPDATE memory_principal_usage
    SET active_record_count=active_record_count-1,
        active_content_bytes=active_content_bytes-OLD.content_bytes,
        updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE principal_id=OLD.principal_id;
END;
"""


MIGRATION_10_SQL = f"""
CREATE TABLE memory_key_epochs (
    epoch_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES memory_principals(principal_id),
    provider_install_id TEXT NOT NULL
        CHECK(length(provider_install_id)=64
              AND provider_install_id=lower(provider_install_id)
              AND provider_install_id NOT GLOB '*[^0-9a-f]*'),
    key_version INTEGER NOT NULL CHECK(key_version BETWEEN 1 AND 2147483647),
    state TEXT NOT NULL CHECK(state IN (
        'PROVISIONING','WRITE_ACTIVE','READABLE','RETIRED',
        'DESTROY_PENDING','DESTROYED','DESTROY_UNCERTAIN'
    )),
    confirmation_challenge TEXT NOT NULL,
    confirmation_hmac BLOB NOT NULL
        CHECK(typeof(confirmation_hmac)='blob' AND length(confirmation_hmac)=32),
    created_at TEXT NOT NULL,
    provider_verified_at TEXT NOT NULL,
    activated_at TEXT,
    retired_at TEXT,
    destroy_requested_at TEXT,
    destroy_verified_at TEXT,
    UNIQUE(principal_id,key_version),
    UNIQUE(provider_install_id,key_version),
    UNIQUE(epoch_id,principal_id,key_version)
);
CREATE UNIQUE INDEX memory_key_epochs_one_write_active
ON memory_key_epochs(principal_id) WHERE state='WRITE_ACTIVE';

CREATE TABLE memory_crypto_state (
    principal_id TEXT PRIMARY KEY REFERENCES memory_principals(principal_id),
    storage_state TEXT NOT NULL
        CHECK(storage_state IN ('LEGACY_PENDING','MIGRATING','ENCRYPTED')),
    provider_install_id TEXT
        CHECK(provider_install_id IS NULL OR (
            length(provider_install_id)=64
            AND provider_install_id=lower(provider_install_id)
            AND provider_install_id NOT GLOB '*[^0-9a-f]*'
        )),
    write_epoch_id TEXT REFERENCES memory_key_epochs(epoch_id),
    memory_revision INTEGER NOT NULL DEFAULT 0 CHECK(memory_revision>=0),
    active_operation_id TEXT,
    updated_at TEXT NOT NULL,
    CHECK(
        (storage_state='LEGACY_PENDING'
         AND provider_install_id IS NULL AND write_epoch_id IS NULL)
        OR
        (storage_state IN ('MIGRATING','ENCRYPTED')
         AND provider_install_id IS NOT NULL AND write_epoch_id IS NOT NULL)
    )
);

CREATE TABLE memory_crypto_operations (
    operation_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES memory_principals(principal_id),
    kind TEXT NOT NULL CHECK(kind IN ('MIGRATE_V9','ROTATE','DESTROY_EPOCH')),
    phase TEXT NOT NULL CHECK(phase IN (
        'PREPARED','PROVIDER_READY','COPYING','VERIFYING','CUTOVER_READY',
        'DB_ACTIVE_SWITCHED','REWRAPPING','PROVIDER_DELETE_REQUESTED',
        'VERIFYING_ABSENCE','COMPLETE','UNCERTAIN'
    )),
    source_epoch_id TEXT,
    target_epoch_id TEXT,
    expected_provider_version INTEGER,
    cursor_created_at TEXT,
    cursor_record_id TEXT,
    expected_record_count INTEGER NOT NULL DEFAULT 0
        CHECK(expected_record_count>=0),
    processed_record_count INTEGER NOT NULL DEFAULT 0
        CHECK(processed_record_count>=0),
    request_sha256 TEXT NOT NULL CHECK(
        length(request_sha256)=64 AND request_sha256=lower(request_sha256)
        AND request_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    last_error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK(
        (kind='MIGRATE_V9' AND phase IN (
            'PREPARED','PROVIDER_READY','COPYING','VERIFYING',
            'CUTOVER_READY','COMPLETE','UNCERTAIN'
        )) OR
        (kind='ROTATE' AND phase IN (
            'PREPARED','PROVIDER_READY','DB_ACTIVE_SWITCHED',
            'REWRAPPING','VERIFYING','COMPLETE','UNCERTAIN'
        )) OR
        (kind='DESTROY_EPOCH' AND phase IN (
            'PREPARED','PROVIDER_DELETE_REQUESTED','VERIFYING_ABSENCE',
            'COMPLETE','UNCERTAIN'
        ))
    )
);
CREATE UNIQUE INDEX memory_crypto_one_active_operation
ON memory_crypto_operations(principal_id)
WHERE phase!='COMPLETE';

CREATE TABLE memory_records (
    record_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES memory_principals(principal_id),
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    source_event_id TEXT NOT NULL REFERENCES inbox(event_id),
    key_epoch_id TEXT NOT NULL,
    key_version INTEGER NOT NULL CHECK(key_version BETWEEN 1 AND 2147483647),
    record_schema_version INTEGER NOT NULL CHECK(record_schema_version=10),
    envelope_version INTEGER NOT NULL CHECK(envelope_version=1),
    payload_version INTEGER NOT NULL CHECK(payload_version=1),
    payload_nonce BLOB NOT NULL
        CHECK(typeof(payload_nonce)='blob' AND length(payload_nonce)=12),
    payload_ciphertext BLOB NOT NULL
        CHECK(typeof(payload_ciphertext)='blob'
              AND length(payload_ciphertext) BETWEEN 17 AND 16400),
    wrap_nonce BLOB NOT NULL
        CHECK(typeof(wrap_nonce)='blob' AND length(wrap_nonce)=12),
    wrapped_dek BLOB NOT NULL
        CHECK(typeof(wrapped_dek)='blob' AND length(wrapped_dek)=48),
    binding_sha256 TEXT NOT NULL CHECK(
        length(binding_sha256)=64 AND binding_sha256=lower(binding_sha256)
        AND binding_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    envelope_sha256 TEXT NOT NULL CHECK(
        length(envelope_sha256)=64 AND envelope_sha256=lower(envelope_sha256)
        AND envelope_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    content_hmac BLOB NOT NULL
        CHECK(typeof(content_hmac)='blob' AND length(content_hmac)=32),
    normalized_hmac BLOB NOT NULL
        CHECK(typeof(normalized_hmac)='blob' AND length(normalized_hmac)=32),
    content_bytes INTEGER NOT NULL
        CHECK(content_bytes BETWEEN 1 AND {MAX_MEMORY_RECORD_BYTES}),
    term_count INTEGER NOT NULL
        CHECK(term_count BETWEEN 1 AND {MAX_MEMORY_RECORD_TERMS}),
    expires_at TEXT NOT NULL,
    normalizer_version TEXT NOT NULL
        CHECK(normalizer_version='{MEMORY_NORMALIZER_VERSION}'),
    created_at TEXT NOT NULL,
    FOREIGN KEY(key_epoch_id,principal_id,key_version)
        REFERENCES memory_key_epochs(epoch_id,principal_id,key_version),
    UNIQUE(conversation_id,source_event_id),
    UNIQUE(principal_id,key_version,normalized_hmac),
    UNIQUE(record_id,principal_id,key_epoch_id,key_version)
);
CREATE INDEX memory_records_by_principal_expiry
ON memory_records(principal_id,expires_at,record_id);
CREATE INDEX memory_records_by_expiry
ON memory_records(expires_at,record_id);
CREATE INDEX memory_records_by_source_task
ON memory_records(task_id,record_id);
CREATE INDEX memory_records_by_source_conversation
ON memory_records(conversation_id,record_id);
CREATE INDEX memory_records_by_epoch
ON memory_records(key_epoch_id,record_id);
CREATE INDEX memory_records_by_principal_created
ON memory_records(principal_id,created_at DESC,record_id);

CREATE TABLE memory_terms (
    principal_id TEXT NOT NULL,
    key_epoch_id TEXT NOT NULL,
    key_version INTEGER NOT NULL,
    term_hmac BLOB NOT NULL
        CHECK(typeof(term_hmac)='blob' AND length(term_hmac)=32),
    record_id TEXT NOT NULL,
    PRIMARY KEY(principal_id,key_version,term_hmac,record_id),
    FOREIGN KEY(record_id,principal_id,key_epoch_id,key_version)
        REFERENCES memory_records(record_id,principal_id,key_epoch_id,key_version)
        ON DELETE CASCADE
);
CREATE INDEX memory_terms_by_principal_term
ON memory_terms(principal_id,key_version,term_hmac,record_id);
CREATE INDEX memory_terms_by_record
ON memory_terms(record_id,term_hmac);

CREATE TABLE memory_record_tombstones (
    record_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES memory_principals(principal_id),
    source_binding_sha256 TEXT NOT NULL CHECK(
        length(source_binding_sha256)=64
        AND source_binding_sha256=lower(source_binding_sha256)
        AND source_binding_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    reason TEXT NOT NULL CHECK(reason IN (
        'explicit_forget','control_delete','source_conversation_delete','ttl_expiry'
    )),
    delete_request_id TEXT,
    memory_revision INTEGER NOT NULL CHECK(memory_revision>=0),
    deleted_at TEXT NOT NULL
);
CREATE UNIQUE INDEX memory_tombstones_by_delete_request
ON memory_record_tombstones(delete_request_id)
WHERE delete_request_id IS NOT NULL;
CREATE TABLE memory_record_audits (
    audit_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES memory_principals(principal_id),
    record_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN (
        'MIGRATED','CREATED','EXPLICIT_FORGOT','CONTROL_DELETED',
        'SOURCE_CONVERSATION_DELETED','TTL_EXPIRED','REWRAPPED'
    )),
    actor_kind TEXT NOT NULL CHECK(actor_kind IN (
        'explicit_command','control','migration','conversation_lifecycle','ttl_sweep'
    )),
    actor_id TEXT,
    source_binding_sha256 TEXT NOT NULL CHECK(
        length(source_binding_sha256)=64
        AND source_binding_sha256=lower(source_binding_sha256)
        AND source_binding_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    operation_id TEXT NOT NULL,
    old_epoch_id TEXT,
    new_epoch_id TEXT,
    outcome_code TEXT NOT NULL,
    created_at TEXT NOT NULL,
    audit_sha256 TEXT NOT NULL CHECK(
        length(audit_sha256)=64 AND audit_sha256=lower(audit_sha256)
        AND audit_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    UNIQUE(record_id,action,operation_id)
);
CREATE INDEX memory_record_audits_by_record
ON memory_record_audits(principal_id,record_id,created_at,audit_id);

CREATE TABLE memory_crypto_migration_audits (
    operation_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES memory_principals(principal_id),
    source_schema INTEGER NOT NULL CHECK(source_schema=9),
    target_schema INTEGER NOT NULL CHECK(target_schema=10),
    provider_install_id TEXT NOT NULL CHECK(
        length(provider_install_id)=64
        AND provider_install_id=lower(provider_install_id)
        AND provider_install_id NOT GLOB '*[^0-9a-f]*'
    ),
    key_version INTEGER NOT NULL,
    source_record_count INTEGER NOT NULL CHECK(source_record_count>=0),
    encrypted_record_count INTEGER NOT NULL CHECK(encrypted_record_count>=0),
    expired_record_count INTEGER NOT NULL CHECK(expired_record_count>=0),
    source_content_bytes INTEGER NOT NULL CHECK(source_content_bytes>=0),
    encrypted_content_bytes INTEGER NOT NULL CHECK(encrypted_content_bytes>=0),
    started_at TEXT NOT NULL,
    cutover_at TEXT NOT NULL,
    contract_sha256 TEXT NOT NULL CHECK(
        length(contract_sha256)=64 AND contract_sha256=lower(contract_sha256)
        AND contract_sha256 NOT GLOB '*[^0-9a-f]*'
    )
);

CREATE TRIGGER memory_records_quota_guard
BEFORE INSERT ON memory_records
WHEN (SELECT storage_state FROM memory_crypto_state
      WHERE principal_id=NEW.principal_id)='ENCRYPTED'
BEGIN
    SELECT CASE WHEN NOT EXISTS(
        SELECT 1 FROM memory_principal_usage WHERE principal_id=NEW.principal_id
    ) THEN RAISE(ABORT,'memory principal usage missing') END;
    SELECT CASE WHEN (
        SELECT active_record_count FROM memory_principal_usage
        WHERE principal_id=NEW.principal_id
    ) + 1 > {MAX_MEMORY_PRINCIPAL_RECORDS}
    OR (
        SELECT active_content_bytes FROM memory_principal_usage
        WHERE principal_id=NEW.principal_id
    ) + NEW.content_bytes > {MAX_MEMORY_PRINCIPAL_BYTES}
    THEN RAISE(ABORT,'memory principal quota exceeded') END;
END;
CREATE TRIGGER memory_records_usage_insert
AFTER INSERT ON memory_records
WHEN (SELECT storage_state FROM memory_crypto_state
      WHERE principal_id=NEW.principal_id)='ENCRYPTED'
BEGIN
    UPDATE memory_principal_usage
    SET active_record_count=active_record_count+1,
        active_content_bytes=active_content_bytes+NEW.content_bytes,
        updated_at=NEW.created_at
    WHERE principal_id=NEW.principal_id;
    UPDATE memory_crypto_state SET memory_revision=memory_revision+1,
        updated_at=NEW.created_at WHERE principal_id=NEW.principal_id;
END;
CREATE TRIGGER memory_records_usage_delete
AFTER DELETE ON memory_records
WHEN (SELECT storage_state FROM memory_crypto_state
      WHERE principal_id=OLD.principal_id)='ENCRYPTED'
BEGIN
    UPDATE memory_principal_usage
    SET active_record_count=active_record_count-1,
        active_content_bytes=active_content_bytes-OLD.content_bytes,
        updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE principal_id=OLD.principal_id;
    UPDATE memory_crypto_state SET memory_revision=memory_revision+1,
        updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE principal_id=OLD.principal_id;
END;
CREATE TRIGGER memory_epoch_destroy_guard
BEFORE UPDATE OF state ON memory_key_epochs
WHEN NEW.state IN ('DESTROY_PENDING','DESTROYED')
     AND EXISTS(SELECT 1 FROM memory_records WHERE key_epoch_id=OLD.epoch_id)
BEGIN
    SELECT RAISE(ABORT,'memory epoch remains referenced');
END;
"""



__all__ = [
    "MEMORY_V9_FINALIZE_SQL",
    "MIGRATION_2_SQL",
    "MIGRATION_3_SQL",
    "MIGRATION_4_SQL",
    "MIGRATION_7_SQL",
    "MIGRATION_8_SQL",
    "MIGRATION_9_SQL",
    "MIGRATION_10_SQL",
    "SCHEMA_SQL",
]
