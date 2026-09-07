"""SQLite WAL persistence, transactional epoch fences, budgets and outbox."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from .cloud_contract import (
    MONTHLY_HARD_CAP_MICROUSD,
    CloudApprovalBindingV1,
    QualifiedEndpointV1,
    build_dispatch_body,
)
from .cloud_routing import RoutePlanV1, RoutingFactsV1, TaskProfileV1
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
    require_uuid4,
    utc_now,
    utc_text,
)
from .jcs import domain_sha256, jcs_bytes, jcs_text
from .memory import (
    MAX_MEMORY_PRINCIPAL_BYTES,
    MAX_MEMORY_PRINCIPAL_RECORDS,
    MAX_MEMORY_QUERY_BYTES,
    MAX_MEMORY_QUERY_TERMS,
    MAX_MEMORY_RECORD_BYTES,
    MAX_MEMORY_RECORD_TERMS,
    MEMORY_NORMALIZER_VERSION,
    MEMORY_RECORD_TTL_DAYS,
    ExplicitMemoryCommandResult,
    ExplicitMemoryCommandStatus,
    explicit_memory_response,
    memory_normalized_sha256,
    memory_search_terms,
    normalize_explicit_memory_target,
    parse_explicit_memory_command,
)
from .memory_crypto import (
    MEMORY_ENVELOPE_VERSION,
    MEMORY_RECORD_SCHEMA_VERSION,
    MemoryCryptoError,
    MemoryEnvelopeCipherV1,
    MemoryEnvelopeInvalid,
    MemoryEnvelopeV1,
    MemoryEpochSessionV1,
    MemoryRecordBindingV1,
)
from .memory_release import shadow_safety_fields, validate_shadow_audit_payload
from .memory_runtime import MemoryCryptoRuntime
from .private_state import (
    absolute_path_no_resolve,
    assert_private_database,
    prepare_private_database,
)
from .reducer import require_transition
from .research import (
    MAX_RUN_DOCUMENT_BYTES,
    EvidenceBundleV1,
    NormalizedDocumentV1,
    SearchPlanV1,
    reconstruct,
    validate_syntactic_citation_coverage,
)
from .scheduler_store import MIGRATION_5_SQL
from .store_schema import (
    MEMORY_V9_FINALIZE_SQL,
    MIGRATION_2_SQL,
    MIGRATION_3_SQL,
    MIGRATION_4_SQL,
    MIGRATION_7_SQL,
    MIGRATION_8_SQL,
    MIGRATION_9_SQL,
    MIGRATION_10_SQL,
    SCHEMA_SQL,
)
from .tool_provenance import (
    ContentProvenanceV1,
    ContentSourceKind,
    ProvenanceError,
)
from .tool_store import MIGRATION_6_SQL

SCHEMA_VERSION = 10
MEMORY_SWEEP_LIMIT = 256
MEMORY_SWEEP_INTERVAL = timedelta(hours=1)
MEMORY_RECALL_BUSY_TIMEOUT_MS = 100
MEMORY_RECALL_DEADLINE_SECONDS = 0.150
MEMORY_PAYLOAD_CONTRACT = "jarvis.memory-payload/1"
MEMORY_CRYPTO_OPERATION_CONTRACT = "jarvis.memory-crypto-operation/1"
MEMORY_AUDIT_CONTRACT = "jarvis.memory-record-audit/1"
MEMORY_PROVIDER_CONFIRMATION_TEXT = "jarvis-memory-key-confirmation-v1"
SQLITE_MAX_INTEGER = (1 << 63) - 1


class MemoryStorageUnavailable(RuntimeError):
    """Stable fail-closed boundary for an unavailable encrypted authority."""

    def __init__(self, code: str = "memory_storage_unavailable") -> None:
        super().__init__(code)
        self.code = code


class ControlMutationKind(StrEnum):
    """Closed same-SQLite mutations accepted by the HTTP control boundary."""

    CREATE_CONVERSATION = "create_conversation"
    DELETE_CONVERSATION = "delete_conversation"
    DELETE_MEMORY_RECORD = "delete_memory_record"
    CANCEL_TASK = "cancel_task"
    DECIDE_APPROVAL = "decide_approval"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class ControlMutationRequest:
    kind: ControlMutationKind
    resource_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ControlMutationKind):
            raise TypeError("control mutation kind is invalid")
        resource_required = self.kind in {
            ControlMutationKind.DELETE_CONVERSATION,
            ControlMutationKind.DELETE_MEMORY_RECORD,
            ControlMutationKind.CANCEL_TASK,
            ControlMutationKind.DECIDE_APPROVAL,
        }
        if resource_required != (self.resource_id is not None):
            raise ValueError("control mutation resource binding is invalid")
        if self.resource_id is not None and not isinstance(self.resource_id, str):
            raise TypeError("control mutation resource is invalid")


class _MemoryQuotaExceeded(RuntimeError):
    pass


def _execute_script_tx(connection: sqlite3.Connection, script: str) -> None:
    """Execute a script without sqlite3.executescript's implicit COMMIT."""

    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if not sqlite3.complete_statement(pending):
            continue
        statement = pending.strip()
        pending = ""
        if statement:
            connection.execute(statement)
    if pending.strip():
        raise RuntimeError("incomplete database migration statement")


def _memory_term_sha256(term: str) -> str:
    return hashlib.sha256(term.encode("utf-8")).hexdigest()


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
        memory_crypto: MemoryCryptoRuntime | None = None,
        memory_storage_enabled: bool = True,
    ) -> None:
        if not isinstance(tool_authority_enabled, bool):
            raise TypeError("tool_authority_enabled must be boolean")
        if not isinstance(memory_storage_enabled, bool):
            raise TypeError("memory_storage_enabled must be boolean")
        self._memory_crypto = self._validated_memory_runtime(memory_crypto)
        selected_path = absolute_path_no_resolve(path)
        database_preexisting = selected_path.exists()
        self.path = prepare_private_database(path)
        with ExitStack() as cleanup:
            self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
            cleanup.callback(self._connection.close)
            self._connection.row_factory = sqlite3.Row
            self._lock = threading.RLock()
            self.model_request_limit = model_request_limit
            self.output_token_limit = output_token_limit
            self.active_budget_seconds = active_budget_seconds
            self.approval_ttl = approval_ttl
            self.disconnect_ttl = disconnect_ttl
            self.delivery_ttl = delivery_ttl
            self.tool_authority_enabled = tool_authority_enabled
            self._memory_storage_enabled = memory_storage_enabled
            self._memory_epoch_session: MemoryEpochSessionV1 | None = None
            with self._lock:
                self._connection.execute("PRAGMA journal_mode=WAL")
                assert_private_database(self.path)
                self._connection.execute("PRAGMA foreign_keys=ON")
                self._connection.execute("PRAGMA synchronous=FULL")
                self._connection.execute("PRAGMA busy_timeout=2000")
                self._connection.execute("BEGIN IMMEDIATE")
                with self._connection:
                    if not self._memory_storage_enabled and database_preexisting:
                        schema_table = self._connection.execute(
                            "SELECT 1 FROM sqlite_master "
                            "WHERE type='table' AND name='schema_meta'"
                        ).fetchone()
                        if schema_table is None:
                            raise MemoryStorageUnavailable(
                                "memory_disabled_legacy_schema"
                            )
                    self._connection.execute(
                        "CREATE TABLE IF NOT EXISTS schema_meta "
                        "(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
                    )
                    existing_version = self._connection.execute(
                        "SELECT value FROM schema_meta WHERE key='schema_version'"
                    ).fetchone()
                    supported = {str(value) for value in range(1, SCHEMA_VERSION + 1)}
                    if (
                        existing_version is not None
                        and existing_version["value"] not in supported
                    ):
                        raise RuntimeError(
                            "unsupported orchestrator database schema version"
                        )
                    source_schema_version = (
                        1 if existing_version is None else int(existing_version["value"])
                    )
                    if (
                        not self._memory_storage_enabled
                        and database_preexisting
                        and (
                            existing_version is None
                            or source_schema_version < SCHEMA_VERSION
                        )
                    ):
                        # A memory-disabled runtime may inspect schema_meta, but it
                        # must never read, rename, migrate, sweep, or delete legacy
                        # plaintext memory.  An operator must explicitly enable
                        # and complete the encrypted migration first.
                        raise MemoryStorageUnavailable(
                            "memory_disabled_legacy_schema"
                        )
                    _execute_script_tx(self._connection, SCHEMA_SQL)
                    if existing_version is None:
                        self._connection.execute(
                            "INSERT INTO schema_meta(key,value) "
                            "VALUES('schema_version','1')"
                        )
                    current_version = source_schema_version
                    if current_version == 1:
                        _execute_script_tx(self._connection, MIGRATION_2_SQL)
                        self._set_schema_version_tx(self._connection, 1, 2)
                        current_version = 2
                    if current_version == 2:
                        _execute_script_tx(self._connection, MIGRATION_3_SQL)
                        self._set_schema_version_tx(self._connection, 2, 3)
                        current_version = 3
                    if current_version == 3:
                        task_columns = {
                            row["name"]
                            for row in self._connection.execute(
                                "PRAGMA table_info(tasks)"
                            ).fetchall()
                        }
                        if "cloud_dispatch_count" not in task_columns:
                            self._connection.execute(
                                "ALTER TABLE tasks ADD COLUMN cloud_dispatch_count "
                                "INTEGER NOT NULL DEFAULT 0 CHECK("
                                "cloud_dispatch_count >= 0 AND cloud_dispatch_count <= 1)"
                            )
                        _execute_script_tx(self._connection, MIGRATION_4_SQL)
                        self._set_schema_version_tx(self._connection, 3, 4)
                        current_version = 4
                    if current_version == 4:
                        _execute_script_tx(self._connection, MIGRATION_5_SQL)
                        self._set_schema_version_tx(self._connection, 4, 5)
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
                            if stripped.startswith(
                                "ALTER TABLE tool_executions ADD COLUMN "
                            ):
                                column_name = stripped.split()[5]
                                if column_name in tool_columns:
                                    continue
                            migration_6_lines.append(statement_line)
                        _execute_script_tx(
                            self._connection, "\n".join(migration_6_lines)
                        )
                        self._set_schema_version_tx(self._connection, 5, 6)
                        current_version = 6
                    if current_version == 6:
                        _execute_script_tx(self._connection, MIGRATION_7_SQL)
                        self._set_schema_version_tx(self._connection, 6, 7)
                        current_version = 7
                    if current_version == 7:
                        _execute_script_tx(self._connection, MIGRATION_8_SQL)
                        self._set_schema_version_tx(self._connection, 7, 8)
                        current_version = 8
                    if current_version == 8:
                        self._migrate_memory_v9_tx(
                            self._connection,
                            source_schema_version=source_schema_version,
                            now=utc_now(),
                        )
                        self._set_schema_version_tx(self._connection, 8, 9)
                        current_version = 9
                    if current_version == 9:
                        self._migrate_memory_v10_tx(
                            self._connection,
                            now=utc_now(),
                        )
                        self._set_schema_version_tx(self._connection, 9, 10)
                        current_version = 10
                    if current_version != SCHEMA_VERSION:
                        raise RuntimeError("orchestrator database migration incomplete")
                assert_private_database(self.path)
            cleanup.pop_all()
        # Sweeping is deliberately deferred until an enabled runtime has
        # opened and verified its encrypted epoch session.  In disabled mode
        # even a schema-v10 LEGACY_PENDING table is completely untouched.

    @staticmethod
    def _validated_memory_runtime(
        runtime: MemoryCryptoRuntime | None,
    ) -> MemoryCryptoRuntime | None:
        if runtime is None:
            return None
        # A fresh platform provider intentionally has no installation id until
        # ``prepare_memory_storage`` reaches its journaled bootstrap phase.
        # Validate only the immutable cipher seam here; the exact id is bound
        # and checked immediately after bootstrap or on encrypted reopen.
        try:
            cipher = runtime.cipher
        except (AttributeError, MemoryCryptoError):
            raise TypeError("memory crypto runtime is invalid") from None
        if not isinstance(cipher, MemoryEnvelopeCipherV1):
            raise TypeError("memory crypto runtime is invalid")
        return runtime

    @staticmethod
    def _set_schema_version_tx(
        connection: sqlite3.Connection, old: int, new: int
    ) -> None:
        changed = connection.execute(
            "UPDATE schema_meta SET value=? "
            "WHERE key='schema_version' AND value=?",
            (str(new), str(old)),
        )
        if changed.rowcount != 1:
            raise RuntimeError("orchestrator database schema version changed")

    @staticmethod
    def _memory_content_fields(content: str) -> dict[str, Any]:
        if not isinstance(content, str) or not content or nfc(content) != content:
            raise ValueError("memory content is invalid")
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > MAX_MEMORY_RECORD_BYTES:
            raise ValueError("memory content exceeds the byte limit")
        normalized = normalize_explicit_memory_target(content)
        if not normalized:
            raise ValueError("memory content normalizes to empty")
        terms = memory_search_terms(content)
        if not terms:
            raise ValueError("memory content has no searchable terms")
        if len(terms) > MAX_MEMORY_RECORD_TERMS:
            raise ValueError("memory content has too many unique terms")
        return {
            "content": content,
            "content_sha256": canonical_sha256({"content": content}),
            "normalized_sha256": memory_normalized_sha256(content),
            "content_bytes": content_bytes,
            "terms": terms,
            "term_count": len(terms),
            "normalizer_version": MEMORY_NORMALIZER_VERSION,
        }

    def _migrate_memory_v9_tx(
        self,
        connection: sqlite3.Connection,
        *,
        source_schema_version: int,
        now: datetime,
    ) -> None:
        """Rebuild legacy memory under one stable local-owner principal."""

        required_columns = {
            "record_id",
            "conversation_id",
            "task_id",
            "source_event_id",
            "scoped_user_id",
            "content",
            "content_sha256",
            "metadata_json",
            "created_at",
        }
        observed_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(memory_records)"
            ).fetchall()
        }
        if observed_columns != required_columns:
            raise RuntimeError("legacy memory schema is malformed")
        forbidden = {
            "memory_principals",
            "conversation_principals",
            "memory_principal_usage",
            "memory_records_v9",
            "memory_terms",
            "memory_migration_audits",
        }
        existing_v9 = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if row["name"] in forbidden
        }
        if existing_v9:
            raise RuntimeError("partial memory v9 schema is present")

        conversations = connection.execute(
            "SELECT conversation_id,created_at FROM conversations "
            "ORDER BY conversation_id"
        ).fetchall()
        conversation_ids = {row["conversation_id"] for row in conversations}
        task_bindings = {
            row["task_id"]: row
            for row in connection.execute(
                """SELECT t.task_id,t.conversation_id,t.epoch,t.turn_id,t.state,
                          ct.user_language,ct.status AS turn_status
                   FROM tasks t JOIN canonical_turns ct USING(task_id)"""
            ).fetchall()
        }
        event_bindings = {
            (row["event_id"], row["task_id"], row["turn_id"])
            for row in connection.execute(
                "SELECT event_id,task_id,turn_id FROM inbox"
            ).fetchall()
        }
        legacy_rows = connection.execute(
            "SELECT * FROM memory_records ORDER BY created_at,record_id"
        ).fetchall()
        selected_at = parse_utc(utc_text(now))
        prepared: list[dict[str, Any]] = []
        expired_count = 0
        for row in legacy_rows:
            try:
                content = row["content"]
                fields = self._memory_content_fields(content)
                if row["content_sha256"] != fields["content_sha256"]:
                    raise ValueError("legacy content digest does not match")
                metadata = json.loads(row["metadata_json"])
                if (
                    not isinstance(metadata, dict)
                    or canonical_json(metadata) != row["metadata_json"]
                ):
                    raise ValueError("legacy metadata is not canonical")
                created = parse_utc(row["created_at"])
                if utc_text(created) != row["created_at"]:
                    raise ValueError("legacy timestamp is not canonical")
                conversation_id = row["conversation_id"]
                task_id = row["task_id"]
                source_event_id = row["source_event_id"]
                task_binding = task_bindings.get(task_id)
                expected_scope = f"anamnesis::jarvis::{conversation_id}"
                expected_metadata_keys = {
                    "conversation_id",
                    "task_id",
                    "task_epoch",
                    "turn_id",
                    "source_event_id",
                    "language",
                    "_anamnesis_namespace",
                    "_anamnesis_user_id",
                }
                if (
                    conversation_id not in conversation_ids
                    or task_binding is None
                    or task_binding["conversation_id"] != conversation_id
                    or task_binding["state"] != TaskState.COMPLETED.value
                    or task_binding["turn_status"] != "COMPLETED"
                    or (
                        source_event_id,
                        task_id,
                        task_binding["turn_id"],
                    )
                    not in event_bindings
                    or row["scoped_user_id"] != expected_scope
                    or set(metadata) != expected_metadata_keys
                    or metadata.get("conversation_id") != conversation_id
                    or metadata.get("task_id") != task_id
                    or metadata.get("task_epoch") != str(task_binding["epoch"])
                    or metadata.get("turn_id") != task_binding["turn_id"]
                    or metadata.get("source_event_id") != source_event_id
                    or metadata.get("language") != task_binding["user_language"]
                    or metadata.get("_anamnesis_namespace") != "jarvis"
                    or metadata.get("_anamnesis_user_id") != conversation_id
                ):
                    raise ValueError("legacy source binding is invalid")
                expires = created + timedelta(days=MEMORY_RECORD_TTL_DAYS)
                if expires <= selected_at:
                    expired_count += 1
                    continue
                prepared.append(
                    {
                        **fields,
                        "record_id": row["record_id"],
                        "conversation_id": conversation_id,
                        "task_id": task_id,
                        "source_event_id": source_event_id,
                        "metadata_json": row["metadata_json"],
                        "created": created,
                        "created_at": row["created_at"],
                        "expires_at": utc_text(expires),
                    }
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError("legacy memory record is malformed") from exc

        retained: list[dict[str, Any]] = []
        seen_normalized: set[str] = set()
        duplicate_count = 0
        for item in sorted(
            prepared, key=lambda value: (value["created"], value["record_id"])
        ):
            normalized_sha256 = item["normalized_sha256"]
            if normalized_sha256 in seen_normalized:
                duplicate_count += 1
                continue
            seen_normalized.add(normalized_sha256)
            retained.append(item)
        retained_bytes = sum(item["content_bytes"] for item in retained)
        if (
            len(retained) > MAX_MEMORY_PRINCIPAL_RECORDS
            or retained_bytes > MAX_MEMORY_PRINCIPAL_BYTES
        ):
            raise RuntimeError("legacy memory exceeds the local-owner quota")

        _execute_script_tx(connection, MIGRATION_9_SQL)
        principal_id = new_uuid4()
        stamp = utc_text(selected_at)
        connection.execute(
            "INSERT INTO memory_principals("
            "principal_id,owner_kind,owner_key,created_at) VALUES("
            "?,'local_owner','local-owner',?)",
            (principal_id, stamp),
        )
        connection.execute(
            "INSERT INTO memory_principal_usage("
            "principal_id,active_record_count,active_content_bytes,updated_at) "
            "VALUES(?,0,0,?)",
            (principal_id, stamp),
        )
        for conversation in conversations:
            connection.execute(
                "INSERT INTO conversation_principals("
                "conversation_id,principal_id,bound_at) VALUES(?,?,?)",
                (
                    conversation["conversation_id"],
                    principal_id,
                    conversation["created_at"],
                ),
            )
        for item in retained:
            connection.execute(
                """INSERT INTO memory_records_v9(
                   record_id,principal_id,conversation_id,task_id,source_event_id,
                   content,content_sha256,normalized_sha256,content_bytes,
                   term_count,expires_at,normalizer_version,metadata_json,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    item["record_id"],
                    principal_id,
                    item["conversation_id"],
                    item["task_id"],
                    item["source_event_id"],
                    item["content"],
                    item["content_sha256"],
                    item["normalized_sha256"],
                    item["content_bytes"],
                    item["term_count"],
                    item["expires_at"],
                    item["normalizer_version"],
                    item["metadata_json"],
                    item["created_at"],
                ),
            )
            for term in item["terms"]:
                connection.execute(
                    "INSERT INTO memory_terms("
                    "principal_id,term_sha256,record_id) VALUES(?,?,?)",
                    (principal_id, _memory_term_sha256(term), item["record_id"]),
                )
        connection.execute(
            "UPDATE memory_principal_usage SET active_record_count=?,"
            "active_content_bytes=?,updated_at=? WHERE principal_id=?",
            (len(retained), retained_bytes, stamp, principal_id),
        )
        connection.execute("ALTER TABLE memory_records RENAME TO memory_records_v8")
        connection.execute("ALTER TABLE memory_records_v9 RENAME TO memory_records")
        connection.execute("DROP TABLE memory_records_v8")
        _execute_script_tx(connection, MEMORY_V9_FINALIZE_SQL)
        connection.execute(
            """INSERT INTO memory_migration_audits(
               schema_version,source_schema_version,source_record_count,
               migrated_record_count,collapsed_duplicate_count,
               expired_record_count,conversation_binding_count,created_at
               ) VALUES(9,?,?,?,?,?,?,?)""",
            (
                source_schema_version,
                len(legacy_rows),
                len(retained),
                duplicate_count,
                expired_count,
                len(conversations),
                stamp,
            ),
        )

    @staticmethod
    def _migrate_memory_v10_tx(
        connection: sqlite3.Connection,
        *,
        now: datetime,
    ) -> None:
        """Install encrypted memory shape without touching the key provider.

        The legacy v9 authority is renamed and becomes inaccessible to normal
        runtime methods.  ``prepare_memory_storage`` performs the separately
        journaled provider-backed conversion before encrypted memory is usable.
        """

        expected_v9_columns = {
            "record_id",
            "principal_id",
            "conversation_id",
            "task_id",
            "source_event_id",
            "content",
            "content_sha256",
            "normalized_sha256",
            "content_bytes",
            "term_count",
            "expires_at",
            "normalizer_version",
            "metadata_json",
            "created_at",
        }
        observed = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(memory_records)"
            ).fetchall()
        }
        if observed != expected_v9_columns:
            raise RuntimeError("memory v9 schema is malformed")
        partial_names = {
            "memory_records_v9_legacy",
            "memory_terms_v9_legacy",
            "memory_crypto_state",
            "memory_key_epochs",
            "memory_crypto_operations",
            "memory_record_tombstones",
            "memory_record_audits",
            "memory_crypto_migration_audits",
        }
        existing = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if row["name"] in partial_names
        }
        if existing:
            raise RuntimeError("partial memory v10 schema is present")
        principals = connection.execute(
            "SELECT principal_id,owner_kind,owner_key FROM memory_principals"
        ).fetchall()
        if (
            len(principals) != 1
            or principals[0]["owner_kind"] != "local_owner"
            or principals[0]["owner_key"] != "local-owner"
        ):
            raise RuntimeError("memory v10 principal binding is invalid")

        for trigger in (
            "memory_records_quota_guard",
            "memory_records_usage_insert",
            "memory_records_usage_delete",
        ):
            connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        for index in (
            "memory_records_by_principal_expiry",
            "memory_records_by_expiry",
            "memory_records_by_source_task",
            "memory_terms_by_principal_term",
        ):
            connection.execute(f"DROP INDEX IF EXISTS {index}")
        connection.execute(
            "ALTER TABLE memory_terms RENAME TO memory_terms_v9_legacy"
        )
        connection.execute(
            "ALTER TABLE memory_records RENAME TO memory_records_v9_legacy"
        )
        _execute_script_tx(connection, MIGRATION_10_SQL)
        stamp = utc_text(now)
        connection.execute(
            """INSERT INTO memory_crypto_state(
               principal_id,storage_state,provider_install_id,write_epoch_id,
               memory_revision,active_operation_id,updated_at
               ) VALUES(?,'LEGACY_PENDING',NULL,NULL,0,NULL,?)""",
            (principals[0]["principal_id"], stamp),
        )

    @staticmethod
    def _local_owner_principal_tx(connection: sqlite3.Connection) -> str:
        rows = connection.execute(
            "SELECT principal_id FROM memory_principals "
            "WHERE owner_kind='local_owner' AND owner_key='local-owner'"
        ).fetchall()
        if len(rows) != 1:
            raise RuntimeError("local-owner memory principal is unavailable")
        return str(rows[0]["principal_id"])

    @classmethod
    def _principal_for_conversation_tx(
        cls, connection: sqlite3.Connection, conversation_id: str
    ) -> str:
        row = connection.execute(
            "SELECT principal_id FROM conversation_principals "
            "WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("conversation memory principal is unavailable")
        principal_id = str(row["principal_id"])
        if principal_id != cls._local_owner_principal_tx(connection):
            raise RuntimeError("conversation memory principal is invalid")
        return principal_id

    @classmethod
    def _bind_conversation_principal_tx(
        cls,
        connection: sqlite3.Connection,
        conversation_id: str,
        bound_at: str,
    ) -> str:
        principal_id = cls._local_owner_principal_tx(connection)
        connection.execute(
            "INSERT INTO conversation_principals("
            "conversation_id,principal_id,bound_at) VALUES(?,?,?)",
            (conversation_id, principal_id, bound_at),
        )
        return principal_id

    def _memory_runtime_or_unavailable(self) -> MemoryCryptoRuntime:
        if not self._memory_storage_enabled:
            raise MemoryStorageUnavailable("memory_disabled")
        runtime = self._memory_crypto
        if runtime is None:
            raise MemoryStorageUnavailable("memory_crypto_unavailable")
        return runtime

    def _memory_session_or_unavailable(self) -> MemoryEpochSessionV1:
        session = self._memory_epoch_session
        if session is None:
            raise MemoryStorageUnavailable("memory_crypto_session_unavailable")
        return session

    @staticmethod
    def _memory_crypto_state_tx(
        connection: sqlite3.Connection, principal_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM memory_crypto_state WHERE principal_id=?",
            (principal_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("memory crypto state is unavailable")
        return row

    @classmethod
    def _require_current_memory_session_tx(
        cls,
        connection: sqlite3.Connection,
        *,
        principal_id: str,
        session: MemoryEpochSessionV1,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state = dict(cls._memory_crypto_state_tx(connection, principal_id))
        if state["storage_state"] != "ENCRYPTED":
            raise MemoryStorageUnavailable("memory_storage_not_prepared")
        epoch_row = connection.execute(
            "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
            (state["write_epoch_id"],),
        ).fetchone()
        if epoch_row is None:
            raise MemoryStorageUnavailable("memory_key_epoch_missing")
        epoch = dict(epoch_row)
        cls._require_session_epoch(session, state, epoch)
        return state, epoch

    def memory_storage_status(self) -> dict[str, Any]:
        """Return a content-free cached storage status without provider calls."""

        if not self._memory_storage_enabled:
            return {
                "storage_state": "DISABLED",
                "ready": False,
                "memory_revision": 0,
                "referenced_epoch_count": 0,
                "pending_operation_count": 0,
            }

        with self._lock:
            principal_id = self._local_owner_principal_tx(self._connection)
            state = self._memory_crypto_state_tx(self._connection, principal_id)
            pending = self._connection.execute(
                "SELECT COUNT(*) FROM memory_crypto_operations "
                "WHERE principal_id=? AND phase!='COMPLETE'",
                (principal_id,),
            ).fetchone()[0]
            epochs = self._connection.execute(
                "SELECT COUNT(*) FROM memory_key_epochs "
                "WHERE principal_id=? AND state!='DESTROYED'",
                (principal_id,),
            ).fetchone()[0]
        ready = (
            state["storage_state"] == "ENCRYPTED"
            and self._memory_epoch_session is not None
        )
        return {
            "storage_state": state["storage_state"],
            "ready": ready,
            "memory_revision": int(state["memory_revision"]),
            "referenced_epoch_count": int(epochs),
            "pending_operation_count": int(pending),
        }

    def memory_crypto_readiness(self) -> dict[str, Any]:
        """Live, content-free provider/key/epoch verification for readiness."""

        if not self._memory_storage_enabled:
            document = {
                "contract": "jarvis.memory-crypto-readiness/1",
                "enabled": False,
                "storage_state": "DISABLED",
            }
            return {
                "ready": False,
                "code": "memory_disabled",
                "state_sha256": canonical_sha256(document),
            }
        try:
            runtime = self._memory_runtime_or_unavailable()
            with self._lock:
                principal_id = self._local_owner_principal_tx(self._connection)
                state = dict(
                    self._memory_crypto_state_tx(self._connection, principal_id)
                )
                if state["storage_state"] != "ENCRYPTED":
                    raise MemoryStorageUnavailable("memory_storage_not_prepared")
                epoch_row = self._connection.execute(
                    "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
                    (state["write_epoch_id"],),
                ).fetchone()
                if epoch_row is None:
                    raise MemoryStorageUnavailable("memory_key_epoch_missing")
                epoch = dict(epoch_row)
            session = self._open_qualified_epoch_session(runtime, state, epoch)
            try:
                attestation = runtime.cipher.provider.attest()
            except MemoryCryptoError as exc:
                raise MemoryStorageUnavailable(exc.code) from exc
            if (
                attestation.install_id != session.install_id
                or attestation.active_version != session.key_version
                or attestation.high_water < attestation.active_version
            ):
                raise MemoryStorageUnavailable("memory_key_epoch_mismatch")
            with self._lock:
                current_state = dict(
                    self._memory_crypto_state_tx(self._connection, principal_id)
                )
                current_epoch_row = self._connection.execute(
                    "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
                    (current_state["write_epoch_id"],),
                ).fetchone()
                if current_epoch_row is None:
                    raise MemoryStorageUnavailable("memory_key_epoch_missing")
                current_epoch = dict(current_epoch_row)
                self._require_session_epoch(session, current_state, current_epoch)
                self._memory_epoch_session = session
            document = {
                "confirmation_hmac_sha256": hashlib.sha256(
                    bytes(current_epoch["confirmation_hmac"])
                ).hexdigest(),
                "contract": "jarvis.memory-crypto-readiness/1",
                "epoch_id": str(current_epoch["epoch_id"]),
                "key_version": int(current_epoch["key_version"]),
                "provider_high_water": int(attestation.high_water),
                "provider_install_id": str(attestation.install_id),
                "storage_state": str(current_state["storage_state"]),
            }
            return {
                "ready": True,
                "code": "memory_crypto_ready",
                "state_sha256": canonical_sha256(document),
            }
        except MemoryStorageUnavailable as exc:
            return {
                "ready": False,
                "code": exc.code,
                "state_sha256": canonical_sha256(
                    {
                        "contract": "jarvis.memory-crypto-readiness/1",
                        "failure_code": exc.code,
                    }
                ),
            }

    @staticmethod
    def _memory_source_binding_sha256(row: Mapping[str, Any]) -> str:
        return canonical_sha256(
            {
                "content_bytes": int(row["content_bytes"]),
                "contract": "jarvis.memory-source-binding/1",
                "conversation_id": str(row["conversation_id"]),
                "created_at": str(row["created_at"]),
                "expires_at": str(row["expires_at"]),
                "normalizer_version": str(row["normalizer_version"]),
                "principal_id": str(row["principal_id"]),
                "record_id": str(row["record_id"]),
                "source_event_id": str(row["source_event_id"]),
                "task_id": str(row["task_id"]),
                "term_count": int(row["term_count"]),
            }
        )

    @staticmethod
    def _append_memory_record_audit_tx(
        connection: sqlite3.Connection,
        *,
        row: Mapping[str, Any],
        action: str,
        actor_kind: str,
        actor_id: str | None,
        outcome_code: str,
        now: datetime,
        operation_id: str | None = None,
        old_epoch_id: str | None = None,
        new_epoch_id: str | None = None,
    ) -> None:
        source_binding = OrchestratorStore._memory_source_binding_sha256(row)
        selected_operation_id = operation_id or actor_id or str(row["record_id"])
        require_uuid4(selected_operation_id)
        audit_id = new_uuid4()
        created_at = utc_text(now)
        document = {
            "action": action,
            "audit_id": audit_id,
            "actor_id": actor_id,
            "actor_kind": actor_kind,
            "contract": MEMORY_AUDIT_CONTRACT,
            "new_epoch_id": new_epoch_id,
            "old_epoch_id": old_epoch_id,
            "operation_id": selected_operation_id,
            "outcome_code": outcome_code,
            "principal_id": str(row["principal_id"]),
            "record_id": str(row["record_id"]),
            "source_binding_sha256": source_binding,
            "created_at": created_at,
        }
        connection.execute(
            """INSERT OR IGNORE INTO memory_record_audits(
               audit_id,principal_id,record_id,action,actor_kind,actor_id,
               source_binding_sha256,operation_id,old_epoch_id,new_epoch_id,
               outcome_code,created_at,audit_sha256
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                audit_id,
                row["principal_id"],
                row["record_id"],
                action,
                actor_kind,
                actor_id,
                source_binding,
                selected_operation_id,
                old_epoch_id,
                new_epoch_id,
                outcome_code,
                created_at,
                canonical_sha256(document),
            ),
        )

    @staticmethod
    def _require_memory_audit_row(row: Mapping[str, Any]) -> None:
        document = {
            "action": row["action"],
            "audit_id": row["audit_id"],
            "actor_id": row["actor_id"],
            "actor_kind": row["actor_kind"],
            "contract": MEMORY_AUDIT_CONTRACT,
            "new_epoch_id": row["new_epoch_id"],
            "old_epoch_id": row["old_epoch_id"],
            "operation_id": row["operation_id"],
            "outcome_code": row["outcome_code"],
            "principal_id": row["principal_id"],
            "record_id": row["record_id"],
            "source_binding_sha256": row["source_binding_sha256"],
            "created_at": row["created_at"],
        }
        if canonical_sha256(document) != row["audit_sha256"]:
            raise MemoryStorageUnavailable("memory_audit_binding_invalid")

    @staticmethod
    def _memory_binding_from_row(row: Mapping[str, Any]) -> MemoryRecordBindingV1:
        return MemoryRecordBindingV1(
            principal_id=str(row["principal_id"]),
            record_id=str(row["record_id"]),
            conversation_id=str(row["conversation_id"]),
            task_id=str(row["task_id"]),
            source_event_id=str(row["source_event_id"]),
            created_at=str(row["created_at"]),
            expires_at=str(row["expires_at"]),
            content_bytes=int(row["content_bytes"]),
            term_count=int(row["term_count"]),
            normalizer_version=str(row["normalizer_version"]),
            record_schema_version=int(row["record_schema_version"]),
        )

    @staticmethod
    def _memory_envelope_from_row(row: Mapping[str, Any]) -> MemoryEnvelopeV1:
        envelope = MemoryEnvelopeV1(
            key_version=int(row["key_version"]),
            payload_nonce=bytes(row["payload_nonce"]),
            payload_ciphertext_and_tag=bytes(row["payload_ciphertext"]),
            wrap_nonce=bytes(row["wrap_nonce"]),
            wrapped_dek_and_tag=bytes(row["wrapped_dek"]),
            binding_sha256=str(row["binding_sha256"]),
        )
        if hashlib.sha256(envelope.to_jcs_bytes()).hexdigest() != row[
            "envelope_sha256"
        ]:
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        return envelope

    @classmethod
    def _decrypt_memory_row(
        cls,
        session: MemoryEpochSessionV1,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        selected_row = dict(row)
        if (
            int(selected_row.get("record_schema_version", -1))
            != MEMORY_RECORD_SCHEMA_VERSION
            or int(selected_row.get("envelope_version", -1))
            != MEMORY_ENVELOPE_VERSION
            or int(selected_row.get("payload_version", -1)) != 1
        ):
            raise MemoryEnvelopeInvalid("memory_envelope_version_invalid")
        binding = cls._memory_binding_from_row(selected_row)
        envelope = cls._memory_envelope_from_row(selected_row)
        if envelope.key_version != int(row["key_version"]):
            raise MemoryEnvelopeInvalid("memory_envelope_invalid")
        plaintext = session.decrypt(envelope, binding)
        try:
            document = json.loads(plaintext.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MemoryEnvelopeInvalid("memory_payload_invalid") from exc
        if (
            not isinstance(document, dict)
            or set(document) != {"contract", "content", "metadata"}
            or document.get("contract") != MEMORY_PAYLOAD_CONTRACT
            or jcs_bytes(document) != plaintext
            or not isinstance(document.get("content"), str)
            or not isinstance(document.get("metadata"), dict)
        ):
            raise MemoryEnvelopeInvalid("memory_payload_invalid")
        content = document["content"]
        metadata = document["metadata"]
        fields = cls._memory_content_fields(content)
        expected_metadata_keys = {
            "conversation_id",
            "task_id",
            "task_epoch",
            "turn_id",
            "source_event_id",
            "language",
            "_anamnesis_namespace",
            "_anamnesis_user_id",
        }
        if (
            fields["content_bytes"] != int(row["content_bytes"])
            or fields["term_count"] != int(row["term_count"])
            or row["normalizer_version"] != MEMORY_NORMALIZER_VERSION
            or set(metadata) != expected_metadata_keys
            or metadata.get("conversation_id") != row["conversation_id"]
            or metadata.get("task_id") != row["task_id"]
            or metadata.get("source_event_id") != row["source_event_id"]
            or metadata.get("_anamnesis_namespace") != "jarvis"
            or metadata.get("_anamnesis_user_id") != row["conversation_id"]
        ):
            raise MemoryEnvelopeInvalid("memory_payload_binding_invalid")
        required_source_fields = {
            "source_task_epoch",
            "source_turn_id",
            "source_task_state",
            "source_task_conversation_id",
            "source_language",
            "source_turn_status",
            "source_turn_conversation_id",
            "source_turn_task_id",
            "source_canonical_turn_id",
            "source_inbox_task_id",
            "source_inbox_turn_id",
        }
        if not required_source_fields.issubset(selected_row) or (
            metadata.get("task_epoch") != str(selected_row["source_task_epoch"])
            or metadata.get("turn_id") != selected_row["source_turn_id"]
            or metadata.get("language") != selected_row["source_language"]
            or selected_row["source_task_state"] != TaskState.COMPLETED.value
            or selected_row["source_turn_status"] != "COMPLETED"
            or selected_row["source_task_conversation_id"]
            != selected_row["conversation_id"]
            or selected_row["source_turn_conversation_id"]
            != selected_row["conversation_id"]
            or selected_row["source_turn_task_id"] != selected_row["task_id"]
            or selected_row["source_canonical_turn_id"]
            != selected_row["source_turn_id"]
            or selected_row["source_inbox_task_id"] != selected_row["task_id"]
            or selected_row["source_inbox_turn_id"]
            != selected_row["source_turn_id"]
        ):
            raise MemoryEnvelopeInvalid("memory_payload_binding_invalid")
        tokens = session.blind_tokens(
            content=content,
            normalized_content=normalize_explicit_memory_target(content),
            terms=fields["terms"],
        )
        if (
            tokens.content_hmac != bytes(row["content_hmac"])
            or tokens.normalized_hmac != bytes(row["normalized_hmac"])
        ):
            raise MemoryEnvelopeInvalid("memory_index_binding_invalid")
        return {
            "record_id": str(row["record_id"]),
            "principal_id": str(row["principal_id"]),
            "conversation_id": str(row["conversation_id"]),
            "task_id": str(row["task_id"]),
            "source_event_id": str(row["source_event_id"]),
            "content": content,
            "content_bytes": int(row["content_bytes"]),
            "term_count": int(row["term_count"]),
            "expires_at": str(row["expires_at"]),
            "normalizer_version": str(row["normalizer_version"]),
            "created_at": str(row["created_at"]),
            "metadata": dict(metadata),
            "metadata_json": canonical_json(metadata),
            "terms": fields["terms"],
            "key_version": int(row["key_version"]),
            "key_epoch_id": str(row["key_epoch_id"]),
            "envelope_sha256": str(row["envelope_sha256"]),
        }

    @classmethod
    def _project_verified_memory_row_tx(
        cls,
        connection: sqlite3.Connection,
        session: MemoryEpochSessionV1,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        projected = cls._decrypt_memory_row(session, row)
        observed = tuple(
            bytes(item["term_hmac"])
            for item in connection.execute(
                "SELECT term_hmac FROM memory_terms "
                "INDEXED BY memory_terms_by_record WHERE record_id=? "
                "ORDER BY term_hmac",
                (row["record_id"],),
            ).fetchall()
        )
        expected = tuple(
            sorted(
                session.blind_tokens(
                    content=projected["content"],
                    normalized_content=normalize_explicit_memory_target(
                        projected["content"]
                    ),
                    terms=projected["terms"],
                ).term_hmacs
            )
        )
        if observed != expected:
            raise MemoryStorageUnavailable("memory_term_index_binding_invalid")
        return projected

    @staticmethod
    def _memory_row_select_sql(*, where: str) -> str:
        return f"""SELECT r.*,t.epoch AS source_task_epoch,
                          t.turn_id AS source_turn_id,
                          t.state AS source_task_state,
                          t.conversation_id AS source_task_conversation_id,
                          ct.user_language AS source_language,
                          ct.status AS source_turn_status,
                          ct.conversation_id AS source_turn_conversation_id,
                          ct.task_id AS source_turn_task_id,
                          ct.turn_id AS source_canonical_turn_id,
                          i.task_id AS source_inbox_task_id,
                          i.turn_id AS source_inbox_turn_id
                   FROM memory_records r
                   JOIN tasks t ON t.task_id=r.task_id
                   JOIN canonical_turns ct ON ct.task_id=r.task_id
                   JOIN inbox i ON i.event_id=r.source_event_id
                   WHERE {where}"""

    @classmethod
    def _legacy_memory_rows_for_migration(
        cls,
        connection: sqlite3.Connection,
        *,
        principal_id: str,
        cutoff: datetime,
    ) -> tuple[list[dict[str, Any]], int]:
        rows = connection.execute(
            """SELECT r.*,t.epoch AS source_task_epoch,
                      t.turn_id AS source_turn_id,t.state AS source_task_state,
                      t.conversation_id AS source_task_conversation_id,
                      ct.user_language AS source_language,
                      ct.status AS source_turn_status,
                      ct.conversation_id AS source_turn_conversation_id,
                      ct.task_id AS source_turn_task_id,
                      ct.turn_id AS source_canonical_turn_id,
                      i.task_id AS source_inbox_task_id,
                      i.turn_id AS source_inbox_turn_id
               FROM memory_records_v9_legacy r
               LEFT JOIN tasks t ON t.task_id=r.task_id
               LEFT JOIN canonical_turns ct ON ct.task_id=r.task_id
               LEFT JOIN inbox i ON i.event_id=r.source_event_id
               WHERE r.principal_id=? ORDER BY r.created_at,r.record_id""",
            (principal_id,),
        ).fetchall()
        prepared: list[dict[str, Any]] = []
        expired = 0
        for raw in rows:
            row = dict(raw)
            try:
                fields = cls._memory_content_fields(row["content"])
                metadata = json.loads(row["metadata_json"])
                created = parse_utc(row["created_at"])
                expires = parse_utc(row["expires_at"])
                expected_terms = {
                    _memory_term_sha256(term) for term in fields["terms"]
                }
                observed_terms = {
                    item["term_sha256"]
                    for item in connection.execute(
                        "SELECT term_sha256 FROM memory_terms_v9_legacy "
                        "WHERE principal_id=? AND record_id=?",
                        (principal_id, row["record_id"]),
                    ).fetchall()
                }
                if (
                    row["principal_id"] != principal_id
                    or row["content_sha256"] != fields["content_sha256"]
                    or row["normalized_sha256"] != fields["normalized_sha256"]
                    or int(row["content_bytes"]) != fields["content_bytes"]
                    or int(row["term_count"]) != fields["term_count"]
                    or row["normalizer_version"] != MEMORY_NORMALIZER_VERSION
                    or observed_terms != expected_terms
                    or utc_text(created) != row["created_at"]
                    or utc_text(expires) != row["expires_at"]
                    or expires
                    != created + timedelta(days=MEMORY_RECORD_TTL_DAYS)
                    or not isinstance(metadata, dict)
                    or canonical_json(metadata) != row["metadata_json"]
                    or set(metadata)
                    != {
                        "conversation_id",
                        "task_id",
                        "task_epoch",
                        "turn_id",
                        "source_event_id",
                        "language",
                        "_anamnesis_namespace",
                        "_anamnesis_user_id",
                    }
                    or metadata.get("conversation_id") != row["conversation_id"]
                    or metadata.get("task_id") != row["task_id"]
                    or metadata.get("task_epoch")
                    != str(row["source_task_epoch"])
                    or metadata.get("turn_id") != row["source_turn_id"]
                    or metadata.get("source_event_id") != row["source_event_id"]
                    or metadata.get("language") != row["source_language"]
                    or metadata.get("_anamnesis_namespace") != "jarvis"
                    or metadata.get("_anamnesis_user_id")
                    != row["conversation_id"]
                    or row["source_task_state"] != TaskState.COMPLETED.value
                    or row["source_turn_status"] != "COMPLETED"
                    or row["source_task_conversation_id"]
                    != row["conversation_id"]
                    or row["source_turn_conversation_id"]
                    != row["conversation_id"]
                    or row["source_turn_task_id"] != row["task_id"]
                    or row["source_canonical_turn_id"] != row["source_turn_id"]
                    or row["source_inbox_task_id"] != row["task_id"]
                    or row["source_inbox_turn_id"] != row["source_turn_id"]
                ):
                    raise ValueError("legacy memory binding is invalid")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise MemoryStorageUnavailable(
                    "memory_legacy_record_invalid"
                ) from exc
            row.update(fields)
            row["metadata"] = metadata
            row["term_sha256s"] = tuple(sorted(observed_terms))
            row["expired_at_cutoff"] = expires <= cutoff
            expired += int(row["expired_at_cutoff"])
            prepared.append(row)
        active = [row for row in prepared if not row["expired_at_cutoff"]]
        if (
            len(active) > MAX_MEMORY_PRINCIPAL_RECORDS
            or sum(int(row["content_bytes"]) for row in active)
            > MAX_MEMORY_PRINCIPAL_BYTES
        ):
            raise MemoryStorageUnavailable("memory_legacy_quota_invalid")
        return prepared, expired

    @staticmethod
    def _legacy_memory_source_digest(
        *,
        operation_id: str,
        principal_id: str,
        cutoff: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> str:
        """Bind one migration operation to the exact validated legacy source."""

        return canonical_sha256(
            {
                "contract": MEMORY_CRYPTO_OPERATION_CONTRACT,
                "cutoff": cutoff,
                "kind": "MIGRATE_V9",
                "operation_id": operation_id,
                "principal_id": principal_id,
                "records": [
                    {
                        key: row[key]
                        for key in (
                            "record_id",
                            "principal_id",
                            "conversation_id",
                            "task_id",
                            "source_event_id",
                            "content",
                            "content_sha256",
                            "normalized_sha256",
                            "content_bytes",
                            "term_count",
                            "expires_at",
                            "normalizer_version",
                            "metadata_json",
                            "created_at",
                            "source_task_epoch",
                            "source_turn_id",
                            "source_task_state",
                            "source_task_conversation_id",
                            "source_language",
                            "source_turn_status",
                            "source_turn_conversation_id",
                            "source_turn_task_id",
                            "source_canonical_turn_id",
                            "source_inbox_task_id",
                            "source_inbox_turn_id",
                            "term_sha256s",
                            "expired_at_cutoff",
                        )
                    }
                    for row in rows
                ],
                "source_schema": 9,
                "target_schema": 10,
            }
        )

    @staticmethod
    def _require_encrypted_source_equivalence(
        *,
        encrypted: Mapping[str, Any],
        projected: Mapping[str, Any],
        source: Mapping[str, Any],
        epoch: Mapping[str, Any],
    ) -> None:
        """Fail closed unless one ciphertext is the exact legacy projection."""

        source_fields = OrchestratorStore._memory_content_fields(
            str(source["content"])
        )
        if (
            projected["record_id"] != source["record_id"]
            or projected["principal_id"] != source["principal_id"]
            or projected["conversation_id"] != source["conversation_id"]
            or projected["task_id"] != source["task_id"]
            or projected["source_event_id"] != source["source_event_id"]
            or projected["content"] != source["content"]
            or projected["metadata"] != source["metadata"]
            or projected["content_bytes"] != int(source["content_bytes"])
            or projected["term_count"] != int(source["term_count"])
            or projected["expires_at"] != source["expires_at"]
            or projected["normalizer_version"] != source["normalizer_version"]
            or projected["created_at"] != source["created_at"]
            or projected["terms"] != source_fields["terms"]
            or tuple(sorted(source["term_sha256s"]))
            != tuple(
                sorted(_memory_term_sha256(term) for term in projected["terms"])
            )
            or encrypted["key_epoch_id"] != epoch["epoch_id"]
            or int(encrypted["key_version"]) != int(epoch["key_version"])
        ):
            raise MemoryStorageUnavailable("memory_migration_target_mismatch")

    @classmethod
    def _encrypted_record_values(
        cls,
        session: MemoryEpochSessionV1,
        *,
        principal_id: str,
        epoch_id: str,
        key_version: int,
        source: Mapping[str, Any],
    ) -> dict[str, Any]:
        binding = MemoryRecordBindingV1(
            principal_id=principal_id,
            record_id=str(source["record_id"]),
            conversation_id=str(source["conversation_id"]),
            task_id=str(source["task_id"]),
            source_event_id=str(source["source_event_id"]),
            created_at=str(source["created_at"]),
            expires_at=str(source["expires_at"]),
            content_bytes=int(source["content_bytes"]),
            term_count=int(source["term_count"]),
            normalizer_version=MEMORY_NORMALIZER_VERSION,
            record_schema_version=MEMORY_RECORD_SCHEMA_VERSION,
        )
        payload = jcs_bytes(
            {
                "contract": MEMORY_PAYLOAD_CONTRACT,
                "content": source["content"],
                "metadata": source["metadata"],
            }
        )
        if (
            session.principal_id != principal_id
            or session.key_version != key_version
        ):
            raise MemoryStorageUnavailable("memory_key_epoch_mismatch")
        envelope = session.encrypt(payload, binding)
        tokens = session.blind_tokens(
            content=str(source["content"]),
            normalized_content=normalize_explicit_memory_target(
                str(source["content"])
            ),
            terms=tuple(source["terms"]),
        )
        return {
            "record_id": source["record_id"],
            "principal_id": principal_id,
            "conversation_id": source["conversation_id"],
            "task_id": source["task_id"],
            "source_event_id": source["source_event_id"],
            "key_epoch_id": epoch_id,
            "key_version": key_version,
            "record_schema_version": MEMORY_RECORD_SCHEMA_VERSION,
            "envelope_version": MEMORY_ENVELOPE_VERSION,
            "payload_version": 1,
            "payload_nonce": envelope.payload_nonce,
            "payload_ciphertext": envelope.payload_ciphertext_and_tag,
            "wrap_nonce": envelope.wrap_nonce,
            "wrapped_dek": envelope.wrapped_dek_and_tag,
            "binding_sha256": envelope.binding_sha256,
            "envelope_sha256": hashlib.sha256(
                envelope.to_jcs_bytes()
            ).hexdigest(),
            "content_hmac": tokens.content_hmac,
            "normalized_hmac": tokens.normalized_hmac,
            "content_bytes": source["content_bytes"],
            "term_count": source["term_count"],
            "expires_at": source["expires_at"],
            "normalizer_version": MEMORY_NORMALIZER_VERSION,
            "created_at": source["created_at"],
            "term_hmacs": tokens.term_hmacs,
        }

    @staticmethod
    def _insert_encrypted_memory_row_tx(
        connection: sqlite3.Connection,
        values: Mapping[str, Any],
    ) -> None:
        connection.execute(
            """INSERT INTO memory_records(
               record_id,principal_id,conversation_id,task_id,source_event_id,
               key_epoch_id,key_version,record_schema_version,envelope_version,
               payload_version,payload_nonce,payload_ciphertext,wrap_nonce,
               wrapped_dek,binding_sha256,envelope_sha256,content_hmac,
               normalized_hmac,content_bytes,term_count,expires_at,
               normalizer_version,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            tuple(
                values[key]
                for key in (
                    "record_id",
                    "principal_id",
                    "conversation_id",
                    "task_id",
                    "source_event_id",
                    "key_epoch_id",
                    "key_version",
                    "record_schema_version",
                    "envelope_version",
                    "payload_version",
                    "payload_nonce",
                    "payload_ciphertext",
                    "wrap_nonce",
                    "wrapped_dek",
                    "binding_sha256",
                    "envelope_sha256",
                    "content_hmac",
                    "normalized_hmac",
                    "content_bytes",
                    "term_count",
                    "expires_at",
                    "normalizer_version",
                    "created_at",
                )
            ),
        )
        for token in values["term_hmacs"]:
            connection.execute(
                """INSERT INTO memory_terms(
                   principal_id,key_epoch_id,key_version,term_hmac,record_id
                   ) VALUES(?,?,?,?,?)""",
                (
                    values["principal_id"],
                    values["key_epoch_id"],
                    values["key_version"],
                    token,
                    values["record_id"],
                ),
            )

    @staticmethod
    def _require_session_epoch(
        session: MemoryEpochSessionV1,
        state: Mapping[str, Any],
        epoch: Mapping[str, Any],
        *,
        allowed_epoch_states: frozenset[str] = frozenset({"WRITE_ACTIVE"}),
    ) -> None:
        if (
            state["provider_install_id"] != session.install_id
            or epoch["provider_install_id"] != session.install_id
            or str(epoch["principal_id"]) != session.principal_id
            or state["write_epoch_id"] != epoch["epoch_id"]
            or str(epoch["state"]) not in allowed_epoch_states
            or session.key_version != int(epoch["key_version"])
        ):
            raise MemoryStorageUnavailable("memory_key_epoch_mismatch")
        confirmation = session.blind_token(
            kind="content",
            value=str(epoch["confirmation_challenge"]),
        )
        if confirmation != bytes(epoch["confirmation_hmac"]):
            raise MemoryStorageUnavailable("memory_key_confirmation_failed")

    @classmethod
    def _open_qualified_epoch_session(
        cls,
        runtime: MemoryCryptoRuntime,
        state: Mapping[str, Any],
        epoch: Mapping[str, Any],
        *,
        allowed_epoch_states: frozenset[str] = frozenset({"WRITE_ACTIVE"}),
    ) -> MemoryEpochSessionV1:
        """Load/attest a provider key before any store lock or transaction."""

        try:
            install_id = runtime.install_id
        except MemoryCryptoError as exc:
            raise MemoryStorageUnavailable(exc.code) from exc
        if (
            not isinstance(install_id, str)
            or len(install_id) != 64
            or install_id != install_id.lower()
            or any(character not in "0123456789abcdef" for character in install_id)
        ):
            raise MemoryStorageUnavailable("memory_key_attestation_invalid")
        if (
            state["provider_install_id"] != install_id
            or epoch["provider_install_id"] != install_id
            or state["write_epoch_id"] != epoch["epoch_id"]
            or str(epoch["state"]) not in allowed_epoch_states
        ):
            raise MemoryStorageUnavailable("memory_key_epoch_mismatch")
        try:
            key_version = int(epoch["key_version"])
            if runtime.cipher.active_version() != key_version:
                raise MemoryStorageUnavailable("memory_key_epoch_mismatch")
            session = runtime.cipher.open_epoch_session(
                principal_id=str(epoch["principal_id"]),
                key_version=key_version,
            )
        except MemoryCryptoError as exc:
            raise MemoryStorageUnavailable(exc.code) from exc
        cls._require_session_epoch(
            session,
            state,
            epoch,
            allowed_epoch_states=allowed_epoch_states,
        )
        return session

    def prepare_memory_storage(
        self,
        runtime: MemoryCryptoRuntime | None = None,
        *,
        now: datetime | None = None,
        _crash_hook: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Explicitly migrate v9 plaintext into one verified encrypted epoch.

        This is the only method allowed to bootstrap a key.  It is resumable
        from every committed phase and is intended to run before serving.
        """

        if not self._memory_storage_enabled:
            raise MemoryStorageUnavailable("memory_disabled")

        selected_runtime = self._validated_memory_runtime(runtime)
        if selected_runtime is not None:
            if (
                self._memory_crypto is not None
                and self._memory_crypto is not selected_runtime
                and self._memory_crypto.install_id != selected_runtime.install_id
            ):
                raise MemoryStorageUnavailable("memory_provider_binding_changed")
            self._memory_crypto = selected_runtime
        active_runtime = self._memory_runtime_or_unavailable()
        selected = now or utc_now()
        hook = _crash_hook or (lambda _phase: None)

        with self._lock:
            principal_id = self._local_owner_principal_tx(self._connection)
            state = dict(
                self._memory_crypto_state_tx(self._connection, principal_id)
            )
            if state["storage_state"] == "ENCRYPTED":
                epoch_row = self._connection.execute(
                    "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
                    (state["write_epoch_id"],),
                ).fetchone()
                if epoch_row is None:
                    raise MemoryStorageUnavailable("memory_key_epoch_missing")
                epoch = dict(epoch_row)
            else:
                epoch = None
        if epoch is not None:
            session = self._open_qualified_epoch_session(
                active_runtime,
                state,
                epoch,
            )
            with self._transaction() as connection:
                current_state = dict(
                    self._memory_crypto_state_tx(connection, principal_id)
                )
                current_epoch_row = connection.execute(
                    "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
                    (current_state["write_epoch_id"],),
                ).fetchone()
                if current_epoch_row is None:
                    raise MemoryStorageUnavailable("memory_key_epoch_missing")
                self._require_session_epoch(
                    session,
                    current_state,
                    dict(current_epoch_row),
                )
            self._memory_epoch_session = session
            self.sweep_expired_memory(now=selected, limit=MEMORY_SWEEP_LIMIT)
            return self.memory_storage_status()

        with self._transaction() as connection:
            state = self._memory_crypto_state_tx(connection, principal_id)
            operation = connection.execute(
                "SELECT * FROM memory_crypto_operations "
                "WHERE principal_id=? AND kind='MIGRATE_V9' "
                "ORDER BY created_at DESC LIMIT 1",
                (principal_id,),
            ).fetchone()
            if operation is None:
                if state["storage_state"] != "LEGACY_PENDING":
                    raise MemoryStorageUnavailable("memory_migration_state_invalid")
                operation_id = new_uuid4()
                stamp = utc_text(selected)
                legacy_rows, _expired_count = self._legacy_memory_rows_for_migration(
                    connection,
                    principal_id=principal_id,
                    cutoff=parse_utc(stamp),
                )
                request_sha256 = self._legacy_memory_source_digest(
                    operation_id=operation_id,
                    principal_id=principal_id,
                    cutoff=stamp,
                    rows=legacy_rows,
                )
                connection.execute(
                    """INSERT INTO memory_crypto_operations(
                       operation_id,principal_id,kind,phase,source_epoch_id,
                       target_epoch_id,expected_provider_version,
                       cursor_created_at,cursor_record_id,expected_record_count,
                       processed_record_count,request_sha256,last_error_code,
                       created_at,updated_at,completed_at
                       ) VALUES(?,?,'MIGRATE_V9','PREPARED',NULL,NULL,NULL,
                                NULL,NULL,?,0,?,NULL,?,?,NULL)""",
                    (
                        operation_id,
                        principal_id,
                        len(legacy_rows),
                        request_sha256,
                        stamp,
                        stamp,
                    ),
                )
                operation = connection.execute(
                    "SELECT * FROM memory_crypto_operations WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()
            if operation is None:
                raise MemoryStorageUnavailable("memory_migration_state_invalid")
            frozen_cutoff = str(operation["created_at"])
            try:
                frozen_cutoff_value = parse_utc(frozen_cutoff)
            except (TypeError, ValueError) as exc:
                raise MemoryStorageUnavailable(
                    "memory_migration_state_invalid"
                ) from exc
            legacy_rows, _expired_count = self._legacy_memory_rows_for_migration(
                connection,
                principal_id=principal_id,
                cutoff=frozen_cutoff_value,
            )
            observed_request_sha256 = self._legacy_memory_source_digest(
                operation_id=str(operation["operation_id"]),
                principal_id=principal_id,
                cutoff=frozen_cutoff,
                rows=legacy_rows,
            )
            if (
                int(operation["expected_record_count"]) != len(legacy_rows)
                or operation["request_sha256"] != observed_request_sha256
            ):
                raise MemoryStorageUnavailable("memory_migration_source_changed")
            if operation["phase"] == "UNCERTAIN":
                raise MemoryStorageUnavailable("memory_migration_uncertain")
            operation_id = str(operation["operation_id"])
        hook("operation_prepared")

        with self._lock:
            operation = self._connection.execute(
                "SELECT * FROM memory_crypto_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            assert operation is not None
            phase = str(operation["phase"])
        if phase == "PREPARED":
            try:
                key_version = active_runtime.cipher.bootstrap_epoch()
                migration_session = active_runtime.cipher.open_epoch_session(
                    principal_id=principal_id,
                    key_version=key_version,
                )
                if (
                    migration_session.install_id != active_runtime.install_id
                    or migration_session.principal_id != principal_id
                    or migration_session.key_version != key_version
                ):
                    raise MemoryStorageUnavailable("memory_key_epoch_mismatch")
                confirmation = migration_session.blind_token(
                    kind="content",
                    value=MEMORY_PROVIDER_CONFIRMATION_TEXT,
                )
            except MemoryCryptoError as exc:
                raise MemoryStorageUnavailable(exc.code) from exc
            epoch_id = new_uuid4()
            with self._transaction() as connection:
                current = connection.execute(
                    "SELECT * FROM memory_crypto_operations WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()
                if current is None or current["phase"] != "PREPARED":
                    raise MemoryStorageUnavailable("memory_migration_state_changed")
                stamp = utc_text(selected)
                connection.execute(
                    """INSERT INTO memory_key_epochs(
                       epoch_id,principal_id,provider_install_id,key_version,state,
                       confirmation_challenge,confirmation_hmac,created_at,
                       provider_verified_at,activated_at,retired_at,
                       destroy_requested_at,destroy_verified_at
                       ) VALUES(?,?,?,?,'PROVISIONING',?,?,?, ?,NULL,NULL,NULL,NULL)""",
                    (
                        epoch_id,
                        principal_id,
                        active_runtime.install_id,
                        key_version,
                        MEMORY_PROVIDER_CONFIRMATION_TEXT,
                        confirmation,
                        stamp,
                        stamp,
                    ),
                )
                connection.execute(
                    """UPDATE memory_crypto_state
                       SET storage_state='MIGRATING',provider_install_id=?,
                           write_epoch_id=?,active_operation_id=?,updated_at=?
                       WHERE principal_id=? AND storage_state='LEGACY_PENDING'""",
                    (
                        active_runtime.install_id,
                        epoch_id,
                        operation_id,
                        stamp,
                        principal_id,
                    ),
                )
                connection.execute(
                    """UPDATE memory_crypto_operations
                       SET phase='PROVIDER_READY',target_epoch_id=?,
                           expected_provider_version=?,updated_at=?
                       WHERE operation_id=? AND phase='PREPARED'""",
                    (epoch_id, key_version, stamp, operation_id),
                )
            hook("provider_ready")

        with self._lock:
            operation_row = self._connection.execute(
                "SELECT * FROM memory_crypto_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            assert operation_row is not None
            operation = dict(operation_row)
            state = dict(
                self._memory_crypto_state_tx(self._connection, principal_id)
            )
            epoch_row = self._connection.execute(
                "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
                (operation["target_epoch_id"],),
            ).fetchone()
            if epoch_row is None:
                raise MemoryStorageUnavailable("memory_key_epoch_missing")
            epoch = dict(epoch_row)
        migration_session = self._open_qualified_epoch_session(
            active_runtime,
            state,
            epoch,
            allowed_epoch_states=frozenset({"PROVISIONING", "WRITE_ACTIVE"}),
        )

        if operation["phase"] in {"PROVIDER_READY", "COPYING"}:
            with self._transaction() as connection:
                connection.execute(
                    "UPDATE memory_crypto_operations SET phase='COPYING',updated_at=? "
                    "WHERE operation_id=? AND phase IN ('PROVIDER_READY','COPYING')",
                    (utc_text(selected), operation_id),
                )
            cursor = (
                operation["cursor_created_at"],
                operation["cursor_record_id"],
            )
            for source in legacy_rows:
                identity = (source["created_at"], source["record_id"])
                if cursor[0] is not None and identity <= cursor:
                    continue
                values = None
                if not source["expired_at_cutoff"]:
                    try:
                        values = self._encrypted_record_values(
                            migration_session,
                            principal_id=principal_id,
                            epoch_id=str(epoch["epoch_id"]),
                            key_version=int(epoch["key_version"]),
                            source=source,
                        )
                    except MemoryCryptoError as exc:
                        raise MemoryStorageUnavailable(exc.code) from exc
                with self._transaction() as connection:
                    current_source = connection.execute(
                        "SELECT * FROM memory_records_v9_legacy WHERE record_id=?",
                        (source["record_id"],),
                    ).fetchone()
                    if current_source is None or any(
                        current_source[key] != source[key]
                        for key in (
                            "principal_id",
                            "conversation_id",
                            "task_id",
                            "source_event_id",
                            "content",
                            "content_sha256",
                            "normalized_sha256",
                            "content_bytes",
                            "term_count",
                            "expires_at",
                            "normalizer_version",
                            "metadata_json",
                            "created_at",
                        )
                    ):
                        raise MemoryStorageUnavailable(
                            "memory_migration_source_changed"
                        )
                    existing_target = connection.execute(
                        "SELECT record_id FROM memory_records WHERE record_id=?",
                        (source["record_id"],),
                    ).fetchone()
                    if values is not None and existing_target is None:
                        self._insert_encrypted_memory_row_tx(connection, values)
                        self._append_memory_record_audit_tx(
                            connection,
                            row=values,
                            action="MIGRATED",
                            actor_kind="migration",
                            actor_id=operation_id,
                            operation_id=operation_id,
                            old_epoch_id=None,
                            new_epoch_id=str(epoch["epoch_id"]),
                            outcome_code="encrypted",
                            now=selected,
                        )
                    elif values is None:
                        if existing_target is not None:
                            raise MemoryStorageUnavailable(
                                "memory_migration_expired_target_present"
                            )
                        self._record_memory_tombstone_tx(
                            connection,
                            source,
                            reason="ttl_expiry",
                            action="TTL_EXPIRED",
                            actor_kind="migration",
                            actor_id=operation_id,
                            operation_id=operation_id,
                            now=selected,
                            revision_increment=0,
                        )
                    connection.execute(
                        """UPDATE memory_crypto_operations
                           SET cursor_created_at=?,cursor_record_id=?,
                               processed_record_count=processed_record_count+1,
                               updated_at=?
                           WHERE operation_id=? AND phase='COPYING'""",
                        (
                            source["created_at"],
                            source["record_id"],
                            utc_text(selected),
                            operation_id,
                        ),
                    )
                hook("record_copied")
            with self._transaction() as connection:
                changed = connection.execute(
                    """UPDATE memory_crypto_operations
                       SET phase='VERIFYING',updated_at=?
                       WHERE operation_id=? AND phase='COPYING'
                         AND processed_record_count=expected_record_count""",
                    (utc_text(selected), operation_id),
                )
                if changed.rowcount != 1:
                    raise MemoryStorageUnavailable(
                        "memory_migration_copy_incomplete"
                    )

        with self._lock:
            operation = dict(
                self._connection.execute(
                    "SELECT * FROM memory_crypto_operations WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()
            )
            encrypted_rows = [
                dict(row)
                for row in self._connection.execute(
                    self._memory_row_select_sql(where="r.principal_id=?")
                    + " ORDER BY r.created_at,r.record_id",
                    (principal_id,),
                ).fetchall()
            ]
        if operation["phase"] == "VERIFYING":
            expected_active = [
                row for row in legacy_rows if not row["expired_at_cutoff"]
            ]
            if len(encrypted_rows) != len(expected_active):
                raise MemoryStorageUnavailable("memory_migration_count_mismatch")
            expected_ids = {row["record_id"] for row in expected_active}
            if {row["record_id"] for row in encrypted_rows} != expected_ids:
                raise MemoryStorageUnavailable("memory_migration_identity_mismatch")
            expected_by_id = {
                str(source["record_id"]): source for source in expected_active
            }
            for encrypted in encrypted_rows:
                projected = self._decrypt_memory_row(migration_session, encrypted)
                self._require_encrypted_source_equivalence(
                    encrypted=encrypted,
                    projected=projected,
                    source=expected_by_id[str(encrypted["record_id"])],
                    epoch=epoch,
                )
                with self._lock:
                    term_rows = self._connection.execute(
                        "SELECT term_hmac FROM memory_terms WHERE record_id=? "
                        "ORDER BY term_hmac",
                        (encrypted["record_id"],),
                    ).fetchall()
                expected_tokens = migration_session.blind_tokens(
                    content=projected["content"],
                    normalized_content=normalize_explicit_memory_target(
                        projected["content"]
                    ),
                    terms=projected["terms"],
                ).term_hmacs
                if [bytes(item["term_hmac"]) for item in term_rows] != sorted(
                    expected_tokens
                ):
                    raise MemoryStorageUnavailable(
                        "memory_migration_term_index_mismatch"
                    )
            with self._transaction() as connection:
                connection.execute(
                    "UPDATE memory_crypto_operations SET phase='CUTOVER_READY',"
                    "updated_at=? WHERE operation_id=? AND phase='VERIFYING'",
                    (utc_text(selected), operation_id),
                )
            hook("cutover_ready")

        with self._transaction() as connection:
            operation = connection.execute(
                "SELECT * FROM memory_crypto_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise MemoryStorageUnavailable("memory_migration_state_invalid")
            if operation["phase"] == "CUTOVER_READY":
                final_cutoff = str(operation["created_at"])
                try:
                    final_cutoff_value = parse_utc(final_cutoff)
                except (TypeError, ValueError) as exc:
                    raise MemoryStorageUnavailable(
                        "memory_migration_state_invalid"
                    ) from exc
                final_legacy_rows, final_expired_count = (
                    self._legacy_memory_rows_for_migration(
                        connection,
                        principal_id=principal_id,
                        cutoff=final_cutoff_value,
                    )
                )
                if (
                    len(final_legacy_rows)
                    != int(operation["expected_record_count"])
                    or self._legacy_memory_source_digest(
                        operation_id=operation_id,
                        principal_id=principal_id,
                        cutoff=final_cutoff,
                        rows=final_legacy_rows,
                    )
                    != operation["request_sha256"]
                ):
                    raise MemoryStorageUnavailable(
                        "memory_migration_source_changed"
                    )
                final_active_rows = {
                    str(item["record_id"]): item
                    for item in final_legacy_rows
                    if not item["expired_at_cutoff"]
                }
                final_encrypted_rows = [
                    dict(item)
                    for item in connection.execute(
                        self._memory_row_select_sql(where="r.principal_id=?")
                        + " ORDER BY r.created_at,r.record_id",
                        (principal_id,),
                    ).fetchall()
                ]
                if {
                    str(item["record_id"]) for item in final_encrypted_rows
                } != set(final_active_rows):
                    raise MemoryStorageUnavailable(
                        "memory_migration_identity_mismatch"
                    )
                for encrypted_row in final_encrypted_rows:
                    projected = self._project_verified_memory_row_tx(
                        connection,
                        migration_session,
                        encrypted_row,
                    )
                    source = final_active_rows[str(encrypted_row["record_id"])]
                    self._require_encrypted_source_equivalence(
                        encrypted=encrypted_row,
                        projected=projected,
                        source=source,
                        epoch=epoch,
                    )
                    audits = connection.execute(
                        "SELECT * FROM memory_record_audits "
                        "WHERE record_id=? AND action='MIGRATED' "
                        "AND operation_id=?",
                        (encrypted_row["record_id"], operation_id),
                    ).fetchall()
                    if (
                        len(audits) != 1
                        or audits[0]["source_binding_sha256"]
                        != self._memory_source_binding_sha256(source)
                        or audits[0]["actor_kind"] != "migration"
                        or audits[0]["actor_id"] != operation_id
                        or audits[0]["new_epoch_id"] != epoch["epoch_id"]
                    ):
                        raise MemoryStorageUnavailable(
                            "memory_migration_audit_invalid"
                        )
                    self._require_memory_audit_row(dict(audits[0]))
                for legacy_row in final_legacy_rows:
                    if not legacy_row["expired_at_cutoff"]:
                        continue
                    source_binding = self._memory_source_binding_sha256(
                        legacy_row
                    )
                    tombstone = connection.execute(
                        "SELECT * FROM memory_record_tombstones WHERE record_id=?",
                        (legacy_row["record_id"],),
                    ).fetchone()
                    audits = connection.execute(
                        "SELECT * FROM memory_record_audits "
                        "WHERE record_id=? AND action='TTL_EXPIRED' "
                        "AND operation_id=?",
                        (legacy_row["record_id"], operation_id),
                    ).fetchall()
                    if (
                        tombstone is None
                        or tombstone["principal_id"] != principal_id
                        or tombstone["source_binding_sha256"] != source_binding
                        or tombstone["reason"] != "ttl_expiry"
                        or tombstone["delete_request_id"] is not None
                        or len(audits) != 1
                        or audits[0]["source_binding_sha256"] != source_binding
                        or audits[0]["actor_kind"] != "migration"
                        or audits[0]["actor_id"] != operation_id
                    ):
                        raise MemoryStorageUnavailable(
                            "memory_migration_expiry_audit_invalid"
                        )
                    self._require_memory_audit_row(dict(audits[0]))
                count_row = connection.execute(
                    "SELECT COUNT(*) AS count,COALESCE(SUM(content_bytes),0) AS bytes "
                    "FROM memory_records WHERE principal_id=?",
                    (principal_id,),
                ).fetchone()
                active_count = int(count_row["count"])
                active_bytes = int(count_row["bytes"])
                connection.execute("DROP TABLE memory_terms_v9_legacy")
                connection.execute("DROP TABLE memory_records_v9_legacy")
                stamp = utc_text(selected)
                connection.execute(
                    """UPDATE memory_principal_usage
                       SET active_record_count=?,active_content_bytes=?,updated_at=?
                       WHERE principal_id=?""",
                    (active_count, active_bytes, stamp, principal_id),
                )
                connection.execute(
                    """UPDATE memory_key_epochs
                       SET state='WRITE_ACTIVE',activated_at=?
                       WHERE epoch_id=? AND state='PROVISIONING'""",
                    (stamp, epoch["epoch_id"]),
                )
                connection.execute(
                    """UPDATE memory_crypto_state
                       SET storage_state='ENCRYPTED',active_operation_id=NULL,
                           memory_revision=0,updated_at=?
                       WHERE principal_id=? AND storage_state='MIGRATING'
                         AND write_epoch_id=?""",
                    (stamp, principal_id, epoch["epoch_id"]),
                )
                connection.execute(
                    """UPDATE memory_crypto_operations
                       SET phase='COMPLETE',request_sha256=?,updated_at=?,completed_at=?
                       WHERE operation_id=? AND phase='CUTOVER_READY'""",
                    (
                        canonical_sha256(
                            {
                                "contract": MEMORY_CRYPTO_OPERATION_CONTRACT,
                                "encrypted_content_bytes": active_bytes,
                                "encrypted_record_count": active_count,
                                "expired_record_count": final_expired_count,
                                "final_state": "COMPLETE",
                                "key_version": int(epoch["key_version"]),
                                "kind": "MIGRATE_V9",
                                "operation_id": operation_id,
                                "principal_id": principal_id,
                                "source_record_count": len(final_legacy_rows),
                                "source_schema": 9,
                                "target_epoch_id": str(epoch["epoch_id"]),
                                "target_schema": 10,
                            }
                        ),
                        stamp,
                        stamp,
                        operation_id,
                    ),
                )
                connection.execute(
                    """INSERT INTO memory_crypto_migration_audits(
                       operation_id,principal_id,source_schema,target_schema,
                       provider_install_id,key_version,source_record_count,
                       encrypted_record_count,expired_record_count,
                       source_content_bytes,encrypted_content_bytes,started_at,
                       cutover_at,contract_sha256
                       ) VALUES(?,?,9,10,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        operation_id,
                        principal_id,
                        active_runtime.install_id,
                        epoch["key_version"],
                        len(final_legacy_rows),
                        active_count,
                        final_expired_count,
                        sum(
                            int(row["content_bytes"])
                            for row in final_legacy_rows
                        ),
                        active_bytes,
                        operation["created_at"],
                        stamp,
                        canonical_sha256(
                            {
                                "contract": MEMORY_PAYLOAD_CONTRACT,
                                "envelope_version": MEMORY_ENVELOPE_VERSION,
                                "normalizer_version": MEMORY_NORMALIZER_VERSION,
                                "record_schema_version": MEMORY_RECORD_SCHEMA_VERSION,
                            }
                        ),
                    ),
                )
        hook("cutover_committed")
        with self._lock:
            final_state = dict(
                self._memory_crypto_state_tx(self._connection, principal_id)
            )
            final_epoch_row = self._connection.execute(
                "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
                (final_state["write_epoch_id"],),
            ).fetchone()
            if final_epoch_row is None:
                raise MemoryStorageUnavailable("memory_key_epoch_missing")
            self._require_session_epoch(
                migration_session,
                final_state,
                dict(final_epoch_row),
            )
            self._memory_epoch_session = migration_session
        self.sweep_expired_memory(now=selected, limit=MEMORY_SWEEP_LIMIT)
        return self.memory_storage_status()

    @staticmethod
    def rotate_memory_storage_key() -> None:
        raise OrchestratorError("memory_rotation_not_supported", status=409)

    @staticmethod
    def destroy_memory_storage_key() -> None:
        raise OrchestratorError("memory_key_destruction_not_supported", status=409)

    @classmethod
    def _delete_saved_memory_row_tx(
        cls,
        connection: sqlite3.Connection,
        row: Mapping[str, Any],
        *,
        reason: str,
        action: str,
        actor_kind: str,
        actor_id: str | None,
        now: datetime,
        encrypted: bool,
        delete_request_id: str | None = None,
        operation_id: str | None = None,
    ) -> bool:
        cls._record_memory_tombstone_tx(
            connection,
            row,
            reason=reason,
            action=action,
            actor_kind=actor_kind,
            actor_id=actor_id,
            now=now,
            revision_increment=int(encrypted),
            delete_request_id=delete_request_id,
            operation_id=operation_id,
        )
        table = "memory_records" if encrypted else "memory_records_v9_legacy"
        deleted = connection.execute(
            f"DELETE FROM {table} WHERE record_id=?",
            (row["record_id"],),
        )
        return deleted.rowcount == 1

    @classmethod
    def _record_memory_tombstone_tx(
        cls,
        connection: sqlite3.Connection,
        row: Mapping[str, Any],
        *,
        reason: str,
        action: str,
        actor_kind: str,
        actor_id: str | None,
        now: datetime,
        revision_increment: int,
        delete_request_id: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        """Write one replay-safe logical deletion marker without deleting a row."""

        if revision_increment not in {0, 1}:
            raise ValueError("memory revision increment is invalid")
        source_binding = cls._memory_source_binding_sha256(row)
        existing_tombstone = connection.execute(
            "SELECT * FROM memory_record_tombstones WHERE record_id=?",
            (row["record_id"],),
        ).fetchone()
        if existing_tombstone is not None:
            if (
                existing_tombstone["principal_id"] != row["principal_id"]
                or existing_tombstone["source_binding_sha256"] != source_binding
                or existing_tombstone["reason"] != reason
                or existing_tombstone["delete_request_id"] != delete_request_id
            ):
                raise OrchestratorError("memory_delete_conflict", status=409)
        else:
            connection.execute(
                """INSERT INTO memory_record_tombstones(
                   record_id,principal_id,source_binding_sha256,reason,
                   delete_request_id,memory_revision,deleted_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (
                    row["record_id"],
                    row["principal_id"],
                    source_binding,
                    reason,
                    delete_request_id,
                    int(
                        connection.execute(
                            "SELECT memory_revision FROM memory_crypto_state "
                            "WHERE principal_id=?",
                            (row["principal_id"],),
                        ).fetchone()[0]
                    )
                    + revision_increment,
                    utc_text(now),
                ),
            )
            cls._append_memory_record_audit_tx(
                connection,
                row=row,
                action=action,
                actor_kind=actor_kind,
                actor_id=actor_id,
                outcome_code="removed_from_saved_memory",
                now=now,
                operation_id=operation_id,
            )

    def _sweep_expired_memory_tx(
        self,
        connection: sqlite3.Connection,
        *,
        now: datetime,
        limit: int,
    ) -> int:
        if not self._memory_storage_enabled:
            return 0
        stamp = utc_text(now)
        principal_id = self._local_owner_principal_tx(connection)
        state = self._memory_crypto_state_tx(connection, principal_id)
        if state["storage_state"] == "MIGRATING":
            return 0
        encrypted = state["storage_state"] == "ENCRYPTED"
        table = "memory_records" if encrypted else "memory_records_v9_legacy"
        rows = connection.execute(
            f"SELECT * FROM {table} WHERE expires_at<=? "
            "ORDER BY expires_at,record_id LIMIT ?",
            (stamp, limit),
        ).fetchall()
        for row in rows:
            if not self._delete_saved_memory_row_tx(
                connection,
                row,
                reason="ttl_expiry",
                action="TTL_EXPIRED",
                actor_kind="ttl_sweep",
                actor_id=None,
                now=now,
                encrypted=encrypted,
            ):
                raise RuntimeError("expired memory changed during sweep")
        if not encrypted:
            usage = connection.execute(
                """SELECT COUNT(*) AS count,COALESCE(SUM(content_bytes),0) AS bytes
                   FROM memory_records_v9_legacy
                   WHERE principal_id=? AND expires_at>?""",
                (principal_id, stamp),
            ).fetchone()
            connection.execute(
                """UPDATE memory_principal_usage
                   SET active_record_count=?,active_content_bytes=?,updated_at=?
                   WHERE principal_id=?""",
                (int(usage["count"]), int(usage["bytes"]), stamp, principal_id),
            )
        connection.execute(
            "INSERT INTO schema_meta(key,value) VALUES('memory_last_sweep_at',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (stamp,),
        )
        return len(rows)

    def _maybe_sweep_expired_memory_tx(
        self,
        connection: sqlite3.Connection,
        *,
        now: datetime,
    ) -> int:
        if not self._memory_storage_enabled:
            return 0
        row = connection.execute(
            "SELECT value FROM schema_meta WHERE key='memory_last_sweep_at'"
        ).fetchone()
        if row is not None:
            try:
                last_sweep = parse_utc(row["value"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("memory sweep timestamp is malformed") from exc
            if now < last_sweep + MEMORY_SWEEP_INTERVAL:
                return 0
        return self._sweep_expired_memory_tx(
            connection, now=now, limit=MEMORY_SWEEP_LIMIT
        )

    def sweep_expired_memory(
        self,
        *,
        now: datetime | None = None,
        limit: int = MEMORY_SWEEP_LIMIT,
    ) -> int:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MEMORY_SWEEP_LIMIT
        ):
            raise ValueError("memory sweep limit is invalid")
        selected = now or utc_now()
        with self._transaction() as connection:
            return self._sweep_expired_memory_tx(
                connection, now=selected, limit=limit
            )

    def sweep_expired_memory_if_due(
        self, *, now: datetime | None = None
    ) -> int:
        selected = now or utc_now()
        with self._transaction() as connection:
            return self._maybe_sweep_expired_memory_tx(connection, now=selected)

    def _insert_memory_record_tx(
        self,
        connection: sqlite3.Connection,
        *,
        session: MemoryEpochSessionV1,
        principal_id: str,
        conversation_id: str,
        task_id: str,
        source_event_id: str,
        content: str,
        metadata_json: str,
        now: datetime,
    ) -> tuple[dict[str, Any], bool]:
        fields = self._memory_content_fields(content)
        stamp = utc_text(now)
        state = dict(self._memory_crypto_state_tx(connection, principal_id))
        if state["storage_state"] != "ENCRYPTED":
            raise MemoryStorageUnavailable("memory_storage_not_prepared")
        epoch_row = connection.execute(
            "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
            (state["write_epoch_id"],),
        ).fetchone()
        if epoch_row is None:
            raise MemoryStorageUnavailable("memory_key_epoch_missing")
        epoch = dict(epoch_row)
        self._require_session_epoch(session, state, epoch)
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("memory metadata is invalid") from exc
        if not isinstance(metadata, dict) or canonical_json(metadata) != metadata_json:
            raise ValueError("memory metadata is invalid")

        expired_source_rows = connection.execute(
            "SELECT * FROM memory_records WHERE conversation_id=? "
            "AND source_event_id=? AND expires_at<=?",
            (conversation_id, source_event_id, stamp),
        ).fetchall()
        for expired in expired_source_rows:
            self._delete_saved_memory_row_tx(
                connection,
                expired,
                reason="ttl_expiry",
                action="TTL_EXPIRED",
                actor_kind="ttl_sweep",
                actor_id=None,
                now=now,
                encrypted=True,
            )
        same_source = connection.execute(
            self._memory_row_select_sql(
                where="r.conversation_id=? AND r.source_event_id=? AND r.expires_at>?"
            ),
            (conversation_id, source_event_id, stamp),
        ).fetchone()
        if same_source is not None:
            projected = self._project_verified_memory_row_tx(
                connection, session, dict(same_source)
            )
            if projected["content"] != content or projected["metadata"] != metadata:
                raise OrchestratorError("idempotency_conflict", status=409)
            return projected, False
        blind = session.blind_tokens(
            content=content,
            normalized_content=normalize_explicit_memory_target(content),
            terms=fields["terms"],
        )
        expired_duplicates = connection.execute(
            """SELECT * FROM memory_records
               WHERE principal_id=? AND key_version=? AND normalized_hmac=?
                 AND expires_at<=?""",
            (principal_id, epoch["key_version"], blind.normalized_hmac, stamp),
        ).fetchall()
        for expired in expired_duplicates:
            self._delete_saved_memory_row_tx(
                connection,
                expired,
                reason="ttl_expiry",
                action="TTL_EXPIRED",
                actor_kind="ttl_sweep",
                actor_id=None,
                now=now,
                encrypted=True,
            )
        duplicate = connection.execute(
            self._memory_row_select_sql(
                where=(
                    "r.principal_id=? AND r.key_version=? "
                    "AND r.normalized_hmac=? AND r.expires_at>?"
                )
            ),
            (principal_id, epoch["key_version"], blind.normalized_hmac, stamp),
        ).fetchone()
        if duplicate is not None:
            return (
                self._project_verified_memory_row_tx(
                    connection, session, dict(duplicate)
                ),
                False,
            )
        usage = connection.execute(
            "SELECT active_record_count,active_content_bytes "
            "FROM memory_principal_usage WHERE principal_id=?",
            (principal_id,),
        ).fetchone()
        if usage is None:
            raise RuntimeError("memory principal usage is unavailable")
        if (
            int(usage["active_record_count"]) + 1
            > MAX_MEMORY_PRINCIPAL_RECORDS
            or int(usage["active_content_bytes"]) + fields["content_bytes"]
            > MAX_MEMORY_PRINCIPAL_BYTES
        ):
            raise _MemoryQuotaExceeded
        record_id = new_uuid4()
        source = {
            "record_id": record_id,
            "principal_id": principal_id,
            "conversation_id": conversation_id,
            "task_id": task_id,
            "source_event_id": source_event_id,
            "content": content,
            "metadata": metadata,
            "content_bytes": fields["content_bytes"],
            "term_count": fields["term_count"],
            "terms": fields["terms"],
            "expires_at": utc_text(
                now + timedelta(days=MEMORY_RECORD_TTL_DAYS)
            ),
            "created_at": stamp,
        }
        values = self._encrypted_record_values(
            session,
            principal_id=principal_id,
            epoch_id=str(epoch["epoch_id"]),
            key_version=int(epoch["key_version"]),
            source=source,
        )
        if values["normalized_hmac"] != blind.normalized_hmac:
            raise RuntimeError("memory blind index derivation is inconsistent")
        self._insert_encrypted_memory_row_tx(connection, values)
        self._append_memory_record_audit_tx(
            connection,
            row=values,
            action="CREATED",
            actor_kind="explicit_command",
            actor_id=task_id,
            outcome_code="stored",
            now=now,
            new_epoch_id=str(epoch["epoch_id"]),
        )
        return {
            **{key: value for key, value in values.items() if key != "term_hmacs"},
            "content": content,
            "metadata": metadata,
            "metadata_json": metadata_json,
            "terms": fields["terms"],
        }, True

    async def search_memory_records(
        self,
        *,
        conversation_id: str,
        query: str,
        limit: int,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Run encrypted indexed recall on a cancellable read-only connection."""

        query_bytes = len(query.encode("utf-8"))
        terms = memory_search_terms(query)
        if not query or query_bytes > MAX_MEMORY_QUERY_BYTES:
            raise ValueError("memory recall query is empty or too large")
        if len(terms) > MAX_MEMORY_QUERY_TERMS:
            raise ValueError("memory recall query has too many unique terms")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("memory recall limit is invalid")
        if not terms:
            return []
        session = self._memory_session_or_unavailable()
        with self._lock:
            principal_id = self._principal_for_conversation_tx(
                self._connection, conversation_id
            )
            state = dict(self._memory_crypto_state_tx(self._connection, principal_id))
            if state["storage_state"] != "ENCRYPTED":
                raise MemoryStorageUnavailable("memory_storage_not_prepared")
            epoch_row = self._connection.execute(
                "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
                (state["write_epoch_id"],),
            ).fetchone()
            if epoch_row is None:
                raise MemoryStorageUnavailable("memory_key_epoch_missing")
            epoch = dict(epoch_row)
        selected = now or utc_now()
        expires_after = utc_text(selected)
        cancelled = threading.Event()
        deadline = time.monotonic() + MEMORY_RECALL_DEADLINE_SECONDS
        database_uri = self.path.resolve().as_uri() + "?mode=ro"

        def query_read_only() -> list[dict[str, Any]]:
            if cancelled.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("memory recall deadline exceeded")
            try:
                self._require_session_epoch(session, state, epoch)
                term_tokens = session.blind_term_tokens(terms=terms)
            except MemoryCryptoError as exc:
                raise MemoryStorageUnavailable(exc.code) from exc
            if cancelled.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("memory recall deadline exceeded")
            connection = sqlite3.connect(
                database_uri,
                uri=True,
                isolation_level=None,
                timeout=MEMORY_RECALL_BUSY_TIMEOUT_MS / 1000,
            )
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only=ON")
                connection.execute(
                    f"PRAGMA busy_timeout={MEMORY_RECALL_BUSY_TIMEOUT_MS}"
                )
                connection.set_progress_handler(
                    lambda: int(
                        cancelled.is_set() or time.monotonic() >= deadline
                    ),
                    100,
                )
                current_state_row = connection.execute(
                    "SELECT * FROM memory_crypto_state WHERE principal_id=?",
                    (principal_id,),
                ).fetchone()
                current_epoch_row = connection.execute(
                    "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
                    (state["write_epoch_id"],),
                ).fetchone()
                if current_state_row is None or current_epoch_row is None:
                    raise MemoryStorageUnavailable(
                        "memory_key_epoch_missing"
                    )
                self._require_session_epoch(
                    session,
                    dict(current_state_row),
                    dict(current_epoch_row),
                )
                placeholders = ",".join("?" for _ in term_tokens)
                rows = connection.execute(
                    f"""SELECT r.*,COUNT(*) AS match_count,
                               st.epoch AS source_task_epoch,
                               st.turn_id AS source_turn_id,
                               st.state AS source_task_state,
                               st.conversation_id AS source_task_conversation_id,
                               ct.user_language AS source_language,
                               ct.status AS source_turn_status,
                               ct.conversation_id AS source_turn_conversation_id,
                               ct.task_id AS source_turn_task_id,
                               ct.turn_id AS source_canonical_turn_id,
                               i.task_id AS source_inbox_task_id,
                               i.turn_id AS source_inbox_turn_id
                       FROM memory_terms AS t
                       INDEXED BY memory_terms_by_principal_term
                       JOIN memory_records AS r
                         ON r.record_id=t.record_id
                        AND r.principal_id=t.principal_id
                        AND r.key_epoch_id=t.key_epoch_id
                        AND r.key_version=t.key_version
                       JOIN tasks st ON st.task_id=r.task_id
                       JOIN canonical_turns ct ON ct.task_id=r.task_id
                       JOIN inbox i ON i.event_id=r.source_event_id
                       WHERE t.principal_id=?
                         AND t.key_version=?
                         AND t.term_hmac IN ({placeholders})
                         AND r.expires_at>?
                       GROUP BY r.record_id
                       ORDER BY match_count DESC,r.created_at DESC,r.record_id
                       LIMIT ?""",
                    (
                        principal_id,
                        epoch["key_version"],
                        *term_tokens,
                        expires_after,
                        limit,
                    ),
                ).fetchall()
                selected_rows: list[dict[str, Any]] = []
                for row in rows:
                    item = dict(row)
                    item["stored_term_hmacs"] = tuple(
                        bytes(token["term_hmac"])
                        for token in connection.execute(
                            """SELECT term_hmac FROM memory_terms
                               INDEXED BY memory_terms_by_record
                               WHERE record_id=? ORDER BY term_hmac""",
                            (row["record_id"],),
                        ).fetchall()
                    )
                    selected_rows.append(item)
            except sqlite3.OperationalError:
                if cancelled.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError(
                        "memory recall deadline exceeded"
                    ) from None
                raise
            finally:
                connection.close()
            projected_rows: list[dict[str, Any]] = []
            for row in selected_rows:
                if cancelled.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("memory recall deadline exceeded")
                try:
                    projected = self._decrypt_memory_row(session, row)
                    expected_term_tokens = session.blind_term_tokens(
                        terms=projected["terms"]
                    )
                except MemoryCryptoError as exc:
                    raise MemoryStorageUnavailable(exc.code) from exc
                if tuple(sorted(expected_term_tokens)) != row[
                    "stored_term_hmacs"
                ]:
                    raise MemoryStorageUnavailable(
                        "memory_term_index_binding_invalid"
                    )
                actual_matches = len(set(terms).intersection(projected["terms"]))
                if actual_matches != int(row["match_count"]):
                    raise MemoryStorageUnavailable(
                        "memory_recall_match_binding_invalid"
                    )
                projected["match_count"] = actual_matches
                projected_rows.append(projected)
            return projected_rows

        worker = asyncio.create_task(asyncio.to_thread(query_read_only))

        def consume_worker_result(done: asyncio.Task[list[dict[str, Any]]]) -> None:
            if not done.cancelled():
                with suppress(BaseException):
                    done.exception()

        worker.add_done_callback(consume_worker_result)
        try:
            return await asyncio.wait_for(
                asyncio.shield(worker),
                timeout=max(0.001, deadline - time.monotonic()),
            )
        except (asyncio.CancelledError, TimeoutError):
            cancelled.set()
            raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold one SQLite snapshot across a bounded multi-query read."""

        with self._lock:
            self._connection.execute("BEGIN")
            with self._connection:
                yield self._connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            with self._connection:
                yield self._connection

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
                        """SELECT record_id,principal_id,task_id,source_event_id,
                                  key_epoch_id,key_version,binding_sha256,
                                  envelope_sha256,content_bytes,term_count,
                                  expires_at,normalizer_version,created_at
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
        session = self._memory_session_or_unavailable()
        self._memory_content_fields(content)
        metadata_json = canonical_json(metadata)
        expected_scope = f"anamnesis::jarvis::{conversation_id}"
        if scoped_user_id != expected_scope:
            raise ValueError("memory scope does not match conversation")
        with self._transaction() as connection:
            self._sweep_expired_memory_tx(
                connection, now=selected, limit=MEMORY_SWEEP_LIMIT
            )
            task = connection.execute(
                """SELECT t.*,c.current_epoch,c.state AS conversation_state,
                          ct.status AS turn_status,ct.user_language,
                          i.event_id AS bound_event_id
                   FROM tasks t JOIN conversations c USING(conversation_id)
                   JOIN canonical_turns ct USING(task_id)
                   JOIN inbox i USING(task_id)
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
                or task["bound_event_id"] != source_event_id
            ):
                raise StaleEpochError()
            expected_metadata = {
                "conversation_id": conversation_id,
                "task_id": task_id,
                "task_epoch": str(epoch),
                "turn_id": task["turn_id"],
                "source_event_id": source_event_id,
                "language": task["user_language"],
                "_anamnesis_namespace": "jarvis",
                "_anamnesis_user_id": conversation_id,
            }
            if dict(metadata) != expected_metadata:
                raise ValueError("memory metadata binding is invalid")
            principal_id = self._principal_for_conversation_tx(
                connection, conversation_id
            )
            try:
                row, _ = self._insert_memory_record_tx(
                    connection,
                    session=session,
                    principal_id=principal_id,
                    conversation_id=conversation_id,
                    task_id=task_id,
                    source_event_id=source_event_id,
                    content=content,
                    metadata_json=metadata_json,
                    now=selected,
                )
            except _MemoryQuotaExceeded as exc:
                raise OrchestratorError(
                    "memory_quota_exceeded", status=409
                ) from exc
            result = dict(row)
            result["scoped_user_id"] = expected_scope
            return result

    def complete_explicit_memory_command(
        self,
        task_id: str,
        epoch: int,
        *,
        enabled: bool,
        locale: str,
        now: datetime | None = None,
        delivery_ttl: timedelta | None = None,
    ) -> tuple[TaskSnapshot, ExplicitMemoryCommandResult]:
        """Atomically mutate explicit memory and commit its deterministic reply.

        ``task_events`` is the durable command journal and the normal response
        outbox is the delivery journal.  A process failure before SQLite commit
        leaves both absent; a failure afterwards can only replay the persisted
        outcome and response events.
        """

        if not isinstance(enabled, bool):
            raise TypeError("explicit memory enabled flag must be boolean")
        if not locale or nfc(locale) != locale:
            raise ValueError("explicit memory response locale is invalid")
        selected = now or utc_now()
        prepared_session = self._memory_epoch_session
        stamp = utc_text(selected)
        delivery = utc_text(selected + (delivery_ttl or self.delivery_ttl))
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT t.*,c.current_epoch,c.state AS conversation_state,
                          ct.user_text,ct.user_language,ct.status AS turn_status,
                          ct.assistant_text,ct.effects_json,i.event_id AS source_event_id
                   FROM tasks t JOIN conversations c USING(conversation_id)
                   JOIN canonical_turns ct USING(task_id)
                   JOIN inbox i USING(task_id)
                   WHERE t.task_id=? AND t.epoch=? AND t.tombstoned=0""",
                (task_id, epoch),
            ).fetchone()
            if (
                row is None
                or int(row["current_epoch"]) != epoch
                or row["conversation_state"] != "ACTIVE"
            ):
                raise StaleEpochError()
            parsed = parse_explicit_memory_command(str(row["user_text"]))
            if parsed is None:
                raise InvalidStateError()
            operation, target = parsed
            target_sha256 = hashlib.sha256(target.encode("utf-8")).hexdigest()
            existing_rows = connection.execute(
                """SELECT event_type,payload_json FROM task_events
                   WHERE task_id=? AND event_type LIKE 'memory.command.%'""",
                (task_id,),
            ).fetchall()
            if len(existing_rows) > 1:
                raise OrchestratorError("memory_command_audit_invalid", status=409)
            existing = existing_rows[0] if existing_rows else None
            if existing is not None:
                try:
                    payload = json.loads(existing["payload_json"])
                except json.JSONDecodeError as exc:
                    raise OrchestratorError(
                        "memory_command_audit_invalid", status=409
                    ) from exc
                expected_keys = {
                    "operation",
                    "status",
                    "target_sha256",
                    "matched_count",
                    "changed",
                    "outcome_sha256",
                    "response_sha256",
                }
                if not isinstance(payload, dict) or set(payload) != expected_keys:
                    raise OrchestratorError("memory_command_audit_invalid", status=409)
                try:
                    status = ExplicitMemoryCommandStatus(payload["status"])
                    result = ExplicitMemoryCommandResult(
                        operation=payload["operation"],
                        status=status,
                        matched_count=payload["matched_count"],
                        changed=payload["changed"],
                        target_sha256=payload["target_sha256"],
                    )
                except (TypeError, ValueError) as exc:
                    raise OrchestratorError(
                        "memory_command_audit_invalid", status=409
                    ) from exc
                response_text = explicit_memory_response(
                    result.operation, result.status, str(row["user_language"])
                )
                expected_outcome_sha256 = canonical_sha256(
                    {
                        "operation": result.operation,
                        "status": result.status.value,
                        "target_sha256": result.target_sha256,
                        "matched_count": result.matched_count,
                        "changed": result.changed,
                    }
                )
                expected_event_type = (
                    f"memory.command.{operation}.{result.status.value}"
                )
                if (
                    existing["event_type"] != expected_event_type
                    or result.operation != operation
                    or result.target_sha256 != target_sha256
                    or row["state"] != TaskState.COMPLETED.value
                    or row["turn_status"] != "COMPLETED"
                    or row["assistant_text"] != response_text
                    or payload["outcome_sha256"] != expected_outcome_sha256
                    or payload["response_sha256"]
                    != hashlib.sha256(response_text.encode("utf-8")).hexdigest()
                ):
                    raise OrchestratorError("memory_command_replay_invalid", status=409)
                journal = connection.execute(
                    """SELECT event_type,payload_json,task_id,task_epoch,
                              guard_epoch,turn_id,response_id
                       FROM outbox
                       WHERE response_id=? AND event_type LIKE 'response.%'
                       ORDER BY ordinal""",
                    (row["response_id"],),
                ).fetchall()
                expected_types = [
                    "response.started",
                    "response.text.delta",
                    "response.text.done",
                    "response.speech.commit",
                    "response.speech.done",
                ]
                if (
                    [item["event_type"] for item in journal] != expected_types
                    or any(
                        item["task_id"] != task_id
                        or item["task_epoch"] != epoch
                        or item["guard_epoch"] != epoch
                        or item["turn_id"] != row["turn_id"]
                        or item["response_id"] != row["response_id"]
                        for item in journal
                    )
                ):
                    raise OrchestratorError(
                        "memory_command_response_journal_invalid", status=409
                    )
                try:
                    journal_payloads = [
                        json.loads(item["payload_json"]) for item in journal
                    ]
                except json.JSONDecodeError as exc:
                    raise OrchestratorError(
                        "memory_command_response_journal_invalid", status=409
                    ) from exc
                if any(not isinstance(item, dict) for item in journal_payloads):
                    raise OrchestratorError(
                        "memory_command_response_journal_invalid", status=409
                    )
                committed_locale = journal_payloads[0].get("locale")
                if (
                    set(journal_payloads[0]) != {"locale"}
                    or not isinstance(committed_locale, str)
                    or not committed_locale
                    or nfc(committed_locale) != committed_locale
                    or journal_payloads[1]
                    != {"chunk_index": 0, "text": response_text}
                    or journal_payloads[2] != {"chunk_count": 1}
                    or journal_payloads[3]
                    != {
                        "unit_index": 1,
                        "locale": committed_locale,
                        "text": response_text,
                    }
                    or journal_payloads[4] != {"last_unit_index": 1}
                ):
                    raise OrchestratorError(
                        "memory_command_response_journal_invalid", status=409
                    )
                expected_effect = asdict(
                    CanonicalEffectV1(
                        kind="memory",
                        name=operation,
                        status=result.status.value,
                        arguments_sha256=target_sha256,
                        result_sha256=expected_outcome_sha256,
                    )
                )
                try:
                    effects = json.loads(row["effects_json"])
                except json.JSONDecodeError as exc:
                    raise OrchestratorError(
                        "memory_command_replay_invalid", status=409
                    ) from exc
                provenance_row = connection.execute(
                    """SELECT provenance_json,provenance_sha256,
                              policy_label,egress_disposition
                       FROM content_provenance
                       WHERE entity_kind='canonical_turn' AND entity_id=?""",
                    (row["turn_id"],),
                ).fetchone()
                expected_provenance = self._explicit_memory_command_provenance(
                    task_id=task_id,
                    turn_id=row["turn_id"],
                    user_text=row["user_text"],
                    response_text=response_text,
                )
                try:
                    provenance = (
                        None
                        if provenance_row is None
                        else ContentProvenanceV1.from_dict(
                            json.loads(provenance_row["provenance_json"])
                        )
                    )
                except (json.JSONDecodeError, ProvenanceError) as exc:
                    raise OrchestratorError(
                        "memory_command_replay_invalid", status=409
                    ) from exc
                if (
                    effects != [expected_effect]
                    or provenance is None
                    or provenance != expected_provenance
                    or provenance.sha256 != provenance_row["provenance_sha256"]
                    or provenance_row["policy_label"]
                    != expected_provenance.policy_label.value
                    or provenance_row["egress_disposition"]
                    != expected_provenance.egress.value
                ):
                    raise OrchestratorError("memory_command_replay_invalid", status=409)
                return self._task_from_row(row), result

            if (
                row["state"] != TaskState.EXECUTING.value
                or row["turn_status"] != "PENDING"
            ):
                raise StaleEpochError()

            self._sweep_expired_memory_tx(
                connection, now=selected, limit=MEMORY_SWEEP_LIMIT
            )
            matches: list[dict[str, Any]] = []
            principal_id: str | None = None
            session: MemoryEpochSessionV1 | None = None
            crypto_available = False
            over_limit = (
                enabled and len(target.encode("utf-8")) > MAX_MEMORY_RECORD_BYTES
            )
            target_terms = () if not enabled or over_limit else memory_search_terms(
                target
            )
            invalid_content = enabled and not over_limit and (
                not target_terms or len(target_terms) > MAX_MEMORY_RECORD_TERMS
            )
            if not enabled:
                matched_count = 0
                status = ExplicitMemoryCommandStatus.NOT_ENABLED
                changed = False
            elif over_limit:
                matched_count = 0
                status = ExplicitMemoryCommandStatus.LIMIT_EXCEEDED
                changed = False
            elif invalid_content:
                matched_count = 0
                status = ExplicitMemoryCommandStatus.INVALID_CONTENT
                changed = False
            else:
                principal_id = self._principal_for_conversation_tx(
                    connection, row["conversation_id"]
                )
                state = dict(self._memory_crypto_state_tx(connection, principal_id))
                session = prepared_session
                if state["storage_state"] == "ENCRYPTED" and session is not None:
                    epoch_row = connection.execute(
                        "SELECT * FROM memory_key_epochs WHERE epoch_id=?",
                        (state["write_epoch_id"],),
                    ).fetchone()
                    try:
                        if epoch_row is None:
                            raise MemoryStorageUnavailable(
                                "memory_key_epoch_missing"
                            )
                        epoch_row = dict(epoch_row)
                        self._require_session_epoch(session, state, epoch_row)
                        normalized_hmac = session.blind_token(
                            kind="normalized",
                            value=normalize_explicit_memory_target(target),
                        )
                        selected_matches = connection.execute(
                            self._memory_row_select_sql(
                                where=(
                                    "r.principal_id=? AND r.key_version=? "
                                    "AND r.normalized_hmac=? AND r.expires_at>?"
                                )
                            )
                            + " ORDER BY r.record_id",
                            (
                                principal_id,
                                epoch_row["key_version"],
                                normalized_hmac,
                                stamp,
                            ),
                        ).fetchall()
                        for selected_match in selected_matches:
                            projected = self._project_verified_memory_row_tx(
                                connection, session, dict(selected_match)
                            )
                            if normalize_explicit_memory_target(
                                projected["content"]
                            ) != normalize_explicit_memory_target(target):
                                raise MemoryStorageUnavailable(
                                    "memory_index_binding_invalid"
                                )
                            matches.append(dict(selected_match))
                        matched_count = len(matches)
                        crypto_available = True
                    except (MemoryCryptoError, MemoryStorageUnavailable):
                        matched_count = 0
                        status = ExplicitMemoryCommandStatus.UNAVAILABLE
                        changed = False
                else:
                    matched_count = 0
                    status = ExplicitMemoryCommandStatus.UNAVAILABLE
                    changed = False
            if not enabled or over_limit or invalid_content or not crypto_available:
                pass
            elif operation == "remember":
                if matches:
                    status = ExplicitMemoryCommandStatus.ALREADY_EXISTS
                    changed = False
                else:
                    status = ExplicitMemoryCommandStatus.STORED
                    changed = True
                    metadata = {
                        "conversation_id": row["conversation_id"],
                        "task_id": task_id,
                        "task_epoch": str(epoch),
                        "turn_id": row["turn_id"],
                        "source_event_id": row["source_event_id"],
                        "language": row["user_language"],
                        "_anamnesis_namespace": "jarvis",
                        "_anamnesis_user_id": row["conversation_id"],
                    }
                    try:
                        self._insert_memory_record_tx(
                            connection,
                            session=session,
                            principal_id=principal_id,
                            conversation_id=row["conversation_id"],
                            task_id=task_id,
                            source_event_id=row["source_event_id"],
                            content=target,
                            metadata_json=canonical_json(metadata),
                            now=selected,
                        )
                    except _MemoryQuotaExceeded:
                        status = ExplicitMemoryCommandStatus.QUOTA_EXCEEDED
                        changed = False
                    except (MemoryCryptoError, MemoryStorageUnavailable):
                        status = ExplicitMemoryCommandStatus.UNAVAILABLE
                        changed = False
                        matched_count = 0
            elif matched_count == 0:
                status = ExplicitMemoryCommandStatus.NOT_FOUND
                changed = False
            elif matched_count > 1:
                status = ExplicitMemoryCommandStatus.AMBIGUOUS
                changed = False
            else:
                status = ExplicitMemoryCommandStatus.DELETED
                changed = True
                assert principal_id is not None
                if not self._delete_saved_memory_row_tx(
                    connection,
                    matches[0],
                    reason="explicit_forget",
                    action="EXPLICIT_FORGOT",
                    actor_kind="explicit_command",
                    actor_id=task_id,
                    now=selected,
                    encrypted=True,
                ):
                    raise OrchestratorError(
                        "memory_delete_target_changed", status=409
                    )

            result = ExplicitMemoryCommandResult(
                operation=operation,
                status=status,
                matched_count=matched_count,
                changed=changed,
                target_sha256=target_sha256,
            )
            outcome_document = {
                "operation": result.operation,
                "status": result.status.value,
                "target_sha256": result.target_sha256,
                "matched_count": result.matched_count,
                "changed": result.changed,
            }
            outcome_sha256 = canonical_sha256(outcome_document)
            response_text = explicit_memory_response(
                operation, status, str(row["user_language"])
            )
            command_payload = {
                **outcome_document,
                "outcome_sha256": outcome_sha256,
                "response_sha256": hashlib.sha256(
                    response_text.encode("utf-8")
                ).hexdigest(),
            }
            self._append_task_event_tx(
                connection,
                task_id,
                epoch,
                f"memory.command.{operation}.{status.value}",
                command_payload,
                selected,
            )
            if operation == "forget" and status in {
                ExplicitMemoryCommandStatus.DELETED,
                ExplicitMemoryCommandStatus.NOT_FOUND,
                ExplicitMemoryCommandStatus.AMBIGUOUS,
            }:
                legacy_status = {
                    ExplicitMemoryCommandStatus.DELETED: "completed",
                    ExplicitMemoryCommandStatus.NOT_FOUND: "empty",
                    ExplicitMemoryCommandStatus.AMBIGUOUS: "ambiguous",
                }[status]
                self._append_task_event_tx(
                    connection,
                    task_id,
                    epoch,
                    f"memory.delete.{legacy_status}",
                    {
                        "status": legacy_status,
                        "target_sha256": target_sha256,
                        "matched_count": matched_count,
                        "deleted_count": 1 if changed else 0,
                        "authoritative_state_changed": changed,
                        "selection": "exact_normalized_content",
                    },
                    selected,
                )

            effects = json.loads(row["effects_json"])
            effects.append(
                asdict(
                    CanonicalEffectV1(
                        kind="memory",
                        name=operation,
                        status=status.value,
                        arguments_sha256=target_sha256,
                        result_sha256=outcome_sha256,
                    )
                )
            )
            provenance = self._explicit_memory_command_provenance(
                task_id=task_id,
                turn_id=row["turn_id"],
                user_text=row["user_text"],
                response_text=response_text,
            )
            self._append_task_event_tx(
                connection, task_id, epoch, "RESPONDING", {}, selected
            )
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
            self._append_outbox_tx(
                connection,
                conversation_id=row["conversation_id"],
                task_id=task_id,
                task_epoch=epoch,
                guard_epoch=epoch,
                turn_id=row["turn_id"],
                response_id=row["response_id"],
                event_type="response.text.delta",
                payload={"chunk_index": 0, "text": response_text},
                now=selected,
            )
            self._append_task_event_tx(
                connection, task_id, epoch, "COMMITTING", {}, selected
            )
            self._append_outbox_tx(
                connection,
                conversation_id=row["conversation_id"],
                task_id=task_id,
                task_epoch=epoch,
                guard_epoch=epoch,
                turn_id=row["turn_id"],
                response_id=row["response_id"],
                event_type="response.text.done",
                payload={"chunk_count": 1},
                delivery_expires_at=delivery,
                now=selected,
            )
            self._append_outbox_tx(
                connection,
                conversation_id=row["conversation_id"],
                task_id=task_id,
                task_epoch=epoch,
                guard_epoch=epoch,
                turn_id=row["turn_id"],
                response_id=row["response_id"],
                event_type="response.speech.commit",
                payload={"unit_index": 1, "locale": locale, "text": response_text},
                delivery_expires_at=delivery,
                now=selected,
            )
            self._append_outbox_tx(
                connection,
                conversation_id=row["conversation_id"],
                task_id=task_id,
                task_epoch=epoch,
                guard_epoch=epoch,
                turn_id=row["turn_id"],
                response_id=row["response_id"],
                event_type="response.speech.done",
                payload={"last_unit_index": 1},
                delivery_expires_at=delivery,
                now=selected,
            )
            elapsed = float(row["active_elapsed_seconds"])
            if row["active_started_at"] is not None:
                elapsed += max(
                    0.0,
                    (selected - parse_utc(row["active_started_at"])).total_seconds(),
                )
            updated = connection.execute(
                """UPDATE tasks SET state='COMPLETED',partial_output=1,
                          active_elapsed_seconds=?,active_started_at=NULL,
                          delivery_expires_at=?,disconnect_expires_at=NULL,updated_at=?
                   WHERE task_id=? AND epoch=? AND state='EXECUTING' AND tombstoned=0""",
                (elapsed, delivery, stamp, task_id, epoch),
            )
            if updated.rowcount != 1:
                raise StaleEpochError()
            connection.execute(
                """UPDATE outbox SET delivery_expires_at=?
                   WHERE task_id=? AND state='REPLAYABLE'""",
                (delivery, task_id),
            )
            connection.execute(
                """UPDATE canonical_turns
                   SET assistant_text=?,effects_json=?,status='COMPLETED',completed_at=?
                   WHERE task_id=? AND status='PENDING'""",
                (response_text, canonical_json(effects), stamp, task_id),
            )
            connection.execute(
                """INSERT INTO content_provenance(
                   entity_kind,entity_id,provenance_json,provenance_sha256,
                   policy_label,egress_disposition,created_at
                   ) VALUES('canonical_turn',?,?,?,?,?,?)""",
                (
                    row["turn_id"],
                    jcs_text(provenance.to_dict()),
                    provenance.sha256,
                    provenance.policy_label.value,
                    provenance.egress.value,
                    stamp,
                ),
            )
            connection.execute(
                """UPDATE conversations SET revision=revision+1,updated_at=?
                   WHERE conversation_id=? AND current_epoch=?""",
                (stamp, row["conversation_id"], epoch),
            )
            self._append_task_event_tx(
                connection, task_id, epoch, "COMPLETED", {}, selected
            )
            completed = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            return self._task_from_row(completed), result

    def memory_command_audit(self, task_id: str) -> dict[str, Any] | None:
        """Return the content-free committed explicit command result."""

        with self._lock:
            row = self._connection.execute(
                """SELECT epoch,event_type,payload_json,created_at
                   FROM task_events
                   WHERE task_id=? AND event_type LIKE 'memory.command.%'""",
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

    def control_memory_record_snapshot(
        self,
        record_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Read one unexpired local-owner record and its revision atomically."""

        require_uuid4(record_id)
        session = self._memory_session_or_unavailable()
        selected = utc_text(now or utc_now())
        with self._read_transaction() as connection:
            principal_id = self._local_owner_principal_tx(connection)
            state, _ = self._require_current_memory_session_tx(
                connection,
                principal_id=principal_id,
                session=session,
            )
            row = connection.execute(
                self._memory_row_select_sql(
                    where=(
                        "r.record_id=? AND r.principal_id=? "
                        "AND r.expires_at>?"
                    )
                ),
                (record_id, principal_id, selected),
            ).fetchone()
            record = (
                None
                if row is None
                else self._project_verified_memory_row_tx(
                    connection,
                    session,
                    dict(row),
                )
            )
            revision = int(state["memory_revision"])
        return {"record": record, "memory_revision": revision}

    def control_memory_record_page(
        self,
        *,
        limit: int = 50,
        after: tuple[str, str] | None = None,
        expected_revision: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Read one revision-bound, principal-wide local-owner page."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("memory page limit is invalid")
        if after is not None and (
            not isinstance(after, tuple)
            or len(after) != 2
            or not all(isinstance(item, str) and item for item in after)
        ):
            raise ValueError("memory page cursor is invalid")
        if expected_revision is not None and (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("memory revision is invalid")
        session = self._memory_session_or_unavailable()
        selected = utc_text(now or utc_now())
        with self._read_transaction() as connection:
            principal_id = self._local_owner_principal_tx(connection)
            state, _ = self._require_current_memory_session_tx(
                connection,
                principal_id=principal_id,
                session=session,
            )
            revision = int(state["memory_revision"])
            if expected_revision is not None and revision != expected_revision:
                raise OrchestratorError("memory_revision_conflict", status=409)
            parameters: list[Any] = [principal_id, selected]
            where = "r.principal_id=? AND r.expires_at>?"
            if after is not None:
                parse_utc(after[0])
                require_uuid4(after[1])
                where += (
                    " AND (r.created_at<? OR "
                    "(r.created_at=? AND r.record_id>?))"
                )
                parameters.extend((after[0], after[0], after[1]))
            parameters.append(limit + 1)
            rows = connection.execute(
                self._memory_row_select_sql(where=where)
                + " ORDER BY r.created_at DESC,r.record_id LIMIT ?",
                tuple(parameters),
            ).fetchall()
            has_more = len(rows) > limit
            page_rows = rows[:limit]
            items = [
                self._project_verified_memory_row_tx(
                    connection,
                    session,
                    dict(row),
                )
                for row in page_rows
            ]
        cursor = None
        if has_more and items:
            cursor = (items[-1]["created_at"], items[-1]["record_id"])
        return {
            "items": items,
            "memory_revision": revision,
            "next_cursor": cursor,
        }

    def memory_record(self, record_id: str) -> dict[str, Any] | None:
        session = self._memory_session_or_unavailable()
        with self._lock:
            principal_id = self._local_owner_principal_tx(self._connection)
            self._require_current_memory_session_tx(
                self._connection,
                principal_id=principal_id,
                session=session,
            )
            row = self._connection.execute(
                self._memory_row_select_sql(
                    where="r.record_id=? AND r.principal_id=?"
                ),
                (record_id, principal_id),
            ).fetchone()
            if row is None:
                return None
            result = self._project_verified_memory_row_tx(
                self._connection, session, dict(row)
            )
        result["scoped_user_id"] = (
            f"anamnesis::jarvis::{result['conversation_id']}"
        )
        return result

    def memory_records_for_scope(self, scoped_user_id: str) -> list[dict[str, Any]]:
        prefix = "anamnesis::jarvis::"
        if not scoped_user_id.startswith(prefix) or len(scoped_user_id) == len(prefix):
            raise ValueError("memory scope is invalid")
        conversation_id = scoped_user_id[len(prefix) :]
        session = self._memory_session_or_unavailable()
        with self._lock:
            principal_id = self._principal_for_conversation_tx(
                self._connection, conversation_id
            )
            self._require_current_memory_session_tx(
                self._connection,
                principal_id=principal_id,
                session=session,
            )
            rows = self._connection.execute(
                self._memory_row_select_sql(
                    where="r.principal_id=? AND r.expires_at>?"
                )
                + " ORDER BY r.created_at DESC,r.record_id",
                (principal_id, utc_text(utc_now())),
            ).fetchall()
            result = [
                self._project_verified_memory_row_tx(
                    self._connection, session, dict(row)
                )
                for row in rows
            ]
        for item in result:
            item["scoped_user_id"] = scoped_user_id
        return result

    def memory_records_matching(
        self,
        *,
        conversation_id: str,
        target: str,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        session = self._memory_session_or_unavailable()
        with self._lock:
            principal_id = self._principal_for_conversation_tx(
                self._connection, conversation_id
            )
            _, epoch = self._require_current_memory_session_tx(
                self._connection,
                principal_id=principal_id,
                session=session,
            )
            normalized_hmac = session.blind_token(
                kind="normalized",
                value=normalize_explicit_memory_target(target),
            )
            rows = self._connection.execute(
                self._memory_row_select_sql(
                    where=(
                        "r.principal_id=? AND r.key_version=? "
                        "AND r.normalized_hmac=? AND r.expires_at>?"
                    )
                )
                + " ORDER BY r.record_id",
                (
                    principal_id,
                    epoch["key_version"],
                    normalized_hmac,
                    utc_text(now or utc_now()),
                ),
            ).fetchall()
            return [
                self._project_verified_memory_row_tx(
                    self._connection, session, dict(row)
                )
                for row in rows
            ]

    def delete_memory_record(self, record_id: str) -> bool:
        session = self._memory_session_or_unavailable()
        selected = utc_now()
        with self._transaction() as connection:
            principal_id = self._local_owner_principal_tx(connection)
            self._require_current_memory_session_tx(
                connection,
                principal_id=principal_id,
                session=session,
            )
            row = connection.execute(
                self._memory_row_select_sql(
                    where="r.record_id=? AND r.principal_id=?"
                ),
                (record_id, principal_id),
            ).fetchone()
            if row is None:
                return False
            self._project_verified_memory_row_tx(connection, session, dict(row))
            return self._delete_saved_memory_row_tx(
                connection,
                dict(row),
                reason="control_delete",
                action="CONTROL_DELETED",
                actor_kind="control",
                actor_id=None,
                now=selected,
                encrypted=True,
            )

    def memory_record_page(
        self,
        *,
        conversation_id: str,
        limit: int = 50,
        after: tuple[str, str] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """List a bounded principal-wide page using a stable keyset cursor."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("memory page limit is invalid")
        if after is not None and (
            not isinstance(after, tuple)
            or len(after) != 2
            or not all(isinstance(item, str) and item for item in after)
        ):
            raise ValueError("memory page cursor is invalid")
        session = self._memory_session_or_unavailable()
        selected = utc_text(now or utc_now())
        with self._lock:
            principal_id = self._principal_for_conversation_tx(
                self._connection, conversation_id
            )
            state, _ = self._require_current_memory_session_tx(
                self._connection,
                principal_id=principal_id,
                session=session,
            )
            parameters: list[Any] = [principal_id, selected]
            where = "r.principal_id=? AND r.expires_at>?"
            if after is not None:
                parse_utc(after[0])
                where += (
                    " AND (r.created_at<? OR "
                    "(r.created_at=? AND r.record_id>?))"
                )
                parameters.extend((after[0], after[0], after[1]))
            parameters.append(limit + 1)
            rows = self._connection.execute(
                self._memory_row_select_sql(where=where)
                + " ORDER BY r.created_at DESC,r.record_id LIMIT ?",
                tuple(parameters),
            ).fetchall()
            has_more = len(rows) > limit
            page_rows = rows[:limit]
            items = [
                self._project_verified_memory_row_tx(
                    self._connection, session, dict(row)
                )
                for row in page_rows
            ]
            revision = int(state["memory_revision"])
        cursor = None
        if has_more and items:
            cursor = (items[-1]["created_at"], items[-1]["record_id"])
        return {
            "items": items,
            "memory_revision": revision,
            "next_cursor": cursor,
        }

    @classmethod
    def _delete_memory_record_by_id_tx(
        cls,
        connection: sqlite3.Connection,
        record_id: str,
        *,
        expected_revision: int,
        request_id: str,
        session: MemoryEpochSessionV1 | None,
        now: datetime,
    ) -> dict[str, Any]:
        """Apply a replay-safe saved-memory deletion inside the caller's tx."""

        principal_id = cls._local_owner_principal_tx(connection)
        tombstone = connection.execute(
            "SELECT * FROM memory_record_tombstones WHERE record_id=?",
            (record_id,),
        ).fetchone()
        if tombstone is not None:
            if tombstone["reason"] != "control_delete":
                raise OrchestratorError("not_found", status=404)
            if (
                tombstone["principal_id"] != principal_id
                or tombstone["delete_request_id"] != request_id
                or int(tombstone["memory_revision"]) != expected_revision + 1
            ):
                raise OrchestratorError("idempotency_conflict", status=409)
            return {
                "record_id": record_id,
                "status": "removed_from_saved_memory",
                "chat_history_changed": False,
                "physical_erasure_claimed": False,
                "memory_revision": int(tombstone["memory_revision"]),
            }
        reused_request = connection.execute(
            "SELECT record_id FROM memory_record_tombstones "
            "WHERE delete_request_id=?",
            (request_id,),
        ).fetchone()
        if reused_request is not None:
            raise OrchestratorError("idempotency_conflict", status=409)
        if session is None:
            raise MemoryStorageUnavailable("memory_crypto_session_unavailable")
        state, _ = cls._require_current_memory_session_tx(
            connection,
            principal_id=principal_id,
            session=session,
        )
        if int(state["memory_revision"]) != expected_revision:
            raise OrchestratorError("memory_revision_conflict", status=409)
        row = connection.execute(
            cls._memory_row_select_sql(
                where=(
                    "r.record_id=? AND r.principal_id=? AND r.expires_at>?"
                )
            ),
            (record_id, principal_id, utc_text(now)),
        ).fetchone()
        if row is None:
            raise OrchestratorError("not_found", status=404)
        cls._project_verified_memory_row_tx(connection, session, dict(row))
        if not cls._delete_saved_memory_row_tx(
            connection,
            dict(row),
            reason="control_delete",
            action="CONTROL_DELETED",
            actor_kind="control",
            actor_id=request_id,
            delete_request_id=request_id,
            now=now,
            encrypted=True,
        ):
            raise OrchestratorError("memory_delete_target_changed", status=409)
        current = connection.execute(
            "SELECT memory_revision FROM memory_crypto_state WHERE principal_id=?",
            (principal_id,),
        ).fetchone()
        revision = int(current["memory_revision"])
        return {
            "record_id": record_id,
            "status": "removed_from_saved_memory",
            "chat_history_changed": False,
            "physical_erasure_claimed": False,
            "memory_revision": revision,
        }

    def delete_memory_record_by_id(
        self,
        record_id: str,
        *,
        expected_revision: int,
        request_id: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Commit logical removal, audit and revision CAS in one transaction."""

        require_uuid4(record_id)
        require_uuid4(request_id)
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("memory revision is invalid")
        prepared_session = self._memory_epoch_session
        selected = now or utc_now()
        with self._transaction() as connection:
            return self._delete_memory_record_by_id_tx(
                connection,
                record_id,
                expected_revision=expected_revision,
                request_id=request_id,
                session=prepared_session,
                now=selected,
            )

    def record_explicit_memory_delete(
        self,
        *,
        task_id: str,
        conversation_id: str,
        epoch: int,
        target_sha256: str,
        record_id: str | None,
        expected_envelope_sha256: str | None,
        matched_count: int,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Atomically audit and apply one exact explicit retrospective deletion."""

        if matched_count < 0 or isinstance(matched_count, bool):
            raise ValueError("memory delete match count is invalid")
        if (record_id is None) != (expected_envelope_sha256 is None):
            raise ValueError("memory delete record binding is incomplete")
        if (matched_count == 1) != (record_id is not None):
            raise ValueError("memory delete record binding is inconsistent")
        selected = now or utc_now()
        prepared_session = self._memory_epoch_session
        with self._transaction() as connection:
            task = self._guarded_task_tx(
                connection, task_id, epoch, (TaskState.COMPLETED,)
            )
            if task["conversation_id"] != conversation_id:
                raise StaleEpochError()
            principal_id = self._principal_for_conversation_tx(
                connection, conversation_id
            )
            existing = connection.execute(
                """SELECT payload_json FROM task_events
                   WHERE task_id=? AND event_type LIKE 'memory.delete.%'""",
                (task_id,),
            ).fetchone()
            if existing is not None:
                previous = json.loads(existing["payload_json"])
                if previous.get("target_sha256") != target_sha256:
                    raise OrchestratorError("idempotency_conflict", status=409)
                return dict(previous)
            deleted_count = 0
            status = "empty" if matched_count == 0 else "ambiguous"
            if record_id is not None:
                if prepared_session is None:
                    raise MemoryStorageUnavailable(
                        "memory_crypto_session_unavailable"
                    )
                self._require_current_memory_session_tx(
                    connection,
                    principal_id=principal_id,
                    session=prepared_session,
                )
                row = connection.execute(
                    self._memory_row_select_sql(
                        where=(
                            "r.record_id=? AND r.principal_id=? "
                            "AND r.envelope_sha256=?"
                        )
                    ),
                    (record_id, principal_id, expected_envelope_sha256),
                ).fetchone()
                if row is None:
                    raise OrchestratorError("memory_delete_target_changed", status=409)
                self._project_verified_memory_row_tx(
                    connection, prepared_session, dict(row)
                )
                if not self._delete_saved_memory_row_tx(
                    connection,
                    dict(row),
                    reason="explicit_forget",
                    action="EXPLICIT_FORGOT",
                    actor_kind="explicit_command",
                    actor_id=task_id,
                    now=selected,
                    encrypted=True,
                ):
                    raise OrchestratorError("memory_delete_target_changed", status=409)
                deleted_count = 1
                status = "completed"
            payload = {
                "status": status,
                "target_sha256": target_sha256,
                "matched_count": matched_count,
                "deleted_count": deleted_count,
                "authoritative_state_changed": deleted_count == 1,
                "selection": "exact_normalized_content",
            }
            self._append_task_event_tx(
                connection,
                task_id,
                epoch,
                f"memory.delete.{status}",
                payload,
                selected,
            )
            return dict(payload)

    def memory_delete_audit(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT epoch,event_type,payload_json,created_at
                   FROM task_events
                   WHERE task_id=? AND event_type LIKE 'memory.delete.%'""",
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

    def memory_record_count(self, conversation_id: str) -> int:
        with self._lock:
            return int(
                self._connection.execute(
                    """SELECT COUNT(*) FROM conversation_principals cp
                       JOIN memory_records r USING(principal_id)
                       WHERE cp.conversation_id=? AND r.expires_at>?""",
                    (conversation_id, utc_text(utc_now())),
                ).fetchone()[0]
            )

    def memory_principal_usage(self, conversation_id: str) -> dict[str, int]:
        with self._lock:
            row = self._connection.execute(
                """SELECT u.active_record_count,u.active_content_bytes
                   FROM conversation_principals cp
                   JOIN memory_principal_usage u USING(principal_id)
                   WHERE cp.conversation_id=?""",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise OrchestratorError("not_found", status=404)
        return {
            "active_record_count": int(row["active_record_count"]),
            "active_content_bytes": int(row["active_content_bytes"]),
        }

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

    def _create_conversation_tx(
        self,
        connection: sqlite3.Connection,
        expected_epoch: int,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        stamp = utc_text(now)
        conversation_id = new_uuid4()
        self._maybe_sweep_expired_memory_tx(connection, now=now)
        current = connection.execute(
            "SELECT * FROM conversations WHERE state='ACTIVE'"
        ).fetchone()
        if current is None:
            if expected_epoch != 0:
                raise StaleEpochError()
        else:
            if current["current_epoch"] != expected_epoch:
                raise StaleEpochError()
            if self._active_task_tx(connection, current["conversation_id"]) is not None:
                raise OrchestratorError(
                    "conversation_busy", status=409, retryable=True
                )
            connection.execute(
                "UPDATE conversations SET state='CLOSED',updated_at=? "
                "WHERE conversation_id=? AND current_epoch=?",
                (stamp, current["conversation_id"], expected_epoch),
            )
            connection.execute(
                """UPDATE outbox SET state='NON_REPLAYABLE',claimed_by=NULL,
                          claim_expires_at=NULL
                   WHERE conversation_id=?
                     AND state IN ('REPLAYABLE','CLAIMED')""",
                (current["conversation_id"],),
            )
        connection.execute(
            """INSERT INTO conversations(
                   conversation_id,state,current_epoch,revision,created_at,updated_at
               ) VALUES(?,'ACTIVE',0,0,?,?)""",
            (conversation_id, stamp, stamp),
        )
        self._bind_conversation_principal_tx(connection, conversation_id, stamp)
        return {"conversation_id": conversation_id, "epoch": 0, "state": "ACTIVE"}

    def create_conversation(
        self, expected_epoch: int, *, now: datetime | None = None
    ) -> dict[str, Any]:
        selected = now or utc_now()
        with self._transaction() as connection:
            return self._create_conversation_tx(
                connection, expected_epoch, now=selected
            )

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
            self._maybe_sweep_expired_memory_tx(connection, now=selected)
            existing = connection.execute("SELECT * FROM inbox WHERE event_id=?", (event_id,)).fetchone()
            conversation = connection.execute("SELECT * FROM conversations WHERE state='ACTIVE'").fetchone()
            if conversation is None:
                conversation_id = new_uuid4()
                stamp = utc_text(selected)
                connection.execute(
                    "INSERT INTO conversations(conversation_id,state,current_epoch,revision,created_at,updated_at) VALUES(?,'ACTIVE',0,0,?,?)",
                    (conversation_id, stamp, stamp),
                )
                self._bind_conversation_principal_tx(
                    connection, conversation_id, stamp
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

    @staticmethod
    def _explicit_memory_command_provenance(
        *,
        task_id: str,
        turn_id: str,
        user_text: str,
        response_text: str,
    ) -> ContentProvenanceV1:
        """Bind a code-owned memory reply only to its current input and template."""

        return ContentProvenanceV1.derive(
            (
                ContentProvenanceV1.source(
                    ContentSourceKind.USER_INPUT,
                    turn_id,
                    user_text,
                ),
                ContentProvenanceV1.source(
                    ContentSourceKind.CODE_TEMPLATE,
                    task_id,
                    response_text,
                ),
            )
        )

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
        if any(value.startswith("memory.command.") for value in event_types):
            inputs.append(
                ContentProvenanceV1.source(
                    ContentSourceKind.CODE_TEMPLATE, task_id, assistant_text
                )
            )
            return ContentProvenanceV1.derive(inputs)
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

    def _cancel_task_request_tx(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        expected_epoch: int,
        reason: ReasonCode,
        *,
        terminal_state: TaskState = TaskState.CANCELLED,
        now: datetime,
    ) -> dict[str, Any]:
        row = connection.execute(
            """SELECT t.*,c.current_epoch
               FROM tasks t JOIN conversations c USING(conversation_id)
               WHERE task_id=? AND t.epoch=? AND c.current_epoch=?
                 AND c.state='ACTIVE'""",
            (task_id, expected_epoch, expected_epoch),
        ).fetchone()
        if row is None:
            raise StaleEpochError()
        if TaskState(row["state"]) in TERMINAL_STATES:
            raise InvalidStateError()
        new_epoch = expected_epoch + 1
        changed = connection.execute(
            "UPDATE conversations SET current_epoch=?,updated_at=? "
            "WHERE conversation_id=? AND current_epoch=?",
            (new_epoch, utc_text(now), row["conversation_id"], expected_epoch),
        )
        if changed.rowcount != 1:
            raise StaleEpochError()
        self._cancel_task_tx(
            connection,
            row,
            guard_epoch=new_epoch,
            state=terminal_state,
            reason=reason,
            now=now,
        )
        return {
            "task_id": task_id,
            "previous_epoch": expected_epoch,
            "epoch": new_epoch,
            "state": terminal_state.value,
        }

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
            return self._cancel_task_request_tx(
                connection,
                task_id,
                expected_epoch,
                reason,
                terminal_state=terminal_state,
                now=selected,
            )

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

    def _decide_approval_tx(
        self,
        connection: sqlite3.Connection,
        approval_id: str,
        expected_epoch: int,
        decision: str,
        *,
        client_connected: bool,
        now: datetime,
    ) -> dict[str, Any]:
        if decision not in {"approve", "deny"}:
            raise OrchestratorError("invalid_request", status=400)
        approval = connection.execute(
            "SELECT * FROM approvals WHERE approval_id=?", (approval_id,)
        ).fetchone()
        if approval is None:
            raise OrchestratorError("not_found", status=404)
        if approval["epoch"] != expected_epoch:
            raise StaleEpochError()
        if approval["status"] != "PENDING":
            raise InvalidStateError()
        task = self._guarded_task_tx(
            connection,
            approval["task_id"],
            expected_epoch,
            (TaskState.AWAITING_APPROVAL,),
        )
        approval_status = (
            ApprovalStatus.APPROVED
            if decision == "approve"
            else ApprovalStatus.DENIED
        )
        target = (
            TaskState.EXECUTING
            if client_connected
            else TaskState.DECISION_WAITING_RECONNECT
        )
        reconnect_deadline = (
            None if client_connected else utc_text(now + self.disconnect_ttl)
        )
        connection.execute(
            "UPDATE approvals SET status=?,decided_at=? "
            "WHERE approval_id=? AND status='PENDING'",
            (approval_status.value, utc_text(now), approval_id),
        )
        if approval["kind"] == "research":
            connection.execute(
                """UPDATE research_runs SET stage=?,updated_at=?
                   WHERE approval_id=? AND stage='AWAITING_APPROVAL'""",
                (
                    "APPROVED"
                    if approval_status == ApprovalStatus.APPROVED
                    else "DENIED",
                    utc_text(now),
                    approval_id,
                ),
            )
        connection.execute(
            """UPDATE tasks SET state=?,active_started_at=?,
                      disconnect_expires_at=?,updated_at=?
               WHERE task_id=? AND epoch=? AND state='AWAITING_APPROVAL'""",
            (
                target.value,
                utc_text(now) if client_connected else None,
                reconnect_deadline,
                utc_text(now),
                task["task_id"],
                expected_epoch,
            ),
        )
        turn = connection.execute(
            "SELECT effects_json FROM canonical_turns WHERE task_id=?",
            (task["task_id"],),
        ).fetchone()
        effects = json.loads(turn["effects_json"])
        effects.append(
            {
                "kind": (
                    "ToolDenied"
                    if approval_status == ApprovalStatus.DENIED
                    else approval["kind"]
                ),
                "name": approval["name"],
                "status": approval_status.value,
                "arguments_sha256": approval["arguments_sha256"],
                "result_sha256": "",
            }
        )
        connection.execute(
            "UPDATE canonical_turns SET effects_json=? WHERE task_id=?",
            (canonical_json(effects), task["task_id"]),
        )
        self._append_task_event_tx(
            connection,
            task["task_id"],
            expected_epoch,
            target.value,
            {"approval_id": approval_id, "decision": decision},
            now,
        )
        return {
            "approval_id": approval_id,
            "task_id": task["task_id"],
            "epoch": expected_epoch,
            "approval_status": approval_status.value,
            "task_state": target.value,
        }

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
        expired = False
        result: dict[str, Any] | None = None
        with self._transaction() as connection:
            before = connection.execute(
                "SELECT expires_at,status FROM approvals WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            if (
                before is not None
                and before["status"] == "PENDING"
                and parse_utc(before["expires_at"]) <= selected
            ):
                self._expire_approval_tx(connection, approval_id, now=selected)
                expired = True
            else:
                result = self._decide_approval_tx(
                    connection,
                    approval_id,
                    expected_epoch,
                    decision,
                    client_connected=client_connected,
                    now=selected,
                )
        if expired:
            raise OrchestratorError("approval_expired", status=410)
        assert result is not None
        return result

    def _expire_approval_tx(
        self,
        connection: sqlite3.Connection,
        approval_id: str,
        *,
        now: datetime,
    ) -> bool:
        approval = connection.execute(
            "SELECT * FROM approvals WHERE approval_id=? "
            "AND status='PENDING' AND expires_at<=?",
            (approval_id, utc_text(now)),
        ).fetchone()
        if approval is None:
            return False
        task = connection.execute(
            """SELECT t.*,c.current_epoch
               FROM tasks t JOIN conversations c USING(conversation_id)
               WHERE t.task_id=? AND t.epoch=? AND t.tombstoned=0
                 AND c.current_epoch=? AND c.state='ACTIVE'""",
            (approval["task_id"], approval["epoch"], approval["epoch"]),
        ).fetchone()
        if task is None:
            connection.execute(
                "UPDATE approvals SET status='EXPIRED',decided_at=? "
                "WHERE approval_id=?",
                (utc_text(now), approval_id),
            )
            return True
        new_epoch = int(approval["epoch"]) + 1
        changed = connection.execute(
            "UPDATE conversations SET current_epoch=?,updated_at=? "
            "WHERE conversation_id=? AND current_epoch=?",
            (new_epoch, utc_text(now), task["conversation_id"], approval["epoch"]),
        )
        if changed.rowcount != 1:
            raise StaleEpochError()
        connection.execute(
            "UPDATE approvals SET status='EXPIRED',decided_at=? "
            "WHERE approval_id=?",
            (utc_text(now), approval_id),
        )
        self._cancel_task_tx(
            connection,
            task,
            guard_epoch=new_epoch,
            state=TaskState.CANCELLED,
            reason=ReasonCode.APPROVAL_EXPIRED,
            now=now,
        )
        return True

    def expire_approval(
        self, approval_id: str, *, now: datetime | None = None
    ) -> bool:
        selected = now or utc_now()
        with self._transaction() as connection:
            return self._expire_approval_tx(
                connection, approval_id, now=selected
            )

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
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT t.* FROM tasks t JOIN conversations c USING(conversation_id)
                   WHERE t.tombstoned=0 AND t.epoch=c.current_epoch AND c.state='ACTIVE'
                   AND t.state NOT IN ('COMPLETED','CANCELLED','INTERRUPTED','BUDGET_EXHAUSTED','FAILED')""",
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
                self._cancel_task_request_tx(
                    connection, row["task_id"], row["epoch"], reason,
                    terminal_state=terminal, now=selected,
                )
                count += 1
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

    def record_voice_playback_terminal(
        self,
        *,
        event_id: str,
        semantic_sha256: str,
        response_id: str,
        terminal_type: str,
        client_sent_at: str,
        now: datetime | None = None,
    ) -> str:
        """Persist content-free playback evidence before retiring delivery.

        The receipt deliberately stores no payload, transcript, response text,
        locale, unit index, session identifier, or connection identifier.  An
        exact replay is idempotent; a changed event identity or a second
        terminal for the same response fails closed.
        """

        if terminal_type not in {"playback.completed", "playback.interrupted"}:
            raise ValueError("playback terminal type is invalid")
        if (
            not isinstance(semantic_sha256, str)
            or len(semantic_sha256) != 64
            or any(character not in "0123456789abcdef" for character in semantic_sha256)
        ):
            raise ValueError("playback semantic digest is invalid")
        try:
            parse_utc(client_sent_at)
        except (TypeError, ValueError) as exc:
            raise ValueError("playback client timestamp is invalid") from exc
        selected = now or utc_now()
        with self._transaction() as connection:
            task = connection.execute(
                "SELECT task_id,epoch,state FROM tasks WHERE response_id=?",
                (response_id,),
            ).fetchone()
            if task is None:
                return "stale"
            existing_event = connection.execute(
                """SELECT semantic_sha256,task_id,epoch,response_id,terminal_type,
                          client_sent_at
                   FROM voice_playback_receipts WHERE event_id=?""",
                (event_id,),
            ).fetchone()
            expected = (
                semantic_sha256,
                task["task_id"],
                int(task["epoch"]),
                response_id,
                terminal_type,
            )
            if existing_event is not None:
                observed = (
                    existing_event["semantic_sha256"],
                    existing_event["task_id"],
                    int(existing_event["epoch"]),
                    existing_event["response_id"],
                    existing_event["terminal_type"],
                )
                if observed != expected:
                    raise OrchestratorError(
                        "protocol.conflicting_duplicate", status=409
                    )
                return "duplicate"
            if connection.execute(
                "SELECT 1 FROM voice_playback_receipts WHERE response_id=?",
                (response_id,),
            ).fetchone() is not None:
                raise OrchestratorError(
                    "protocol.conflicting_duplicate", status=409
                )
            if terminal_type == "playback.completed":
                speech_done = connection.execute(
                    """SELECT 1 FROM outbox
                       WHERE response_id=? AND event_type='response.speech.done'""",
                    (response_id,),
                ).fetchone()
                if task["state"] != TaskState.COMPLETED.value or speech_done is None:
                    raise InvalidStateError()
            connection.execute(
                """INSERT INTO voice_playback_receipts(
                   event_id,semantic_sha256,task_id,epoch,response_id,
                   terminal_type,client_sent_at,received_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    event_id,
                    semantic_sha256,
                    task["task_id"],
                    int(task["epoch"]),
                    response_id,
                    terminal_type,
                    client_sent_at,
                    utc_text(selected),
                ),
            )
            self._append_task_event_tx(
                connection,
                task["task_id"],
                int(task["epoch"]),
                "voice.playback.completed"
                if terminal_type == "playback.completed"
                else "voice.playback.interrupted",
                {},
                selected,
            )
            connection.execute(
                """UPDATE outbox SET state='NON_REPLAYABLE',claimed_by=NULL,
                          claim_expires_at=NULL
                   WHERE response_id=? AND state IN ('REPLAYABLE','CLAIMED')""",
                (response_id,),
            )
            return "accepted"

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

    def recent_canonical_turns(
        self, conversation_id: str, *, limit_pairs: int = 8
    ) -> tuple[CanonicalTurnV1, ...]:
        """Return a bounded chronological window of completed canonical pairs."""

        if (
            isinstance(limit_pairs, bool)
            or not isinstance(limit_pairs, int)
            or not 1 <= limit_pairs <= 8
        ):
            raise ValueError("canonical history limit must be between 1 and 8")
        with self._lock:
            rows = self._connection.execute(
                """SELECT turn_id,user_text,user_language,assistant_text,effects_json
                   FROM canonical_turns
                   WHERE conversation_id=? AND status='COMPLETED'
                   ORDER BY created_at DESC,turn_id DESC LIMIT ?""",
                (conversation_id, limit_pairs),
            ).fetchall()
        return tuple(
            CanonicalTurnV1(
                row["turn_id"],
                row["user_text"],
                row["user_language"],
                row["assistant_text"],
                tuple(
                    CanonicalEffectV1(**item)
                    for item in json.loads(row["effects_json"])
                ),
            )
            for row in reversed(rows)
        )

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

    @staticmethod
    def _lookup_command_tx(
        connection: sqlite3.Connection,
        key: str,
        method: str,
        path: str,
        request_sha256: str,
        expected_epoch: int,
        *,
        now: datetime,
    ) -> tuple[int, str | None] | None:
        stamp = utc_text(now)
        row = connection.execute(
            "SELECT * FROM idempotency_commands WHERE idempotency_key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] <= stamp:
            connection.execute(
                "DELETE FROM idempotency_commands "
                "WHERE idempotency_key=? AND expires_at<=?",
                (key, stamp),
            )
            return None
        if (
            row["method"] != method
            or row["canonical_path"] != path
            or row["request_sha256"] != request_sha256
            or row["expected_epoch"] != expected_epoch
        ):
            raise OrchestratorError("idempotency_conflict", status=409)
        return row["response_status"], row["response_json"]

    @staticmethod
    def _store_command_tx(
        connection: sqlite3.Connection,
        key: str,
        method: str,
        path: str,
        request_sha256: str,
        expected_epoch: int,
        status: int,
        response_json: str | None,
        *,
        now: datetime,
    ) -> None:
        connection.execute(
            """INSERT INTO idempotency_commands(
                   idempotency_key,method,canonical_path,request_sha256,
                   expected_epoch,response_status,response_json,created_at,
                   expires_at
               ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                key,
                method,
                path,
                request_sha256,
                expected_epoch,
                status,
                response_json,
                utc_text(now),
                utc_text(now + timedelta(hours=24)),
            ),
        )

    @staticmethod
    def _control_object(body: Any, fields: set[str]) -> dict[str, Any]:
        if not isinstance(body, dict) or set(body) != fields:
            raise OrchestratorError("invalid_request", status=400)
        return body

    @staticmethod
    def _control_error(error: OrchestratorError) -> dict[str, Any]:
        return {
            "error": {
                "code": error.code,
                "retryable": error.retryable,
            }
        }

    @classmethod
    def _require_durable_memory_delete_binding_tx(
        cls,
        connection: sqlite3.Connection,
        *,
        idempotency_key: str,
        request: ControlMutationRequest,
        body: Any,
        expected_epoch: int,
    ) -> None:
        """Keep a durable memory-delete key bound after its receipt expires."""

        tombstone = connection.execute(
            "SELECT * FROM memory_record_tombstones "
            "WHERE delete_request_id=?",
            (idempotency_key,),
        ).fetchone()
        if tombstone is None:
            return
        exact_memory_delete = (
            request.kind is ControlMutationKind.DELETE_MEMORY_RECORD
            and request.resource_id == tombstone["record_id"]
            and body is None
            and not isinstance(expected_epoch, bool)
            and isinstance(expected_epoch, int)
            and 0 <= expected_epoch <= SQLITE_MAX_INTEGER
            and int(tombstone["memory_revision"]) == expected_epoch + 1
            and tombstone["reason"] == "control_delete"
            and tombstone["principal_id"]
            == cls._local_owner_principal_tx(connection)
        )
        if not exact_memory_delete:
            raise OrchestratorError("idempotency_conflict", status=409)

    def _apply_control_mutation_tx(
        self,
        connection: sqlite3.Connection,
        request: ControlMutationRequest,
        body: Any,
        expected_epoch: int,
        *,
        idempotency_key: str,
        client_connected: bool,
        now: datetime,
    ) -> tuple[int, Any | None]:
        if request.kind is ControlMutationKind.CREATE_CONVERSATION:
            self._control_object(body, set())
            return 201, self._create_conversation_tx(
                connection, expected_epoch, now=now
            )
        if request.kind is ControlMutationKind.DELETE_CONVERSATION:
            if body is not None:
                raise OrchestratorError("invalid_request", status=400)
            assert request.resource_id is not None
            self._delete_conversation_tx(
                connection,
                require_uuid4(request.resource_id),
                expected_epoch,
                now=now,
            )
            return 204, None
        if request.kind is ControlMutationKind.DELETE_MEMORY_RECORD:
            if body is not None:
                prior_delete = connection.execute(
                    "SELECT record_id FROM memory_record_tombstones "
                    "WHERE delete_request_id=?",
                    (idempotency_key,),
                ).fetchone()
                if prior_delete is not None:
                    raise OrchestratorError("idempotency_conflict", status=409)
                raise OrchestratorError("invalid_request", status=400)
            if (
                isinstance(expected_epoch, bool)
                or not isinstance(expected_epoch, int)
                or expected_epoch < 0
                or expected_epoch > SQLITE_MAX_INTEGER
            ):
                raise OrchestratorError("invalid_request", status=400)
            assert request.resource_id is not None
            try:
                return 200, self._delete_memory_record_by_id_tx(
                    connection,
                    require_uuid4(request.resource_id),
                    expected_revision=expected_epoch,
                    request_id=idempotency_key,
                    session=self._memory_epoch_session,
                    now=now,
                )
            except OrchestratorError:
                raise
            except (
                MemoryCryptoError,
                MemoryStorageUnavailable,
                RuntimeError,
            ) as error:
                raise OrchestratorError(
                    "memory_storage_unavailable",
                    status=503,
                    retryable=True,
                ) from error
        if request.kind is ControlMutationKind.CANCEL_TASK:
            payload = self._control_object(body, {"reason_code"})
            if payload["reason_code"] != ReasonCode.USER_CANCELLED.value:
                raise OrchestratorError("invalid_request", status=400)
            assert request.resource_id is not None
            return 200, self._cancel_task_request_tx(
                connection,
                require_uuid4(request.resource_id),
                expected_epoch,
                ReasonCode.USER_CANCELLED,
                now=now,
            )
        if request.kind is ControlMutationKind.DECIDE_APPROVAL:
            payload = self._control_object(body, {"decision"})
            decision = payload["decision"]
            if decision not in {"approve", "deny"}:
                raise OrchestratorError("invalid_request", status=400)
            assert request.resource_id is not None
            approval_id = require_uuid4(request.resource_id)
            approval = connection.execute(
                "SELECT kind,status,expires_at FROM approvals "
                "WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            if approval is not None and approval["kind"] == "cloud_model":
                raise OrchestratorError("forbidden", status=403)
            if approval is not None and approval["kind"] == "tool":
                raise OrchestratorError(
                    "tool.approval_route_required", status=409
                )
            if (
                approval is not None
                and approval["status"] == "PENDING"
                and parse_utc(approval["expires_at"]) <= now
            ):
                self._expire_approval_tx(connection, approval_id, now=now)
                error = OrchestratorError("approval_expired", status=410)
                return error.status, self._control_error(error)
            return 200, self._decide_approval_tx(
                connection,
                approval_id,
                expected_epoch,
                decision,
                client_connected=client_connected,
                now=now,
            )
        if request.kind is ControlMutationKind.METHOD_NOT_ALLOWED:
            raise OrchestratorError("method_not_allowed", status=405)
        if request.kind is ControlMutationKind.NOT_FOUND:
            raise OrchestratorError("not_found", status=404)
        raise RuntimeError("unreachable control mutation")

    @staticmethod
    def _validate_control_mutation_binding(
        method: str,
        path: str,
        request: ControlMutationRequest,
    ) -> None:
        if request.kind is ControlMutationKind.CREATE_CONVERSATION:
            expected = ("POST", "/v1/conversations")
        elif request.kind in {
            ControlMutationKind.DELETE_CONVERSATION,
            ControlMutationKind.DELETE_MEMORY_RECORD,
            ControlMutationKind.CANCEL_TASK,
            ControlMutationKind.DECIDE_APPROVAL,
        }:
            assert request.resource_id is not None
            try:
                canonical_id = require_uuid4(request.resource_id)
            except OrchestratorError as error:
                raise ValueError("control mutation HTTP binding is invalid") from error
            if canonical_id != request.resource_id:
                raise ValueError("control mutation HTTP binding is invalid")
            if request.kind is ControlMutationKind.DELETE_CONVERSATION:
                expected = (
                    "DELETE",
                    f"/v1/conversations/{canonical_id}",
                )
            elif request.kind is ControlMutationKind.DELETE_MEMORY_RECORD:
                expected = ("DELETE", f"/v1/memory/records/{canonical_id}")
            elif request.kind is ControlMutationKind.CANCEL_TASK:
                expected = ("POST", f"/v1/tasks/{canonical_id}/cancel")
            else:
                expected = (
                    "POST",
                    f"/v1/approvals/{canonical_id}/decision",
                )
        else:
            return
        if (method, path) != expected:
            raise ValueError("control mutation HTTP binding is invalid")

    def execute_control_mutation(
        self,
        key: str,
        method: str,
        path: str,
        body: Any,
        expected_epoch: int,
        request: ControlMutationRequest,
        *,
        client_connected: bool,
        now: datetime | None = None,
    ) -> tuple[int, str | None]:
        """Commit one same-SQLite mutation and its exact receipt atomically."""

        if not isinstance(request, ControlMutationRequest):
            raise TypeError("control mutation request is invalid")
        if not isinstance(client_connected, bool):
            raise TypeError("control client state is invalid")
        if request.kind is ControlMutationKind.DELETE_MEMORY_RECORD:
            require_uuid4(key)
            if (
                isinstance(expected_epoch, bool)
                or not isinstance(expected_epoch, int)
                or expected_epoch < 0
                or expected_epoch > SQLITE_MAX_INTEGER
            ):
                raise OrchestratorError("invalid_request", status=400)
        self._validate_control_mutation_binding(method, path, request)
        selected = now or utc_now()
        request_sha256 = canonical_sha256(body)
        should_checkpoint = False
        with self._transaction() as connection:
            replay = self._lookup_command_tx(
                connection,
                key,
                method,
                path,
                request_sha256,
                expected_epoch,
                now=selected,
            )
            if replay is not None:
                return replay
            self._require_durable_memory_delete_binding_tx(
                connection,
                idempotency_key=key,
                request=request,
                body=body,
                expected_epoch=expected_epoch,
            )
            connection.execute("SAVEPOINT control_mutation")
            try:
                status, result = self._apply_control_mutation_tx(
                    connection,
                    request,
                    body,
                    expected_epoch,
                    idempotency_key=key,
                    client_connected=client_connected,
                    now=selected,
                )
            except OrchestratorError as error:
                connection.execute("ROLLBACK TO control_mutation")
                connection.execute("RELEASE control_mutation")
                status = error.status
                result = self._control_error(error)
            else:
                connection.execute("RELEASE control_mutation")
            response_json = (
                None if result is None else canonical_json(result)
            )
            self._store_command_tx(
                connection,
                key,
                method,
                path,
                request_sha256,
                expected_epoch,
                status,
                response_json,
                now=selected,
            )
            should_checkpoint = (
                request.kind is ControlMutationKind.DELETE_CONVERSATION
                and status == 204
            )
        if should_checkpoint:
            # Logical deletion and its receipt are already durable. WAL
            # compaction is deliberately outside that transaction.
            with suppress(sqlite3.Error):
                self.checkpoint()
        return status, response_json

    def lookup_command(
        self,
        key: str,
        method: str,
        path: str,
        body: Any,
        expected_epoch: int,
        *,
        now: datetime | None = None,
    ) -> tuple[int, str | None] | None:
        selected = now or utc_now()
        with self._transaction() as connection:
            return self._lookup_command_tx(
                connection,
                key,
                method,
                path,
                canonical_sha256(body),
                expected_epoch,
                now=selected,
            )

    def store_command(
        self,
        key: str,
        method: str,
        path: str,
        body: Any,
        expected_epoch: int,
        status: int,
        response_json: str | None,
        *,
        now: datetime | None = None,
    ) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            self._store_command_tx(
                connection,
                key,
                method,
                path,
                canonical_sha256(body),
                expected_epoch,
                status,
                response_json,
                now=selected,
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
            expires_at = utc_text(selected + self.delivery_ttl)
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
            used = self._cloud_monthly_used_tx(connection, utc_month)
            if used + reserved_microusd > MONTHLY_HARD_CAP_MICROUSD:
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

    @staticmethod
    def _cloud_monthly_used_tx(connection: sqlite3.Connection, utc_month: str) -> int:
        used = connection.execute(
            """SELECT COALESCE(SUM(CASE
                   WHEN state='RECONCILED' THEN actual_microusd
                   WHEN state IN ('RESERVED','DISPATCHED','AMBIGUOUS') THEN reserved_microusd
                   ELSE 0 END),0)
               FROM cloud_cost_reservations WHERE utc_month=?""",
            (utc_month,),
        ).fetchone()[0]
        retired = connection.execute(
            "SELECT value FROM schema_meta WHERE key=?",
            ("cloud_retired_microusd:" + utc_month,),
        ).fetchone()
        return int(used) + (int(retired[0]) if retired is not None else 0)

    def cloud_monthly_remaining(self, utc_month: str) -> int:
        with self._read_transaction() as connection:
            return max(0, MONTHLY_HARD_CAP_MICROUSD - self._cloud_monthly_used_tx(connection, utc_month))

    def _delete_conversation_tx(
        self,
        connection: sqlite3.Connection,
        conversation_id: str,
        expected_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None:
        row = connection.execute(
            "SELECT * FROM conversations "
            "WHERE conversation_id=? AND state!='DELETED'",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise OrchestratorError("not_found", status=404)
        if row["current_epoch"] != expected_epoch:
            raise StaleEpochError()
        if self._active_task_tx(connection, conversation_id) is not None:
            raise OrchestratorError(
                "conversation_busy", status=409, retryable=True
            )
        principal_id = self._principal_for_conversation_tx(
            connection, conversation_id
        )
        memory_state = self._memory_crypto_state_tx(connection, principal_id)
        if memory_state["storage_state"] == "MIGRATING":
            raise OrchestratorError(
                "memory_migration_in_progress", status=409, retryable=True
            )
        encrypted_memory = memory_state["storage_state"] == "ENCRYPTED"
        memory_table = (
            "memory_records"
            if encrypted_memory
            else "memory_records_v9_legacy"
        )
        memory_rows = connection.execute(
            f"SELECT * FROM {memory_table} WHERE conversation_id=? "
            "ORDER BY record_id",
            (conversation_id,),
        ).fetchall()
        selected = now or utc_now()
        for memory_row in memory_rows:
            if not self._delete_saved_memory_row_tx(
                connection,
                memory_row,
                reason="source_conversation_delete",
                action="SOURCE_CONVERSATION_DELETED",
                actor_kind="conversation_lifecycle",
                actor_id=conversation_id,
                now=selected,
                encrypted=encrypted_memory,
            ):
                raise OrchestratorError(
                    "memory_delete_target_changed", status=409
                )
        if not encrypted_memory and memory_rows:
            usage = connection.execute(
                """SELECT COUNT(*) AS count,COALESCE(SUM(content_bytes),0) AS bytes
                   FROM memory_records_v9_legacy WHERE principal_id=?""",
                (principal_id,),
            ).fetchone()
            connection.execute(
                """UPDATE memory_principal_usage
                   SET active_record_count=?,active_content_bytes=?,updated_at=?
                   WHERE principal_id=?""",
                (
                    int(usage["count"]),
                    int(usage["bytes"]),
                    utc_text(selected),
                    principal_id,
                ),
            )
        # Preserve only monthly charges when deleting their conversation records.
        connection.execute(
            """INSERT INTO schema_meta(key,value)
               SELECT 'cloud_retired_microusd:' || r.utc_month,
                      CAST(SUM(CASE WHEN r.state='RECONCILED' THEN r.actual_microusd
                                    ELSE r.reserved_microusd END) AS TEXT)
               FROM cloud_cost_reservations r JOIN tasks t USING(task_id)
               WHERE t.conversation_id=? AND r.state IN ('DISPATCHED','AMBIGUOUS','RECONCILED')
               GROUP BY r.utc_month
               ON CONFLICT(key) DO UPDATE SET value=CAST(
                   CAST(schema_meta.value AS INTEGER) + CAST(excluded.value AS INTEGER) AS TEXT)""",
            (conversation_id,),
        )
        task_filter = "SELECT task_id FROM tasks WHERE conversation_id=?"
        connection.execute(
            f"""DELETE FROM model_request_provenance
                WHERE reservation_id IN (
                    SELECT reservation_id FROM reservations
                    WHERE task_id IN ({task_filter})
                )""",
            (conversation_id,),
        )
        connection.execute(
            f"""DELETE FROM cloud_extension_inbox WHERE approval_id IN (
               SELECT approval_id FROM approvals
               WHERE task_id IN ({task_filter})
               )""",
            (conversation_id,),
        )
        connection.execute(
            f"DELETE FROM research_documents "
            f"WHERE task_id IN ({task_filter})",
            (conversation_id,),
        )
        connection.execute(
            f"DELETE FROM research_runs WHERE task_id IN ({task_filter})",
            (conversation_id,),
        )
        for table in (
            "scheduler_task_links",
            "voice_playback_receipts",
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
                f"DELETE FROM {table} WHERE task_id IN ({task_filter})",
                (conversation_id,),
            )
        connection.execute(
            "DELETE FROM cloud_extension_outbox WHERE conversation_id=?",
            (conversation_id,),
        )
        connection.execute(
            """DELETE FROM outbox_attempts WHERE event_id IN (
                   SELECT event_id FROM outbox WHERE conversation_id=?
               )""",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM outbox WHERE conversation_id=?", (conversation_id,)
        )
        connection.execute(
            """DELETE FROM content_provenance
               WHERE entity_kind='canonical_turn' AND entity_id IN (
                   SELECT turn_id FROM canonical_turns WHERE conversation_id=?
               )""",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM canonical_turns WHERE conversation_id=?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM tasks WHERE conversation_id=?", (conversation_id,)
        )
        connection.execute(
            "DELETE FROM conversations "
            "WHERE conversation_id=? AND current_epoch=?",
            (conversation_id, expected_epoch),
        )

    def delete_conversation(
        self,
        conversation_id: str,
        expected_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None:
        selected = now or utc_now()
        with self._transaction() as connection:
            self._delete_conversation_tx(
                connection, conversation_id, expected_epoch, now=selected
            )
        self.checkpoint()

    def checkpoint(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
