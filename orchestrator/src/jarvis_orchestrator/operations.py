"""Operational safety primitives for a long-running local Orchestrator.

This module deliberately exposes only aggregate, content-free diagnostics.
No transcript, prompt, response text, UUID, approval argument, memory value or
provider payload is emitted by any report or log record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import math
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from .memory_backup import (
    MAX_MEMORY_BACKUP_MANIFEST_BYTES,
    MEMORY_BACKUP_MANIFEST_SUFFIX,
    EncryptedMemoryBackupStateV1,
    MemoryBackupEpochV1,
    MemoryBackupError,
    MemoryBackupManifestV1,
    MemoryBackupProviderAttestor,
    verify_memory_backup_provider,
)
from .private_state import (
    PRIVATE_FILE_MODE,
    PrivateStateError,
    _open_private_fd,
    absolute_path_no_resolve,
    assert_private_database,
    assert_private_log_family,
    assert_private_regular_file,
    ensure_private_directory,
    open_private_append,
    open_private_lock,
    unlink_private_file,
)


class OperationalError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating handler whose newly created streams are exact owner-only files."""

    def _open(self) -> Any:
        descriptor = open_private_append(self.baseFilename)
        return os.fdopen(
            descriptor,
            mode="a",
            encoding=self.encoding,
            errors=self.errors,
        )


class ContentFreeOperationalLog:
    """Bounded JSONL log accepting only codes and scalar counters."""

    def __init__(
        self,
        path: str | Path,
        *,
        maximum_bytes: int = 1_048_576,
        backup_count: int = 3,
    ) -> None:
        try:
            selected = assert_private_log_family(path)
            descriptor = open_private_append(selected)
            os.close(descriptor)
            assert_private_log_family(selected)
        except PrivateStateError as exc:
            raise OperationalError(exc.code) from exc
        self._path = selected
        self._logger = logging.getLogger(f"jarvis.operations.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        handler = _PrivateRotatingFileHandler(
            selected,
            maxBytes=maximum_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def emit(self, code: str, **metrics: float | bool | None) -> None:
        if not code or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
            for character in code
        ):
            raise ValueError("operational log code is invalid")
        if any(isinstance(value, str) for value in metrics.values()):
            raise ValueError("operational logs accept no free-form strings")
        document: dict[str, Any] = {"code": code, "observed_at": _utc_now()}
        for key, value in metrics.items():
            if not key or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for character in key
            ):
                raise ValueError("operational metric name is invalid")
            if (value is not None and not isinstance(value, (bool, int, float))) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError("operational metric value is invalid")
            document[key] = value
        try:
            assert_private_log_family(self._path)
            self._logger.info(
                json.dumps(document, sort_keys=True, separators=(",", ":"))
            )
            for handler in self._logger.handlers:
                handler.flush()
            assert_private_log_family(self._path)
            assert_private_regular_file(self._path)
        except PrivateStateError as exc:
            raise OperationalError(exc.code) from exc

    def close(self) -> None:
        for handler in tuple(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)
        try:
            assert_private_log_family(self._path)
            assert_private_regular_file(self._path)
        except PrivateStateError as exc:
            raise OperationalError(exc.code) from exc


class ProcessLock:
    """Cross-platform, non-blocking single-process fence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: Any | None = None

    def acquire(self) -> None:
        try:
            descriptor = open_private_lock(self.path)
            handle = os.fdopen(descriptor, "a+b")
        except PrivateStateError as exc:
            raise OperationalError(exc.code) from exc
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            raise OperationalError("service_already_running") from exc
        self._file = handle

    def release(self) -> None:
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class DatabaseDiagnosticsV1:
    version: str
    ready: bool
    integrity_ok: bool
    integrity_issue_count: int
    invariant_violation_count: int
    invariant_counts: Mapping[str, int]
    schema_version: int
    database_bytes: int
    wal_bytes: int
    shm_bytes: int
    page_count: int
    freelist_count: int
    conversation_count: int
    task_count: int
    active_task_count: int
    lifecycle_event_count: int
    outbox_count: int
    replayable_outbox_count: int
    claimed_outbox_count: int
    extension_outbox_count: int
    replayable_extension_outbox_count: int
    pending_approval_count: int
    pending_model_reservation_count: int
    pending_capability_reservation_count: int
    enabled_cloud_endpoint_count: int
    operational_log_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_count": self.conversation_count,
            "active_task_count": self.active_task_count,
            "claimed_outbox_count": self.claimed_outbox_count,
            "database_bytes": self.database_bytes,
            "enabled_cloud_endpoint_count": self.enabled_cloud_endpoint_count,
            "extension_outbox_count": self.extension_outbox_count,
            "freelist_count": self.freelist_count,
            "integrity_issue_count": self.integrity_issue_count,
            "integrity_ok": self.integrity_ok,
            "invariant_counts": dict(sorted(self.invariant_counts.items())),
            "invariant_violation_count": self.invariant_violation_count,
            "lifecycle_event_count": self.lifecycle_event_count,
            "outbox_count": self.outbox_count,
            "operational_log_bytes": self.operational_log_bytes,
            "page_count": self.page_count,
            "pending_approval_count": self.pending_approval_count,
            "pending_capability_reservation_count": self.pending_capability_reservation_count,
            "pending_model_reservation_count": self.pending_model_reservation_count,
            "ready": self.ready,
            "replayable_outbox_count": self.replayable_outbox_count,
            "replayable_extension_outbox_count": self.replayable_extension_outbox_count,
            "schema_version": self.schema_version,
            "shm_bytes": self.shm_bytes,
            "task_count": self.task_count,
            "version": self.version,
            "wal_bytes": self.wal_bytes,
        }


def _scalar(
    connection: sqlite3.Connection, query: str, parameters: tuple[Any, ...] = ()
) -> int:
    row = connection.execute(query, parameters).fetchone()
    return int(row[0] if row is not None else 0)


def _memory_invariants(
    connection: sqlite3.Connection,
    *,
    schema_version: int,
) -> dict[str, int]:
    """Return content-free encrypted-memory integrity counts for schema v10+."""

    if schema_version < 10:
        return {}
    return {
        "memory_principal_binding": _scalar(
            connection,
            """SELECT ABS((SELECT COUNT(*) FROM memory_principals)-1)
                      + ABS((SELECT COUNT(*) FROM memory_principal_usage)-1)
                      + (SELECT COUNT(*) FROM conversations c
                         LEFT JOIN conversation_principals b
                           USING(conversation_id)
                         LEFT JOIN memory_principals p
                           USING(principal_id)
                         WHERE b.conversation_id IS NULL
                            OR p.owner_kind!='local_owner')""",
        ),
        "memory_crypto_state_cardinality": _scalar(
            connection,
            """SELECT ABS(
                   (SELECT COUNT(*) FROM memory_principals)
                   - (SELECT COUNT(*) FROM memory_crypto_state)
               )""",
        ),
        "memory_epoch_state_binding": _scalar(
            connection,
            """SELECT COUNT(*) FROM memory_crypto_state s
               LEFT JOIN memory_key_epochs e ON e.epoch_id=s.write_epoch_id
               WHERE (s.storage_state='LEGACY_PENDING' AND (
                        s.provider_install_id IS NOT NULL OR s.write_epoch_id IS NOT NULL
                     )) OR (s.storage_state IN ('MIGRATING','ENCRYPTED') AND (
                        e.epoch_id IS NULL OR e.principal_id!=s.principal_id
                        OR e.provider_install_id!=s.provider_install_id
                        OR (s.storage_state='ENCRYPTED' AND e.state!='WRITE_ACTIVE')
                     ))""",
        ),
        "memory_pending_crypto_operations": _scalar(
            connection,
            "SELECT COUNT(*) FROM memory_crypto_operations WHERE phase!='COMPLETE'",
        ),
        "memory_usage_mismatch": _scalar(
            connection,
            """SELECT COUNT(*) FROM memory_principal_usage u
               LEFT JOIN (
                   SELECT principal_id,COUNT(*) AS record_count,
                          COALESCE(SUM(content_bytes),0) AS content_bytes
                   FROM memory_records GROUP BY principal_id
               ) r USING(principal_id)
               JOIN memory_crypto_state s USING(principal_id)
               WHERE s.storage_state='ENCRYPTED' AND (
                   u.active_record_count!=COALESCE(r.record_count,0)
                   OR u.active_content_bytes!=COALESCE(r.content_bytes,0)
               )""",
        ),
        "memory_term_count_mismatch": _scalar(
            connection,
            """SELECT COUNT(*) FROM memory_records r
               LEFT JOIN (
                   SELECT record_id,COUNT(*) AS indexed_terms
                   FROM memory_terms GROUP BY record_id
               ) t USING(record_id)
               WHERE r.term_count!=COALESCE(t.indexed_terms,0)""",
        ),
        "memory_epoch_reference_invalid": _scalar(
            connection,
            """SELECT COUNT(*) FROM memory_records r
               LEFT JOIN memory_key_epochs e ON e.epoch_id=r.key_epoch_id
               WHERE e.epoch_id IS NULL OR e.principal_id!=r.principal_id
                  OR e.key_version!=r.key_version
                  OR e.state NOT IN ('WRITE_ACTIVE','READABLE')""",
        ),
        "memory_source_binding_invalid": _scalar(
            connection,
            """SELECT COUNT(*) FROM memory_records r
               LEFT JOIN tasks t ON t.task_id=r.task_id
               LEFT JOIN canonical_turns c ON c.task_id=r.task_id
               LEFT JOIN inbox i ON i.event_id=r.source_event_id
               WHERE t.task_id IS NULL OR c.task_id IS NULL OR i.event_id IS NULL
                  OR t.conversation_id!=r.conversation_id
                  OR c.conversation_id!=r.conversation_id
                  OR c.turn_id!=t.turn_id OR i.task_id!=r.task_id
                  OR i.turn_id!=t.turn_id OR t.state!='COMPLETED'
                  OR c.status!='COMPLETED'""",
        ),
        "memory_tombstone_live_collision": _scalar(
            connection,
            """SELECT COUNT(*) FROM memory_record_tombstones d
               JOIN memory_records r USING(record_id)""",
        ),
        "memory_record_audit_missing": _scalar(
            connection,
            """SELECT COUNT(*) FROM memory_records r
               WHERE NOT EXISTS (
                   SELECT 1 FROM memory_record_audits a
                   WHERE a.record_id=r.record_id
                     AND a.principal_id=r.principal_id
                     AND a.action IN ('CREATED','MIGRATED')
               )""",
        ),
        "memory_tombstone_audit_missing": _scalar(
            connection,
            """SELECT COUNT(*) FROM memory_record_tombstones d
               WHERE NOT EXISTS (
                   SELECT 1 FROM memory_record_audits a
                   WHERE a.record_id=d.record_id
                     AND a.principal_id=d.principal_id
                     AND a.action IN (
                         'EXPLICIT_FORGOT','CONTROL_DELETED',
                         'SOURCE_CONVERSATION_DELETED','TTL_EXPIRED'
                     )
               )""",
        ),
    }


def _database_diagnostics(
    path: str | Path,
    *,
    immutable: bool = False,
) -> DatabaseDiagnosticsV1:
    """Run read-only integrity and lifecycle invariant checks."""

    selected = absolute_path_no_resolve(path)
    try:
        assert_private_database(selected)
    except PrivateStateError as exc:
        raise OperationalError(exc.code) from exc
    connection: sqlite3.Connection | None = None
    try:
        immutable_query = "&immutable=1" if immutable else ""
        connection = sqlite3.connect(
            f"{selected.as_uri()}?mode=ro{immutable_query}", uri=True
        )
        connection.execute("PRAGMA query_only=ON")
        integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
        integrity_ok = len(integrity_rows) == 1 and integrity_rows[0][0] == "ok"
        integrity_issue_count = 0 if integrity_ok else len(integrity_rows)
        schema_version = int(
            connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
        )
        invariants = {
            "multiple_active_conversations": max(
                0,
                _scalar(
                    connection,
                    "SELECT COUNT(*) FROM conversations WHERE state='ACTIVE'",
                )
                - 1,
            ),
            "multiple_active_tasks": max(
                0,
                _scalar(
                    connection,
                    """SELECT COUNT(*) FROM tasks WHERE tombstoned=0 AND state IN (
                       'ADMITTED','ROUTING','LOADING_CONTEXT','EXECUTING','RESPONDING',
                       'COMMITTING','AWAITING_APPROVAL','DECISION_WAITING_RECONNECT')""",
                )
                - 1,
            ),
            "cloud_dispatch_overflow": _scalar(
                connection, "SELECT COUNT(*) FROM tasks WHERE cloud_dispatch_count>1"
            ),
            "mutually_exclusive_terminals": _scalar(
                connection,
                """SELECT COUNT(*) FROM (
                   SELECT response_id FROM outbox
                   WHERE event_type IN ('response.speech.done','response.cancelled')
                   GROUP BY response_id HAVING COUNT(DISTINCT event_type)>1)""",
            ),
            "output_after_terminal": _scalar(
                connection,
                """SELECT COUNT(*) FROM outbox later JOIN outbox terminal
                   ON later.response_id=terminal.response_id AND later.ordinal>terminal.ordinal
                   WHERE terminal.event_type IN ('response.speech.done','response.cancelled')
                   AND later.event_type IN (
                     'response.started','response.text.delta','response.text.done',
                     'response.speech.commit','response.speech.done')""",
            ),
            "completed_without_done": _scalar(
                connection,
                """SELECT COUNT(*) FROM tasks t WHERE t.state='COMPLETED' AND (
                   NOT EXISTS(SELECT 1 FROM outbox o WHERE o.task_id=t.task_id AND o.event_type='response.text.done')
                   OR NOT EXISTS(SELECT 1 FROM outbox o WHERE o.task_id=t.task_id AND o.event_type='response.speech.done'))""",
            ),
            "duplicate_text_chunks": _scalar(
                connection,
                """SELECT COUNT(*) FROM (
                   SELECT response_id,json_extract(payload_json,'$.chunk_index') AS position
                   FROM outbox WHERE event_type='response.text.delta'
                   GROUP BY response_id,position HAVING COUNT(*)>1)""",
            ),
            "duplicate_speech_units": _scalar(
                connection,
                """SELECT COUNT(*) FROM (
                   SELECT response_id,json_extract(payload_json,'$.unit_index') AS position
                   FROM outbox WHERE event_type='response.speech.commit'
                   GROUP BY response_id,position HAVING COUNT(*)>1)""",
            ),
        }
        invariants.update(_memory_invariants(connection, schema_version=schema_version))
        violation_count = sum(invariants.values())
        result = DatabaseDiagnosticsV1(
            version="operational-diagnostics/1",
            ready=integrity_ok and violation_count == 0,
            integrity_ok=integrity_ok,
            integrity_issue_count=integrity_issue_count,
            invariant_violation_count=violation_count,
            invariant_counts=invariants,
            schema_version=schema_version,
            database_bytes=_file_bytes(selected),
            wal_bytes=_file_bytes(Path(str(selected) + "-wal")),
            shm_bytes=_file_bytes(Path(str(selected) + "-shm")),
            page_count=_scalar(connection, "PRAGMA page_count"),
            freelist_count=_scalar(connection, "PRAGMA freelist_count"),
            conversation_count=_scalar(
                connection, "SELECT COUNT(*) FROM conversations"
            ),
            task_count=_scalar(connection, "SELECT COUNT(*) FROM tasks"),
            active_task_count=_scalar(
                connection,
                """SELECT COUNT(*) FROM tasks WHERE tombstoned=0 AND state IN (
                   'ADMITTED','ROUTING','LOADING_CONTEXT','EXECUTING','RESPONDING',
                   'COMMITTING','AWAITING_APPROVAL','DECISION_WAITING_RECONNECT')""",
            ),
            lifecycle_event_count=_scalar(
                connection, "SELECT COUNT(*) FROM task_events"
            ),
            outbox_count=_scalar(connection, "SELECT COUNT(*) FROM outbox"),
            replayable_outbox_count=_scalar(
                connection, "SELECT COUNT(*) FROM outbox WHERE state='REPLAYABLE'"
            ),
            claimed_outbox_count=_scalar(
                connection, "SELECT COUNT(*) FROM outbox WHERE state='CLAIMED'"
            ),
            extension_outbox_count=_scalar(
                connection, "SELECT COUNT(*) FROM cloud_extension_outbox"
            ),
            replayable_extension_outbox_count=_scalar(
                connection,
                "SELECT COUNT(*) FROM cloud_extension_outbox WHERE state='REPLAYABLE'",
            ),
            pending_approval_count=_scalar(
                connection, "SELECT COUNT(*) FROM approvals WHERE status='PENDING'"
            ),
            pending_model_reservation_count=_scalar(
                connection,
                "SELECT COUNT(*) FROM reservations WHERE state IN ('RESERVED','DISPATCHED')",
            ),
            pending_capability_reservation_count=_scalar(
                connection,
                "SELECT COUNT(*) FROM capability_reservations WHERE status='RESERVED'",
            ),
            enabled_cloud_endpoint_count=_scalar(
                connection,
                "SELECT COUNT(*) FROM endpoint_qualifications WHERE enabled=1",
            ),
            operational_log_bytes=sum(
                _file_bytes(item)
                for item in (selected.parent / "operations").glob(
                    "orchestrator.operations.jsonl*"
                )
            ),
        )
        return result
    except (sqlite3.DatabaseError, OSError, TypeError, ValueError, IndexError) as exc:
        raise OperationalError("database_integrity_unavailable") from exc
    finally:
        if connection is not None:
            connection.close()


def database_diagnostics(path: str | Path) -> DatabaseDiagnosticsV1:
    """Run diagnostics against the live SQLite view, including its WAL."""

    return _database_diagnostics(path, immutable=False)


def _sha256_file(path: Path) -> str:
    try:
        with open(path, "rb", opener=lambda name, flags: _open_private_fd(Path(name), flags, create=False)) as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except PrivateStateError as exc:
        raise OperationalError(exc.code) from exc


@dataclass(frozen=True, slots=True)
class _MemoryBackupInspectionV1:
    schema_version: int
    storage_state: str | None
    encrypted: EncryptedMemoryBackupStateV1 | None


def memory_backup_manifest_path(backup: str | Path) -> Path:
    """Return the fixed sidecar path for one SQLite backup path."""

    selected = absolute_path_no_resolve(backup)
    return Path(f"{selected}{MEMORY_BACKUP_MANIFEST_SUFFIX}")


def _inspect_memory_backup_database(
    path: Path,
    *,
    immutable: bool = False,
) -> _MemoryBackupInspectionV1:
    """Read only content-free v10 backup facts, then close SQLite."""

    connection: sqlite3.Connection | None = None
    try:
        immutable_query = "&immutable=1" if immutable else ""
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro{immutable_query}", uri=True
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        row = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            raise OperationalError("memory_backup_state_invalid")
        schema_version = int(row[0])
        if schema_version < 10:
            return _MemoryBackupInspectionV1(schema_version, None, None)
        if schema_version != 10:
            raise OperationalError("memory_backup_schema_unsupported")
        pending = _scalar(
            connection,
            "SELECT COUNT(*) FROM memory_crypto_operations WHERE phase!='COMPLETE'",
        )
        if pending:
            raise OperationalError("memory_backup_crypto_operation_pending")
        states = connection.execute(
            "SELECT storage_state,provider_install_id,write_epoch_id,"
            "memory_revision,active_operation_id FROM memory_crypto_state"
        ).fetchall()
        if len(states) != 1:
            raise OperationalError("memory_backup_state_invalid")
        state = states[0]
        storage_state = str(state["storage_state"])
        if storage_state == "MIGRATING":
            raise OperationalError("memory_backup_migration_incomplete")
        if state["active_operation_id"] is not None:
            raise OperationalError("memory_backup_state_invalid")
        if storage_state == "LEGACY_PENDING":
            legacy_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='memory_records_v9_legacy'"
            ).fetchone()
            if legacy_table is None:
                raise OperationalError("memory_backup_state_invalid")
            if _scalar(connection, "SELECT COUNT(*) FROM memory_records_v9_legacy"):
                raise OperationalError("memory_backup_legacy_plaintext_refused")
            return _MemoryBackupInspectionV1(
                schema_version,
                storage_state,
                None,
            )
        if storage_state != "ENCRYPTED":
            raise OperationalError("memory_backup_state_invalid")
        epoch_rows = connection.execute(
            """SELECT DISTINCT e.epoch_id,e.key_version,e.provider_install_id
               FROM memory_key_epochs e
               WHERE e.epoch_id=? OR e.epoch_id IN (
                   SELECT DISTINCT key_epoch_id FROM memory_records
               )
               ORDER BY e.key_version,e.epoch_id""",
            (state["write_epoch_id"],),
        ).fetchall()
        referenced_epoch_ids = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT key_epoch_id FROM memory_records"
            ).fetchall()
        }
        expected_epoch_ids = referenced_epoch_ids | {str(state["write_epoch_id"])}
        observed_epoch_ids = {str(epoch["epoch_id"]) for epoch in epoch_rows}
        if expected_epoch_ids != observed_epoch_ids or any(
            epoch["provider_install_id"] != state["provider_install_id"]
            for epoch in epoch_rows
        ):
            raise OperationalError("memory_backup_state_invalid")
        try:
            encrypted = EncryptedMemoryBackupStateV1(
                schema_version=schema_version,
                provider_install_id=state["provider_install_id"],
                write_epoch_id=state["write_epoch_id"],
                memory_revision=int(state["memory_revision"]),
                referenced_epochs=tuple(
                    MemoryBackupEpochV1(
                        key_version=int(epoch["key_version"]),
                        epoch_id=str(epoch["epoch_id"]),
                    )
                    for epoch in epoch_rows
                ),
            )
        except MemoryBackupError as exc:
            raise OperationalError(exc.code) from exc
        return _MemoryBackupInspectionV1(
            schema_version,
            storage_state,
            encrypted,
        )
    except OperationalError:
        raise
    except (sqlite3.DatabaseError, OSError, TypeError, ValueError) as exc:
        raise OperationalError("memory_backup_inspection_failed") from exc
    finally:
        if connection is not None:
            connection.close()


def _fsync_file(path: Path) -> None:
    try:
        with open(path, "rb", opener=lambda name, flags: _open_private_fd(Path(name), flags, create=False)) as source:
            os.fsync(source.fileno())
    except PrivateStateError as exc:
        raise OperationalError(exc.code) from exc


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":  # Windows lacks a portable directory flush seam.
        return
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_private_temporary(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
    payload: bytes,
) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    temporary = Path(name)
    try:
        if os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    os.close(descriptor)
    try:
        assert_private_regular_file(temporary)
    except PrivateStateError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise OperationalError(exc.code) from exc
    return temporary


def _read_private_bounded(path: Path, *, maximum: int) -> bytes:
    try:
        descriptor = _open_private_fd(path, os.O_RDONLY, create=False)
        try:
            opened = os.fstat(descriptor)
            if not 1 <= opened.st_size <= maximum:
                raise OperationalError("memory_backup_manifest_invalid")
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) != opened.st_size or len(payload) > maximum:
                raise OperationalError("memory_backup_manifest_invalid")
            return payload
        finally:
            os.close(descriptor)
    except PrivateStateError as exc:
        raise OperationalError(exc.code) from exc
    except OSError as exc:
        raise OperationalError("memory_backup_manifest_unavailable") from exc


def _private_file_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise OperationalError("memory_backup_manifest_unavailable") from exc
    return True


def _require_no_sqlite_sidecars(path: Path, *, code: str) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        candidate = Path(f"{path}{suffix}")
        try:
            assert_private_regular_file(candidate, required=False)
        except PrivateStateError as exc:
            raise OperationalError(exc.code) from exc
        if _private_file_exists(candidate):
            raise OperationalError(code)


def _require_manifest_matches_database(
    manifest: MemoryBackupManifestV1,
    *,
    database: Path,
    inspection: _MemoryBackupInspectionV1,
) -> None:
    if (
        inspection.storage_state != "ENCRYPTED"
        or inspection.encrypted is None
        or manifest.memory_state != inspection.encrypted
        or manifest.backup_bytes != _file_bytes(database)
        or manifest.backup_sha256 != _sha256_file(database)
    ):
        raise OperationalError("memory_backup_manifest_mismatch")


def _load_backup_manifest(
    backup: Path,
    inspection: _MemoryBackupInspectionV1,
) -> tuple[MemoryBackupManifestV1 | None, bytes | None]:
    sidecar = memory_backup_manifest_path(backup)
    exists = _private_file_exists(sidecar)
    if inspection.storage_state != "ENCRYPTED":
        if exists:
            raise OperationalError("memory_backup_manifest_unexpected")
        return None, None
    if not exists:
        raise OperationalError("memory_backup_manifest_missing")
    payload = _read_private_bounded(
        sidecar,
        maximum=MAX_MEMORY_BACKUP_MANIFEST_BYTES,
    )
    try:
        manifest = MemoryBackupManifestV1.from_jcs_bytes(payload)
    except MemoryBackupError as exc:
        raise OperationalError(exc.code) from exc
    _require_manifest_matches_database(
        manifest,
        database=backup,
        inspection=inspection,
    )
    return manifest, payload


def _copy_private_database_exact(source: Path, destination: Path) -> str:
    """Copy one private regular DB to an existing private temporary file."""

    try:
        assert_private_database(source)
        source_descriptor = _open_private_fd(source, os.O_RDONLY, create=False)
        try:
            destination_descriptor = _open_private_fd(destination, os.O_WRONLY | os.O_TRUNC, create=False)
            digest = hashlib.sha256()
            try:
                while True:
                    block = os.read(source_descriptor, 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    view = memoryview(block)
                    while view:
                        written = os.write(destination_descriptor, view)
                        if written <= 0:
                            raise OSError
                        view = view[written:]
                os.fsync(destination_descriptor)
            finally:
                os.close(destination_descriptor)
            return digest.hexdigest()
        finally:
            os.close(source_descriptor)
    except PrivateStateError as exc:
        raise OperationalError(exc.code) from exc


def backup_database(
    source: str | Path,
    destination: str | Path,
    *,
    memory_provider_attestor: MemoryBackupProviderAttestor | None = None,
) -> dict[str, Any]:
    """Create an atomic SQLite backup and committed v10 memory sidecar."""

    source_path = absolute_path_no_resolve(source)
    destination_path = absolute_path_no_resolve(destination)
    if source_path == destination_path:
        raise OperationalError("backup_path_invalid")
    try:
        assert_private_database(source_path)
        ensure_private_directory(destination_path.parent, create=True)
        assert_private_regular_file(destination_path, required=False)
        for suffix in ("-wal", "-shm", "-journal"):
            assert_private_regular_file(
                Path(f"{destination_path}{suffix}"),
                required=False,
            )
        assert_private_regular_file(
            memory_backup_manifest_path(destination_path),
            required=False,
        )
    except PrivateStateError as exc:
        raise OperationalError(exc.code) from exc
    _require_no_sqlite_sidecars(
        destination_path,
        code="memory_backup_destination_sqlite_sidecar_present",
    )
    before = database_diagnostics(source_path)
    if not before.integrity_ok:
        raise OperationalError("source_database_corrupt")
    _inspect_memory_backup_database(source_path)
    temporary = _write_private_temporary(
        destination_path.parent, prefix=f".{destination_path.name}.", suffix=".tmp", payload=b""
    )
    manifest_temporary: Path | None = None
    try:
        with closing(sqlite3.connect(
            f"{source_path.as_uri()}?mode=ro", uri=True
        )) as source_connection, closing(sqlite3.connect(temporary)) as target_connection:
            source_connection.backup(target_connection, pages=256, sleep=0.01)
        _fsync_file(temporary)
        inspection = _inspect_memory_backup_database(temporary, immutable=True)
        verified = _database_diagnostics(temporary, immutable=True)
        if not verified.integrity_ok or verified.invariant_violation_count:
            raise OperationalError("backup_verification_failed")
        backup_bytes = _file_bytes(temporary)
        backup_sha256 = _sha256_file(temporary)
        manifest: MemoryBackupManifestV1 | None = None
        if inspection.encrypted is not None:
            try:
                provider = verify_memory_backup_provider(
                    inspection.encrypted,
                    memory_provider_attestor,
                )
                manifest = MemoryBackupManifestV1(
                    backup_sha256=backup_sha256,
                    backup_bytes=backup_bytes,
                    memory_state=inspection.encrypted,
                    provider_active_version=provider.active_version,
                    provider_high_water=provider.high_water,
                )
                manifest_temporary = _write_private_temporary(
                    destination_path.parent,
                    prefix=f".{destination_path.name}.",
                    suffix=".manifest.tmp",
                    payload=manifest.to_jcs_bytes(),
                )
            except MemoryBackupError as exc:
                raise OperationalError(exc.code) from exc
        sidecar = memory_backup_manifest_path(destination_path)
        _require_no_sqlite_sidecars(destination_path, code="memory_backup_destination_sqlite_sidecar_present")
        try:
            os.replace(temporary, destination_path)
            if manifest is not None:
                if manifest_temporary is None:
                    raise OperationalError("memory_backup_manifest_unavailable")
                os.replace(manifest_temporary, sidecar)
            else:
                unlink_private_file(sidecar, missing_ok=True)
            _fsync_directory(destination_path.parent)
            assert_private_regular_file(destination_path)
            if manifest is not None:
                assert_private_regular_file(sidecar)
        except OperationalError:
            raise
        except (OSError, PrivateStateError) as exc:
            raise OperationalError("backup_commit_uncertain") from exc
        report: dict[str, Any] = {
            "backup_bytes": backup_bytes,
            "backup_sha256": backup_sha256,
            "integrity_ok": True,
            "memory_manifest": manifest is not None,
            "schema_version": verified.schema_version,
            "version": "sqlite-backup/2",
        }
        if manifest is not None:
            report["manifest_sha256"] = hashlib.sha256(
                manifest.to_jcs_bytes()
            ).hexdigest()
        return report
    except (sqlite3.DatabaseError, OSError) as exc:
        raise OperationalError("backup_failed") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        if manifest_temporary is not None:
            try:
                manifest_temporary.unlink(missing_ok=True)
            except OSError:
                pass


def restore_database(
    backup: str | Path,
    destination: str | Path,
    *,
    service_lock: str | Path,
    memory_provider_attestor: MemoryBackupProviderAttestor | None = None,
) -> dict[str, Any]:
    """Verify backup, sidecar, and provider before atomic offline publish."""

    backup_path = absolute_path_no_resolve(backup)
    destination_path = absolute_path_no_resolve(destination)
    if backup_path == destination_path:
        raise OperationalError("restore_path_invalid")
    try:
        assert_private_database(backup_path)
        ensure_private_directory(destination_path.parent, create=True)
        for candidate in (
            destination_path,
            Path(f"{destination_path}-wal"),
            Path(f"{destination_path}-shm"),
            Path(f"{destination_path}-journal"),
        ):
            assert_private_regular_file(candidate, required=False)
    except PrivateStateError as exc:
        raise OperationalError(exc.code) from exc
    _require_no_sqlite_sidecars(
        backup_path,
        code="memory_backup_sqlite_sidecar_present",
    )
    _require_no_sqlite_sidecars(
        destination_path,
        code="restore_destination_sqlite_sidecar_present",
    )
    verified = _database_diagnostics(backup_path, immutable=True)
    if not verified.integrity_ok or verified.invariant_violation_count:
        raise OperationalError("backup_verification_failed")
    inspection = _inspect_memory_backup_database(backup_path, immutable=True)
    manifest, manifest_payload = _load_backup_manifest(backup_path, inspection)
    if inspection.encrypted is not None:
        try:
            provider = verify_memory_backup_provider(
                inspection.encrypted,
                memory_provider_attestor,
                minimum_high_water=(
                    None if manifest is None else manifest.provider_high_water
                ),
            )
        except MemoryBackupError as exc:
            raise OperationalError(exc.code) from exc
        if (
            manifest is None
            or provider.active_version != manifest.provider_active_version
        ):
            raise OperationalError("memory_backup_provider_active_mismatch")
    with ProcessLock(service_lock):
        temporary = _write_private_temporary(
            destination_path.parent, prefix=f".{destination_path.name}.", suffix=".restore", payload=b""
        )
        try:
            copied_sha256 = _copy_private_database_exact(backup_path, temporary)
            restored_inspection = _inspect_memory_backup_database(
                temporary,
                immutable=True,
            )
            restored = _database_diagnostics(temporary, immutable=True)
            if not restored.ready:
                raise OperationalError("restore_verification_failed")
            current_manifest, current_payload = _load_backup_manifest(
                backup_path,
                inspection,
            )
            _require_no_sqlite_sidecars(
                backup_path,
                code="memory_backup_sqlite_sidecar_present",
            )
            if (
                current_manifest != manifest
                or current_payload != manifest_payload
                or restored_inspection != inspection
            ):
                raise OperationalError("memory_backup_manifest_mismatch")
            if manifest is not None:
                _require_manifest_matches_database(
                    manifest,
                    database=temporary,
                    inspection=restored_inspection,
                )
                if copied_sha256 != manifest.backup_sha256:
                    raise OperationalError("memory_backup_manifest_mismatch")
                if restored_inspection.encrypted is None:
                    raise OperationalError("memory_backup_manifest_mismatch")
                try:
                    final_provider = verify_memory_backup_provider(
                        restored_inspection.encrypted,
                        memory_provider_attestor,
                        minimum_high_water=manifest.provider_high_water,
                    )
                except MemoryBackupError as exc:
                    raise OperationalError(exc.code) from exc
                if final_provider.active_version != manifest.provider_active_version:
                    raise OperationalError("memory_backup_provider_active_mismatch")
            _require_no_sqlite_sidecars(destination_path, code="restore_destination_sqlite_sidecar_present")
            try:
                os.replace(temporary, destination_path)
                _fsync_directory(destination_path.parent)
                assert_private_database(destination_path)
            except (OSError, PrivateStateError) as exc:
                raise OperationalError("restore_commit_uncertain") from exc
            return {
                "database_bytes": _file_bytes(destination_path),
                "database_sha256": _sha256_file(destination_path),
                "integrity_ok": True,
                "memory_manifest": manifest is not None,
                "schema_version": restored.schema_version,
                "version": "sqlite-restore/2",
            }
        except (sqlite3.DatabaseError, OSError) as exc:
            raise OperationalError("restore_failed") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def operational_paths(database_path: str | Path) -> dict[str, Path]:
    database = absolute_path_no_resolve(database_path)
    state = database.parent / "operations"
    return {
        "lock": state / "orchestrator.lock",
        "log": state / "orchestrator.operations.jsonl",
        "stop": state / "orchestrator.stop-request",
        "state": state,
    }


__all__ = [
    "ContentFreeOperationalLog",
    "DatabaseDiagnosticsV1",
    "MemoryBackupManifestV1",
    "MemoryBackupProviderAttestor",
    "OperationalError",
    "ProcessLock",
    "backup_database",
    "database_diagnostics",
    "memory_backup_manifest_path",
    "operational_paths",
    "restore_database",
    "verify_memory_backup_provider",
]
