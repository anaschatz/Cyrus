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
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


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


class ContentFreeOperationalLog:
    """Bounded JSONL log accepting only codes and scalar counters."""

    def __init__(
        self,
        path: str | Path,
        *,
        maximum_bytes: int = 1_048_576,
        backup_count: int = 3,
    ) -> None:
        selected = Path(path)
        selected.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"jarvis.operations.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        handler = logging.handlers.RotatingFileHandler(
            selected,
            maxBytes=maximum_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def emit(self, code: str, **metrics: int | float | bool | None) -> None:
        if not code or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in code):
            raise ValueError("operational log code is invalid")
        if any(isinstance(value, str) for value in metrics.values()):
            raise ValueError("operational logs accept no free-form strings")
        document: dict[str, Any] = {"code": code, "observed_at": _utc_now()}
        for key, value in metrics.items():
            if not key or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in key):
                raise ValueError("operational metric name is invalid")
            if (
                value is not None
                and not isinstance(value, (bool, int, float))
            ) or (isinstance(value, float) and not math.isfinite(value)):
                raise ValueError("operational metric value is invalid")
            document[key] = value
        self._logger.info(json.dumps(document, sort_keys=True, separators=(",", ":")))

    def close(self) -> None:
        for handler in tuple(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)


class ProcessLock:
    """Cross-platform, non-blocking single-process fence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: Any | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
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

    def __enter__(self) -> "ProcessLock":
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


def _scalar(connection: sqlite3.Connection, query: str, parameters: tuple[Any, ...] = ()) -> int:
    row = connection.execute(query, parameters).fetchone()
    return int(row[0] if row is not None else 0)


def database_diagnostics(path: str | Path) -> DatabaseDiagnosticsV1:
    """Run read-only integrity and lifecycle invariant checks."""

    selected = Path(path).resolve()
    if not selected.is_file():
        raise OperationalError("database_unavailable")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{selected.as_posix()}?mode=ro", uri=True)
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
                _scalar(connection, "SELECT COUNT(*) FROM conversations WHERE state='ACTIVE'") - 1,
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
            conversation_count=_scalar(connection, "SELECT COUNT(*) FROM conversations"),
            task_count=_scalar(connection, "SELECT COUNT(*) FROM tasks"),
            active_task_count=_scalar(
                connection,
                """SELECT COUNT(*) FROM tasks WHERE tombstoned=0 AND state IN (
                   'ADMITTED','ROUTING','LOADING_CONTEXT','EXECUTING','RESPONDING',
                   'COMMITTING','AWAITING_APPROVAL','DECISION_WAITING_RECONNECT')""",
            ),
            lifecycle_event_count=_scalar(connection, "SELECT COUNT(*) FROM task_events"),
            outbox_count=_scalar(connection, "SELECT COUNT(*) FROM outbox"),
            replayable_outbox_count=_scalar(
                connection, "SELECT COUNT(*) FROM outbox WHERE state='REPLAYABLE'"
            ),
            claimed_outbox_count=_scalar(
                connection, "SELECT COUNT(*) FROM outbox WHERE state='CLAIMED'"
            ),
            extension_outbox_count=_scalar(connection, "SELECT COUNT(*) FROM cloud_extension_outbox"),
            replayable_extension_outbox_count=_scalar(
                connection,
                "SELECT COUNT(*) FROM cloud_extension_outbox WHERE state='REPLAYABLE'",
            ),
            pending_approval_count=_scalar(
                connection, "SELECT COUNT(*) FROM approvals WHERE status='PENDING'"
            ),
            pending_model_reservation_count=_scalar(
                connection, "SELECT COUNT(*) FROM reservations WHERE state IN ('RESERVED','DISPATCHED')"
            ),
            pending_capability_reservation_count=_scalar(
                connection, "SELECT COUNT(*) FROM capability_reservations WHERE status='RESERVED'"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backup_database(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Create and verify an atomic SQLite online backup."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path or not source_path.is_file():
        raise OperationalError("backup_path_invalid")
    before = database_diagnostics(source_path)
    if not before.integrity_ok:
        raise OperationalError("source_database_corrupt")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        source_connection = sqlite3.connect(f"file:{source_path.as_posix()}?mode=ro", uri=True)
        target_connection = sqlite3.connect(temporary)
        try:
            source_connection.backup(target_connection, pages=256, sleep=0.01)
        finally:
            target_connection.close()
            source_connection.close()
        verified = database_diagnostics(temporary)
        if not verified.integrity_ok or verified.invariant_violation_count:
            raise OperationalError("backup_verification_failed")
        os.replace(temporary, destination_path)
        return {
            "backup_bytes": _file_bytes(destination_path),
            "backup_sha256": _sha256_file(destination_path),
            "integrity_ok": True,
            "schema_version": verified.schema_version,
            "version": "sqlite-backup/1",
        }
    except (sqlite3.DatabaseError, OSError) as exc:
        raise OperationalError("backup_failed") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def restore_database(
    backup: str | Path,
    destination: str | Path,
    *,
    service_lock: str | Path,
) -> dict[str, Any]:
    """Restore only while the service lock proves the daemon is quiescent."""

    backup_path = Path(backup).resolve()
    destination_path = Path(destination).resolve()
    if backup_path == destination_path or not backup_path.is_file():
        raise OperationalError("restore_path_invalid")
    verified = database_diagnostics(backup_path)
    if not verified.integrity_ok or verified.invariant_violation_count:
        raise OperationalError("backup_verification_failed")
    with ProcessLock(service_lock):
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.", suffix=".restore", dir=destination_path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            source_connection = sqlite3.connect(f"file:{backup_path.as_posix()}?mode=ro", uri=True)
            target_connection = sqlite3.connect(temporary)
            try:
                source_connection.backup(target_connection, pages=256, sleep=0.01)
            finally:
                target_connection.close()
                source_connection.close()
            restored = database_diagnostics(temporary)
            if not restored.ready:
                raise OperationalError("restore_verification_failed")
            os.replace(temporary, destination_path)
            for suffix in ("-wal", "-shm"):
                Path(str(destination_path) + suffix).unlink(missing_ok=True)
            return {
                "database_bytes": _file_bytes(destination_path),
                "database_sha256": _sha256_file(destination_path),
                "integrity_ok": True,
                "schema_version": restored.schema_version,
                "version": "sqlite-restore/1",
            }
        except (sqlite3.DatabaseError, OSError) as exc:
            raise OperationalError("restore_failed") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def operational_paths(database_path: str | Path) -> dict[str, Path]:
    database = Path(database_path).resolve()
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
    "OperationalError",
    "ProcessLock",
    "backup_database",
    "database_diagnostics",
    "operational_paths",
    "restore_database",
]
