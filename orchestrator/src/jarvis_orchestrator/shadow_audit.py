"""Read-only, content-free reporting for real JARVIS shadow sessions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .anamnesis_shadow import reconstruct_active_shadow_memories
from .domain import canonical_json
from .memory_release import (
    MEMORY_RELEASE_POLICY_SHA256,
    MEMORY_RELEASE_POLICY_VERSION,
    validate_shadow_audit_payload,
)


SHADOW_AUDIT_REPORT_VERSION = "jarvis.shadow-audit-report.v1"
MINIMUM_REAL_SESSION_CONVERSATIONS = 20
MINIMUM_REAL_SESSION_TURNS = 100
_AUDIT_EVENTS = {
    "memory.shadow.pending",
    "memory.shadow.proposed",
    "memory.shadow.failed",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _open_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("shadow audit database must be a regular file")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def build_shadow_trial_report(database_path: str | Path) -> dict[str, Any]:
    """Aggregate a stopped/local database without returning user content or IDs."""

    connection = _open_read_only(Path(database_path))
    try:
        schema = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        if schema is None or schema["value"] not in {"2", "3", "4", "5", "6", "7"}:
            raise RuntimeError("unsupported shadow audit database schema")
        rows = connection.execute(
            """SELECT e.event_sequence,e.epoch,e.event_type,e.payload_json,e.created_at,
                      k.conversation_id,t.user_text,t.user_language,i.event_id
               FROM task_events e
               JOIN tasks k USING(task_id)
               LEFT JOIN canonical_turns t USING(task_id)
               LEFT JOIN inbox i USING(task_id)
               WHERE e.event_type IN (
                   'memory.shadow.pending',
                   'memory.shadow.proposed',
                   'memory.shadow.failed'
               )
               ORDER BY e.event_sequence"""
        ).fetchall()
        completed_turns = int(
            connection.execute(
                "SELECT COUNT(*) FROM canonical_turns WHERE status='COMPLETED'"
            ).fetchone()[0]
        )
        authoritative_memory_records = int(
            connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
        )
    finally:
        connection.close()

    event_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    writer_contract_counts: Counter[str] = Counter()
    release_policy_counts: Counter[str] = Counter()
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audited_conversations: set[str] = set()
    invalid_rows = 0
    authority_violations = 0
    content_free_audit_projection: list[dict[str, Any]] = []

    for row in rows:
        event_type = str(row["event_type"])
        if event_type not in _AUDIT_EVENTS:
            invalid_rows += 1
            continue
        event_counts[event_type] += 1
        audited_conversations.add(str(row["conversation_id"]))
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            invalid_rows += 1
            continue
        if not isinstance(payload, dict):
            invalid_rows += 1
            continue
        if (
            payload.get("release_policy_version") != MEMORY_RELEASE_POLICY_VERSION
            or payload.get("release_policy_sha256")
            != MEMORY_RELEASE_POLICY_SHA256
        ):
            invalid_rows += 1
            continue
        release_policy_counts[MEMORY_RELEASE_POLICY_SHA256] += 1
        if event_type != "memory.shadow.pending":
            if (
                payload.get("authoritative_state_changed") is not False
                or payload.get("automatic_side_effect_requested") is not False
                or payload.get("action_evidence_emitted") is not False
                or payload.get("authority_state_before_sha256")
                != payload.get("authority_state_after_sha256")
            ):
                authority_violations += 1
            try:
                validate_shadow_audit_payload(event_type, payload)
            except (TypeError, ValueError):
                invalid_rows += 1
                continue
        content_free_audit_projection.append(
            {
                "event_sequence": row["event_sequence"],
                "epoch": row["epoch"],
                "event_type": event_type,
                "payload_sha256": _digest(payload),
                "created_at": row["created_at"],
            }
        )
        if event_type != "memory.shadow.proposed":
            continue
        proposal = payload.get("proposal")
        if not isinstance(proposal, dict):
            invalid_rows += 1
            continue
        operation = proposal.get("operation")
        writer_contract = proposal.get("writer_contract_sha256")
        if operation not in {"upsert", "cancel", "ignore"}:
            invalid_rows += 1
        else:
            operation_counts[str(operation)] += 1
        if (
            not isinstance(writer_contract, str)
            or len(writer_contract) != 64
            or any(character not in "0123456789abcdef" for character in writer_contract)
        ):
            invalid_rows += 1
        else:
            writer_contract_counts[writer_contract] += 1
        if (
            row["user_text"] is None
            or row["user_language"] is None
            or row["event_id"] is None
        ):
            invalid_rows += 1
            continue
        histories[str(row["conversation_id"])].append(
            {
                "event_sequence": row["event_sequence"],
                "epoch": row["epoch"],
                "payload": payload,
                "created_at": row["created_at"],
                "source_text": row["user_text"],
                "source_language": row["user_language"],
                "canonical_source_event_id": row["event_id"],
            }
        )

    active_shadow_memories = 0
    invalid_histories = 0
    for history in histories.values():
        try:
            active_shadow_memories += len(
                reconstruct_active_shadow_memories(history)
            )
        except (TypeError, ValueError):
            invalid_histories += 1

    audited_turns = sum(event_counts.values())
    coverage = audited_turns / completed_turns if completed_turns else 0.0
    integrity_passed = (
        invalid_rows == 0
        and invalid_histories == 0
        and authority_violations == 0
        and event_counts["memory.shadow.pending"] == 0
        and event_counts["memory.shadow.failed"] == 0
    )
    sample_sufficient = (
        len(audited_conversations) >= MINIMUM_REAL_SESSION_CONVERSATIONS
        and audited_turns >= MINIMUM_REAL_SESSION_TURNS
    )
    if not integrity_passed:
        decision = "shadow_trial_rejected"
    elif not sample_sufficient:
        decision = "insufficient_real_session_evidence"
    else:
        decision = "awaiting_human_correctness_review"

    return {
        "report_version": SHADOW_AUDIT_REPORT_VERSION,
        "contains_user_content": False,
        "contains_stable_user_or_conversation_ids": False,
        "manual_correctness_review_required": True,
        "completed_turns": completed_turns,
        "audited_turns": audited_turns,
        "audited_conversations": len(audited_conversations),
        "audit_coverage": coverage,
        "event_counts": {
            event_type: event_counts[event_type]
            for event_type in sorted(_AUDIT_EVENTS)
        },
        "operation_counts": {
            operation: operation_counts[operation]
            for operation in ("upsert", "cancel", "ignore")
        },
        "writer_contract_counts": dict(sorted(writer_contract_counts.items())),
        "release_policy_counts": dict(sorted(release_policy_counts.items())),
        "active_shadow_memories": active_shadow_memories,
        "invalid_rows": invalid_rows,
        "invalid_histories": invalid_histories,
        "authority_violations": authority_violations,
        "authoritative_memory_records": authoritative_memory_records,
        "retrospective_memory_records": authoritative_memory_records,
        "minimum_conversations": MINIMUM_REAL_SESSION_CONVERSATIONS,
        "minimum_turns": MINIMUM_REAL_SESSION_TURNS,
        "sample_sufficient": sample_sufficient,
        "integrity_passed": integrity_passed,
        "audit_projection_sha256": _digest(content_free_audit_projection),
        "decision": decision,
    }


__all__ = [
    "MINIMUM_REAL_SESSION_CONVERSATIONS",
    "MINIMUM_REAL_SESSION_TURNS",
    "SHADOW_AUDIT_REPORT_VERSION",
    "build_shadow_trial_report",
]
