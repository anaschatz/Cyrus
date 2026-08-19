"""Content-free verifier for one new physical macOS Voice turn.

The verifier intentionally never selects transcript, response, prompt, or
payload columns.  Its report contains only fixed codes, booleans, counts,
timestamps, durations, and the private-stack deployment digest.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .private_state import assert_private_database
from .service import (
    BETA_SESSION_SCHEMA_VERSION,
    valid_voice_connection_tracker_id,
)
from .stack_config import PrivateStackConfig


PHYSICAL_TURN_GATE_VERSION = "jarvis-macos-physical-turn/2"
REQUIRED_RESPONSE_EVENTS = (
    "response.started",
    "response.text.done",
    "response.speech.commit",
    "response.speech.done",
)
TERMINAL_FAILURE_STATES = frozenset(
    {"CANCELLED", "INTERRUPTED", "BUDGET_EXHAUSTED", "FAILED"}
)


@dataclass(frozen=True, slots=True)
class VoiceConnectionState:
    connected: bool
    tracker_id: str
    generation: int


def _utc_now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return None
    return parsed


def _open_read_only(path: Path) -> sqlite3.Connection:
    assert_private_database(path)
    connection = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro", uri=True, timeout=0.5
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=500")
    return connection


def _live_voice_state(config: OrchestratorConfig) -> VoiceConnectionState | None:
    request = urllib.request.Request(
        f"http://{config.control_host}:{config.control_port}/v1/beta-session",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {config.control_bearer_token}",
            "Host": f"127.0.0.1:{config.control_port}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=1.0) as response:
            raw = response.read(4097)
        if response.status != 200 or len(raw) > 4096:
            return None
        payload = json.loads(raw.decode("utf-8", errors="strict"))
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {
                "core_ready",
                "effective_capabilities",
                "schema_version",
                "voice_client_connected",
                "voice_connection_generation",
                "voice_connection_tracker_id",
            }
            or payload.get("schema_version")
            != BETA_SESSION_SCHEMA_VERSION
            or payload.get("core_ready") is not True
            or type(payload.get("voice_client_connected")) is not bool
            or type(payload.get("voice_connection_generation")) is not int
            or payload["voice_connection_generation"] < 0
            or not valid_voice_connection_tracker_id(
                payload.get("voice_connection_tracker_id")
            )
            or not isinstance(payload.get("effective_capabilities"), dict)
            or set(payload["effective_capabilities"])
            != {
                "cloud",
                "memory_retrospective",
                "memory_shadow",
                "research",
                "tools",
            }
            or any(
                type(value) is not bool
                for value in payload["effective_capabilities"].values()
            )
        ):
            return None
        return VoiceConnectionState(
            connected=payload["voice_client_connected"],
            tracker_id=payload["voice_connection_tracker_id"],
            generation=payload["voice_connection_generation"],
        )
    except (OSError, UnicodeError, ValueError, urllib.error.URLError):
        return None


def _baseline(connection: sqlite3.Connection) -> int:
    version = connection.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    if version is None or int(version[0]) < 8:
        raise RuntimeError("playback_receipt_schema_unavailable")
    return int(
        connection.execute("SELECT COALESCE(MAX(rowid),0) FROM inbox").fetchone()[0]
    )


def _new_admissions(
    connection: sqlite3.Connection, baseline_rowid: int
) -> list[sqlite3.Row]:
    # Never add payload-bearing columns to this projection.
    return connection.execute(
        """SELECT rowid,event_id,turn_id,task_id,created_at
           FROM inbox WHERE rowid>? ORDER BY rowid""",
        (baseline_rowid,),
    ).fetchall()


def _turn_snapshot(connection: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    task = connection.execute(
        """SELECT task_id,conversation_id,turn_id,response_id,epoch,state,
                  tombstoned,admitted_at,created_at,updated_at
           FROM tasks WHERE task_id=?""",
        (task_id,),
    ).fetchone()
    if task is None:
        return {"missing": True}
    canonical = connection.execute(
        "SELECT status,created_at,completed_at FROM canonical_turns WHERE task_id=?",
        (task_id,),
    ).fetchone()
    reservations = connection.execute(
        """SELECT purpose,state,created_at,dispatched_at,reconciled_at
           FROM reservations WHERE task_id=? AND kind='model'
           ORDER BY created_at""",
        (task_id,),
    ).fetchall()
    outbox = connection.execute(
        """SELECT event_id,event_type,ordinal,state,original_sent_at,
                  ack_status
           FROM outbox WHERE task_id=? ORDER BY ordinal""",
        (task_id,),
    ).fetchall()
    attempts = connection.execute(
        """SELECT o.event_id,o.event_type,a.completed,a.finished_at
           FROM outbox o JOIN outbox_attempts a ON a.event_id=o.event_id
           WHERE o.task_id=? ORDER BY o.ordinal,a.attempt_id""",
        (task_id,),
    ).fetchall()
    receipts = connection.execute(
        """SELECT terminal_type,client_sent_at,received_at
           FROM voice_playback_receipts WHERE task_id=? ORDER BY received_at""",
        (task_id,),
    ).fetchall()
    task_events = connection.execute(
        "SELECT event_type,created_at FROM task_events WHERE task_id=? ORDER BY event_sequence",
        (task_id,),
    ).fetchall()
    current = connection.execute(
        """SELECT current_epoch,state FROM conversations
           WHERE conversation_id=?""",
        (task["conversation_id"],),
    ).fetchone()
    cloud_dispatches = int(
        connection.execute(
            """SELECT COUNT(*) FROM cloud_dispatch_permits
               WHERE task_id=? AND state='CONSUMED'""",
            (task_id,),
        ).fetchone()[0]
    )

    event_counts: dict[str, int] = {}
    event_ordinals: dict[str, list[int]] = {}
    successful_writes: set[str] = set()
    for row in outbox:
        kind = str(row["event_type"])
        event_counts[kind] = event_counts.get(kind, 0) + 1
        event_ordinals.setdefault(kind, []).append(int(row["ordinal"]))
    for row in attempts:
        if int(row["completed"]) == 1:
            successful_writes.add(str(row["event_id"]))
    required_ids = {
        str(row["event_id"])
        for row in outbox
        if row["event_type"] in REQUIRED_RESPONSE_EVENTS
        or (row["event_type"] == "server.ack" and row["ack_status"] == "accepted")
    }
    final_reservations = [row for row in reservations if row["purpose"] == "final"]
    completed_receipts = [
        row for row in receipts if row["terminal_type"] == "playback.completed"
    ]
    interrupted_receipts = [
        row for row in receipts if row["terminal_type"] == "playback.interrupted"
    ]
    started = event_ordinals.get("response.started", [])
    text_done = event_ordinals.get("response.text.done", [])
    speech_done = event_ordinals.get("response.speech.done", [])
    speech_commits = event_ordinals.get("response.speech.commit", [])
    ordered = bool(
        len(started) == len(text_done) == len(speech_done) == 1
        and speech_commits
        and started[0] < text_done[0] < speech_done[0]
        and all(started[0] < ordinal < speech_done[0] for ordinal in speech_commits)
    )
    event_types = {str(row["event_type"]) for row in task_events}
    return {
        "missing": False,
        "state": str(task["state"]),
        "tombstoned": bool(task["tombstoned"]),
        "admitted_at": str(task["admitted_at"]),
        "completed_at": (
            str(canonical["completed_at"])
            if canonical is not None and canonical["completed_at"] is not None
            else None
        ),
        "canonical_completed": bool(
            canonical is not None and canonical["status"] == "COMPLETED"
        ),
        "current_epoch": bool(
            current is not None
            and current["state"] == "ACTIVE"
            and int(current["current_epoch"]) == int(task["epoch"])
        ),
        "final_model_dispatch_count": sum(
            1 for row in final_reservations if row["dispatched_at"] is not None
        ),
        "final_model_reconciled_count": sum(
            1
            for row in final_reservations
            if row["state"] == "RECONCILED"
            and row["dispatched_at"] is not None
            and row["reconciled_at"] is not None
        ),
        "cloud_dispatch_count": cloud_dispatches,
        "event_counts": event_counts,
        "required_events_ordered": ordered,
        "required_event_writes_complete": bool(
            required_ids and required_ids.issubset(successful_writes)
        ),
        "accepted_ack_count": sum(
            1
            for row in outbox
            if row["event_type"] == "server.ack"
            and row["ack_status"] == "accepted"
        ),
        "duplicate_ack_count": sum(
            1
            for row in outbox
            if row["event_type"] == "server.ack"
            and row["ack_status"] == "duplicate"
        ),
        "stale_output_count": sum(1 for row in outbox if row["state"] == "STALE"),
        "unretired_output_count": sum(
            1 for row in outbox if row["state"] != "NON_REPLAYABLE"
        ),
        "cancel_or_error_count": sum(
            event_counts.get(kind, 0)
            for kind in ("response.cancelled", "server.error")
        ),
        "failure_event_count": sum(
            1
            for kind in event_types
            if kind == "diagnostic.internal_failure"
            or kind in TERMINAL_FAILURE_STATES
        ),
        "playback_completed_count": len(completed_receipts),
        "playback_interrupted_count": len(interrupted_receipts),
        "playback_received_at": (
            str(completed_receipts[0]["received_at"])
            if len(completed_receipts) == 1
            else None
        ),
    }


def _evidence(snapshot: dict[str, Any]) -> dict[str, Any]:
    counts = snapshot.get("event_counts", {})
    return {
        "admitted_task": not snapshot.get("missing", True),
        "task_completed": snapshot.get("state") == "COMPLETED",
        "canonical_turn_completed": bool(snapshot.get("canonical_completed")),
        "task_is_current_and_not_tombstoned": bool(
            snapshot.get("current_epoch") and not snapshot.get("tombstoned")
        ),
        "one_local_final_model_request": bool(
            snapshot.get("final_model_dispatch_count") == 1
            and snapshot.get("final_model_reconciled_count") == 1
            and snapshot.get("cloud_dispatch_count") == 0
        ),
        "accepted_transcript_ack": snapshot.get("accepted_ack_count") == 1,
        "response_started": counts.get("response.started", 0) == 1,
        "response_text_done": counts.get("response.text.done", 0) == 1,
        "response_speech_commit": counts.get("response.speech.commit", 0) >= 1,
        "response_speech_done": counts.get("response.speech.done", 0) == 1,
        "response_events_ordered": bool(snapshot.get("required_events_ordered")),
        "required_events_written_to_voice_socket": bool(
            snapshot.get("required_event_writes_complete")
        ),
        "playback_completed_receipt": bool(
            snapshot.get("playback_completed_count") == 1
            and snapshot.get("playback_interrupted_count") == 0
        ),
        "delivery_retired_after_playback": snapshot.get("unretired_output_count") == 0,
        "no_duplicate_submission": snapshot.get("duplicate_ack_count") == 0,
        "no_stale_output": snapshot.get("stale_output_count") == 0,
        "no_cancel_error_or_failure": bool(
            snapshot.get("cancel_or_error_count") == 0
            and snapshot.get("failure_event_count") == 0
        ),
    }


def run_macos_physical_turn_gate(
    stack: PrivateStackConfig,
    *,
    timeout_seconds: float = 180.0,
    poll_interval_seconds: float = 0.1,
    settle_seconds: float = 0.35,
    voice_state: Callable[[], VoiceConnectionState | None] | None = None,
) -> dict[str, Any]:
    """Wait for and verify exactly one admission created after invocation."""

    if timeout_seconds <= 0 or poll_interval_seconds <= 0 or settle_seconds < 0:
        raise ValueError("physical turn gate timing is invalid")
    config = OrchestratorConfig.load(stack.orchestrator_config_path)
    probe = voice_state or (lambda: _live_voice_state(config))
    deployment_before = stack.deployment_identity()
    started_at = _utc_now_text()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + timeout_seconds
    blockers: list[str] = []
    snapshot: dict[str, Any] = {"missing": True}
    admission_count = 0
    voice_start = probe()
    voice_at_start = bool(voice_start is not None and voice_start.connected)
    voice_seen_disconnected = not voice_at_start
    voice_state_unavailable = voice_start is None
    voice_generation_changed = False

    def observe_voice_state() -> VoiceConnectionState | None:
        nonlocal voice_generation_changed
        nonlocal voice_seen_disconnected
        nonlocal voice_state_unavailable
        current = probe()
        if current is None:
            voice_state_unavailable = True
            voice_seen_disconnected = True
            return None
        if not current.connected:
            voice_seen_disconnected = True
        if voice_start is None or (
            current.tracker_id != voice_start.tracker_id
            or current.generation != voice_start.generation
        ):
            voice_generation_changed = True
        return current

    baseline_rowid = 0
    task_id: str | None = None
    try:
        with closing(_open_read_only(config.database_path)) as connection:
            baseline_rowid = _baseline(connection)
        settled_until: float | None = None
        while time.monotonic() < deadline:
            observe_voice_state()
            with closing(_open_read_only(config.database_path)) as connection:
                admissions = _new_admissions(connection, baseline_rowid)
                admission_count = len(admissions)
                if admission_count > 1:
                    blockers.append("multiple_new_voice_turns")
                    break
                if admission_count == 1:
                    task_id = str(admissions[0]["task_id"])
                    snapshot = _turn_snapshot(connection, task_id)
                    if snapshot.get("state") in TERMINAL_FAILURE_STATES:
                        blockers.append("turn_reached_failure_terminal")
                        break
                    if snapshot.get("playback_completed_count") == 1:
                        if settled_until is None:
                            settled_until = time.monotonic() + settle_seconds
                        elif time.monotonic() >= settled_until:
                            break
            time.sleep(poll_interval_seconds)
        else:
            blockers.append("physical_turn_timeout")
        with closing(_open_read_only(config.database_path)) as connection:
            admissions = _new_admissions(connection, baseline_rowid)
            admission_count = len(admissions)
            if admission_count == 1:
                task_id = str(admissions[0]["task_id"])
                snapshot = _turn_snapshot(connection, task_id)
    except (OSError, sqlite3.Error, RuntimeError, ValueError):
        blockers.append("physical_turn_evidence_unavailable")

    finished_at = _utc_now_text()
    elapsed_ms = int(round((time.monotonic() - started_monotonic) * 1000))
    voice_end = observe_voice_state()
    voice_at_end = bool(voice_end is not None and voice_end.connected)
    voice_continuity_proved = bool(
        voice_at_start
        and voice_at_end
        and not voice_seen_disconnected
        and not voice_state_unavailable
        and not voice_generation_changed
    )
    try:
        deployment_after = stack.deployment_identity()
        deployment_stable = (
            deployment_before["deployment_sha256"]
            == deployment_after["deployment_sha256"]
        )
    except Exception:
        deployment_stable = False
    evidence = _evidence(snapshot)
    timeline = tuple(
        _parse_utc(value)
        for value in (
            started_at,
            snapshot.get("admitted_at"),
            snapshot.get("completed_at"),
            snapshot.get("playback_received_at"),
            finished_at,
        )
    )
    timestamp_chain_valid = bool(
        all(value is not None for value in timeline)
        and all(
            timeline[index] <= timeline[index + 1]  # type: ignore[operator]
            for index in range(len(timeline) - 1)
        )
    )
    evidence.update(
        {
            "exactly_one_new_turn": admission_count == 1,
            "voice_connected_at_start": voice_at_start,
            "voice_connected_at_end": voice_at_end,
            "voice_connection_continuous_during_gate": voice_continuity_proved,
            "deployment_identity_stable": deployment_stable,
            "bounded_completion": elapsed_ms <= int(timeout_seconds * 1000),
            "timestamp_chain_bound_to_gate": timestamp_chain_valid,
        }
    )
    for name, passed in evidence.items():
        if not passed:
            blockers.append(f"evidence_failed.{name}")
    blockers = sorted(set(blockers))
    return {
        "version": PHYSICAL_TURN_GATE_VERSION,
        "ready": not blockers,
        "deployment_sha256": deployment_before["deployment_sha256"],
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": elapsed_ms,
        "timeout_seconds": timeout_seconds,
        "admitted_at": snapshot.get("admitted_at"),
        "completed_at": snapshot.get("completed_at"),
        "playback_received_at": snapshot.get("playback_received_at"),
        "evidence": evidence,
        "event_counts": {
            name: int(snapshot.get("event_counts", {}).get(name, 0))
            for name in REQUIRED_RESPONSE_EVENTS
        },
        "blockers": blockers,
        "proof_limits": {
            "evidence_scope": "operator_observed_authenticated_voice_transport",
            "per_server_event_client_ack_supported": False,
            "native_binary_remote_attestation_supported": False,
            "playback_terminal_persisted_content_free": True,
        },
    }


__all__ = [
    "PHYSICAL_TURN_GATE_VERSION",
    "VoiceConnectionState",
    "run_macos_physical_turn_gate",
]
