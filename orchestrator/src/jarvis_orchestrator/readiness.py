"""Executable operational readiness gates.

Reports are deliberately content-free: they contain counts, durations,
bounded resource measurements and stable reason codes, never turn text or
wire identifiers.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

from .domain import ReasonCode, TaskState
from .operations import (
    ContentFreeOperationalLog,
    OperationalError,
    backup_database,
    database_diagnostics,
    operational_paths,
    restore_database,
)
from .private_state import PrivateStateError, create_or_replace_private_file
from .readiness_evidence import (
    AGGREGATE_KIND,
    LIVE_KIND,
    NATIVE_KIND,
    READINESS_AGGREGATE_VERSION,
    RECOVERY_KIND,
    RESTART_KIND,
    SOAK_KIND,
    SYNTHETIC_KIND,
    ReadinessEvidenceSigner,
    ReadinessEvidenceV2,
    ReadinessEvidenceVerifier,
    ReadinessIdentityV2,
    build_readiness_evidence,
    canonical_json_bytes,
    qualification_binding_for_kind,
    validate_readiness_cohort_receipt,
    validated_readiness_database_diagnostics,
    verify_readiness_evidence,
    write_readiness_evidence,
)
from .store import OrchestratorStore


SYNTHETIC_CANCELLATION_STAGES = (
    "ADMITTED",
    "ROUTING",
    "LOADING_CONTEXT",
    "EXECUTING",
    "RESPONDING",
    "AWAITING_APPROVAL",
    "DECISION_WAITING_RECONNECT",
)

MAXIMUM_EVIDENCE_BYTES = 1024 * 1024
MAXIMUM_OPERATIONAL_LOG_BYTES = 8 * 1024 * 1024


def write_readiness_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Persist a legacy operational report that is never gate-eligible."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > MAXIMUM_EVIDENCE_BYTES:
        raise OperationalError("readiness_report_too_large")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_signed_readiness_report(
    path: str | Path,
    evidence: ReadinessEvidenceV2,
    *,
    verifier: ReadinessEvidenceVerifier,
    cohort_state_directory: str | Path,
    current_state_provider: (
        Callable[[], tuple[ReadinessIdentityV2, Any]] | None
    ) = None,
    final_time_provider: Callable[[], datetime] | None = None,
) -> Path:
    """Persist only a structurally closed envelope trusted by the verifier."""

    return write_readiness_evidence(
        path,
        evidence,
        verifier=verifier,
        cohort_state_directory=cohort_state_directory,
        current_state_provider=current_state_provider,
        final_time_provider=final_time_provider,
    )


def _synthetic_payload(index: int) -> dict[str, Any]:
    return {
        "asr_score": {
            "calibrated": False,
            "method": "synthetic_unqualified",
            "value": 1.0,
        },
        "language": {"code": "en", "score": 1.0},
        "text": f"synthetic lifecycle turn {index}",
    }


def run_synthetic_lifecycle_gate(*, turns: int = 1000) -> dict[str, Any]:
    """Exercise every durable cancellation phase and successful completion."""

    if isinstance(turns, bool) or not 1 <= turns <= 100_000:
        raise ValueError("turns must be from 1 to 100000")
    started = time.monotonic()
    outcomes: Counter[str] = Counter()
    invariant_checks = 0
    maximum_database_bytes = 0
    with tempfile.TemporaryDirectory(prefix="jarvis-ready-gate-") as directory:
        path = Path(directory) / "synthetic.sqlite3"
        store = OrchestratorStore(path, tool_authority_enabled=True)
        try:
            for index in range(turns):
                task = store.admit_transcript(
                    event_id=str(uuid.uuid4()),
                    turn_id=str(uuid.uuid4()),
                    payload=_synthetic_payload(index),
                ).task
                if task is None:
                    raise OperationalError("synthetic_admission_failed")
                scenario = index % (len(SYNTHETIC_CANCELLATION_STAGES) + 2)
                if scenario == 0:
                    store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED)
                    outcomes["cancelled_admitted"] += 1
                else:
                    task = store.transition(
                        task.task_id, task.epoch, TaskState.ADMITTED, TaskState.ROUTING
                    )
                    if scenario == 1:
                        store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED)
                        outcomes["cancelled_routing"] += 1
                    else:
                        task = store.transition(
                            task.task_id,
                            task.epoch,
                            TaskState.ROUTING,
                            TaskState.LOADING_CONTEXT,
                        )
                        if scenario == 2:
                            store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED)
                            outcomes["cancelled_loading_context"] += 1
                        else:
                            task = store.transition(
                                task.task_id,
                                task.epoch,
                                TaskState.LOADING_CONTEXT,
                                TaskState.EXECUTING,
                            )
                            if scenario == 3:
                                store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED)
                                outcomes["cancelled_executing"] += 1
                            elif scenario == 4:
                                store.start_response(task.task_id, task.epoch, "en-US")
                                store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED)
                                outcomes["cancelled_responding"] += 1
                            elif scenario == 5:
                                store.create_approval(
                                    task.task_id,
                                    task.epoch,
                                    kind="tool",
                                    name="synthetic_noop",
                                    arguments={},
                                )
                                store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED)
                                outcomes["cancelled_awaiting_approval"] += 1
                            elif scenario == 6:
                                approval_id = store.create_approval(
                                    task.task_id,
                                    task.epoch,
                                    kind="tool",
                                    name="synthetic_noop",
                                    arguments={},
                                )
                                store.decide_approval(
                                    approval_id,
                                    task.epoch,
                                    "deny",
                                    client_connected=False,
                                )
                                store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED)
                                outcomes["cancelled_decision_waiting_reconnect"] += 1
                            else:
                                store.start_response(task.task_id, task.epoch, "en-US")
                                store.append_response_event(
                                    task.task_id,
                                    task.epoch,
                                    "response.text.delta",
                                    {"chunk_index": 0, "text": "Synthetic response."},
                                )
                                store.complete_response(
                                    task.task_id,
                                    task.epoch,
                                    "Synthetic response.",
                                    chunk_count=1,
                                    trailing_speech=[(1, "en-US", "Synthetic response.")],
                                    last_unit_index=1,
                                )
                                outcomes["completed"] += 1
                with store._lock:
                    active_conversations = store._connection.execute(
                        "SELECT COUNT(*) FROM conversations WHERE state='ACTIVE'"
                    ).fetchone()[0]
                    active_tasks = store._connection.execute(
                        """SELECT COUNT(*) FROM tasks WHERE tombstoned=0 AND state IN (
                           'ADMITTED','ROUTING','LOADING_CONTEXT','EXECUTING','RESPONDING',
                           'COMMITTING','AWAITING_APPROVAL','DECISION_WAITING_RECONNECT')"""
                    ).fetchone()[0]
                    duplicate_terminals = store._connection.execute(
                        """SELECT COUNT(*) FROM (
                           SELECT response_id FROM outbox
                           WHERE event_type IN ('response.speech.done','response.cancelled')
                           GROUP BY response_id HAVING COUNT(DISTINCT event_type)>1)"""
                    ).fetchone()[0]
                if active_conversations > 1 or active_tasks > 1 or duplicate_terminals:
                    raise OperationalError("synthetic_invariant_violation")
                invariant_checks += 1
                maximum_database_bytes = max(
                    maximum_database_bytes,
                    path.stat().st_size + Path(str(path) + "-wal").stat().st_size
                    if Path(str(path) + "-wal").exists()
                    else path.stat().st_size,
                )
            store.checkpoint()
            final = database_diagnostics(path)
        finally:
            store.close()
    elapsed_ms = int((time.monotonic() - started) * 1000)
    ready = (
        invariant_checks == turns
        and final.ready
        and sum(outcomes.values()) == turns
        and all(outcomes[f"cancelled_{stage.lower()}"] > 0 for stage in SYNTHETIC_CANCELLATION_STAGES)
    )
    return {
        "cancellation_stage_counts": {
            stage: outcomes[f"cancelled_{stage.lower()}"]
            for stage in SYNTHETIC_CANCELLATION_STAGES
        },
        "completed_turn_count": outcomes["completed"],
        "elapsed_ms": elapsed_ms,
        "invariant_check_count": invariant_checks,
        "invariant_violation_count": final.invariant_violation_count,
        "maximum_database_bytes": maximum_database_bytes,
        "ready": ready,
        "turn_count": turns,
        "version": "synthetic-lifecycle-ready-gate/1",
    }


def run_sqlite_recovery_gate(database_path: str | Path) -> dict[str, Any]:
    """Exercise corruption detection and restore against disposable copies.

    The configured database is read through SQLite's online backup API but is
    never replaced or damaged by this drill.
    """

    source = Path(database_path).resolve()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="jarvis-recovery-gate-") as directory:
        root = Path(directory)
        backup = root / "verified-backup.sqlite3"
        incident = root / "incident.sqlite3"
        service_lock = root / "offline.lock"
        backup_report = backup_database(source, backup)
        try:
            create_or_replace_private_file(
                incident,
                b"intentional-corruption-fixture",
            )
        except PrivateStateError as exc:
            raise OperationalError(exc.code) from exc
        corruption_detected = False
        try:
            database_diagnostics(incident)
        except OperationalError:
            corruption_detected = True
        restore_report = restore_database(
            backup,
            incident,
            service_lock=service_lock,
        )
        restored = database_diagnostics(incident)
    return {
        "backup_integrity_ok": backup_report["integrity_ok"] is True,
        "corruption_detected": corruption_detected,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "production_database_mutated": False,
        "ready": (
            backup_report["integrity_ok"] is True
            and corruption_detected
            and restore_report["integrity_ok"] is True
            and restored.ready
        ),
        "restore_integrity_ok": restore_report["integrity_ok"] is True,
        "schema_version": restored.schema_version,
        "version": "sqlite-recovery-ready-gate/1",
    }


async def run_live_sequential_gate(
    *,
    host: str,
    port: int,
    auth_key: Any,
    turns: int = 100,
    timeout_seconds: float = 120.0,
    maximum_database_bytes: int | None = None,
    database_path: str | Path | None = None,
    request_text: str | None = None,
    include_latency_profile: bool = False,
) -> dict[str, Any]:
    """Run real sequential local-model turns through the voice WebSocket.

    This validates the wire path through finalized speech commits. Actual
    speaker playback remains a voice-runtime qualification signal.
    """

    if host != "127.0.0.1" or isinstance(turns, bool) or not 1 <= turns <= 10_000:
        raise ValueError("live gate parameters are invalid")
    if request_text is not None and (
        not isinstance(request_text, str)
        or not request_text.strip()
        or len(request_text) > 256
    ):
        raise ValueError("live gate request text is invalid")
    try:
        from websockets.asyncio.client import connect
        from jarvis_voice.protocol import Envelope, EventFactory
        from jarvis_voice.voice_auth_transport import authenticate_async_client
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise OperationalError("live_gate_dependency_unavailable") from exc
    started = time.monotonic()
    session_id = str(uuid.uuid4())
    factory = EventFactory(session_id=session_id)
    seen_event_ids: set[str] = set()
    response_count = 0
    speech_unit_count = 0
    maximum_turn_ms = 0
    latency_samples: dict[str, list[int]] = {
        "transcript_to_ack": [],
        "transcript_to_response_started": [],
        "transcript_to_first_text_delta": [],
        "transcript_to_first_speech_commit": [],
        "transcript_to_speech_done": [],
        "ack_to_first_speech_commit": [],
    }
    uri = f"ws://{host}:{port}"
    async with connect(
        uri,
        compression=None,
        proxy=None,
        open_timeout=2,
        close_timeout=1,
        max_size=64 * 1024,
        max_queue=32,
    ) as websocket:
        hello = factory.create(
            "client.hello",
            {
                "capabilities": ["speech_commits", "speech_done", "output_epochs"],
                "protocol": "jarvis.voice/1",
            },
        )
        await authenticate_async_client(websocket, auth_key, hello.to_json())
        ready = Envelope.from_json(await asyncio.wait_for(websocket.recv(), timeout=2.0))
        if ready.type != "server.ready":
            raise OperationalError("live_gate_handshake_failed")
        for index in range(turns):
            turn_started = time.monotonic()
            payload = _synthetic_payload(index)
            if request_text is not None:
                payload = {**payload, "text": request_text}
            transcript = factory.create(
                "input.transcript.final",
                payload,
                turn_id=str(uuid.uuid4()),
            )
            await websocket.send(transcript.to_json())
            accepted = False
            started_response = False
            text_done = False
            text_chunks = 0
            speech_units = 0
            response_id: str | None = None
            markers: dict[str, float] = {}
            while True:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                event = Envelope.from_json(raw)
                if event.event_id in seen_event_ids:
                    raise OperationalError("duplicate_server_publication")
                seen_event_ids.add(event.event_id)
                if event.type == "server.ack" and event.payload["acked_event_id"] == transcript.event_id:
                    accepted = True
                    markers.setdefault("ack", time.monotonic())
                    response_id = event.response_id
                    if response_id is None:
                        raise OperationalError("live_ack_binding_missing")
                elif event.type == "server.error" and accepted and event.response_id in {
                    None,
                    response_id,
                }:
                    raise OperationalError("live_turn_failed")
                elif response_id is not None and event.response_id == response_id:
                    if event.type == "response.started":
                        if started_response or text_chunks or speech_units:
                            raise OperationalError("live_started_order_violation")
                        started_response = True
                        markers.setdefault("response_started", time.monotonic())
                    elif event.type == "response.text.delta":
                        if (
                            not started_response
                            or text_done
                            or event.payload["chunk_index"] != text_chunks
                        ):
                            raise OperationalError("live_text_sequence_violation")
                        text_chunks += 1
                        markers.setdefault("first_text_delta", time.monotonic())
                    elif event.type == "response.speech.commit":
                        if (
                            not started_response
                            or event.payload["unit_index"] != speech_units + 1
                        ):
                            raise OperationalError("live_speech_sequence_violation")
                        speech_units += 1
                        markers.setdefault("first_speech_commit", time.monotonic())
                    elif event.type == "response.text.done":
                        if (
                            not started_response
                            or text_done
                            or event.payload["chunk_count"] != text_chunks
                        ):
                            raise OperationalError("live_text_terminal_violation")
                        text_done = True
                    elif event.type == "response.cancelled":
                        raise OperationalError("live_turn_failed")
                    elif event.type == "response.speech.done":
                        if (
                            not accepted
                            or not started_response
                            or not text_done
                            or event.payload["last_unit_index"] != speech_units
                        ):
                            raise OperationalError("live_terminal_violation")
                        speech_done_at = time.monotonic()
                        markers["speech_done"] = speech_done_at
                        required_markers = {
                            "ack", "response_started", "first_text_delta",
                            "first_speech_commit", "speech_done",
                        }
                        if not required_markers.issubset(markers):
                            raise OperationalError("live_latency_sequence_incomplete")
                        for name in (
                            "ack", "response_started", "first_text_delta",
                            "first_speech_commit", "speech_done",
                        ):
                            latency_samples[f"transcript_to_{name}"].append(
                                int(round((markers[name] - turn_started) * 1000))
                            )
                        latency_samples["ack_to_first_speech_commit"].append(
                            int(round((markers["first_speech_commit"] - markers["ack"]) * 1000))
                        )
                        playback = factory.create(
                            "playback.completed",
                            {"epoch": 0},
                            turn_id=transcript.turn_id,
                            response_id=response_id,
                        )
                        await websocket.send(playback.to_json())
                        response_count += 1
                        speech_unit_count += speech_units
                        maximum_turn_ms = max(
                            maximum_turn_ms, int((time.monotonic() - turn_started) * 1000)
                        )
                        break
    diagnostics = database_diagnostics(database_path) if database_path is not None else None
    bounded = (
        diagnostics is None
        or maximum_database_bytes is None
        or diagnostics.database_bytes + diagnostics.wal_bytes + diagnostics.shm_bytes
        <= maximum_database_bytes
    )
    report = {
        "database_bounded": bounded,
        "duplicate_publication_count": 0,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "maximum_turn_ms": maximum_turn_ms,
        "physical_speaker_playback_observed": False,
        "playback_completion_mode": "protocol_probe",
        "ready": response_count == turns and bounded and (diagnostics is None or diagnostics.ready),
        "response_count": response_count,
        "speech_unit_count": speech_unit_count,
        "turn_count": turns,
        "voice_auth_protocol": "jarvis.voice-auth/1",
        "version": "live-sequential-ready-gate/2",
    }
    if include_latency_profile:
        def distribution(values: list[int]) -> dict[str, int]:
            ordered = sorted(values)
            return {
                "count": len(ordered),
                "p50": ordered[max(0, math.ceil(len(ordered) * 0.50) - 1)],
                "p95": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
                "maximum": ordered[-1],
            }

        report["latency_profile_ms"] = {
            name: distribution(values) for name, values in latency_samples.items()
        }
    return report


async def run_live_model_failure_gate(
    *,
    host: str,
    port: int,
    auth_key: Any,
    timeout_seconds: float,
    expected_error_code: str = "backend_failure",
) -> dict[str, Any]:
    """Verify the exact terminal wire contract while local inference is down.

    The caller owns stopping and restarting the inference appliance. This
    probe only admits one bounded turn and reports content-free counts/codes.
    """

    if (
        host != "127.0.0.1"
        or isinstance(timeout_seconds, bool)
        or not 0 < timeout_seconds <= 120
        or expected_error_code != "backend_failure"
    ):
        raise ValueError("model failure gate parameters are invalid")
    try:
        from websockets.asyncio.client import connect
        from jarvis_voice.protocol import Envelope, EventFactory
        from jarvis_voice.voice_auth_transport import authenticate_async_client
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise OperationalError("live_gate_dependency_unavailable") from exc

    started = time.monotonic()
    factory = EventFactory(session_id=str(uuid.uuid4()))
    uri = f"ws://{host}:{port}"
    event_types: list[str] = []
    event_ids: set[str] = set()
    error_code: str | None = None
    reason_code: str | None = None
    response_id: str | None = None
    async with connect(
        uri,
        compression=None,
        proxy=None,
        open_timeout=2,
        close_timeout=1,
        max_size=64 * 1024,
        max_queue=32,
    ) as websocket:
        hello = factory.create(
            "client.hello",
            {
                "capabilities": [
                    "speech_commits",
                    "speech_done",
                    "output_epochs",
                ],
                "protocol": "jarvis.voice/1",
            },
        )
        await authenticate_async_client(websocket, auth_key, hello.to_json())
        ready = Envelope.from_json(
            await asyncio.wait_for(websocket.recv(), timeout=2.0)
        )
        if ready.type != "server.ready":
            raise OperationalError("failure_gate_handshake_failed")
        transcript = factory.create(
            "input.transcript.final",
            {
                **_synthetic_payload(0),
                "text": "Reply exactly: local failure gate.",
            },
            turn_id=str(uuid.uuid4()),
        )
        await websocket.send(transcript.to_json())
        deadline = time.monotonic() + timeout_seconds + 2.0
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            event = Envelope.from_json(
                await asyncio.wait_for(websocket.recv(), timeout=remaining)
            )
            if event.event_id in event_ids:
                raise OperationalError("duplicate_server_publication")
            event_ids.add(event.event_id)
            if (
                event.type == "server.ack"
                and event.payload["acked_event_id"] == transcript.event_id
            ):
                response_id = event.response_id
                event_types.append(event.type)
                continue
            if response_id is None or event.response_id != response_id:
                continue
            event_types.append(event.type)
            if event.type == "server.error":
                error_code = event.payload["code"]
            elif event.type == "response.cancelled":
                reason_code = event.payload["reason_code"]
                break

    expected_order = ["server.ack", "server.error", "response.cancelled"]
    forbidden_output = {
        "response.started",
        "response.text.delta",
        "response.text.done",
        "response.speech.commit",
        "response.speech.done",
    }
    bounded = (time.monotonic() - started) <= timeout_seconds + 2.0
    return {
        "bounded_timeout": bounded,
        "duplicate_publication_count": 0,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "error_code": error_code,
        "event_count": len(event_types),
        "failure_deterministic": event_types == expected_order,
        "late_output_count": sum(item in forbidden_output for item in event_types),
        "local_only": True,
        "ready": (
            bounded
            and event_types == expected_order
            and error_code == expected_error_code
            and reason_code == "backend_failure"
        ),
        "reason_code": reason_code,
        "voice_auth_protocol": "jarvis.voice-auth/1",
        "version": "live-model-failure-ready-gate/2",
    }


async def run_live_soak_gate(
    *,
    host: str,
    port: int,
    auth_key: Any,
    database_path: str | Path,
    duration_seconds: int = 86_400,
    batch_turns: int = 10,
    maximum_database_bytes: int = 512 * 1024 * 1024,
) -> dict[str, Any]:
    """Run bounded sequential batches until the requested wall-clock duration."""

    if duration_seconds < 60 or batch_turns < 1:
        raise ValueError("soak bounds are invalid")
    started = time.monotonic()
    batches = 0
    turns = 0
    maximum_observed_bytes = 0
    operations_state = operational_paths(database_path)["state"]
    maximum_observed_log_bytes = 0
    qualification_log = ContentFreeOperationalLog(operations_state / "soak.operations.jsonl")
    try:
        qualification_log.emit("soak_started", duration_seconds=duration_seconds)
        while time.monotonic() - started < duration_seconds:
            report = await run_live_sequential_gate(
                host=host,
                port=port,
                auth_key=auth_key,
                turns=batch_turns,
                database_path=database_path,
                maximum_database_bytes=maximum_database_bytes,
            )
            if not report["ready"]:
                raise OperationalError("soak_batch_failed")
            diagnostics = database_diagnostics(database_path)
            observed = diagnostics.database_bytes + diagnostics.wal_bytes + diagnostics.shm_bytes
            maximum_observed_bytes = max(maximum_observed_bytes, observed)
            if observed > maximum_database_bytes:
                raise OperationalError("soak_database_bound_exceeded")
            observed_logs = sum(
                item.stat().st_size
                for item in operations_state.glob("*.jsonl*")
                if item.is_file()
            )
            maximum_observed_log_bytes = max(
                maximum_observed_log_bytes, observed_logs
            )
            if observed_logs > MAXIMUM_OPERATIONAL_LOG_BYTES:
                raise OperationalError("soak_log_bound_exceeded")
            batches += 1
            turns += batch_turns
            qualification_log.emit(
                "soak_batch_completed",
                batch_count=batches,
                database_bytes=observed,
                invariant_violation_count=diagnostics.invariant_violation_count,
                turn_count=turns,
            )
            await asyncio.sleep(0.05)
        final = database_diagnostics(database_path)
        elapsed = int(time.monotonic() - started)
        qualification_log.emit(
            "soak_completed",
            duration_seconds=elapsed,
            turn_count=turns,
        )
    finally:
        qualification_log.close()
    return {
        "batch_count": batches,
        "database_bounded": maximum_observed_bytes <= maximum_database_bytes,
        "duration_seconds": elapsed,
        "invariant_violation_count": final.invariant_violation_count,
        "maximum_database_bytes": maximum_observed_bytes,
        "maximum_operational_log_bytes": maximum_observed_log_bytes,
        "operational_log_bounded": (
            maximum_observed_log_bytes <= MAXIMUM_OPERATIONAL_LOG_BYTES
        ),
        "physical_speaker_playback_observed": False,
        "playback_completion_mode": "protocol_probe",
        "ready": (
            elapsed >= duration_seconds
            and final.ready
            and maximum_observed_log_bytes <= MAXIMUM_OPERATIONAL_LOG_BYTES
        ),
        "turn_count": turns,
        "voice_auth_protocol": "jarvis.voice-auth/1",
        "version": "live-soak-ready-gate/2",
    }


async def run_live_restart_gate(
    *,
    host: str,
    port: int,
    auth_key: Any,
    stop_service: Callable[[], object],
    start_service: Callable[[], object],
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Restart the installed daemon during generation and pending playback."""

    if host != "127.0.0.1":
        raise ValueError("restart gate requires loopback")
    try:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed
        from jarvis_voice.protocol import Envelope, EventFactory
        from jarvis_voice.voice_auth import VoiceAuthError
        from jarvis_voice.voice_auth_transport import authenticate_async_client
    except ImportError as exc:  # pragma: no cover
        raise OperationalError("live_gate_dependency_unavailable") from exc
    uri = f"ws://{host}:{port}"
    session_id = str(uuid.uuid4())

    async def open_connection():
        deadline = time.monotonic() + 15.0
        while True:
            websocket = None
            try:
                websocket = await connect(
                    uri,
                    compression=None,
                    proxy=None,
                    open_timeout=2,
                    close_timeout=1,
                    max_size=64 * 1024,
                    max_queue=32,
                )
                factory = EventFactory(session_id=session_id)
                hello = factory.create(
                    "client.hello",
                    {
                        "capabilities": [
                            "speech_commits",
                            "speech_done",
                            "output_epochs",
                        ],
                        "protocol": "jarvis.voice/1",
                    },
                )
                await authenticate_async_client(
                    websocket,
                    auth_key,
                    hello.to_json(),
                )
                ready = Envelope.from_json(
                    await asyncio.wait_for(websocket.recv(), timeout=2.0)
                )
                if ready.type != "server.ready":
                    raise OperationalError("restart_gate_handshake_failed")
                return websocket, factory
            except (ConnectionClosed, OSError, TimeoutError, VoiceAuthError):
                if websocket is not None:
                    with suppress(Exception):
                        await websocket.close()
                if time.monotonic() >= deadline:
                    raise OperationalError("restart_gate_reconnect_timeout")
                await asyncio.sleep(0.1)

    websocket, factory = await open_connection()
    generation_interrupted = False
    playback_replay_exact = False
    duplicate_publications = 0
    try:
        generation_turn = factory.create(
            "input.transcript.final",
            _synthetic_payload(1),
            turn_id=str(uuid.uuid4()),
        )
        await websocket.send(generation_turn.to_json())
        generation_response_id: str | None = None
        while generation_response_id is None:
            event = Envelope.from_json(
                await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
            )
            if (
                event.type == "server.ack"
                and event.payload["acked_event_id"] == generation_turn.event_id
            ):
                generation_response_id = event.response_id
        await asyncio.to_thread(stop_service)
        with suppress(Exception):
            await websocket.close()
        await asyncio.to_thread(start_service)
        websocket, factory = await open_connection()
        while True:
            event = Envelope.from_json(
                await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
            )
            if event.response_id != generation_response_id:
                continue
            if event.type == "response.cancelled":
                generation_interrupted = (
                    event.payload["reason_code"] == "service_interrupted"
                )
                break
            if event.type == "response.speech.done":
                raise OperationalError("restart_generation_completed_before_stop")

        playback_turn = factory.create(
            "input.transcript.final",
            _synthetic_payload(2),
            turn_id=str(uuid.uuid4()),
        )
        await websocket.send(playback_turn.to_json())
        playback_response_id: str | None = None
        original: dict[str, tuple[str, str, str]] = {}
        while True:
            event = Envelope.from_json(
                await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
            )
            if (
                event.type == "server.ack"
                and event.payload["acked_event_id"] == playback_turn.event_id
            ):
                playback_response_id = event.response_id
            if playback_response_id is None or event.response_id != playback_response_id:
                continue
            semantic = (
                event.type,
                event.sent_at,
                str(sorted(event.payload.items())),
            )
            if event.event_id in original:
                duplicate_publications += 1
            original[event.event_id] = semantic
            if event.type == "response.cancelled":
                raise OperationalError("restart_playback_turn_failed")
            if event.type == "response.speech.done":
                break
        await asyncio.to_thread(stop_service)
        with suppress(Exception):
            await websocket.close()
        await asyncio.to_thread(start_service)
        websocket, factory = await open_connection()
        replay: dict[str, tuple[str, str, str]] = {}
        while True:
            event = Envelope.from_json(
                await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
            )
            if event.response_id != playback_response_id:
                continue
            semantic = (
                event.type,
                event.sent_at,
                str(sorted(event.payload.items())),
            )
            if event.event_id in replay:
                duplicate_publications += 1
            replay[event.event_id] = semantic
            if event.type == "response.speech.done":
                break
        playback_replay_exact = replay == original
        await websocket.send(
            factory.create(
                "playback.completed",
                {"epoch": 0},
                turn_id=playback_turn.turn_id,
                response_id=playback_response_id,
            ).to_json()
        )
    finally:
        with suppress(Exception):
            await websocket.close()
    return {
        "duplicate_publication_count": duplicate_publications,
        "generation_interrupted": generation_interrupted,
        "playback_replay_exact": playback_replay_exact,
        "ready": generation_interrupted
        and playback_replay_exact
        and duplicate_publications == 0,
        "voice_auth_protocol": "jarvis.voice-auth/1",
        "version": "live-restart-ready-gate/2",
    }


def evaluate_operational_ready_gate(
    *,
    synthetic_evidence: str | Path,
    live_evidence: str | Path,
    restart_evidence: str | Path,
    recovery_evidence: str | Path,
    soak_evidence: str | Path,
    native_voice_evidence: str | Path,
    verifier: ReadinessEvidenceVerifier,
    expected_identity: ReadinessIdentityV2,
    aggregate_signer: ReadinessEvidenceSigner,
    cohort_state_directory: str | Path,
    current_state_provider: Callable[[], tuple[ReadinessIdentityV2, Any]],
    final_time_provider: Callable[[], datetime] | None = None,
    now: datetime | None = None,
    aggregate_freshness_seconds: int = 24 * 60 * 60,
) -> ReadinessEvidenceV2:
    """Verify six independently signed leaves and sign their exact aggregate."""

    selected_now = now or datetime.now(UTC)
    paths = {
        SYNTHETIC_KIND: synthetic_evidence,
        LIVE_KIND: live_evidence,
        RESTART_KIND: restart_evidence,
        RECOVERY_KIND: recovery_evidence,
        SOAK_KIND: soak_evidence,
        NATIVE_KIND: native_voice_evidence,
    }
    verified = {
        kind: verify_readiness_evidence(
            path,
            verifier=verifier,
            expected_kind=kind,
            expected_identity=expected_identity,
            now=selected_now,
        )
        for kind, path in paths.items()
    }
    qualifications = [verified[kind].qualification for kind in paths]
    cohort_ids = {item.cohort_id for item in qualifications}
    window_starts = {item.window_started_at_utc for item in qualifications}
    window_expires = {item.window_expires_at_utc for item in qualifications}
    sequences = {item.sequence for item in qualifications}
    captured_by_sequence = sorted(
        (
            evidence.qualification.sequence,
            datetime.strptime(
                evidence.captured_at_utc,
                "%Y-%m-%dT%H:%M:%SZ",
            ).replace(tzinfo=UTC),
        )
        for evidence in verified.values()
    )
    if (
        len(cohort_ids) != 1
        or len(window_starts) != 1
        or len(window_expires) != 1
        or len(sequences) != len(paths)
        or [sequence for sequence, _ in captured_by_sequence]
        != list(range(1, len(paths) + 1))
        or any(
            later < earlier
            for (_, earlier), (_, later) in pairwise(captured_by_sequence)
        )
    ):
        raise OperationalError("readiness_qualification_cohort_invalid")
    validate_readiness_cohort_receipt(
        cohort_state_directory,
        {kind: (paths[kind], verified[kind]) for kind in paths},
    )
    synthetic = verified[SYNTHETIC_KIND].payload
    live = verified[LIVE_KIND].payload
    restart = verified[RESTART_KIND].payload
    recovery = verified[RECOVERY_KIND].payload
    soak = verified[SOAK_KIND].payload
    native = verified[NATIVE_KIND].payload

    checks = {
        "synthetic_lifecycle": (
            synthetic.get("version") == "synthetic-lifecycle-ready-gate/1"
            and synthetic.get("ready") is True
            and isinstance(synthetic.get("turn_count"), int)
            and synthetic["turn_count"] >= 1000
            and synthetic.get("invariant_check_count") == synthetic["turn_count"]
            and synthetic.get("invariant_violation_count") == 0
            and synthetic.get("completed_turn_count")
            + sum(synthetic["cancellation_stage_counts"].values())
            == synthetic["turn_count"]
            and all(
                count > 0
                for count in synthetic["cancellation_stage_counts"].values()
            )
        ),
        "sequential_local_model": (
            live.get("version") == "live-sequential-ready-gate/2"
            and live.get("voice_auth_protocol") == "jarvis.voice-auth/1"
            and live.get("ready") is True
            and isinstance(live.get("turn_count"), int)
            and live["turn_count"] >= 100
            and live.get("response_count") == live["turn_count"]
            and live.get("speech_unit_count", 0) >= live["response_count"]
            and live.get("database_bounded") is True
            and live.get("duplicate_publication_count") == 0
        ),
        "restart_generation_playback": (
            restart.get("version") == "live-restart-ready-gate/2"
            and restart.get("voice_auth_protocol") == "jarvis.voice-auth/1"
            and restart.get("ready") is True
            and restart.get("generation_interrupted") is True
            and restart.get("playback_replay_exact") is True
            and restart.get("duplicate_publication_count") == 0
        ),
        "sqlite_recovery": (
            recovery.get("version") == "sqlite-recovery-ready-gate/1"
            and recovery.get("ready") is True
            and recovery.get("backup_integrity_ok") is True
            and recovery.get("corruption_detected") is True
            and recovery.get("production_database_mutated") is False
            and recovery.get("restore_integrity_ok") is True
            and recovery.get("schema_version")
            == expected_identity.database_schema_version
        ),
        "twenty_four_hour_soak": (
            soak.get("version") == "live-soak-ready-gate/2"
            and soak.get("voice_auth_protocol") == "jarvis.voice-auth/1"
            and soak.get("ready") is True
            and isinstance(soak.get("duration_seconds"), int)
            and soak["duration_seconds"] >= 86_400
            and soak.get("database_bounded") is True
            and soak.get("operational_log_bounded") is True
            and soak.get("invariant_violation_count") == 0
            and soak.get("batch_count", 0) > 0
            and soak.get("turn_count", 0) > 0
        ),
        "native_voice_playback": native.get("ready") is True,
    }
    payload = {
        "checks": {name: {"ready": passed} for name, passed in sorted(checks.items())},
        "content_captured": False,
        "ready": all(checks.values()),
        "version": READINESS_AGGREGATE_VERSION,
    }
    if current_state_provider is None:
        raise OperationalError("readiness_current_state_required")
    try:
        current_identity, current_database_diagnostics = current_state_provider()
    except OperationalError:
        raise
    except Exception:  # noqa: BLE001 - injected current-state failures fail closed
        raise OperationalError("readiness_current_state_unavailable") from None
    if current_identity != expected_identity:
        raise OperationalError("readiness_identity_changed_during_gate")
    try:
        final_now = (
            final_time_provider()
            if final_time_provider is not None
            else datetime.now(UTC)
        )
    except Exception:  # noqa: BLE001 - injected clock failures fail closed
        raise OperationalError("readiness_final_time_unavailable") from None
    if final_now.tzinfo is None:
        raise OperationalError("readiness_final_time_invalid")
    final_now = final_now.astimezone(UTC)
    last_leaf_capture = captured_by_sequence[-1][1]
    aggregate_capture = last_leaf_capture + timedelta(seconds=1)
    window_expiry = datetime.strptime(
        next(iter(window_expires)),
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=UTC)
    if aggregate_capture > final_now or final_now >= window_expiry:
        raise OperationalError("readiness_qualification_window_invalid")
    reverified = {
        kind: verify_readiness_evidence(
            path,
            verifier=verifier,
            expected_kind=kind,
            expected_identity=expected_identity,
            now=aggregate_capture,
        )
        for kind, path in paths.items()
    }
    if any(
        reverified[kind].evidence_sha256 != verified[kind].evidence_sha256
        for kind in paths
    ):
        raise OperationalError("readiness_parent_evidence_changed")
    current_diagnostics_document = validated_readiness_database_diagnostics(
        current_database_diagnostics,
        expected_schema_version=current_identity.database_schema_version,
    )
    current_diagnostics_bytes = canonical_json_bytes(current_diagnostics_document)
    if current_diagnostics_bytes != canonical_json_bytes(
        dict(reverified[NATIVE_KIND].database_diagnostics)
    ):
        raise OperationalError("readiness_database_state_changed")
    aggregate = build_readiness_evidence(
        evidence_kind=AGGREGATE_KIND,
        payload=payload,
        identity=current_identity,
        database_diagnostics=current_database_diagnostics,
        qualification=qualification_binding_for_kind(
            AGGREGATE_KIND,
            cohort_id=next(iter(cohort_ids)),
            window_started_at=next(iter(window_starts)),
        ),
        signer=aggregate_signer,
        captured_at=aggregate_capture,
        freshness_seconds=aggregate_freshness_seconds,
        parent_evidence_sha256s={
            kind: evidence.evidence_sha256 for kind, evidence in reverified.items()
        },
    )
    return aggregate


__all__ = [
    "evaluate_operational_ready_gate",
    "run_live_restart_gate",
    "run_live_sequential_gate",
    "run_live_soak_gate",
    "run_sqlite_recovery_gate",
    "run_synthetic_lifecycle_gate",
    "write_readiness_report",
    "write_signed_readiness_report",
]
