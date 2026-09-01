#!/usr/bin/env python3
"""Run a content-free remember/recall/forget/recall-miss live probe.

All mutations go through the public voice/control surfaces.  The disposable
SQLite database is opened read-only only to verify the memory/audit contract.
No transcript, response, identifier, bearer token, database path, or memory
content is printed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from jarvis_voice.protocol import Envelope, EventFactory
from jarvis_voice.voice_auth import load_voice_auth_key
from jarvis_voice.voice_auth_transport import authenticate_async_client

_CLOSED_FAILURE_CODES = frozenset(
    {
        "probe_failed",
        "probe_not_accepted",
        "probe_response_unidentified",
        "probe_target_unsafe",
        *(
            f"phase_{phase}_{suffix}"
            for phase in range(1, 5)
            for suffix in (
                "evidence_timeout",
                "response_cancelled_backend_failure",
                "response_cancelled_internal_failure",
                "response_cancelled_other",
                "server_error_backend_failure",
                "server_error_internal_failure",
                "server_error_other",
            )
        ),
    }
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="run-live-memory-crud-probe")
    result.add_argument("--url", default="ws://127.0.0.1:8865")
    result.add_argument("--control-url", default="http://127.0.0.1:8867")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--database", type=Path, required=True)
    result.add_argument("--timeout", type=float, default=180.0)
    return result


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_only(
    database: Path, sql: str, values: tuple[object, ...] = ()
) -> list[sqlite3.Row]:
    uri = (
        "file:"
        + urllib.parse.quote(database.resolve().as_posix(), safe="/")
        + "?mode=ro"
    )
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        return connection.execute(sql, values).fetchall()
    finally:
        connection.close()


def _task_evidence(database: Path, turn_id: str) -> dict[str, Any]:
    task = _read_only(
        database,
        """SELECT t.task_id,t.state,t.epoch,t.conversation_id
           FROM tasks t JOIN canonical_turns c USING(task_id)
           WHERE c.turn_id=?""",
        (turn_id,),
    )
    if len(task) != 1:
        raise RuntimeError("probe task identity is missing or ambiguous")
    selected = task[0]
    events = _read_only(
        database,
        """SELECT event_type,payload_json FROM task_events
           WHERE task_id=? AND (
             event_type LIKE 'memory.recall.%'
             OR event_type LIKE 'memory.delete.%'
             OR event_type LIKE 'memory.command.%'
           ) ORDER BY event_sequence""",
        (selected["task_id"],),
    )
    recall = next(
        (row for row in events if str(row["event_type"]).startswith("memory.recall.")),
        None,
    )
    deletion = next(
        (row for row in events if str(row["event_type"]).startswith("memory.delete.")),
        None,
    )
    command = next(
        (row for row in events if str(row["event_type"]).startswith("memory.command.")),
        None,
    )
    recall_payload = {} if recall is None else json.loads(recall["payload_json"])
    delete_payload = {} if deletion is None else json.loads(deletion["payload_json"])
    command_payload = {} if command is None else json.loads(command["payload_json"])
    memory_count = _read_only(
        database,
        "SELECT COUNT(*) AS n FROM memory_records WHERE conversation_id=?",
        (selected["conversation_id"],),
    )[0]["n"]
    return {
        "task_id": selected["task_id"],
        "conversation_id": selected["conversation_id"],
        "state": selected["state"],
        "epoch": selected["epoch"],
        "memory_count": int(memory_count),
        "recall_event": None if recall is None else recall["event_type"],
        "recall_count": int(recall_payload.get("result_count", 0)),
        "command_event": None if command is None else command["event_type"],
        "command_status": command_payload.get("status"),
        "command_changed": command_payload.get("changed"),
        "command_matched": int(command_payload.get("matched_count", 0)),
        "delete_event": None if deletion is None else deletion["event_type"],
        "delete_matched": int(delete_payload.get("matched_count", 0)),
        "delete_count": int(delete_payload.get("deleted_count", 0)),
        "delete_selection": delete_payload.get("selection"),
        "delete_authoritative": delete_payload.get("authoritative_state_changed"),
        "delete_target_sha256": delete_payload.get("target_sha256"),
    }


def _wait_evidence(
    database: Path,
    turn_id: str,
    phase: int,
    timeout: float,
    expected: tuple[str | None, int, str | None, str | None, int, int],
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            evidence = _task_evidence(database, turn_id)
            (
                recall_event,
                recall_count,
                command_event,
                delete_event,
                delete_count,
                memory_count,
            ) = expected
            if (
                evidence["state"] == "COMPLETED"
                and evidence["recall_event"] == recall_event
                and evidence["recall_count"] == recall_count
                and evidence["command_event"] == command_event
                and evidence["delete_event"] == delete_event
                and evidence["delete_count"] == delete_count
                and evidence["memory_count"] == memory_count
            ):
                return evidence
        except (RuntimeError, sqlite3.Error) as exc:
            last = exc
        time.sleep(0.05)
    raise RuntimeError(f"phase_{phase}_evidence_timeout") from last


async def _turn(
    websocket: Any,
    factory: EventFactory,
    *,
    text: str,
    marker: str,
    phase: int,
    timeout: float,
) -> dict[str, Any]:
    turn_id = str(uuid.uuid4())
    transcript = factory.create(
        "input.transcript.final",
        {
            "text": text,
            "language": {"code": "en", "score": 1.0},
            "asr_score": {
                "value": 1.0,
                "method": "manual_memory_crud_probe",
                "raw_avg_logprob": 0.0,
                "calibrated": False,
            },
        },
        turn_id=turn_id,
    )
    accepted = False
    response_id: str | None = None
    chunks: list[str] = []
    event_count = 0
    await websocket.send(transcript.to_json())
    async with asyncio.timeout(timeout):
        while True:
            event = Envelope.from_json(await websocket.recv())
            if event.type == "server.error":
                raw_code = event.payload.get("code")
                code = (
                    raw_code
                    if raw_code in {"backend_failure", "internal_failure"}
                    else "other"
                )
                raise RuntimeError(f"phase_{phase}_server_error_{code}")
            if (
                event.type == "server.ack"
                and event.payload.get("acked_event_id") == transcript.event_id
            ):
                accepted = event.payload.get("status") == "accepted"
                continue
            if event.turn_id != turn_id:
                continue
            event_count += 1
            response_id = event.response_id or response_id
            if event.type == "response.text.delta":
                chunks.append(str(event.payload["text"]))
            if event.type == "response.speech.done":
                break
            if event.type == "response.cancelled":
                raw_reason = event.payload.get("reason_code")
                reason = (
                    raw_reason
                    if raw_reason in {"backend_failure", "internal_failure"}
                    else "other"
                )
                raise RuntimeError(f"phase_{phase}_response_cancelled_{reason}")
    if not accepted:
        raise RuntimeError("probe_not_accepted")
    if response_id is None:
        raise RuntimeError("probe_response_unidentified")
    await websocket.send(
        factory.create("playback.completed", {}, response_id=response_id).to_json()
    )
    response = unicodedata.normalize("NFC", "".join(chunks))
    return {
        "turn_id": turn_id,
        "accepted": accepted,
        "response_contains_marker": marker.casefold() in response.casefold(),
        "response_sha256": _sha256(response),
        "response_bytes": len(response.encode("utf-8")),
        "wire_event_count": event_count,
    }


def _delete_conversation(
    control_url: str, config: Path, conversation_id: str, epoch: int
) -> int:
    document = json.loads(config.read_text(encoding="utf-8"))
    token = document["control_bearer_token"]
    request = urllib.request.Request(
        f"{control_url}/v1/conversations/{conversation_id}",
        method="DELETE",
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": str(uuid.uuid4()),
            "If-Match-Epoch": str(epoch),
        },
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        response.read()
        return response.status


def _cancel_task(control_url: str, config: Path, task_id: str, epoch: int) -> int:
    document = json.loads(config.read_text(encoding="utf-8"))
    encoded = b'{"reason_code":"user_cancelled"}'
    request = urllib.request.Request(
        f"{control_url}/v1/tasks/{task_id}/cancel",
        data=encoded,
        method="POST",
        headers={
            "Authorization": f"Bearer {document['control_bearer_token']}",
            "Content-Type": "application/json",
            "Idempotency-Key": str(uuid.uuid4()),
            "If-Match-Epoch": str(epoch),
        },
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        response.read()
        return response.status


def _best_effort_cleanup(args: argparse.Namespace) -> bool:
    try:
        _validate_disposable_target(args)
        conversations = _read_only(
            args.database,
            "SELECT conversation_id,current_epoch FROM conversations WHERE state='ACTIVE'",
        )
        if not conversations:
            return True
        if len(conversations) != 1:
            return False
        conversation_id = str(conversations[0]["conversation_id"])
        epoch = int(conversations[0]["current_epoch"])
        active = _read_only(
            args.database,
            """SELECT task_id FROM tasks WHERE conversation_id=? AND tombstoned=0
               AND state NOT IN ('COMPLETED','CANCELLED','INTERRUPTED',
                                 'BUDGET_EXHAUSTED','FAILED')""",
            (conversation_id,),
        )
        if len(active) > 1:
            return False
        if active:
            if (
                _cancel_task(
                    args.control_url, args.config, str(active[0]["task_id"]), epoch
                )
                != 200
            ):
                return False
            refreshed = _read_only(
                args.database,
                "SELECT current_epoch FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            )
            if len(refreshed) != 1:
                return False
            epoch = int(refreshed[0]["current_epoch"])
        if (
            _delete_conversation(args.control_url, args.config, conversation_id, epoch)
            != 204
        ):
            return False
        return not _read_only(
            args.database,
            "SELECT 1 FROM conversations WHERE conversation_id=?",
            (conversation_id,),
        )
    except Exception:  # noqa: BLE001 - cleanup must fail closed for every provider error
        return False


def _authority_counts(database: Path) -> dict[str, int]:
    tables = (
        "approvals",
        "capability_reservations",
        "cloud_dispatch_permits",
        "cloud_provider_receipts",
        "research_runs",
        "scheduler_jobs",
        "scheduler_occurrences",
        "scheduler_proposals",
        "tool_executions",
        "memory_records",
    )
    counts = {
        table: int(_read_only(database, f"SELECT COUNT(*) AS n FROM {table}")[0]["n"])
        for table in tables
    }
    counts["canonical_effects"] = int(
        _read_only(
            database,
            "SELECT COALESCE(SUM(json_array_length(effects_json)),0) AS n FROM canonical_turns",
        )[0]["n"]
    )
    return counts


def _canonical_effect_summary(
    database: Path, conversation_id: str
) -> list[tuple[str, str]]:
    rows = _read_only(
        database,
        """SELECT effects_json FROM canonical_turns
           WHERE conversation_id=? ORDER BY created_at,turn_id""",
        (conversation_id,),
    )
    result: list[tuple[str, str]] = []
    for row in rows:
        effects = json.loads(row["effects_json"])
        if not isinstance(effects, list):
            raise RuntimeError("memory CRUD canonical effects were invalid")
        for effect in effects:
            if not isinstance(effect, dict):
                raise RuntimeError("memory CRUD canonical effects were invalid")
            result.append((str(effect.get("kind")), str(effect.get("status"))))
    return result


_DISPOSABLE_ROOT = Path("/private/tmp")


def _validate_disposable_target(args: argparse.Namespace) -> None:
    database = args.database.resolve(strict=True)
    config = args.config.resolve(strict=True)
    disposable_root = _DISPOSABLE_ROOT.resolve(strict=True)
    try:
        database.relative_to(disposable_root)
        config.relative_to(disposable_root)
    except ValueError as exc:
        raise RuntimeError("probe target is not disposable") from exc
    document = json.loads(config.read_text(encoding="utf-8"))
    if Path(document["database_path"]).resolve() != database:
        raise RuntimeError("probe config/database mismatch")
    voice = urllib.parse.urlsplit(args.url)
    control = urllib.parse.urlsplit(args.control_url)
    if (
        voice.scheme != "ws"
        or voice.hostname != "127.0.0.1"
        or voice.port != int(document["voice_port"])
        or control.scheme != "http"
        or control.hostname != "127.0.0.1"
        or control.port != int(document["control_port"])
        or voice.port != 8865
        or control.port != 8867
    ):
        raise RuntimeError("probe endpoint is not disposable")
    if (
        document.get("memory_enabled") is not True
        or document.get("memory_shadow_enabled") is not False
        or document.get("research_enabled", False)
        or document.get("cloud_enabled", False)
        or document.get("tools_enabled", False)
    ):
        raise RuntimeError("probe capability boundary is unsafe")


def _validate_disposable_inputs(args: argparse.Namespace) -> None:
    _validate_disposable_target(args)
    database = args.database.resolve(strict=True)
    before = _authority_counts(database)
    conversation_count = int(
        _read_only(database, "SELECT COUNT(*) AS n FROM conversations")[0]["n"]
    )
    task_count = int(_read_only(database, "SELECT COUNT(*) AS n FROM tasks")[0]["n"])
    if conversation_count or task_count or any(before.values()):
        raise RuntimeError("probe database is not empty")


def _closed_failure_code(exc: BaseException) -> str:
    candidate = str(exc)
    if candidate in _CLOSED_FAILURE_CODES:
        return candidate
    if candidate == "memory CRUD probe was not accepted or identified":
        return "probe_not_accepted"
    return "probe_failed"


async def run(
    args: argparse.Namespace, *, disposable_inputs_validated: bool = False
) -> dict[str, Any]:
    from websockets.asyncio.client import connect

    if not disposable_inputs_validated:
        await asyncio.to_thread(_validate_disposable_inputs, args)
    marker = "jmemprobe" + uuid.uuid4().hex
    fact = f"my synthetic verification nonce is {marker}"
    texts = (
        f"Remember that {fact}.",
        "What is my synthetic verification nonce? Reply with only the nonce.",
        f"Forget that {fact}.",
        "Is there a stored synthetic verification nonce? Reply only yes or no.",
    )
    expected = (
        (None, 0, "memory.command.remember.stored", None, 0, 1),
        ("memory.recall.completed", 1, None, None, 0, 1),
        (None, 0, "memory.command.forget.deleted", "memory.delete.completed", 1, 0),
        ("memory.recall.empty", 0, None, None, 0, 0),
    )
    factory = EventFactory()
    turns: list[dict[str, Any]] = []
    before = _authority_counts(args.database)
    config_document = json.loads(args.config.read_text(encoding="utf-8"))
    auth_key = load_voice_auth_key(config_document["voice_auth_key_file"])
    async with connect(
        args.url,
        origin=None,
        compression=None,
        proxy=None,
        open_timeout=3,
        close_timeout=1,
        ping_interval=10,
        ping_timeout=10,
        max_size=64 * 1024,
    ) as websocket:
        hello = factory.create(
            "client.hello",
            {
                "protocol": "jarvis.voice/1",
                "capabilities": ["speech_commits", "speech_done", "output_epochs"],
            },
        )
        await authenticate_async_client(websocket, auth_key, hello.to_json())
        ready = Envelope.from_json(await websocket.recv())
        if ready.type != "server.ready":
            raise RuntimeError("memory CRUD probe did not receive server.ready")
        for phase, (text, phase_contract) in enumerate(
            zip(texts, expected, strict=True), start=1
        ):
            wire = await _turn(
                websocket,
                factory,
                text=text,
                marker=marker,
                phase=phase,
                timeout=args.timeout,
            )
            evidence = await asyncio.to_thread(
                _wait_evidence,
                args.database,
                str(wire.pop("turn_id")),
                phase,
                args.timeout,
                phase_contract,
            )
            turns.append({**wire, **evidence})
        await websocket.send(factory.create("client.goodbye").to_json())

    for index, (turn, contract) in enumerate(zip(turns, expected, strict=True)):
        (
            recall_event,
            recall_count,
            command_event,
            delete_event,
            delete_count,
            memory_count,
        ) = contract
        if (
            turn["state"] != "COMPLETED"
            or turn["recall_event"] != recall_event
            or turn["recall_count"] != recall_count
            or turn["command_event"] != command_event
            or turn["delete_event"] != delete_event
            or turn["delete_count"] != delete_count
            or turn["memory_count"] != memory_count
        ):
            raise RuntimeError(f"memory CRUD contract failed at phase {index + 1}")
    if (
        turns[0]["command_status"] != "stored"
        or turns[0]["command_changed"] is not True
        or turns[0]["command_matched"] != 0
    ):
        raise RuntimeError("memory CRUD remember command evidence was invalid")
    if turns[2]["delete_selection"] != "exact_normalized_content":
        raise RuntimeError("memory CRUD deletion selection was not deterministic")
    if (
        turns[2]["command_status"] != "deleted"
        or turns[2]["command_changed"] is not True
        or turns[2]["command_matched"] != 1
        or turns[2]["delete_matched"] != 1
        or turns[2]["delete_authoritative"] is not True
        or turns[2]["delete_target_sha256"] != _sha256(fact)
    ):
        raise RuntimeError("memory CRUD deletion provenance was invalid")
    conversation_ids = {str(turn["conversation_id"]) for turn in turns}
    if len(conversation_ids) != 1:
        raise RuntimeError("memory CRUD probe crossed conversation scope")
    conversation_id = conversation_ids.pop()
    epoch = int(turns[-1]["epoch"])
    after = _authority_counts(args.database)
    if any(
        after[name] != 0
        for name in (
            "approvals",
            "capability_reservations",
            "cloud_dispatch_permits",
            "cloud_provider_receipts",
            "scheduler_jobs",
            "scheduler_occurrences",
            "scheduler_proposals",
            "tool_executions",
            "research_runs",
        )
    ):
        raise RuntimeError("memory CRUD probe created an unrelated side effect")
    canonical_effects = _canonical_effect_summary(args.database, conversation_id)
    if after["canonical_effects"] != 2 or canonical_effects != [
        ("memory", "stored"),
        ("memory", "deleted"),
    ]:
        raise RuntimeError("memory CRUD canonical effects were invalid")
    status = await asyncio.to_thread(
        _delete_conversation, args.control_url, args.config, conversation_id, epoch
    )
    if status != 204:
        raise RuntimeError("memory CRUD cleanup did not return 204")
    scoped_after_cleanup = {
        "conversations": len(
            _read_only(
                args.database,
                "SELECT 1 FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            )
        ),
        "tasks": len(
            _read_only(
                args.database,
                "SELECT 1 FROM tasks WHERE conversation_id=?",
                (conversation_id,),
            )
        ),
        "turns": len(
            _read_only(
                args.database,
                "SELECT 1 FROM canonical_turns WHERE conversation_id=?",
                (conversation_id,),
            )
        ),
        "memory": len(
            _read_only(
                args.database,
                "SELECT 1 FROM memory_records WHERE conversation_id=?",
                (conversation_id,),
            )
        ),
    }
    if any(scoped_after_cleanup.values()):
        raise RuntimeError("memory CRUD cleanup left scoped authority rows")

    return {
        "probe_version": "jarvis.live-memory-crud.v1",
        "ready": True,
        "marker_sha256": _sha256(marker),
        "contains_user_content": False,
        "remember_stored": turns[0]["memory_count"] == 1,
        "recall_completed": turns[1]["recall_event"] == "memory.recall.completed",
        "recall_response_contains_marker": turns[1]["response_contains_marker"],
        "delete_completed": turns[2]["delete_event"] == "memory.delete.completed",
        "canonical_memory_effects_verified": True,
        "post_delete_recall_empty": turns[3]["recall_event"] == "memory.recall.empty",
        "post_delete_response_contains_marker": turns[3]["response_contains_marker"],
        "response_marker_checks_isolating": False,
        "turns_completed": sum(turn["state"] == "COMPLETED" for turn in turns),
        "wire_event_counts": [turn["wire_event_count"] for turn in turns],
        "response_byte_counts": [turn["response_bytes"] for turn in turns],
        "response_sha256": [turn["response_sha256"] for turn in turns],
        "side_effect_counts_before": before,
        "side_effect_counts_after": after,
        "conversation_cleanup_status": status,
        "conversation_cleanup_scoped_counts": scoped_after_cleanup,
    }


def main() -> int:
    args = parser().parse_args()
    cleanup_armed = False
    try:
        _validate_disposable_inputs(args)
        cleanup_armed = True
        report = asyncio.run(run(args, disposable_inputs_validated=True))
        status = 0
    except Exception as exc:  # noqa: BLE001 - emit only a closed, content-free failure
        cleanup_attempted = False
        cleanup_succeeded = False
        if cleanup_armed:
            cleanup_attempted = True
            cleanup_succeeded = _best_effort_cleanup(args)
        report = {
            "cleanup_attempted": cleanup_attempted,
            "cleanup_succeeded": cleanup_succeeded,
            "contains_user_content": False,
            "error_code": _closed_failure_code(exc),
            "probe_version": "jarvis.live-memory-crud.v1",
            "ready": False,
        }
        status = 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
