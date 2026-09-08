#!/usr/bin/env python3
"""Content-safe real WebSocket smoke for JARVIS local memory.

The script deliberately prints only structural booleans and identifiers' counts;
it never prints transcripts, model output, bearer tokens, or memory contents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import unicodedata
import uuid
from pathlib import Path

from jarvis_voice.protocol import Envelope, EventFactory
from jarvis_voice.voice_auth import load_voice_auth_key
from jarvis_voice.voice_auth_transport import authenticate_async_client


REMEMBER_TEXT = "Να θυμάσαι ότι το αγαπημένο μου χρώμα είναι μπλε."
RECALL_TEXT = "Ποιο είναι το αγαπημένο μου χρώμα;"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--voice-auth-key", type=Path, required=True)
    parser.add_argument("--phase", choices=("remember", "recall"), required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def _task_evidence(database: Path, turn_id: str) -> dict[str, object]:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        task = connection.execute(
            """SELECT t.task_id,t.state,t.epoch,t.conversation_id
               FROM tasks t JOIN canonical_turns c USING(task_id)
               WHERE c.turn_id=?""",
            (turn_id,),
        ).fetchone()
        if task is None:
            raise RuntimeError("admitted task is missing")
        recall = connection.execute(
            """SELECT event_type,payload_json FROM task_events
               WHERE task_id=? AND event_type IN
               ('memory.recall.completed','memory.recall.empty','memory.recall.failed')
               ORDER BY event_sequence DESC LIMIT 1""",
            (task["task_id"],),
        ).fetchone()
        memory_count = connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE conversation_id=?",
            (task["conversation_id"],),
        ).fetchone()[0]
        recall_payload = json.loads(recall["payload_json"]) if recall else {}
        return {
            "task_state": task["state"],
            "epoch": task["epoch"],
            "memory_record_count": memory_count,
            "recall_event": None if recall is None else recall["event_type"],
            "recall_result_count": int(recall_payload.get("result_count", 0)),
        }
    finally:
        connection.close()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    from websockets.asyncio.client import connect

    text = REMEMBER_TEXT if args.phase == "remember" else RECALL_TEXT
    turn_id = str(uuid.uuid4())
    factory = EventFactory()
    transcript_event_id: str | None = None
    event_types: list[str] = []
    assistant_chunks: list[str] = []
    accepted = False
    response_id: str | None = None
    terminal: str | None = None
    auth_key = load_voice_auth_key(args.voice_auth_key)

    async with asyncio.timeout(args.timeout):
        async with connect(
            args.url,
            origin=None,
            compression=None,
            proxy=None,
            open_timeout=2,
            close_timeout=1,
            ping_interval=10,
            ping_timeout=10,
            max_size=64 * 1024,
        ) as websocket:
            hello = factory.create(
                "client.hello",
                {
                    "protocol": "jarvis.voice/1",
                    "capabilities": [
                        "speech_commits",
                        "speech_done",
                        "output_epochs",
                    ],
                },
            )
            await authenticate_async_client(websocket, auth_key, hello.to_json())
            ready = Envelope.from_json(await websocket.recv())
            if ready.type != "server.ready":
                raise RuntimeError("server.ready was not first")
            transcript = factory.create(
                "input.transcript.final",
                {
                    "text": text,
                    "language": {"code": "el", "score": 1.0},
                    "asr_score": {
                        "value": 1.0,
                        "method": "manual_smoke",
                        "raw_avg_logprob": 0.0,
                        "calibrated": False,
                    },
                },
                turn_id=turn_id,
            )
            transcript_event_id = transcript.event_id
            await websocket.send(transcript.to_json())
            while terminal is None:
                event = Envelope.from_json(await websocket.recv())
                if event.type == "server.ack":
                    if event.payload["acked_event_id"] == transcript_event_id:
                        accepted = event.payload["status"] == "accepted"
                    continue
                if event.turn_id != turn_id:
                    continue
                event_types.append(event.type)
                response_id = event.response_id or response_id
                if event.type == "response.text.delta":
                    assistant_chunks.append(str(event.payload["text"]))
                elif event.type in {"response.speech.done", "response.cancelled"}:
                    terminal = event.type

    evidence = _task_evidence(args.database, turn_id)
    answer = unicodedata.normalize("NFC", "".join(assistant_chunks)).casefold()
    result = {
        "phase": args.phase,
        "accepted_ack": accepted,
        "terminal": terminal,
        "completed": evidence["task_state"] == "COMPLETED",
        "memory_record_count": evidence["memory_record_count"],
        "recall_event": evidence["recall_event"],
        "recall_result_count": evidence["recall_result_count"],
        "expected_fact_present": "μπλε" in answer if args.phase == "recall" else None,
        "wire_event_count": len(event_types),
        "has_response_id": response_id is not None,
    }
    if not accepted or terminal != "response.speech.done" or not result["completed"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.phase == "remember" and int(result["memory_record_count"]) < 1:
        raise RuntimeError("completed remember turn did not create a memory record")
    if args.phase == "recall" and (
        result["recall_event"] != "memory.recall.completed"
        or int(result["recall_result_count"]) < 1
        or result["expected_fact_present"] is not True
    ):
        raise RuntimeError("memory recall acceptance failed")
    return result


def main() -> int:
    args = _parser().parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
