#!/usr/bin/env python3
"""Content-safe real WebSocket smoke for Orchestrator -> inference.

The fixed synthetic transcript is intentionally non-personal. Output contains
only structural evidence and never includes the transcript, model response,
protocol identifiers, database paths, or bearer tokens.
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


FIXED_TRANSCRIPT = "Απάντησε μόνο με τη λέξη έτοιμο."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smoke-orchestrator-text")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--voice-auth-key", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def _durable_evidence(database: Path, turn_id: str) -> dict[str, object]:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        task = connection.execute(
            """SELECT t.task_id,t.state,c.status AS turn_status
               FROM tasks t JOIN canonical_turns c USING(task_id)
               WHERE c.turn_id=?""",
            (turn_id,),
        ).fetchone()
        if task is None:
            raise RuntimeError("smoke task is missing")
        reservations = connection.execute(
            """SELECT state,actual_output_tokens FROM reservations
               WHERE task_id=? ORDER BY created_at""",
            (task["task_id"],),
        ).fetchall()
        return {
            "task_state": task["state"],
            "turn_status": task["turn_status"],
            "reservation_count": len(reservations),
            "usage_reconciled": bool(reservations)
            and all(
                row["state"] == "RECONCILED"
                and type(row["actual_output_tokens"]) is int
                and row["actual_output_tokens"] >= 0
                for row in reservations
            ),
        }
    finally:
        connection.close()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    from websockets.asyncio.client import connect

    turn_id = str(uuid.uuid4())
    factory = EventFactory()
    event_types: list[str] = []
    assistant_chunks: list[str] = []
    accepted = False
    terminal: str | None = None
    response_id: str | None = None
    received_events = 0
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
                    "capabilities": ["speech_commits", "speech_done", "output_epochs"],
                },
            )
            await authenticate_async_client(websocket, auth_key, hello.to_json())
            ready = Envelope.from_json(await websocket.recv())
            if ready.type != "server.ready" or ready.payload != {"protocol": "jarvis.voice/1"}:
                raise RuntimeError("server.ready was invalid")
            await websocket.send(
                factory.create(
                    "input.transcript.final",
                    {
                        "text": FIXED_TRANSCRIPT,
                        "language": {"code": "el", "score": 1.0},
                        "asr_score": {
                            "value": 1.0,
                            "method": "synthetic_text_smoke",
                            "raw_avg_logprob": 0.0,
                            "calibrated": False,
                        },
                    },
                    turn_id=turn_id,
                ).to_json()
            )
            while terminal is None:
                event = Envelope.from_json(await websocket.recv())
                received_events += 1
                if received_events > 4096:
                    raise RuntimeError("wire replay exceeded the smoke bound")
                belongs_to_turn = event.turn_id == turn_id
                belongs_to_response = response_id is not None and event.response_id == response_id
                if not belongs_to_turn and not belongs_to_response:
                    continue
                if event.type == "response.started":
                    if event.response_id is None:
                        raise RuntimeError("response.started had no response_id")
                    response_id = event.response_id
                event_types.append(event.type)
                if event.type == "server.ack":
                    accepted = event.payload["status"] == "accepted"
                elif event.type == "response.text.delta":
                    assistant_chunks.append(str(event.payload["text"]))
                elif event.type in {"response.speech.done", "response.cancelled", "server.error"}:
                    terminal = event.type

    durable = _durable_evidence(args.database, turn_id)
    normalized_answer = unicodedata.normalize("NFC", "".join(assistant_chunks)).casefold()
    required_events = {
        "server.ack",
        "response.started",
        "response.text.delta",
        "response.text.done",
        "response.speech.commit",
        "response.speech.done",
    }
    result = {
        "ready": bool(
            accepted
            and terminal == "response.speech.done"
            and required_events.issubset(event_types)
            and durable["task_state"] == "COMPLETED"
            and durable["turn_status"] == "COMPLETED"
            and durable["usage_reconciled"] is True
            and "έτοιμο" in normalized_answer
        ),
        "accepted_ack": accepted,
        "terminal": terminal,
        "task_state": durable["task_state"],
        "turn_status": durable["turn_status"],
        "reservation_count": durable["reservation_count"],
        "usage_reconciled": durable["usage_reconciled"],
        "wire_event_count": len(event_types),
        "text_delta_count": event_types.count("response.text.delta"),
        "speech_commit_count": event_types.count("response.speech.commit"),
        "expected_token_present": "έτοιμο" in normalized_answer,
        "has_response_id": response_id is not None,
        "ignored_replay_count": received_events - len(event_types),
        "content_emitted": False,
    }
    if result["ready"] is not True:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def main() -> int:
    args = _parser().parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
