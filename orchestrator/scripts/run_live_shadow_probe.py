#!/usr/bin/env python3
"""Run one benign create/replace/cancel sequence through the live voice gateway."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from jarvis_voice.protocol import Envelope, EventFactory
from jarvis_voice.voice_auth import load_voice_auth_key
from jarvis_voice.voice_auth_transport import authenticate_async_client


PROBE_TURNS = (
    ("en", "Remember that my shadow-probe color is blue."),
    ("el", "Διόρθωση: το χρώμα του shadow probe είναι πράσινο."),
    ("en", "Forget my shadow-probe color."),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="run-live-shadow-probe")
    result.add_argument("--url", default="ws://127.0.0.1:8865")
    result.add_argument("--voice-auth-key", type=Path, required=True)
    return result


async def run(url: str, voice_auth_key: Path) -> dict[str, object]:
    from websockets.asyncio.client import connect

    factory = EventFactory()
    auth_key = load_voice_auth_key(voice_auth_key)
    event_counts: dict[str, int] = {}
    completed = 0
    async with connect(
        url,
        origin=None,
        compression=None,
        proxy=None,
        max_size=64 * 1024,
        open_timeout=3,
        close_timeout=1,
        ping_interval=10,
        ping_timeout=10,
    ) as websocket:
        hello = factory.create("client.hello")
        await authenticate_async_client(websocket, auth_key, hello.to_json())
        ready = Envelope.from_json(await asyncio.wait_for(websocket.recv(), timeout=3))
        if ready.type != "server.ready":
            raise RuntimeError("shadow probe did not receive server.ready")
        for language, text in PROBE_TURNS:
            transcript = factory.create(
                "input.transcript.final",
                {
                    "text": text,
                    "language": {"code": language, "score": 1.0},
                    "asr_score": {
                        "value": 1.0,
                        "method": "manual_shadow_probe",
                        "calibrated": False,
                    },
                },
                turn_id=str(uuid.uuid4()),
            )
            await websocket.send(transcript.to_json())
            response_id = None
            while True:
                event = Envelope.from_json(
                    await asyncio.wait_for(websocket.recv(), timeout=180)
                )
                event_counts[event.type] = event_counts.get(event.type, 0) + 1
                if event.type == "server.error":
                    raise RuntimeError(
                        f"shadow probe server error: {event.payload['code']}"
                    )
                if event.response_id is not None:
                    response_id = event.response_id
                if event.type == "response.speech.done":
                    break
            if response_id is None:
                raise RuntimeError("shadow probe response identity is missing")
            await websocket.send(
                factory.create(
                    "playback.completed",
                    {},
                    response_id=response_id,
                ).to_json()
            )
            completed += 1
        await websocket.send(factory.create("client.goodbye").to_json())
    return {
        "probe": "jarvis.live-shadow-probe.v1",
        "turns_completed": completed,
        "event_counts": dict(sorted(event_counts.items())),
        "contains_response_text": False,
        "authoritative_memory_requested": False,
    }


def main(argv: list[str] | None = None) -> int:
    import json

    parsed = parser().parse_args(argv)
    result = asyncio.run(run(parsed.url, parsed.voice_auth_key))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["turns_completed"] == len(PROBE_TURNS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
