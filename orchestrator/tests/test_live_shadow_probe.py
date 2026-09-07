from __future__ import annotations

import asyncio
import runpy
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import _bootstrap  # noqa: F401
from jarvis_voice.protocol import Envelope, EventFactory

SCRIPT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_live_shadow_probe.py")
)


class FakeSocket:
    def __init__(self, mode):
        self.mode = mode
        self.factory = EventFactory()
        self.queue = [
            self.factory.create(
                "server.ready", {"protocol": "jarvis.voice/1"}
            ).to_json()
        ]
        self.expected = []
        self.played = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def recv(self):
        if not self.queue:
            raise AssertionError(
                "probe consumed all synthetic events without finishing"
            )
        return self.queue.pop(0)

    def emit(self, kind, payload, turn, response=None):
        self.queue.append(
            self.factory.create(
                kind, payload, turn_id=turn, response_id=response
            ).to_json()
        )

    async def send(self, raw):
        event = Envelope.from_json(raw)
        if event.type == "playback.completed":
            self.played.append(event.response_id)
        if event.type != "input.transcript.final":
            return
        response = str(uuid.uuid4())
        self.expected.append(response)
        if self.mode == "replay":
            stale_turn, stale_response = str(uuid.uuid4()), str(uuid.uuid4())
            self.emit("response.started", {"locale": "en"}, stale_turn, stale_response)
            self.emit(
                "response.speech.done",
                {"last_unit_index": 0},
                stale_turn,
                stale_response,
            )
        ack_id = str(uuid.uuid4()) if self.mode == "wrong_ack" else event.event_id
        self.emit(
            "server.ack",
            {"acked_event_id": ack_id, "status": "accepted"},
            event.turn_id,
        )
        self.emit("response.started", {"locale": "en"}, event.turn_id, response)
        if self.mode == "wrong_response":
            self.emit(
                "response.speech.done",
                {"last_unit_index": 0},
                event.turn_id,
                str(uuid.uuid4()),
            )
        if self.mode == "cancelled":
            self.emit(
                "response.cancelled",
                {"reason_code": "user_cancelled", "partial_output": False},
                event.turn_id,
                response,
            )
        self.emit(
            "response.speech.done", {"last_unit_index": 0}, event.turn_id, response
        )


class LiveShadowProbeTests(unittest.TestCase):
    def run_probe(self, mode):
        socket = FakeSocket(mode)
        with (
            patch.dict(
                SCRIPT["run"].__globals__,
                {
                    "load_voice_auth_key": lambda path: b"synthetic-key",
                    "authenticate_async_client": AsyncMock(),
                },
            ),
            patch("websockets.asyncio.client.connect", return_value=socket),
        ):
            result = asyncio.run(
                SCRIPT["run"]("ws://127.0.0.1:8865", Path("unused-key"))
            )
        return result, socket

    def test_completes_only_the_acknowledged_turn_and_response(self):
        for mode in ("valid", "replay", "wrong_response"):
            with self.subTest(mode=mode):
                result, socket = self.run_probe(mode)
                self.assertEqual(result["turns_completed"], 3)
                self.assertEqual(len(socket.expected), 3)
                self.assertEqual(socket.played, socket.expected)

    def test_missing_ack_and_cancellation_cannot_report_success(self):
        for mode in ("wrong_ack", "cancelled"):
            with self.subTest(mode=mode):
                with self.assertRaises(RuntimeError):
                    self.run_probe(mode)


if __name__ == "__main__":
    unittest.main()
