from __future__ import annotations

import asyncio
import unittest
from collections import deque

import _bootstrap  # noqa: F401

from jarvis_voice.backends import AsrScore, LanguageMetadata, TranscriptResult
from jarvis_voice.errors import BackendError
from jarvis_voice.protocol import EventFactory
from jarvis_voice.transport import LoopbackFakeServer, LoopbackWebSocketTransport


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.incoming: deque[str] = deque()
        self.closed = False

    def send(self, data: str) -> None:
        self.sent.append(data)

    def recv(self, timeout: float | None = None) -> str:
        del timeout
        if not self.incoming:
            raise TimeoutError
        return self.incoming.popleft()

    def close(self) -> None:
        self.closed = True


class FakeAsyncWebSocket:
    def __init__(self, incoming: list[str]) -> None:
        self.incoming = iter(incoming)
        self.sent: list[str] = []

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return next(self.incoming)
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, data: str) -> None:
        self.sent.append(data)


def transcript(factory: EventFactory, *, event_id: str | None = None):
    result = TranscriptResult("Άνοιξέ το.", LanguageMetadata("el", 0.9), AsrScore(0.8, -0.2))
    return factory.create("input.transcript.final", result.to_payload(), event_id=event_id)


class TransportTests(unittest.TestCase):
    def test_transport_rejects_non_loopback(self) -> None:
        with self.assertRaisesRegex(BackendError, "transport.non_loopback_url"):
            LoopbackWebSocketTransport("ws://example.com:8765")

    def test_injected_connection_serializes_and_receives_envelopes(self) -> None:
        connection = FakeConnection()
        transport = LoopbackWebSocketTransport("ws://127.0.0.1:8765", connection_factory=lambda url: connection)
        factory = EventFactory(); hello = factory.create("client.hello", {})
        transport.send(hello)
        self.assertEqual(connection.sent, [hello.to_json()])
        server = EventFactory(session_id=hello.session_id).create(
            "server.ready", {"protocol": "jarvis.voice/1"}
        )
        connection.incoming.append(server.to_json())
        self.assertEqual(transport.receive(), server)
        self.assertIsNone(transport.receive())
        transport.close(); self.assertTrue(connection.closed)

    def test_fake_server_streams_greek_commits_before_speech_done(self) -> None:
        factory = EventFactory()
        hello = factory.create("client.hello", {})
        final = transcript(factory)
        websocket = FakeAsyncWebSocket([hello.to_json(), final.to_json()])
        asyncio.run(LoopbackFakeServer().handle(websocket))
        types = [__import__("json").loads(raw)["type"] for raw in websocket.sent]
        self.assertIn("response.speech.commit", types)
        self.assertLess(types.index("response.speech.commit"), types.index("response.speech.done"))
        self.assertEqual(types[0:2], ["server.ready", "server.ack"])

    def test_duplicate_event_id_is_not_processed_as_new_turn_on_reconnect(self) -> None:
        server = LoopbackFakeServer()
        first_factory = EventFactory(); original = transcript(first_factory)
        first = FakeAsyncWebSocket([original.to_json()]); asyncio.run(server.handle(first))
        reconnect = EventFactory(session_id=original.session_id)
        retry = transcript(reconnect, event_id=original.event_id)
        second = FakeAsyncWebSocket([retry.to_json()]); asyncio.run(server.handle(second))
        import json
        self.assertEqual(json.loads(first.sent[0])["payload"]["status"], "accepted")
        self.assertEqual(json.loads(second.sent[0])["payload"]["status"], "duplicate")
        self.assertEqual(retry.seq, 1)
        self.assertEqual(len(second.sent), 1)


if __name__ == "__main__":
    unittest.main()
