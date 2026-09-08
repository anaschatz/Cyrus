from __future__ import annotations

import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.backends import AsrScore, LanguageMetadata, TranscriptResult
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.runtime import MacVoiceRuntime
from jarvis_voice.protocol import EventFactory
from jarvis_voice.transport import LoopbackFakeServer, LoopbackWebSocketTransport
from jarvis_voice.voice_auth import (
    VoiceAuthKey,
    create_client_response,
    create_server_challenge,
    create_server_ready,
)


AUTH_KEY = VoiceAuthKey(1, bytes(range(32)))
OTHER_AUTH_KEY = VoiceAuthKey(1, bytes(reversed(range(32))))


class FakeConnection:
    def __init__(self, *, server_key: VoiceAuthKey = AUTH_KEY) -> None:
        self.sent: list[str] = []
        self.server_key = server_key
        self.challenge = create_server_challenge(server_key).to_json()
        self.incoming: deque[str] = deque([self.challenge])
        self.closed = False

    def send(self, data: str) -> None:
        self.sent.append(data)
        if len(self.sent) == 2:
            self.incoming.append(
                create_server_ready(
                    self.server_key,
                    self.challenge,
                    self.sent[0],
                    self.sent[1],
                ).to_json()
            )

    def recv(self, timeout: float | None = None) -> str:
        del timeout
        if not self.incoming:
            raise TimeoutError
        return self.incoming.popleft()

    def close(self) -> None:
        self.closed = True


class FakeAsyncWebSocket:
    def __init__(self, incoming: list[str], *, client_key: VoiceAuthKey = AUTH_KEY) -> None:
        self.incoming = incoming
        self.client_key = client_key
        self.cursor = 1
        self.auth_step = 0
        self.sent: list[str] = []

    async def recv(self) -> str:
        if self.auth_step == 0:
            self.auth_step = 1
            return create_client_response(
                self.client_key,
                self.sent[0],
                self.incoming[0],
            ).to_json()
        if self.auth_step == 1:
            self.auth_step = 2
            return self.incoming[0]
        if self.cursor >= len(self.incoming):
            raise StopAsyncIteration
        value = self.incoming[self.cursor]
        self.cursor += 1
        return value

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self.cursor >= len(self.incoming):
            raise StopAsyncIteration
        value = self.incoming[self.cursor]
        self.cursor += 1
        return value

    async def send(self, data: str) -> None:
        self.sent.append(data)


def transcript(factory: EventFactory, *, event_id: str | None = None):
    result = TranscriptResult("Άνοιξέ το.", LanguageMetadata("el", 0.9), AsrScore(0.8, -0.2))
    return factory.create("input.transcript.final", result.to_payload(), event_id=event_id)


class TransportTests(unittest.TestCase):
    def test_authenticated_connection_rejects_other_session_frames(self) -> None:
        connection = FakeConnection()
        transport = LoopbackWebSocketTransport(
            "ws://127.0.0.1:8765", auth_key=AUTH_KEY,
            connection_factory=lambda _: connection,
        )
        hello = EventFactory().create("client.hello", {})
        transport.send(hello)
        other = EventFactory()
        with self.assertRaisesRegex(BackendError, "transport.session_mismatch"):
            transport.send(transcript(other))
        self.assertEqual(len(connection.sent), 2)
        connection.incoming.append(other.create("server.ready", {"protocol": "jarvis.voice/1"}).to_json())
        with self.assertRaisesRegex(BackendError, "transport.session_mismatch"):
            transport.receive()

    def test_transport_rejects_non_loopback(self) -> None:
        with self.assertRaisesRegex(BackendError, "transport.non_loopback_url"):
            LoopbackWebSocketTransport("ws://example.com:8765", auth_key=AUTH_KEY)

    def test_injected_connection_serializes_and_receives_envelopes(self) -> None:
        connection = FakeConnection()
        transport = LoopbackWebSocketTransport(
            "ws://127.0.0.1:8765",
            auth_key=AUTH_KEY,
            connection_factory=lambda url: connection,
        )
        factory = EventFactory(); hello = factory.create("client.hello", {})
        transport.send(hello)
        self.assertEqual(connection.sent[1], hello.to_json())
        server = EventFactory(session_id=hello.session_id).create(
            "server.ready", {"protocol": "jarvis.voice/1"}
        )
        connection.incoming.append(server.to_json())
        self.assertEqual(transport.receive(), server)
        self.assertIsNone(transport.receive())
        transport.close(); self.assertTrue(connection.closed)

    def test_receive_failure_reports_only_bounded_close_code(self) -> None:
        class ClosedConnection(FakeConnection):
            def recv(self, timeout: float | None = None) -> str:
                if self.incoming:
                    return super().recv(timeout)
                error = OSError("untrusted detail")
                error.rcvd = SimpleNamespace(code=1011, reason="untrusted detail")  # type: ignore[attr-defined]
                raise error

        connection = ClosedConnection()
        transport = LoopbackWebSocketTransport(
            "ws://127.0.0.1:8765",
            auth_key=AUTH_KEY,
            connection_factory=lambda url: connection,
        )
        transport.send(EventFactory().create("client.hello", {}))

        with self.assertRaises(BackendError) as raised:
            transport.receive()

        self.assertEqual(raised.exception.code, "transport.receive_failed")
        self.assertEqual(raised.exception.safe_context, {"close_code": 1011})
        self.assertNotIn("untrusted detail", raised.exception.to_log_record().values())

    def test_macos_pending_transcript_reconnects_once_without_closing_journal(self) -> None:
        runtime = MacVoiceRuntime.__new__(MacVoiceRuntime)
        event_id = "pending-event"
        runtime.transport = SimpleNamespace(
            receive=mock.Mock(
                side_effect=BackendError(
                    "transport.receive_failed",
                    "closed",
                    subsystem="transport",
                    retryable=True,
                )
            ),
            close=mock.Mock(),
            send=mock.Mock(),
        )
        runtime.reliable = SimpleNamespace(
            can_reconnect=True,
            original=SimpleNamespace(event_id=event_id),
            reconnect_event=mock.Mock(return_value="retry"),
        )
        runtime.delivery_journal = object()
        runtime.core = SimpleNamespace(
            begin_new_connection=mock.Mock(),
            client_hello=mock.Mock(return_value="hello"),
        )
        runtime.cloud_approval = SimpleNamespace(begin_new_connection=mock.Mock())
        runtime.action_approval = SimpleNamespace(begin_new_connection=mock.Mock())
        runtime.scheduler_runtime = None
        runtime._transport_reconnect_event_id = None
        runtime._record_timing = mock.Mock()

        self.assertIsNone(runtime._receive_or_reconnect())
        self.assertEqual(runtime.transport.send.call_args_list, [mock.call("hello"), mock.call("retry")])
        runtime._record_timing.assert_has_calls(
            [mock.call("transport_reconnect_started"), mock.call("transport_reconnect_completed")]
        )
        with self.assertRaisesRegex(BackendError, "transport.receive_failed"):
            runtime._receive_or_reconnect()

    def test_wrong_key_fails_before_hello_is_accepted(self) -> None:
        connection = FakeConnection(server_key=AUTH_KEY)
        transport = LoopbackWebSocketTransport(
            "ws://127.0.0.1:8765",
            auth_key=OTHER_AUTH_KEY,
            connection_factory=lambda url: connection,
        )
        with self.assertRaisesRegex(BackendError, "transport.authentication_failed"):
            transport.send(EventFactory().create("client.hello", {}))
        self.assertTrue(connection.closed)
        self.assertEqual(connection.sent, [])

    def test_close_failure_cannot_mask_authentication_failure_or_reuse_socket(self) -> None:
        class CloseFailureConnection(FakeConnection):
            def close(self) -> None:
                self.closed = True
                raise OSError("close failed")

        connection = CloseFailureConnection(server_key=AUTH_KEY)
        transport = LoopbackWebSocketTransport(
            "ws://127.0.0.1:8765",
            auth_key=OTHER_AUTH_KEY,
            connection_factory=lambda url: connection,
        )

        with self.assertRaisesRegex(BackendError, "transport.authentication_failed"):
            transport.send(EventFactory().create("client.hello", {}))

        self.assertTrue(connection.closed)
        self.assertIsNone(transport._connection)
        self.assertFalse(transport._authenticated)

    def test_transport_rejects_business_event_before_authenticated_hello(self) -> None:
        connection = FakeConnection()
        transport = LoopbackWebSocketTransport(
            "ws://127.0.0.1:8765",
            auth_key=AUTH_KEY,
            connection_factory=lambda url: connection,
        )
        with self.assertRaisesRegex(BackendError, "transport.authentication_required"):
            transport.send(transcript(EventFactory()))
        self.assertEqual(connection.sent, [])
        self.assertTrue(connection.closed)

    def test_fake_server_streams_greek_commits_before_speech_done(self) -> None:
        factory = EventFactory()
        hello = factory.create("client.hello", {})
        final = transcript(factory)
        websocket = FakeAsyncWebSocket([hello.to_json(), final.to_json()])
        asyncio.run(LoopbackFakeServer(auth_key=AUTH_KEY).handle(websocket))
        documents = [__import__("json").loads(raw) for raw in websocket.sent]
        types = [
            row["type"] for row in documents
            if row.get("protocol") != "jarvis.voice-auth/1"
        ]
        self.assertIn("response.speech.commit", types)
        self.assertLess(types.index("response.speech.commit"), types.index("response.speech.done"))
        self.assertEqual(types[0:2], ["server.ready", "server.ack"])

    def test_duplicate_event_id_is_not_processed_as_new_turn_on_reconnect(self) -> None:
        server = LoopbackFakeServer(auth_key=AUTH_KEY)
        first_factory = EventFactory()
        first_hello = first_factory.create("client.hello", {})
        original = transcript(first_factory)
        first = FakeAsyncWebSocket([first_hello.to_json(), original.to_json()]); asyncio.run(server.handle(first))
        reconnect = EventFactory(session_id=original.session_id)
        reconnect_hello = reconnect.create("client.hello", {})
        retry = transcript(reconnect, event_id=original.event_id)
        second = FakeAsyncWebSocket([reconnect_hello.to_json(), retry.to_json()]); asyncio.run(server.handle(second))
        import json
        first_ack = next(json.loads(raw) for raw in first.sent if json.loads(raw).get("type") == "server.ack")
        second_ack = next(json.loads(raw) for raw in second.sent if json.loads(raw).get("type") == "server.ack")
        self.assertEqual(first_ack["payload"]["status"], "accepted")
        self.assertEqual(second_ack["payload"]["status"], "duplicate")
        self.assertEqual(retry.seq, 2)


if __name__ == "__main__":
    unittest.main()
