from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import _bootstrap  # noqa: F401
from jarvis_voice import voice_auth_transport as transport
from jarvis_voice.voice_auth import VoiceAuthError
from test_transport import AUTH_KEY, FakeAsyncWebSocket, FakeConnection
from test_voice_auth import CONNECTION_ID, HELLO


class VoiceAuthTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_deadlines_fail_before_transport_io(self):
        for timeout in (None, "2", True, False, float("nan"), float("inf"), -1, 0, 10**500):
            for kind in ("sync", "client", "server"):
                with self.subTest(timeout=timeout, kind=kind):
                    wire = SimpleNamespace(send=Mock(), recv=Mock())
                    with self.assertRaisesRegex(VoiceAuthError, "voice_auth.timeout_invalid"):
                        if kind == "sync":
                            transport.authenticate_sync_client(wire, AUTH_KEY, HELLO, timeout_seconds=timeout)
                        elif kind == "client":
                            await transport.authenticate_async_client(wire, AUTH_KEY, HELLO, timeout_seconds=timeout)
                        else:
                            await transport.authenticate_async_server(wire, AUTH_KEY, connection_id=CONNECTION_ID, timeout_seconds=timeout)
                    wire.send.assert_not_called()
                    wire.recv.assert_not_called()

    async def test_valid_and_late_handshakes_respect_the_total_deadline(self):
        for kind in ("sync", "client", "server"):
            for late in (False, True):
                with self.subTest(kind=kind, late=late):
                    clock = SimpleNamespace(value=0.0)
                    if kind == "server":
                        wire = FakeAsyncWebSocket([HELLO])
                        send = wire.send
                        async def finish(data):
                            await send(data)
                            if late and len(wire.sent) == 2:
                                clock.value = 1.0
                        wire.send = finish
                    else:
                        wire = FakeConnection()
                        receive = wire.recv
                        def finish(*args, **kwargs):
                            value = receive(*args, **kwargs)
                            if late and wire.sent:
                                clock.value = 1.0
                            return value
                        wire.recv = finish
                        if kind == "client":
                            wire.send = AsyncMock(side_effect=wire.send)
                            wire.recv = AsyncMock(side_effect=wire.recv)
                    with patch.object(transport, "time", SimpleNamespace(monotonic=lambda: clock.value)):
                        async def authenticate():
                            if kind == "sync":
                                transport.authenticate_sync_client(wire, AUTH_KEY, HELLO, timeout_seconds=1.0)
                            elif kind == "client":
                                await transport.authenticate_async_client(wire, AUTH_KEY, HELLO, timeout_seconds=1.0)
                            else:
                                self.assertEqual(await transport.authenticate_async_server(wire, AUTH_KEY, connection_id=CONNECTION_ID, timeout_seconds=1.0), HELLO)
                        if late:
                            with self.assertRaisesRegex(VoiceAuthError, "voice_auth.timeout"):
                                await authenticate()
                        else:
                            await authenticate()

    async def test_expired_response_send_does_not_dispatch_the_hello(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                wire = FakeConnection()
                send = wire.send
                clock = SimpleNamespace(value=0.0)
                def expire(data):
                    send(data)
                    clock.value = 1.0
                wire.send = AsyncMock(side_effect=expire) if asynchronous else Mock(side_effect=expire)
                if asynchronous:
                    wire.recv = AsyncMock(side_effect=wire.recv)
                with patch.object(transport, "time", SimpleNamespace(monotonic=lambda: clock.value)):
                    with self.assertRaisesRegex(VoiceAuthError, "voice_auth.timeout"):
                        if asynchronous:
                            await transport.authenticate_async_client(wire, AUTH_KEY, HELLO, timeout_seconds=1.0)
                        else:
                            transport.authenticate_sync_client(wire, AUTH_KEY, HELLO, timeout_seconds=1.0)
                wire.send.assert_called_once()

    async def test_async_receive_timeout_cancels_the_pending_operation(self):
        cancelled = asyncio.Event()
        async def receive():
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        wire = SimpleNamespace(recv=receive, send=AsyncMock())
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.timeout"):
            await transport.authenticate_async_client(wire, AUTH_KEY, HELLO, timeout_seconds=0.01)
        self.assertTrue(cancelled.is_set())
        wire.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
