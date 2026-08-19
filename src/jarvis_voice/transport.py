"""Loopback-only WebSocket transport and deterministic fake server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Protocol

from .config import is_loopback_websocket_url
from .errors import BackendError
from .fakes import FakeOrchestrator
from .protocol import Envelope
from .cloud_approval import CloudApprovalEnvelope
from .action_approval import ACTION_APPROVAL_PROTOCOL, ActionApprovalEnvelope
from .scheduler_protocol import SCHEDULER_PROTOCOL, SchedulerEnvelope
from .voice_auth import VoiceAuthError, VoiceAuthKey
from .voice_auth_transport import authenticate_async_server, authenticate_sync_client


class SyncConnection(Protocol):
    def send(self, data: str) -> None: ...
    def recv(self, timeout: float | None = None) -> str | bytes: ...
    def close(self) -> None: ...


class LoopbackWebSocketTransport:
    def __init__(
        self,
        url: str,
        *,
        auth_key: VoiceAuthKey,
        connection_factory: Callable[[str], SyncConnection] | None = None,
    ) -> None:
        if not is_loopback_websocket_url(url):
            raise BackendError("transport.non_loopback_url", "Only loopback WebSockets are permitted", subsystem="transport")
        if not isinstance(auth_key, VoiceAuthKey):
            raise BackendError(
                "transport.authentication_required",
                "Voice authentication key is required",
                subsystem="transport",
            )
        self.url = url
        self._auth_key = auth_key
        self._factory = connection_factory
        self._connection: SyncConnection | None = None
        self._authenticated = False

    def connect(self) -> None:
        if self._connection is not None:
            return
        factory = self._factory
        if factory is None:
            try:
                from websockets.sync.client import connect  # type: ignore[import-not-found]
            except ImportError as exc:
                raise BackendError("transport.runtime_missing", "websockets is not installed", subsystem="transport") from exc
            factory = lambda selected_url: connect(
                selected_url,
                origin=None,
                compression=None,
                proxy=None,
                open_timeout=2,
                close_timeout=1,
                ping_interval=10,
                ping_timeout=10,
                max_size=64 * 1024,
                max_queue=32,
            )
        self._connection = factory(self.url)

    def send(self, event: Envelope | CloudApprovalEnvelope | ActionApprovalEnvelope | SchedulerEnvelope) -> None:
        try:
            self.connect()
            assert self._connection is not None
            if not self._authenticated:
                if not isinstance(event, Envelope) or event.type != "client.hello" or event.seq != 1:
                    with suppress(Exception):
                        self.close()
                    raise BackendError(
                        "transport.authentication_required",
                        "The first frame must be the authenticated Voice hello",
                        subsystem="transport",
                    )
                authenticate_sync_client(
                    self._connection,
                    self._auth_key,
                    event.to_json(),
                )
                self._authenticated = True
                return
            self._connection.send(event.to_json())
        except BackendError:
            raise
        except VoiceAuthError as exc:
            with suppress(Exception):
                self.close()
            raise BackendError(
                "transport.authentication_failed",
                "Voice WebSocket authentication failed",
                subsystem="transport",
                retryable=True,
            ) from exc
        except Exception as exc:
            raise BackendError("transport.send_failed", "WebSocket send failed", subsystem="transport", retryable=True) from exc

    def receive(self) -> Envelope | CloudApprovalEnvelope | ActionApprovalEnvelope | SchedulerEnvelope | None:
        try:
            if self._connection is None or not self._authenticated:
                raise BackendError(
                    "transport.authentication_required",
                    "Voice WebSocket is not authenticated",
                    subsystem="transport",
                )
            raw = self._connection.recv(timeout=0)
        except TimeoutError:
            return None
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError("transport.receive_failed", "WebSocket receive failed", subsystem="transport", retryable=True) from exc
        try:
            preview = json.loads(raw)
        except Exception as exc:
            raise BackendError("transport.invalid_json", "WebSocket frame is invalid", subsystem="transport") from exc
        if isinstance(preview, dict) and preview.get("protocol") == ACTION_APPROVAL_PROTOCOL:
            return ActionApprovalEnvelope.from_json(raw)
        if isinstance(preview, dict) and preview.get("protocol") == SCHEDULER_PROTOCOL:
            return SchedulerEnvelope.from_json(raw)
        if isinstance(preview, dict) and "protocol" in preview:
            return CloudApprovalEnvelope.from_json(raw)
        return Envelope.from_json(raw)

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        self._authenticated = False
        if connection is not None:
            connection.close()


class LoopbackFakeServer:
    """A local test server; acknowledged event IDs persist across reconnects."""

    def __init__(self, *, auth_key: VoiceAuthKey) -> None:
        if not isinstance(auth_key, VoiceAuthKey):
            raise BackendError(
                "transport.authentication_required",
                "Voice authentication key is required",
                subsystem="transport",
            )
        self.auth_key = auth_key
        self.acknowledged_event_ids: set[str] = set()

    async def handle(self, websocket: Any) -> None:
        from .protocol import new_uuid4

        hello_raw = await authenticate_async_server(
            websocket,
            self.auth_key,
            connection_id=new_uuid4(),
        )
        hello = Envelope.from_json(hello_raw)
        if hello.type != "client.hello" or hello.seq != 1:
            raise VoiceAuthError("voice_auth.hello_invalid")
        orchestrator = FakeOrchestrator(session_id=hello.session_id)
        orchestrator.acknowledged_event_ids = self.acknowledged_event_ids
        for reply in orchestrator.accept(hello):
            await websocket.send(reply.to_json())
        async for raw in websocket:
            event = Envelope.from_json(raw)
            was_acknowledged = event.event_id in self.acknowledged_event_ids
            for reply in orchestrator.accept(event):
                await websocket.send(reply.to_json())
            if event.type == "input.transcript.final" and not was_acknowledged:
                language = event.payload.get("language", {}).get("code", "en")
                greek = language == "el"
                sentences = ["Βεβαίως.", "Το εκτελώ τώρα."] if greek else ["Certainly.", "I will do that now."]
                locale = "el-GR" if greek else "en-US"
                scripted = orchestrator.script_response(sentences, locale=locale, turn_id=event.turn_id, interleave_speech=True)
                for reply in scripted.events:
                    await websocket.send(reply.to_json())

    def run(self, host: str, port: int) -> None:
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise BackendError("transport.non_loopback_bind", "Fake server must bind to loopback", subsystem="transport")

        async def serve_forever() -> None:
            try:
                from websockets.asyncio.server import serve  # type: ignore[import-not-found]
            except ImportError as exc:
                raise BackendError("transport.runtime_missing", "websockets is not installed", subsystem="transport") from exc
            async with serve(self.handle, host, port):
                await asyncio.Future()

        asyncio.run(serve_forever())
