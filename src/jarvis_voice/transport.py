"""Loopback-only WebSocket transport and deterministic fake server."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

from .config import is_loopback_websocket_url
from .errors import BackendError
from .fakes import FakeOrchestrator
from .protocol import Envelope


class SyncConnection(Protocol):
    def send(self, data: str) -> None: ...
    def recv(self, timeout: float | None = None) -> str | bytes: ...
    def close(self) -> None: ...


class LoopbackWebSocketTransport:
    def __init__(self, url: str, *, connection_factory: Callable[[str], SyncConnection] | None = None) -> None:
        if not is_loopback_websocket_url(url):
            raise BackendError("transport.non_loopback_url", "Only loopback WebSockets are permitted", subsystem="transport")
        self.url = url
        self._factory = connection_factory
        self._connection: SyncConnection | None = None

    def connect(self) -> None:
        if self._connection is not None:
            return
        factory = self._factory
        if factory is None:
            try:
                from websockets.sync.client import connect  # type: ignore[import-not-found]
            except ImportError as exc:
                raise BackendError("transport.runtime_missing", "websockets is not installed", subsystem="transport") from exc
            factory = connect
        self._connection = factory(self.url)

    def send(self, event: Envelope) -> None:
        try:
            self.connect()
            assert self._connection is not None
            self._connection.send(event.to_json())
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError("transport.send_failed", "WebSocket send failed", subsystem="transport", retryable=True) from exc

    def receive(self) -> Envelope | None:
        try:
            self.connect()
            assert self._connection is not None
            raw = self._connection.recv(timeout=0)
        except TimeoutError:
            return None
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError("transport.receive_failed", "WebSocket receive failed", subsystem="transport", retryable=True) from exc
        return Envelope.from_json(raw)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close(); self._connection = None


class LoopbackFakeServer:
    """A local test server; acknowledged event IDs persist across reconnects."""

    def __init__(self) -> None:
        self.acknowledged_event_ids: set[str] = set()

    async def handle(self, websocket: Any) -> None:
        orchestrator: FakeOrchestrator | None = None
        async for raw in websocket:
            event = Envelope.from_json(raw)
            if orchestrator is None:
                orchestrator = FakeOrchestrator(session_id=event.session_id)
                orchestrator.acknowledged_event_ids = self.acknowledged_event_ids
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
