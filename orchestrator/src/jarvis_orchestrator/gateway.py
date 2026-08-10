"""Loopback ``jarvis.voice/1`` WebSocket gateway and durable outbox dispatcher."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from .domain import OrchestratorError, ReasonCode, TaskState, new_uuid4
from .service import OrchestratorService


class VoiceGateway:
    def __init__(self, service: OrchestratorService) -> None:
        self.service = service
        self._publication_locks: dict[str, asyncio.Lock] = {}
        self._execution_tasks: dict[str, asyncio.Task[Any]] = {}
        self._server: Any = None
        self._sweeper: asyncio.Task[Any] | None = None

    def _lock(self, conversation_id: str) -> asyncio.Lock:
        return self._publication_locks.setdefault(conversation_id, asyncio.Lock())

    async def start(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("websockets>=15,<16 is required") from exc
        loop = asyncio.get_running_loop()
        self.service.set_execution_scheduler(
            lambda task_id: loop.call_soon_threadsafe(self._track_execution, task_id)
        )
        self._server = await serve(
            self.handle,
            self.service.config.voice_host,
            self.service.config.voice_port,
            origins=[None],
            max_size=64 * 1024,
            max_queue=32,
            open_timeout=2,
            close_timeout=1,
            ping_interval=10,
            ping_timeout=10,
        )
        self._sweeper = asyncio.create_task(self._sweep_loop(), name="jarvis-deadline-sweeper")

    async def stop(self) -> None:
        self.service.set_execution_scheduler(None)
        if self._sweeper is not None:
            self._sweeper.cancel()
            with suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None
        for task_id in tuple(self._execution_tasks):
            with suppress(OrchestratorError):
                snapshot = self.service.store.get_task(task_id)
                self.service.store.cancel_task(
                    task_id,
                    snapshot.epoch,
                    ReasonCode.SERVICE_INTERRUPTED,
                    terminal_state=TaskState.INTERRUPTED,
                )
        for task in tuple(self._execution_tasks.values()):
            task.cancel()
        if self._execution_tasks:
            await asyncio.gather(*self._execution_tasks.values(), return_exceptions=True)
            self._execution_tasks.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _sweep_loop(self) -> None:
        while True:
            self.service.sweep()
            await asyncio.sleep(0.25)

    def _track_execution(self, task_id: str) -> None:
        existing = self._execution_tasks.get(task_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(self.service.execute_task(task_id), name=f"jarvis-task-{task_id}")
        self._execution_tasks[task_id] = task
        task.add_done_callback(lambda completed: self._execution_tasks.pop(task_id, None))

    def _cancel_execution(self, task_id: str) -> None:
        task = self._execution_tasks.get(task_id)
        if task is not None:
            task.cancel()

    def _cancel_superseded_executions(self, current_task_id: str) -> None:
        for task_id, task in tuple(self._execution_tasks.items()):
            if task_id != current_task_id:
                task.cancel()

    async def _publish(self, websocket: Any, session_id: str, factory: Any, connection_id: str, sent: set[str]) -> None:
        from jarvis_voice.protocol import Envelope

        while True:
            conversation = self.service.store.active_conversation()
            if conversation is None:
                await asyncio.sleep(0.02)
                continue
            events = self.service.store.claim_outbox(
                conversation["conversation_id"],
                connection_id,
                exclude_event_ids=tuple(sent),
            )
            if not events:
                await asyncio.sleep(0.02)
                continue
            lock = self._lock(conversation["conversation_id"])
            for logical in events:
                envelope = Envelope(
                    type=logical.event_type,
                    event_id=logical.event_id,
                    session_id=session_id,
                    seq=factory.next_seq,
                    sent_at=logical.original_sent_at,
                    turn_id=logical.turn_id,
                    response_id=logical.response_id,
                    payload=logical.payload,
                )
                complete = False
                current = True
                try:
                    async with asyncio.timeout(1.0):
                        async with lock:
                            current = self.service.store.outbox_claim_is_current(
                                logical.event_id, connection_id
                            )
                            if current:
                                await websocket.send(envelope.to_json())
                    if not current:
                        continue
                    factory.consume_sequence()
                    sent.add(logical.event_id)
                    complete = True
                finally:
                    with suppress(OrchestratorError):
                        self.service.store.finish_outbox_attempt(logical.event_id, connection_id, complete=complete)
                if not complete:
                    return

    async def _cancel_with_publication_fence(self, websocket: Any, writer: asyncio.Task[Any], task_id: str, epoch: int, conversation_id: str) -> None:
        lock = self._lock(conversation_id)
        acquired = False
        try:
            async with asyncio.timeout(1.0):
                await lock.acquire()
                acquired = True
        except TimeoutError:
            writer.cancel()
            with suppress(asyncio.CancelledError, Exception):
                async with asyncio.timeout(1.0):
                    await websocket.close(code=1011, reason="publication_timeout")
                    await writer
        try:
            self.service.cancel_task(task_id, epoch)
            self._cancel_execution(task_id)
        finally:
            if acquired:
                lock.release()

    async def handle(self, websocket: Any) -> None:
        from jarvis_voice.protocol import Envelope, EventFactory
        from jarvis_voice.sequencing import SequenceResult, SequenceTracker

        peer = websocket.remote_address
        if not peer or peer[0] != "127.0.0.1":
            await websocket.close(code=1008, reason="loopback_required")
            return
        session_id: str | None = None
        writer: asyncio.Task[Any] | None = None
        connection_id = new_uuid4()
        try:
            raw_hello = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            hello = Envelope.from_json(raw_hello)
            if hello.type != "client.hello" or hello.seq != 1:
                await websocket.close(code=1002, reason="hello_required")
                return
            inbound_sequence = SequenceTracker()
            inbound_sequence.observe(hello)
            session_id = hello.session_id
            self.service.attach_client(connection_id)
            factory = EventFactory(session_id=session_id)
            await asyncio.wait_for(
                websocket.send(factory.create("server.ready", {"protocol": "jarvis.voice/1"}).to_json()),
                timeout=1.0,
            )
            sent: set[str] = set()
            writer = asyncio.create_task(self._publish(websocket, session_id, factory, connection_id, sent), name=f"jarvis-publisher-{connection_id}")
            async for raw in websocket:
                event = Envelope.from_json(raw)
                if event.session_id != session_id:
                    await websocket.close(code=1002, reason="session_changed")
                    break
                if inbound_sequence.observe(event) == SequenceResult.DUPLICATE:
                    continue
                if event.type == "input.transcript.final":
                    admission = self.service.admit_envelope(event)
                    if admission.task is not None:
                        self._cancel_superseded_executions(admission.task.task_id)
                        self._track_execution(admission.task.task_id)
                elif event.type == "response.cancel":
                    conversation = self.service.store.active_conversation()
                    if conversation and conversation["active_task_id"]:
                        active = self.service.store.get_task(conversation["active_task_id"])
                        if event.response_id != active.response_id:
                            continue
                        await self._cancel_with_publication_fence(
                            websocket,
                            writer,
                            conversation["active_task_id"],
                            conversation["epoch"],
                            conversation["conversation_id"],
                        )
                elif event.type in {"playback.completed", "playback.interrupted"} and event.response_id:
                    self.service.handle_playback_terminal(event.response_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            with suppress(Exception):
                await websocket.close(code=1011, reason="service_error")
        finally:
            if writer is not None:
                writer.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await writer
            if session_id is not None:
                self.service.detach_client(connection_id)
