"""Loopback ``jarvis.voice/1`` WebSocket gateway and durable outbox dispatcher."""

from __future__ import annotations

import asyncio
import json
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
        self._cloud_connections: dict[str, Any] = {}
        self._action_connections: dict[str, Any] = {}

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

    async def _publish(
        self,
        websocket: Any,
        session_id: str,
        factory: Any,
        connection_id: str,
        sent: set[str],
        *,
        cloud_factory: Any | None = None,
        scheduler_factory: Any | None = None,
        action_factory: Any | None = None,
    ) -> None:
        from jarvis_voice.protocol import Envelope
        from jarvis_voice.cloud_approval import CLOUD_APPROVAL_PROTOCOL, CloudApprovalEnvelope
        from jarvis_voice.scheduler_protocol import SCHEDULER_PROTOCOL, SchedulerEnvelope
        from jarvis_voice.action_approval import ACTION_APPROVAL_PROTOCOL, ActionApprovalEnvelope

        while True:
            if scheduler_factory is not None and self.service.scheduler is not None:
                self.service.scheduler.promote_next_approval(
                    session_id, supported_protocols=(SCHEDULER_PROTOCOL,)
                )
            if cloud_factory is not None:
                self.service.store.claim_cloud_presentation_owner(session_id)
            if action_factory is not None and self.service.tool_store is not None:
                self.service.tool_store.promote_next_action_approval(session_id)
            conversation = self.service.store.active_conversation()
            events = (
                self.service.store.claim_outbox(
                    conversation["conversation_id"], connection_id, exclude_event_ids=tuple(sent)
                ) if conversation is not None else []
            )
            cloud_events = (
                self.service.store.claim_cloud_extension_events(conversation["conversation_id"])
                if cloud_factory is not None and conversation is not None
                else []
            )
            scheduler_events = (
                self.service.scheduler.claim_outbox(connection_id)
                if scheduler_factory is not None and self.service.scheduler is not None
                else []
            )
            action_events = (
                self.service.tool_store.claim_action_extension_events()
                if action_factory is not None and self.service.tool_store is not None
                else []
            )
            publications = [
                (event.ordinal, "base", event) for event in events
            ] + [
                (int(event["ordinal"]), "cloud", event) for event in cloud_events
            ] + [
                (index, "scheduler", event) for index, event in enumerate(scheduler_events)
            ] + [
                (int(event["ordinal"]), "action", event) for event in action_events
            ]
            publications.sort(key=lambda item: item[0])
            if not publications:
                await asyncio.sleep(0.02)
                continue
            lock = self._lock(conversation["conversation_id"] if conversation is not None else f"scheduler:{session_id}")
            for _ordinal, protocol_kind, logical in publications:
                if protocol_kind == "base":
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
                elif protocol_kind == "cloud":
                    envelope = CloudApprovalEnvelope(
                        protocol=CLOUD_APPROVAL_PROTOCOL,
                        type=logical["event_type"],
                        event_id=logical["event_id"],
                        session_id=session_id,
                        seq=cloud_factory.next_seq,
                        sent_at=logical["original_sent_at"],
                        payload=logical["payload"],
                    )
                elif protocol_kind == "action":
                    envelope = ActionApprovalEnvelope(
                        protocol=ACTION_APPROVAL_PROTOCOL,
                        type=logical["event_type"],
                        event_id=logical["event_id"],
                        session_id=session_id,
                        seq=action_factory.next_seq,
                        sent_at=logical["original_sent_at"],
                        payload=logical["payload"],
                    )
                else:
                    envelope = SchedulerEnvelope(
                        protocol=SCHEDULER_PROTOCOL,
                        type=logical["event_type"],
                        event_id=logical["event_id"],
                        session_id=session_id,
                        seq=scheduler_factory.next_seq,
                        sent_at=logical["original_sent_at"],
                        payload=logical["payload"],
                    )
                complete = False
                current = True
                try:
                    async with asyncio.timeout(1.0):
                        async with lock:
                            current = (
                                self.service.store.outbox_claim_is_current(logical.event_id, connection_id)
                                if protocol_kind == "base"
                                else (
                                    self.service.store.cloud_extension_is_current(logical["event_id"])
                                    if protocol_kind == "cloud" else True
                                )
                            )
                            if current:
                                await websocket.send(envelope.to_json())
                    if not current:
                        continue
                    if protocol_kind == "base":
                        factory.consume_sequence()
                        sent.add(logical.event_id)
                    elif protocol_kind == "cloud":
                        cloud_factory.consume_sequence()
                    elif protocol_kind == "action":
                        action_factory.consume_sequence()
                    else:
                        scheduler_factory.consume_sequence()
                    complete = True
                finally:
                    with suppress(OrchestratorError):
                        if protocol_kind == "base":
                            self.service.store.finish_outbox_attempt(logical.event_id, connection_id, complete=complete)
                        elif protocol_kind == "cloud":
                            self.service.store.finish_cloud_extension_write(logical["event_id"], complete=complete)
                        elif protocol_kind == "action" and self.service.tool_store is not None:
                            self.service.tool_store.finish_action_extension_write(logical["event_id"], complete=complete)
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
        from jarvis_voice.cloud_approval import (
            CLIENT_EXTENSION_EVENTS,
            CLOUD_APPROVAL_PROTOCOL,
            CloudApprovalEnvelope,
            CloudApprovalEventFactory,
        )
        from jarvis_voice.errors import ProtocolError
        from jarvis_voice.scheduler_protocol import (
            APPROVAL_ARBITER_PROTOCOL,
            CLIENT_SCHEDULER_EVENTS,
            SCHEDULER_PROTOCOL,
            SchedulerEnvelope,
            SchedulerEventFactory,
        )
        from jarvis_voice.action_approval import (
            ACTION_APPROVAL_PROTOCOL,
            CLIENT_EVENTS as CLIENT_ACTION_EVENTS,
            ActionApprovalEnvelope,
            ActionApprovalEventFactory,
        )
        from jarvis_voice.sequencing import SequenceResult, SequenceTracker

        peer = websocket.remote_address
        if not peer or peer[0] != "127.0.0.1":
            await websocket.close(code=1008, reason="loopback_required")
            return
        session_id: str | None = None
        writer: asyncio.Task[Any] | None = None
        cloud_negotiated = False
        scheduler_negotiated = False
        scheduler_speech_negotiated = False
        scheduler_actions_negotiated = False
        action_negotiated = False
        capture_idle = True
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
            capabilities = hello.payload.get("capabilities", [])
            cloud_negotiated = isinstance(capabilities, list) and CLOUD_APPROVAL_PROTOCOL in capabilities
            action_negotiated = (
                self.service.tool_store is not None
                and isinstance(capabilities, list)
                and ACTION_APPROVAL_PROTOCOL in capabilities
                and APPROVAL_ARBITER_PROTOCOL in capabilities
            )
            scheduler_negotiated = (
                self.service.scheduler is not None
                and self.service.scheduler.readiness()["state"] == "READY"
                and bool(self.service.config.voice_scheduler_compatibility_sha256)
                and isinstance(capabilities, list)
                and SCHEDULER_PROTOCOL in capabilities
                and APPROVAL_ARBITER_PROTOCOL in capabilities
                and hello.payload.get("scheduler_manifest_sha256")
                    == self.service.config.voice_scheduler_compatibility_sha256
            )
            scheduler_speech_negotiated = scheduler_negotiated and "scheduler_notification_speech" in capabilities
            scheduler_actions_negotiated = (
                scheduler_negotiated
                and "scheduler_notification_actions" in capabilities
                and bool(self.service.config.voice_scheduler_callback_qualification_sha256)
                and hello.payload.get("scheduler_callback_qualification_sha256")
                == self.service.config.voice_scheduler_callback_qualification_sha256
            )
            factory = EventFactory(session_id=session_id)
            await asyncio.wait_for(
                websocket.send(factory.create("server.ready", {"protocol": "jarvis.voice/1"}).to_json()),
                timeout=1.0,
            )
            cloud_factory = CloudApprovalEventFactory(session_id) if cloud_negotiated else None
            scheduler_factory = SchedulerEventFactory(session_id) if scheduler_negotiated else None
            action_factory = ActionApprovalEventFactory(session_id) if action_negotiated else None
            if cloud_factory is not None:
                await asyncio.wait_for(
                    websocket.send(
                        cloud_factory.create(
                            "cloud.approval.ready",
                            {"protocol": CLOUD_APPROVAL_PROTOCOL},
                        ).to_json()
                    ),
                    timeout=1.0,
                )
                ownership = self.service.store.claim_cloud_presentation_owner(session_id)
                old_socket = None
                if ownership is not None and ownership["old_session_id"]:
                    old_socket = self._cloud_connections.get(ownership["old_session_id"])
                self._cloud_connections[session_id] = websocket
                if old_socket is not None and old_socket is not websocket:
                    with suppress(Exception):
                        await old_socket.close(code=4001, reason="session_superseded")
            if scheduler_factory is not None:
                await asyncio.wait_for(
                    websocket.send(scheduler_factory.create("scheduler.ready", {
                        "protocol": SCHEDULER_PROTOCOL,
                        "maximum_frame_bytes": 16 * 1024,
                        "sequence_namespace": "protocol_direction_connection",
                    }).to_json()), timeout=1.0,
                )
                self.service.scheduler.promote_next_approval(
                    session_id,
                    supported_protocols=(SCHEDULER_PROTOCOL,),
                )
            if action_factory is not None and self.service.tool_store is not None:
                await asyncio.wait_for(
                    websocket.send(
                        action_factory.create(
                            "action.approval.ready",
                            {"protocol": ACTION_APPROVAL_PROTOCOL},
                        ).to_json()
                    ),
                    timeout=1.0,
                )
                self._action_connections[session_id] = websocket
                self.service.tool_store.reset_action_extension_replay()
                self.service.tool_store.promote_next_action_approval(session_id)
            self.service.attach_client(
                connection_id,
                cloud_session_id=session_id if cloud_negotiated else None,
                scheduler_session_id=session_id if scheduler_negotiated else None,
                scheduler_action_session_id=(session_id if scheduler_actions_negotiated else None),
            )
            active_after_handshake = self.service.store.active_conversation()
            if cloud_negotiated and active_after_handshake is not None:
                self.service.store.reset_cloud_extension_replay(
                    active_after_handshake["conversation_id"]
                )
            sent: set[str] = set()
            writer = asyncio.create_task(
                self._publish(
                    websocket,
                    session_id,
                    factory,
                    connection_id,
                    sent,
                    cloud_factory=cloud_factory,
                    scheduler_factory=scheduler_factory,
                    action_factory=action_factory,
                ),
                name=f"jarvis-publisher-{connection_id}",
            )
            extension_next_seq = 1
            extension_seen: dict[int, str] = {}
            scheduler_next_seq = 1
            scheduler_seen: dict[int, str] = {}
            action_next_seq = 1
            action_seen: dict[int, str] = {}
            async for raw in websocket:
                try:
                    preview = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                    await websocket.close(code=1002, reason="invalid_json")
                    break
                if isinstance(preview, dict) and "protocol" in preview:
                    protocol = preview.get("protocol")
                    if protocol == ACTION_APPROVAL_PROTOCOL:
                        if not action_negotiated or self.service.tool_store is None:
                            await websocket.close(code=1002, reason="extension_not_negotiated")
                            break
                        event = ActionApprovalEnvelope.from_json(raw)
                        if event.type not in CLIENT_ACTION_EVENTS or event.session_id != session_id:
                            await websocket.close(code=1002, reason="invalid_extension_event")
                            break
                        fingerprint = event.to_json()
                        if event.seq == action_next_seq:
                            action_seen[event.seq] = fingerprint
                            action_next_seq += 1
                        elif event.seq < action_next_seq and action_seen.get(event.seq) == fingerprint:
                            continue
                        else:
                            await websocket.close(code=1002, reason="extension_sequence")
                            break
                        if event.type == "action.approval.event_ack":
                            self.service.tool_store.acknowledge_action_extension(
                                event.payload["acked_event_id"]
                            )
                        elif event.type == "action.approval.decision":
                            if not capture_idle:
                                await websocket.close(code=1002, reason="capture_active")
                                break
                            outcome = self.service.tool_store.decide_action_transport(
                                decision_event_id=event.event_id,
                                semantic_sha256=event.semantic_sha256,
                                approval_id=event.payload["approval_id"],
                                presentation_id=event.payload["presentation_id"],
                                requested_event_id=event.payload["requested_event_id"],
                                approval_binding_sha256=event.payload["approval_binding_sha256"],
                                decision_nonce=event.payload["decision_nonce"],
                                decision=event.payload["decision"],
                                session_id=session_id,
                            )
                            if not outcome.get("duplicate"):
                                self._track_execution(outcome["task_id"])
                            self.service.tool_store.promote_next_action_approval(session_id)
                        continue
                    if protocol == SCHEDULER_PROTOCOL:
                        if not scheduler_negotiated or self.service.scheduler is None:
                            await websocket.close(code=1002, reason="extension_not_negotiated")
                            break
                        event = SchedulerEnvelope.from_json(raw)
                        if event.type not in CLIENT_SCHEDULER_EVENTS or event.session_id != session_id:
                            await websocket.close(code=1002, reason="invalid_extension_event")
                            break
                        fingerprint = event.to_json()
                        if event.seq == scheduler_next_seq:
                            scheduler_seen[event.seq] = fingerprint
                            scheduler_next_seq += 1
                        elif event.seq < scheduler_next_seq and scheduler_seen.get(event.seq) == fingerprint:
                            continue
                        else:
                            await websocket.close(code=1002, reason="extension_sequence")
                            break
                        if event.type == "scheduler.event_ack":
                            self.service.scheduler.acknowledge_scheduler_event(
                                event.payload["acked_event_id"], status=event.payload["status"],
                                speech_supported=scheduler_speech_negotiated,
                            )
                        elif event.type == "scheduler.approval.decision":
                            if not capture_idle:
                                await websocket.close(code=1002, reason="capture_active")
                                break
                            scheduler_result = self.service.scheduler.decide_approval(
                                approval_id=event.payload["approval_id"],
                                proposal_id=event.payload["proposal_id"],
                                proposal_sha256=event.payload["proposal_sha256"],
                                decision=event.payload["decision"],
                                cause_event_id=event.event_id,
                                owner_session_id=session_id,
                                presentation_id=event.payload["presentation_id"],
                                requested_event_id=event.payload["requested_event_id"],
                                decision_nonce=event.payload["decision_nonce"],
                            )
                            task_id = self.service.scheduler.task_id_for_proposal(
                                scheduler_result["proposal_id"]
                            )
                            if task_id is not None:
                                self._track_execution(task_id)
                            self.service.scheduler.promote_next_approval(
                                session_id, supported_protocols=(SCHEDULER_PROTOCOL,)
                            )
                            if cloud_negotiated:
                                self.service.store.claim_cloud_presentation_owner(session_id)
                        elif event.type == "scheduler.notification.receipt":
                            self.service.scheduler.record_delivery_receipt(
                                event_id=event.event_id,
                                semantic_sha256=event.semantic_sha256,
                                notification=event.payload["notification_id"],
                                attempt_id=event.payload["attempt_id"],
                                stage=event.payload["stage"],
                                reason_code=event.payload["reason_code"],
                                observed_at_utc=event.payload["observed_at_utc"],
                            )
                        elif event.type == "scheduler.notification.action":
                            if not scheduler_actions_negotiated:
                                await websocket.close(code=1002, reason="extension_not_negotiated")
                                break
                            self.service.scheduler.apply_notification_action(
                                event_id=event.event_id,
                                semantic_sha256=event.semantic_sha256,
                                action=event.payload["action"],
                                job_id=event.payload["job_id"],
                                job_revision=event.payload["job_revision"],
                                occurrence=event.payload["occurrence_id"],
                                expected_generation=event.payload["snooze_generation"],
                                requested_due_at_utc=event.payload["requested_due_at_utc"],
                            )
                        continue
                    if protocol != CLOUD_APPROVAL_PROTOCOL or not cloud_negotiated:
                        await websocket.close(code=1002, reason="extension_not_negotiated")
                        break
                    event = CloudApprovalEnvelope.from_json(raw)
                    if event.type not in CLIENT_EXTENSION_EVENTS or event.session_id != session_id:
                        await websocket.close(code=1002, reason="invalid_extension_event")
                        break
                    fingerprint = event.to_json()
                    if event.seq == extension_next_seq:
                        extension_seen[event.seq] = fingerprint
                        extension_next_seq += 1
                    elif event.seq < extension_next_seq and extension_seen.get(event.seq) == fingerprint:
                        continue
                    else:
                        await websocket.close(code=1002, reason="extension_sequence")
                        break
                    if event.type == "cloud.approval.event_ack":
                        self.service.store.acknowledge_cloud_extension(event.payload["acked_event_id"])
                    elif event.type == "cloud.approval.decision":
                        outcome = self.service.store.decide_cloud_approval(
                            decision_event_id=event.event_id,
                            semantic_sha256=event.semantic_sha256,
                            approval_id=event.payload["approval_id"],
                            presentation_id=event.payload["presentation_id"],
                            requested_event_id=event.payload["requested_event_id"],
                            approval_binding_sha256=event.payload["approval_binding_sha256"],
                            decision_nonce=event.payload["decision_nonce"],
                            decision=event.payload["decision"],
                            session_id=session_id,
                            capture_idle=capture_idle,
                        )
                        if outcome["result"]["result"] == "applied":
                            task = self.service.store.resume_cloud_decision(event.payload["approval_id"])
                            self._track_execution(task.task_id)
                            if scheduler_negotiated and self.service.scheduler is not None:
                                self.service.scheduler.promote_next_approval(
                                    session_id, supported_protocols=(SCHEDULER_PROTOCOL,)
                                )
                            if cloud_negotiated:
                                self.service.store.claim_cloud_presentation_owner(session_id)
                    continue
                event = Envelope.from_json(raw)
                if event.session_id != session_id:
                    await websocket.close(code=1002, reason="session_changed")
                    break
                if inbound_sequence.observe(event) == SequenceResult.DUPLICATE:
                    continue
                if event.type == "input.transcript.final":
                    capture_idle = True
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
                elif event.type == "input.capture.started":
                    capture_idle = False
                elif event.type == "input.no_speech":
                    capture_idle = True
                elif event.type in {"playback.completed", "playback.interrupted"} and event.response_id:
                    self.service.handle_playback_terminal(event.response_id)
        except asyncio.CancelledError:
            raise
        except ProtocolError as exc:
            with suppress(Exception):
                close_code = 1009 if exc.code == "protocol.frame_too_large" else 1002
                await websocket.close(code=close_code, reason="protocol_error")
        except Exception:
            with suppress(Exception):
                await websocket.close(code=1011, reason="service_error")
        finally:
            if writer is not None:
                writer.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await writer
            if session_id is not None:
                if self.service.scheduler is not None:
                    self.service.scheduler.mark_owner_disconnected(session_id)
                self.service.detach_client(connection_id)
                if self._cloud_connections.get(session_id) is websocket:
                    self._cloud_connections.pop(session_id, None)
                if self._action_connections.get(session_id) is websocket:
                    self._action_connections.pop(session_id, None)
