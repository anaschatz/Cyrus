"""Loopback ``jarvis.voice/1`` WebSocket gateway and durable outbox dispatcher."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from .domain import (
    OrchestratorError,
    ReasonCode,
    TaskState,
    canonical_sha256,
    new_uuid4,
)
from .runtime_supervisor import RuntimeProcessDeadline, RuntimeSupervisor
from .service import OrchestratorService

MAX_PENDING_AUTHENTICATIONS = 8
AUTH_CAPACITY_WAIT_SECONDS = 0.05
GATEWAY_SWEEPER_COMPONENT = "gateway_sweeper"
GATEWAY_SWEEPER_FAILURE = "runtime_gateway_sweeper_failed"
GATEWAY_LISTENER_COMPONENT = "gateway_listener"
GATEWAY_LISTENER_FAILURE = "runtime_gateway_listener_failed"


class VoiceGateway:
    def __init__(
        self,
        service: OrchestratorService,
        *,
        auth_key: Any,
        business_admission: Callable[[], bool] | None = None,
        service_error_reporter: Callable[[str, int | None], None] | None = None,
    ) -> None:
        from jarvis_voice.voice_auth import VoiceAuthKey

        if not isinstance(auth_key, VoiceAuthKey):
            raise RuntimeError("voice_auth_key_required")
        self.service = service
        self.auth_key = auth_key
        # Admission is fail-closed unless a lifecycle owner explicitly injects
        # the shared readiness predicate.
        self._business_admission = business_admission or (lambda: False)
        self._service_error_reporter = service_error_reporter
        self._publication_locks: dict[str, asyncio.Lock] = {}
        self._execution_tasks: dict[str, asyncio.Task[Any]] = {}
        self._server: Any = None
        self._sweeper: asyncio.Task[Any] | None = None
        self._server_monitor: asyncio.Task[Any] | None = None
        self._cloud_connections: dict[str, Any] = {}
        self._action_connections: dict[str, Any] = {}
        self._auth_slots = asyncio.BoundedSemaphore(MAX_PENDING_AUTHENTICATIONS)

    def _report_service_error(self, phase: str, exc: Exception) -> None:
        """Publish only a bounded error class and optional SQLite code."""

        reporter = self._service_error_reporter
        if reporter is None:
            return
        if isinstance(exc, sqlite3.IntegrityError):
            category = "sqlite_integrity"
            detail = getattr(exc, "sqlite_errorcode", None)
        elif isinstance(exc, sqlite3.OperationalError):
            category = "sqlite_operational"
            detail = getattr(exc, "sqlite_errorcode", None)
        elif isinstance(exc, OrchestratorError):
            category = "orchestrator"
            detail = exc.status
        elif isinstance(exc, RuntimeError):
            category = "runtime"
            detail = None
        elif isinstance(exc, (KeyError, TypeError, ValueError)):
            category = "validation"
            detail = None
        else:
            category = "unexpected"
            detail = None
        with suppress(Exception):
            reporter(f"gateway_service_error_{phase}_{category}", detail)

    def _lock(self, conversation_id: str) -> asyncio.Lock:
        return self._publication_locks.setdefault(conversation_id, asyncio.Lock())

    async def start(
        self,
        *,
        supervisor: RuntimeSupervisor | None = None,
        shutdown_deadline: RuntimeProcessDeadline | None = None,
    ) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("websockets>=15,<16 is required") from exc
        loop = asyncio.get_running_loop()
        self.service.set_execution_scheduler(
            lambda task_id: loop.call_soon_threadsafe(self._track_execution, task_id)
        )
        try:
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
            server = self._server
            if supervisor is None:
                # Standalone gateways retain the public API; the production CLI
                # always supplies the process supervisor.
                self._sweeper = asyncio.create_task(
                    self._sweep_loop(), name="jarvis-deadline-sweeper"
                )
                self._server_monitor = asyncio.create_task(
                    self._watch_server(server), name="jarvis-gateway-listener-monitor"
                )
            else:
                self._sweeper = supervisor.start_task(
                    GATEWAY_SWEEPER_COMPONENT,
                    lambda: self._sweep_loop(
                        lambda: supervisor.acknowledge_ready(
                            GATEWAY_SWEEPER_COMPONENT
                        )
                    ),
                    failure_code=GATEWAY_SWEEPER_FAILURE,
                )
                self._server_monitor = supervisor.start_task(
                    GATEWAY_LISTENER_COMPONENT,
                    lambda: self._watch_server(
                        server,
                        lambda: supervisor.acknowledge_ready(
                            GATEWAY_LISTENER_COMPONENT
                        ),
                    ),
                    failure_code=GATEWAY_LISTENER_FAILURE,
                )
        except BaseException:
            cleanup_supervisor = supervisor or RuntimeSupervisor(
                shutdown_timeout_seconds=3.0
            )
            cleanup_supervisor.begin_shutdown()
            owned_deadline = shutdown_deadline is None
            deadline = shutdown_deadline or cleanup_supervisor.process_deadline(3.0)
            deadline.begin()
            try:
                await cleanup_supervisor.run_bounded_cleanup(
                    "gateway_startup",
                    self._rollback_failed_start,
                    failure_code="runtime_gateway_startup_cleanup_failed",
                    timeout_code="runtime_gateway_startup_cleanup_timeout",
                    timeout_seconds=deadline.remaining(),
                )
            finally:
                if owned_deadline:
                    deadline.disarm()
            raise

    async def _rollback_failed_start(self) -> None:
        self.service.set_execution_scheduler(None)
        startup_tasks = tuple(
            task
            for task in (self._sweeper, self._server_monitor)
            if task is not None
        )
        for task in startup_tasks:
            task.cancel()
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)
        self._sweeper = None
        self._server_monitor = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

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
        if self._server_monitor is not None:
            with suppress(asyncio.CancelledError):
                await self._server_monitor
            self._server_monitor = None

    @staticmethod
    async def _watch_server(
        server: Any,
        ready: Callable[[], None] | None = None,
    ) -> None:
        if ready is not None:
            ready()
        await server.wait_closed()

    async def _sweep_loop(
        self,
        ready: Callable[[], None] | None = None,
    ) -> None:
        acknowledged = False
        while True:
            self.service.sweep()
            if not acknowledged and ready is not None:
                ready()
                acknowledged = True
            await asyncio.sleep(0.25)

    def _track_execution(self, task_id: str) -> None:
        existing = self._execution_tasks.get(task_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(self.service.execute_task(task_id), name=f"jarvis-task-{task_id}")
        self._execution_tasks[task_id] = task
        def discard_completed(completed: asyncio.Task[Any]) -> None:
            if self._execution_tasks.get(task_id) is completed:
                self._execution_tasks.pop(task_id)

        task.add_done_callback(discard_completed)

    def _cancel_execution(self, task_id: str) -> None:
        task = self._execution_tasks.get(task_id)
        if task is not None:
            task.cancel()

    async def _cancel_superseded_executions(self, current_task_id: str) -> None:
        superseded = tuple(
            task
            for task_id, task in self._execution_tasks.items()
            if task_id != current_task_id and not task.done()
        )
        for task in superseded:
            task.cancel()
        if superseded:
            # A cancelled streaming request owns the inference connection until
            # its async context exits.  Starting the replacement turn before
            # that cleanup settles can leave the single-request appliance busy
            # behind an already-superseded generation.
            await asyncio.gather(*superseded, return_exceptions=True)

    async def _publish(
        self,
        websocket: Any,
        session_id: str,
        factory: Any,
        connection_id: str,
        *,
        cloud_factory: Any | None = None,
        scheduler_factory: Any | None = None,
        action_factory: Any | None = None,
    ) -> None:
        from jarvis_voice.protocol import Envelope
        from jarvis_voice.cloud_approval import CLOUD_APPROVAL_PROTOCOL, CloudApprovalEnvelope
        from jarvis_voice.scheduler_protocol import SCHEDULER_PROTOCOL, SchedulerEnvelope
        from jarvis_voice.action_approval import ACTION_APPROVAL_PROTOCOL, ActionApprovalEnvelope

        try:
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
                        conversation["conversation_id"], connection_id
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
        except Exception as exc:
            self._report_service_error("publication", exc)
            with suppress(Exception):
                await websocket.close(code=1011, reason="publication_failed")

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
        from jarvis_voice.voice_auth import VoiceAuthError
        from jarvis_voice.voice_auth_transport import (
            AUTH_PROBE_FRAME,
            AUTH_PROBE_PATH,
            AUTH_PROBE_SUCCESS_FRAME,
            authenticate_async_server,
        )

        peer = websocket.remote_address
        if not peer or peer[0] != "127.0.0.1":
            await websocket.close(code=1008, reason="loopback_required")
            return
        request = getattr(websocket, "request", None)
        request_path = getattr(request, "path", "/")
        if request_path not in {"/", AUTH_PROBE_PATH}:
            await websocket.close(code=1008, reason="endpoint_not_allowed")
            return
        try:
            await asyncio.wait_for(
                self._auth_slots.acquire(),
                timeout=AUTH_CAPACITY_WAIT_SECONDS,
            )
        except TimeoutError:
            await websocket.close(code=1013, reason="authentication_capacity")
            return
        try:
            connection_id = new_uuid4()
            try:
                raw_hello = await authenticate_async_server(
                    websocket,
                    self.auth_key,
                    connection_id=connection_id,
                )
            except VoiceAuthError:
                with suppress(Exception):
                    await websocket.close(code=1008, reason="authentication_failed")
                return
        finally:
            self._auth_slots.release()
        if request_path == AUTH_PROBE_PATH:
            expected_probe = AUTH_PROBE_FRAME.encode("ascii")
            received_probe = (
                raw_hello.encode("utf-8")
                if isinstance(raw_hello, str)
                else raw_hello
            )
            if received_probe != expected_probe:
                with suppress(Exception):
                    await websocket.close(code=1008, reason="authentication_probe_invalid")
                return
            await websocket.send(AUTH_PROBE_SUCCESS_FRAME)
            return
        if not self._business_admission():
            await websocket.close(code=1013, reason="runtime_not_ready")
            return
        session_id: str | None = None
        writer: asyncio.Task[Any] | None = None
        cloud_negotiated = False
        scheduler_negotiated = False
        scheduler_speech_negotiated = False
        scheduler_actions_negotiated = False
        action_negotiated = False
        capture_idle = True
        service_phase = "handshake"
        try:
            hello = Envelope.from_json(raw_hello)
            if hello.type != "client.hello" or hello.seq != 1:
                await websocket.close(code=1002, reason="hello_required")
                return
            inbound_sequence = SequenceTracker()
            inbound_sequence.observe(hello)
            session_id = hello.session_id
            capabilities = hello.payload.get("capabilities", [])
            cloud_negotiated = (
                self.service.cloud_approval_available()
                and isinstance(capabilities, list)
                and CLOUD_APPROVAL_PROTOCOL in capabilities
            )
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
            writer = asyncio.create_task(
                self._publish(
                    websocket,
                    session_id,
                    factory,
                    connection_id,
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
                except (ValueError, TypeError, RecursionError):
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
                service_phase = "voice_event"
                if event.session_id != session_id:
                    await websocket.close(code=1002, reason="session_changed")
                    break
                if inbound_sequence.observe(event) == SequenceResult.DUPLICATE:
                    continue
                if event.type == "input.transcript.final":
                    capture_idle = True
                    service_phase = "transcript_admission"
                    admission = self.service.admit_envelope(event)
                    if admission.task is not None:
                        service_phase = "transcript_supersede"
                        await self._cancel_superseded_executions(admission.task.task_id)
                        service_phase = "transcript_schedule"
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
                    self.service.handle_playback_terminal(
                        event_id=event.event_id,
                        semantic_sha256=canonical_sha256(
                            {
                                "type": event.type,
                                "turn_id": event.turn_id,
                                "response_id": event.response_id,
                                "payload": dict(event.payload),
                            }
                        ),
                        response_id=event.response_id,
                        terminal_type=event.type,
                        client_sent_at=event.sent_at,
                    )
        except asyncio.CancelledError:
            raise
        except ProtocolError as exc:
            with suppress(Exception):
                close_code = 1009 if exc.code == "protocol.frame_too_large" else 1002
                await websocket.close(code=close_code, reason="protocol_error")
        except Exception as exc:
            self._report_service_error(service_phase, exc)
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
