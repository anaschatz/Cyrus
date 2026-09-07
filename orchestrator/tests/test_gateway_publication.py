from __future__ import annotations

import asyncio
import json
import sqlite3
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401
from helpers import StoreCase
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import utc_now
from jarvis_orchestrator.gateway import VoiceGateway
from jarvis_orchestrator.service import OrchestratorService
from jarvis_voice.protocol import EventFactory
from test_planner_scheduler_tool_e2e import FinalModel
from test_voice_gateway_auth import AUTH_KEY, AuthWebSocket


class WaitingWebSocket(AuthWebSocket):
    def __init__(self, *, fail_send=False):
        super().__init__(EventFactory().create("client.hello", {
            "capabilities": ["speech_commits", "speech_done", "output_epochs"],
            "protocol": "jarvis.voice/1",
        }).to_json())
        self.fail_send = fail_send
        self.closed_event = asyncio.Event()
        self.delivered = asyncio.Queue()

    async def __anext__(self):
        await self.closed_event.wait()
        raise StopAsyncIteration

    async def send(self, raw):
        if json.loads(raw)["type"] == "response.text.delta":
            if self.fail_send:
                raise OSError("synthetic private detail")
            self.delivered.put_nowait(json.loads(raw)["payload"]["chunk_index"])
        await super().send(raw)

    async def close(self, *, code, reason):
        await super().close(code=code, reason=reason)
        self.closed_event.set()


class GatewayPublicationTests(StoreCase, unittest.IsolatedAsyncioTestCase):
    def setup_gateway(self):
        store = self.make_store()
        task = self.executing(store, self.admitted(
            store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4()), now=utc_now(),
        ), now=utc_now())
        store.start_response(task.task_id, task.epoch, "el-GR")
        store.append_response_event(task.task_id, task.epoch, "response.text.delta",
                                    {"chunk_index": 0, "text": "Δοκιμή."})
        service = OrchestratorService(OrchestratorConfig(
            database_path=store.path, control_bearer_token="control",
            inference_bearer_token="inference",
        ), store, FinalModel())
        reports = []
        gateway = VoiceGateway(service, auth_key=AUTH_KEY, business_admission=lambda: True,
                               service_error_reporter=lambda *args: reports.append(args))
        return gateway, store, task, reports

    async def test_publication_failure_closes_connection_and_detaches_client(self):
        for phase in ("claim", "send"):
            with self.subTest(phase=phase):
                gateway, store, task, reports = self.setup_gateway()
                socket = WaitingWebSocket(fail_send=phase == "send")
                with patch.object(store, "claim_outbox", wraps=store.claim_outbox) as claim:
                    if phase == "claim":
                        claim.side_effect = sqlite3.OperationalError("synthetic private detail")
                    await asyncio.wait_for(gateway.handle(socket), 0.5)
                self.assertEqual(socket.closed, [(1011, "publication_failed")])
                self.assertFalse(gateway.service.has_client())
                self.assertEqual(len(reports), 1)
                self.assertTrue(reports[0][0].startswith("gateway_service_error_publication_"))
                self.assertNotIn("private", repr(reports))
                replay = store.claim_outbox(task.conversation_id, "reconnected")
                self.assertTrue(any(event.event_type == "response.text.delta" for event in replay))

    async def test_old_completion_cannot_untrack_replacement_execution(self):
        started = asyncio.Event()
        executions = []

        async def execute(task_id):
            executions.append(asyncio.current_task())
            if len(executions) == 1:
                asyncio.get_running_loop().call_soon(gateway._track_execution, task_id)
                return
            started.set()
            await asyncio.Event().wait()

        gateway = VoiceGateway(SimpleNamespace(execute_task=execute), auth_key=AUTH_KEY)
        gateway._track_execution("same-task")
        try:
            await asyncio.wait_for(started.wait(), 0.5)
            self.assertIs(gateway._execution_tasks.get("same-task"), executions[1])
            gateway._cancel_execution("same-task")
            with self.assertRaises(asyncio.CancelledError):
                await executions[1]
            self.assertNotIn("same-task", gateway._execution_tasks)
        finally:
            for execution in executions:
                execution.cancel()
            await asyncio.gather(*executions, return_exceptions=True)

    async def test_only_completed_attempts_suppress_same_connection_replay(self):
        _gateway, store, task, _reports = self.setup_gateway()
        first = store.claim_outbox(task.conversation_id, "first")
        self.assertTrue(first)
        for event in first:
            store.finish_outbox_attempt(event.event_id, "first", complete=False)
        retry = store.claim_outbox(task.conversation_id, "first")
        self.assertEqual(retry, first)
        for event in retry:
            store.finish_outbox_attempt(event.event_id, "first", complete=True)
        self.assertEqual(store.claim_outbox(task.conversation_id, "first"), [])
        self.assertEqual(store.claim_outbox(task.conversation_id, "reconnected"), first)

    async def test_long_connection_keeps_publishing_with_bounded_sql_parameters(self):
        gateway, store, task, reports = self.setup_gateway()
        for index in range(1, 40):
            store.append_response_event(task.task_id, task.epoch, "response.text.delta",
                                        {"chunk_index": index, "text": "Δοκιμή."})
        store._connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 32)
        socket = WaitingWebSocket()
        handler = asyncio.create_task(gateway.handle(socket))
        try:
            for index in range(40):
                self.assertEqual(await asyncio.wait_for(socket.delivered.get(), 0.5), index)
            store.append_response_event(task.task_id, task.epoch, "response.text.delta",
                                        {"chunk_index": 40, "text": "Τέλος."})
            self.assertEqual(await asyncio.wait_for(socket.delivered.get(), 0.5), 40)
            self.assertTrue(socket.delivered.empty())
            self.assertEqual(reports, [])
            self.assertEqual(socket.closed, [])
        finally:
            socket.closed_event.set()
            await asyncio.wait_for(handler, 0.5)


if __name__ == "__main__":
    unittest.main()
