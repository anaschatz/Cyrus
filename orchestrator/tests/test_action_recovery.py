from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
import uuid
from contextlib import ExitStack
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import _bootstrap  # noqa: F401
import test_action_approval_v1
import test_m1c_cloud_contract
import test_tool_store_v6
import test_voice_gateway_auth
from helpers import NOW
from jarvis_orchestrator.domain import OrchestratorError
from jarvis_orchestrator.gateway import VoiceGateway
from jarvis_voice.protocol import EventFactory
from jarvis_voice.action_approval import ActionApprovalController, ActionApprovalEventFactory, ActionApprovalJournal
from jarvis_voice.approval_arbiter import GlobalApprovalController, LocalApprovalState
from jarvis_voice.cloud_approval import CloudApprovalController, CloudApprovalEventFactory, CloudApprovalJournal


class ActionRecoveryTests(unittest.TestCase):
    def test_acknowledged_request_reopens_and_transfers_to_the_new_session(self):
        fixture = test_tool_store_v6.ToolStoreV6Tests()
        self.addCleanup(fixture.doCleanups)
        _, _, tools, _, _, old_decision = fixture.prepared_action()
        row, = tools.claim_action_extension_events()
        requested = ActionApprovalEventFactory(old_decision["session_id"]).create(
            row["event_type"], row["payload"], event_id=row["event_id"], sent_at=row["original_sent_at"])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for restarting in (False, True):
                with ExitStack() as stack:
                    cloud_journal = CloudApprovalJournal(folder / "cloud.sqlite3")
                    stack.callback(cloud_journal.close)
                    journal = ActionApprovalJournal(folder / "action.sqlite3")
                    stack.callback(journal.close)
                    arbiter = GlobalApprovalController(cloud_journal)
                    session_id = str(uuid.uuid4()) if restarting else requested.session_id
                    client = ActionApprovalController(session_id=session_id, journal=journal, approval_arbiter=arbiter)
                    if restarting:
                        self.assertEqual(arbiter.presentation.state, LocalApprovalState.STAGED)
                        self.assertEqual(tools.reset_action_extension_replay(session_id, now=NOW), requested.session_id)
                        replay, = tools.claim_action_extension_events()
                        self.assertEqual(replay["event_id"], requested.event_id)
                    ack = client.handle_server_event(replace(requested, session_id=session_id), key_counter=0, both_keys_up=True)
                    self.assertEqual(ack[0].payload["status"], "duplicate" if restarting else "applied")
                    tools.acknowledge_action_extension(requested.event_id)
                    if restarting:
                        with self.assertRaises(OrchestratorError):
                            tools.decide_action_transport(**old_decision)
                        decision = client.key_down("F11", key_counter=1, capture_idle=True, autorepeat=False)
                        self.assertIsNotNone(decision)
                        outcome = tools.decide_action_transport(decision_event_id=decision.event_id,
                            semantic_sha256=decision.semantic_sha256, session_id=session_id, now=NOW, **decision.payload)
                        self.assertEqual(outcome["decision"], "deny")
            self.assertIsNone(tools.reset_action_extension_replay(str(uuid.uuid4()), now=NOW))
            self.assertNotIn("action.approval.requested", {item["event_type"] for item in tools.claim_action_extension_events()})

    def test_gateway_recovery_closes_the_previous_authenticated_owner(self):
        fixture = test_tool_store_v6.ToolStoreV6Tests()
        self.addCleanup(fixture.doCleanups)
        store, _, tools, _, _, decision = fixture.prepared_action()
        service = test_voice_gateway_auth.GuardedService()
        service.store, service.tool_store = store, tools
        gateway = VoiceGateway(service, auth_key=test_voice_gateway_auth.AUTH_KEY, business_admission=lambda: True)
        factory = EventFactory()
        hello = factory.create("client.hello", {"protocol": "jarvis.voice/1", "capabilities": [
            "speech_commits", "speech_done", "output_epochs", "jarvis.action-approval/1", "jarvis.approval-arbiter/1"]}).to_json()
        old_socket = test_voice_gateway_auth.AuthWebSocket(hello)
        socket = test_voice_gateway_auth.AuthWebSocket(hello)
        gateway._action_connections[decision["session_id"]] = old_socket
        with patch("jarvis_orchestrator.tool_store.utc_now", return_value=NOW):
            asyncio.run(gateway.handle(socket))
        self.assertEqual(old_socket.closed, [(4001, "session_superseded")])
        self.assertEqual(socket.closed, [])
        self.assertTrue(service.attached)
        with store._lock:
            owner = store._connection.execute("SELECT owner_session_id FROM approval_arbiter_queue WHERE approval_id=?", (decision["approval_id"],)).fetchone()[0]
        self.assertEqual(owner, factory.session_id)

    def test_superseded_socket_cannot_ack_the_new_owners_replay(self):
        fixture = test_tool_store_v6.ToolStoreV6Tests()
        self.addCleanup(fixture.doCleanups)
        store, _, tools, _, _, decision = fixture.prepared_action()
        request, = tools.claim_action_extension_events()
        tools.acknowledge_action_extension(request["event_id"])
        service = test_voice_gateway_auth.GuardedService()
        service.store, service.tool_store = store, tools
        gateway = VoiceGateway(service, auth_key=test_voice_gateway_auth.AUTH_KEY, business_admission=lambda: True)
        gateway._publish = AsyncMock()
        def hello(session_id):
            return EventFactory(session_id=session_id).create("client.hello", {"protocol": "jarvis.voice/1", "capabilities": [
                "speech_commits", "speech_done", "output_epochs", "jarvis.action-approval/1", "jarvis.approval-arbiter/1"]}).to_json()
        async def exercise():
            attached = asyncio.Event()
            original_attach = service.attach_client
            def attach(*args, **kwargs):
                original_attach(*args, **kwargs)
                attached.set()
            service.attach_client = attach
            class OldSocket(test_voice_gateway_auth.AuthWebSocket):
                def __init__(self):
                    super().__init__(hello(decision["session_id"]))
                    self.frames = asyncio.Queue()
                async def __anext__(self):
                    frame = await self.frames.get()
                    if frame is None:
                        raise StopAsyncIteration
                    return frame
                async def close(self, *, code, reason):
                    await super().close(code=code, reason=reason)
                    if len(self.closed) == 1:
                        frame = ActionApprovalEventFactory(decision["session_id"]).create("action.approval.event_ack", {
                            "acked_event_id": request["event_id"], "status": "duplicate"})
                        self.frames.put_nowait(frame.to_json())
                        self.frames.put_nowait(None)
            old_socket = OldSocket()
            task = asyncio.create_task(gateway.handle(old_socket))
            try:
                await asyncio.wait_for(attached.wait(), 1)
                tools.acknowledge_action_extension(request["event_id"])
                new_socket = test_voice_gateway_auth.AuthWebSocket(hello(str(uuid.uuid4())))
                await gateway.handle(new_socket)
                await asyncio.wait_for(task, 1)
                self.assertEqual(new_socket.closed, [])
                replay, = tools.claim_action_extension_events()
                self.assertEqual(replay["event_id"], request["event_id"])
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        with patch("jarvis_orchestrator.tool_store.utc_now", return_value=NOW):
            asyncio.run(exercise())

    def test_owner_change_and_requested_replay_roll_back_together(self):
        fixture = test_tool_store_v6.ToolStoreV6Tests()
        self.addCleanup(fixture.doCleanups)
        store, _, tools, _, _, _ = fixture.prepared_action()
        row, = tools.claim_action_extension_events()
        tools.acknowledge_action_extension(row["event_id"])
        def snapshot():
            with store._lock:
                return {name: [tuple(row) for row in store._connection.execute("SELECT * FROM " + name)]
                        for name in ("approval_arbiter_queue", "action_approval_presentations", "action_approval_outbox")}
        before = snapshot()
        with store._lock:
            store._connection.execute("""CREATE TRIGGER fail_recovery BEFORE UPDATE ON action_approval_outbox
                WHEN NEW.state='PENDING' BEGIN SELECT RAISE(ABORT,'injected replay failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            tools.reset_action_extension_replay(str(uuid.uuid4()), now=NOW)
        self.assertEqual(snapshot(), before)

    def test_pending_replay_recovers_after_observation_before_local_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = ActionApprovalJournal(Path(directory) / "action.sqlite3")
            self.addCleanup(journal.close)
            requested = test_action_approval_v1.ActionApprovalV1Tests().requested()
            journal.observe(requested)
            client = ActionApprovalController(session_id=requested.session_id, journal=journal, approval_arbiter=GlobalApprovalController())
            client.handle_server_event(requested, key_counter=10, both_keys_up=True)
            self.assertIsNone(client.key_down("F10", key_counter=10, capture_idle=True, autorepeat=False))
            self.assertIsNotNone(client.key_down("F11", key_counter=11, capture_idle=True, autorepeat=False))

    def test_replay_does_not_rearm_decided_or_invalidated_requests(self):
        for state in ("PENDING", "ACKED", "RESOLVED", "INVALIDATED"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                journal = ActionApprovalJournal(Path(directory) / "action.sqlite3")
                self.addCleanup(journal.close)
                server = ActionApprovalEventFactory(str(uuid.uuid4()))
                requested = server.create("action.approval.requested", test_action_approval_v1.ActionApprovalV1Tests().requested().payload)
                arbiter = GlobalApprovalController()
                client = ActionApprovalController(session_id=server.session_id, journal=journal, approval_arbiter=arbiter)
                client.handle_server_event(requested, key_counter=0, both_keys_up=True)
                if state == "INVALIDATED":
                    terminal = server.create("action.approval.invalidated", {
                        **{k: requested.payload[k] for k in ("approval_id", "presentation_id", "approval_binding_sha256")},
                        "reason_code": "task_cancelled"})
                    journal.observe(terminal)
                else:
                    decision = client.key_down("F11", key_counter=1, capture_idle=True, autorepeat=False)
                    if state == "ACKED":
                        journal.mark_ack(decision.event_id)
                    elif state == "RESOLVED":
                        journal.mark_result(decision.event_id, "denied")
                restarted = ActionApprovalController(session_id=server.session_id, journal=journal, approval_arbiter=arbiter)
                restarted.handle_server_event(requested, key_counter=2, both_keys_up=True)
                self.assertIsNone(restarted.requested)
                self.assertIsNone(restarted.key_down("F10", key_counter=3, capture_idle=True, autorepeat=False))
                if state == "INVALIDATED":
                    restarted.handle_server_event(terminal, key_counter=2, both_keys_up=True)
                    self.assertIsNone(arbiter.presentation)

    def test_expired_request_is_not_replayed_or_reassigned(self):
        fixture = test_tool_store_v6.ToolStoreV6Tests()
        self.addCleanup(fixture.doCleanups)
        _, _, tools, _, _, decision = fixture.prepared_action()
        row, = tools.claim_action_extension_events()
        tools.acknowledge_action_extension(row["event_id"])
        self.assertIsNone(tools.reset_action_extension_replay(str(uuid.uuid4()), now=NOW + timedelta(seconds=3)))
        self.assertEqual(tools.claim_action_extension_events(), [])

    def test_cloud_request_survives_arrival_while_approval_key_is_held(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = CloudApprovalJournal(Path(directory) / "cloud.sqlite3")
            self.addCleanup(journal.close)
            client = CloudApprovalController(session_id=str(uuid.uuid4()), journal=journal)
            requested = test_m1c_cloud_contract.ExtensionProtocolTests().requested(CloudApprovalEventFactory(client.factory.session_id))
            client.handle_server_event(requested, key_counter=3, both_keys_up=False)
            self.assertIsNone(client.key_down("F10", key_counter=4, capture_idle=True, autorepeat=False))
            self.assertFalse(client.rearm_after_capture(key_counter=4, both_keys_up=False))
            self.assertTrue(client.rearm_after_capture(key_counter=5, both_keys_up=True))
            self.assertIsNone(client.key_down("F10", key_counter=5, capture_idle=True, autorepeat=False))
            self.assertIsNotNone(client.key_down("F11", key_counter=6, capture_idle=True, autorepeat=False))


if __name__ == "__main__":
    unittest.main()
