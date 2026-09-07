from __future__ import annotations

import asyncio
import sqlite3
import unittest
import uuid
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.domain import ReasonCode, TaskState
from jarvis_orchestrator.gateway import VoiceGateway
from jarvis_orchestrator.tool_contracts import ToolRegistryV1
from jarvis_orchestrator.tool_crypto import InMemoryToolKeyProvider, ToolPayloadCipherV1
from jarvis_orchestrator.tool_store import ToolStoreV1
from jarvis_voice.action_approval import ActionApprovalEventFactory
from jarvis_voice.protocol import EventFactory
from test_voice_gateway_auth import AUTH_KEY


class ActionPublicationTests(StoreCase, unittest.IsolatedAsyncioTestCase):
    def pending(self):
        store = self.make_store(tool_authority_enabled=True)
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        tools = ToolStoreV1(store, cipher=ToolPayloadCipherV1(InMemoryToolKeyProvider()),
                            registry=ToolRegistryV1.build(root_ids=("documents",), app_ids=("textedit",)))
        created = tools.create_tool_run(
            task_id=task.task_id, epoch=task.epoch, tool_call_id=str(uuid.uuid4()),
            tool_id="note.capture@1", arguments={"title": "Title", "body": "Body"},
            preview={"title": "Title", "body": "Body"}, idempotency_key=str(uuid.uuid4()),
            policy_sha256="a" * 64, scheduler_contract_sha256="b" * 64,
            user_intent_binding_sha256="c" * 64, now=NOW,
        )
        return store, tools, task, created

    async def test_cancellation_retires_action_presentation_and_replay_atomically(self):
        for sent in (None, False, True):
            with self.subTest(sent=sent):
                store, tools, task, created = self.pending()
                if sent is not None:
                    presented = tools.promote_next_action_approval(str(uuid.uuid4()), now=NOW)
                if sent:
                    tools.finish_action_extension_write(presented["event_id"], complete=True)
                table = "action_approval_presentations" if sent is None else "action_approval_outbox"
                store._connection.execute(f"""CREATE TRIGGER reject_retirement BEFORE UPDATE ON {table}
                                          WHEN NEW.state='INVALIDATED' BEGIN SELECT RAISE(ABORT,'test failure'); END""")
                with self.assertRaises(sqlite3.IntegrityError):
                    store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
                self.assertEqual(store.get_task(task.task_id).state, TaskState.AWAITING_APPROVAL)
                self.assertEqual(store.active_conversation()["epoch"], task.epoch)
                store._connection.execute("DROP TRIGGER reject_retirement")
                store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
                self.assertEqual(tools.claim_action_extension_events(), [])
                tables = ("action_approval_presentations",) if sent is None else ("action_approval_presentations", "action_approval_outbox")
                for table in tables:
                    self.assertEqual(store._connection.execute(
                        f"SELECT state FROM {table} WHERE approval_id=?", (created["approval_id"],),
                    ).fetchone()[0], "INVALIDATED")

    async def test_decision_receipts_survive_task_cancellation_until_acknowledged(self):
        store, tools, task, created = self.pending()
        session_id = str(uuid.uuid4())
        presented = tools.promote_next_action_approval(session_id, now=NOW)
        event = ActionApprovalEventFactory(session_id).create("action.approval.decision", {
            "approval_id": created["approval_id"], "presentation_id": created["presentation_id"],
            "requested_event_id": presented["event_id"],
            "approval_binding_sha256": created["approval_binding_sha256"],
            "decision_nonce": presented["payload"]["decision_nonce"], "decision": "approve",
        })
        tools.decide_action_transport(decision_event_id=event.event_id, semantic_sha256=event.semantic_sha256,
                                      session_id=session_id, now=NOW, **event.payload)
        store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
        receipts = tools.claim_action_extension_events()
        self.assertEqual({item["event_type"] for item in receipts},
                         {"action.approval.decision_ack", "action.approval.decision_result"})
        with patch("jarvis_orchestrator.store.utc_now", return_value=NOW + timedelta(minutes=11)):
            for receipt in receipts:
                self.assertTrue(store.action_extension_is_current(receipt["event_id"]))
                tools.acknowledge_action_extension(receipt["event_id"])
                self.assertFalse(store.action_extension_is_current(receipt["event_id"]))
        self.assertEqual(tools.claim_action_extension_events(), [])

    async def test_publisher_rechecks_action_approval_after_waiting_for_lock(self):
        for state in ("live", "cancelled", "decided", "expired"):
            with self.subTest(state=state):
                store, tools, task, created = self.pending()
                session_id = str(uuid.uuid4())
                gateway = VoiceGateway(SimpleNamespace(store=store, tool_store=tools), auth_key=AUTH_KEY)
                socket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
                factory = ActionApprovalEventFactory(session_id)
                claimed, processed = asyncio.Event(), asyncio.Event()
                original_claim = tools.claim_action_extension_events

                def claim():
                    if claimed.is_set():
                        processed.set()
                        return []
                    events = original_claim()
                    self.assertEqual(len(events), 1)
                    claimed.set()
                    return events

                lock = gateway._lock(task.conversation_id)
                await lock.acquire()
                with patch("jarvis_orchestrator.tool_store.utc_now", return_value=NOW), \
                        patch("jarvis_orchestrator.store.utc_now", return_value=NOW) as clock, \
                        patch.object(tools, "claim_action_extension_events", side_effect=claim):
                    writer = asyncio.create_task(gateway._publish(
                        socket, session_id, EventFactory(), "connection", action_factory=factory,
                    ))
                    try:
                        await asyncio.wait_for(claimed.wait(), 0.5)
                        if state == "cancelled":
                            store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
                        elif state == "decided":
                            tools.decide_action_approval(approval_id=created["approval_id"], epoch=task.epoch,
                                                         decision="deny", now=NOW)
                        elif state == "expired":
                            clock.return_value = NOW + timedelta(minutes=10)
                        lock.release()
                        await asyncio.wait_for(processed.wait(), 0.5)
                        requests = [call for call in socket.send.call_args_list
                                    if '"action.approval.requested"' in call.args[0]]
                        self.assertEqual(len(requests), int(state == "live"))
                        self.assertEqual(factory.next_seq, 2 if state == "live" else 1)
                        socket.close.assert_not_awaited()
                    finally:
                        if lock.locked():
                            lock.release()
                        writer.cancel()
                        await asyncio.gather(writer, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
