from __future__ import annotations

import sqlite3
import tempfile
import unittest
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
import test_action_approval_v1
import test_tool_store_v6
from helpers import NOW
from jarvis_voice.action_approval import ActionApprovalController, ActionApprovalEventFactory, ActionApprovalJournal
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.errors import ProtocolError


class ActionDecisionRetryTests(unittest.TestCase):
    def test_retry_and_restart_keep_the_server_decision_identity(self):
        fixture = test_tool_store_v6.ToolStoreV6Tests()
        self.addCleanup(fixture.doCleanups)
        _, _, tools, _, _, arguments = fixture.prepared_action()
        server_event = tools.claim_action_extension_events()[0]
        server = ActionApprovalEventFactory(arguments["session_id"])
        requested = server.create(server_event["event_type"], server_event["payload"],
                                  event_id=server_event["event_id"], sent_at=server_event["original_sent_at"])
        def receive(event):
            return tools.decide_action_transport(
                decision_event_id=event.event_id, semantic_sha256=event.semantic_sha256,
                session_id=event.session_id, now=NOW, **event.payload,
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.sqlite3"
            with closing(ActionApprovalJournal(path)) as journal:
                client = ActionApprovalController(session_id=server.session_id, journal=journal,
                                                  approval_arbiter=GlobalApprovalController())
                client.handle_server_event(requested, key_counter=0, both_keys_up=True)
                with patch("jarvis_voice.action_approval.datetime", wraps=datetime) as clock:
                    clock.now.return_value = NOW
                    first = client.key_down("F11", key_counter=1, capture_idle=True, autorepeat=False)
                self.assertEqual(receive(first)["decision"], "deny")
                journal.mark_sent(first.event_id, now=100.0)
                self.assertEqual(client.due_decision_retransmissions(now=100.5), [])
                retry, = client.due_decision_retransmissions(now=101.0)
                self.assertEqual(receive(retry), {"duplicate": True})
                self.assertEqual((retry.event_id, retry.sent_at, retry.semantic_sha256),
                                 (first.event_id, first.sent_at, first.semantic_sha256))
                self.assertGreater(retry.seq, first.seq)
            with closing(ActionApprovalJournal(path)) as journal:
                client = ActionApprovalController(session_id=server.session_id, journal=journal,
                                                  approval_arbiter=GlobalApprovalController())
                retry, = client.due_decision_retransmissions(now=102.0)
                self.assertEqual(receive(retry), {"duplicate": True})
                self.assertEqual(retry.sent_at, first.sent_at)
                journal.mark_ack(retry.event_id)
                self.assertEqual(client.due_decision_retransmissions(now=103.0), [])

    def test_reboot_does_not_reuse_a_previous_monotonic_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.sqlite3"
            with closing(ActionApprovalJournal(path)) as journal:
                first = journal.persist_decision(test_action_approval_v1.ActionApprovalV1Tests().requested(), "deny")
                journal.mark_sent(first.event_id, now=100000.0)
            with closing(ActionApprovalJournal(path)) as journal:
                client = ActionApprovalController(session_id=str(uuid.uuid4()), journal=journal,
                                                  approval_arbiter=GlobalApprovalController())
                retry, = client.due_decision_retransmissions(now=1.0)
                self.assertEqual(retry.semantic_sha256, first.semantic_sha256)

    def test_invalidation_stops_only_the_matching_decision_retry(self):
        for acked in (False, True):
            with self.subTest(acked=acked), tempfile.TemporaryDirectory() as directory, closing(ActionApprovalJournal(Path(directory) / "action.sqlite3")) as journal:
                server = ActionApprovalEventFactory(str(uuid.uuid4()))
                requested = server.create("action.approval.requested", test_action_approval_v1.ActionApprovalV1Tests().requested().payload)
                client = ActionApprovalController(session_id=server.session_id, journal=journal,
                                                  approval_arbiter=GlobalApprovalController())
                client.handle_server_event(requested, key_counter=0, both_keys_up=True)
                decision = client.key_down("F11", key_counter=1, capture_idle=True, autorepeat=False)
                if acked:
                    journal.mark_ack(decision.event_id)
                payload = {name: requested.payload[name] for name in ("approval_id", "presentation_id", "approval_binding_sha256")}
                payload["reason_code"] = "task_cancelled"
                unrelated = server.create("action.approval.invalidated", dict(payload, approval_id=str(uuid.uuid4())))
                client.handle_server_event(unrelated, key_counter=1, both_keys_up=True)
                self.assertIsNotNone(client.requested)
                self.assertEqual(len(journal.pending_decisions()), 0 if acked else 1)
                terminal = server.create("action.approval.invalidated", payload)
                client.handle_server_event(terminal, key_counter=1, both_keys_up=True)
                self.assertEqual(client.due_decision_retransmissions(now=100.0), [])
                with journal._lock:
                    row = journal._connection.execute("SELECT state,result_code FROM action_decisions WHERE decision_event_id=?", (decision.event_id,)).fetchone()
                self.assertEqual(tuple(row), ("INVALIDATED", "task_cancelled"))

    def test_legacy_pending_decisions_are_retained_without_inventing_a_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.sqlite3"
            with closing(ActionApprovalJournal(path)) as journal:
                first = journal.persist_decision(test_action_approval_v1.ActionApprovalV1Tests().requested(), "deny")
            with closing(sqlite3.connect(path, isolation_level=None)) as connection:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(action_decisions)")}
                if "sent_at" in columns:
                    connection.execute("ALTER TABLE action_decisions DROP COLUMN sent_at")
            for _ in range(2):
                with closing(ActionApprovalJournal(path)) as journal:
                    client = ActionApprovalController(session_id=str(uuid.uuid4()), journal=journal,
                                                      approval_arbiter=GlobalApprovalController())
                    with self.assertRaises(ProtocolError) as raised:
                        client.due_decision_retransmissions(now=1.0)
                    self.assertEqual(raised.exception.code, "action_journal.replay_identity_missing")
                    pending, = journal.pending_decisions()
                    self.assertEqual((pending["decision_event_id"], pending["decision"], pending["state"]),
                                     (first.event_id, "deny", "PENDING"))
                    self.assertEqual(client.factory.next_seq, 1)
            with closing(ActionApprovalJournal(path)) as journal:
                journal.mark_ack(first.event_id)
            with closing(ActionApprovalJournal(path)) as journal:
                client = ActionApprovalController(session_id=str(uuid.uuid4()), journal=journal,
                                                  approval_arbiter=GlobalApprovalController())
                self.assertEqual(client.due_decision_retransmissions(now=1.0), [])


if __name__ == "__main__":
    unittest.main()
