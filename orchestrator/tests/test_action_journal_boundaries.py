from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import test_action_approval_v1
from jarvis_voice.action_approval import ActionApprovalController, ActionApprovalJournal
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.errors import ProtocolError


class ActionJournalBoundaryTests(unittest.TestCase):
    def test_duplicate_write_never_returns_an_unpersisted_decision(self):
        with tempfile.TemporaryDirectory() as directory, closing(ActionApprovalJournal(Path(directory) / "action.sqlite3")) as journal:
            requested = test_action_approval_v1.ActionApprovalV1Tests().requested()
            first = journal.persist_decision(requested, "deny")
            before = journal.pending_decisions()
            for decision in ("deny", "approve"):
                with self.subTest(decision=decision):
                    with self.assertRaises(sqlite3.IntegrityError):
                        journal.persist_decision(requested, decision)
                    self.assertEqual(journal.pending_decisions(), before)
            row, = journal.pending_decisions()
            self.assertEqual(row["decision_event_id"], first.event_id)
            self.assertEqual(row["decision"], "deny")

    def test_sequence_history_does_not_retain_complete_wire_metadata(self):
        with tempfile.TemporaryDirectory() as directory, closing(ActionApprovalJournal(Path(directory) / "action.sqlite3")) as journal:
            requested = test_action_approval_v1.ActionApprovalV1Tests().requested()
            marker = "private-metadata-" * 100
            requested = replace(requested, payload=dict(requested.payload, tool_id=marker))
            client = ActionApprovalController(session_id=requested.session_id, journal=journal,
                                              approval_arbiter=GlobalApprovalController())
            client.handle_server_event(requested, key_counter=0, both_keys_up=True)
            self.assertEqual(client.handle_server_event(requested, key_counter=0, both_keys_up=True), [])
            changed = replace(requested, payload=dict(requested.payload, tool_id="changed"))
            with self.assertRaises(ProtocolError):
                client.handle_server_event(changed, key_counter=0, both_keys_up=True)
            self.assertNotIn(marker, repr(client._seen))
            self.assertLessEqual(sum(len(value) for value in client._seen.values()), 64)
            client.begin_new_connection()
            self.assertEqual(client._seen, {})


if __name__ == "__main__":
    unittest.main()
