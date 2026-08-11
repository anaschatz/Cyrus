from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.action_approval import (
    ACTION_APPROVAL_PROTOCOL,
    ActionApprovalEnvelope,
    ActionApprovalEventFactory,
    ActionApprovalJournal,
)


class ActionApprovalV1Tests(unittest.TestCase):
    def requested(self) -> ActionApprovalEnvelope:
        factory = ActionApprovalEventFactory(str(uuid.uuid4()))
        return factory.create(
            "action.approval.requested",
            {
                "approval_id": str(uuid.uuid4()),
                "presentation_id": str(uuid.uuid4()),
                "task_id": str(uuid.uuid4()),
                "tool_call_id": str(uuid.uuid4()),
                "epoch": 3,
                "tool_id": "email.draft@1",
                "tool_version": 1,
                "approval_binding_sha256": "a" * 64,
                "preview_sha256": "b" * 64,
                "arguments_sha256": "c" * 64,
                "user_intent_binding_sha256": "d" * 64,
                "registry_sha256": "e" * 64,
                "tool_schema_sha256": "f" * 64,
                "policy_sha256": "0" * 64,
                "expires_at": "2026-08-11T16:00:00.000Z",
                "decision_nonce": str(uuid.uuid4()),
                "preview_fetch_nonce": str(uuid.uuid4()),
                "approve_key": "F10",
                "deny_key": "F11",
            },
        )

    def test_wire_contains_only_ids_hashes_and_bounded_metadata(self) -> None:
        event = self.requested()
        serialized = event.to_json()
        for secret in ("recipient@example.com", "calendar body", "note body", "/Users/private"):
            self.assertNotIn(secret, serialized)
        self.assertEqual(ActionApprovalEnvelope.from_json(serialized), event)

    def test_unknown_requested_field_is_rejected(self) -> None:
        value = self.requested().to_dict()
        value["payload"]["body"] = "secret"
        with self.assertRaises(Exception):
            ActionApprovalEnvelope.from_json(__import__("json").dumps(value, separators=(",", ":")))

    def test_decision_is_persisted_before_send_and_duplicate_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = ActionApprovalJournal(Path(directory) / "journal.sqlite3")
            requested = self.requested()
            self.assertTrue(journal.observe(requested))
            self.assertFalse(journal.observe(requested))
            decision = journal.persist_decision(requested, "approve")
            self.assertEqual(decision.protocol, ACTION_APPROVAL_PROTOCOL)
            self.assertEqual(decision.payload["decision"], "approve")
            with journal._lock:
                row = journal._connection.execute("SELECT state FROM action_decisions WHERE decision_event_id=?", (decision.event_id,)).fetchone()
            self.assertEqual(row["state"], "PENDING")


if __name__ == "__main__":
    unittest.main()
