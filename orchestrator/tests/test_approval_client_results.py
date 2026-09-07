from __future__ import annotations

import tempfile
import unittest
import uuid
from contextlib import closing
from pathlib import Path

import _bootstrap  # noqa: F401
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_voice.action_approval import ActionApprovalController, ActionApprovalEventFactory, ActionApprovalJournal
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.scheduler_client import SchedulerClientController
from jarvis_voice.scheduler_journal import SchedulerClientJournal
from jarvis_voice.scheduler_protocol import SchedulerEventFactory
import test_action_approval_v1


class ApprovalClientResultTests(unittest.TestCase):
    def test_scheduler_replayed_result_preserves_the_new_approval(self):
        with tempfile.TemporaryDirectory() as directory, closing(SchedulerClientJournal(
            Path(directory) / "scheduler.sqlite3", cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
        )) as journal:
            server = SchedulerEventFactory(str(uuid.uuid4()))
            arbiter = GlobalApprovalController()
            client = SchedulerClientController(session_id=server.session_id, journal=journal, approval_arbiter=arbiter)
            def request():
                return server.create("scheduler.approval.requested", {
                    "approval_id": str(uuid.uuid4()), "proposal_id": str(uuid.uuid4()),
                    "proposal_sha256": "a" * 64, "operation": "CREATE", "job_id": None,
                    "expected_job_revision": None, "occurrence_id": None, "preview": {},
                    "expires_at_utc": "2030-01-01T00:00:00.000Z", "decision_nonce": str(uuid.uuid4()),
                    "presentation_id": str(uuid.uuid4()), "approve_key": "F10", "deny_key": "F11",
                })
            first = request()
            client.handle_server_event(first, key_counter=0, both_keys_up=True)
            decision = client.key_down("F10", key_counter=1, capture_idle=True, autorepeat=False)
            result = server.create("scheduler.approval.decision_result", {
                "decision_event_id": decision.event_id, "approval_id": first.payload["approval_id"],
                "proposal_id": first.payload["proposal_id"], "proposal_sha256": "a" * 64,
                "result": "APPLIED", "reason_code": "approved", "job_id": None,
                "job_revision": None, "occurrence_id": None, "effective_due_at_utc": None,
            })
            client.handle_server_event(result, key_counter=1, both_keys_up=True)
            second = request()
            client.handle_server_event(second, key_counter=2, both_keys_up=True)
            replay = server.create(result.type, result.payload, event_id=result.event_id, sent_at=result.sent_at)
            ack = client.handle_server_event(replay, key_counter=2, both_keys_up=True)
            self.assertEqual(ack[0].payload["status"], "duplicate")
            self.assertEqual(arbiter.presentation.approval_id, second.payload["approval_id"])
            decision = client.key_down("F10", key_counter=3, capture_idle=True, autorepeat=False)
            self.assertIsNotNone(decision)
            self.assertEqual(decision.payload["approval_id"], second.payload["approval_id"])

    def test_late_action_terminal_events_preserve_the_new_approval(self):
        for late_kind in ("action.approval.invalidated", "action.approval.decision_result"):
            with self.subTest(late_kind=late_kind), tempfile.TemporaryDirectory() as directory, closing(ActionApprovalJournal(Path(directory) / "action.sqlite3")) as journal:
                server = ActionApprovalEventFactory(str(uuid.uuid4()))
                arbiter = GlobalApprovalController()
                client = ActionApprovalController(session_id=server.session_id, journal=journal, approval_arbiter=arbiter)
                def request():
                    return server.create("action.approval.requested", test_action_approval_v1.ActionApprovalV1Tests().requested().payload)
                def terminal(requested, decision, kind):
                    payload = {name: requested.payload[name] for name in ("approval_id", "presentation_id", "approval_binding_sha256")}
                    payload["reason_code"] = "approval_expired"
                    if kind == "action.approval.decision_result":
                        payload.update(decision_event_id=decision.event_id, result="rejected")
                    return server.create(kind, payload)
                first = request()
                client.handle_server_event(first, key_counter=0, both_keys_up=True)
                first_decision = client.key_down("F10", key_counter=1, capture_idle=True, autorepeat=False)
                initial_kind = "action.approval.decision_result" if late_kind.endswith("invalidated") else "action.approval.invalidated"
                client.handle_server_event(terminal(first, first_decision, initial_kind), key_counter=1, both_keys_up=True)
                second = request()
                client.handle_server_event(second, key_counter=2, both_keys_up=True)
                ack = client.handle_server_event(terminal(first, first_decision, late_kind), key_counter=2, both_keys_up=True)
                self.assertEqual(ack[0].payload["status"], "applied")
                self.assertEqual(arbiter.presentation.approval_id, second.payload["approval_id"])
                decision = client.key_down("F10", key_counter=3, capture_idle=True, autorepeat=False)
                self.assertIsNotNone(decision)
                self.assertEqual(decision.payload["approval_id"], second.payload["approval_id"])
                client.handle_server_event(terminal(second, decision, "action.approval.decision_result"), key_counter=3, both_keys_up=True)
                self.assertIsNone(client.requested)
                self.assertIsNone(arbiter.presentation)


if __name__ == "__main__":
    unittest.main()
