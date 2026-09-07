from __future__ import annotations

import sqlite3
import unittest
import uuid
from datetime import timedelta

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.domain import OrchestratorError, TaskState, parse_utc, utc_text
from jarvis_orchestrator.scheduler_contract import PackagedTimezoneProvider, compile_proposal
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.scheduler_store import SchedulerStore
from test_scheduler_v5 import proposal_input, proposal_output


class SchedulerApprovalExpiryTests(StoreCase, unittest.TestCase):
    def pending(self, *, minutes):
        store = self.make_store()
        scheduler = SchedulerStore(store, cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
                                   zones=PackagedTimezoneProvider())
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        request_id = str(uuid.uuid4())
        compiled = compile_proposal(
            proposal_input(request_id, utc_text(NOW)), proposal_output(request_id, minutes=minutes),
            compiler_qualification_sha256="1" * 64, zones=scheduler.zones,
        )
        proposal = scheduler.create_proposal(compiled, task_id=task.task_id, epoch=task.epoch, now=NOW)
        presented = scheduler.promote_next_approval(str(uuid.uuid4()), supported_protocols=("jarvis.scheduler/1",), now=NOW)
        return store, scheduler, task, proposal, presented

    def test_all_expiry_paths_cancel_linked_task_and_revoke_presentation(self):
        for boundary in ("ttl", "minimum_lead"):
            for path in ("sweep", "promote", "decision"):
                with self.subTest(boundary=boundary, path=path):
                    store, scheduler, task, proposal, presented = self.pending(minutes=20 if boundary == "ttl" else 1)
                    selected = parse_utc(proposal["expires_at_utc"]) if boundary == "ttl" else NOW + timedelta(seconds=51)
                    if path == "sweep":
                        self.assertEqual(scheduler.expire_approvals(now=selected), 1)
                    elif path == "promote":
                        self.assertIsNone(scheduler.promote_next_approval(
                            str(uuid.uuid4()), supported_protocols=("jarvis.scheduler/1",), now=selected,
                        ))
                    else:
                        with self.assertRaisesRegex(OrchestratorError, "approval_expired|minimum_lead_violated"):
                            scheduler.decide_approval(
                                approval_id=proposal["approval_id"], proposal_id=proposal["proposal_id"],
                                proposal_sha256=proposal["proposal_sha256"], decision="approve",
                                cause_event_id=str(uuid.uuid4()), now=selected,
                            )
                    self.assertEqual(store.get_task(task.task_id).state, TaskState.CANCELLED)
                    self.assertEqual(scheduler.task_run(task.task_id, task.epoch)["state"], "EXPIRED")
                    self.assertEqual(store.active_conversation()["epoch"], task.epoch + 1)
                    self.assertEqual(store._connection.execute(
                        "SELECT state FROM scheduler_outbox WHERE event_id=?", (presented["event_id"],),
                    ).fetchone()[0], "REVOKED")
                    self.assertEqual(store._connection.execute(
                        "SELECT state FROM approval_arbiter_queue WHERE approval_id=?", (proposal["approval_id"],),
                    ).fetchone()[0], "EXPIRED")
                    events = store.claim_outbox(task.conversation_id, "client", now=selected)
                    cancelled = [event for event in events if event.event_type == "response.cancelled"]
                    self.assertEqual(len(cancelled), 1)
                    self.assertEqual(cancelled[0].payload["reason_code"], "approval_expired")
                    self.assertEqual(scheduler.expire_approvals(now=selected), 0)
                    self.assertEqual(scheduler.list_jobs(), [])

    def test_failed_task_cancellation_rolls_back_entire_expiry(self):
        store, scheduler, task, proposal, presented = self.pending(minutes=20)
        store._connection.execute("""CREATE TRIGGER reject_cancellation BEFORE UPDATE ON tasks
                                  WHEN NEW.state='CANCELLED' BEGIN SELECT RAISE(ABORT,'test failure'); END""")
        selected = parse_utc(proposal["expires_at_utc"])
        with self.assertRaises(sqlite3.IntegrityError):
            scheduler.expire_approvals(now=selected)
        self.assertEqual(store.get_task(task.task_id).state, TaskState.AWAITING_APPROVAL)
        self.assertEqual(scheduler.task_run(task.task_id, task.epoch)["proposal_state"], "PENDING")
        self.assertEqual(store._connection.execute(
            "SELECT state FROM approval_arbiter_queue WHERE approval_id=?", (proposal["approval_id"],),
        ).fetchone()[0], "PRESENTING")
        self.assertEqual(store._connection.execute(
            "SELECT state FROM scheduler_outbox WHERE event_id=?", (presented["event_id"],),
        ).fetchone()[0], "PENDING")
        store._connection.execute("DROP TRIGGER reject_cancellation")
        self.assertEqual(scheduler.expire_approvals(now=selected), 1)
        self.assertEqual(store.get_task(task.task_id).state, TaskState.CANCELLED)


if __name__ == "__main__":
    unittest.main()
