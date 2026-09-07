from __future__ import annotations

import unittest
import uuid
from datetime import timedelta

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.domain import OrchestratorError, parse_utc, utc_text
from jarvis_orchestrator.scheduler_contract import PackagedTimezoneProvider, compile_proposal, notification_id
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.scheduler_store import SchedulerStore
from jarvis_voice.scheduler_protocol import SchedulerEventFactory
from test_scheduler_v5 import proposal_input, proposal_output


class SchedulerReceiptBindingTests(StoreCase, unittest.TestCase):
    def setUp(self):
        self.store = self.make_store()
        self.scheduler = SchedulerStore(
            self.store, cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            zones=PackagedTimezoneProvider(),
        )
        self.factory = SchedulerEventFactory(str(uuid.uuid4()))
        self.due = NOW + timedelta(minutes=20)

    def delivery(self):
        request_id = str(uuid.uuid4())
        output = proposal_output(request_id)
        output["notification"]["speakable_text"] = "Ώρα για διάλειμμα."
        compiled = compile_proposal(
            proposal_input(request_id, utc_text(NOW)), output,
            compiler_qualification_sha256="1" * 64, zones=self.scheduler.zones,
        )
        proposal = self.scheduler.create_proposal(compiled, now=NOW)
        approved = self.scheduler.decide_approval(
            approval_id=proposal["approval_id"], proposal_id=proposal["proposal_id"],
            proposal_sha256=proposal["proposal_sha256"], decision="approve",
            cause_event_id=str(uuid.uuid4()), now=NOW,
        )
        self.scheduler.fire_due(now=self.due)
        return notification_id(approved["occurrence_id"], 0)

    def receipt(self, notification, attempt, stage):
        event = self.factory.create("scheduler.notification.receipt", {
            "notification_id": notification, "attempt_id": attempt, "stage": stage,
            "reason_code": "test_receipt", "observed_at_utc": utc_text(self.due),
        })
        return self.scheduler.record_delivery_receipt(
            event_id=event.event_id, semantic_sha256=event.semantic_sha256,
            notification=notification, attempt_id=attempt, stage=stage,
            reason_code=event.payload["reason_code"], observed_at_utc=event.payload["observed_at_utc"],
        )

    def test_attempt_cannot_be_rebound_to_another_notification(self):
        first, second = self.delivery(), self.delivery()
        attempt = str(uuid.uuid4())
        self.receipt(first, attempt, "DISPATCHING")
        tables = ("scheduler_inbox", "scheduler_delivery_attempts", "scheduler_deliveries",
                  "scheduler_occurrences", "scheduler_jobs", "scheduler_ledger")
        before = {table: [tuple(row) for row in self.store._connection.execute(f"SELECT * FROM {table}")]
                  for table in tables}
        with self.assertRaisesRegex(OrchestratorError, "binding_mismatch"):
            self.receipt(second, attempt, "OS_ACCEPTED")
        for table in tables:
            self.assertEqual([tuple(row) for row in self.store._connection.execute(f"SELECT * FROM {table}")],
                             before[table])
        self.assertTrue(self.receipt(first, attempt, "OS_ACCEPTED"))
        self.assertTrue(self.receipt(second, str(uuid.uuid4()), "OS_ACCEPTED"))

    def test_only_live_ack_can_create_notification_speech(self):
        for phase in ("live", "revoked", "expired", "deadline"):
            with self.subTest(phase=phase):
                notification = self.delivery()
                requested = next(row for row in self.scheduler.claim_outbox(str(uuid.uuid4()), now=self.due)
                                 if row["event_type"] == "scheduler.notification.requested"
                                 and row["payload"]["notification_id"] == notification)
                selected = self.due
                if phase == "revoked":
                    self.scheduler.cancel_job(requested["payload"]["job_id"], 1,
                                              cause_event_id=str(uuid.uuid4()), now=selected)
                elif phase in {"expired", "deadline"}:
                    selected = parse_utc(requested["payload"]["expires_at_utc"])
                    if phase == "expired":
                        self.scheduler.claim_outbox(str(uuid.uuid4()), now=selected)
                before = self.store._connection.execute(
                    "SELECT state FROM scheduler_outbox WHERE event_id=?", (requested["event_id"],),
                ).fetchone()[0]
                for replay in (False, True):
                    accepted = self.scheduler.acknowledge_scheduler_event(
                        requested["event_id"], status="applied", speech_supported=True, now=selected,
                    )
                    self.assertEqual(accepted, phase == "live" and not replay)
                self.assertEqual(self.store._connection.execute(
                    "SELECT COUNT(*) FROM scheduler_outbox WHERE entity_id=? AND event_type='scheduler.notification.speech'",
                    (notification,),
                ).fetchone()[0], int(phase == "live"))
                self.assertEqual(self.store._connection.execute(
                    "SELECT state FROM scheduler_outbox WHERE event_id=?", (requested["event_id"],),
                ).fetchone()[0], "ACKED" if phase == "live" else before)

    def test_late_progress_cannot_regress_a_finished_or_newer_attempt(self):
        for later_attempt in (False, True):
            for stage in ("STAGED", "DISPATCHING", "AMBIGUOUS_OS_DISPATCH", "OS_REJECTED"):
                with self.subTest(later_attempt=later_attempt, stage=stage):
                    notification = self.delivery()
                    first = str(uuid.uuid4())
                    self.receipt(notification, first, "DISPATCHING")
                    self.receipt(notification, first, "OS_REJECTED")
                    if later_attempt:
                        self.receipt(notification, str(uuid.uuid4()), "DISPATCHING")
                    before = tuple(self.store._connection.execute(
                        "SELECT state,updated_at FROM scheduler_deliveries WHERE notification_id=?", (notification,),
                    ).fetchone())
                    self.assertFalse(self.receipt(notification, first, stage))
                    self.assertEqual(tuple(self.store._connection.execute(
                        "SELECT state,updated_at FROM scheduler_deliveries WHERE notification_id=?", (notification,),
                    ).fetchone()), before)
                    self.assertEqual(self.store._connection.execute(
                        "SELECT state FROM scheduler_delivery_attempts WHERE attempt_id=?", (first,),
                    ).fetchone()[0], "OS_REJECTED")
                    ignored = self.store._connection.execute(
                        "SELECT event_type FROM scheduler_ledger ORDER BY ledger_seq DESC LIMIT 1",
                    ).fetchone()[0]
                    self.assertEqual(ignored, "scheduler.delivery.late_receipt_ignored")

    def test_late_progress_preserves_dispatch_and_recovery_state(self):
        for initial, late in (("DISPATCHING", "STAGED"), ("AMBIGUOUS_OS_DISPATCH", "STAGED"),
                              ("AMBIGUOUS_OS_DISPATCH", "DISPATCHING")):
            with self.subTest(initial=initial, late=late):
                notification, attempt = self.delivery(), str(uuid.uuid4())
                self.receipt(notification, attempt, initial)
                self.assertFalse(self.receipt(notification, attempt, late))
                self.assertEqual(self.store._connection.execute(
                    "SELECT state FROM scheduler_delivery_attempts WHERE attempt_id=?", (attempt,),
                ).fetchone()[0], initial)
                self.assertTrue(self.receipt(notification, attempt, "OS_ACCEPTED"))

    def test_late_ambiguous_attempt_cannot_regress_new_dispatch_but_can_confirm_acceptance(self):
        notification = self.delivery()
        first, second = str(uuid.uuid4()), str(uuid.uuid4())
        self.receipt(notification, first, "DISPATCHING")
        self.scheduler.recover_after_restart(now=self.due)
        self.receipt(notification, second, "DISPATCHING")
        self.assertFalse(self.receipt(notification, first, "AMBIGUOUS_OS_DISPATCH"))
        self.assertEqual(self.store._connection.execute(
            "SELECT state FROM scheduler_deliveries WHERE notification_id=?", (notification,),
        ).fetchone()[0], "DISPATCHING")
        self.assertTrue(self.receipt(notification, first, "OS_ACCEPTED"))
        self.assertFalse(self.receipt(notification, second, "OS_REJECTED"))
        self.assertEqual(self.store._connection.execute(
            "SELECT state FROM scheduler_deliveries WHERE notification_id=?", (notification,),
        ).fetchone()[0], "OS_ACCEPTED")

    def test_staged_then_dispatching_attempt_is_recovered_after_restart(self):
        notification = self.delivery()
        attempt = str(uuid.uuid4())
        self.receipt(notification, attempt, "STAGED")
        self.receipt(notification, attempt, "DISPATCHING")
        recovered = self.scheduler.recover_after_restart(now=self.due + timedelta(seconds=1))
        self.assertEqual(recovered["ambiguous_attempt_count"], 1)
        row = self.store._connection.execute(
            "SELECT notification_id,state FROM scheduler_delivery_attempts WHERE attempt_id=?", (attempt,),
        ).fetchone()
        self.assertEqual(tuple(row), (notification, "AMBIGUOUS_OS_DISPATCH"))
        row = self.store._connection.execute(
            "SELECT state FROM scheduler_deliveries WHERE notification_id=?", (notification,),
        ).fetchone()
        self.assertEqual(row[0], "RETRY_WAIT")
        self.assertEqual(self.scheduler.recover_after_restart(now=self.due)["ambiguous_attempt_count"], 0)
        self.assertTrue(self.receipt(notification, attempt, "OS_ACCEPTED"))


if __name__ == "__main__":
    unittest.main()
