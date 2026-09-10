from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_orchestrator.control import ControlApplication
from jarvis_orchestrator.schedule_compiler import CompilerQualification, LocalScheduleCompiler
from jarvis_orchestrator.scheduler_contract import (PackagedTimezoneProvider, ScheduleContractError,
                                                    compile_proposal)
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.scheduler_store import SchedulerStore
from jarvis_orchestrator.store import OrchestratorStore
from jarvis_orchestrator.domain import OrchestratorError
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.scheduler_client import SchedulerClientController
from jarvis_voice.scheduler_callback import sign_callback
from jarvis_voice.scheduler_runtime import SchedulerNativeRuntime
from jarvis_voice.scheduler_journal import SchedulerClientJournal
from jarvis_voice.scheduler_protocol import SchedulerEnvelope, SchedulerEventFactory


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def proposal_input(request_id: str, received: str) -> dict:
    return {
        "schema_version": "schedule-proposal-input/1",
        "request_id": request_id,
        "request_text": "Θύμισέ μου να κάνω διάλειμμα σε είκοσι λεπτά",
        "request_received_at_utc": received,
        "default_timezone": "Europe/Athens",
        "default_locale": "el-GR",
        "selected_target": {"job_id": None, "job_revision": None, "occurrence_id": None, "snooze_generation": None},
        "policy": {"minimum_lead_ms": 10000, "minimum_interval_ms": 60000, "maximum_snooze_generations": 8},
    }


def temporal_after(minutes: int) -> dict:
    return {
        "kind": "AFTER_DURATION", "local_datetime": None, "year": None, "month": None,
        "day": None, "local_time": None, "timezone": None, "amount": minutes,
        "unit": "MINUTE", "relation": None, "weekday": None,
    }


def temporal_none() -> dict:
    return {"kind": "NONE", "local_datetime": None, "year": None, "month": None, "day": None,
            "local_time": None, "timezone": None, "amount": None, "unit": None,
            "relation": None, "weekday": None}


def proposal_output(request_id: str, *, minutes: int = 20, recurrence: str = "NONE", count: int | None = None) -> dict:
    return {
        "schema_version": "schedule-proposal/1", "request_id": request_id, "outcome": "PROPOSAL",
        "operation": "CREATE_JOB", "target_job_id": None, "target_job_revision": None,
        "target_occurrence_id": None, "time_expression": temporal_after(minutes),
        "recurrence": {
            "kind": recurrence,
            "interval_amount": 1 if recurrence == "FIXED_INTERVAL" else None,
            "interval_unit": "MINUTE" if recurrence == "FIXED_INTERVAL" else None,
            "weekdays": [], "day_of_month": None, "count": count, "until": temporal_none(),
        },
        "notification": {"title": "Υπενθύμιση", "body": "Κάνε το προγραμματισμένο διάλειμμα.",
                         "speakable_text": None, "locale": "el-GR", "visibility": "private"},
        "clarification_reason": "NONE", "clarification_question": None,
    }


class SchedulerV5Tests(unittest.TestCase):
    def test_native_action_dedup_survives_transport_changes_and_legacy_upgrade(self) -> None:
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                activated = self.activate()
                payload = {
                    "action": "CANCEL", "job_id": activated["job_id"], "job_revision": 1,
                    "occurrence_id": activated["occurrence_id"], "snooze_generation": 0,
                    "requested_due_at_utc": None,
                }
                first = SchedulerEventFactory(str(uuid.uuid4())).create(
                    "scheduler.notification.action", payload, sent_at="2030-01-10T12:00:00.000Z",
                )

                def apply(event, selected_payload=None):
                    p = selected_payload or event.payload
                    return self.scheduler.apply_notification_action(
                        event_id=event.event_id, semantic_sha256=event.semantic_sha256,
                        action=p["action"], job_id=p["job_id"], job_revision=p["job_revision"],
                        occurrence=p["occurrence_id"], expected_generation=p["snooze_generation"],
                        requested_due_at_utc=p["requested_due_at_utc"], now=self.reference,
                    )

                self.assertTrue(apply(first))
                if legacy:
                    self.base._connection.execute(
                        "UPDATE scheduler_inbox SET content_record_id=NULL WHERE event_id=?", (first.event_id,),
                    )
                    self.assertFalse(apply(first))
                retry = SchedulerEventFactory(str(uuid.uuid4())).create(
                    first.type, payload, event_id=first.event_id, sent_at="2030-01-10T12:00:01.000Z",
                )
                self.assertNotEqual(first.semantic_sha256, retry.semantic_sha256)
                self.assertFalse(apply(retry))
                original_hash = self.base._connection.execute(
                    "SELECT semantic_sha256 FROM scheduler_inbox WHERE event_id=?", (first.event_id,),
                ).fetchone()[0]
                self.assertEqual(original_hash, first.semantic_sha256)
                with self.assertRaisesRegex(OrchestratorError, "protocol.conflicting_duplicate"):
                    apply(first, {**payload, "job_revision": 2})

    def test_legacy_native_action_rejects_unverifiable_transport_change(self) -> None:
        activated = self.activate()
        arguments = dict(
            event_id=str(uuid.uuid4()), action="CANCEL", job_id=activated["job_id"],
            job_revision=1, occurrence=activated["occurrence_id"], expected_generation=0,
            requested_due_at_utc=None, now=self.reference,
        )
        self.scheduler.apply_notification_action(**arguments, semantic_sha256="a" * 64)
        self.base._connection.execute(
            "UPDATE scheduler_inbox SET content_record_id=NULL WHERE event_id=?", (arguments["event_id"],),
        )
        with self.assertRaisesRegex(OrchestratorError, "protocol.conflicting_duplicate"):
            self.scheduler.apply_notification_action(**arguments, semantic_sha256="b" * 64)
        row = self.base._connection.execute(
            "SELECT state,semantic_sha256,content_record_id FROM scheduler_inbox WHERE event_id=?",
            (arguments["event_id"],),
        ).fetchone()
        self.assertEqual(tuple(row), ("QUARANTINED", "a" * 64, None))

    def test_native_action_round_trip_recovers_after_process_restart(self) -> None:
        activated = self.activate()
        now = self.reference + timedelta(minutes=20)
        self.scheduler.fire_due(now=now)
        notification = next(row for row in self.scheduler.claim_outbox(str(uuid.uuid4()), now=now)
                            if row["event_type"] == "scheduler.notification.requested")
        cipher = SchedulerContentCipher(InMemorySchedulerKeyProvider())
        path = Path(self.temporary.name) / "callback.sqlite3"
        original = None
        for attempt in range(3):
            with closing(SchedulerClientJournal(path, cipher=cipher)) as journal:
                controller = SchedulerClientController(
                    session_id=str(uuid.uuid4()), journal=journal,
                    approval_arbiter=GlobalApprovalController(),
                )
                runtime = SchedulerNativeRuntime(
                    controller=controller, delivery=SimpleNamespace(), capabilities=(),
                    manifest_sha256="a" * 64,
                )
                server = SchedulerEventFactory(controller.factory.session_id)
                if attempt == 0:
                    requested = server.create(notification["event_type"], notification["payload"],
                                              event_id=notification["event_id"])
                    controller.handle_server_event(requested, key_counter=0, both_keys_up=True)
                    callback = sign_callback({
                        "callback_id": str(uuid.uuid4()),
                        "notification_id": notification["payload"]["notification_id"],
                        "action": "CANCEL", "requested_due_at_utc": None,
                        "occurred_at_utc": "2030-01-10T12:20:00.000Z",
                    }, key=bytes(range(32)))
                    runtime.ingest_native_callback(callback, key=bytes(range(32)))
                pending = runtime.pump_native_callbacks()
                if attempt == 2:
                    self.assertEqual(pending, [])
                    continue
                self.assertEqual(len(pending), 1)
                event = SchedulerEnvelope.from_json(pending[0].to_json())
                if original is None:
                    original = event
                else:
                    self.assertEqual(event.event_id, original.event_id)
                    self.assertEqual(event.payload, original.payload)
                    self.assertNotEqual(event.semantic_sha256, original.semantic_sha256)
                payload = event.payload
                applied = self.scheduler.apply_notification_action(
                    event_id=event.event_id, semantic_sha256=event.semantic_sha256,
                    action=payload["action"], job_id=payload["job_id"],
                    job_revision=payload["job_revision"], occurrence=payload["occurrence_id"],
                    expected_generation=payload["snooze_generation"],
                    requested_due_at_utc=payload["requested_due_at_utc"], now=now,
                )
                self.assertEqual(applied, attempt == 0)
                if attempt == 1:
                    result = next(row for row in self.scheduler.claim_outbox(str(uuid.uuid4()), now=now)
                                  if row["event_type"] == "scheduler.notification.action_result"
                                  and row["payload"]["result"] == "DUPLICATE")
                    runtime.handle_server_event(
                        server.create(result["event_type"], result["payload"], event_id=result["event_id"]),
                        key_counter=0, both_keys_up=True, send=lambda _event: None,
                    )
        count = self.base._connection.execute(
            "SELECT COUNT(*) FROM scheduler_ledger WHERE entity_id=? AND event_type='scheduler.job.cancelled'",
            (activated["job_id"],),
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_native_action_identity_failure_rolls_back_cancellation(self) -> None:
        activated = self.activate()
        event_id = str(uuid.uuid4())
        original_state = self.base._connection.execute(
            "SELECT state FROM scheduler_jobs WHERE job_id=?", (activated["job_id"],),
        ).fetchone()[0]
        put = self.scheduler._put_content_tx

        def fail_identity(connection, **kwargs):
            if kwargs["record_type"] == "NATIVE_ACTION_IDENTITY":
                raise RuntimeError("identity write failed")
            return put(connection, **kwargs)

        with patch.object(self.scheduler, "_put_content_tx", side_effect=fail_identity):
            with self.assertRaisesRegex(RuntimeError, "identity write failed"):
                self.scheduler.apply_notification_action(
                    event_id=event_id, semantic_sha256="a" * 64, action="CANCEL",
                    job_id=activated["job_id"], job_revision=1,
                    occurrence=activated["occurrence_id"], expected_generation=0,
                    requested_due_at_utc=None, now=self.reference,
                )
        self.assertEqual(self.base._connection.execute(
            "SELECT state FROM scheduler_jobs WHERE job_id=?", (activated["job_id"],),
        ).fetchone()[0], original_state)
        for table, column in (("scheduler_inbox", "event_id"), ("scheduler_outbox", "entity_id"),
                              ("scheduler_ledger", "cause_event_id")):
            self.assertEqual(self.base._connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column}=?", (event_id,),
            ).fetchone()[0], 0)

    def test_real_approval_round_trips_through_voice_controller(self) -> None:
        session_id = str(uuid.uuid4())
        self.scheduler.create_proposal(self.compile(), now=self.reference)
        promoted = self.scheduler.promote_next_approval(
            session_id, supported_protocols=("jarvis.scheduler/1",), now=self.reference
        )
        self.assertIsNotNone(promoted)
        journal = SchedulerClientJournal(
            Path(self.temporary.name) / "voice.sqlite3",
            cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
        )
        self.addCleanup(journal.close)
        controller = SchedulerClientController(
            session_id=session_id, journal=journal, approval_arbiter=GlobalApprovalController(),
        )
        server = SchedulerEventFactory(session_id)
        requested = server.create(
            "scheduler.approval.requested", promoted["payload"],
            event_id=promoted["event_id"], sent_at="2030-01-10T12:00:00.000Z",
        )
        replies = controller.handle_server_event(
            SchedulerEnvelope.from_json(requested.to_json()), key_counter=0, both_keys_up=True,
        )
        self.assertEqual([event.type for event in replies], ["scheduler.event_ack"])
        self.scheduler.acknowledge_scheduler_event(requested.event_id, status=replies[0].payload["status"], now=self.reference)
        decision = controller.key_down("F10", key_counter=1, capture_idle=True, autorepeat=False)
        self.assertIsNotNone(decision)
        result = self.scheduler.decide_approval(
            **{name: decision.payload[name] for name in (
                "approval_id", "proposal_id", "proposal_sha256", "decision", "presentation_id",
                "requested_event_id", "decision_nonce",
            )},
            cause_event_id=decision.event_id, owner_session_id=session_id, now=self.reference,
        )
        self.assertIsNotNone(result["job_id"])
        outbox = self.scheduler.claim_outbox(str(uuid.uuid4()), now=self.reference)
        finished = next(row for row in outbox if row["event_type"] == "scheduler.approval.decision_result")
        response = server.create("scheduler.approval.decision_result", finished["payload"])
        controller.handle_server_event(SchedulerEnvelope.from_json(response.to_json()), key_counter=1, both_keys_up=False)
        self.assertIsNone(controller.approval)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "orchestrator.sqlite3"
        self.base = OrchestratorStore(self.path)
        self.addCleanup(self.base.close)
        self.scheduler = SchedulerStore(
            self.base,
            cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            zones=PackagedTimezoneProvider(),
        )
        self.reference = utc("2030-01-10T12:00:00.000Z")

    def compile(self, *, recurrence: str = "NONE", count: int | None = None, minutes: int = 20):
        request_id = str(uuid.uuid4())
        return compile_proposal(
            proposal_input(request_id, "2030-01-10T12:00:00.000Z"),
            proposal_output(request_id, minutes=minutes, recurrence=recurrence, count=count),
            compiler_qualification_sha256="1" * 64,
            zones=self.scheduler.zones,
        )

    def activate(self, *, minutes: int = 20) -> dict:
        created = self.scheduler.create_proposal(self.compile(minutes=minutes), now=self.reference)
        return self.scheduler.decide_approval(
            approval_id=created["approval_id"], proposal_id=created["proposal_id"],
            proposal_sha256=created["proposal_sha256"], decision="approve",
            cause_event_id=str(uuid.uuid4()), now=self.reference,
        )

    def test_approval_does_not_create_notification_outbox(self) -> None:
        created = self.scheduler.create_proposal(self.compile(), now=self.reference)
        promoted = self.scheduler.promote_next_approval(
            str(uuid.uuid4()), supported_protocols=("jarvis.scheduler/1",), now=self.reference
        )
        self.assertIsNotNone(promoted)
        self.scheduler.acknowledge_scheduler_event(promoted["event_id"], status="applied", now=self.reference)
        result = self.scheduler.decide_approval(
            approval_id=created["approval_id"], proposal_id=created["proposal_id"],
            proposal_sha256=created["proposal_sha256"], decision="approve",
            cause_event_id=str(uuid.uuid4()), now=self.reference,
        )
        types = [row[0] for row in self.base._connection.execute(
            "SELECT event_type FROM scheduler_outbox ORDER BY ordinal"
        ).fetchall()]
        self.assertEqual(types, ["scheduler.approval.requested", "scheduler.approval.decision_result"])
        self.assertIsNotNone(result["job_id"])

    def test_proposal_api_idempotency_replays_and_quarantines_conflicts(self) -> None:
        event_id = str(uuid.uuid4())
        semantic = "a" * 64
        first = self.scheduler.create_proposal(
            self.compile(minutes=20),
            now=self.reference,
            api_event_id=event_id,
            api_semantic_sha256=semantic,
        )
        replay = self.scheduler.create_proposal(
            self.compile(minutes=21),
            now=self.reference + timedelta(seconds=1),
            api_event_id=event_id,
            api_semantic_sha256=semantic,
        )
        self.assertEqual(replay, first)
        self.assertEqual(self.scheduler.lookup_api_result(event_id, semantic), {
            "result": first,
            "status": 201,
        })
        self.assertEqual(self.base._connection.execute(
            "SELECT COUNT(*) FROM scheduler_proposals"
        ).fetchone()[0], 1)
        self.assertEqual(self.base._connection.execute(
            "SELECT COUNT(*) FROM approval_arbiter_queue"
        ).fetchone()[0], 1)

        with self.assertRaisesRegex(OrchestratorError, "protocol.conflicting_duplicate"):
            self.scheduler.lookup_api_result(event_id, "b" * 64)
        self.assertEqual(self.base._connection.execute(
            "SELECT state FROM scheduler_inbox WHERE event_id=?", (event_id,)
        ).fetchone()[0], "QUARANTINED")

    def test_compile_api_replay_does_not_call_model_twice(self) -> None:
        compiled = self.compile(minutes=20)

        class Compiler:
            def __init__(self) -> None:
                self.calls = 0

            def compile(self, _model_input):
                self.calls += 1
                return compiled

        compiler = Compiler()
        application = ControlApplication(
            SimpleNamespace(),
            SimpleNamespace(scheduler=self.scheduler, schedule_compiler=compiler),
        )
        body = {
            "request_text": "Θύμισέ μου να κάνω διάλειμμα σε είκοσι λεπτά",
            "selected_target": {
                "job_id": None,
                "job_revision": None,
                "occurrence_id": None,
                "snooze_generation": None,
            },
            "default_timezone": "Europe/Athens",
            "default_locale": "el-GR",
        }
        event_id = str(uuid.uuid4())
        first = application.scheduler_mutate(
            "POST", "/v1/scheduler/compile", body, event_id
        )
        replay = application.scheduler_mutate(
            "POST", "/v1/scheduler/compile", body, event_id
        )
        self.assertEqual(replay, first)
        self.assertEqual(first[0], 201)
        self.assertEqual(compiler.calls, 1)
        self.assertEqual(self.base._connection.execute(
            "SELECT COUNT(*) FROM scheduler_proposals"
        ).fetchone()[0], 1)

    def test_fire_is_one_logical_delivery_and_ciphertext_hides_body(self) -> None:
        created = self.scheduler.create_proposal(self.compile(), now=self.reference)
        result = self.scheduler.decide_approval(
            approval_id=created["approval_id"], proposal_id=created["proposal_id"],
            proposal_sha256=created["proposal_sha256"], decision="approve",
            cause_event_id=str(uuid.uuid4()), now=self.reference,
        )
        due = self.reference + timedelta(minutes=20)
        self.assertEqual(len(self.scheduler.fire_due(now=due)), 1)
        self.assertEqual(self.scheduler.fire_due(now=due), [])
        self.assertEqual(self.base._connection.execute("SELECT COUNT(*) FROM scheduler_deliveries").fetchone()[0], 1)
        requested = self.base._connection.execute(
            "SELECT event_id FROM scheduler_outbox WHERE event_type='scheduler.notification.requested'"
        ).fetchone()[0]
        self.scheduler.acknowledge_scheduler_event(requested, status="applied", now=due)
        self.assertEqual(self.base._connection.execute(
            "SELECT state FROM scheduler_deliveries"
        ).fetchone()[0], "STAGED")
        stored_schedule = json.loads(self.base._connection.execute(
            "SELECT schedule_json FROM scheduler_jobs WHERE job_id=?", (result["job_id"],)
        ).fetchone()[0])
        self.assertEqual(stored_schedule["recurrence_contract_version"], "scheduler-recurrence/1")
        self.assertEqual(stored_schedule["timezone_contract"]["distribution"], "2026.3")
        self.assertEqual(len(stored_schedule["timezone_contract"]["tzif_sha256"]), 64)
        self.assertEqual(stored_schedule["dtstart_fold"], 0)
        self.base._connection.execute("PRAGMA wal_checkpoint(FULL)")
        data = self.path.read_bytes()
        self.assertNotIn("Κάνε το προγραμματισμένο διάλειμμα".encode(), data)
        self.assertEqual(self.scheduler.list_jobs()[0]["job_id"], result["job_id"])

    def test_finite_recurrence_drains_then_completes(self) -> None:
        created = self.scheduler.create_proposal(
            self.compile(recurrence="FIXED_INTERVAL", count=2), now=self.reference
        )
        result = self.scheduler.decide_approval(
            approval_id=created["approval_id"], proposal_id=created["proposal_id"],
            proposal_sha256=created["proposal_sha256"], decision="approve",
            cause_event_id=str(uuid.uuid4()), now=self.reference,
        )
        first = self.reference + timedelta(minutes=20)
        self.scheduler.fire_due(now=first)
        self.assertEqual(self.scheduler.list_jobs()[0]["state"], "DRAINING")
        self.scheduler.fire_due(now=first + timedelta(minutes=1))
        rows = self.base._connection.execute(
            "SELECT notification_id FROM scheduler_deliveries ORDER BY created_at"
        ).fetchall()
        for row in rows:
            self.base._connection.execute(
                "UPDATE scheduler_deliveries SET state='OS_ACCEPTED' WHERE notification_id=?", (row[0],)
            )
        self.base._connection.execute(
            "UPDATE scheduler_occurrences SET action_deadline_at=? WHERE state='FIRED'",
            ("2030-01-11T12:00:00.000Z",),
        )
        self.scheduler.expire_deliveries(now=utc("2030-01-11T12:00:01.000Z"))
        self.assertEqual(result["job_id"], self.scheduler.list_jobs()[0]["job_id"])
        self.assertEqual(self.scheduler.list_jobs()[0]["state"], "COMPLETED")

    def test_global_approval_fifo(self) -> None:
        first = self.scheduler.create_proposal(self.compile(minutes=20), now=self.reference)
        second = self.scheduler.create_proposal(self.compile(minutes=21), now=self.reference)
        owner = str(uuid.uuid4())
        promoted = self.scheduler.promote_next_approval(owner, supported_protocols=("jarvis.scheduler/1",), now=self.reference)
        self.assertEqual(promoted["payload"]["approval_id"], first["approval_id"])
        self.assertIsNone(self.scheduler.promote_next_approval(owner, supported_protocols=("jarvis.scheduler/1",), now=self.reference))
        self.scheduler.decide_approval(
            approval_id=first["approval_id"], proposal_id=first["proposal_id"],
            proposal_sha256=first["proposal_sha256"], decision="deny",
            cause_event_id=str(uuid.uuid4()), now=self.reference,
        )
        promoted = self.scheduler.promote_next_approval(owner, supported_protocols=("jarvis.scheduler/1",), now=self.reference)
        self.assertEqual(promoted["payload"]["approval_id"], second["approval_id"])

    def test_incompatible_cloud_head_blocks_scheduler_without_invalidation(self) -> None:
        cloud_approval = str(uuid.uuid4())
        self.base._connection.execute(
            """INSERT INTO approval_arbiter_queue(
               approval_id,producer_protocol,binding_sha256,state,created_at,expires_at,updated_at
               ) VALUES(?,'jarvis.cloud-approval/1',?,'WAITING_FOR_CHANNEL',?,?,?)""",
            (cloud_approval, "c" * 64, "2030-01-10T12:00:00.000Z",
             "2030-01-10T12:10:00.000Z", "2030-01-10T12:00:00.000Z"),
        )
        scheduler = self.scheduler.create_proposal(self.compile(), now=self.reference)
        self.assertIsNone(self.scheduler.promote_next_approval(
            str(uuid.uuid4()), supported_protocols=("jarvis.scheduler/1",), now=self.reference
        ))
        states = dict(self.base._connection.execute(
            "SELECT approval_id,state FROM approval_arbiter_queue"
        ).fetchall())
        self.assertEqual(states[cloud_approval], "WAITING_FOR_CHANNEL")
        self.assertEqual(states[scheduler["approval_id"]], "WAITING_FOR_CHANNEL")

    def test_owner_reconnect_rotates_presentation_and_nonce(self) -> None:
        created = self.scheduler.create_proposal(self.compile(), now=self.reference)
        session_id = str(uuid.uuid4())
        first = self.scheduler.promote_next_approval(
            session_id, supported_protocols=("jarvis.scheduler/1",), now=self.reference
        )
        self.assertTrue(self.scheduler.mark_owner_disconnected(session_id, now=self.reference))
        second = self.scheduler.promote_next_approval(
            session_id, supported_protocols=("jarvis.scheduler/1",), now=self.reference
        )
        self.assertNotEqual(first["event_id"], second["event_id"])
        self.assertNotEqual(first["payload"]["presentation_id"], second["payload"]["presentation_id"])
        self.assertNotEqual(first["payload"]["decision_nonce"], second["payload"]["decision_nonce"])
        self.assertEqual(second["payload"]["approval_id"], created["approval_id"])

    def test_unrepresentable_next_interval_does_not_block_the_due_notification(self) -> None:
        request_id = str(uuid.uuid4())
        output = proposal_output(request_id, recurrence="FIXED_INTERVAL")
        output["recurrence"].update(interval_amount=10_000_000, interval_unit="WEEK")
        compiled = compile_proposal(
            proposal_input(request_id, "2030-01-10T12:00:00.000Z"), output,
            compiler_qualification_sha256="1" * 64, zones=self.scheduler.zones,
        )
        created = self.scheduler.create_proposal(compiled, now=self.reference)
        decision = self.scheduler.decide_approval(
            approval_id=created["approval_id"], proposal_id=created["proposal_id"],
            proposal_sha256=created["proposal_sha256"], decision="approve",
            cause_event_id=str(uuid.uuid4()), now=self.reference,
        )
        fired = self.scheduler.fire_due(now=utc(compiled.first_due_at_utc))
        self.assertEqual(len(fired), 1)
        cursor = self.base._connection.execute(
            "SELECT exhausted,next_scheduled_at_utc FROM scheduler_recurrence_cursors WHERE job_id=?",
            (decision["job_id"],),
        ).fetchone()
        self.assertEqual(tuple(cursor), (1, None))
        self.assertEqual(self.scheduler.fire_due(now=utc(compiled.first_due_at_utc)), [])

    def test_monthly_calendar_exhaustion_keeps_the_last_supported_occurrence(self) -> None:
        request_id = str(uuid.uuid4())
        reference = utc("9999-12-27T00:00:00.000Z")
        incoming = proposal_input(request_id, "9999-12-27T00:00:00.000Z")
        incoming["default_timezone"] = "UTC"
        output = proposal_output(request_id, recurrence="MONTHLY")
        output["time_expression"] = {
            **temporal_none(), "kind": "ABSOLUTE_LOCAL", "local_datetime": "9999-12-28T09:00:00.000",
        }
        compiled = compile_proposal(incoming, output, compiler_qualification_sha256="1" * 64,
                                    zones=self.scheduler.zones)
        created = self.scheduler.create_proposal(compiled, now=reference)
        decision = self.scheduler.decide_approval(
            approval_id=created["approval_id"], proposal_id=created["proposal_id"],
            proposal_sha256=created["proposal_sha256"], decision="approve",
            cause_event_id=str(uuid.uuid4()), now=reference,
        )
        self.assertEqual(len(self.scheduler.fire_due(now=utc(compiled.first_due_at_utc))), 1)
        cursor = self.base._connection.execute(
            "SELECT exhausted,next_scheduled_at_utc FROM scheduler_recurrence_cursors WHERE job_id=?",
            (decision["job_id"],),
        ).fetchone()
        self.assertEqual(tuple(cursor), (1, None))

    def test_dst_gap_fold_and_monthly_skip_are_frozen(self) -> None:
        def absolute(local: str) -> dict:
            value = temporal_none()
            value.update({"kind": "ABSOLUTE_LOCAL", "local_datetime": local,
                          "timezone": "Europe/Athens"})
            return value

        request_id = str(uuid.uuid4())
        input_value = proposal_input(request_id, "2030-03-01T00:00:00.000Z")
        output = proposal_output(request_id)
        output["time_expression"] = absolute("2030-03-31T03:30:00.000")
        with self.assertRaisesRegex(ScheduleContractError, "one_shot_dst_gap"):
            compile_proposal(input_value, output, compiler_qualification_sha256="1" * 64,
                             zones=self.scheduler.zones)

        output["recurrence"]["kind"] = "DAILY"
        compiled = compile_proposal(input_value, output, compiler_qualification_sha256="1" * 64,
                                    zones=self.scheduler.zones)
        self.assertEqual(compiled.first_due_at_utc, "2030-04-01T00:30:00.000Z")

        fold_id = str(uuid.uuid4())
        fold_input = proposal_input(fold_id, "2030-10-01T00:00:00.000Z")
        fold_output = proposal_output(fold_id)
        fold_output["time_expression"] = absolute("2030-10-27T03:30:00.000")
        fold = compile_proposal(fold_input, fold_output, compiler_qualification_sha256="1" * 64,
                                zones=self.scheduler.zones)
        self.assertEqual(fold.first_due_at_utc, "2030-10-27T00:30:00.000Z")

        month_id = str(uuid.uuid4())
        month_input = proposal_input(month_id, "2030-01-01T00:00:00.000Z")
        month_output = proposal_output(month_id)
        month_output["time_expression"] = absolute("2030-01-31T09:00:00.000")
        month_output["recurrence"].update({"kind": "MONTHLY", "day_of_month": 31, "count": 3})
        compiled_month = compile_proposal(
            month_input, month_output, compiler_qualification_sha256="1" * 64,
            zones=self.scheduler.zones,
        )
        created = self.scheduler.create_proposal(compiled_month, now=utc("2030-01-01T00:00:00.000Z"))
        decision = self.scheduler.decide_approval(
            approval_id=created["approval_id"], proposal_id=created["proposal_id"],
            proposal_sha256=created["proposal_sha256"], decision="approve",
            cause_event_id=str(uuid.uuid4()), now=utc("2030-01-01T00:00:00.000Z"),
        )
        self.scheduler.fire_due(now=utc("2030-01-31T07:00:00.000Z"))
        self.scheduler.fire_due(now=utc("2030-03-31T06:00:00.000Z"))
        rows = self.base._connection.execute(
            "SELECT scheduled_at_utc FROM scheduler_occurrences WHERE job_id=? ORDER BY sequence",
            (decision["job_id"],),
        ).fetchall()
        self.assertEqual([row[0] for row in rows], [
            "2030-01-31T07:00:00.000Z", "2030-03-31T06:00:00.000Z", "2030-05-31T06:00:00.000Z",
        ])

    def test_snooze_generations_permission_expiry_and_ambiguous_restart(self) -> None:
        before = self.activate(minutes=20)
        occurrence = before["occurrence_id"]
        for generation in range(8):
            result = self.scheduler.snooze_occurrence(
                job_id=before["job_id"], job_revision=1, occurrence=occurrence,
                expected_generation=generation,
                snooze_until_at_utc=f"2030-01-10T13:{generation:02d}:00.000Z",
                cause_event_id=str(uuid.uuid4()), now=self.reference,
            )
            self.assertEqual(result["delivery_generation"], generation + 1)
        with self.assertRaisesRegex(OrchestratorError, "snooze_limit_reached"):
            self.scheduler.snooze_occurrence(
                job_id=before["job_id"], job_revision=1, occurrence=occurrence,
                expected_generation=8, snooze_until_at_utc="2030-01-10T14:00:00.000Z",
                cause_event_id=str(uuid.uuid4()), now=self.reference,
            )

        after = self.activate(minutes=21)
        due = self.reference + timedelta(minutes=21)
        self.scheduler.fire_due(now=due)
        delivery = self.base._connection.execute(
            "SELECT notification_id FROM scheduler_deliveries WHERE occurrence_id=?",
            (after["occurrence_id"],),
        ).fetchone()[0]
        accepted_attempt = str(uuid.uuid4())
        self.scheduler.record_delivery_receipt(
            event_id=str(uuid.uuid4()), semantic_sha256="2" * 64,
            notification=delivery, attempt_id=accepted_attempt, stage="OS_ACCEPTED",
            reason_code="native_api_accepted", observed_at_utc="2030-01-10T12:21:00.000Z",
        )
        snoozed = self.scheduler.snooze_occurrence(
            job_id=after["job_id"], job_revision=1, occurrence=after["occurrence_id"],
            expected_generation=0, snooze_until_at_utc="2030-01-10T13:00:00.000Z",
            cause_event_id=str(uuid.uuid4()), now=due,
        )
        self.assertEqual(snoozed["delivery_generation"], 1)
        self.assertEqual(self.base._connection.execute(
            """SELECT COUNT(*) FROM scheduler_outbox
               WHERE event_type='scheduler.notification.tombstone' AND entity_id=?""",
            (delivery,),
        ).fetchone()[0], 1)
        self.scheduler.fire_due(now=utc("2030-01-10T13:00:00.000Z"))
        self.assertEqual(self.base._connection.execute(
            "SELECT COUNT(*) FROM scheduler_deliveries WHERE occurrence_id=?", (after["occurrence_id"],)
        ).fetchone()[0], 2)

        blocked = self.activate(minutes=22)
        blocked_due = self.reference + timedelta(minutes=22)
        self.scheduler.fire_due(now=blocked_due)
        blocked_id = self.base._connection.execute(
            "SELECT notification_id FROM scheduler_deliveries WHERE occurrence_id=?", (blocked["occurrence_id"],)
        ).fetchone()[0]
        self.scheduler.record_delivery_receipt(
            event_id=str(uuid.uuid4()), semantic_sha256="3" * 64,
            notification=blocked_id, attempt_id=str(uuid.uuid4()), stage="OS_REJECTED",
            reason_code="permission_denied", observed_at_utc="2030-01-10T12:22:00.000Z",
        )
        self.scheduler.expire_deliveries(now=utc("2030-01-11T12:22:00.001Z"))
        self.assertEqual(self.base._connection.execute(
            "SELECT state FROM scheduler_deliveries WHERE notification_id=?", (blocked_id,)
        ).fetchone()[0], "EXPIRED_UNDELIVERED")

        ambiguous = self.activate(minutes=23)
        ambiguous_due = self.reference + timedelta(minutes=23)
        self.scheduler.fire_due(now=ambiguous_due)
        ambiguous_id = self.base._connection.execute(
            "SELECT notification_id FROM scheduler_deliveries WHERE occurrence_id=?", (ambiguous["occurrence_id"],)
        ).fetchone()[0]
        ambiguous_attempt = str(uuid.uuid4())
        self.scheduler.record_delivery_receipt(
            event_id=str(uuid.uuid4()), semantic_sha256="4" * 64,
            notification=ambiguous_id, attempt_id=ambiguous_attempt, stage="DISPATCHING",
            reason_code="native_call_starting", observed_at_utc="2030-01-10T12:23:00.000Z",
        )
        recovery = self.scheduler.recover_after_restart(now=utc("2030-01-10T12:23:01.000Z"))
        self.assertEqual(recovery["ambiguous_attempt_count"], 1)
        self.assertEqual(self.base._connection.execute(
            "SELECT state FROM scheduler_delivery_attempts WHERE attempt_id=?", (ambiguous_attempt,)
        ).fetchone()[0], "AMBIGUOUS_OS_DISPATCH")
        self.scheduler.record_delivery_receipt(
            event_id=str(uuid.uuid4()), semantic_sha256="5" * 64,
            notification=ambiguous_id, attempt_id=str(uuid.uuid4()), stage="OS_ACCEPTED",
            reason_code="native_api_accepted", observed_at_utc="2030-01-10T12:23:02.000Z",
        )
        self.assertEqual(self.base._connection.execute(
            "SELECT COUNT(*) FROM scheduler_delivery_attempts WHERE notification_id=?", (ambiguous_id,)
        ).fetchone()[0], 2)

    def test_cancel_revokes_pending_notification_and_emits_tombstone(self) -> None:
        activated = self.activate()
        due = self.reference + timedelta(minutes=20)
        self.scheduler.fire_due(now=due)
        delivery = self.base._connection.execute(
            "SELECT notification_id FROM scheduler_deliveries WHERE occurrence_id=?",
            (activated["occurrence_id"],),
        ).fetchone()[0]
        self.scheduler.cancel_job(
            activated["job_id"], 1, cause_event_id=str(uuid.uuid4()), now=due
        )
        rows = self.base._connection.execute(
            "SELECT event_type,state FROM scheduler_outbox WHERE entity_id=? ORDER BY ordinal",
            (delivery,),
        ).fetchall()
        self.assertEqual([tuple(row) for row in rows], [
            ("scheduler.notification.requested", "REVOKED"),
            ("scheduler.notification.tombstone", "PENDING"),
        ])
        self.assertEqual(self.base._connection.execute(
            "SELECT state FROM scheduler_deliveries WHERE notification_id=?", (delivery,)
        ).fetchone()[0], "REVOKED")

    def test_native_notification_action_is_atomic_and_idempotent(self) -> None:
        activated = self.activate()
        action_event_id = str(uuid.uuid4())
        applied = self.scheduler.apply_notification_action(
            event_id=action_event_id,
            semantic_sha256="9" * 64,
            action="CANCEL",
            job_id=activated["job_id"],
            job_revision=1,
            occurrence=activated["occurrence_id"],
            expected_generation=0,
            requested_due_at_utc=None,
            now=self.reference,
        )
        duplicate = self.scheduler.apply_notification_action(
            event_id=action_event_id,
            semantic_sha256="9" * 64,
            action="CANCEL",
            job_id=activated["job_id"],
            job_revision=1,
            occurrence=activated["occurrence_id"],
            expected_generation=0,
            requested_due_at_utc=None,
            now=self.reference,
        )
        self.assertTrue(applied)
        self.assertFalse(duplicate)
        self.assertEqual(self.base._connection.execute(
            "SELECT COUNT(*) FROM scheduler_inbox WHERE event_id=?", (action_event_id,)
        ).fetchone()[0], 1)
        results = self.base._connection.execute(
            """SELECT event_type FROM scheduler_outbox
               WHERE entity_id=? ORDER BY ordinal""", (action_event_id,)
        ).fetchall()
        self.assertEqual([row[0] for row in results], [
            "scheduler.notification.action_result",
            "scheduler.notification.action_result",
        ])

    def test_terminal_delivery_and_tombstone_precedence_ignore_late_receipt(self) -> None:
        activated = self.activate()
        due = self.reference + timedelta(minutes=20)
        self.scheduler.fire_due(now=due)
        delivery = self.base._connection.execute(
            "SELECT notification_id FROM scheduler_deliveries WHERE occurrence_id=?",
            (activated["occurrence_id"],),
        ).fetchone()[0]
        attempt = str(uuid.uuid4())
        self.scheduler.record_delivery_receipt(
            event_id=str(uuid.uuid4()), semantic_sha256="6" * 64,
            notification=delivery, attempt_id=attempt, stage="OS_ACCEPTED",
            reason_code="native_api_accepted", observed_at_utc="2030-01-10T12:20:00.000Z",
        )
        self.assertFalse(self.scheduler.record_delivery_receipt(
            event_id=str(uuid.uuid4()), semantic_sha256="7" * 64,
            notification=delivery, attempt_id=attempt, stage="AMBIGUOUS_OS_DISPATCH",
            reason_code="late_restart_receipt", observed_at_utc="2030-01-10T12:20:01.000Z",
        ))
        self.assertEqual(self.base._connection.execute(
            "SELECT state FROM scheduler_deliveries WHERE notification_id=?", (delivery,)
        ).fetchone()[0], "OS_ACCEPTED")

    def test_recovery_coalesces_to_exactly_one_latest_slot(self) -> None:
        created = self.scheduler.create_proposal(
            self.compile(recurrence="FIXED_INTERVAL", count=5), now=self.reference
        )
        decision = self.scheduler.decide_approval(
            approval_id=created["approval_id"], proposal_id=created["proposal_id"],
            proposal_sha256=created["proposal_sha256"], decision="approve",
            cause_event_id=str(uuid.uuid4()), now=self.reference,
        )
        fired = self.scheduler.fire_due(now=utc("2030-01-10T12:25:00.000Z"))
        self.assertEqual(len(fired), 1)
        rows = self.base._connection.execute(
            "SELECT state,scheduled_at_utc FROM scheduler_occurrences WHERE job_id=? ORDER BY sequence",
            (decision["job_id"],),
        ).fetchall()
        self.assertEqual([row[0] for row in rows], [
            "MISSED_COALESCED", "MISSED_COALESCED", "MISSED_COALESCED", "MISSED_COALESCED", "FIRED",
        ])
        self.assertEqual(rows[-1][1], "2030-01-10T12:24:00.000Z")
        self.assertEqual(self.base._connection.execute(
            "SELECT COUNT(*) FROM scheduler_deliveries WHERE occurrence_id=?", (fired[0],)
        ).fetchone()[0], 1)

    def test_key_rotation_reencrypts_bounded_records_without_plaintext(self) -> None:
        self.scheduler.create_proposal(self.compile(), now=self.reference)
        before = self.base._connection.execute(
            "SELECT COUNT(*) FROM scheduler_content WHERE key_version=1"
        ).fetchone()[0]
        self.assertGreater(before, 0)
        result = self.scheduler.rotate_content_key(batch_size=1)
        self.assertEqual(result, {"key_version": 2, "rotated_record_count": before})
        self.assertEqual(self.base._connection.execute(
            "SELECT COUNT(*) FROM scheduler_content WHERE key_version<>2"
        ).fetchone()[0], 0)
        promoted = self.scheduler.promote_next_approval(
            str(uuid.uuid4()), supported_protocols=("jarvis.scheduler/1",), now=self.reference
        )
        self.assertEqual(promoted["payload"]["preview"]["notification"]["body"],
                         "Κάνε το προγραμματισμένο διάλειμμα.")

    def test_missing_key_blocks_without_ack_and_recovers_before_expiry(self) -> None:
        activated = self.activate()
        provider = self.scheduler.cipher.provider
        key = provider._keys.pop(1)
        due = self.reference + timedelta(minutes=20)
        self.scheduler.fire_due(now=due)
        delivery = self.base._connection.execute(
            "SELECT * FROM scheduler_deliveries WHERE occurrence_id=?",
            (activated["occurrence_id"],),
        ).fetchone()
        self.assertEqual(delivery["state"], "BLOCKED_KEY_UNAVAILABLE")
        self.assertEqual(self.base._connection.execute(
            "SELECT COUNT(*) FROM scheduler_outbox WHERE event_type='scheduler.notification.requested'"
        ).fetchone()[0], 0)
        provider._keys[1] = key
        self.assertEqual(self.scheduler.retry_key_blocked_deliveries(now=due), 1)
        self.assertEqual(self.base._connection.execute(
            "SELECT state FROM scheduler_deliveries WHERE notification_id=?", (delivery["notification_id"],)
        ).fetchone()[0], "OUTBOXED")

    def test_minimum_lead_boundary_is_inclusive_and_persistently_expires(self) -> None:
        def ten_second_compiled():
            request_id = str(uuid.uuid4())
            output = proposal_output(request_id)
            output["time_expression"] = temporal_after(10)
            output["time_expression"]["unit"] = "SECOND"
            return compile_proposal(
                proposal_input(request_id, "2030-01-10T12:00:00.000Z"), output,
                compiler_qualification_sha256="1" * 64, zones=self.scheduler.zones,
            )

        exact = self.scheduler.create_proposal(ten_second_compiled(), now=self.reference)
        self.scheduler.decide_approval(
            approval_id=exact["approval_id"], proposal_id=exact["proposal_id"],
            proposal_sha256=exact["proposal_sha256"], decision="approve",
            cause_event_id=str(uuid.uuid4()), now=self.reference,
        )
        stale = self.scheduler.create_proposal(ten_second_compiled(), now=self.reference)
        with self.assertRaisesRegex(OrchestratorError, "minimum_lead_violated"):
            self.scheduler.decide_approval(
                approval_id=stale["approval_id"], proposal_id=stale["proposal_id"],
                proposal_sha256=stale["proposal_sha256"], decision="approve",
                cause_event_id=str(uuid.uuid4()),
                now=self.reference + timedelta(milliseconds=1),
            )
        self.assertEqual(self.base._connection.execute(
            "SELECT state FROM scheduler_proposals WHERE proposal_id=?", (stale["proposal_id"],)
        ).fetchone()[0], "EXPIRED")

    def test_local_compiler_is_one_strict_retry_free_loopback_request(self) -> None:
        request_id = str(uuid.uuid4())
        model_input = proposal_input(request_id, "2030-01-10T12:00:00.000Z")
        model_output = proposal_output(request_id)
        calls: list[dict] = []

        import httpx
        client_type = httpx.AsyncClient

        def response(request):
            calls.append({"url": str(request.url), "json": json.loads(request.content)})
            return httpx.Response(200, json={"choices": [{
                "index": 0, "message": {"role": "assistant", "content": json.dumps(model_output)},
                "finish_reason": "stop",
            }]}, headers={"X-Jarvis-Compiler-Identity": request.headers["X-Jarvis-Compiler-Identity"]})

        def client(**kwargs):
            calls.append({"client": kwargs})
            return client_type(transport=httpx.MockTransport(response), **kwargs)

        from test_schedule_compiler import qualification_document
        document = qualification_document()
        compiler = LocalScheduleCompiler(
            base_url="http://127.0.0.1:8765/v1",
            bearer_token="local-token",
            model_alias="jarvis-local",
            qualification=CompilerQualification(document["qualification_sha256"], document),
            timeout_seconds=5,
            zones=self.scheduler.zones,
        )
        with patch("httpx.AsyncClient", client):
            compiled = compiler.compile(model_input)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["client"], {
            "timeout": 5, "trust_env": False, "follow_redirects": False,
        })
        request = calls[1]
        self.assertEqual(request["url"], "http://127.0.0.1:8765/v1/chat/completions")
        self.assertEqual(request["json"]["model"], "jarvis-local")
        self.assertEqual(request["json"]["max_completion_tokens"], 512)
        self.assertNotIn("tools", request["json"])
        self.assertTrue(request["json"]["response_format"]["json_schema"]["strict"])
        self.assertEqual(compiled.first_due_at_utc, "2030-01-10T12:20:00.000Z")

    def test_local_compiler_rejects_userinfo_and_noncanonical_alias(self) -> None:
        qualification = CompilerQualification("1" * 64, {})
        with self.assertRaisesRegex(ScheduleContractError, "compiler_not_loopback"):
            LocalScheduleCompiler(
                base_url="http://127.0.0.1:8765@public.example/v1",
                bearer_token="local-token",
                model_alias="jarvis-local",
                qualification=qualification,
                timeout_seconds=5,
                zones=self.scheduler.zones,
            )
        with self.assertRaisesRegex(ScheduleContractError, "compiler_model_alias_mismatch"):
            LocalScheduleCompiler(
                base_url="http://127.0.0.1:8765/v1",
                bearer_token="local-token",
                model_alias="other-model",
                qualification=qualification,
                timeout_seconds=5,
                zones=self.scheduler.zones,
            )


if __name__ == "__main__":
    unittest.main()
