from __future__ import annotations

import tempfile
import json
import unittest
import uuid
from contextlib import closing
from pathlib import Path

from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.cloud_approval import CloudApprovalEventFactory
from jarvis_voice.errors import ProtocolError
from jarvis_voice.protocol import EventFactory
from jarvis_voice.scheduler_journal import SchedulerClientJournal
from jarvis_voice.scheduler_client import SchedulerClientController, select_notification_presentation
from jarvis_voice.scheduler_protocol import SchedulerEnvelope, SchedulerEventFactory


class SchedulerTransportTests(unittest.TestCase):
    def test_notification_replay_preserves_payload_identity_across_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory, closing(SchedulerClientJournal(
                Path(directory) / "voice.sqlite3", cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            )) as journal:
            payload = {"occurrence_id": "a" * 64, "notification_id": "b" * 64,
                       "unit_index": 1, "locale": "en-US", "speakable_text": "Original speech."}
            first = SchedulerEventFactory(str(uuid.uuid4())).create(
                "scheduler.notification.speech", payload, sent_at="2030-01-10T12:00:00.000Z",
            )
            controller = SchedulerClientController(session_id=first.session_id, journal=journal, approval_arbiter=GlobalApprovalController())
            controller.handle_server_event(first, key_counter=0, both_keys_up=True)
            server = SchedulerEventFactory(str(uuid.uuid4()))
            controller = SchedulerClientController(session_id=server.session_id, journal=journal, approval_arbiter=GlobalApprovalController())
            replay = server.create(first.type, payload, event_id=first.event_id, sent_at="2030-01-10T12:00:01.000Z")
            ack = controller.handle_server_event(replay, key_counter=0, both_keys_up=True)
            self.assertEqual(ack[0].payload["status"], "duplicate")
            self.assertEqual(journal._db.execute("SELECT semantic_sha256 FROM encrypted_frames WHERE event_id=?", (first.event_id,)).fetchone()[0], first.semantic_sha256)
            changed = server.create(first.type, {**payload, "speakable_text": "Changed speech."}, event_id=first.event_id)
            with self.assertRaisesRegex(ValueError, "protocol.conflicting_duplicate"):
                controller.handle_server_event(changed, key_counter=0, both_keys_up=True)

    def test_invalid_scheduler_values_are_typed_and_do_not_consume_sequence(self) -> None:
        ready = {"protocol": "jarvis.scheduler/1", "maximum_frame_bytes": 16384,
                 "sequence_namespace": "protocol_direction_connection"}
        for event_type, payload in (
            ([], ready),
            ("scheduler.event_ack", {"acked_event_id": str(uuid.uuid4()), "status": []}),
            ("scheduler.notification.action_result", {"action_event_id": str(uuid.uuid4()), "result": {}, "reason_code": "ok"}),
            ("scheduler.notification.speech", {"occurrence_id": "a" * 64, "notification_id": "b" * 64,
                "unit_index": 1, "locale": "en-US", "speakable_text": "\ud800"}),
        ):
            with self.subTest(event_type=event_type):
                factory = SchedulerEventFactory(str(uuid.uuid4()))
                with self.assertRaises(ProtocolError):
                    factory.create(event_type, payload)
                self.assertEqual(factory.create("scheduler.ready", ready).seq, 1)
        valid = SchedulerEventFactory(str(uuid.uuid4())).create("scheduler.ready", ready).to_dict()
        valid["type"] = []
        with self.assertRaises(ProtocolError):
            SchedulerEnvelope.from_json(json.dumps(valid))
        with self.assertRaises(ProtocolError):
            SchedulerEnvelope.from_json("\ud800")

    def test_action_requires_typed_target_and_matching_due_time(self) -> None:
        payload = {"action": "CANCEL", "job_id": str(uuid.uuid4()), "job_revision": 1,
                   "occurrence_id": "a" * 64, "snooze_generation": 0,
                   "requested_due_at_utc": None}
        for field, invalid in (
            ("job_id", None),
            *(("job_revision", value) for value in (None, True, 1.0, "1", [], 0, -1, 2**63)),
            *(("snooze_generation", value) for value in (None, False, 0.0, "0", [], -1, 2**63)),
            ("requested_due_at_utc", "2030-01-10T12:20:00.000Z"),
            ("action", "SNOOZE"),
        ):
            with self.subTest(field=field, invalid=invalid):
                factory = SchedulerEventFactory(str(uuid.uuid4()))
                with self.assertRaises(ProtocolError):
                    factory.create("scheduler.notification.action", {**payload, field: invalid})
                self.assertEqual(factory.next_seq, 1)
        factory = SchedulerEventFactory(str(uuid.uuid4()))
        first = factory.create("scheduler.notification.action", payload)
        second = factory.create("scheduler.notification.action", {
            **payload, "action": "SNOOZE", "requested_due_at_utc": "2030-01-10T12:20:00.000Z",
        })
        self.assertEqual((first.seq, second.seq), (1, 2))

    def test_each_protocol_starts_at_sequence_one(self) -> None:
        session = str(uuid.uuid4())
        voice = EventFactory(session_id=session).create("client.hello", {"protocol": "jarvis.voice/1", "capabilities": []})
        cloud = CloudApprovalEventFactory(session).create("cloud.approval.ready", {"protocol": "jarvis.cloud-approval/1"})
        scheduler = SchedulerEventFactory(session).create("scheduler.ready", {
            "protocol": "jarvis.scheduler/1", "maximum_frame_bytes": 16384,
            "sequence_namespace": "protocol_direction_connection",
        })
        self.assertEqual((voice.seq, cloud.seq, scheduler.seq), (1, 1, 1))

    def test_conflicting_duplicate_and_frame_bound_fail_closed(self) -> None:
        factory = SchedulerEventFactory(str(uuid.uuid4()))
        event = factory.create("scheduler.ready", {
            "protocol": "jarvis.scheduler/1", "maximum_frame_bytes": 16384,
            "sequence_namespace": "protocol_direction_connection",
        })
        value = event.to_dict()
        value["payload"] = {**value["payload"], "unknown": True}
        with self.assertRaises(ProtocolError):
            SchedulerEnvelope.from_json(__import__("json").dumps(value))

    def test_global_client_arbiter_rejects_second_protocol(self) -> None:
        arbiter = GlobalApprovalController()
        arbiter.stage(
            protocol="jarvis.cloud-approval/1", approval_id=str(uuid.uuid4()),
            presentation_id=str(uuid.uuid4()), requested_event_id=str(uuid.uuid4()),
            key_counter=1, both_keys_up=True,
        )
        with self.assertRaisesRegex(ProtocolError, "second local approval"):
            arbiter.stage(
                protocol="jarvis.scheduler/1", approval_id=str(uuid.uuid4()),
                presentation_id=str(uuid.uuid4()), requested_event_id=str(uuid.uuid4()),
                key_counter=1, both_keys_up=True,
            )

    def test_client_journal_contains_no_plaintext_and_detects_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scheduler.sqlite3"
            journal = SchedulerClientJournal(
                path, cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider())
            )
            event_id = str(uuid.uuid4())
            body = "Πολύ ιδιωτικό περιεχόμενο"
            self.assertEqual(journal.stage(
                event_id=event_id, semantic_sha256="a" * 64,
                event_type="scheduler.notification.requested", payload={"body": body},
            ), "applied")
            journal._db.execute("PRAGMA wal_checkpoint(FULL)")
            self.assertNotIn(body.encode(), path.read_bytes())
            self.assertEqual(journal.read(event_id)["body"], body)
            self.assertEqual(journal.stage(
                event_id=event_id, semantic_sha256="a" * 64,
                event_type="scheduler.notification.requested", payload={"body": body},
            ), "duplicate")
            with self.assertRaises(ValueError):
                journal.stage(
                    event_id=event_id, semantic_sha256="b" * 64,
                    event_type="scheduler.notification.requested", payload={"body": "changed"},
                )
            journal.close()

    def test_lock_privacy_downgrade_is_fail_closed(self) -> None:
        payload = {
            "title": "Approved title",
            "body": "Approved body",
            "private_title": "JARVIS",
            "private_body": "You have a reminder.",
            "visibility": "full",
            "allow_full_body_when_locked": False,
        }
        locked = select_notification_presentation(payload, lock_state="locked")
        unknown = select_notification_presentation(payload, lock_state="unknown")
        unlocked = select_notification_presentation(payload, lock_state="unlocked")
        self.assertEqual((locked["effective_visibility"], locked["body"], locked["tts_allowed"]),
                         ("title_only", "You have a reminder.", False))
        self.assertEqual((unknown["effective_visibility"], unknown["title"], unknown["tts_allowed"]),
                         ("private", "JARVIS", False))
        self.assertEqual((unlocked["effective_visibility"], unlocked["body"], unlocked["tts_allowed"]),
                         ("full", "Approved body", True))
        payload["visibility"] = "private"
        private = select_notification_presentation(payload, lock_state="unlocked")
        self.assertEqual((private["effective_visibility"], private["body"]),
                         ("private", "You have a reminder."))


if __name__ == "__main__":
    unittest.main()
