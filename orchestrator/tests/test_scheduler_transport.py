from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.cloud_approval import CloudApprovalEventFactory
from jarvis_voice.errors import ProtocolError
from jarvis_voice.protocol import EventFactory
from jarvis_voice.scheduler_journal import SchedulerClientJournal
from jarvis_voice.scheduler_client import select_notification_presentation
from jarvis_voice.scheduler_protocol import SchedulerEnvelope, SchedulerEventFactory


class SchedulerTransportTests(unittest.TestCase):
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
                    event_type="scheduler.notification.requested", payload={"body": body},
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
