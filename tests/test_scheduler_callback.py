from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.scheduler_callback import NativeSchedulerCallback, sign_callback
from jarvis_voice.scheduler_protocol import SchedulerEventFactory
from jarvis_voice.scheduler_client import SchedulerClientController
from jarvis_voice.scheduler_journal import SchedulerClientJournal
from jarvis_voice.scheduler_runtime import SchedulerNativeRuntime


class _Delivery:
    def post(self, payload):  # pragma: no cover - unused seam
        raise AssertionError

    def remove(self, notification_id):
        return None


class NativeSchedulerCallbackTests(unittest.TestCase):
    def test_malformed_frames_have_content_free_errors(self) -> None:
        malformed = json.loads(sign_callback(self.payload, key=self.key))
        malformed["occurred_at_utc"] = "private-callback-\ud800"
        for raw in ('[' * 1500 + '0' + ']' * 1500, "\ud800", json.dumps(malformed)):
            with self.subTest(length=len(raw)):
                with self.assertRaisesRegex(ValueError, "^callback.invalid_frame$"):
                    NativeSchedulerCallback.verify(raw, key=self.key)

    def setUp(self) -> None:
        self.key = bytes(range(32))
        self.callback_id = str(uuid.uuid4())
        self.notification_id = "a" * 64
        self.payload = {
            "callback_id": self.callback_id,
            "notification_id": self.notification_id,
            "action": "SNOOZE",
            "requested_due_at_utc": "2026-08-11T12:15:00.000Z",
            "occurred_at_utc": "2026-08-11T12:05:00.000Z",
        }

    def test_exact_authenticated_envelope(self) -> None:
        raw = sign_callback(self.payload, key=self.key)
        parsed = NativeSchedulerCallback.verify(raw, key=self.key)
        self.assertEqual(parsed.notification_id, self.notification_id)
        changed = json.loads(raw)
        changed["action"] = "CANCEL"
        with self.assertRaisesRegex(ValueError, "authentication_failed"):
            NativeSchedulerCallback.verify(json.dumps(changed), key=self.key)

    def test_cancel_forbids_due_time_and_unknown_fields(self) -> None:
        payload = {**self.payload, "action": "CANCEL", "requested_due_at_utc": None}
        raw = json.loads(sign_callback(payload, key=self.key))
        raw["extra"] = True
        with self.assertRaisesRegex(ValueError, "invalid_fields"):
            NativeSchedulerCallback.verify(json.dumps(raw), key=self.key)

    def test_runtime_binds_action_to_authoritative_notification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cipher = SchedulerContentCipher(InMemorySchedulerKeyProvider())
            journal = SchedulerClientJournal(
                Path(temporary) / "scheduler.sqlite3",
                cipher=cipher,
            )
            controller = SchedulerClientController(
                session_id=str(uuid.uuid4()),
                journal=journal,
                approval_arbiter=GlobalApprovalController(),
            )
            runtime = SchedulerNativeRuntime(
                controller=controller,
                delivery=_Delivery(),
                capabilities=(),
                manifest_sha256="b" * 64,
            )
            server_payload = {
                "notification_id": self.notification_id,
                "occurrence_id": "c" * 64,
                "job_id": str(uuid.uuid4()),
                "job_revision": 7,
                "delivery_generation": 2,
            }
            journal.stage(
                event_id=str(uuid.uuid4()),
                semantic_sha256="d" * 64,
                event_type="scheduler.notification.requested",
                payload=server_payload,
            )
            raw = sign_callback(self.payload, key=self.key)
            self.assertEqual(runtime.ingest_native_callback(raw, key=self.key), "applied")
            self.assertEqual(runtime.ingest_native_callback(raw, key=self.key), "duplicate")
            actions = runtime.pump_native_callbacks()
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0].payload, {
                "action": "SNOOZE",
                "job_id": server_payload["job_id"],
                "job_revision": 7,
                "occurrence_id": "c" * 64,
                "snooze_generation": 2,
                "requested_due_at_utc": "2026-08-11T12:15:00.000Z",
            })
            self.assertEqual(runtime.pump_native_callbacks(), [])
            runtime.begin_new_connection()
            retried = runtime.pump_native_callbacks()
            self.assertEqual(len(retried), 1)
            self.assertEqual(retried[0].event_id, actions[0].event_id)
            runtime.close()
            journal = SchedulerClientJournal(Path(temporary) / "scheduler.sqlite3", cipher=cipher)
            controller = SchedulerClientController(
                session_id=str(uuid.uuid4()), journal=journal,
                approval_arbiter=GlobalApprovalController(),
            )
            runtime = SchedulerNativeRuntime(
                controller=controller, delivery=_Delivery(), capabilities=(), manifest_sha256="b" * 64,
            )
            restarted = runtime.pump_native_callbacks()
            self.assertEqual(len(restarted), 1)
            self.assertEqual(restarted[0].event_id, actions[0].event_id)
            self.assertEqual(restarted[0].payload, actions[0].payload)
            self.assertNotEqual(restarted[0].session_id, actions[0].session_id)
            result = SchedulerEventFactory(controller.factory.session_id).create(
                "scheduler.notification.action_result",
                {
                    "action_event_id": actions[0].event_id,
                    "result": "APPLIED",
                    "reason_code": "action_applied",
                },
            )
            runtime.handle_server_event(result, key_counter=0, both_keys_up=True, send=lambda _item: None)
            runtime.begin_new_connection()
            self.assertEqual(runtime.pump_native_callbacks(), [])
            runtime.close()

    def test_pump_rejects_bad_or_retired_callbacks_without_losing_later_actions(self) -> None:
        from contextlib import closing

        frames = []

        class Delivery(_Delivery):
            callback_key = self.key

            def drain_callbacks(self):
                batch = frames[:]
                frames.clear()
                return batch

        with tempfile.TemporaryDirectory() as temporary:
            journal = SchedulerClientJournal(
                Path(temporary) / "scheduler.sqlite3",
                cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            )
            runtime = SchedulerNativeRuntime(
                controller=SchedulerClientController(
                    session_id=str(uuid.uuid4()), journal=journal,
                    approval_arbiter=GlobalApprovalController(),
                ),
                delivery=Delivery(), capabilities=(), manifest_sha256="e" * 64,
            )
            with closing(runtime):
                for notification_id in (self.notification_id, "b" * 64):
                    journal.stage(
                        event_id=str(uuid.uuid4()), semantic_sha256="d" * 64,
                        event_type="scheduler.notification.requested",
                        payload={"notification_id": notification_id, "occurrence_id": "c" * 64,
                                 "job_id": str(uuid.uuid4()), "job_revision": 1,
                                 "delivery_generation": 0},
                    )
                journal.tombstone("b" * 64, "cancelled")
                valid = sign_callback(self.payload, key=self.key)
                frames.extend([
                    b'{"contract":"jarvis.scheduler-native-callback/1"}',
                    sign_callback(self.payload, key=b"x" * 32),
                    sign_callback({**self.payload, "notification_id": "b" * 64}, key=self.key),
                    valid,
                ])
                actions = runtime.pump_native_callbacks()
                self.assertEqual(len(actions), 1)
                self.assertEqual(actions[0].payload["action"], "SNOOZE")
                self.assertEqual(runtime.pump_native_callbacks(), [])
                for error in (OSError("storage unavailable"), ValueError("invalid encrypted data")):
                    frames.append(valid)
                    with patch.object(journal, "stage_native_callback", side_effect=error):
                        with self.assertRaises(type(error)):
                            runtime.pump_native_callbacks()

    def test_unknown_or_tombstoned_notification_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journal = SchedulerClientJournal(
                Path(temporary) / "scheduler.sqlite3",
                cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            )
            controller = SchedulerClientController(
                session_id=str(uuid.uuid4()), journal=journal,
                approval_arbiter=GlobalApprovalController(),
            )
            runtime = SchedulerNativeRuntime(
                controller=controller, delivery=_Delivery(), capabilities=(), manifest_sha256="e" * 64
            )
            with self.assertRaisesRegex(ValueError, "notification_not_active"):
                runtime.ingest_native_callback(sign_callback(self.payload, key=self.key), key=self.key)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
