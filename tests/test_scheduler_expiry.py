"""Exercise expiry at initial delivery and durable restart recovery, without native I/O."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.scheduler_crypto import (
    InMemorySchedulerKeyProvider,
    SchedulerContentCipher,
)
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.scheduler_client import SchedulerClientController
from jarvis_voice.scheduler_journal import SchedulerClientJournal
from jarvis_voice.scheduler_native import QualifiedNativeSchedulerDelivery
from jarvis_voice.scheduler_protocol import SchedulerEventFactory
from jarvis_voice.scheduler_runtime import SchedulerNativeRuntime
from test_scheduler_native import FakeAdapter, UnknownLock


class SchedulerExpiryTests(unittest.TestCase):
    def test_expiry_prevents_native_dispatch_on_receipt_and_after_restart(self):
        now = datetime(2030, 1, 10, 12, tzinfo=UTC)
        for recovered in (False, True):
            for remaining_ms in (-1, 0, 1):
                with self.subTest(recovered=recovered, remaining_ms=remaining_ms):
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "scheduler.sqlite3"
                        cipher = SchedulerContentCipher(InMemorySchedulerKeyProvider())
                        journal = SchedulerClientJournal(path, cipher=cipher)
                        factory = SchedulerEventFactory(str(uuid.uuid4()))
                        requested = factory.create(
                            "scheduler.notification.requested",
                            {
                                "notification_id": "3" * 64,
                                "occurrence_id": "4" * 64,
                                "job_id": str(uuid.uuid4()),
                                "job_revision": 1,
                                "delivery_generation": 0,
                                "scheduled_at_utc": "2030-01-09T12:00:00.000Z",
                                "effective_due_at_utc": "2030-01-09T12:00:00.000Z",
                                "expires_at_utc": (
                                    now + timedelta(milliseconds=remaining_ms)
                                )
                                .isoformat()
                                .replace("+00:00", "Z"),
                                "content_sha256": "5" * 64,
                                "title": "Synthetic title",
                                "body": "Synthetic body",
                                "private_title": "JARVIS",
                                "private_body": "Reminder",
                                "locale": "en-US",
                                "visibility": "private",
                                "allow_full_body_when_locked": False,
                                "silent": True,
                            },
                        )
                        event = requested
                        if recovered:
                            journal.stage(
                                event_id=requested.event_id,
                                semantic_sha256=requested.semantic_sha256,
                                event_type=requested.type,
                                payload=requested.payload,
                            )
                            journal.close()
                            journal = SchedulerClientJournal(path, cipher=cipher)
                            factory = SchedulerEventFactory(str(uuid.uuid4()))
                            event = factory.create(
                                "scheduler.ready",
                                {
                                    "protocol": "jarvis.scheduler/1",
                                    "maximum_frame_bytes": 16384,
                                    "sequence_namespace": "protocol_direction_connection",
                                },
                            )
                        adapter = FakeAdapter()
                        runtime = SchedulerNativeRuntime(
                            controller=SchedulerClientController(
                                session_id=factory.session_id,
                                journal=journal,
                                approval_arbiter=GlobalApprovalController(),
                            ),
                            delivery=QualifiedNativeSchedulerDelivery(
                                adapter=adapter, lock_state=UnknownLock()
                            ),
                            capabilities=(
                                "jarvis.scheduler/1",
                                "jarvis.approval-arbiter/1",
                            ),
                            manifest_sha256="d" * 64,
                        )
                        sent = []
                        try:
                            with patch(
                                "jarvis_voice.scheduler_runtime.datetime",
                                wraps=datetime,
                                create=True,
                            ) as clock:
                                clock.now.return_value = now
                                runtime.handle_server_event(
                                    event,
                                    key_counter=0,
                                    both_keys_up=True,
                                    send=sent.append,
                                )
                            expected = int(remaining_ms > 0)
                            self.assertEqual(len(adapter.posts), expected)
                            self.assertEqual(
                                journal._db.execute(
                                    "SELECT COUNT(*) FROM scheduler_delivery_attempts"
                                ).fetchone()[0],
                                expected,
                            )
                            self.assertEqual(
                                [
                                    item.payload["stage"]
                                    for item in sent
                                    if item.type == "scheduler.notification.receipt"
                                ],
                                ["DISPATCHING", "OS_ACCEPTED"] if expected else [],
                            )
                        finally:
                            runtime.close()


if __name__ == "__main__":
    unittest.main()
