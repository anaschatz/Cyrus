from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
import unittest
import uuid
import plistlib
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.scheduler_client import SchedulerClientController
from jarvis_voice.scheduler_compatibility import SchedulerCompatibilityManifest
from jarvis_voice.scheduler_journal import SchedulerClientJournal
from jarvis_voice.scheduler_native import (
    MACOS_LOCK_PROVIDER_SHA256,
    MACOS_NOTIFICATION_ADAPTER_SHA256,
    MacOSLockStateProvider,
    MacOSNotificationAdapter,
    NativeDispatchResult,
    QualifiedNativeSchedulerDelivery,
    WindowsLockStateProvider,
    WindowsNotificationAdapter,
)
from jarvis_voice.scheduler_native_qualification import load_native_scheduler_qualification
from jarvis_voice.scheduler_protocol import SchedulerEventFactory
from jarvis_voice.scheduler_runtime import SchedulerNativeRuntime
from jarvis_voice.transport import LoopbackWebSocketTransport


class CapturingRunner:
    def __init__(self, returncode: int = 0, stdout: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.calls: list[tuple[list[str], bytes, float]] = []

    def __call__(self, command, *, input, timeout):
        self.calls.append((list(command), bytes(input), float(timeout)))
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, b"")


class FakeAdapter:
    qualification_sha256 = "a" * 64

    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.removed: list[str] = []

    def post(self, **kwargs):
        self.posts.append(dict(kwargs))
        return NativeDispatchResult(True, "native_api_accepted")

    def remove(self, notification_id):
        self.removed.append(notification_id)
        return NativeDispatchResult(True, "native_api_accepted")


class UnknownLock:
    qualification_sha256 = "b" * 64

    def state(self):
        return "unknown"


class SchedulerNativeTests(unittest.TestCase):
    def test_transport_decodes_scheduler_protocol_in_its_own_namespace(self) -> None:
        session_id = str(uuid.uuid4())
        event = SchedulerEventFactory(session_id).create("scheduler.ready", {
            "protocol": "jarvis.scheduler/1",
            "maximum_frame_bytes": 16384,
            "sequence_namespace": "protocol_direction_connection",
        })

        class Connection:
            def recv(self, timeout=0):
                return event.to_json()

            def send(self, data):
                return None

            def close(self):
                return None

        transport = LoopbackWebSocketTransport(
            "ws://127.0.0.1:8765", connection_factory=lambda _url: Connection()
        )
        decoded = transport.receive()
        self.assertEqual(decoded, event)

    def test_macos_lock_probe_is_recursive_and_unknown_on_failure(self) -> None:
        data = plistlib.dumps([{"IORegistryEntryChildren": [{"CGSSessionScreenIsLocked": True}]}])
        self.assertEqual(MacOSLockStateProvider(runner=CapturingRunner(stdout=data)).state(), "locked")
        self.assertEqual(MacOSLockStateProvider(runner=CapturingRunner(returncode=1)).state(), "unknown")

    def test_native_adapters_keep_content_out_of_process_arguments(self) -> None:
        title = "Ιδιωτικός τίτλος"
        body = "Μυστικό σώμα υπενθύμισης"
        notification_id = "1" * 64
        for adapter_type in (MacOSNotificationAdapter, WindowsNotificationAdapter):
            with self.subTest(adapter=adapter_type.__name__):
                runner = CapturingRunner()
                result = adapter_type(runner=runner).post(
                    notification_id=notification_id,
                    title=title,
                    body=body,
                    silent=True,
                )
                self.assertTrue(result.accepted)
                command, stdin, _timeout = runner.calls[0]
                self.assertNotIn(title, " ".join(command))
                self.assertNotIn(body, " ".join(command))
                self.assertNotIn(title.encode(), b"\0".join(item.encode() for item in command))
                self.assertTrue(stdin)

    def test_native_adapters_preserve_silent_policy(self) -> None:
        mac_silent = CapturingRunner()
        MacOSNotificationAdapter(runner=mac_silent).post(
            notification_id="6" * 64, title="Title", body="Body", silent=True
        )
        self.assertIn(b"p.silent ? null", mac_silent.calls[0][1])

        windows_silent = CapturingRunner()
        WindowsNotificationAdapter(runner=windows_silent).post(
            notification_id="7" * 64, title="Title", body="Body", silent=True
        )
        windows_audible = CapturingRunner()
        WindowsNotificationAdapter(runner=windows_audible).post(
            notification_id="8" * 64, title="Title", body="Body", silent=False
        )
        def embedded_payload(runner):
            encoded = re.search(
                rb"FromBase64String\('([^']+)'\)", runner.calls[0][1]
            ).group(1)
            return json.loads(base64.b64decode(encoded))

        self.assertIs(embedded_payload(windows_silent)["silent"], True)
        self.assertIs(embedded_payload(windows_audible)["silent"], False)

    def test_windows_lock_probe_treats_only_access_denied_as_locked(self) -> None:
        class User32:
            def OpenInputDesktop(self, *_args):
                return 0

            def CloseDesktop(self, _desktop):
                raise AssertionError("no desktop should be closed")

        self.assertEqual(
            WindowsLockStateProvider(user32=User32(), get_last_error=lambda: 5).state(),
            "locked",
        )
        self.assertEqual(
            WindowsLockStateProvider(user32=User32(), get_last_error=lambda: 6).state(),
            "unknown",
        )

    def test_unknown_lock_state_forces_private_presentation(self) -> None:
        adapter = FakeAdapter()
        delivery = QualifiedNativeSchedulerDelivery(adapter=adapter, lock_state=UnknownLock())
        result = delivery.post({
            "notification_id": "2" * 64,
            "title": "Sensitive title",
            "body": "Sensitive body",
            "private_title": "JARVIS",
            "private_body": "You have a reminder.",
            "visibility": "full",
            "allow_full_body_when_locked": False,
            "silent": False,
        })
        self.assertTrue(result.accepted)
        self.assertEqual(adapter.posts[0]["title"], "JARVIS")
        self.assertEqual(adapter.posts[0]["body"], "You have a reminder.")

    def test_native_qualification_is_digest_and_compatibility_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.json"
            value = {
                "schema_version": "scheduler-native-qualification/1",
                "platform": "macos",
                "ready": True,
                "permission": "granted",
                "capabilities": ["post", "replace", "remove", "silent", "lock_state"],
                "notification_adapter_sha256": MACOS_NOTIFICATION_ADAPTER_SHA256,
                "lock_state_provider_sha256": MACOS_LOCK_PROVIDER_SHA256,
                "checks": {name: "pass" for name in (
                    "permission_probe", "post", "replace", "remove", "silent",
                    "callback_idempotency", "locked_privacy", "unknown_privacy",
                    "restart_recovery",
                )},
                "qualified_at_utc": "2026-08-11T12:00:00.000Z",
                "artifact_signature_sha256": "c" * 64,
            }
            raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            compatibility = SchedulerCompatibilityManifest(
                sha256="d" * 64,
                capabilities=("jarvis.scheduler/1", "jarvis.approval-arbiter/1"),
                value={
                    "notification_adapter_qualification_sha256": digest,
                    "lock_state_provider_sha256": MACOS_LOCK_PROVIDER_SHA256,
                    "artifact_signature_sha256": "c" * 64,
                },
            )
            loaded = load_native_scheduler_qualification(
                path, digest, platform_name="darwin", compatibility=compatibility
            )
            self.assertEqual(loaded.notification_adapter_sha256, MACOS_NOTIFICATION_ADAPTER_SHA256)
            value["checks"]["remove"] = "blocked"
            path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
            changed = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "checks invalid"):
                load_native_scheduler_qualification(
                    path, changed, platform_name="darwin", compatibility=SchedulerCompatibilityManifest(
                        sha256="d" * 64,
                        capabilities=("jarvis.scheduler/1", "jarvis.approval-arbiter/1"),
                        value={
                            "notification_adapter_qualification_sha256": changed,
                            "lock_state_provider_sha256": MACOS_LOCK_PROVIDER_SHA256,
                            "artifact_signature_sha256": "c" * 64,
                        },
                    )
                )

    def test_runtime_acks_staged_content_before_native_dispatch_and_removes_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal_path = Path(directory) / "scheduler.sqlite3"
            journal = SchedulerClientJournal(
                journal_path,
                cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            )
            session_id = str(uuid.uuid4())
            controller = SchedulerClientController(
                session_id=session_id,
                journal=journal,
                approval_arbiter=GlobalApprovalController(),
            )
            adapter = FakeAdapter()
            runtime = SchedulerNativeRuntime(
                controller=controller,
                delivery=QualifiedNativeSchedulerDelivery(
                    adapter=adapter, lock_state=UnknownLock()
                ),
                capabilities=("jarvis.scheduler/1", "jarvis.approval-arbiter/1"),
                manifest_sha256="d" * 64,
            )
            factory = SchedulerEventFactory(session_id)
            sent = []
            runtime.handle_server_event(
                factory.create("scheduler.ready", {
                    "protocol": "jarvis.scheduler/1",
                    "maximum_frame_bytes": 16384,
                    "sequence_namespace": "protocol_direction_connection",
                }),
                key_counter=0,
                both_keys_up=True,
                send=sent.append,
            )
            notification_id = "3" * 64
            occurrence_id = "4" * 64
            requested = factory.create("scheduler.notification.requested", {
                "notification_id": notification_id,
                "occurrence_id": occurrence_id,
                "job_id": str(uuid.uuid4()),
                "job_revision": 1,
                "delivery_generation": 0,
                "scheduled_at_utc": "2026-08-11T12:00:00.000Z",
                "effective_due_at_utc": "2026-08-11T12:00:00.000Z",
                "expires_at_utc": "2026-08-12T12:00:00.000Z",
                "content_sha256": "5" * 64,
                "title": "Sensitive title",
                "body": "Sensitive body",
                "private_title": "JARVIS",
                "private_body": "You have a reminder.",
                "locale": "en-US",
                "visibility": "full",
                "allow_full_body_when_locked": False,
                "silent": False,
            })
            runtime.handle_server_event(
                requested,
                key_counter=0,
                both_keys_up=True,
                send=sent.append,
            )
            self.assertEqual(
                [event.type for event in sent],
                ["scheduler.event_ack", "scheduler.notification.receipt", "scheduler.notification.receipt"],
            )
            self.assertEqual(sent[1].payload["stage"], "DISPATCHING")
            self.assertEqual(sent[2].payload["stage"], "OS_ACCEPTED")
            self.assertEqual(adapter.posts[0]["body"], "You have a reminder.")
            journal._db.execute("PRAGMA wal_checkpoint(FULL)")
            self.assertNotIn(b"Sensitive body", journal_path.read_bytes())

            tombstone = factory.create("scheduler.notification.tombstone", {
                "notification_id": notification_id,
                "occurrence_id": occurrence_id,
                "reason_code": "job_cancelled",
                "created_at_utc": "2026-08-11T12:01:00.000Z",
            })
            runtime.handle_server_event(
                tombstone,
                key_counter=0,
                both_keys_up=True,
                send=sent.append,
            )
            self.assertEqual(adapter.removed, [notification_id])
            runtime.close()

    def test_scheduler_config_is_disabled_by_default_and_requires_full_binding(self) -> None:
        config = voice_config_from_mapping({})
        self.assertFalse(config.scheduler.enabled)
        with self.assertRaisesRegex(Exception, "qualification_required"):
            voice_config_from_mapping({"scheduler": {"enabled": True}})


if __name__ == "__main__":
    unittest.main()
