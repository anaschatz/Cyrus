from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import sys
import textwrap
import tempfile
import threading
import unittest
import uuid
import plistlib
import re
import os
import queue
import sqlite3
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.macos_code_identity import code_identity_sha256
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
    MacOSSchedulerHostAdapter,
    NativeDispatchResult,
    QualifiedNativeSchedulerDelivery,
    WindowsLockStateProvider,
    WindowsNotificationAdapter,
)
from jarvis_voice.scheduler_native_qualification import load_native_scheduler_qualification
from jarvis_voice.scheduler_protocol import SchedulerEventFactory
from jarvis_voice.scheduler_runtime import SchedulerNativeRuntime
from jarvis_voice.transport import LoopbackWebSocketTransport
from jarvis_voice.protocol import EventFactory
from jarvis_voice.voice_auth import VoiceAuthKey, create_server_challenge, create_server_ready


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
    @unittest.skipUnless(os.name == "posix", "POSIX helper pipes")
    def test_native_command_timeout_covers_a_full_input_pipe(self) -> None:
        script = textwrap.dedent("""
            import os, queue, threading, time
            from types import SimpleNamespace
            from jarvis_voice.scheduler_native import MacOSSchedulerHostAdapter
            reader, writer = os.pipe()
            os.set_blocking(writer, False)
            try:
                while True:
                    os.write(writer, b'x' * 4096)
            except BlockingIOError:
                pass
            os.set_blocking(writer, True)
            stream = os.fdopen(writer, 'wb', buffering=0)
            adapter = MacOSSchedulerHostAdapter.__new__(MacOSSchedulerHostAdapter)
            adapter._process = SimpleNamespace(stdin=stream, poll=lambda: None)
            adapter._timeout = 0.03
            adapter._closed = False
            adapter._command_lock = threading.Lock()
            adapter._receipts = queue.Queue(maxsize=64)
            started = time.monotonic()
            try:
                result = adapter.remove('a' * 64)
                assert result.reason_code == 'native_timeout', result
                assert result.ambiguous
                assert time.monotonic() - started < 0.5
                assert adapter._closed
            finally:
                stream.close()
                os.close(reader)
        """)
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=2)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_native_reader_discards_an_entire_oversized_line(self) -> None:
        adapter = MacOSSchedulerHostAdapter.__new__(MacOSSchedulerHostAdapter)
        adapter._closed = False
        adapter._receipts = queue.Queue()
        adapter._callbacks = queue.Queue()
        oversized_tail = {"type": "command.receipt", "request_id": "oversized"}
        valid = {"type": "command.receipt", "request_id": "valid"}
        adapter._process = SimpleNamespace(stdout=io.BytesIO(
            b' ' * 4097 + json.dumps(oversized_tail).encode() + b'\n'
            + json.dumps(valid).encode() + b'\n',
        ))
        adapter._read_output()
        self.assertEqual(adapter._receipts.get_nowait(), valid)
        self.assertTrue(adapter._receipts.empty())

    def test_native_timeout_rejects_nonfinite_or_invalid_values_before_startup(self) -> None:
        for timeout in (0, -1, True, "5", float("nan"), float("inf"), 10**1000):
            with self.subTest(timeout=timeout), patch.object(MacOSSchedulerHostAdapter, "_validate_helper") as validate:
                with self.assertRaisesRegex(ValueError, "timeout invalid"):
                    MacOSSchedulerHostAdapter(
                        helper_path="/unused/helper", helper_sha256="a" * 64,
                        helper_code_identity_sha256="b" * 64, timeout_seconds=timeout,
                    )
                validate.assert_not_called()

    def test_native_callback_burst_waits_for_capacity_without_losing_actions(self) -> None:
        from jarvis_voice.scheduler_callback import sign_callback
        overflow = threading.Event()

        class ObservedQueue(queue.Queue):
            def put(self, item, block=True, timeout=None):
                try:
                    super().put(item, block, timeout)
                except queue.Full:
                    overflow.set()
                    raise

        frames = [sign_callback({
            "callback_id": str(uuid.uuid4()),
            "notification_id": hashlib.sha256(str(index).encode()).hexdigest(),
            "action": "CANCEL", "requested_due_at_utc": None,
            "occurred_at_utc": "2030-01-10T12:00:00.000Z",
        }, key=bytes(range(32))) for index in range(65)]
        adapter = MacOSSchedulerHostAdapter.__new__(MacOSSchedulerHostAdapter)
        adapter._closed = False
        adapter._receipts = queue.Queue(maxsize=64)
        adapter._callbacks = ObservedQueue(maxsize=64)
        adapter._process = SimpleNamespace(stdout=io.BytesIO(b"\n".join(frames) + b"\n"))
        reader = threading.Thread(target=adapter._read_output, daemon=True)
        reader.start()
        try:
            self.assertTrue(overflow.wait(timeout=2))
            received = adapter.drain_callbacks()
            self.assertLessEqual(len(received), 64)
            reader.join(timeout=2)
            self.assertFalse(reader.is_alive())
            received.extend(adapter.drain_callbacks())
            self.assertEqual(received, frames)
        finally:
            adapter._closed = True
            adapter.drain_callbacks()
            reader.join(timeout=2)

    def test_native_reader_survives_deeply_nested_output(self) -> None:
        receipt = {"type": "command.receipt", "request_id": str(uuid.uuid4()),
                   "accepted": True, "ambiguous": False, "reason_code": "initialized"}
        adapter = MacOSSchedulerHostAdapter.__new__(MacOSSchedulerHostAdapter)
        adapter._closed = False
        adapter._receipts = queue.Queue()
        adapter._callbacks = queue.Queue()
        adapter._process = SimpleNamespace(stdout=io.BytesIO(
            b'[' * 1500 + b'0' + b']' * 1500 + b'\n' + json.dumps(receipt).encode() + b'\n',
        ))
        adapter._read_output()
        self.assertEqual(adapter._receipts.get_nowait(), receipt)

    def test_native_close_reaps_killed_helper_and_closes_output(self) -> None:
        process = Mock(stdin=io.BytesIO(), stdout=io.BytesIO())
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("helper", 2), 0]
        adapter = MacOSSchedulerHostAdapter.__new__(MacOSSchedulerHostAdapter)
        adapter._process = process
        adapter._reader = Mock()
        adapter.close()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        adapter._reader.join.assert_called_once()

    def test_native_initialization_failure_closes_spawned_helper(self) -> None:
        process = Mock(stdin=io.BytesIO(), stdout=io.BytesIO())
        process.poll.return_value = None
        with (
            patch.object(MacOSSchedulerHostAdapter, "_validate_helper"),
            patch("jarvis_voice.scheduler_native.threading.Thread.start", side_effect=RuntimeError("reader unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "reader unavailable"):
                MacOSSchedulerHostAdapter(
                    helper_path="/unused/helper", helper_sha256="a" * 64,
                    helper_code_identity_sha256="b" * 64, popen=lambda *_args, **_kwargs: process,
                )
        process.terminate.assert_called_once()
        process.wait.assert_called_once()
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)

    def test_native_binding_failure_closes_constructed_adapter(self) -> None:
        from jarvis_voice.scheduler_native import load_qualified_native_delivery
        adapter = Mock(qualification_sha256="a" * 64)
        config = SimpleNamespace(
            compatibility_manifest_path="unused", compatibility_manifest_sha256="b" * 64,
            native_qualification_path="unused", native_qualification_sha256="c" * 64,
            callback_qualification_path=None,
        )
        with (
            patch("jarvis_voice.scheduler_compatibility.load_scheduler_compatibility",
                  return_value=SimpleNamespace(capabilities=())),
            patch("jarvis_voice.scheduler_native_qualification.load_native_scheduler_qualification",
                  return_value=SimpleNamespace(notification_adapter_sha256="d" * 64)),
            patch("jarvis_voice.scheduler_native.platform_native_components",
                  return_value=(adapter, UnknownLock())),
        ):
            with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
                load_qualified_native_delivery(config, platform_name="darwin")
        adapter.close.assert_called_once()

    def test_windows_lock_provider_declares_pointer_and_error_contract(self) -> None:
        import ctypes
        from ctypes import wintypes
        user32 = SimpleNamespace(OpenInputDesktop=Mock(return_value=2**40 + 1),
                                 CloseDesktop=Mock(return_value=True))
        with (
            patch.object(ctypes, "windll", SimpleNamespace(user32=user32), create=True),
            patch.object(ctypes, "WinDLL", return_value=user32, create=True) as load,
            patch.object(ctypes, "get_last_error", return_value=5, create=True),
        ):
            provider = WindowsLockStateProvider()
            self.assertEqual(provider.state(), "unlocked")
            user32.CloseDesktop.assert_called_once_with(2**40 + 1)
            load.assert_called_once_with("user32", use_last_error=True)
            self.assertIs(user32.OpenInputDesktop.restype, wintypes.HANDLE)
            self.assertEqual(user32.CloseDesktop.argtypes, [wintypes.HANDLE])
            user32.OpenInputDesktop.return_value = None
            self.assertEqual(provider.state(), "locked")

    def test_scheduler_journal_closes_connection_after_schema_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = sqlite3.connect(":memory:")
            self.addCleanup(connection.close)
            with (
                patch("jarvis_voice.scheduler_journal.sqlite3.connect", return_value=connection),
                patch("jarvis_voice.scheduler_journal.SCHEMA", "INVALID SQL"),
            ):
                with self.assertRaises(sqlite3.DatabaseError):
                    SchedulerClientJournal(Path(directory) / "scheduler.sqlite3", cipher=None)
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_scheduler_releases_native_and_journal_resources_on_failures(self) -> None:
        for failure in ("journal", "controller", "close"):
            with self.subTest(failure=failure):
                adapter, journal = Mock(), Mock()
                delivery = SimpleNamespace(adapter=adapter)
                if failure == "close":
                    adapter.close.side_effect = RuntimeError("close failed")
                    runtime = SchedulerNativeRuntime(
                        controller=SimpleNamespace(journal=journal), delivery=delivery,
                        capabilities=(), manifest_sha256="a" * 64,
                    )
                    with self.assertRaisesRegex(RuntimeError, "close failed"):
                        runtime.close()
                else:
                    with (
                        patch("jarvis_voice.scheduler_runtime.load_qualified_native_delivery", return_value=(
                            SimpleNamespace(capabilities=(), sha256="a" * 64), None, delivery, None,
                        )),
                        patch("jarvis_orchestrator.scheduler_crypto.platform_key_provider"),
                        patch("jarvis_orchestrator.scheduler_crypto.SchedulerContentCipher"),
                        patch("jarvis_voice.scheduler_runtime.prepare_voice_journal_path", return_value=Path("/unused/scheduler.sqlite3")),
                        patch("jarvis_voice.scheduler_runtime.SchedulerClientJournal", return_value=journal) as create_journal,
                        patch("jarvis_voice.scheduler_runtime.SchedulerClientController") as create_controller,
                    ):
                        (create_journal if failure == "journal" else create_controller).side_effect = RuntimeError("startup failed")
                        with self.assertRaisesRegex(RuntimeError, "startup failed"):
                            SchedulerNativeRuntime.from_config(
                                SimpleNamespace(), session_id=str(uuid.uuid4()),
                                approval_arbiter=GlobalApprovalController(),
                            )
                adapter.close.assert_called_once()
                if failure != "journal":
                    journal.close.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "POSIX helper pipes")
    def test_signed_macos_scheduler_host_initializes_and_keeps_content_off_argv(self) -> None:
        helper_script = textwrap.dedent("""
            import json, sys
            for line in sys.stdin:
                message = json.loads(line)
                reason = "initialized" if message["type"] == "initialize" else "native_api_accepted"
                if message.get("body") == "invalid-receipt":
                    reason = {"private": "invalid receipt content"}
                print(json.dumps({
                    "type": "command.receipt", "request_id": message["request_id"],
                    "accepted": True, "ambiguous": False, "reason_code": reason,
                }), flush=True)
        """)
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "jarvis-scheduler-native"
            helper.write_bytes(b"signed-helper")
            helper.chmod(0o700)
            requirements = b"designated => identifier com.jarvis.voice\n"
            signatures, observed, written = [], [], []

            def run(command, **kwargs):
                signatures.append(kwargs)
                output = requirements if "--requirements" in command else b""
                return subprocess.CompletedProcess(command, 0, output, b"")

            def popen(command, **kwargs):
                observed.append(command)
                return subprocess.Popen([sys.executable, "-u", "-c", helper_script], **kwargs)

            write = os.write

            def short_write(descriptor, pending):
                fragment = bytes(pending[:7])
                count = write(descriptor, fragment)
                written.append(fragment[:count])
                return count

            with patch("jarvis_voice.scheduler_native.os.write", side_effect=short_write), closing(
                MacOSSchedulerHostAdapter(
                    helper_path=os.fspath(helper),
                    helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                    helper_code_identity_sha256=code_identity_sha256(requirements),
                    popen=popen, run=run,
                )
            ) as adapter:
                title, body = "Ιδιωτικός τίτλος", "Μυστικό σώμα" * 100
                with ThreadPoolExecutor(max_workers=2) as executor:
                    calls = [executor.submit(adapter.post, notification_id="f" * 64,
                                             title=title, body=body, silent=True) for _ in range(2)]
                    self.assertTrue(all(call.result(timeout=2).accepted for call in calls))
                invalid = adapter.post(notification_id="f" * 64, title=title,
                                       body="invalid-receipt", silent=True)
                self.assertEqual(invalid, NativeDispatchResult(False, "adapter_invalid_receipt", ambiguous=True))
                self.assertEqual(observed, [[os.fspath(helper)]])
                self.assertNotIn(title, observed[0])
                self.assertNotIn(body, observed[0])
                messages = [json.loads(line) for line in b"".join(written).splitlines()]
                self.assertEqual([message["body"] for message in messages[1:3]], [body, body])
                self.assertTrue(all(call["timeout"] == 5.0 for call in signatures))
                self.assertEqual(adapter._receipts.maxsize, 64)

    def test_transport_decodes_scheduler_protocol_in_its_own_namespace(self) -> None:
        session_id = str(uuid.uuid4())
        event = SchedulerEventFactory(session_id).create("scheduler.ready", {
            "protocol": "jarvis.scheduler/1",
            "maximum_frame_bytes": 16384,
            "sequence_namespace": "protocol_direction_connection",
        })

        class Connection:
            def __init__(self):
                self.key = VoiceAuthKey(1, bytes(range(32)))
                self.challenge = create_server_challenge(self.key).to_json()
                self.incoming = [self.challenge]
                self.sent = []

            def recv(self, timeout=0):
                del timeout
                return self.incoming.pop(0)

            def send(self, data):
                self.sent.append(data)
                if len(self.sent) == 2:
                    self.incoming.extend(
                        [
                            create_server_ready(
                                self.key,
                                self.challenge,
                                self.sent[0],
                                self.sent[1],
                            ).to_json(),
                            event.to_json(),
                        ]
                    )

            def close(self):
                return None

        connection = Connection()
        transport = LoopbackWebSocketTransport(
            "ws://127.0.0.1:8765",
            auth_key=connection.key,
            connection_factory=lambda _url: connection,
        )
        transport.send(
            EventFactory(session_id=session_id).create("client.hello", {})
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

        self.assertIn(b"$xml = [Windows.Data.Xml.Dom.XmlDocument]::new()", windows_silent.calls[0][1])
        self.assertIn(b"$xml.DocumentElement.AppendChild($audio)", windows_silent.calls[0][1])
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
            value["platform"] = ""
            raw = json.dumps(value).encode()
            path.write_bytes(raw)
            changed = hashlib.sha256(raw).hexdigest()
            with self.assertRaises(ValueError):
                load_native_scheduler_qualification(
                    path, changed, platform_name="linux", compatibility=SchedulerCompatibilityManifest(
                        sha256=compatibility.sha256, capabilities=compatibility.capabilities,
                        value={**compatibility.value, "notification_adapter_qualification_sha256": changed},
                    ),
                )
            value["platform"] = "macos"
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
                "expires_at_utc": (datetime.now(UTC) + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
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
