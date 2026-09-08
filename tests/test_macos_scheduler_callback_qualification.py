from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_voice.scheduler_callback import sign_callback
from jarvis_voice.scheduler_callback_qualification import load_scheduler_callback_qualification
from jarvis_voice.scheduler_native import NativeDispatchResult


def qualification_module():
    path = Path(__file__).parents[1] / "scripts" / "run_macos_scheduler_callback_qualification.py"
    spec = importlib.util.spec_from_file_location("scheduler_callback_qualification_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SchedulerCallbackRunnerTests(unittest.TestCase):
    def test_invalid_deadline_is_rejected_before_artifact_or_native_access(self):
        module = qualification_module()
        for value in ("nan", "inf", "-inf", "0", "-1"):
            argv = ["qualify", "--helper", "/unused/helper", "--artifact-manifest", "/unused/manifest",
                    "--output", "/unused/output", "--callback-timeout-seconds=" + value]
            with self.subTest(value=value), patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "timeout"):
                    module.main()

    def test_invalid_artifact_is_rejected_before_native_access(self):
        module = qualification_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "helper"
            helper.write_bytes(b"x")
            manifest = root / "artifact.json"
            value = {"schema_version": "jarvis-macos-helper-artifact/1", "binary_sha256": "a" * 64,
                     "code_identity_sha256": "b" * 64, "size_bytes": 1}
            valid = json.dumps(value)
            invalid = (
                json.dumps({**value, "size_bytes": True}),
                json.dumps({**value, "size_bytes": 1.0}),
                valid[:-1] + ', "size_bytes":1}',
                " " * (1024 * 1024) + valid,
                '[' * 1500 + '0' + ']' * 1500,
            )
            argv = ["qualify", "--helper", str(helper), "--artifact-manifest", str(manifest),
                    "--output", str(root / "result.json")]
            for raw in invalid:
                manifest.write_text(raw)
                with self.subTest(length=len(raw)), patch("sys.argv", argv), patch.object(
                    module, "MacOSSchedulerHostAdapter", side_effect=AssertionError("native access")
                ):
                    with self.assertRaises(ValueError):
                        module.main()

    def test_simulated_full_run_requires_visible_checks_and_emits_loadable_report(self):
        module = qualification_module()

        class Adapter:
            callback_key = bytes(range(32))
            closed = False

            def __init__(self):
                self.callbacks = []

            def post(self, *, notification_id, title, body, silent):
                if body.startswith("Click"):
                    snooze = "Snooze" in body
                    self.callbacks.append(sign_callback({
                        "callback_id": str(uuid.uuid4()), "notification_id": notification_id,
                        "action": "SNOOZE" if snooze else "CANCEL",
                        "requested_due_at_utc": "2030-01-10T12:10:00.000Z" if snooze else None,
                        "occurred_at_utc": "2030-01-10T12:00:00.000Z",
                    }, key=self.callback_key))
                return NativeDispatchResult(True, "native_api_accepted")

            def remove(self, _notification_id):
                return NativeDispatchResult(True, "removed")

            def drain_callbacks(self):
                batch, self.callbacks = self.callbacks, []
                return batch

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "helper"
            helper.write_bytes(b"fake-helper")
            manifest = root / "artifact.json"
            manifest.write_text(json.dumps({
                "schema_version": "jarvis-macos-helper-artifact/1",
                "binary_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
                "code_identity_sha256": "b" * 64, "size_bytes": helper.stat().st_size,
            }))
            output = root / "report.json"
            argv = ["qualify", "--helper", str(helper), "--artifact-manifest", str(manifest),
                    "--output", str(output)]
            adapter = Adapter()
            with patch("sys.argv", argv), patch.object(module, "MacOSSchedulerHostAdapter", return_value=adapter), patch.object(
                module, "_confirm"
            ) as confirm, patch("builtins.print"):
                self.assertEqual(module.main(), 0)
            self.assertTrue(adapter.closed)
            self.assertEqual(confirm.call_count, 4)
            report = load_scheduler_callback_qualification(
                output, hashlib.sha256(output.read_bytes()).hexdigest(), platform_name="darwin",
            )
            self.assertEqual(report.value["artifact_manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            output.unlink()
            adapter = Adapter()
            with patch("sys.argv", argv), patch.object(module, "MacOSSchedulerHostAdapter", return_value=adapter), patch.object(
                module, "_confirm", side_effect=RuntimeError("not observed")
            ):
                with self.assertRaisesRegex(RuntimeError, "not observed"):
                    module.main()
            self.assertTrue(adapter.closed)
            self.assertFalse(output.exists())

    def test_replay_checks_payload_and_closes_journal_on_failure(self):
        module = qualification_module()
        key = bytes(range(32))
        raw = sign_callback({"callback_id": str(uuid.uuid4()), "notification_id": "a" * 64,
                             "action": "CANCEL", "requested_due_at_utc": None,
                             "occurred_at_utc": "2030-01-10T12:00:00.000Z"}, key=key)
        original = module.SchedulerNativeRuntime.pump_native_callbacks
        original_close = module.SchedulerClientJournal.close
        journals = []
        sessions = []

        def corrupted_replay(runtime):
            events = original(runtime)
            sessions.append(runtime.controller.factory.session_id)
            if len(sessions) > 1 and events:
                events[0].payload["job_revision"] += 1
            return events

        def close(journal):
            journals.append(journal)
            original_close(journal)

        with patch.object(module.SchedulerNativeRuntime, "pump_native_callbacks", corrupted_replay), patch.object(
            module.SchedulerClientJournal, "close", close
        ):
            with self.assertRaisesRegex(RuntimeError, "reconnect"):
                module._verify_reconnect_replay(raw, key=key, notification_id="a" * 64)
        self.assertEqual(len(journals), 2)
        self.assertGreater(len(set(sessions)), 1)
        module._verify_reconnect_replay(raw, key=key, notification_id="a" * 64)


if __name__ == "__main__":
    unittest.main()
