from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_voice.scheduler_callback_qualification import (
    REQUIRED_CHECKS,
    load_scheduler_callback_qualification,
)


class SchedulerCallbackQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.value = {
            "schema_version": "scheduler-native-callback-qualification/1",
            "platform": "macos",
            "ready": True,
            "native_helper_sha256": "1" * 64,
            "native_helper_code_identity_sha256": "2" * 64,
            "notification_adapter_sha256": "3" * 64,
            "checks": {name: "pass" for name in REQUIRED_CHECKS},
            "qualified_at_utc": "2026-08-11T12:00:00.000Z",
            "artifact_manifest_sha256": "4" * 64,
        }
    def test_exact_pass_manifest_loads_and_failures_are_rejected(self) -> None:
        value = self.value
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(raw)
            loaded = load_scheduler_callback_qualification(
                path, hashlib.sha256(raw).hexdigest(), platform_name="darwin"
            )
            self.assertEqual(loaded.helper_sha256, "1" * 64)
            value["platform"] = ""
            raw = json.dumps(value).encode()
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                load_scheduler_callback_qualification(path, hashlib.sha256(raw).hexdigest(), platform_name="linux")
            value["platform"] = "macos"
            value["checks"]["cancel_callback"] = "fail"
            changed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "checks"):
                load_scheduler_callback_qualification(
                    path, hashlib.sha256(changed).hexdigest(), platform_name="darwin"
                )


    def test_compatibility_manifest_rejects_wrong_scalar_and_capability_types(self) -> None:
        from jarvis_voice.scheduler_compatibility import load_scheduler_compatibility
        protocols = ["jarvis.scheduler/1", "jarvis.approval-arbiter/1"]
        value = {
            "schema_version": "voice-scheduler-compatibility/1",
            "supported_protocols": protocols, "capabilities": protocols,
            "sequence_namespace": "protocol_direction_connection", "cpython": "3.11.14",
            "unicode_db": "14.0.0", "aead_contract": "scheduler-content-aead/1",
            "key_provider_contract": "synthetic-test-provider/1",
            "scheduler_journal_schema": "voice-scheduler-journal/1",
            "scheduler_playback_isolation": True, "lock_state_provider_sha256": "1" * 64,
            "notification_adapter_qualification_sha256": "2" * 64,
            "tts_adapter_qualification_sha256": None, "maximum_frame_bytes": 16384,
            "artifact_signature_sha256": "3" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compatibility.json"
            raw = json.dumps(value).encode(); path.write_bytes(raw)
            self.assertEqual(load_scheduler_compatibility(path, hashlib.sha256(raw).hexdigest()).capabilities,
                             tuple(protocols))
            for field, invalid in (
                ("maximum_frame_bytes", 16384.0), ("key_provider_contract", []),
                ("capabilities", dict.fromkeys(protocols)), ("capabilities", None),
                ("capabilities", protocols + [protocols[0]]), ("capabilities", protocols + [{}]),
                ("lock_state_provider_sha256", None), ("artifact_signature_sha256", 123),
                ("notification_adapter_qualification_sha256", None),
                ("capabilities", protocols + ["scheduler_notification_speech"]),
            ):
                with self.subTest(field=field, invalid=invalid):
                    raw = json.dumps({**value, field: invalid}).encode(); path.write_bytes(raw)
                    with self.assertRaises(ValueError):
                        load_scheduler_compatibility(path, hashlib.sha256(raw).hexdigest())

    def test_callback_qualification_rejects_duplicate_check_keys(self) -> None:
        raw = json.dumps(self.value).replace('"ready": true', '"ready": false, "ready": true').encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "JSON"):
                load_scheduler_callback_qualification(
                    path, hashlib.sha256(raw).hexdigest(), platform_name="darwin",
                )

    @unittest.skipUnless(hasattr(__import__("os"), "mkfifo"), "POSIX FIFO")
    def test_pinned_document_rejects_fifo_without_waiting_for_writer(self) -> None:
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            os.mkfifo(path)
            # Isolate the potentially blocking open so a regression cannot hang the suite.
            script = (
                "from pathlib import Path; from jarvis_voice.scheduler_compatibility import load_pinned_scheduler_json; "
                "import sys; load_pinned_scheduler_json(Path(sys.argv[1]), 'a'*64)"
            )
            result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, timeout=3)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"must be a regular file", result.stderr)

    def test_all_scheduler_loaders_bound_reads_and_translate_nested_json(self) -> None:
        from jarvis_voice.scheduler_compatibility import load_scheduler_compatibility
        from jarvis_voice.scheduler_native_qualification import load_native_scheduler_qualification
        from jarvis_voice.scheduler_tts_qualification import load_scheduler_tts_qualification
        loaders = (
            lambda path, digest: load_scheduler_compatibility(path, digest),
            lambda path, digest: load_scheduler_callback_qualification(path, digest, platform_name="darwin"),
            lambda path, digest: load_native_scheduler_qualification(
                path, digest, platform_name="darwin", compatibility=SimpleNamespace(value={})),
            lambda path, digest: load_scheduler_tts_qualification(
                path, digest, platform_name="darwin", compatibility=SimpleNamespace(value={})),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            path.write_bytes(b'{}')
            for index, load in enumerate(loaders):
                with self.subTest(loader=index, missing_pin=True):
                    with self.assertRaisesRegex(ValueError, "digest"):
                        load(path, None)
            for raw, reason in ((b'[' * 1500 + b'0' + b']' * 1500, "JSON"),
                                (b'{"x": NaN}', "JSON"), (b'{"x": 1e999}', "JSON"),
                                (b'{"x": "\\ud800"}', "JSON"), (b'\xff', "JSON"),
                                (b' ' * (1024 * 1024) + b'{}', "size")):
                path.write_bytes(raw)
                for index, load in enumerate(loaders):
                    with self.subTest(loader=index, reason=reason), patch.object(
                        Path, "read_bytes", side_effect=AssertionError("unbounded read"),
                    ):
                        with self.assertRaisesRegex(ValueError, reason):
                            load(path, hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    unittest.main()
