from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jarvis_voice.scheduler_callback_qualification import (
    REQUIRED_CHECKS,
    load_scheduler_callback_qualification,
)


class SchedulerCallbackQualificationTests(unittest.TestCase):
    def test_exact_pass_manifest_loads_and_failures_are_rejected(self) -> None:
        value = {
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(raw)
            loaded = load_scheduler_callback_qualification(
                path, hashlib.sha256(raw).hexdigest(), platform_name="darwin"
            )
            self.assertEqual(loaded.helper_sha256, "1" * 64)
            value["checks"]["cancel_callback"] = "fail"
            changed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "checks"):
                load_scheduler_callback_qualification(
                    path, hashlib.sha256(changed).hexdigest(), platform_name="darwin"
                )


if __name__ == "__main__":
    unittest.main()
