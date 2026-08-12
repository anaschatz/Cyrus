from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_orchestrator.stack_config import (
    STACK_SCHEMA_VERSION,
    PrivateStackConfig,
    StackConfigError,
    write_private_stack_config,
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class PrivateStackConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.voice = self.private_file(
            "voice.json", {"orchestrator_url": "ws://127.0.0.1:8765"}
        )
        self.orchestrator = self.private_file(
            "orchestrator.json",
            {
                "database_path": str(self.root / "state.sqlite3"),
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
            },
        )
        self.runner = self.root / "runner.py"
        self.runner.write_text("pass\n", encoding="utf-8")
        self.runner.chmod(0o700)
        self.manifest_document = {
            "alias": "jarvis-local",
            "manifest_schema_version": 1,
        }
        self.manifest = self.private_file("manifest.json", self.manifest_document)
        self.report = self.private_file(
            "report.json",
            {
                "status": "pass",
                "identity": {
                    "manifest_digest": hashlib.sha256(
                        canonical(self.manifest_document)
                    ).hexdigest()
                },
            },
        )
        self.stack_path = self.root / "jarvis.private.json"

    def private_file(self, name: str, value: object) -> Path:
        path = self.root / name
        path.write_bytes(canonical(value) + b"\n")
        path.chmod(0o600)
        return path

    def document(self) -> dict[str, object]:
        return {
            "schema_version": STACK_SCHEMA_VERSION,
            "voice_config_path": str(self.voice),
            "orchestrator_config_path": str(self.orchestrator),
            "inference": {
                "runner_path": str(self.runner),
                "manifest_path": str(self.manifest),
                "qualification_report_path": str(self.report),
            },
        }

    def test_write_load_and_hash_are_deterministic_and_private(self) -> None:
        write_private_stack_config(self.stack_path, self.document())
        first = PrivateStackConfig.load(self.stack_path)
        second = PrivateStackConfig.load(self.stack_path)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.voice_config_path, self.voice.resolve())
        self.assertEqual(first.orchestrator_config_path, self.orchestrator.resolve())
        self.assertEqual(first.inference.manifest_path, self.manifest.resolve())
        if os.name != "nt":
            self.assertEqual(self.stack_path.stat().st_mode & 0o777, 0o600)

    def test_endpoint_mismatch_fails_closed(self) -> None:
        self.voice.write_bytes(
            canonical({"orchestrator_url": "ws://127.0.0.1:9999"}) + b"\n"
        )
        self.voice.chmod(0o600)
        self.private_file("jarvis.private.json", self.document())
        with self.assertRaisesRegex(StackConfigError, "voice_orchestrator_mismatch"):
            PrivateStackConfig.load(self.stack_path)

    def test_qualification_digest_mismatch_fails_closed(self) -> None:
        self.report.write_bytes(
            canonical(
                {
                    "status": "pass",
                    "identity": {"manifest_digest": "0" * 64},
                }
            )
            + b"\n"
        )
        self.report.chmod(0o600)
        self.private_file("jarvis.private.json", self.document())
        with self.assertRaisesRegex(
            StackConfigError, "inference_qualification_mismatch"
        ):
            PrivateStackConfig.load(self.stack_path)

    def test_duplicate_json_key_is_rejected(self) -> None:
        self.stack_path.write_text(
            '{"schema_version":"jarvis-private-stack/1",'
            '"schema_version":"jarvis-private-stack/1"}',
            encoding="utf-8",
        )
        self.stack_path.chmod(0o600)
        with self.assertRaisesRegex(StackConfigError, "stack_json_invalid"):
            PrivateStackConfig.load(self.stack_path)


if __name__ == "__main__":
    unittest.main()
