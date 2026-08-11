from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.jcs import jcs_bytes
from jarvis_orchestrator.macos_action_adapters import (
    MacOSActionAdapterConfig,
    MacOSAdapterError,
    MacOSNativeAdapterV1,
)
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider
from jarvis_orchestrator.tool_contracts import ToolRegistryV1


class MacOSActionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        self.documents = self.root / "documents"
        self.documents.mkdir()
        self.helper = self.root / "jarvis-action-native"
        self.helper.write_bytes(b"binary-placeholder")
        os.chmod(self.helper, 0o700)
        self.registry = ToolRegistryV1.build(
            root_ids=("documents",),
            app_ids=("calendar", "finder", "mail", "preview", "reminders", "textedit"),
        )
        self.keys = InMemorySchedulerKeyProvider()
        self.adapter = MacOSNativeAdapterV1(
            config=MacOSActionAdapterConfig(
                helper_path=self.helper,
                helper_sha256=__import__("hashlib").sha256(self.helper.read_bytes()).hexdigest(),
                helper_code_identity_sha256="0" * 64,
                roots={"documents": str(self.documents)},
                app_ids=("finder",),
                state_path=self.root / "refs.sqlite3",
                notes_path=self.root / "notes",
                require_code_signature=False,
            ),
            registry=self.registry,
            auth_keys=self.keys,
        )

    def tearDown(self) -> None:
        self.adapter.close()
        self.temp.cleanup()

    def deadline(self) -> str:
        return (datetime.now(UTC) + timedelta(minutes=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def test_private_note_is_exactly_read_back_and_not_plaintext_at_rest(self) -> None:
        body = "πολύ ιδιωτικό σώμα"
        result = self.adapter.execute(
            tool_id="note.capture@1",
            arguments={"title": "Τίτλος", "body": body},
            execution_id="ignored",
            deadline_utc=self.deadline(),
        )
        self.assertEqual(result.status.value, "VERIFIED")
        self.assertEqual(result.result_payload["storage_domain"], "JARVIS_PRIVATE_NOTES")
        self.adapter.references._db.execute("PRAGMA wal_checkpoint(FULL)")
        self.assertNotIn(body.encode(), (self.root / "refs.sqlite3").read_bytes())

    def test_native_helper_receives_sensitive_content_only_on_stdin(self) -> None:
        response = {
            "status": "VERIFIED",
            "verification": "EXACT_READBACK",
            "reason_code": "ok",
            "result_payload": {
                "app_id": "finder",
                "bundle_id": "com.apple.finder",
                "request_observed_at_utc": "2026-08-11T18:00:00.000Z",
            },
            "reference_bindings": [],
            "os_receipt_sha256": "f" * 64,
        }
        completed = subprocess.CompletedProcess(
            [str(self.helper)], 0, stdout=jcs_bytes(response), stderr=b""
        )
        with patch(
            "jarvis_orchestrator.macos_action_adapters.subprocess.run",
            return_value=completed,
        ) as runner:
            result = self.adapter.execute(
                tool_id="app.open@1",
                arguments={"app_id": "finder"},
                execution_id="ignored",
                deadline_utc=self.deadline(),
            )
        self.assertEqual(result.result_payload["bundle_id"], "com.apple.finder")
        args, kwargs = runner.call_args
        self.assertEqual(args[0], [str(self.helper)])
        self.assertNotIn(b"finder", b"\0".join(item.encode() for item in args[0]))
        self.assertIn(b"finder", kwargs["input"])
        self.assertIs(kwargs["shell"], False)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)

    def test_adapter_revalidates_native_output_schema(self) -> None:
        response = {
            "status": "VERIFIED",
            "verification": "EXACT_READBACK",
            "reason_code": "ok",
            "result_payload": {
                "app_id": "finder",
                "bundle_id": "com.apple.finder",
                "request_observed_at_utc": "2026-08-11T18:00:00.000Z",
                "unexpected": True,
            },
            "reference_bindings": [],
            "os_receipt_sha256": "f" * 64,
        }
        completed = subprocess.CompletedProcess(
            [str(self.helper)], 0, stdout=jcs_bytes(response), stderr=b""
        )
        with patch(
            "jarvis_orchestrator.macos_action_adapters.subprocess.run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(Exception, "tool.output_invalid"):
                self.adapter.execute(
                    tool_id="app.open@1",
                    arguments={"app_id": "finder"},
                    execution_id="ignored",
                    deadline_utc=self.deadline(),
                )

    def test_helper_symlink_is_rejected(self) -> None:
        linked = self.root / "linked-helper"
        linked.symlink_to(self.helper)
        config = MacOSActionAdapterConfig(
            helper_path=linked,
            helper_sha256=__import__("hashlib").sha256(self.helper.read_bytes()).hexdigest(),
            helper_code_identity_sha256="0" * 64,
            roots={"documents": str(self.documents)},
            app_ids=("finder",),
            state_path=self.root / "other.sqlite3",
            notes_path=self.root / "notes2",
            require_code_signature=False,
        )
        with self.assertRaises(MacOSAdapterError):
            config.validate()


if __name__ == "__main__":
    unittest.main()
