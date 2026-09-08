from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.action_host_qualification import (
    NativeQualificationError,
    NativeReadinessV1,
    build_native_qualification,
    load_native_qualification,
    readiness_digest,
)
from jarvis_orchestrator.action_host_v2_protocol import ACTION_HOST_V2_CONTRACT_SHA256
from jarvis_orchestrator.jcs import domain_sha256, jcs_text
from jarvis_orchestrator.macos_code_identity import code_identity_sha256


def utc(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ActionHostQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2030, 1, 1, tzinfo=UTC)
        self.readiness = NativeReadinessV1.from_mapping(
            {
                "contract": "jarvis.action-host-native-readiness/1",
                "observed_at_utc": utc(self.now),
                "calendar": "FULL_ACCESS",
                "reminders": "FULL_ACCESS",
                "mail_automation": "AUTHORIZED",
                "spotlight": "READY",
                "workspace": "READY",
                "private_notes": "READY",
            }
        )
        self.tools = (
            "app.open@1",
            "calendar.create@1",
            "calendar.read@1",
            "email.draft@1",
            "file.open@1",
            "file.search@1",
            "note.capture@1",
            "reminder.complete@1",
            "reminder.create@1",
            "reminder.list@1",
        )

    def build(self):
        return build_native_qualification(
            generated_at_utc=utc(self.now),
            expires_at_utc=utc(self.now + timedelta(days=7)),
            os_build="24G90",
            registry_sha256="a" * 64,
            helper_binary_sha256="b" * 64,
            helper_code_identity_sha256="c" * 64,
            readiness=self.readiness,
            test_manifest_sha256="d" * 64,
            qualified_tools=self.tools,
        )

    def test_manifest_binds_protocol_helper_registry_readiness_and_tests(self) -> None:
        qualification = self.build()
        self.assertEqual(qualification.protocol_sha256, ACTION_HOST_V2_CONTRACT_SHA256)
        self.assertEqual(qualification.readiness_sha256, readiness_digest(self.readiness))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            path.write_text(jcs_text(qualification.to_dict()), encoding="utf-8")
            with patch("jarvis_orchestrator.action_host_qualification.platform.system", return_value="Darwin"), patch(
                "jarvis_orchestrator.action_host_qualification.platform.machine", return_value="arm64"
            ):
                loaded = load_native_qualification(
                    path,
                    expected_sha256=qualification.qualification_sha256,
                    registry_sha256="a" * 64,
                    helper_binary_sha256="b" * 64,
                    helper_code_identity_sha256="c" * 64,
                    expected_os_build="24G90",
                    now=self.now + timedelta(hours=1),
                )
        self.assertEqual(loaded, qualification)

    def test_expiry_identity_and_permission_drift_fail_closed(self) -> None:
        qualification = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            path.write_text(jcs_text(qualification.to_dict()), encoding="utf-8")
            with patch("jarvis_orchestrator.action_host_qualification.platform.system", return_value="Darwin"), patch(
                "jarvis_orchestrator.action_host_qualification.platform.machine", return_value="arm64"
            ):
                with self.assertRaises(NativeQualificationError):
                    load_native_qualification(
                        path,
                        expected_sha256=qualification.qualification_sha256,
                        registry_sha256="a" * 64,
                        helper_binary_sha256="f" * 64,
                        now=self.now,
                    )
                with self.assertRaisesRegex(NativeQualificationError, "native_qualification_expired"):
                    load_native_qualification(
                        path,
                        expected_sha256=qualification.qualification_sha256,
                        registry_sha256="a" * 64,
                        now=self.now + timedelta(days=8),
                    )
        revoked = NativeReadinessV1(
            self.readiness.observed_at_utc,
            "DENIED",
            self.readiness.reminders,
            self.readiness.mail_automation,
            self.readiness.spotlight,
            self.readiness.workspace,
            self.readiness.private_notes,
        )
        self.assertNotEqual(readiness_digest(revoked), qualification.readiness_sha256)
        self.assertFalse(revoked.supports("calendar.read@1"))

    def test_readiness_is_capability_specific_and_never_registers_a_tool(self) -> None:
        limited = NativeReadinessV1(
            self.readiness.observed_at_utc,
            "WRITE_ONLY",
            "NOT_DETERMINED",
            "DENIED",
            "UNAVAILABLE",
            "READY",
            "READY",
        )
        self.assertTrue(limited.supports("calendar.create@1"))
        self.assertFalse(limited.supports("calendar.read@1"))
        self.assertFalse(limited.supports("file.search@1"))
        self.assertTrue(limited.supports("note.capture@1"))
        self.assertFalse(limited.supports("host.readiness@1"))

    def test_code_identity_excludes_install_path_but_not_requirement(self) -> None:
        first = b'Executable=/tmp/build/helper\ndesignated => identifier "com.jarvis.helper" and anchor apple generic\n'
        installed = b'Executable=/Applications/JARVIS.app/Contents/Helpers/helper\ndesignated => identifier "com.jarvis.helper" and anchor apple generic\n'
        changed = b'Executable=/Applications/JARVIS.app/Contents/Helpers/helper\ndesignated => identifier "evil.helper" and anchor apple generic\n'
        self.assertEqual(code_identity_sha256(first), code_identity_sha256(installed))
        self.assertNotEqual(code_identity_sha256(first), code_identity_sha256(changed))



    def test_old_frame_limit_qualification_cannot_enable_new_contract(self):
        value = self.build().inputs | {"protocol_sha256":
            "eab93d5724d271ea33623f098b2abb8f6be7bf00256e3925aa7b89745621c3c4"}
        digest = domain_sha256(value["contract"], value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            path.write_text(jcs_text(value | {"qualification_sha256": digest}))
            with patch("jarvis_orchestrator.action_host_qualification.platform.system", return_value="Darwin"), patch(
                "jarvis_orchestrator.action_host_qualification.platform.machine", return_value="arm64"
            ), self.assertRaisesRegex(NativeQualificationError, "native_qualification_contract_mismatch"):
                load_native_qualification(path, expected_sha256=digest, registry_sha256="a" * 64, now=self.now)


    def test_readiness_rejects_invalid_states_and_unknown_reminder_tools(self):
        for field in ("calendar", "reminders", "mail_automation", "spotlight", "workspace", "private_notes"):
            for value in ([], {}):
                with self.subTest(field=field, value=value), self.assertRaises(NativeQualificationError):
                    NativeReadinessV1.from_mapping(self.readiness.to_dict() | {field: value})
        for tool in ("reminder.delete@1", "reminder.list@2", "reminder."):
            with self.subTest(tool=tool):
                self.assertFalse(self.readiness.supports(tool))

    def test_qualification_rejects_non_string_tools(self):
        for tools in ([None], [False], [["app.open@1"]], ["app.open@1", 1]):
            with self.subTest(tools=tools):
                value = self.build().inputs | {"qualified_tools": tools}
                digest = domain_sha256(value["contract"], value)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "qualification.json"
                    path.write_text(jcs_text(value | {"qualification_sha256": digest}))
                    with patch("jarvis_orchestrator.action_host_qualification.platform.system", return_value="Darwin"), patch(
                        "jarvis_orchestrator.action_host_qualification.platform.machine", return_value="arm64"
                    ), self.assertRaisesRegex(NativeQualificationError, "native_qualified_tools_invalid"):
                        load_native_qualification(path, expected_sha256=digest, registry_sha256="a" * 64, now=self.now)

    def test_qualification_fifo_is_rejected_without_waiting_for_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            os.mkfifo(path, 0o600)
            script = """import sys
from pathlib import Path
from jarvis_orchestrator.action_host_qualification import load_native_qualification, NativeQualificationError
try:
    load_native_qualification(Path(sys.argv[1]), expected_sha256='a'*64, registry_sha256='b'*64)
except NativeQualificationError:
    sys.exit(0)
sys.exit(1)
"""
            result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, timeout=1)
            self.assertEqual(result.returncode, 0, result.stderr.decode())


    def test_generated_qualification_is_loadable_without_rewriting_bytes(self):
        scripts = str(Path(__file__).resolve().parents[2] / "scripts")
        with patch.object(sys, "path", [scripts, *sys.path]):
            from generate_action_host_qualification import write_atomic
        qualification = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            write_atomic(path, qualification.to_dict())
            with patch("jarvis_orchestrator.action_host_qualification.platform.system", return_value="Darwin"), patch(
                "jarvis_orchestrator.action_host_qualification.platform.machine", return_value="arm64"
            ):
                loaded = load_native_qualification(path, expected_sha256=qualification.qualification_sha256,
                                                   registry_sha256="a" * 64, now=self.now)
            self.assertEqual(loaded, qualification)
            self.assertEqual(path.read_bytes(), jcs_text(qualification.to_dict()).encode())

    def test_native_test_manifest_rejects_malformed_json_and_digest_types(self):
        scripts = str(Path(__file__).resolve().parents[2] / "scripts")
        with patch.object(sys, "path", [scripts, *sys.path]):
            from generate_action_host_qualification import load_test_manifest
        import hashlib
        import json
        value = dict(contract="jarvis.action-host-native-tests/1", baseline_commit="a" * 40,
                     completed_at_utc=utc(self.now), results=[dict(tool_id=tool, status="PASS", evidence_sha256="b" * 64)
                                                            for tool in self.tools])
        digest = hashlib.sha256(jcs_text(value).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tests.json"
            path.write_text(json.dumps(value) + "\n")
            self.assertEqual(load_test_manifest(path, digest, self.tools), digest)
            for raw in ("null", '[{}]', '{"x":' + '[' * 1500 + '0' + ']' * 1500 + '}'):
                path.write_text(raw)
                with self.subTest(raw=raw[:20]), self.assertRaises(ValueError):
                    load_test_manifest(path, digest, self.tools)
            value["results"][0]["evidence_sha256"] = False
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_test_manifest(path, digest, self.tools)

if __name__ == "__main__":
    unittest.main()
