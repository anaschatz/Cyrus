from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.beta_preflight import evaluate_macos_beta_preflight
from jarvis_orchestrator.macos_beta_session import MacOSBetaSessionError
from jarvis_orchestrator.private_state import PrivateStateReportV1
from jarvis_orchestrator.stack_cli import _beta_session_error_payload
from jarvis_orchestrator.stack_cli import main as stack_main
from jarvis_orchestrator.stack_config import (
    PrivateStackConfig,
    write_private_stack_config,
)

from jarvis_voice.private_state import VoicePrivateStateReportV1
from jarvis_voice.voice_auth import create_voice_auth_key_file


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


class MacosBetaPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.key = self.root / "auth.json"
        create_voice_auth_key_file(self.key, key=b"k" * 32)
        self.voice = self.private_json(
            "voice.json",
            {
                "orchestrator_url": "ws://127.0.0.1:8765",
                "ptt_key": "CONTROL_OPTION",
                "voice_auth_key_file": str(self.key),
                "macos": {
                    "asr_backend": "faster_whisper_cpu_smoke",
                    "system_voices": {
                        "el-GR": "Melina",
                        "en-US": "Samantha",
                    },
                    "tts_backend": "system_say",
                },
            },
        )
        self.orchestrator_document = {
            "database_path": str(self.root / "state" / "state.sqlite3"),
            "control_bearer_token": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
            "inference_bearer_token": "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            "memory_enabled": False,
            "memory_shadow_enabled": False,
            "voice_auth_key_file": str(self.key),
        }
        self.orchestrator = self.private_json(
            "orchestrator.json", self.orchestrator_document
        )
        self.runner = self.root / "runner"
        self.runner.write_text("pass\n", encoding="utf-8")
        self.runner.chmod(0o700)
        manifest_document = {
            "alias": "jarvis-local",
            "manifest_schema_version": 1,
        }
        self.manifest = self.private_json("manifest.json", manifest_document)
        self.report = self.private_json(
            "qualification.json",
            {
                "status": "pass",
                "identity": {
                    "manifest_digest": hashlib.sha256(
                        canonical(manifest_document)
                    ).hexdigest()
                },
            },
        )
        self.stack_path = self.root / "stack.json"
        write_private_stack_config(
            self.stack_path,
            {
                "schema_version": "jarvis-private-stack/2",
                "voice_config_path": str(self.voice),
                "orchestrator_config_path": str(self.orchestrator),
                "inference": {
                    "runner_path": str(self.runner),
                    "manifest_path": str(self.manifest),
                    "qualification_report_path": str(self.report),
                },
            },
        )

    def private_json(self, name: str, value: object) -> Path:
        path = self.root / name
        path.write_bytes(canonical(value) + b"\n")
        path.chmod(0o600)
        return path

    @staticmethod
    def private_state(ready: bool) -> PrivateStateReportV1:
        return PrivateStateReportV1(
            version="orchestrator-private-state/1",
            ready=ready,
            platform_enforced=True,
            checked_directory_count=2,
            checked_file_count=4,
            missing_directory_count=0,
            missing_file_count=0,
            hardened_directory_count=0,
            hardened_file_count=0,
            violation_counts={} if ready else {"file_permissions": 1},
        )

    @staticmethod
    def voice_private_state(ready: bool) -> VoicePrivateStateReportV1:
        return VoicePrivateStateReportV1(
            version="jarvis-voice-private-state/1",
            ready=ready,
            platform_enforced=True,
            checked_directory_count=5,
            checked_file_count=3,
            missing_directory_count=0,
            missing_file_count=9,
            created_directory_count=0,
            created_file_count=0,
            hardened_directory_count=0,
            hardened_file_count=0,
            violation_counts={} if ready else {"file_permissions": 1},
        )

    def test_safe_static_profile_can_enter_technical_preflight_only(self) -> None:
        report = evaluate_macos_beta_preflight(
            PrivateStackConfig.load(self.stack_path),
            doctor_report={"ready": True},
            private_state_report=self.private_state(True),
            voice_private_state_report=self.voice_private_state(True),
        )

        self.assertTrue(report["technical_beta_preflight_ready"])
        self.assertFalse(report["daily_owner_beta_ready"])
        self.assertFalse(report["production_ready"])
        self.assertEqual(report["version"], "jarvis-macos-beta-preflight/3")
        self.assertEqual(report["voice_environment_blockers"], [])
        self.assertEqual(
            report["voice_environment_probe_scope"],
            "current_process_identity",
        )
        self.assertEqual(
            report["daily_owner_blocker"],
            "bilingual_conversational_quality_not_qualified",
        )
        self.assertNotIn("Melina", json.dumps(report))
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_verified_pocket_profile_can_enter_technical_preflight(self) -> None:
        document = json.loads(self.voice.read_text(encoding="utf-8"))
        document["macos"]["tts_backend"] = "pocket_tts"
        document["pocket_tts"] = {
            "model": {
                "path": str(self.root / "model.safetensors"),
                "size": 1,
                "sha256": "a" * 64,
            },
            "tokenizer": {
                "path": str(self.root / "tokenizer.model"),
                "size": 1,
                "sha256": "b" * 64,
            },
            "voice_state": {
                "path": str(self.root / "george.safetensors"),
                "size": 1,
                "sha256": "c" * 64,
            },
        }
        self.voice.write_bytes(canonical(document) + b"\n")

        report = evaluate_macos_beta_preflight(
            PrivateStackConfig.load(self.stack_path),
            doctor_report={"ready": True},
            private_state_report=self.private_state(True),
            voice_private_state_report=self.voice_private_state(True),
        )

        self.assertTrue(report["technical_beta_preflight_ready"])
        self.assertTrue(report["checks"]["selected_voice_profile"])
        self.assertFalse(report["production_ready"])

    def test_verified_whisper_cpp_profile_can_enter_technical_preflight(self) -> None:
        document = json.loads(self.voice.read_text(encoding="utf-8"))
        document["macos"]["asr_backend"] = "whisper_cpp"
        self.voice.write_bytes(canonical(document) + b"\n")

        report = evaluate_macos_beta_preflight(
            PrivateStackConfig.load(self.stack_path),
            doctor_report={"ready": True},
            private_state_report=self.private_state(True),
            voice_private_state_report=self.voice_private_state(True),
        )

        self.assertTrue(report["technical_beta_preflight_ready"])
        self.assertTrue(report["checks"]["selected_voice_profile"])

    def test_unapproved_tts_backend_cannot_enter_technical_preflight(self) -> None:
        document = json.loads(self.voice.read_text(encoding="utf-8"))
        document["macos"]["tts_backend"] = "kokoro_onnx"
        document["macos"]["kokoro_onnx"] = {
            "model": {
                "path": str(self.root / "model.onnx"),
                "size": 1,
                "sha256": "a" * 64,
            },
            "voices": {
                "path": str(self.root / "voices.bin"),
                "size": 1,
                "sha256": "b" * 64,
            },
        }
        self.voice.write_bytes(canonical(document) + b"\n")

        report = evaluate_macos_beta_preflight(
            PrivateStackConfig.load(self.stack_path),
            doctor_report={"ready": True},
            private_state_report=self.private_state(True),
            voice_private_state_report=self.voice_private_state(True),
        )

        self.assertFalse(report["technical_beta_preflight_ready"])
        self.assertFalse(report["checks"]["selected_voice_profile"])

    def test_memory_flag_without_live_crypto_evidence_is_blocked(self) -> None:
        document = dict(self.orchestrator_document)
        document["memory_enabled"] = True
        self.orchestrator.write_bytes(canonical(document) + b"\n")
        self.orchestrator.chmod(0o600)

        report = evaluate_macos_beta_preflight(
            PrivateStackConfig.load(self.stack_path),
            doctor_report={"ready": True},
            private_state_report=self.private_state(True),
            voice_private_state_report=self.voice_private_state(True),
        )

        self.assertFalse(report["technical_beta_preflight_ready"])
        capability = next(
            item
            for item in report["capabilities"]
            if item["name"] == "memory.retrospective"
        )
        self.assertEqual(capability["effective"], "BLOCKED")
        self.assertEqual(capability["reason_code"], "memory_crypto_runtime_unqualified")

    def test_memory_flag_with_live_crypto_evidence_enters_technical_preflight(
        self,
    ) -> None:
        document = dict(self.orchestrator_document)
        document["memory_enabled"] = True
        self.orchestrator.write_bytes(canonical(document) + b"\n")
        self.orchestrator.chmod(0o600)

        with patch(
            "jarvis_orchestrator.beta_preflight._live_memory_crypto_state",
            return_value="a" * 64,
        ):
            report = evaluate_macos_beta_preflight(
                PrivateStackConfig.load(self.stack_path),
                doctor_report={"ready": True},
                private_state_report=self.private_state(True),
                voice_private_state_report=self.voice_private_state(True),
            )

        self.assertTrue(report["technical_beta_preflight_ready"])
        self.assertTrue(report["checks"]["memory_crypto_qualified"])
        self.assertEqual(report["memory_crypto_state_sha256"], "a" * 64)
        capability = next(
            item
            for item in report["capabilities"]
            if item["name"] == "memory.retrospective"
        )
        self.assertEqual(capability["effective"], "READY")
        self.assertEqual(capability["reason_code"], "memory_crypto_runtime_qualified")

    def test_exposed_state_or_shadow_memory_blocks_technical_preflight(self) -> None:
        stack = PrivateStackConfig.load(self.stack_path)
        exposed = evaluate_macos_beta_preflight(
            stack,
            doctor_report={"ready": True},
            private_state_report=self.private_state(False),
            voice_private_state_report=self.voice_private_state(True),
        )
        self.assertFalse(exposed["technical_beta_preflight_ready"])
        self.assertFalse(exposed["checks"]["owner_only_state"])

        document = dict(self.orchestrator_document)
        document["memory_shadow_enabled"] = True
        self.orchestrator.write_bytes(canonical(document) + b"\n")
        self.orchestrator.chmod(0o600)
        shadow = evaluate_macos_beta_preflight(
            PrivateStackConfig.load(self.stack_path),
            doctor_report={"ready": True},
            private_state_report=self.private_state(True),
            voice_private_state_report=self.voice_private_state(True),
        )
        self.assertFalse(shadow["technical_beta_preflight_ready"])
        self.assertFalse(shadow["checks"]["safe_feature_scope"])

    def test_exposed_voice_journal_state_blocks_owner_only_gate(self) -> None:
        report = evaluate_macos_beta_preflight(
            PrivateStackConfig.load(self.stack_path),
            doctor_report={"ready": True},
            private_state_report=self.private_state(True),
            voice_private_state_report=self.voice_private_state(False),
        )

        self.assertFalse(report["technical_beta_preflight_ready"])
        self.assertFalse(report["checks"]["owner_only_state"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("file_permissions", encoded)
        self.assertNotIn(str(self.root), encoded)

    def test_restricted_native_context_surfaces_only_sanitized_voice_codes(
        self,
    ) -> None:
        report = evaluate_macos_beta_preflight(
            PrivateStackConfig.load(self.stack_path),
            doctor_report={
                "ready": False,
                "checks": {
                    "ptt": {
                        "status": "inconclusive",
                        "code": "environment.restricted_native_io_probe",
                        "private_detail": str(self.root),
                    },
                    "audio": {
                        "status": "inconclusive",
                        "code": "environment.restricted_native_io_probe",
                    },
                    "whisper_cpp": {
                        "status": "deferred",
                        "code": "asr.whisper_cpp_not_selected",
                    },
                },
            },
            private_state_report=self.private_state(True),
            voice_private_state_report=self.voice_private_state(True),
        )

        self.assertFalse(report["technical_beta_preflight_ready"])
        self.assertFalse(report["checks"]["voice_environment"])
        self.assertEqual(
            report["voice_environment_blockers"],
            ["environment.restricted_native_io_probe"],
        )
        self.assertNotIn(str(self.root), json.dumps(report))

        payload = _beta_session_error_payload(
            "beta_preflight_failed",
            preflight_report=report,
        )
        self.assertEqual(
            payload["error"]["code"],
            "beta_preflight_native_probe_inconclusive",
        )
        self.assertEqual(
            payload["error"]["cause_codes"],
            ["environment.restricted_native_io_probe"],
        )
        self.assertFalse(payload["ready"])

    def test_malformed_voice_blocker_is_replaced_and_not_echoed(self) -> None:
        private_detail = f"invalid {self.root}"
        report = evaluate_macos_beta_preflight(
            PrivateStackConfig.load(self.stack_path),
            doctor_report={
                "ready": False,
                "checks": {
                    "audio": {
                        "status": "blocked",
                        "code": private_detail,
                    }
                },
            },
            private_state_report=self.private_state(True),
            voice_private_state_report=self.voice_private_state(True),
        )

        self.assertEqual(
            report["voice_environment_blockers"],
            ["voice.environment_not_ready"],
        )
        self.assertNotIn(private_detail, json.dumps(report))

    def test_beta_run_cli_preserves_restricted_probe_reason_without_starting(
        self,
    ) -> None:
        report = {
            "technical_beta_preflight_ready": False,
            "voice_environment_blockers": ["environment.restricted_native_io_probe"],
        }
        output = io.StringIO()
        with (
            patch(
                "jarvis_orchestrator.macos_beta_session.MacOSBetaSession.run",
                side_effect=MacOSBetaSessionError("beta_preflight_failed"),
            ) as run,
            patch(
                "jarvis_orchestrator.beta_preflight.evaluate_macos_beta_preflight",
                return_value=report,
            ),
            redirect_stdout(output),
        ):
            exit_code = stack_main(
                [
                    "--config",
                    str(self.stack_path),
                    "beta-run",
                    "--runtime-python",
                    sys.executable,
                    "--session-parent",
                    str(self.root),
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            payload["error"]["code"],
            "beta_preflight_native_probe_inconclusive",
        )
        self.assertEqual(
            payload["error"]["cause_codes"],
            ["environment.restricted_native_io_probe"],
        )
        self.assertFalse(payload["ready"])


if __name__ == "__main__":
    unittest.main()
