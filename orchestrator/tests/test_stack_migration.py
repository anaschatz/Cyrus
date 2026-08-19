from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.stack_cli import main as stack_main
from jarvis_orchestrator.stack_config import PrivateStackConfig
from jarvis_orchestrator.stack_migration import (
    LEGACY_STACK_SCHEMA_VERSION,
    ORCHESTRATOR_CONFIG_NAME,
    STACK_CONFIG_NAME,
    VOICE_AUTH_KEY_NAME,
    VOICE_CONFIG_NAME,
    StackMigrationError,
    prepare_auth_v2_stack,
)
from jarvis_orchestrator.windows_private_file import harden_windows_private_path


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class StackMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        if os.name == "nt":  # pragma: no cover - native Windows CI
            harden_windows_private_path(self.root)
        else:
            self.root.chmod(0o700)

        self.voice = self.private_json(
            "voice-v1.json", {"orchestrator_url": "ws://127.0.0.1:8765"}
        )
        self.orchestrator = self.private_json(
            "orchestrator-v1.json",
            {
                "database_path": str(self.root / "state.sqlite3"),
                "control_bearer_token": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
                "inference_bearer_token": "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
            },
        )
        self.runner = self.root / "inference-runner.py"
        self.runner.write_text("pass\n", encoding="utf-8")
        self.runner.chmod(0o700)
        if os.name == "nt":  # pragma: no cover - native Windows CI
            harden_windows_private_path(self.runner)
        self.manifest_document = {
            "alias": "jarvis-local",
            "manifest_schema_version": 1,
        }
        self.manifest = self.private_json(
            "inference-manifest.json", self.manifest_document
        )
        self.report = self.private_json(
            "inference-qualification.json",
            {
                "status": "pass",
                "identity": {
                    "manifest_digest": hashlib.sha256(
                        canonical(self.manifest_document)
                    ).hexdigest()
                },
            },
        )
        self.legacy = self.private_json("jarvis-v1.private.json", self.document())
        self.output = self.root / "prepared-v2"

    def private_json(self, name: str, value: object) -> Path:
        selected = self.root / name
        selected.write_bytes(canonical(value) + b"\n")
        selected.chmod(0o600)
        if os.name == "nt":  # pragma: no cover - native Windows CI
            harden_windows_private_path(selected)
        return selected

    def document(self) -> dict[str, object]:
        return {
            "schema_version": LEGACY_STACK_SCHEMA_VERSION,
            "voice_config_path": str(self.voice),
            "orchestrator_config_path": str(self.orchestrator),
            "inference": {
                "runner_path": str(self.runner),
                "manifest_path": str(self.manifest),
                "qualification_report_path": str(self.report),
            },
        }

    def replace_with_weak_legacy_tokens(self) -> bytes:
        document = json.loads(self.orchestrator.read_text(encoding="utf-8"))
        document["control_bearer_token"] = "legacy-control-token"
        document["inference_bearer_token"] = "legacy-inference-token"
        self.orchestrator.write_bytes(canonical(document) + b"\n")
        self.orchestrator.chmod(0o600)
        return self.orchestrator.read_bytes()

    def test_prepares_independent_private_v2_stack_without_mutating_inputs(
        self,
    ) -> None:
        legacy_snapshot = {
            path: path.read_bytes()
            for path in (
                self.legacy,
                self.voice,
                self.orchestrator,
                self.runner,
                self.manifest,
                self.report,
            )
        }

        result = prepare_auth_v2_stack(self.legacy, self.output)

        self.assertEqual(
            result.canonical(),
            {
                "version": "jarvis-stack-auth-migration/1",
                "status": "prepared",
                "stack_schema_version": "jarvis-private-stack/2",
            },
        )
        self.assertNotIn(str(self.root), json.dumps(result.canonical()))
        self.assertNotIn(
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
            json.dumps(result.canonical()),
        )
        for path, content in legacy_snapshot.items():
            self.assertEqual(path.read_bytes(), content)

        stack = PrivateStackConfig.load(self.output / STACK_CONFIG_NAME)
        self.assertTrue(
            stack.voice_config_path.samefile(self.output / VOICE_CONFIG_NAME)
        )
        self.assertTrue(
            stack.orchestrator_config_path.samefile(
                self.output / ORCHESTRATOR_CONFIG_NAME
            )
        )
        voice_document = json.loads((self.output / VOICE_CONFIG_NAME).read_text())
        orchestrator_document = json.loads(
            (self.output / ORCHESTRATOR_CONFIG_NAME).read_text()
        )
        expected_key = str(self.output / VOICE_AUTH_KEY_NAME)
        self.assertEqual(voice_document["voice_auth_key_file"], expected_key)
        self.assertEqual(orchestrator_document["voice_auth_key_file"], expected_key)
        self.assertEqual(
            set(voice_document) - set(json.loads(self.voice.read_text())),
            {"voice_auth_key_file"},
        )
        self.assertEqual(
            set(orchestrator_document) - set(json.loads(self.orchestrator.read_text())),
            {"voice_auth_key_file"},
        )
        if os.name != "nt":
            self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)
            for name in (
                VOICE_CONFIG_NAME,
                ORCHESTRATOR_CONFIG_NAME,
                VOICE_AUTH_KEY_NAME,
                STACK_CONFIG_NAME,
            ):
                self.assertEqual((self.output / name).stat().st_mode & 0o777, 0o600)

    def test_cli_prepares_bundle_without_printing_paths_or_secrets(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = stack_main(
                [
                    "prepare-auth-v2",
                    "--legacy-config",
                    str(self.legacy),
                    "--output-directory",
                    str(self.output),
                    "--macos-ptt-key",
                    "CONTROL_OPTION",
                    "--safe-beta-scope",
                ]
            )

        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "prepared")
        self.assertNotIn(str(self.root), output.getvalue())
        self.assertNotIn(
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
            output.getvalue(),
        )
        self.assertTrue((self.output / STACK_CONFIG_NAME).is_file())
        voice = json.loads((self.output / VOICE_CONFIG_NAME).read_text())
        self.assertEqual(voice["ptt_key"], "CONTROL_OPTION")
        orchestrator = json.loads((self.output / ORCHESTRATOR_CONFIG_NAME).read_text())
        self.assertFalse(orchestrator.get("memory_enabled", False))
        self.assertFalse(orchestrator["memory_shadow_enabled"])
        for flag in (
            "cloud_enabled",
            "cloud_terra_enabled",
            "research_enabled",
            "research_browser_fallback",
            "tools_enabled",
        ):
            self.assertFalse(orchestrator[flag])
        self.assertIsNone(orchestrator["cloud_secret_file"])
        self.assertIsNone(orchestrator["scheduler_compiler_qualification_file"])

    def test_weak_v1_tokens_require_explicit_rotation_without_mutating_input(
        self,
    ) -> None:
        legacy_orchestrator = self.replace_with_weak_legacy_tokens()

        with self.assertRaisesRegex(
            StackMigrationError,
            "^legacy_bearer_token_rotation_required$",
        ):
            prepare_auth_v2_stack(self.legacy, self.output)

        self.assertFalse(self.output.exists())
        self.assertEqual(self.orchestrator.read_bytes(), legacy_orchestrator)

        rendered = io.StringIO()
        with redirect_stdout(rendered):
            status = stack_main(
                [
                    "prepare-auth-v2",
                    "--legacy-config",
                    str(self.legacy),
                    "--output-directory",
                    str(self.output),
                ]
            )
        self.assertEqual(status, 2)
        self.assertEqual(
            json.loads(rendered.getvalue()),
            {
                "error": {
                    "code": "legacy_bearer_token_rotation_required",
                    "retryable": False,
                },
                "ready": False,
            },
        )
        self.assertNotIn("legacy-control-token", rendered.getvalue())
        self.assertNotIn(str(self.root), rendered.getvalue())
        self.assertEqual(self.orchestrator.read_bytes(), legacy_orchestrator)

    def test_explicit_rotation_writes_only_canonical_tokens_to_fresh_bundle(
        self,
    ) -> None:
        legacy_orchestrator = self.replace_with_weak_legacy_tokens()
        rotated_control = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
        rotated_inference = "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8"
        rendered = io.StringIO()

        with (
            patch(
                "jarvis_orchestrator.stack_migration.secrets.token_urlsafe",
                side_effect=(rotated_control, rotated_inference),
            ) as generate,
            redirect_stdout(rendered),
        ):
            status = stack_main(
                [
                    "prepare-auth-v2",
                    "--legacy-config",
                    str(self.legacy),
                    "--output-directory",
                    str(self.output),
                    "--rotate-weak-bearer-tokens",
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(self.orchestrator.read_bytes(), legacy_orchestrator)
        generated = OrchestratorConfig.load(self.output / ORCHESTRATOR_CONFIG_NAME)
        self.assertEqual(generated.control_bearer_token, rotated_control)
        self.assertEqual(generated.inference_bearer_token, rotated_inference)
        self.assertNotIn(rotated_control, rendered.getvalue())
        self.assertNotIn(rotated_inference, rendered.getvalue())
        self.assertNotIn("legacy-control-token", rendered.getvalue())
        self.assertNotIn(str(self.root), rendered.getvalue())

    def test_rotation_flag_must_be_a_boolean(self) -> None:
        with self.assertRaisesRegex(
            StackMigrationError,
            "^migration_bearer_rotation_flag_invalid$",
        ):
            prepare_auth_v2_stack(
                self.legacy,
                self.output,
                rotate_weak_bearer_tokens=1,  # type: ignore[arg-type]
            )
        self.assertFalse(self.output.exists())

    def test_invalid_macos_ptt_key_fails_before_output_creation(self) -> None:
        with self.assertRaisesRegex(
            StackMigrationError, "migration_macos_ptt_key_invalid"
        ):
            prepare_auth_v2_stack(self.legacy, self.output, macos_ptt_key="F9")
        self.assertFalse(self.output.exists())

    def test_duplicate_key_and_non_finite_json_fail_before_output_creation(
        self,
    ) -> None:
        self.legacy.write_text(
            '{"schema_version":"jarvis-private-stack/1",'
            '"schema_version":"jarvis-private-stack/1"}',
            encoding="utf-8",
        )
        self.legacy.chmod(0o600)
        with self.assertRaisesRegex(StackMigrationError, "legacy_json_invalid"):
            prepare_auth_v2_stack(self.legacy, self.output)
        self.assertFalse(self.output.exists())

        self.legacy.write_bytes(canonical(self.document()) + b"\n")
        self.legacy.chmod(0o600)
        self.voice.write_text(
            '{"orchestrator_url":"ws://127.0.0.1:8765","bad":NaN}',
            encoding="utf-8",
        )
        self.voice.chmod(0o600)
        with self.assertRaisesRegex(StackMigrationError, "legacy_json_invalid"):
            prepare_auth_v2_stack(self.legacy, self.output)
        self.assertFalse(self.output.exists())

    def test_bad_final_qualification_rolls_back_only_fresh_output(self) -> None:
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
        before = self.report.read_bytes()

        with self.assertRaisesRegex(StackMigrationError, "migration_validation_failed"):
            prepare_auth_v2_stack(self.legacy, self.output)

        self.assertFalse(self.output.exists())
        self.assertEqual(self.report.read_bytes(), before)
        self.assertTrue(self.legacy.exists())

    def test_existing_output_is_never_modified_or_removed(self) -> None:
        self.output.mkdir(mode=0o700)
        sentinel = self.output / "keep.txt"
        sentinel.write_text("owned by caller", encoding="utf-8")
        with self.assertRaisesRegex(StackMigrationError, "migration_output_not_fresh"):
            prepare_auth_v2_stack(self.legacy, self.output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "owned by caller")

    @unittest.skipIf(os.name == "nt", "POSIX symlink and mode contract")
    def test_symlink_inputs_outputs_and_non_private_inputs_fail_closed(self) -> None:
        legacy_link = self.root / "legacy-link.json"
        legacy_link.symlink_to(self.legacy)
        with self.assertRaisesRegex(StackMigrationError, "legacy_stack_unavailable"):
            prepare_auth_v2_stack(legacy_link, self.output)
        self.assertFalse(self.output.exists())

        voice_link = self.root / "voice-link.json"
        voice_link.symlink_to(self.voice)
        linked_document = self.document()
        linked_document["voice_config_path"] = str(voice_link)
        linked_legacy = self.private_json("linked.private.json", linked_document)
        with self.assertRaisesRegex(
            StackMigrationError, "legacy_voice_config_unavailable"
        ):
            prepare_auth_v2_stack(linked_legacy, self.output)
        self.assertFalse(self.output.exists())

        output_target = self.root / "existing-target"
        output_target.mkdir(mode=0o700)
        self.output.symlink_to(output_target, target_is_directory=True)
        with self.assertRaisesRegex(StackMigrationError, "migration_output_not_fresh"):
            prepare_auth_v2_stack(self.legacy, self.output)
        self.assertTrue(self.output.is_symlink())
        self.output.unlink()

        self.voice.chmod(0o644)
        with self.assertRaisesRegex(
            StackMigrationError, "legacy_private_file_permissions"
        ):
            prepare_auth_v2_stack(self.legacy, self.output)
        self.assertFalse(self.output.exists())

    def test_relative_paths_unknown_shape_and_existing_auth_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            StackMigrationError, "migration_output_path_invalid"
        ):
            prepare_auth_v2_stack(self.legacy, Path("relative-output"))

        invalid = self.document()
        invalid["unexpected"] = True
        self.legacy.write_bytes(canonical(invalid) + b"\n")
        self.legacy.chmod(0o600)
        with self.assertRaisesRegex(StackMigrationError, "legacy_stack_schema_invalid"):
            prepare_auth_v2_stack(self.legacy, self.output)

        self.legacy.write_bytes(canonical(self.document()) + b"\n")
        self.legacy.chmod(0o600)
        voice = json.loads(self.voice.read_text())
        voice["voice_auth_key_file"] = str(self.root / "old-key.json")
        self.voice.write_bytes(canonical(voice) + b"\n")
        self.voice.chmod(0o600)
        with self.assertRaisesRegex(
            StackMigrationError, "legacy_auth_already_configured"
        ):
            prepare_auth_v2_stack(self.legacy, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
