from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_inference.gemma4 import GEMMA4_PROMPT_PROFILE
from jarvis_voice.manifest import BASE_MODEL_MANIFEST


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_windows_handoff.py"
SPEC = importlib.util.spec_from_file_location("verify_windows_handoff", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

COMMIT = "0d515832d600fdde2b823f54e801f69e3d574495"


class WindowsHandoffVerifierTests(unittest.TestCase):
    def template(self) -> dict[str, object]:
        return MODULE.handoff_template(COMMIT)

    def assert_code(self, expected: str, value: object) -> None:
        with self.assertRaises(MODULE.HandoffError) as raised:
            MODULE.validate_handoff_manifest(value, expected_commit=COMMIT)
        self.assertEqual(raised.exception.code, expected)

    def test_canonical_template_is_ready_only_for_windows_preparation(self) -> None:
        first = self.template()
        second = self.template()
        report = MODULE.validate_handoff_manifest(first, expected_commit=COMMIT)

        self.assertEqual(first, second)
        self.assertTrue(report["handoff_preparation_ready"])
        self.assertFalse(report["native_qualification_complete"])
        self.assertFalse(report["production_ready"])
        self.assertEqual(report["native_gates_pending"], len(MODULE.NATIVE_GATE_NAMES))
        self.assertEqual(len(report["manifest_sha256"]), 64)
        self.assertTrue(all(value == "pending" for value in first["native_gates"].values()))

    def test_known_model_and_asr_artifacts_are_locked_while_native_artifacts_are_pending(self) -> None:
        artifacts = self.template()["artifact_identities"]
        self.assertEqual(artifacts["gemma_primary_model"]["size_bytes"], 6_975_879_296)
        self.assertEqual(
            artifacts["gemma_primary_model"]["sha256"],
            "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b",
        )
        self.assertEqual(
            artifacts["whisper_base_config"],
            {
                "identity_status": "locked",
                "relative_path": "models/faster-whisper-base/config.json",
                "size_bytes": 2_309,
                "sha256": "56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a",
                "source_revision": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
            },
        )
        self.assertEqual(artifacts["whisper_base_model"]["size_bytes"], 145_217_532)
        self.assertEqual(
            artifacts["whisper_base_model"]["sha256"],
            "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
        )
        self.assertEqual(artifacts["whisper_base_tokenizer"]["size_bytes"], 2_203_239)
        self.assertEqual(
            artifacts["whisper_base_tokenizer"]["sha256"],
            "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
        )
        self.assertEqual(artifacts["whisper_base_vocabulary"]["size_bytes"], 459_861)
        self.assertEqual(
            artifacts["whisper_base_vocabulary"]["sha256"],
            "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
        )
        self.assertEqual(artifacts["whisper_model"]["size_bytes"], 1_617_884_929)
        self.assertEqual(artifacts["llama_server_executable"]["identity_status"], "pending_native")
        self.assertIsNone(artifacts["llama_server_executable"]["sha256"])
        for name in (
            "cuda_runtime_bundle_manifest",
            "windows_inference_manifest",
            "native_fixture_runner",
            "asr_qualification_collector",
            "primary_profile_runner",
            "qualification_runner_build_manifest",
            "windows_asr_audio_corpus_manifest",
            "windows_power_profile",
            "windows_asr_config",
            "silero_vad_model",
        ):
            self.assertEqual(artifacts[name]["identity_status"], "pending_native")
            self.assertIsNone(artifacts[name]["sha256"])
        self.assertEqual(
            artifacts["cuda_runtime_bundle_manifest"]["relative_path"],
            "build-identities/cuda-runtime-manifest.json",
        )
        self.assertEqual(
            artifacts["qualification_runner_build_manifest"]["relative_path"],
            "build-identities/qualification-runners-manifest.json",
        )
        self.assertNotIn("gemma_renderer", artifacts)
        self.assertNotIn("gemma_tool_parser", artifacts)
        self.assertEqual(
            self.template()["profiles"]["llm"]["prompt_profile"],
            MODULE.GEMMA_PROMPT_PROFILE,
        )
        self.assertEqual(MODULE.GEMMA_PROMPT_PROFILE, GEMMA4_PROMPT_PROFILE)
        self.assertEqual(
            MODULE.GEMMA_PROMPT_PROFILE["embedded_chat_template"]["sha256"],
            "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        )

    def test_cpu_baseline_profile_is_required_first_and_references_locked_base_pack(self) -> None:
        baseline = self.template()["profiles"]["asr"]["cpu_baseline"]
        self.assertEqual(
            baseline,
            {
                "profile_id": "CpuBaseline",
                "model_profile": "base",
                "model_repository": "Systran/faster-whisper-base",
                "model_revision": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
                "artifact_ids": [
                    "whisper_base_config",
                    "whisper_base_model",
                    "whisper_base_tokenizer",
                    "whisper_base_vocabulary",
                ],
                "execution_profile": "cpu",
                "cpu_compute_type": "int8",
                "status": "required_first_pending_native_qualification",
            },
        )

        changed = self.template()
        changed["profiles"]["asr"]["cpu_baseline"]["artifact_ids"][0] = "whisper_config"
        self.assert_code("handoff.profile_selection_invalid", changed)

    def test_cpu_baseline_lock_matches_voice_manifest_source_of_truth(self) -> None:
        artifacts = self.template()["artifact_identities"]
        artifact_names = {
            "config.json": "whisper_base_config",
            "model.bin": "whisper_base_model",
            "tokenizer.json": "whisper_base_tokenizer",
            "vocabulary.txt": "whisper_base_vocabulary",
        }
        self.assertEqual(BASE_MODEL_MANIFEST.model_id, "Systran/faster-whisper-base")
        self.assertEqual(BASE_MODEL_MANIFEST.revision, MODULE.WHISPER_BASE_REVISION)
        self.assertEqual(set(artifact_names), {item.name for item in BASE_MODEL_MANIFEST.files})
        for item in BASE_MODEL_MANIFEST.files:
            locked = artifacts[artifact_names[item.name]]
            self.assertEqual(locked["size_bytes"], item.size)
            self.assertEqual(locked["sha256"], item.sha256)
            self.assertEqual(locked["source_revision"], BASE_MODEL_MANIFEST.revision)

    def test_commit_and_single_machine_policy_cannot_be_weakened(self) -> None:
        changed = self.template()
        changed["host_policy"]["live_cross_host_components"] = True
        self.assert_code("handoff.single_machine_policy_invalid", changed)

        changed = self.template()
        changed["frozen_source"]["commit"] = "1" * 40
        self.assert_code("handoff.frozen_commit_mismatch", changed)

    def test_model_selection_and_acceptance_thresholds_are_frozen(self) -> None:
        changed = self.template()
        changed["profiles"]["llm"]["model_repository"] = "Qwen/Qwen3-14B-GGUF"
        self.assert_code("handoff.profile_selection_invalid", changed)

        changed = self.template()
        changed["acceptance"]["llm_quality"]["minimum_pass_rate_per_language"] = 0.80
        self.assert_code("handoff.acceptance_thresholds_invalid", changed)

        asr = self.template()["acceptance"]["asr_native"]
        self.assertEqual(asr["clean_greek_utterances_minimum"], 100)
        self.assertEqual(asr["clean_english_utterances_minimum"], 100)
        self.assertEqual(asr["minimum_language_accuracy_per_locale"], 0.95)
        self.assertEqual(asr["maximum_word_error_rate_per_locale"], 0.20)

        changed = self.template()
        changed["acceptance"]["asr_native"]["clean_greek_utterances_minimum"] = 20
        self.assert_code("handoff.acceptance_thresholds_invalid", changed)

    def test_native_pass_claim_and_fabricated_pending_identity_fail_closed(self) -> None:
        changed = self.template()
        changed["native_gates"]["hardware_inventory"] = "pass"
        self.assert_code("handoff.native_gates_not_pending", changed)

        changed = self.template()
        changed["artifact_identities"]["llama_server_executable"]["sha256"] = "a" * 64
        self.assert_code("handoff.pending_identity_fabricated", changed)

    def test_secret_mac_path_and_unknown_fields_are_rejected(self) -> None:
        changed = self.template()
        changed["target"]["native_identity"]["windows_build"] = "/Users/operator/build.txt"
        self.assert_code("handoff.macos_absolute_path_detected", changed)

        changed = self.template()
        changed["target"]["native_identity"]["windows_build"] = "sk-or-v1-abcdefghijklmnop"
        self.assert_code("handoff.secret_detected", changed)

        changed = self.template()
        changed["profiles"]["llm"]["download_url"] = "https://example.invalid/model"
        self.assert_code("handoff.profile_selection_invalid", changed)

    def test_exact_template_rejects_equal_values_with_wrong_json_types(self) -> None:
        for path, replacement, code in (
            (("frozen_source", "untracked_files_included"), 0, "handoff.frozen_source_invalid"),
            (("host_policy", "components_must_share_host"), 1, "handoff.single_machine_policy_invalid"),
            (("target", "gpu_vram_mib"), 16384.0, "handoff.target_invalid"),
            (("profiles", "llm", "context_tokens"), 32768.0, "handoff.profile_selection_invalid"),
            (("profiles", "llm", "prompt_profile", "thinking"), 0, "handoff.profile_selection_invalid"),
            (("acceptance", "llm_quality", "maximum_critical_failures"), False, "handoff.acceptance_thresholds_invalid"),
            (("acceptance", "llm_performance", "p95_ttft_seconds_maximum"), 3, "handoff.acceptance_thresholds_invalid"),
            (("safety_assertions", "production_ready"), 0, "handoff.safety_assertions_invalid"),
        ):
            with self.subTest(path=path):
                changed = self.template()
                parent = changed
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = replacement
                self.assert_code(code, changed)

    def test_artifact_hash_and_paths_are_exact(self) -> None:
        changed = self.template()
        changed["artifact_identities"]["gemma_primary_model"]["sha256"] = "f" * 64
        self.assert_code("handoff.artifact_identities_invalid", changed)

        changed = self.template()
        changed["artifact_identities"]["llama_server_executable"]["relative_path"] = "../llama.exe"
        self.assert_code("handoff.artifact_path_invalid", changed)

    def test_loader_rejects_duplicate_json_keys_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
            with self.assertRaises(MODULE.HandoffError) as raised:
                MODULE.load_manifest(duplicate)
            self.assertEqual(raised.exception.code, "handoff.duplicate_json_key")

            valid = root / "valid.json"
            valid.write_text(json.dumps(self.template()), encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(valid)
            with self.assertRaises(MODULE.HandoffError) as raised:
                MODULE.load_manifest(link)
            self.assertEqual(raised.exception.code, "handoff.manifest_unreadable")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = MODULE.main(
                    ["verify", "--manifest", str(link), "--expected-commit", COMMIT]
                )
            self.assertEqual(result, 2)
            self.assertEqual(json.loads(stdout.getvalue())["code"], "handoff.manifest_unreadable")

    def test_cli_malformed_json_values_return_only_a_failure_report(self) -> None:
        surrogate = self.template()
        surrogate["target"]["native_identity"]["windows_build"] = "\ud800"
        for raw in (
            '{"schema":' + "1" * 5000 + '}',
            json.dumps(surrogate),
            '{"schema":' + "[" * 2000 + "0" + "]" * 2000 + '}',
        ):
            with self.subTest(size=len(raw)), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "invalid.json"
                path.write_text(raw)
                output = io.StringIO()
                with redirect_stdout(output):
                    result = MODULE.main(["verify", "--manifest", str(path), "--expected-commit", COMMIT])
                report = json.loads(output.getvalue())
                self.assertEqual(result, 2)
                self.assertFalse(report["handoff_preparation_ready"])
                self.assertNotIn(str(path), output.getvalue())

    def test_cli_round_trip_is_offline_and_content_free_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "handoff.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = MODULE.main(
                    ["template", "--frozen-commit", COMMIT, "--output", str(output)]
                )
            report = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertTrue(report["handoff_preparation_ready"])
            self.assertFalse(report["network_used"])
            self.assertFalse(report["installation_performed"])

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = MODULE.main(
                    ["verify", "--manifest", str(output), "--expected-commit", "1" * 40]
                )
            report = json.loads(stdout.getvalue())
            self.assertEqual(result, 2)
            self.assertEqual(report["code"], "handoff.frozen_commit_mismatch")
            self.assertNotIn(str(output), stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
