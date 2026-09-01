from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis_inference.cli import verify_readiness
from jarvis_inference.durable import durable_replace, read_report
from jarvis_inference.gemma4 import (
    GEMMA4_PROMPT_PROFILE,
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
)
from jarvis_inference.engine import PreparedPrompt
from jarvis_inference.gateway import Production
from jarvis_inference.manifest import load_manifest
from jarvis_inference.primary_profile import (
    PRIMARY_CORPUS_SHA256,
    PRIMARY_QUALITY_SUITE_VERSION,
    PRIMARY_SYSTEM_PROMPT,
    load_primary_corpus,
)
from jarvis_inference.primary_qualification import _PrimaryQualificationProduction
from jarvis_inference.profile_report import (
    ATTESTATION_LEVEL,
    EVIDENCE_SCHEMA,
    OPERATOR_TRUST_REQUIRED,
    PERFORMANCE_FIXTURE_RELATIVE,
    PRIMARY_BASE_CORPUS_RELATIVE,
    PRIMARY_CORPUS_RELATIVE,
    PROFILE_EVIDENCE_RELATIVE,
    PROFILE_RUNNER_RELATIVE,
    REPORT_SCHEMA,
    expected_profile_identity,
    load_valid_profile_report,
    profile_report_matches,
    validate_profile_document,
)
from jarvis_inference.windows_primary_fixture import SUITE_ID, load_fixture
from jarvis_inference.strict_json import canonical_bytes


REPOSITORY_FIXTURES = Path(__file__).parents[1] / "fixtures"


def _artifact(path: Path, root: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


class QualifiedPrimaryFixture:
    def __init__(self, root: Path):
        self.root = root
        qualification = root / "qualification"
        qualification.mkdir()
        shutil.copyfile(
            REPOSITORY_FIXTURES / "local-primary-quality-v1.json",
            root / PRIMARY_CORPUS_RELATIVE,
        )
        shutil.copyfile(
            REPOSITORY_FIXTURES / "local-fallback-quality-v1.json",
            root / PRIMARY_BASE_CORPUS_RELATIVE,
        )
        shutil.copyfile(
            REPOSITORY_FIXTURES / "windows-primary-performance-soak-v1.json",
            root / PERFORMANCE_FIXTURE_RELATIVE,
        )
        self.profile_runner = root / PROFILE_RUNNER_RELATIVE
        self.profile_runner.write_bytes(b"profile-runner")
        self.generic_runner = qualification / "generic-runner.exe"
        self.generic_runner.write_bytes(b"generic-runner")
        self.asset_paths = {
            "fixture": qualification / "windows-asr-coexistence-v1.json",
            "collector": qualification / "cyrus-windows-asr-collector.exe",
            "audio_corpus": qualification / "audio-corpus-v1.json",
            "power_profile": qualification / "power-profile-v1.json",
            "asr_config": qualification / "asr-config-v1.json",
        }
        self.generic_fixture_document = json.loads(
            (REPOSITORY_FIXTURES / "windows-asr-coexistence-v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.power_document = {
            "windows_power_scheme_guid": "12345678-1234-1234-1234-123456789abc",
            "ac_state": "ac",
            "gpu_power_limit_watts": 360,
            "gpu_clock_policy": (
                "applications_graphics_mhz=2500;"
                "applications_memory_mhz=14000"
            ),
            "driver_version": "fixture-driver",
            "compute_runtime": "CUDA 12.8",
            "require_zero_thermal_power_throttle_reasons": True,
        }
        for name, path in self.asset_paths.items():
            if name == "fixture":
                path.write_bytes(canonical_bytes(self.generic_fixture_document))
            elif name == "power_profile":
                path.write_bytes(canonical_bytes(self.power_document))
            else:
                path.write_bytes((name + "-asset").encode("ascii"))
        executable = root / "llama-server.exe"
        executable.write_bytes(b"server")
        model = root / MODEL_FILENAME
        model.write_bytes(b"model-placeholder")
        document = {
            "manifest_schema_version": 1,
            "alias": "jarvis-local",
            "platform": "windows-x86_64",
            "backend": {
                "type": "llama-server",
                "llama_cpp_commit": "a" * 40,
                "build_id": "fixture",
                "executable": _artifact(executable, root),
                "dlls": [],
                "parameters": {},
            },
            "model": {
                "format": "gguf",
                "context_window": 32_768,
                "artifacts": [
                    {
                        "path": MODEL_FILENAME,
                        "size_bytes": MODEL_SIZE_BYTES,
                        "sha256": MODEL_SHA256,
                    }
                ],
            },
            "python": {},
            "capabilities": {
                "text": True,
                "stream": True,
                "tools": True,
                "json_object": True,
                "json_schema": True,
            },
            "prompt_profile": GEMMA4_PROMPT_PROFILE,
            "qualification_report": "qualification-report.json",
            "profile_qualification_report": "qualification-profile-local-primary.json",
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(document), encoding="utf-8")
        self.manifest = load_manifest(manifest_path)
        self.current_identity = {
            "manifest_digest": self.manifest.digest,
            "gateway_digest": "b" * 64,
            "hardware_runtime": {
                "gpu_device": "RTX 5080",
                "driver": "fixture-driver",
                "compute_runtime": "CUDA 12.8",
            },
        }
        assets = {
            name: _artifact(path, root) for name, path in self.asset_paths.items()
        }
        self.generic_gate = {
            "kind": "windows_asr_coexistence",
            "fixture": {
                **self.generic_fixture_document,
                "power_profile": self.power_document,
            },
            "assets": assets,
            "asr_runtime_identity_digest": assets["asr_config"]["sha256"],
            "audio_corpus_digest": assets["audio_corpus"]["sha256"],
            "greek_utterances": 100,
            "english_utterances": 100,
            "greek_language_correct": 95,
            "english_language_correct": 95,
            "greek_reference_words": 1000,
            "greek_word_errors": 100,
            "english_reference_words": 1000,
            "english_word_errors": 100,
            "forced_cpu_fallbacks": 5,
            "baseline_power_identity_digest": hashlib.sha256(
                canonical_bytes(self.power_document)
            ).hexdigest(),
            "coexistence_power_identity_digest": hashlib.sha256(
                canonical_bytes(self.power_document)
            ).hexdigest(),
            "overlapping_utterances": 80,
            "baseline_p95": 1,
            "coexistence_p95": 1.1,
            "baseline_llm_p95": 5,
            "coexistence_llm_p95": 5.5,
            "minimum_free_vram_mib": 2048,
            "lost_final_transcripts": 0,
            "duplicate_final_transcripts": 0,
            "stale_results": 0,
            "oom": 0,
            "crashes": 0,
            "throttle_reasons": 0,
        }
        self.generic_report = {
            "report_schema_version": 1,
            "status": "pass",
            "identity": self.current_identity,
            "evidence_digest": "e" * 64,
            "fixture_runner": _artifact(self.generic_runner, root),
            "results": {
                "streaming": {"greek_success": 20, "english_success": 20},
                "tools": {"named_success": 10, "required_success": 10},
                "json_schema": {"strict_success": 20},
                "cancellation": {"prefill_success": 5, "mid_stream_success": 5},
                "crash_restart": {"isolated_success": 5},
                "isolation": {"pairs_success": 50, "leaks": 0},
                "protocol_conformance": 1,
                "privacy_leaks": 0,
                "unexpected_oom": 0,
                "unexpected_crashes": 0,
                "unexpected_hangs": 0,
                "platform_gate": self.generic_gate,
            },
        }
        durable_replace(self.manifest.qualification_report, self.generic_report)
        parsed_report = read_report(self.manifest.qualification_report)
        assert isinstance(parsed_report, dict)
        self.generic_report = parsed_report

    def evidence(self) -> dict[str, object]:
        quality_fixtures, _ = load_primary_corpus(self.root / PRIMARY_CORPUS_RELATIVE)
        performance_fixture = load_fixture(self.root / PERFORMANCE_FIXTURE_RELATIVE)
        quality_cases = [
            {"case_id": fixture.case_id, "passed": True, "critical": False}
            for fixture in quality_fixtures
        ]
        performance_cases = []
        for serial, case in enumerate(
            performance_fixture.document["performance_cases"], start=1
        ):
            performance_cases.append({
                "case_id": case["case_id"],
                "target_bin": case["target_rendered_bin"],
                "generation_serial": serial,
                "rendered_token_count": case["target_rendered_bin"],
                "rendered_token_ids_sha256": hashlib.sha256(
                    case["case_id"].encode("ascii")
                ).hexdigest(),
                "ttft_seconds": 1,
                "end_to_end_seconds": 5,
                "backend_prompt_seconds": 0.5,
                "backend_decode_seconds": 4,
                "request_to_engine_final_seconds": 4.5,
                "accepted_decode_tokens": 128,
                "accepted_token_ids_sha256": hashlib.sha256(
                    ("accepted-" + case["case_id"]).encode("ascii")
                ).hexdigest(),
                "decode_tokens_per_second": 32,
                "projected_128_token_e2e_seconds": 1 + 127 / 32,
                "finish_reason": "length",
                "passed": True,
            })
        tokenizer_identity = {
            "schema": "cyrus.windows-native-tokenizer-identity/1",
            "manifest_sha256": self.manifest.digest,
            "runtime_identity_sha256": hashlib.sha256(
                canonical_bytes(self.current_identity)
            ).hexdigest(),
            "backend": {
                "llama_cpp_commit": self.manifest.backend["llama_cpp_commit"],
                "build_id": self.manifest.backend["build_id"],
            },
            "model_artifact_sha256": self.manifest.model["artifacts"][0][
                "sha256"
            ],
            "prompt_profile_id": self.manifest.prompt_profile["id"],
            "renderer_sha256": self.manifest.prompt_profile["renderer"][
                "sha256"
            ],
            "cases": [
                {
                    "case_id": case["case_id"],
                    "rendered_token_count": case["rendered_token_count"],
                    "rendered_token_ids_sha256": case[
                        "rendered_token_ids_sha256"
                    ],
                }
                for case in performance_cases
            ],
        }
        tokenizer_sha = hashlib.sha256(
            canonical_bytes(tokenizer_identity)
        ).hexdigest()
        profile_runner_sha = hashlib.sha256(
            self.profile_runner.read_bytes()
        ).hexdigest()
        identity, _ = expected_profile_identity(
            self.manifest,
            self.current_identity,
            self.generic_report,
            profile_runner_sha256=profile_runner_sha,
            profile_evidence_sha256="0" * 64,
            tokenizer_identity_sha256=tokenizer_sha,
        )
        identity.pop("profile_evidence_sha256")
        return {
            "schema": EVIDENCE_SCHEMA,
            "status": "pass",
            "profile_id": "local-primary",
            "platform": "windows-x86_64",
            "attestation_level": ATTESTATION_LEVEL,
            "operator_trust_required": OPERATOR_TRUST_REQUIRED,
            "identity": identity,
            "provenance": {
                "challenge": "1" * 64,
                "started_unix_ns": "1000000000",
                "completed_unix_ns": "28801000000000",
                "monotonic_elapsed_ns": 28_800 * 1_000_000_000,
                "restart_epochs": list(range(1, 10)),
            },
            "quality": {
                "status": "pass",
                "suite_version": PRIMARY_QUALITY_SUITE_VERSION,
                "corpus_sha256": PRIMARY_CORPUS_SHA256,
                "system_prompt_sha256": hashlib.sha256(
                    PRIMARY_SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
                "sample_count": 120,
                "greek_total": 60,
                "greek_passed": 60,
                "english_total": 60,
                "english_passed": 60,
                "language_pass_rate_gap": 0,
                "critical_failures": 0,
                "protocol_failures": 0,
                "cases": quality_cases,
            },
            "performance": {
                "status": "pass",
                "suite_id": SUITE_ID,
                "fixture_sha256": performance_fixture.fixture_sha256,
                "expanded_requests_sha256": performance_fixture.expanded_requests_sha256,
                "rendered_prompt_texts_sha256": performance_fixture.rendered_prompt_texts_sha256,
                "timed_requests_completed": 100,
                "cold_load_to_ready_samples_seconds": [100] * 8,
                "native_token_validation": {
                    "status": "pass",
                    "validated_cases": 100,
                    "tokenizer_identity_sha256": tokenizer_sha,
                },
                "metrics": {
                    "ttft_p50_seconds": 1,
                    "ttft_p95_seconds": 1,
                    "end_to_end_p50_seconds": 5,
                    "end_to_end_p95_seconds": 5,
                    "projected_128_token_e2e_p95_seconds": 1 + 127 / 32,
                    "decode_p5_tokens_per_second": 32,
                    "decode_p50_tokens_per_second": 32,
                    "decode_p95_tokens_per_second": 32,
                    "cold_load_to_ready_p95_seconds": 100,
                },
                "cases": performance_cases,
            },
            "soak": {
                "status": "pass",
                "monotonic_seconds_observed": 28_800,
                "cycles_completed": 96,
                "full_restarts_completed": 8,
                "event_counts": {
                    "observed_request_failures": 0,
                    "observed_unexpected_process_terminations": 0,
                    "observed_request_timeouts": 0,
                    "observed_cleanup_failures": 0,
                    "observed_circuit_breaker_rejections": 0,
                },
                "memory": {
                    "evaluated": True,
                    "monotonic_growth_detected": False,
                    "baseline_bytes": 100,
                    "final_bytes": 100,
                    "peak_bytes": 200,
                    "epoch_sample_counts": [12] * 8,
                    "max_epoch_growth_fraction": 0.05,
                    "global_growth_fraction": 0.05,
                },
            },
            "windows_coexistence": {
                "status": "pass",
                "greek_utterances": 100,
                "english_utterances": 100,
                "greek_language_correct": 95,
                "english_language_correct": 95,
                "greek_reference_words": 1000,
                "greek_word_errors": 100,
                "english_reference_words": 1000,
                "english_word_errors": 100,
                "forced_cpu_fallbacks": 5,
                "overlapping_utterances": 80,
                "baseline_p95_seconds": 1,
                "coexistence_p95_seconds": 1.1,
                "asr_degradation_ratio": 0.1,
                "baseline_llm_p95_seconds": 5,
                "coexistence_llm_p95_seconds": 5.5,
                "llm_regression_ratio": 0.1,
                "minimum_free_vram_mib": 2048,
                "asr_runtime_identity_sha256": self.generic_gate[
                    "asr_runtime_identity_digest"
                ],
                "lost_final_transcripts": 0,
                "duplicate_final_transcripts": 0,
                "stale_results": 0,
                "oom": 0,
                "crashes": 0,
                "throttle_reasons": 0,
            },
            "privacy": {
                "runner_content_payloads_persisted": 0,
                "runner_content_payloads_logged": 0,
                "evidence_content_free": True,
                "diagnostic_sanitization_passed": True,
            },
        }

    def write_pass(self) -> dict[str, object]:
        evidence = self.evidence()
        durable_replace(self.root / PROFILE_EVIDENCE_RELATIVE, evidence)
        evidence_sha = hashlib.sha256(
            (self.root / PROFILE_EVIDENCE_RELATIVE).read_bytes()
        ).hexdigest()
        report = copy.deepcopy(evidence)
        report["schema"] = REPORT_SCHEMA
        report["authoritative"] = True
        report["production_ready"] = True
        report["identity"]["profile_evidence_sha256"] = evidence_sha
        durable_replace(self.manifest.profile_qualification_report, report)
        return report


class ProfileReportTests(unittest.TestCase):
    def test_qualification_tokenizer_returns_only_count_and_real_id_digest(self) -> None:
        class TokenEngine:
            def ready(self):
                return True

            def prepare(self, request, deadline):
                return PreparedPrompt((7, 11, 13), "sensitive rendered prompt", request)

            def cleanup(self, deadline):
                return None

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            fixture = QualifiedPrimaryFixture(Path(directory))
            production = _PrimaryQualificationProduction(
                fixture.manifest,
                "x" * 40,
                lambda _manifest: TokenEngine(),
                engine=TokenEngine(),
            )
            result = production.qualification_tokenize(object())
            self.assertEqual(
                result,
                {
                    "token_count": 3,
                    "token_ids_sha256": hashlib.sha256(
                        canonical_bytes([7, 11, 13])
                    ).hexdigest(),
                },
            )
            self.assertNotIn("prompt", result)
            self.assertNotIn("token_ids", result)
            self.assertFalse(
                hasattr(
                    Production(
                        fixture.manifest,
                        "x" * 40,
                        lambda _manifest: TokenEngine(),
                    ),
                    "qualification_tokenize",
                )
            )

    def test_exact_identity_bound_pass_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualifiedPrimaryFixture(Path(directory))
            fixture.write_pass()
            report = load_valid_profile_report(
                fixture.manifest,
                fixture.current_identity,
                fixture.generic_report,
            )
            self.assertTrue(report["authoritative"])
            self.assertTrue(report["production_ready"])
            self.assertTrue(
                profile_report_matches(
                    fixture.manifest,
                    fixture.current_identity,
                    fixture.generic_report,
                )
            )

    def test_report_evidence_generic_and_runner_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualifiedPrimaryFixture(Path(directory))
            original = fixture.write_pass()
            for mutation in ("report", "evidence", "generic", "runner"):
                durable_replace(fixture.manifest.profile_qualification_report, original)
                fixture.write_pass()
                if mutation == "report":
                    report = copy.deepcopy(original)
                    report["identity"]["tokenizer_identity_sha256"] = "0" * 64
                    durable_replace(fixture.manifest.profile_qualification_report, report)
                elif mutation == "evidence":
                    evidence = json.loads(
                        (fixture.root / PROFILE_EVIDENCE_RELATIVE).read_text()
                    )
                    evidence["performance"]["metrics"]["ttft_p95_seconds"] = 2
                    durable_replace(fixture.root / PROFILE_EVIDENCE_RELATIVE, evidence)
                elif mutation == "generic":
                    raw = fixture.manifest.qualification_report.read_bytes()
                    fixture.manifest.qualification_report.write_bytes(
                        raw.replace(b"{", b"{ ", 1)
                    )
                else:
                    fixture.profile_runner.write_bytes(b"changed")
                self.assertFalse(
                    profile_report_matches(
                        fixture.manifest,
                        fixture.current_identity,
                        fixture.generic_report,
                    ),
                    mutation,
                )
                if mutation == "runner":
                    fixture.profile_runner.write_bytes(b"profile-runner")
                durable_replace(
                    fixture.manifest.qualification_report, fixture.generic_report
                )

    def test_generic_fixture_power_and_asr_runtime_are_cross_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualifiedPrimaryFixture(Path(directory))
            mutations = []

            changed_fixture = copy.deepcopy(fixture.generic_report)
            changed_fixture["results"]["platform_gate"]["fixture"][
                "schedule"
            ] = "changed"
            mutations.append((changed_fixture, "fixture identity"))

            changed_power = copy.deepcopy(fixture.generic_report)
            changed_power["results"]["platform_gate"][
                "baseline_power_identity_digest"
            ] = "0" * 64
            mutations.append((changed_power, "fixture identity"))

            changed_asr = copy.deepcopy(fixture.generic_report)
            changed_asr["results"]["platform_gate"][
                "asr_runtime_identity_digest"
            ] = "0" * 64
            mutations.append((changed_asr, "ASR runtime identity"))

            for generic, error in mutations:
                with self.subTest(error=error):
                    durable_replace(
                        fixture.manifest.qualification_report, generic
                    )
                    parsed = read_report(fixture.manifest.qualification_report)
                    assert isinstance(parsed, dict)
                    with self.assertRaisesRegex(ValueError, error):
                        expected_profile_identity(
                            fixture.manifest,
                            fixture.current_identity,
                            parsed,
                            profile_runner_sha256="1" * 64,
                            profile_evidence_sha256="2" * 64,
                            tokenizer_identity_sha256="3" * 64,
                        )

    def test_native_thresholds_and_sanitization_are_not_claimable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualifiedPrimaryFixture(Path(directory))
            evidence = fixture.evidence()
            evidence["soak"]["monotonic_seconds_observed"] = 60
            with self.assertRaisesRegex(ValueError, "soak gate"):
                validate_profile_document(
                    evidence,
                    fixture.manifest,
                    fixture.current_identity,
                    fixture.generic_report,
                    evidence=True,
                )

    def test_local_operator_attestation_disclosure_is_exact_and_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualifiedPrimaryFixture(Path(directory))
            mutations = []

            missing_level = fixture.evidence()
            missing_level.pop("attestation_level")
            mutations.append((missing_level, "profile report"))

            overstated_level = fixture.evidence()
            overstated_level["attestation_level"] = "hardware-backed"
            mutations.append((overstated_level, "identity or sanitization"))

            hidden_operator_trust = fixture.evidence()
            hidden_operator_trust["operator_trust_required"] = False
            mutations.append(
                (hidden_operator_trust, "identity or sanitization")
            )

            for evidence, error in mutations:
                with self.subTest(error=error):
                    with self.assertRaisesRegex(ValueError, error):
                        validate_profile_document(
                            evidence,
                            fixture.manifest,
                            fixture.current_identity,
                            fixture.generic_report,
                            evidence=True,
                        )

    def test_performance_and_asr_evidence_are_recomputed_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualifiedPrimaryFixture(Path(directory))

            mutations = []
            outside_bin = fixture.evidence()
            outside_bin["performance"]["cases"][0]["rendered_token_count"] = 1
            mutations.append((outside_bin, "performance case"))

            duplicate_serial = fixture.evidence()
            duplicate_serial["performance"]["cases"][1]["generation_serial"] = 1
            mutations.append((duplicate_serial, "performance case"))

            skipped_serial = fixture.evidence()
            skipped_serial["performance"]["cases"][0]["generation_serial"] = 2
            mutations.append((skipped_serial, "performance case"))

            forged_tokenizer_identity = fixture.evidence()
            forged_tokenizer_identity["performance"]["native_token_validation"][
                "tokenizer_identity_sha256"
            ] = "0" * 64
            forged_tokenizer_identity["identity"][
                "tokenizer_identity_sha256"
            ] = "0" * 64
            mutations.append((forged_tokenizer_identity, "performance gate"))

            dishonest_projection = fixture.evidence()
            dishonest_projection["performance"]["cases"][0][
                "projected_128_token_e2e_seconds"
            ] = 1 + 128 / 32
            mutations.append((dishonest_projection, "projected 128-token"))

            dishonest_decode = fixture.evidence()
            dishonest_decode["performance"]["cases"][0][
                "backend_decode_seconds"
            ] = 3
            mutations.append((dishonest_decode, "backend performance telemetry"))

            impossible_timing_sum = fixture.evidence()
            timing_case = impossible_timing_sum["performance"]["cases"][0]
            timing_case["ttft_seconds"] = 2.1
            timing_case["backend_prompt_seconds"] = 2
            timing_case["backend_decode_seconds"] = 2
            timing_case["request_to_engine_final_seconds"] = 3
            timing_case["decode_tokens_per_second"] = 64
            timing_case["projected_128_token_e2e_seconds"] = 2.1 + 127 / 64
            mutations.append(
                (impossible_timing_sum, "backend performance telemetry")
            )

            forged_cold_load_aggregate = fixture.evidence()
            forged_cold_load_aggregate["performance"]["metrics"][
                "cold_load_to_ready_p95_seconds"
            ] = 99
            mutations.append(
                (forged_cold_load_aggregate, "cold-load aggregate mismatch")
            )

            slow_cold_load = fixture.evidence()
            slow_cold_load["performance"][
                "cold_load_to_ready_samples_seconds"
            ] = [121] * 8
            slow_cold_load["performance"]["metrics"][
                "cold_load_to_ready_p95_seconds"
            ] = 121
            mutations.append((slow_cold_load, "performance gate"))

            extra_cold_load = fixture.evidence()
            extra_cold_load["performance"][
                "cold_load_to_ready_samples_seconds"
            ].append(100)
            mutations.append((extra_cold_load, "cold-load samples"))

            old_aggregate = fixture.evidence()
            metrics = old_aggregate["performance"]["metrics"]
            metrics["p95_end_to_end_128_seconds"] = metrics.pop(
                "projected_128_token_e2e_p95_seconds"
            )
            mutations.append((old_aggregate, "performance metrics"))

            excessive_wer = fixture.evidence()
            excessive_wer["windows_coexistence"]["greek_word_errors"] = 201
            mutations.append((excessive_wer, "Windows coexistence"))

            wrong_fallback_count = fixture.evidence()
            wrong_fallback_count["windows_coexistence"][
                "forced_cpu_fallbacks"
            ] = 4
            mutations.append((wrong_fallback_count, "Windows coexistence"))

            wrong_epoch_count = fixture.evidence()
            wrong_epoch_count["soak"]["memory"]["epoch_sample_counts"].append(12)
            mutations.append((wrong_epoch_count, "soak gate"))

            incomplete_restart_epoch = fixture.evidence()
            incomplete_restart_epoch["soak"]["cycles_completed"] = 97
            mutations.append((incomplete_restart_epoch, "soak gate"))

            for evidence, error in mutations:
                with self.subTest(error=error):
                    with self.assertRaisesRegex(ValueError, error):
                        validate_profile_document(
                            evidence,
                            fixture.manifest,
                            fixture.current_identity,
                            fixture.generic_report,
                            evidence=True,
                        )
            evidence = fixture.evidence()
            evidence["quality"]["content"] = "forbidden"
            with self.assertRaisesRegex(ValueError, "identity or sanitization"):
                validate_profile_document(
                    evidence,
                    fixture.manifest,
                    fixture.current_identity,
                    fixture.generic_report,
                    evidence=True,
                )

    def test_content_free_readiness_cli_fails_fast_on_missing_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualifiedPrimaryFixture(Path(directory))
            with (
                patch("jarvis_inference.cli.assert_runtime_platform"),
                patch("jarvis_inference.cli.verify_artifacts"),
                patch(
                    "jarvis_inference.cli.runtime_identity",
                    return_value=fixture.current_identity,
                ),
            ):
                with self.assertRaisesRegex(Exception, "profile qualification"):
                    verify_readiness(fixture.manifest)
                fixture.write_pass()
                result = verify_readiness(fixture.manifest)
            self.assertEqual(
                result,
                {
                    "production_ready": True,
                    "manifest_digest": fixture.manifest.digest,
                    "attestation_level": ATTESTATION_LEVEL,
                    "operator_trust_required": OPERATOR_TRUST_REQUIRED,
                },
            )

    def test_non_primary_legacy_manifest_does_not_gain_a_new_report_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "llama-server.exe"
            model = root / "model.gguf"
            executable.write_bytes(b"server")
            model.write_bytes(b"model")
            document = {
                "manifest_schema_version": 1,
                "alias": "jarvis-local",
                "platform": "windows-x86_64",
                "backend": {
                    "type": "llama-server",
                    "llama_cpp_commit": "a" * 40,
                    "build_id": "legacy",
                    "executable": _artifact(executable, root),
                    "dlls": [],
                    "parameters": {},
                },
                "model": {
                    "format": "gguf",
                    "context_window": 16,
                    "artifacts": [_artifact(model, root)],
                },
                "python": {},
                "capabilities": {
                    "text": True,
                    "stream": True,
                    "tools": False,
                    "json_object": False,
                    "json_schema": False,
                },
                "qualification_report": "qualification-report.json",
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            manifest = load_manifest(path)
            self.assertIsNone(manifest.profile_qualification_report)
            self.assertTrue(profile_report_matches(manifest, {}, {}))


if __name__ == "__main__":
    unittest.main()
