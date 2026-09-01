#!/usr/bin/env python3
"""Create or verify the strict, offline Cyrus Windows handoff template."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath


SCHEMA = "cyrus.windows-handoff/1"
MAX_MANIFEST_BYTES = 512 * 1024
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_MAC_PATH_PREFIXES = (
    "/Applications/",
    "/Library/",
    "/System/",
    "/Users/",
    "/Volumes/",
    "/private/",
    "file:///",
    "~/",
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN (?:ENCRYPTED )?PRIVATE KEY-----"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}={0,2}\b"),
    re.compile(r"\b(?:sk|sk-or-v1)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)
_SECRET_KEY = re.compile(
    r"(?i)(?:^|_)(?:api_?key|auth_?key|password|private_?key|secret|token)(?:$|_)"
)


class HandoffError(ValueError):
    """Content-free validation failure with a stable machine-readable code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _locked_artifact(
    relative_path: str,
    size_bytes: int,
    sha256: str,
    source_revision: str,
) -> dict[str, object]:
    return {
        "identity_status": "locked",
        "relative_path": relative_path,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "source_revision": source_revision,
    }


def _pending_artifact(relative_path: str) -> dict[str, object]:
    return {
        "identity_status": "pending_native",
        "relative_path": relative_path,
        "size_bytes": None,
        "sha256": None,
        "source_revision": None,
    }


GEMMA_REVISION = "29d097773436b69ff9feafd636ab4cf873786537"
WHISPER_BASE_REVISION = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
WHISPER_TURBO_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
GEMMA_PROMPT_PROFILE = {
    "id": "google-gemma4-12b-it-qat-q4_0-v1",
    "model_repository": "google/gemma-4-12B-it-qat-q4_0-gguf",
    "model_revision": GEMMA_REVISION,
    "model_artifact": {
        "filename": "gemma-4-12b-it-qat-q4_0.gguf",
        "size_bytes": 6_975_879_296,
        "sha256": "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b",
    },
    "embedded_chat_template": {
        "metadata_key": "tokenizer.chat_template",
        "size_bytes": 18_683,
        "sha256": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
    },
    "renderer": {
        "id": "cyrus.gemma4-appliance-renderer/1",
        "sha256": "3ce59b2dadfeeff369b337da785deb096ca50d8a2182570cc9db36f7f63a1951",
    },
    "tool_parser": {
        "id": "cyrus.gemma4-single-tool-parser/1",
        "sha256": "e118a2ab2d39113fd2a18964f8f6ade87692afeed26c87d95d87a0dcb5c6afd4",
    },
    "thinking": False,
}

LOCKED_ARTIFACTS: dict[str, dict[str, object]] = {
    "gemma_primary_model": _locked_artifact(
        "models/gemma-4-12b-it-qat-q4_0.gguf",
        6_975_879_296,
        "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b",
        GEMMA_REVISION,
    ),
    "whisper_base_config": _locked_artifact(
        "models/faster-whisper-base/config.json",
        2_309,
        "56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a",
        WHISPER_BASE_REVISION,
    ),
    "whisper_base_model": _locked_artifact(
        "models/faster-whisper-base/model.bin",
        145_217_532,
        "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
        WHISPER_BASE_REVISION,
    ),
    "whisper_base_tokenizer": _locked_artifact(
        "models/faster-whisper-base/tokenizer.json",
        2_203_239,
        "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
        WHISPER_BASE_REVISION,
    ),
    "whisper_base_vocabulary": _locked_artifact(
        "models/faster-whisper-base/vocabulary.txt",
        459_861,
        "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
        WHISPER_BASE_REVISION,
    ),
    "whisper_config": _locked_artifact(
        "models/faster-whisper-large-v3-turbo/config.json",
        2_263,
        "b0253ea6c0d3bea6b1e19e91a02acfd3b53f4467362efcb5a3e6b16c9b3a9b7e",
        WHISPER_TURBO_REVISION,
    ),
    "whisper_model": _locked_artifact(
        "models/faster-whisper-large-v3-turbo/model.bin",
        1_617_884_929,
        "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
        WHISPER_TURBO_REVISION,
    ),
    "whisper_preprocessor": _locked_artifact(
        "models/faster-whisper-large-v3-turbo/preprocessor_config.json",
        340,
        "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
        WHISPER_TURBO_REVISION,
    ),
    "whisper_tokenizer": _locked_artifact(
        "models/faster-whisper-large-v3-turbo/tokenizer.json",
        2_710_337,
        "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd",
        WHISPER_TURBO_REVISION,
    ),
    "whisper_vocabulary": _locked_artifact(
        "models/faster-whisper-large-v3-turbo/vocabulary.json",
        1_068_114,
        "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1",
        WHISPER_TURBO_REVISION,
    ),
}

PENDING_ARTIFACTS: dict[str, dict[str, object]] = {
    "python_runtime": _pending_artifact("runtime/python/python.exe"),
    "cuda_runtime_bundle_manifest": _pending_artifact(
        "build-identities/cuda-runtime-manifest.json"
    ),
    "llama_server_executable": _pending_artifact("runtime/llama/llama-server.exe"),
    "llama_server_dll_manifest": _pending_artifact("runtime/llama/dll-manifest.json"),
    "windows_inference_manifest": _pending_artifact("inference/manifest.json"),
    "native_fixture_runner": _pending_artifact(
        "qualification/native-fixture-runner.exe"
    ),
    "asr_qualification_collector": _pending_artifact(
        "qualification/cyrus-windows-asr-collector.exe"
    ),
    "primary_profile_runner": _pending_artifact(
        "qualification/cyrus-windows-primary-profile-runner.exe"
    ),
    "qualification_runner_build_manifest": _pending_artifact(
        "build-identities/qualification-runners-manifest.json"
    ),
    "windows_asr_audio_corpus_manifest": _pending_artifact(
        "qualification/audio-corpus-v1.json"
    ),
    "windows_power_profile": _pending_artifact(
        "qualification/power-profile-v1.json"
    ),
    "windows_asr_config": _pending_artifact(
        "qualification/asr-config-v1.json"
    ),
    "silero_vad_model": _pending_artifact(
        "models/silero-vad/silero_vad.onnx"
    ),
    "qwen3_14b_comparison": _pending_artifact("models/Qwen3-14B-Q4_K_M.gguf"),
    "voice_wheel": _pending_artifact("packages/jarvis_voice.whl"),
    "orchestrator_wheel": _pending_artifact("packages/jarvis_orchestrator.whl"),
    "inference_wheel": _pending_artifact("packages/jarvis_inference.whl"),
    "anamnesis_wheel": _pending_artifact("packages/anamnesis.whl"),
    "research_wheel": _pending_artifact("packages/jarvis_research.whl"),
    "windows_distribution_manifest": _pending_artifact(
        "distribution/JARVIS Voice/distribution-manifest.json"
    ),
}

NATIVE_GATE_NAMES = (
    "hardware_inventory",
    "dependency_matrix_operator_approval",
    "artifact_materialization",
    "llm_profile_lock",
    "gemma_renderer_and_parser",
    "llm_bilingual_quality",
    "llm_structured_output",
    "llm_performance",
    "llm_eight_hour_soak",
    "asr_cpu_baseline",
    "asr_cuda_turn_scoped",
    "cuda_llm_asr_coexistence",
    "voice_two_hundred_turns",
    "voice_acoustic_barge_in",
    "voice_device_and_sleep_recovery",
    "windows_signed_package",
    "operational_synthetic",
    "operational_live_model",
    "operational_restart",
    "operational_backup_restore",
    "operational_twenty_four_hour_soak",
    "signed_operational_aggregate",
)


def handoff_template(frozen_commit: str) -> dict[str, object]:
    """Return the one canonical pre-Windows handoff for a frozen source commit."""

    if not isinstance(frozen_commit, str) or _COMMIT.fullmatch(frozen_commit) is None:
        raise HandoffError("handoff.frozen_commit_invalid")
    artifacts = copy.deepcopy(LOCKED_ARTIFACTS)
    artifacts.update(copy.deepcopy(PENDING_ARTIFACTS))
    return {
        "schema": SCHEMA,
        "frozen_source": {
            "commit": frozen_commit,
            "protocol_changes_during_bringup": "forbidden",
            "baseline_platform": "macos",
            "baseline_status": "technical_beta",
            "untracked_files_included": False,
        },
        "host_policy": {
            "runtime_scope": "single_machine",
            "components_must_share_host": True,
            "windows_role": "independent_primary",
            "macos_role": "independent_portable_fallback",
            "live_cross_host_components": False,
            "cross_host_rpc": False,
            "state_transfer_between_hosts": False,
            "credential_reuse_between_hosts": False,
        },
        "target": {
            "platform": "windows",
            "os_family": "Windows 11",
            "architecture": "x86_64",
            "gpu": "NVIDIA GeForce RTX 5080",
            "gpu_vram_mib": 16_384,
            "python_major_minor": "3.11",
            "native_identity": {
                "status": "pending_native",
                "windows_build": None,
                "python_version": None,
                "nvidia_driver_version": None,
                "cuda_runtime_version": None,
                "cudnn_version": None,
                "ctranslate2_version": None,
                "llama_cpp_commit": None,
                "llama_server_build_id": None,
            },
        },
        "profiles": {
            "llm": {
                "profile_id": "local-primary",
                "public_model_alias": "jarvis-local",
                "model_repository": "google/gemma-4-12B-it-qat-q4_0-gguf",
                "model_revision": GEMMA_REVISION,
                "model_artifact": "gemma_primary_model",
                "format": "GGUF",
                "quantization": "QAT_Q4_0",
                "runtime": "llama-server",
                "runtime_identity": "pending_native",
                # The inference wheel owns these implementations. There are no
                # separately materialized runtime/adapters files and serving
                # never evaluates the GGUF's embedded Jinja.
                "prompt_profile": copy.deepcopy(GEMMA_PROMPT_PROFILE),
                "context_tokens": 32_768,
                "completion_tokens": 4_096,
                "capabilities": {
                    "text": True,
                    "stream": True,
                    "tools": True,
                    "json_object": True,
                    "json_schema": True,
                },
                "status": "selected_not_qualified",
                "comparison_control": {
                    "model_repository": "Qwen/Qwen3-14B-GGUF",
                    "file": "Qwen3-14B-Q4_K_M.gguf",
                    "artifact": "qwen3_14b_comparison",
                    "status": "disabled_comparison_only",
                    "activation_requires_new_profile_and_qualification": True,
                },
            },
            "asr": {
                "cpu_baseline": {
                    "profile_id": "CpuBaseline",
                    "model_profile": "base",
                    "model_repository": "Systran/faster-whisper-base",
                    "model_revision": WHISPER_BASE_REVISION,
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
                "model_profile": "large-v3-turbo",
                "model_repository": "dropbox-dash/faster-whisper-large-v3-turbo",
                "model_revision": WHISPER_TURBO_REVISION,
                "runtime": "faster-whisper/ctranslate2",
                "runtime_version": "faster-whisper==1.2.1",
                "cpu_compute_type": "int8",
                "bringup_execution_profile": "cpu",
                "production_candidate_execution_profile": "cuda",
                "cuda_compute_type": "int8_float16",
                "cuda_residency": "turn_scoped",
                "cuda_device_index": 0,
                "minimum_free_vram_mib": 2_048,
                "admission_free_vram_mib": 4_096,
                "fallback_to_cpu": True,
                "status": "selected_not_qualified",
            },
            "tts": {
                "bringup_backend": "sapi5",
                "active_backend_at_handoff": "sapi5",
                "english_candidate": "pocket_tts",
                "english_candidate_status": "disabled_pending_native_qualification",
                "unsupported_locale_fallback": "sapi5",
                "status": "selected_not_qualified",
            },
        },
        "artifact_identities": artifacts,
        "acceptance": {
            "llm_quality": {
                "greek_cases": 60,
                "english_cases": 60,
                "minimum_pass_rate_per_language": 0.90,
                "maximum_language_gap_percentage_points": 10,
                "maximum_critical_failures": 0,
            },
            "llm_streaming": {
                "greek_cases": 20,
                "english_cases": 20,
                "required_passes": 40,
            },
            "llm_json_schema": {
                "cases": 20,
                "required_parse_and_validate": 20,
                "maximum_extra_text_cases": 0,
            },
            "llm_tools": {
                "cases": 20,
                "required_wire_and_schema_valid": 20,
                "minimum_exact_tool_and_arguments": 19,
            },
            "llm_performance": {
                "prompts": 100,
                "greek_prompts": 50,
                "english_prompts": 50,
                "p95_ttft_seconds_maximum": 3.0,
                "p95_128_token_e2e_seconds_maximum": 10.0,
                "p5_decode_tokens_per_second_minimum": 25.0,
                "p95_cold_load_seconds_maximum": 120.0,
                "required_timed_successes": 100,
            },
            "llm_soak": {
                "duration_seconds_minimum": 28_800,
                "cycles_minimum": 96,
                "full_restarts_minimum": 8,
                "maximum_unexpected_oom_crash_hang_or_identity_drift": 0,
            },
            "asr_native": {
                "clean_greek_utterances_minimum": 100,
                "clean_english_utterances_minimum": 100,
                "minimum_language_accuracy_per_locale": 0.95,
                "maximum_word_error_rate_per_locale": 0.20,
                "no_speech_turns_minimum": 5,
                "interruptions_minimum": 10,
                "forced_cpu_fallbacks_minimum": 5,
                "worker_restart_simulations_minimum": 5,
            },
            "cuda_coexistence": {
                "minimum_free_vram_mib": 2_048,
                "maximum_asr_p95_degradation_fraction": 0.15,
                "maximum_llm_p95_regression_fraction": 0.15,
                "maximum_oom_crashes_lost_turns_or_stale_outputs": 0,
            },
            "voice_native": {
                "real_voice_turns_minimum": 200,
                "acoustic_barge_in_samples_minimum": 20,
                "ptt_to_acoustic_silence_ms_strict_maximum": 200.0,
                "microphone_disconnect_recoveries_minimum": 1,
                "sleep_resume_recoveries_minimum": 1,
                "soak_seconds_minimum": 28_800,
                "rss_samples_minimum": 97,
                "rss_slope_bytes_per_hour_maximum": 1_048_576,
                "rss_end_growth_bytes_maximum": 33_554_432,
                "maximum_duplicates_stuck_ptt_or_crashes": 0,
            },
            "operational": {
                "synthetic_turns_minimum": 1_000,
                "real_local_model_turns_minimum": 100,
                "restart_gate_required": True,
                "backup_corruption_restore_gate_required": True,
                "whole_system_soak_seconds_minimum": 86_400,
                "maximum_database_bytes": 536_870_912,
                "signed_aggregate_required": True,
            },
        },
        "native_gates": {name: "pending" for name in NATIVE_GATE_NAMES},
        "safety_assertions": {
            "bitlocker_required": True,
            "defender_required": True,
            "firewall_required": True,
            "run_as_administrator": False,
            "loopback_bindings_only": True,
            "owner_only_dacl_for_private_state": True,
            "secrets_embedded": False,
            "macos_secrets_migrated": False,
            "runtime_downloads_allowed": False,
            "machine_wide_path_mutation_allowed": False,
            "tools_actions_memory_research_cloud_enabled_during_bringup": False,
            "native_qualification_complete": False,
            "production_ready": False,
        },
    }


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _walk(value: object, *, key: str | None = None, depth: int = 0) -> None:
    if depth > 64:
        raise HandoffError("handoff.manifest_too_deep")
    if key is not None and _SECRET_KEY.search(key) is not None:
        if isinstance(value, str) and value:
            raise HandoffError("handoff.secret_detected")
        if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)) and value:
            raise HandoffError("handoff.secret_detected")
    if isinstance(value, str):
        lowered = value.lower()
        if any(lowered.startswith(prefix.lower()) for prefix in _MAC_PATH_PREFIXES):
            raise HandoffError("handoff.macos_absolute_path_detected")
        if any(pattern.search(value) is not None for pattern in _SECRET_VALUE_PATTERNS):
            raise HandoffError("handoff.secret_detected")
        return
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise HandoffError("handoff.schema_invalid")
            _walk(child, key=child_key, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _walk(child, depth=depth + 1)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise HandoffError("handoff.non_finite_number")


def _valid_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    if _WINDOWS_DRIVE_PATH.match(value) is not None:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def _validate_artifacts(raw: object) -> None:
    if not isinstance(raw, Mapping):
        raise HandoffError("handoff.artifact_identities_invalid")
    expected_names = set(LOCKED_ARTIFACTS) | set(PENDING_ARTIFACTS)
    if set(raw) != expected_names:
        raise HandoffError("handoff.artifact_identities_invalid")
    keys = {"identity_status", "relative_path", "size_bytes", "sha256", "source_revision"}
    for artifact in raw.values():
        if not isinstance(artifact, Mapping) or set(artifact) != keys:
            raise HandoffError("handoff.artifact_identity_invalid")
        if not _valid_relative_path(artifact.get("relative_path")):
            raise HandoffError("handoff.artifact_path_invalid")
        status = artifact.get("identity_status")
        if status == "locked":
            size = artifact.get("size_bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise HandoffError("handoff.artifact_identity_invalid")
            if not isinstance(artifact.get("sha256"), str) or _SHA256.fullmatch(artifact["sha256"]) is None:
                raise HandoffError("handoff.artifact_identity_invalid")
            revision = artifact.get("source_revision")
            if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
                raise HandoffError("handoff.artifact_identity_invalid")
        elif status == "pending_native":
            if any(artifact.get(name) is not None for name in ("size_bytes", "sha256", "source_revision")):
                raise HandoffError("handoff.pending_identity_fabricated")
        else:
            raise HandoffError("handoff.artifact_identity_invalid")


def validate_handoff_manifest(
    raw: object,
    *,
    expected_commit: str,
) -> dict[str, object]:
    """Fail closed unless *raw* is the exact safe pre-native handoff contract."""

    if not isinstance(expected_commit, str) or _COMMIT.fullmatch(expected_commit) is None:
        raise HandoffError("handoff.expected_commit_invalid")
    if not isinstance(raw, Mapping):
        raise HandoffError("handoff.schema_invalid")
    _walk(raw)
    top_keys = {
        "schema",
        "frozen_source",
        "host_policy",
        "target",
        "profiles",
        "artifact_identities",
        "acceptance",
        "native_gates",
        "safety_assertions",
    }
    if set(raw) != top_keys or raw.get("schema") != SCHEMA:
        raise HandoffError("handoff.schema_invalid")
    frozen = raw.get("frozen_source")
    if not isinstance(frozen, Mapping) or frozen.get("commit") != expected_commit:
        raise HandoffError("handoff.frozen_commit_mismatch")
    expected = handoff_template(expected_commit)
    section_codes = {
        "frozen_source": "handoff.frozen_source_invalid",
        "host_policy": "handoff.single_machine_policy_invalid",
        "target": "handoff.target_invalid",
        "profiles": "handoff.profile_selection_invalid",
        "acceptance": "handoff.acceptance_thresholds_invalid",
        "native_gates": "handoff.native_gates_not_pending",
        "safety_assertions": "handoff.safety_assertions_invalid",
    }
    _validate_artifacts(raw.get("artifact_identities"))
    if raw.get("artifact_identities") != expected["artifact_identities"]:
        raise HandoffError("handoff.artifact_identities_invalid")
    for name, code in section_codes.items():
        if raw.get(name) != expected[name]:
            raise HandoffError(code)
    digest = manifest_sha256(raw)
    return {
        "schema": SCHEMA,
        "handoff_preparation_ready": True,
        "native_qualification_complete": False,
        "production_ready": False,
        "native_gates_pending": len(NATIVE_GATE_NAMES),
        "manifest_sha256": digest,
        "code": "handoff.windows_preparation_ready",
        "network_used": False,
        "installation_performed": False,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HandoffError("handoff.duplicate_json_key")
        result[key] = value
    return result


def load_manifest(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HandoffError("handoff.manifest_unreadable") from exc
    if path.is_symlink() or not path.is_file() or metadata.st_size > MAX_MANIFEST_BYTES:
        raise HandoffError("handoff.manifest_unreadable")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                HandoffError("handoff.non_finite_number")
            ),
        )
    except HandoffError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise HandoffError("handoff.manifest_invalid_json") from exc
    if not isinstance(value, dict):
        raise HandoffError("handoff.schema_invalid")
    return value


def write_new_template(path: Path, value: Mapping[str, object]) -> None:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists() or path.is_symlink():
        raise HandoffError("handoff.output_invalid")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise HandoffError("handoff.output_invalid") from exc
    except OSError as exc:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        raise HandoffError("handoff.output_write_failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _absolute_without_resolving(path: Path) -> Path:
    """Make a CLI path absolute without following a final symlink."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="write the canonical pending template")
    template.add_argument("--frozen-commit", required=True)
    template.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="validate an existing handoff template")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == "template":
            value = handoff_template(args.frozen_commit)
            write_new_template(_absolute_without_resolving(args.output), value)
            report = validate_handoff_manifest(value, expected_commit=args.frozen_commit)
            report["code"] = "handoff.windows_template_created"
        else:
            value = load_manifest(_absolute_without_resolving(args.manifest))
            report = validate_handoff_manifest(value, expected_commit=args.expected_commit)
    except HandoffError as exc:
        report = {
            "schema": SCHEMA,
            "handoff_preparation_ready": False,
            "native_qualification_complete": False,
            "production_ready": False,
            "code": exc.code,
            "network_used": False,
            "installation_performed": False,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["handoff_preparation_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
