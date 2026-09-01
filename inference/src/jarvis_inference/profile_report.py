from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from .errors import ApplianceError
from .gemma4 import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODEL_SHA256,
    PROFILE_ID,
    RENDERER_SHA256,
    TOOL_PARSER_SHA256,
    uses_gemma4_profile,
)
from .limits import BODY_BYTES
from .manifest import Manifest, hash_file, resolve_local
from .primary_profile import (
    PRIMARY_CORPUS_SHA256,
    PRIMARY_QUALITY_SUITE_VERSION,
    PRIMARY_SYSTEM_PROMPT,
    load_primary_corpus,
)
from .profile_performance import validate_performance
from .strict_json import canonical_bytes, parse_body
from .windows_primary_fixture import load_fixture


REPORT_SCHEMA = "cyrus.inference-primary-profile-qualification/1"
EVIDENCE_SCHEMA = "cyrus.inference-primary-profile-evidence/1"
ATTESTATION_LEVEL = "identity-bound-local-operator"
OPERATOR_TRUST_REQUIRED = True
PROFILE_REPORT_FILENAME = "qualification-profile-local-primary.json"
PROFILE_EVIDENCE_RELATIVE = (
    "qualification/qualification-evidence-profile-local-primary.json"
)
PROFILE_RUNNER_RELATIVE = "qualification/cyrus-windows-primary-profile-runner.exe"
PRIMARY_CORPUS_RELATIVE = "qualification/local-primary-quality-v1.json"
PRIMARY_BASE_CORPUS_RELATIVE = "qualification/local-fallback-quality-v1.json"
PERFORMANCE_FIXTURE_RELATIVE = (
    "qualification/windows-primary-performance-soak-v1.json"
)

_SHA = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_CHALLENGE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_ZERO_EVENTS = (
    "observed_request_failures",
    "observed_unexpected_process_terminations",
    "observed_request_timeouts",
    "observed_cleanup_failures",
    "observed_circuit_breaker_rejections",
)
_GENERIC_ASSET_PATHS = {
    "fixture": "qualification/windows-asr-coexistence-v1.json",
    "collector": "qualification/cyrus-windows-asr-collector.exe",
    "audio_corpus": "qualification/audio-corpus-v1.json",
    "power_profile": "qualification/power-profile-v1.json",
    "asr_config": "qualification/asr-config-v1.json",
}
_FORBIDDEN_CONTENT_KEYS = {
    "content",
    "messages",
    "output",
    "prompt",
    "raw_output",
    "response",
    "tool_arguments",
}


class ProfileQualificationFailure(ValueError):
    """A sanitized, fail-closed primary-profile report failure."""


def profile_report_required(manifest: Manifest) -> bool:
    return manifest.platform == "windows-x86_64" and uses_gemma4_profile(manifest)


def _fail(message: str) -> ProfileQualificationFailure:
    return ProfileQualificationFailure(message)


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise _fail(f"invalid {name} shape")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise _fail(f"invalid {name}")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"invalid {name}")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise _fail(f"invalid {name}")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise _fail(f"invalid {name}")
    return result


def _safe_bytes(path: Path, name: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise _fail(f"missing {name}") from exc
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = bool(
        os.name == "nt"
        and attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or reparse
        or info.st_size <= 0
        or info.st_size > BODY_BYTES
    ):
        raise _fail(f"unsafe {name}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _fail(f"unreadable {name}") from exc
    if len(raw) != info.st_size:
        raise _fail(f"unstable {name}")
    return raw


def _contains_content_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key.casefold() in _FORBIDDEN_CONTENT_KEYS
            or _contains_content_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_content_key(child) for child in value)
    return False


def _generic_context(
    manifest: Manifest,
    current_identity: dict[str, Any],
    generic_report: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    raw = _safe_bytes(manifest.qualification_report, "generic qualification report")
    try:
        parsed = parse_body(raw)
    except ApplianceError as exc:
        raise _fail("invalid generic qualification report") from exc
    if parsed != generic_report:
        raise _fail("generic qualification report changed")
    if (
        generic_report.get("report_schema_version") != 1
        or generic_report.get("status") != "pass"
        or generic_report.get("identity") != current_identity
    ):
        raise _fail("generic qualification is not a current pass")
    evidence_sha = _sha(
        generic_report.get("evidence_digest"), "generic evidence digest"
    )
    runner = _exact(
        generic_report.get("fixture_runner"),
        {"path", "size_bytes", "sha256"},
        "generic fixture runner",
    )
    runner_sha = _sha(runner["sha256"], "generic fixture runner digest")
    runner_path = resolve_local(manifest.root, runner["path"], "fixture_runner")
    runner_size, actual_runner_sha = hash_file(runner_path)
    if (
        type(runner["size_bytes"]) is not int
        or runner["size_bytes"] != runner_size
        or actual_runner_sha != runner_sha
    ):
        raise _fail("generic fixture runner identity mismatch")
    results = generic_report.get("results")
    results = _exact(
        results,
        {
            "streaming",
            "tools",
            "json_schema",
            "cancellation",
            "crash_restart",
            "isolation",
            "protocol_conformance",
            "privacy_leaks",
            "unexpected_oom",
            "unexpected_crashes",
            "unexpected_hangs",
            "platform_gate",
        },
        "generic qualification results",
    )
    if (
        results["streaming"]
        != {"greek_success": 20, "english_success": 20}
        or results["tools"]
        != {"named_success": 10, "required_success": 10}
        or results["json_schema"] != {"strict_success": 20}
        or results["cancellation"]
        != {"prefill_success": 5, "mid_stream_success": 5}
        or results["crash_restart"] != {"isolated_success": 5}
        or results["isolation"] != {"pairs_success": 50, "leaks": 0}
        or _number(results["protocol_conformance"], "generic protocol") != 1
        or any(
            type(results[name]) is not int or results[name] != 0
            for name in (
                "privacy_leaks",
                "unexpected_oom",
                "unexpected_crashes",
                "unexpected_hangs",
            )
        )
    ):
        raise _fail("generic qualification gates are not a pass")
    gate = _exact(
        results.get("platform_gate"),
        {
            "kind",
            "fixture",
            "assets",
            "asr_runtime_identity_digest",
            "audio_corpus_digest",
            "greek_utterances",
            "english_utterances",
            "greek_language_correct",
            "english_language_correct",
            "greek_reference_words",
            "greek_word_errors",
            "english_reference_words",
            "english_word_errors",
            "forced_cpu_fallbacks",
            "baseline_power_identity_digest",
            "coexistence_power_identity_digest",
            "overlapping_utterances",
            "baseline_p95",
            "coexistence_p95",
            "baseline_llm_p95",
            "coexistence_llm_p95",
            "minimum_free_vram_mib",
            "lost_final_transcripts",
            "duplicate_final_transcripts",
            "stale_results",
            "oom",
            "crashes",
            "throttle_reasons",
        },
        "generic Windows platform gate",
    )
    if gate["kind"] != "windows_asr_coexistence":
        raise _fail("generic Windows platform gate is not a pass")
    assets = _exact(gate["assets"], set(_GENERIC_ASSET_PATHS), "ASR assets")
    digests: dict[str, str] = {}
    documents: dict[str, Any] = {}
    for name, relative in _GENERIC_ASSET_PATHS.items():
        artifact = _exact(
            assets[name], {"path", "size_bytes", "sha256"}, f"{name} asset"
        )
        if artifact["path"] != relative:
            raise _fail("ASR asset path mismatch")
        path = resolve_local(manifest.root, relative, f"qualification.{name}")
        size, digest = hash_file(path)
        if (
            type(artifact["size_bytes"]) is not int
            or artifact["size_bytes"] != size
            or artifact["sha256"] != digest
        ):
            raise _fail("ASR asset identity mismatch")
        digests[name] = digest
        if name in {"fixture", "power_profile"}:
            try:
                documents[name] = parse_body(
                    _safe_bytes(path, f"generic {name} asset")
                )
            except ApplianceError as exc:
                raise _fail("invalid generic Windows fixture asset") from exc
    fixture_document = documents.get("fixture")
    power_document = documents.get("power_profile")
    if not isinstance(fixture_document, dict) or not isinstance(
        power_document, dict
    ):
        raise _fail("invalid generic Windows fixture asset")
    expected_gate_fixture = {
        **fixture_document,
        "power_profile": power_document,
    }
    power_identity_sha = hashlib.sha256(
        canonical_bytes(power_document)
    ).hexdigest()
    hardware = current_identity.get("hardware_runtime")
    if (
        gate["fixture"] != expected_gate_fixture
        or gate["baseline_power_identity_digest"] != power_identity_sha
        or gate["coexistence_power_identity_digest"] != power_identity_sha
        or not isinstance(hardware, dict)
        or power_document.get("driver_version") != hardware.get("driver")
        or power_document.get("compute_runtime") != hardware.get(
            "compute_runtime"
        )
    ):
        raise _fail("generic Windows fixture identity mismatch")
    if gate["audio_corpus_digest"] != digests["audio_corpus"]:
        raise _fail("ASR corpus digest mismatch")
    asr_runtime_sha = _sha(
        gate["asr_runtime_identity_digest"], "ASR runtime identity digest"
    )
    if asr_runtime_sha != digests["asr_config"]:
        raise _fail("ASR runtime identity mismatch")
    if (
        gate["greek_utterances"] != 100
        or gate["english_utterances"] != 100
        or type(gate["greek_language_correct"]) is not int
        or gate["greek_language_correct"] < 95
        or type(gate["english_language_correct"]) is not int
        or gate["english_language_correct"] < 95
        or type(gate["greek_reference_words"]) is not int
        or gate["greek_reference_words"] <= 0
        or type(gate["greek_word_errors"]) is not int
        or gate["greek_word_errors"] < 0
        or gate["greek_word_errors"] * 100
        > gate["greek_reference_words"] * 20
        or type(gate["english_reference_words"]) is not int
        or gate["english_reference_words"] <= 0
        or type(gate["english_word_errors"]) is not int
        or gate["english_word_errors"] < 0
        or gate["english_word_errors"] * 100
        > gate["english_reference_words"] * 20
        or gate["forced_cpu_fallbacks"] != 5
        or type(gate["overlapping_utterances"]) is not int
        or gate["overlapping_utterances"] < 80
        or type(gate["minimum_free_vram_mib"]) is not int
        or gate["minimum_free_vram_mib"] < 2048
        or any(
            type(gate[name]) is not int or gate[name] != 0
            for name in (
                "lost_final_transcripts",
                "duplicate_final_transcripts",
                "stale_results",
                "oom",
                "crashes",
                "throttle_reasons",
            )
        )
    ):
        raise _fail("generic Windows coexistence counters failed")
    baseline = _number(gate["baseline_p95"], "baseline ASR p95", positive=True)
    coexistence = _number(
        gate["coexistence_p95"], "coexistence ASR p95", positive=True
    )
    if coexistence / baseline - 1 > 0.15:
        raise _fail("generic ASR degradation exceeded its threshold")
    baseline_llm = _number(
        gate["baseline_llm_p95"], "baseline LLM p95", positive=True
    )
    coexistence_llm = _number(
        gate["coexistence_llm_p95"], "coexistence LLM p95", positive=True
    )
    if coexistence_llm / baseline_llm - 1 > 0.15:
        raise _fail("generic LLM degradation exceeded its threshold")
    return (
        {
            "generic_qualification_report_sha256": hashlib.sha256(raw).hexdigest(),
            "generic_evidence_sha256": evidence_sha,
            "generic_fixture_runner_sha256": runner_sha,
            "asr_fixture_sha256": digests["fixture"],
            "asr_collector_sha256": digests["collector"],
            "asr_config_sha256": digests["asr_config"],
            "audio_corpus_sha256": digests["audio_corpus"],
            "power_profile_sha256": digests["power_profile"],
            "asr_runtime_identity_sha256": asr_runtime_sha,
        },
        gate,
    )


def expected_profile_identity(
    manifest: Manifest,
    current_identity: dict[str, Any],
    generic_report: dict[str, Any],
    *,
    profile_runner_sha256: str,
    profile_evidence_sha256: str,
    tokenizer_identity_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not profile_report_required(manifest):
        raise _fail("manifest is not the exact Windows primary profile")
    generic, gate = _generic_context(manifest, current_identity, generic_report)
    quality_path = resolve_local(
        manifest.root, PRIMARY_CORPUS_RELATIVE, "profile_quality_fixture"
    )
    base_path = resolve_local(
        manifest.root, PRIMARY_BASE_CORPUS_RELATIVE, "profile_quality_base_fixture"
    )
    # The primary loader independently verifies the composite and referenced
    # base corpus. Resolve both first so links/reparse points fail explicitly.
    del base_path
    _, quality_sha = load_primary_corpus(quality_path)
    performance_path = resolve_local(
        manifest.root, PERFORMANCE_FIXTURE_RELATIVE, "profile_performance_fixture"
    )
    performance = load_fixture(performance_path)
    identity = {
        "manifest_sha256": manifest.digest,
        "runtime_identity_sha256": hashlib.sha256(
            canonical_bytes(current_identity)
        ).hexdigest(),
        "generic_qualification_report_sha256": generic[
            "generic_qualification_report_sha256"
        ],
        "generic_evidence_sha256": generic["generic_evidence_sha256"],
        "generic_fixture_runner_sha256": generic[
            "generic_fixture_runner_sha256"
        ],
        "profile_fixture_runner_sha256": _sha(
            profile_runner_sha256, "profile fixture runner digest"
        ),
        "profile_evidence_sha256": _sha(
            profile_evidence_sha256, "profile evidence digest"
        ),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_artifact_sha256": MODEL_SHA256,
        "prompt_profile_id": PROFILE_ID,
        "renderer_sha256": RENDERER_SHA256,
        "tool_parser_sha256": TOOL_PARSER_SHA256,
        "tokenizer_identity_sha256": _sha(
            tokenizer_identity_sha256, "tokenizer identity digest"
        ),
        "quality_corpus_sha256": quality_sha,
        "performance_fixture_sha256": performance.fixture_sha256,
        "expanded_requests_sha256": performance.expanded_requests_sha256,
        "rendered_prompt_texts_sha256": performance.rendered_prompt_texts_sha256,
        "asr_fixture_sha256": generic["asr_fixture_sha256"],
        "asr_collector_sha256": generic["asr_collector_sha256"],
        "asr_config_sha256": generic["asr_config_sha256"],
        "audio_corpus_sha256": generic["audio_corpus_sha256"],
        "power_profile_sha256": generic["power_profile_sha256"],
    }
    return identity, gate


def _validate_provenance(
    value: Any, expected: dict[str, Any] | None
) -> dict[str, Any]:
    provenance = _exact(
        value,
        {
            "challenge",
            "started_unix_ns",
            "completed_unix_ns",
            "monotonic_elapsed_ns",
            "restart_epochs",
        },
        "profile provenance",
    )
    if not isinstance(provenance["challenge"], str) or _CHALLENGE.fullmatch(
        provenance["challenge"]
    ) is None:
        raise _fail("invalid profile challenge")
    times: list[int] = []
    for name in ("started_unix_ns", "completed_unix_ns"):
        raw = provenance[name]
        if not isinstance(raw, str) or re.fullmatch(r"[1-9][0-9]{0,19}", raw) is None:
            raise _fail("invalid profile wall-clock provenance")
        times.append(int(raw))
    elapsed = _integer(
        provenance["monotonic_elapsed_ns"], "profile monotonic elapsed", 1
    )
    epochs = provenance["restart_epochs"]
    if (
        not isinstance(epochs, list)
        or len(epochs) < 9
        or any(type(epoch) is not int or epoch < 1 for epoch in epochs)
        or epochs != list(range(1, len(epochs) + 1))
    ):
        raise _fail("invalid profile restart epochs")
    if times[1] < times[0] or elapsed < 28_800 * 1_000_000_000:
        raise _fail("profile provenance does not contain an eight-hour run")
    if expected is not None:
        if (
            provenance["challenge"] != expected["challenge"]
            or times[0] < expected["not_before_unix_ns"]
            or times[1] > expected["not_after_unix_ns"]
            or elapsed > expected["max_elapsed_ns"]
        ):
            raise _fail("profile provenance is outside the exclusive run")
    return provenance


def _validate_quality(value: Any, fixtures: tuple[Any, ...]) -> dict[str, Any]:
    quality = _exact(
        value,
        {
            "status",
            "suite_version",
            "corpus_sha256",
            "system_prompt_sha256",
            "sample_count",
            "greek_total",
            "greek_passed",
            "english_total",
            "english_passed",
            "language_pass_rate_gap",
            "critical_failures",
            "protocol_failures",
            "cases",
        },
        "quality",
    )
    cases = quality["cases"]
    if not isinstance(cases, list) or len(cases) != 120:
        raise _fail("invalid quality cases")
    expected_ids = [fixture.case_id for fixture in fixtures]
    passed_by_language = Counter()
    totals_by_language = Counter()
    critical = 0
    for expected_id, raw, fixture in zip(expected_ids, cases, fixtures, strict=True):
        case = _exact(raw, {"case_id", "passed", "critical"}, "quality case")
        if (
            case["case_id"] != expected_id
            or type(case["passed"]) is not bool
            or type(case["critical"]) is not bool
        ):
            raise _fail("quality case identity mismatch")
        totals_by_language[fixture.language] += 1
        passed_by_language[fixture.language] += int(case["passed"])
        critical += int(case["critical"])
    gap = abs(passed_by_language["el"] / 60 - passed_by_language["en"] / 60)
    if (
        quality["status"] != "pass"
        or quality["suite_version"] != PRIMARY_QUALITY_SUITE_VERSION
        or quality["corpus_sha256"] != PRIMARY_CORPUS_SHA256
        or quality["system_prompt_sha256"]
        != hashlib.sha256(PRIMARY_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        or quality["sample_count"] != 120
        or quality["greek_total"] != totals_by_language["el"]
        or quality["english_total"] != totals_by_language["en"]
        or quality["greek_passed"] != passed_by_language["el"]
        or quality["english_passed"] != passed_by_language["en"]
        or passed_by_language["el"] < 54
        or passed_by_language["en"] < 54
        or _number(quality["language_pass_rate_gap"], "language gap") != gap
        or gap > 0.10
        or quality["critical_failures"] != critical
        or critical != 0
        or quality["protocol_failures"] != 0
    ):
        raise _fail("quality gate failed")
    return quality


def _validate_soak(value: Any, provenance: dict[str, Any]) -> dict[str, Any]:
    soak = _exact(
        value,
        {
            "status",
            "monotonic_seconds_observed",
            "cycles_completed",
            "full_restarts_completed",
            "event_counts",
            "memory",
        },
        "soak",
    )
    events = _exact(soak["event_counts"], set(_ZERO_EVENTS), "soak events")
    memory = _exact(
        soak["memory"],
        {
            "evaluated",
            "monotonic_growth_detected",
            "baseline_bytes",
            "final_bytes",
            "peak_bytes",
            "epoch_sample_counts",
            "max_epoch_growth_fraction",
            "global_growth_fraction",
        },
        "soak memory",
    )
    seconds = _number(
        soak["monotonic_seconds_observed"], "soak duration", positive=True
    )
    cycles = _integer(soak["cycles_completed"], "soak cycles")
    restarts = _integer(soak["full_restarts_completed"], "soak restarts")
    for name in ("baseline_bytes", "final_bytes", "peak_bytes"):
        _integer(memory[name], f"soak memory {name}", 1)
    epoch_sample_counts = memory["epoch_sample_counts"]
    growth = _number(
        memory["max_epoch_growth_fraction"], "epoch memory growth fraction"
    )
    global_growth = _number(
        memory["global_growth_fraction"], "global memory growth fraction"
    )
    if (
        soak["status"] != "pass"
        or seconds < 28_800
        or seconds * 1_000_000_000 > provenance["monotonic_elapsed_ns"]
        or cycles < 96
        or cycles % 12 != 0
        or restarts < 8
        or restarts != cycles // 12
        or len(provenance["restart_epochs"]) != restarts + 1
        or any(type(count) is not int or count != 0 for count in events.values())
        or memory["evaluated"] is not True
        or memory["monotonic_growth_detected"] is not False
        or memory["peak_bytes"] < max(memory["baseline_bytes"], memory["final_bytes"])
        or not isinstance(epoch_sample_counts, list)
        # Each completed 12-cycle workload epoch is sampled before its
        # deliberate restart. The final restart proves recovery and adds a
        # process epoch ID, but intentionally has no post-restart workload.
        or len(epoch_sample_counts) != restarts
        or any(type(count) is not int or count < 12 for count in epoch_sample_counts)
        or growth < 0
        or growth > 0.05
        or global_growth < 0
        or global_growth > 0.05
    ):
        raise _fail("eight-hour soak gate failed")
    return soak


def _validate_coexistence(value: Any, gate: dict[str, Any]) -> dict[str, Any]:
    coexistence = _exact(
        value,
        {
            "status",
            "greek_utterances",
            "english_utterances",
            "greek_language_correct",
            "english_language_correct",
            "greek_reference_words",
            "greek_word_errors",
            "english_reference_words",
            "english_word_errors",
            "forced_cpu_fallbacks",
            "overlapping_utterances",
            "baseline_p95_seconds",
            "coexistence_p95_seconds",
            "asr_degradation_ratio",
            "baseline_llm_p95_seconds",
            "coexistence_llm_p95_seconds",
            "llm_regression_ratio",
            "minimum_free_vram_mib",
            "asr_runtime_identity_sha256",
            "lost_final_transcripts",
            "duplicate_final_transcripts",
            "stale_results",
            "oom",
            "crashes",
            "throttle_reasons",
        },
        "Windows coexistence",
    )
    baseline = _number(
        coexistence["baseline_p95_seconds"], "profile baseline ASR p95", positive=True
    )
    concurrent = _number(
        coexistence["coexistence_p95_seconds"],
        "profile coexistence ASR p95",
        positive=True,
    )
    ratio = concurrent / baseline - 1
    observed_ratio = _number(
        coexistence["asr_degradation_ratio"], "ASR degradation"
    )
    baseline_llm = _number(
        coexistence["baseline_llm_p95_seconds"],
        "profile baseline LLM p95",
        positive=True,
    )
    concurrent_llm = _number(
        coexistence["coexistence_llm_p95_seconds"],
        "profile coexistence LLM p95",
        positive=True,
    )
    llm_ratio = concurrent_llm / baseline_llm - 1
    observed_llm_ratio = _number(
        coexistence["llm_regression_ratio"], "LLM regression"
    )
    zeroes = (
        "lost_final_transcripts",
        "duplicate_final_transcripts",
        "stale_results",
        "oom",
        "crashes",
        "throttle_reasons",
    )
    if (
        coexistence["status"] != "pass"
        or coexistence["greek_utterances"] != gate["greek_utterances"]
        or coexistence["english_utterances"] != gate["english_utterances"]
        or coexistence["greek_language_correct"]
        != gate["greek_language_correct"]
        or coexistence["english_language_correct"]
        != gate["english_language_correct"]
        or coexistence["greek_reference_words"] != gate["greek_reference_words"]
        or coexistence["greek_word_errors"] != gate["greek_word_errors"]
        or coexistence["english_reference_words"] != gate["english_reference_words"]
        or coexistence["english_word_errors"] != gate["english_word_errors"]
        or coexistence["forced_cpu_fallbacks"] != gate["forced_cpu_fallbacks"]
        or coexistence["forced_cpu_fallbacks"] != 5
        or coexistence["overlapping_utterances"] != gate["overlapping_utterances"]
        or baseline != float(gate["baseline_p95"])
        or concurrent != float(gate["coexistence_p95"])
        or not math.isclose(observed_ratio, ratio, rel_tol=1e-9, abs_tol=1e-9)
        or ratio > 0.15
        or baseline_llm != float(gate["baseline_llm_p95"])
        or concurrent_llm != float(gate["coexistence_llm_p95"])
        or not math.isclose(
            observed_llm_ratio, llm_ratio, rel_tol=1e-9, abs_tol=1e-9
        )
        or llm_ratio > 0.15
        or coexistence["minimum_free_vram_mib"]
        != gate["minimum_free_vram_mib"]
        or coexistence["minimum_free_vram_mib"] < 2048
        or coexistence["asr_runtime_identity_sha256"]
        != gate["asr_runtime_identity_digest"]
        or any(type(coexistence[name]) is not int or coexistence[name] != 0 for name in zeroes)
    ):
        raise _fail("Windows coexistence gate failed")
    return coexistence


def _validate_privacy(value: Any) -> dict[str, Any]:
    privacy = _exact(
        value,
        {
            "runner_content_payloads_persisted",
            "runner_content_payloads_logged",
            "evidence_content_free",
            "diagnostic_sanitization_passed",
        },
        "privacy",
    )
    if privacy != {
        "runner_content_payloads_persisted": 0,
        "runner_content_payloads_logged": 0,
        "evidence_content_free": True,
        "diagnostic_sanitization_passed": True,
    }:
        raise _fail("privacy gate failed")
    return privacy


def validate_profile_document(
    document: dict[str, Any],
    manifest: Manifest,
    current_identity: dict[str, Any],
    generic_report: dict[str, Any],
    *,
    evidence: bool,
    expected_provenance: dict[str, Any] | None = None,
    profile_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    top_fields = {
        "schema",
        "status",
        "profile_id",
        "platform",
        "attestation_level",
        "operator_trust_required",
        "identity",
        "provenance",
        "quality",
        "performance",
        "soak",
        "windows_coexistence",
        "privacy",
    }
    if not evidence:
        top_fields |= {"authoritative", "production_ready"}
    root = _exact(document, top_fields, "profile report")
    if (
        root["schema"] != (EVIDENCE_SCHEMA if evidence else REPORT_SCHEMA)
        or root["status"] != "pass"
        or root["profile_id"] != "local-primary"
        or root["platform"] != "windows-x86_64"
        or root["attestation_level"] != ATTESTATION_LEVEL
        or root["operator_trust_required"] is not OPERATOR_TRUST_REQUIRED
        or (not evidence and root["authoritative"] is not True)
        or (not evidence and root["production_ready"] is not True)
        or _contains_content_key(root)
    ):
        raise _fail("profile report identity or sanitization failed")
    provenance = _validate_provenance(root["provenance"], expected_provenance)
    quality_path = resolve_local(
        manifest.root, PRIMARY_CORPUS_RELATIVE, "profile_quality_fixture"
    )
    fixtures, _ = load_primary_corpus(quality_path)
    performance_fixture = load_fixture(
        resolve_local(
            manifest.root,
            PERFORMANCE_FIXTURE_RELATIVE,
            "profile_performance_fixture",
        )
    )
    quality = _validate_quality(root["quality"], fixtures)
    performance, tokenizer_sha = validate_performance(
        root["performance"], performance_fixture, manifest, current_identity
    )
    runner = resolve_local(
        manifest.root, PROFILE_RUNNER_RELATIVE, "profile_fixture_runner"
    )
    _, runner_sha = hash_file(runner)
    supplied_identity = root["identity"]
    if not isinstance(supplied_identity, dict):
        raise _fail("invalid profile identity")
    evidence_sha = (
        "0" * 64 if evidence else _sha(profile_evidence_sha256, "profile evidence")
    )
    expected_identity, gate = expected_profile_identity(
        manifest,
        current_identity,
        generic_report,
        profile_runner_sha256=runner_sha,
        profile_evidence_sha256=evidence_sha,
        tokenizer_identity_sha256=tokenizer_sha,
    )
    if evidence:
        expected_identity.pop("profile_evidence_sha256")
    if supplied_identity != expected_identity:
        raise _fail("profile qualification identity mismatch")
    soak = _validate_soak(root["soak"], provenance)
    if (
        len(performance["cold_load_to_ready_samples_seconds"])
        != soak["full_restarts_completed"]
    ):
        raise _fail("cold-load samples do not match completed restarts")
    coexistence = _validate_coexistence(root["windows_coexistence"], gate)
    privacy = _validate_privacy(root["privacy"])
    return {
        **root,
        "identity": supplied_identity,
        "provenance": provenance,
        "quality": quality,
        "performance": performance,
        "soak": soak,
        "windows_coexistence": coexistence,
        "privacy": privacy,
    }


def load_valid_profile_report(
    manifest: Manifest,
    current_identity: dict[str, Any],
    generic_report: dict[str, Any],
) -> dict[str, Any]:
    if manifest.profile_qualification_report is None:
        raise _fail("profile qualification report is not declared")
    raw = _safe_bytes(
        manifest.profile_qualification_report, "profile qualification report"
    )
    evidence_path = resolve_local(
        manifest.root, PROFILE_EVIDENCE_RELATIVE, "profile_qualification_evidence"
    )
    evidence_raw = _safe_bytes(evidence_path, "profile qualification evidence")
    evidence_sha = hashlib.sha256(evidence_raw).hexdigest()
    try:
        report = parse_body(raw)
        evidence = parse_body(evidence_raw)
    except ApplianceError as exc:
        raise _fail("invalid profile qualification JSON") from exc
    validated = validate_profile_document(
        report,
        manifest,
        current_identity,
        generic_report,
        evidence=False,
        profile_evidence_sha256=evidence_sha,
    )
    evidence_validated = validate_profile_document(
        evidence,
        manifest,
        current_identity,
        generic_report,
        evidence=True,
    )
    comparable = {
        key: value
        for key, value in validated.items()
        if key not in {"schema", "authoritative", "production_ready"}
    }
    evidence_comparable = {
        key: value for key, value in evidence_validated.items() if key != "schema"
    }
    evidence_comparable["identity"] = {
        key: value
        for key, value in evidence_comparable["identity"].items()
        if key != "profile_evidence_sha256"
    }
    comparable["identity"] = {
        key: value
        for key, value in comparable["identity"].items()
        if key != "profile_evidence_sha256"
    }
    if comparable != evidence_comparable:
        raise _fail("profile report does not match its preserved evidence")
    return validated


def profile_report_matches(
    manifest: Manifest,
    current_identity: dict[str, Any],
    generic_report: dict[str, Any],
) -> bool:
    if not profile_report_required(manifest):
        return True
    try:
        load_valid_profile_report(manifest, current_identity, generic_report)
        return True
    except (ApplianceError, OSError, TypeError, ValueError):
        return False
