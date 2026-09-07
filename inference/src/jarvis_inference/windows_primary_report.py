from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .errors import ApplianceError
from .manifest import _read_manifest_file
from .strict_json import parse_body
from .windows_primary_fixture import (
    SUITE_ID,
    LoadedWindowsPrimaryFixture,
    WindowsPrimaryFixtureFailure,
)


REPORT_SCHEMA_VERSION = 1
FROZEN_REPORT_SKELETON_SHA256 = (
    "c4cffbf65a443e06cb269931a6db01c442b15e4a7c28f4521be91e99d38ac8b9"
)
REPORT_BYTES = 65_536
_CONTENT_KEYS = {
    "content",
    "messages",
    "output",
    "prompt",
    "raw_output",
    "response",
    "schema",
    "tool_arguments",
}


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise WindowsPrimaryFixtureFailure(f"invalid {name} shape")
    return value


def _safe_regular_file(path: Path) -> bytes:
    try:
        raw = _read_manifest_file(path, maximum=REPORT_BYTES)
    except (ApplianceError, OSError) as exc:
        raise WindowsPrimaryFixtureFailure("unsafe report skeleton") from exc
    if not raw:
        raise WindowsPrimaryFixtureFailure("unsafe report skeleton")
    return raw


def _contains_content_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key.casefold() in _CONTENT_KEYS or _contains_content_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_content_key(child) for child in value)
    return False


def validate_pending_report_skeleton(
    document: dict[str, Any], fixture: LoadedWindowsPrimaryFixture
) -> dict[str, Any]:
    root = _exact(
        document,
        {
            "report_schema_version",
            "suite_id",
            "status",
            "authoritative_qualification",
            "production_ready",
            "fixture_sha256",
            "expanded_requests_sha256",
            "rendered_prompt_texts_sha256",
            "profile_id",
            "platform",
            "native_identity",
            "performance",
            "soak",
            "windows_coexistence",
            "privacy",
        },
        "pending report",
    )
    if (
        root["report_schema_version"] != REPORT_SCHEMA_VERSION
        or root["suite_id"] != SUITE_ID
        or root["status"] != "pending_native"
        or root["authoritative_qualification"] is not False
        or root["production_ready"] is not False
        or root["fixture_sha256"] != fixture.fixture_sha256
        or root["expanded_requests_sha256"] != fixture.expanded_requests_sha256
        or root["rendered_prompt_texts_sha256"]
        != fixture.rendered_prompt_texts_sha256
        or root["profile_id"] != "local-primary"
        or root["platform"] != "windows-x86_64"
    ):
        raise WindowsPrimaryFixtureFailure("pending report identity mismatch")
    identity = _exact(
        root["native_identity"],
        {
            "manifest_sha256",
            "runtime_identity_sha256",
            "tokenizer_identity_sha256",
            "driver_version",
            "compute_runtime",
            "windows_power_scheme_guid",
        },
        "native identity",
    )
    if any(value is not None for value in identity.values()):
        raise WindowsPrimaryFixtureFailure("pending report claims native identity")
    performance = _exact(
        root["performance"],
        {
            "status",
            "timed_requests_completed",
            "cold_load_to_ready_samples_seconds",
            "native_rendered_token_validation",
            "per_case_observations",
            "metrics",
            "thresholds_evaluated",
        },
        "pending performance",
    )
    metrics = _exact(
        performance["metrics"],
        {
            "ttft_p50_seconds",
            "ttft_p95_seconds",
            "end_to_end_p50_seconds",
            "end_to_end_p95_seconds",
            "projected_128_token_e2e_p95_seconds",
            "decode_p5_tokens_per_second",
            "decode_p50_tokens_per_second",
            "decode_p95_tokens_per_second",
            "cold_load_to_ready_p95_seconds",
        },
        "pending performance metrics",
    )
    if (
        performance["status"] != "pending_native"
        or performance["timed_requests_completed"] is not None
        or performance["cold_load_to_ready_samples_seconds"] != []
        or performance["native_rendered_token_validation"] != "pending_native"
        or performance["per_case_observations"] != []
        or performance["thresholds_evaluated"] is not False
        or any(value is not None for value in metrics.values())
    ):
        raise WindowsPrimaryFixtureFailure("pending performance contains evidence")
    soak = _exact(
        root["soak"],
        {
            "status",
            "monotonic_seconds_observed",
            "cycles_completed",
            "full_restarts_completed",
            "event_counts",
            "memory_growth_evaluated",
            "epoch_sample_counts",
            "max_epoch_growth_fraction",
            "global_growth_fraction",
        },
        "pending soak",
    )
    if soak != {
        "status": "pending_native",
        "monotonic_seconds_observed": None,
        "cycles_completed": None,
        "full_restarts_completed": None,
        "event_counts": None,
        "memory_growth_evaluated": False,
        "epoch_sample_counts": [],
        "max_epoch_growth_fraction": None,
        "global_growth_fraction": None,
    }:
        raise WindowsPrimaryFixtureFailure("pending soak contains evidence")
    coexistence = _exact(
        root["windows_coexistence"],
        {
            "status",
            "minimum_free_vram_mib_observed",
            "overlapping_utterances",
            "greek_language_correct",
            "english_language_correct",
            "greek_reference_words",
            "greek_word_errors",
            "english_reference_words",
            "english_word_errors",
            "forced_cpu_fallbacks",
            "asr_p95_degradation_ratio",
            "llm_p95_regression_ratio",
            "duplicate_final_transcripts",
            "stale_results",
            "thresholds_evaluated",
        },
        "pending coexistence",
    )
    if coexistence != {
        "status": "pending_native",
        "minimum_free_vram_mib_observed": None,
        "overlapping_utterances": None,
        "greek_language_correct": None,
        "english_language_correct": None,
        "greek_reference_words": None,
        "greek_word_errors": None,
        "english_reference_words": None,
        "english_word_errors": None,
        "forced_cpu_fallbacks": None,
        "asr_p95_degradation_ratio": None,
        "llm_p95_regression_ratio": None,
        "duplicate_final_transcripts": None,
        "stale_results": None,
        "thresholds_evaluated": False,
    }:
        raise WindowsPrimaryFixtureFailure("pending coexistence contains evidence")
    privacy = _exact(
        root["privacy"],
        {"content_payloads_in_report", "content_logging_gate"},
        "pending privacy",
    )
    if privacy != {
        "content_payloads_in_report": False,
        "content_logging_gate": "pending_native",
    }:
        raise WindowsPrimaryFixtureFailure("pending privacy contract mismatch")
    if _contains_content_key(root):
        raise WindowsPrimaryFixtureFailure("pending report contains content fields")
    return root


def load_pending_report_skeleton(
    path: Path, fixture: LoadedWindowsPrimaryFixture
) -> dict[str, Any]:
    raw = _safe_regular_file(path)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FROZEN_REPORT_SKELETON_SHA256:
        raise WindowsPrimaryFixtureFailure("report skeleton digest mismatch")
    return validate_pending_report_skeleton(parse_body(raw), fixture)
