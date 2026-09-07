from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .contracts import Capabilities, normalize_request
from .gemma4 import (
    PROFILE_ID,
    RENDERER_ID,
    RENDERER_SHA256,
    render_gemma4_prompt,
)
from .errors import ApplianceError
from .manifest import _read_manifest_file
from .strict_json import canonical_bytes, parse_body


SUITE_ID = "cyrus-windows-primary-performance-soak-v1"
FIXTURE_SCHEMA_VERSION = 1
FROZEN_FIXTURE_SHA256 = "75572ca3e34874aa93dc210bc9efd34e11fb28c19776ff7d3b489c15c86a8c0f"
FIXTURE_BYTES = 524_288

_TARGET_BINS = (512, 2048, 4096)
_BIN_REPETITIONS = {512: 8, 2048: 32, 4096: 64}
_VARIANTS = (
    "chat_history",
    "json_schema",
    "tool_named",
    "tool_required",
    "tool_history",
)
_ZERO_EVENTS = (
    "observed_request_failures",
    "observed_unexpected_process_terminations",
    "observed_request_timeouts",
    "observed_cleanup_failures",
    "observed_circuit_breaker_rejections",
)


class WindowsPrimaryFixtureFailure(RuntimeError):
    """A content-free frozen-fixture validation failure."""


@dataclass(frozen=True, slots=True)
class LoadedWindowsPrimaryFixture:
    document: dict[str, Any]
    requests: tuple[dict[str, Any], ...]
    fixture_sha256: str
    expanded_requests_sha256: str
    rendered_prompt_texts_sha256: str


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise WindowsPrimaryFixtureFailure(f"invalid {name} shape")
    return value


def _safe_regular_file(path: Path, *, maximum: int, name: str) -> bytes:
    try:
        raw = _read_manifest_file(path, maximum=maximum)
    except (ApplianceError, OSError) as exc:
        raise WindowsPrimaryFixtureFailure(f"unsafe {name}") from exc
    if not raw:
        raise WindowsPrimaryFixtureFailure(f"unsafe {name}")
    return raw


def _hex_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WindowsPrimaryFixtureFailure(f"invalid {name}")
    return value


def _locale_templates(value: Any) -> dict[str, dict[str, Any]]:
    root = _exact(value, {"el", "en"}, "locale templates")
    result: dict[str, dict[str, Any]] = {}
    fields = {
        "prior_user",
        "prior_assistant",
        "context_heading",
        "context_fragments",
        "final_requests",
        "tool_history_user",
        "tool_history_result",
    }
    for language in ("el", "en"):
        item = _exact(root[language], fields, f"{language} locale template")
        strings = (
            item["prior_user"],
            item["prior_assistant"],
            item["context_heading"],
            item["tool_history_user"],
            item["tool_history_result"],
        )
        if any(
            not isinstance(text, str)
            or not text
            or len(text.encode("utf-8")) > 512
            for text in strings
        ):
            raise WindowsPrimaryFixtureFailure("invalid locale template text")
        for key in ("context_fragments", "final_requests"):
            values = item[key]
            expected = 8 if key == "context_fragments" else 10
            if (
                not isinstance(values, list)
                or len(values) != expected
                or any(
                    not isinstance(text, str)
                    or not text
                    or len(text.encode("utf-8")) > 512
                    for text in values
                )
            ):
                raise WindowsPrimaryFixtureFailure("invalid locale template list")
        result[language] = item
    return result


def _validate_schema_and_tool(templates: dict[str, Any]) -> None:
    schema = _exact(
        templates["structured_schema"],
        {"name", "description", "strict", "schema"},
        "structured template",
    )
    if (
        schema["name"] != "performance_summary"
        or not isinstance(schema["description"], str)
        or not schema["description"]
        or schema["strict"] is not True
    ):
        raise WindowsPrimaryFixtureFailure("invalid structured template")
    tool = _exact(
        templates["tool"],
        {"type", "function"},
        "tool template",
    )
    function = _exact(
        tool["function"],
        {"name", "description", "strict", "parameters"},
        "tool function template",
    )
    if (
        tool["type"] != "function"
        or function["name"] != "inspect_schedule_snapshot"
        or not isinstance(function["description"], str)
        or not function["description"]
        or function["strict"] is not True
    ):
        raise WindowsPrimaryFixtureFailure("invalid tool template")


def _context(case: dict[str, Any], locale: dict[str, Any]) -> str:
    fragments = locale["context_fragments"]
    lines = [locale["context_heading"]]
    for index in range(case["context_repetitions"]):
        fragment = fragments[(index + case["scenario_index"]) % len(fragments)]
        lines.append(
            f"[{case['case_id']}-context-{index + 1:03d}] {fragment}"
        )
    lines.append(locale["final_requests"][case["scenario_index"]])
    return "\n".join(lines)


def _materialize_request(
    case: dict[str, Any], templates: dict[str, Any]
) -> dict[str, Any]:
    locale = templates["locales"][case["language"]]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": templates["system_prompt"]},
        {"role": "user", "content": locale["prior_user"]},
        {"role": "assistant", "content": locale["prior_assistant"]},
    ]
    request: dict[str, Any] = {
        "model": "jarvis-local",
        "messages": messages,
        "max_completion_tokens": 128,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    variant = case["variant"]
    if variant == "json_schema":
        request["stream"] = False
        request.pop("stream_options")
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": templates["structured_schema"],
        }
    elif variant in {"tool_named", "tool_required", "tool_history"}:
        request["tools"] = [templates["tool"]]
        if variant == "tool_named":
            request["tool_choice"] = {
                "type": "function",
                "function": {"name": "inspect_schedule_snapshot"},
            }
        elif variant == "tool_required":
            request["tool_choice"] = "required"
        else:
            request["tool_choice"] = "auto"
            call_id = f"hist_{case['case_id'].replace('-', '_')}"
            messages.extend(
                [
                    {"role": "user", "content": locale["tool_history_user"]},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "inspect_schedule_snapshot",
                                    "arguments": '{"date":"2030-01-15","scope":"day"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": locale["tool_history_result"],
                    },
                ]
            )
    messages.append({"role": "user", "content": _context(case, locale)})
    return request


def _validate_case(value: Any, index: int) -> dict[str, Any]:
    case = _exact(
        value,
        {
            "case_id",
            "language",
            "locale",
            "target_rendered_bin",
            "variant",
            "scenario_index",
            "context_repetitions",
            "deterministic_seed",
        },
        f"performance case {index}",
    )
    language = "el" if index < 50 else "en"
    ordinal = index + 1 if index < 50 else index - 49
    target = _TARGET_BINS[(ordinal - 1) % len(_TARGET_BINS)]
    variant = _VARIANTS[(ordinal - 1) % len(_VARIANTS)]
    expected = {
        "case_id": f"wp-{language}-{ordinal:03d}",
        "language": language,
        "locale": "el-GR" if language == "el" else "en-US",
        "target_rendered_bin": target,
        "variant": variant,
        "scenario_index": (ordinal - 1) % 10,
        "context_repetitions": _BIN_REPETITIONS[target],
        "deterministic_seed": 730_000 + index,
    }
    if case != expected:
        raise WindowsPrimaryFixtureFailure("performance case identity mismatch")
    return case


def derive_fixture_digests(document: dict[str, Any]) -> tuple[str, str]:
    templates = document["templates"]
    requests = tuple(
        _materialize_request(case, templates)
        for case in document["performance_cases"]
    )
    request_rows: list[dict[str, Any]] = []
    rendered_rows: list[dict[str, Any]] = []
    capabilities = Capabilities(
        text=True,
        stream=True,
        tools=True,
        json_object=True,
        json_schema=True,
    )
    for case, raw in zip(document["performance_cases"], requests, strict=True):
        normalized = normalize_request(
            raw,
            alias="jarvis-local",
            capabilities=capabilities,
            context_window=32_768,
        )
        rendered = render_gemma4_prompt(normalized)
        request_rows.append(
            {
                "case_id": case["case_id"],
                "request_sha256": hashlib.sha256(canonical_bytes(raw)).hexdigest(),
            }
        )
        rendered_rows.append(
            {
                "case_id": case["case_id"],
                "rendered_prompt_utf8_sha256": hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
            }
        )
    return (
        hashlib.sha256(canonical_bytes(request_rows)).hexdigest(),
        hashlib.sha256(canonical_bytes(rendered_rows)).hexdigest(),
    )


def validate_fixture(document: dict[str, Any]) -> LoadedWindowsPrimaryFixture:
    root = _exact(
        document,
        {
            "fixture_schema_version",
            "suite_id",
            "profile_id",
            "platform",
            "public_model_alias",
            "prompt_profile",
            "execution",
            "acceptance",
            "templates",
            "performance_cases",
            "soak",
            "expanded_requests_sha256",
            "rendered_prompt_texts_sha256",
        },
        "fixture",
    )
    if (
        root["fixture_schema_version"] != FIXTURE_SCHEMA_VERSION
        or root["suite_id"] != SUITE_ID
        or root["profile_id"] != "local-primary"
        or root["platform"] != "windows-x86_64"
        or root["public_model_alias"] != "jarvis-local"
    ):
        raise WindowsPrimaryFixtureFailure("fixture identity mismatch")
    prompt_profile = _exact(
        root["prompt_profile"],
        {"id", "renderer_id", "renderer_sha256"},
        "prompt profile",
    )
    if prompt_profile != {
        "id": PROFILE_ID,
        "renderer_id": RENDERER_ID,
        "renderer_sha256": RENDERER_SHA256,
    }:
        raise WindowsPrimaryFixtureFailure("prompt profile identity mismatch")

    execution = _exact(
        root["execution"],
        {
            "request_count",
            "language_counts",
            "target_rendered_bins",
            "target_bin_counts",
            "native_token_count_ranges",
            "max_completion_tokens",
            "concurrency",
            "sampling",
            "performance_measurement",
            "native_tokenization",
        },
        "execution",
    )
    tokenization = _exact(
        execution["native_tokenization"],
        {
            "status",
            "exact_counts_claimed",
            "tokenizer_identity_source",
            "required_observations",
        },
        "native tokenization",
    )
    if (
        execution["request_count"] != 100
        or execution["language_counts"] != {"el": 50, "en": 50}
        or execution["target_rendered_bins"] != [512, 2048, 4096]
        or execution["target_bin_counts"]
        != {"512": 34, "2048": 34, "4096": 32}
        or execution["native_token_count_ranges"]
        != {
            "512": {"minimum": 256, "maximum": 1023},
            "2048": {"minimum": 1024, "maximum": 3071},
            "4096": {"minimum": 3072, "maximum": 6144},
        }
        or execution["max_completion_tokens"] != 128
        or execution["concurrency"] != 1
        or execution["sampling"] != {"temperature": 0, "top_p": 1}
        or execution["performance_measurement"]
        != {
            "ttft_source": "gateway_request_start_to_first_accepted_sse_event",
            "decode_source": "validated_final_llama_timings",
            "end_to_end_source": "client_monotonic_request_to_complete",
            "projected_128_token_formula": (
                "ttft_seconds + 127 / decode_tokens_per_second"
            ),
        }
        or tokenization
        != {
            "status": "pending_native",
            "exact_counts_claimed": False,
            "tokenizer_identity_source": "qualified_manifest",
            "required_observations": [
                "generation_serial",
                "rendered_token_count",
                "rendered_token_ids_sha256",
                "accepted_decode_tokens",
                "accepted_token_ids_sha256",
                "tokenizer_identity_sha256",
            ],
        }
    ):
        raise WindowsPrimaryFixtureFailure("execution contract mismatch")

    acceptance = _exact(
        root["acceptance"],
        {"performance", "soak", "windows_coexistence"},
        "acceptance",
    )
    if acceptance["performance"] != {
        "timed_requests_required": 100,
        "p95_ttft_seconds_max": 3,
        "projected_128_token_e2e_p95_seconds_max": 10,
        "p5_accepted_decode_tokens_per_second_min": 25,
        "p95_cold_load_to_ready_seconds_max": 120,
    }:
        raise WindowsPrimaryFixtureFailure("performance acceptance mismatch")
    if acceptance["soak"] != {
        "minimum_monotonic_seconds": 28_800,
        "minimum_cycles": 96,
        "minimum_full_restarts": 8,
        "maximum_event_counts": {event: 0 for event in _ZERO_EVENTS},
        "monotonic_memory_growth_allowed": False,
        "maximum_epoch_working_set_growth_fraction": Decimal("0.05"),
        "maximum_global_working_set_growth_fraction": Decimal("0.05"),
    }:
        raise WindowsPrimaryFixtureFailure("soak acceptance mismatch")
    if acceptance["windows_coexistence"] != {
        "minimum_free_vram_mib": 2048,
        "maximum_asr_p95_degradation_ratio": Decimal("0.15"),
        "maximum_llm_p95_regression_ratio": Decimal("0.15"),
        "minimum_overlapping_utterances": 80,
        "minimum_greek_language_correct_per_100": 95,
        "minimum_english_language_correct_per_100": 95,
        "maximum_duplicate_final_transcripts": 0,
        "maximum_stale_results": 0,
        "required_forced_cpu_fallbacks": 5,
    }:
        raise WindowsPrimaryFixtureFailure("coexistence acceptance mismatch")

    templates = _exact(
        root["templates"],
        {"system_prompt", "locales", "structured_schema", "tool"},
        "templates",
    )
    if (
        not isinstance(templates["system_prompt"], str)
        or not templates["system_prompt"]
        or len(templates["system_prompt"].encode("utf-8")) > 4096
    ):
        raise WindowsPrimaryFixtureFailure("invalid system prompt")
    templates["locales"] = _locale_templates(templates["locales"])
    _validate_schema_and_tool(templates)

    cases = root["performance_cases"]
    if not isinstance(cases, list) or len(cases) != 100:
        raise WindowsPrimaryFixtureFailure("invalid performance case count")
    validated_cases = [_validate_case(case, index) for index, case in enumerate(cases)]
    if len({case["case_id"] for case in validated_cases}) != 100:
        raise WindowsPrimaryFixtureFailure("duplicate performance case identifier")
    if Counter(case["language"] for case in validated_cases) != {"el": 50, "en": 50}:
        raise WindowsPrimaryFixtureFailure("language balance mismatch")
    if Counter(case["target_rendered_bin"] for case in validated_cases) != {
        512: 34,
        2048: 34,
        4096: 32,
    }:
        raise WindowsPrimaryFixtureFailure("target-bin balance mismatch")
    if Counter(case["variant"] for case in validated_cases) != {
        variant: 20 for variant in _VARIANTS
    }:
        raise WindowsPrimaryFixtureFailure("request-variant balance mismatch")

    soak = _exact(
        root["soak"],
        {
            "status",
            "minimum_monotonic_seconds",
            "minimum_cycles",
            "cycle_operations",
            "request_set",
            "full_restart_every_cycles",
            "minimum_full_restarts",
            "memory_sampling",
            "required_zero_events",
            "content_persistence",
        },
        "soak",
    )
    if soak != {
        "status": "pending_native",
        "minimum_monotonic_seconds": 28_800,
        "minimum_cycles": 96,
        "cycle_operations": [
            "request_set",
            "per_request_slot_cleanup",
            "combined_gateway_engine_working_set_sample",
        ],
        "request_set": "all_cases_in_frozen_order",
        "full_restart_every_cycles": 12,
        "minimum_full_restarts": 8,
        "memory_sampling": {
            "scope": "combined_gateway_and_engine_working_set",
            "minimum_samples_per_completed_epoch": 12,
            "maximum_epoch_growth_fraction": Decimal("0.05"),
            "maximum_global_growth_fraction": Decimal("0.05"),
            "growth_estimator": "robust_windows_plus_theil_sen",
        },
        "required_zero_events": list(_ZERO_EVENTS),
        "content_persistence": "forbidden",
    }:
        raise WindowsPrimaryFixtureFailure("soak contract mismatch")

    expanded = _hex_digest(root["expanded_requests_sha256"], "expanded request digest")
    rendered = _hex_digest(
        root["rendered_prompt_texts_sha256"], "rendered prompt digest"
    )
    calculated_expanded, calculated_rendered = derive_fixture_digests(root)
    if expanded != calculated_expanded or rendered != calculated_rendered:
        raise WindowsPrimaryFixtureFailure("materialized prompt identity mismatch")
    requests = tuple(_materialize_request(case, templates) for case in validated_cases)
    return LoadedWindowsPrimaryFixture(
        document=root,
        requests=requests,
        fixture_sha256="",
        expanded_requests_sha256=expanded,
        rendered_prompt_texts_sha256=rendered,
    )


def load_fixture(path: Path) -> LoadedWindowsPrimaryFixture:
    raw = _safe_regular_file(path, maximum=FIXTURE_BYTES, name="fixture")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FROZEN_FIXTURE_SHA256:
        raise WindowsPrimaryFixtureFailure("fixture digest mismatch")
    loaded = validate_fixture(parse_body(raw))
    return LoadedWindowsPrimaryFixture(
        document=loaded.document,
        requests=loaded.requests,
        fixture_sha256=digest,
        expanded_requests_sha256=loaded.expanded_requests_sha256,
        rendered_prompt_texts_sha256=loaded.rendered_prompt_texts_sha256,
    )
