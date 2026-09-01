from __future__ import annotations

import math
import re
import hashlib
from decimal import Decimal
from typing import Any

from .profile_preflight import percentile
from .strict_json import canonical_bytes
from .windows_primary_fixture import FROZEN_FIXTURE_SHA256, SUITE_ID


_SHA = re.compile(r"^[0-9a-f]{64}$", re.ASCII)


class ProfilePerformanceFailure(ValueError):
    """A sanitized, fail-closed native performance-evidence failure."""


def _fail(message: str) -> ProfilePerformanceFailure:
    return ProfilePerformanceFailure(message)


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


def validate_performance(
    value: Any,
    fixture: Any,
    manifest: Any,
    current_identity: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate and independently recompute sanitized native performance."""

    performance = _exact(
        value,
        {
            "status",
            "suite_id",
            "fixture_sha256",
            "expanded_requests_sha256",
            "rendered_prompt_texts_sha256",
            "timed_requests_completed",
            "cold_load_to_ready_samples_seconds",
            "native_token_validation",
            "metrics",
            "cases",
        },
        "performance",
    )
    tokenization = _exact(
        performance["native_token_validation"],
        {"status", "validated_cases", "tokenizer_identity_sha256"},
        "native token validation",
    )
    tokenizer_sha = _sha(
        tokenization["tokenizer_identity_sha256"], "tokenizer identity digest"
    )
    cases = performance["cases"]
    if not isinstance(cases, list) or len(cases) != 100:
        raise _fail("invalid performance cases")
    ttft: list[float] = []
    end_to_end: list[float] = []
    decode: list[float] = []
    projected_128: list[float] = []
    previous_serial = 0
    for contract, raw in zip(
        fixture.document["performance_cases"], cases, strict=True
    ):
        case = _exact(
            raw,
            {
                "case_id",
                "target_bin",
                "generation_serial",
                "rendered_token_count",
                "rendered_token_ids_sha256",
                "ttft_seconds",
                "end_to_end_seconds",
                "backend_prompt_seconds",
                "backend_decode_seconds",
                "request_to_engine_final_seconds",
                "accepted_decode_tokens",
                "accepted_token_ids_sha256",
                "decode_tokens_per_second",
                "projected_128_token_e2e_seconds",
                "finish_reason",
                "passed",
            },
            "performance case",
        )
        target_range = fixture.document["execution"]["native_token_count_ranges"]
        target_range = target_range[str(contract["target_rendered_bin"])]
        generation_serial = _integer(
            case["generation_serial"], "generation serial", 1
        )
        if (
            case["case_id"] != contract["case_id"]
            or case["target_bin"] != contract["target_rendered_bin"]
            or generation_serial != previous_serial + 1
            or type(case["rendered_token_count"]) is not int
            or not target_range["minimum"]
            <= case["rendered_token_count"]
            <= target_range["maximum"]
            or not isinstance(case["rendered_token_ids_sha256"], str)
            or _SHA.fullmatch(case["rendered_token_ids_sha256"]) is None
            or type(case["accepted_decode_tokens"]) is not int
            or not 1 <= case["accepted_decode_tokens"] <= 128
            or not isinstance(case["accepted_token_ids_sha256"], str)
            or _SHA.fullmatch(case["accepted_token_ids_sha256"]) is None
            or case["finish_reason"] not in {"stop", "length", "tool_calls"}
            or case["passed"] is not True
        ):
            raise _fail("performance case failed")
        previous_serial = generation_serial
        case_ttft = _number(case["ttft_seconds"], "TTFT", positive=True)
        ttft.append(case_ttft)
        case_end_to_end = _number(
            case["end_to_end_seconds"], "end-to-end latency", positive=True
        )
        end_to_end.append(case_end_to_end)
        backend_prompt = _number(
            case["backend_prompt_seconds"], "backend prompt latency", positive=True
        )
        backend_decode = _number(
            case["backend_decode_seconds"], "backend decode latency", positive=True
        )
        request_to_final = _number(
            case["request_to_engine_final_seconds"],
            "request-to-engine-final latency",
            positive=True,
        )
        case_decode = _number(
            case["decode_tokens_per_second"], "decode throughput", positive=True
        )
        expected_decode = case["accepted_decode_tokens"] / backend_decode
        if (
            not math.isclose(
                case_decode, expected_decode, rel_tol=1e-9, abs_tol=1e-9
            )
            or request_to_final < case_ttft
            or request_to_final < backend_prompt
            or request_to_final < backend_decode
            or case_ttft + 0.05 < backend_prompt
            or request_to_final + 0.05 < backend_prompt + backend_decode
            or case_end_to_end < request_to_final
        ):
            raise _fail("backend performance telemetry mismatch")
        decode.append(case_decode)
        # TTFT includes the first accepted token. A 128-token projection adds
        # only the remaining 127 decoded-token intervals.
        projected = case_ttft + 127 / case_decode
        observed_projected = _number(
            case["projected_128_token_e2e_seconds"],
            "projected 128-token latency",
            positive=True,
        )
        if not math.isclose(
            observed_projected, projected, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise _fail("projected 128-token case latency mismatch")
        projected_128.append(projected)
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
        "performance metrics",
    )
    calculated = {
        "ttft_p50_seconds": percentile(ttft, 0.50),
        "ttft_p95_seconds": percentile(ttft, 0.95),
        "end_to_end_p50_seconds": percentile(end_to_end, 0.50),
        "end_to_end_p95_seconds": percentile(end_to_end, 0.95),
        "projected_128_token_e2e_p95_seconds": percentile(projected_128, 0.95),
        "decode_p5_tokens_per_second": percentile(decode, 0.05),
        "decode_p50_tokens_per_second": percentile(decode, 0.50),
        "decode_p95_tokens_per_second": percentile(decode, 0.95),
    }
    for name, expected in calculated.items():
        observed = _number(metrics[name], name, positive=True)
        if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise _fail("performance aggregate mismatch")
    cold_samples_raw = performance["cold_load_to_ready_samples_seconds"]
    if not isinstance(cold_samples_raw, list) or len(cold_samples_raw) < 8:
        raise _fail("invalid cold-load sample set")
    cold_samples = [
        _number(value, "cold-load sample", positive=True)
        for value in cold_samples_raw
    ]
    cold = _number(
        metrics["cold_load_to_ready_p95_seconds"], "cold-load p95", positive=True
    )
    if not math.isclose(
        cold, percentile(cold_samples, 0.95), rel_tol=1e-9, abs_tol=1e-9
    ):
        raise _fail("cold-load aggregate mismatch")
    artifact = manifest.model["artifacts"][0]
    expected_tokenizer_identity = {
        "schema": "cyrus.windows-native-tokenizer-identity/1",
        "manifest_sha256": manifest.digest,
        "runtime_identity_sha256": hashlib.sha256(
            canonical_bytes(current_identity)
        ).hexdigest(),
        "backend": {
            "llama_cpp_commit": manifest.backend["llama_cpp_commit"],
            "build_id": manifest.backend["build_id"],
        },
        "model_artifact_sha256": artifact["sha256"],
        "prompt_profile_id": manifest.prompt_profile["id"],
        "renderer_sha256": manifest.prompt_profile["renderer"]["sha256"],
        "cases": [
            {
                "case_id": case["case_id"],
                "rendered_token_count": case["rendered_token_count"],
                "rendered_token_ids_sha256": case[
                    "rendered_token_ids_sha256"
                ],
            }
            for case in cases
        ],
    }
    expected_tokenizer_sha = hashlib.sha256(
        canonical_bytes(expected_tokenizer_identity)
    ).hexdigest()
    if (
        performance["status"] != "pass"
        or performance["suite_id"] != SUITE_ID
        or performance["fixture_sha256"] != FROZEN_FIXTURE_SHA256
        or performance["expanded_requests_sha256"]
        != fixture.expanded_requests_sha256
        or performance["rendered_prompt_texts_sha256"]
        != fixture.rendered_prompt_texts_sha256
        or performance["timed_requests_completed"] != 100
        or tokenization["status"] != "pass"
        or tokenization["validated_cases"] != 100
        or tokenizer_sha != expected_tokenizer_sha
        or calculated["ttft_p95_seconds"] > 3.0
        or calculated["projected_128_token_e2e_p95_seconds"] > 10.0
        or calculated["decode_p5_tokens_per_second"] < 25.0
        or cold > 120.0
    ):
        raise _fail("performance gate failed")
    return performance, expected_tokenizer_sha
