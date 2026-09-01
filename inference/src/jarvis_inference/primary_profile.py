from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .profile_preflight import Fixture, LoopbackClient, PreflightFailure, percentile
from .profile_quality import (
    CATEGORIES,
    FROZEN_CORPUS_SHA256,
    _groups,
    _protocol_failure,
    _sanitized_result,
    load_corpus as load_fallback_corpus,
)
from .strict_json import parse_body


PRIMARY_QUALITY_SUITE_VERSION = "cyrus-local-primary-quality-v1"
PRIMARY_CORPUS_SHA256 = "03223a4ce1c0afdefa62c2189e575e594f3a0341c10f8f9ddc48d6e9dacd3d62"
PRIMARY_SYSTEM_PROMPT = (
    "You are the CYRUS local primary assistant. Answer in the user's language. "
    "Follow every requested word, sentence, and output-format limit exactly. "
    "Be concise, factually accurate, and honest about uncertainty. Never claim "
    "that a tool or action completed unless a tool result in the conversation proves it."
)
_EXTENSION_ID = re.compile(
    r"^(el|en)_(instruction|extraction|summary|reasoning|knowledge)_(09|10|11|12)$",
    re.ASCII,
)


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PreflightFailure(f"invalid {name} shape")
    return value


def _extension_fixture(value: Any, index: int) -> Fixture:
    item = _exact(
        value,
        {
            "case_id",
            "language",
            "category",
            "prompt",
            "max_tokens",
            "exact",
            "required_groups",
            "forbidden_groups",
            "max_words",
        },
        f"primary extension case {index}",
    )
    case_id = item["case_id"]
    language = item["language"]
    category = item["category"]
    prompt = item["prompt"]
    max_tokens = item["max_tokens"]
    exact = item["exact"]
    max_words = item["max_words"]
    if (
        not isinstance(case_id, str)
        or _EXTENSION_ID.fullmatch(case_id) is None
        or language not in {"el", "en"}
        or category not in CATEGORIES
        or not case_id.startswith(f"{language}_{category}_")
        or not isinstance(prompt, str)
        or not prompt
        or len(prompt.encode("utf-8")) > 1_024
        or type(max_tokens) is not int
        or not 1 <= max_tokens <= 128
        or (
            exact is not None
            and (
                not isinstance(exact, str)
                or not exact
                or len(exact.encode("utf-8")) > 128
            )
        )
        or (
            max_words is not None
            and (type(max_words) is not int or not 1 <= max_words <= 32)
        )
    ):
        raise PreflightFailure("invalid primary extension case")
    required = _groups(item["required_groups"], "required groups")
    forbidden = _groups(item["forbidden_groups"], "forbidden groups")
    if exact is None and not required:
        raise PreflightFailure("primary extension case has no deterministic expectation")
    return Fixture(
        case_id=case_id,
        language=language,
        prompt=prompt,
        max_tokens=max_tokens,
        exact=exact,
        required_groups=required,
        forbidden_groups=forbidden,
        max_words=max_words,
        category=category,
    )


def load_primary_corpus(path: Path) -> tuple[tuple[Fixture, ...], str]:
    if path.is_symlink() or not path.is_file():
        raise PreflightFailure("primary corpus path is unsafe")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PRIMARY_CORPUS_SHA256:
        raise PreflightFailure("primary corpus digest mismatch")
    document = _exact(
        parse_body(raw),
        {
            "fixture_schema_version",
            "suite_id",
            "system_prompt",
            "base_corpus",
            "extension_cases",
        },
        "primary corpus",
    )
    base = _exact(document["base_corpus"], {"path", "sha256", "cases"}, "base corpus")
    if (
        document["fixture_schema_version"] != 1
        or document["suite_id"] != PRIMARY_QUALITY_SUITE_VERSION
        or document["system_prompt"] != PRIMARY_SYSTEM_PROMPT
        or base
        != {
            "path": "local-fallback-quality-v1.json",
            "sha256": FROZEN_CORPUS_SHA256,
            "cases": 80,
        }
        or not isinstance(document["extension_cases"], list)
        or len(document["extension_cases"]) != 40
    ):
        raise PreflightFailure("primary corpus identity mismatch")
    base_path = path.parent / base["path"]
    if base_path.is_symlink() or base_path.parent.resolve() != path.parent.resolve():
        raise PreflightFailure("primary base corpus path is unsafe")
    base_fixtures, base_digest = load_fallback_corpus(base_path)
    if base_digest != base["sha256"]:
        raise PreflightFailure("primary base corpus identity mismatch")
    extension = tuple(
        _extension_fixture(item, index)
        for index, item in enumerate(document["extension_cases"])
    )
    ids = [fixture.case_id for fixture in (*base_fixtures, *extension)]
    if len(ids) != len(set(ids)):
        raise PreflightFailure("primary corpus has duplicate case identifiers")
    counts = Counter(
        (fixture.language, fixture.category)
        for fixture in (*base_fixtures, *extension)
    )
    if any(
        counts[(language, category)] != 12
        for language in ("el", "en")
        for category in CATEGORIES
    ):
        raise PreflightFailure("primary corpus category balance mismatch")
    return (*base_fixtures, *extension), digest


def run_primary_quality_gate(
    client: LoopbackClient,
    fixtures: tuple[Fixture, ...],
    corpus_digest: str,
    *,
    warmups: int,
) -> dict[str, Any]:
    if len(fixtures) != 120 or not 1 <= warmups <= 20:
        raise ValueError("primary quality gate counts are invalid")
    client.readiness()
    client.capabilities()
    warmup = Fixture(
        "warmup", "en", "Reply with only this word: ready", 16, exact="ready"
    )
    for _ in range(warmups):
        client.stream(warmup)
    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        try:
            results.append(_sanitized_result(client.stream(fixture)))
        except (OSError, PreflightFailure):
            results.append(_protocol_failure(fixture))
    by_language: dict[str, dict[str, Any]] = {}
    for language in ("el", "en"):
        subset = [item for item in results if item["language"] == language]
        passed = sum(bool(item["quality_pass"]) for item in subset)
        by_language[language] = {
            "passed": passed,
            "total": len(subset),
            "pass_rate": passed / len(subset),
        }
    by_category: dict[str, dict[str, int]] = {}
    for category in sorted(CATEGORIES):
        subset = [item for item in results if item["category"] == category]
        by_category[category] = {
            "passed": sum(bool(item["quality_pass"]) for item in subset),
            "total": len(subset),
        }
    critical_failures = sum(
        not bool(item["forbidden_content_ok"]) for item in results
    )
    completed = [item for item in results if item["protocol_pass"]]
    if not completed:
        raise PreflightFailure("primary quality suite produced no complete cases")
    ttft = [float(item["ttft_seconds"]) for item in completed]
    end_to_end = [float(item["end_to_end_seconds"]) for item in completed]
    decode = [float(item["decode_tokens_per_second"]) for item in completed]
    if any(not math.isfinite(value) for value in (*ttft, *end_to_end, *decode)):
        raise PreflightFailure("primary quality metrics are not finite")
    gap = abs(
        float(by_language["el"]["pass_rate"])
        - float(by_language["en"]["pass_rate"])
    )
    thresholds = {
        "greek_pass_rate_at_least_90_percent": by_language["el"]["pass_rate"]
        >= 0.90,
        "english_pass_rate_at_least_90_percent": by_language["en"]["pass_rate"]
        >= 0.90,
        "language_gap_at_most_10_percentage_points": gap <= 0.10,
        "critical_failures_zero": critical_failures == 0,
        "protocol_complete_120_of_120": len(completed) == 120,
    }
    return {
        "report_schema_version": 1,
        "suite_version": PRIMARY_QUALITY_SUITE_VERSION,
        "authoritative_qualification": False,
        "profile_id": "local-primary",
        "public_model_alias": "jarvis-local",
        "corpus_sha256": corpus_digest,
        "system_prompt_sha256": hashlib.sha256(
            PRIMARY_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "warmups": warmups,
        "sample_count": len(results),
        "protocol_failures": len(results) - len(completed),
        "language_results": by_language,
        "category_results": by_category,
        "language_pass_rate_gap": gap,
        "critical_failures": critical_failures,
        "metrics": {
            "ttft_p50_seconds": statistics.median(ttft),
            "ttft_p95_seconds": percentile(ttft, 0.95),
            "end_to_end_p95_seconds": percentile(end_to_end, 0.95),
            "decode_p5_tokens_per_second": percentile(decode, 0.05),
            "decode_p50_tokens_per_second": statistics.median(decode),
        },
        "thresholds": thresholds,
        "cases": results,
        "status": (
            "deterministic_quality_pass"
            if all(thresholds.values())
            else "deterministic_quality_failed"
        ),
    }


def _bearer_from_config(path: Path) -> str:
    value = parse_body(path.read_bytes())
    bearer = value.get("inference_bearer_token") if isinstance(value, dict) else None
    if not isinstance(bearer, str) or len(bearer.encode("utf-8")) < 32:
        raise PreflightFailure("orchestrator config has no valid private bearer")
    return bearer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run-primary-profile-quality")
    parser.add_argument("--base-url", default="http://127.0.0.1:8766/v1")
    parser.add_argument("--orchestrator-config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        fixtures, digest = load_primary_corpus(args.corpus)
        report = run_primary_quality_gate(
            LoopbackClient(
                args.base_url,
                _bearer_from_config(args.orchestrator_config),
                expected_capabilities={
                    "text": True,
                    "stream": True,
                    "tools": True,
                    "json_object": True,
                    "json_schema": True,
                },
                system_prompt=PRIMARY_SYSTEM_PROMPT,
            ),
            fixtures,
            digest,
            warmups=args.warmups,
        )
    except (OSError, ValueError, PreflightFailure):
        print(
            json.dumps(
                {
                    "status": "quality_error",
                    "authoritative_qualification": False,
                },
                separators=(",", ":"),
            )
        )
        return 2
    public_report = (
        {key: value for key, value in report.items() if key != "cases"}
        if args.summary_only
        else report
    )
    print(
        json.dumps(
            public_report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if report["status"] == "deterministic_quality_pass" else 1


__all__ = [
    "PRIMARY_CORPUS_SHA256",
    "PRIMARY_QUALITY_SUITE_VERSION",
    "PRIMARY_SYSTEM_PROMPT",
    "load_primary_corpus",
    "run_primary_quality_gate",
]
