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

from .profile_preflight import FALLBACK_SYSTEM_PROMPT, Fixture, LoopbackClient, PreflightFailure, percentile
from .strict_json import parse_body


QUALITY_SUITE_VERSION = "jarvis-local-fallback-quality-v1"
FROZEN_CORPUS_SHA256 = "cc3814b05807908f1235db815f078f5ac2a4034ad3844e9dcc03a85c5d1fdd71"
CATEGORIES = {"instruction", "extraction", "summary", "reasoning", "knowledge"}
CASE_ID_RE = re.compile(r"^(el|en)_(instruction|extraction|summary|reasoning|knowledge)_[0-9]{2}$", re.ASCII)


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PreflightFailure(f"invalid {name} shape")
    return value


def _groups(value: Any, name: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or len(value) > 8:
        raise PreflightFailure(f"invalid {name}")
    result: list[tuple[str, ...]] = []
    for group in value:
        if (
            not isinstance(group, list)
            or not group
            or len(group) > 8
            or any(not isinstance(item, str) or not item or len(item.encode("utf-8")) > 128 for item in group)
        ):
            raise PreflightFailure(f"invalid {name}")
        result.append(tuple(group))
    return tuple(result)


def load_corpus(path: Path) -> tuple[tuple[Fixture, ...], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FROZEN_CORPUS_SHA256:
        raise PreflightFailure("quality corpus digest mismatch")
    document = _exact(
        parse_body(raw),
        {"fixture_schema_version", "suite_id", "system_prompt", "cases"},
        "quality corpus",
    )
    if (
        document["fixture_schema_version"] != 1
        or document["suite_id"] != QUALITY_SUITE_VERSION
        or document["system_prompt"] != FALLBACK_SYSTEM_PROMPT
        or not isinstance(document["cases"], list)
        or len(document["cases"]) != 80
    ):
        raise PreflightFailure("quality corpus identity mismatch")
    fixtures: list[Fixture] = []
    ids: set[str] = set()
    for index, value in enumerate(document["cases"]):
        item = _exact(
            value,
            {
                "case_id", "language", "category", "prompt", "max_tokens", "exact",
                "required_groups", "forbidden_groups", "max_words",
            },
            f"quality case {index}",
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
            or not CASE_ID_RE.fullmatch(case_id)
            or case_id in ids
            or language not in {"el", "en"}
            or category not in CATEGORIES
            or not case_id.startswith(language + "_" + category + "_")
            or not isinstance(prompt, str)
            or not prompt
            or len(prompt.encode("utf-8")) > 1024
            or type(max_tokens) is not int
            or not 1 <= max_tokens <= 128
            or (exact is not None and (not isinstance(exact, str) or not exact or len(exact.encode("utf-8")) > 128))
            or (max_words is not None and (type(max_words) is not int or not 1 <= max_words <= 32))
        ):
            raise PreflightFailure("invalid quality case")
        required = _groups(item["required_groups"], "required groups")
        forbidden = _groups(item["forbidden_groups"], "forbidden groups")
        if exact is None and not required:
            raise PreflightFailure("quality case has no deterministic expectation")
        ids.add(case_id)
        fixtures.append(Fixture(
            case_id=case_id,
            language=language,
            prompt=prompt,
            max_tokens=max_tokens,
            exact=exact,
            required_groups=required,
            forbidden_groups=forbidden,
            max_words=max_words,
            category=category,
        ))
    counts = Counter((item.language, item.category) for item in fixtures)
    if any(counts[(language, category)] != 8 for language in ("el", "en") for category in CATEGORIES):
        raise PreflightFailure("quality corpus category balance mismatch")
    return tuple(fixtures), digest


def _sanitized_result(item: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "case_id", "language", "category", "protocol_pass", "language_ok", "exact_ok",
        "required_content_ok", "forbidden_content_ok", "word_limit_ok", "completed_ok",
        "quality_pass", "finish_reason", "ttft_seconds", "end_to_end_seconds",
        "completion_tokens", "decode_tokens_per_second", "output_utf8_bytes",
    }
    if set(item) != allowed:
        raise PreflightFailure("quality result has an unsafe shape")
    return {key: item[key] for key in sorted(allowed)}


def run_quality_gate(
    client: LoopbackClient,
    fixtures: tuple[Fixture, ...],
    corpus_digest: str,
    *,
    warmups: int,
) -> dict[str, Any]:
    if len(fixtures) != 80 or not 1 <= warmups <= 20:
        raise ValueError("quality gate counts are invalid")
    client.readiness()
    client.capabilities()
    warmup = Fixture("warmup", "en", "Reply with only this word: ready", 16, exact="ready")
    for _ in range(warmups):
        client.stream(warmup)
    results = [_sanitized_result(client.stream(fixture)) for fixture in fixtures]
    by_language: dict[str, dict[str, Any]] = {}
    for language in ("el", "en"):
        subset = [item for item in results if item["language"] == language]
        passed = sum(bool(item["quality_pass"]) for item in subset)
        by_language[language] = {"passed": passed, "total": len(subset), "pass_rate": passed / len(subset)}
    by_category: dict[str, dict[str, int]] = {}
    for category in sorted(CATEGORIES):
        subset = [item for item in results if item["category"] == category]
        by_category[category] = {
            "passed": sum(bool(item["quality_pass"]) for item in subset),
            "total": len(subset),
        }
    # Tool/action/research routing is an orchestrator pre-inference gate and is
    # deliberately not scored by this stateless model corpus. A critical model
    # failure here is observable forbidden content in any deterministic case.
    critical_failures = sum(not bool(item["forbidden_content_ok"]) for item in results)
    ttft = [float(item["ttft_seconds"]) for item in results]
    end_to_end = [float(item["end_to_end_seconds"]) for item in results]
    decode = [float(item["decode_tokens_per_second"]) for item in results]
    if any(not math.isfinite(value) for value in (*ttft, *end_to_end, *decode)):
        raise PreflightFailure("quality metrics are not finite")
    parity_gap = abs(float(by_language["el"]["pass_rate"]) - float(by_language["en"]["pass_rate"]))
    thresholds = {
        "greek_pass_rate_at_least_85_percent": by_language["el"]["pass_rate"] >= 0.85,
        "english_pass_rate_at_least_85_percent": by_language["en"]["pass_rate"] >= 0.85,
        "language_gap_at_most_10_percentage_points": parity_gap <= 0.10,
        "critical_failures_zero": critical_failures == 0,
        "protocol_complete_80_of_80": all(bool(item["protocol_pass"]) for item in results),
    }
    return {
        "report_schema_version": 1,
        "suite_version": QUALITY_SUITE_VERSION,
        "authoritative_qualification": False,
        "profile_id": "local-fallback",
        "public_model_alias": "jarvis-local",
        "corpus_sha256": corpus_digest,
        "system_prompt_sha256": hashlib.sha256(FALLBACK_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "warmups": warmups,
        "sample_count": len(results),
        "language_results": by_language,
        "category_results": by_category,
        "language_pass_rate_gap": parity_gap,
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
        "status": "deterministic_quality_pass" if all(thresholds.values()) else "deterministic_quality_failed",
    }


def _bearer_from_config(path: Path) -> str:
    value = parse_body(path.read_bytes())
    bearer = value.get("inference_bearer_token") if isinstance(value, dict) else None
    if not isinstance(bearer, str) or len(bearer.encode("utf-8")) < 32:
        raise PreflightFailure("orchestrator config has no valid private bearer")
    return bearer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run-profile-quality")
    parser.add_argument("--base-url", default="http://127.0.0.1:8766/v1")
    parser.add_argument("--orchestrator-config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        fixtures, digest = load_corpus(args.corpus)
        report = run_quality_gate(
            LoopbackClient(args.base_url, _bearer_from_config(args.orchestrator_config)),
            fixtures,
            digest,
            warmups=args.warmups,
        )
    except (OSError, ValueError, PreflightFailure):
        print(json.dumps({"status": "quality_error", "authoritative_qualification": False}, separators=(",", ":")))
        return 2
    public_report = {key: value for key, value in report.items() if key != "cases"} if args.summary_only else report
    print(json.dumps(public_report, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "deterministic_quality_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
