"""Fail-closed online retrieval quality qualification for Research System v1.

The module deliberately separates benchmark review from execution.  An
unreviewed benchmark, an unpinned image, or missing safety evidence can only
produce a REJECTED record.  It never invents relevance judgements and never
contains a model client.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx

from jarvis_orchestrator.domain import canonical_json


ONLINE_GATE_VERSION = "online-retrieval-quality/1"
BENCHMARK_VERSION = "research-retrieval-benchmark/1"
IMAGE_MANIFEST_VERSION = "research-image-manifest/1"
EVIDENCE_SCHEMA_VERSION = 1
LANGUAGES = frozenset({"el", "en"})
REVIEW_STATES = frozenset({"UNREVIEWED", "REVIEWED"})
RESULT_STATUSES = frozenset({"SUCCESS", "TECHNICAL_FAILURE", "CANCELLED"})
IMAGE_NAMES = frozenset({"research-api", "egress-guard", "searxng"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE_REF = re.compile(
    r"(?P<repository>[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+)@sha256:(?P<digest>[0-9a-f]{64})"
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "overall_recall_at_5": 0.75,
    "language_recall_at_5": 0.70,
    "overall_recall_at_10": 0.85,
    "language_recall_at_10": 0.80,
    "overall_mrr_at_10": 0.65,
    "language_mrr_at_10": 0.60,
    "primary_source_recall": 0.80,
    "quote_reconstruction": 1.0,
    "citation_coverage": 1.0,
    "p95_latency_seconds": 45.0,
}
THRESHOLD_MANIFEST_VERSION = "online-retrieval-thresholds/1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("json_artifact_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("json_artifact_invalid")
    return value


def _safe_id(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", value) is None:
        raise ValueError(field + "_invalid")
    return value


def _iso_datetime(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError(field + "_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(field + "_invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(field + "_invalid")
    return value


@dataclass(frozen=True, slots=True)
class RegistryImageV1:
    name: str
    reference: str
    platform: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        if self.name not in IMAGE_NAMES:
            raise ValueError("image_name_invalid")
        if _IMAGE_REF.fullmatch(self.reference) is None:
            raise ValueError("image_reference_not_digest_pinned")
        if self.platform not in {"linux/amd64", "linux/arm64"}:
            raise ValueError("image_platform_invalid")
        if _SHA256.fullmatch(self.configuration_sha256) is None:
            raise ValueError("image_configuration_digest_invalid")

    @property
    def manifest_sha256(self) -> str:
        match = _IMAGE_REF.fullmatch(self.reference)
        assert match is not None
        return match.group("digest")


def parse_image_manifest(value: Mapping[str, object]) -> tuple[RegistryImageV1, ...]:
    if set(value) != {"version", "images"} or value.get("version") != IMAGE_MANIFEST_VERSION:
        raise ValueError("image_manifest_schema_invalid")
    raw = value.get("images")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("image_manifest_schema_invalid")
    images: list[RegistryImageV1] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {
            "name", "reference", "platform", "configuration_sha256"
        }:
            raise ValueError("image_manifest_schema_invalid")
        images.append(RegistryImageV1(**item))
    if {item.name for item in images} != IMAGE_NAMES:
        raise ValueError("image_manifest_schema_invalid")
    return tuple(sorted(images, key=lambda item: item.name))


def parse_threshold_manifest(value: Mapping[str, object]) -> dict[str, float]:
    if set(value) != {"version", "thresholds"} or value.get("version") != THRESHOLD_MANIFEST_VERSION:
        raise ValueError("threshold_manifest_invalid")
    raw = value.get("thresholds")
    if not isinstance(raw, dict) or set(raw) != set(DEFAULT_THRESHOLDS):
        raise ValueError("threshold_manifest_invalid")
    result: dict[str, float] = {}
    for key, item in raw.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError("threshold_manifest_invalid")
        selected = float(item)
        if key == "p95_latency_seconds":
            if not 1.0 <= selected <= 45.0:
                raise ValueError("threshold_manifest_invalid")
        elif not 0.0 <= selected <= 1.0:
            raise ValueError("threshold_manifest_invalid")
        result[key] = selected
    return result


def verify_runtime_image(
    declared: RegistryImageV1,
    *,
    repo_digests: Sequence[str],
    configuration_id: str,
    os_name: str,
    architecture: str,
) -> bool:
    expected_config = "sha256:" + declared.configuration_sha256
    return (
        declared.reference in repo_digests
        and configuration_id == expected_config
        and f"{os_name}/{architecture}" == declared.platform
    )


def inspect_runtime_images(images: Sequence[RegistryImageV1]) -> tuple[str, dict[str, object]]:
    docker = shutil.which("docker", path=os.environ.get("PATH"))
    if docker is None:
        return "DOCKER_UNAVAILABLE", {}
    observed: dict[str, object] = {}
    for image in images:
        try:
            result = subprocess.run(
                [docker, "image", "inspect", image.reference],
                text=True, capture_output=True, check=False, timeout=15,
                env={"PATH": os.path.dirname(docker) + ":/usr/local/bin:/usr/bin:/bin"},
            )
            raw = json.loads(result.stdout)
            info = raw[0]
            passed = result.returncode == 0 and verify_runtime_image(
                image,
                repo_digests=info["RepoDigests"],
                configuration_id=info["Id"],
                os_name=info["Os"],
                architecture=info["Architecture"],
            )
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, IndexError, TypeError):
            passed = False
        observed[image.name] = {
            "manifest_sha256": image.manifest_sha256,
            "configuration_sha256": image.configuration_sha256,
            "platform": image.platform,
            "runtime_match": passed,
        }
    return ("VERIFIED" if all(item["runtime_match"] for item in observed.values()) else "MISMATCH"), observed


@dataclass(frozen=True, slots=True)
class QrelV1:
    qrel_id: str
    target_kind: str
    target: str
    relevance: int
    primary: bool
    facet_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCaseV1:
    case_id: str
    language: str
    canonical_query: str
    category: str
    facets: tuple[tuple[str, str], ...]
    freshness: str
    reference_time: str
    timezone: str
    qrels: tuple[QrelV1, ...]
    acceptable_primary_domains: tuple[str, ...]
    known_independence_groups: tuple[tuple[str, ...], ...]
    expected_conflict: str
    sensitivity: str
    review_state: str
    review_provenance: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkV1:
    version: str
    review_state: str
    review_provenance: str | None
    cases: tuple[BenchmarkCaseV1, ...]
    sha256: str
    qrels_sha256: str


def parse_benchmark(value: Mapping[str, object]) -> BenchmarkV1:
    if set(value) != {"version", "review_state", "review_provenance", "cases"}:
        raise ValueError("benchmark_schema_invalid")
    if value.get("version") != BENCHMARK_VERSION or value.get("review_state") not in REVIEW_STATES:
        raise ValueError("benchmark_schema_invalid")
    root_provenance = value.get("review_provenance")
    if root_provenance is not None and (not isinstance(root_provenance, str) or not 1 <= len(root_provenance) <= 256):
        raise ValueError("benchmark_schema_invalid")
    raw_cases = value.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 100:
        raise ValueError("benchmark_case_count_invalid")
    cases: list[BenchmarkCaseV1] = []
    ids: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict) or set(raw) != {
            "case_id", "language", "canonical_query", "category", "facets", "freshness",
            "reference_time", "timezone", "qrels", "acceptable_primary_domains",
            "known_independence_groups", "expected_conflict", "sensitivity",
            "review_state", "review_provenance",
        }:
            raise ValueError("benchmark_case_schema_invalid")
        case_id = _safe_id(raw["case_id"], "case_id")
        if case_id in ids:
            raise ValueError("benchmark_duplicate_case")
        ids.add(case_id)
        language = raw["language"]
        query = raw["canonical_query"]
        if language not in LANGUAGES or not isinstance(query, str) or not 1 <= len(query) <= 512:
            raise ValueError("benchmark_query_invalid")
        if query != __import__("unicodedata").normalize("NFC", query):
            raise ValueError("benchmark_query_invalid")
        raw_facets = raw["facets"]
        if not isinstance(raw_facets, list) or not 1 <= len(raw_facets) <= 6:
            raise ValueError("benchmark_facets_invalid")
        facets: list[tuple[str, str]] = []
        for facet in raw_facets:
            if not isinstance(facet, dict) or set(facet) != {"facet_id", "label"}:
                raise ValueError("benchmark_facets_invalid")
            facet_id = _safe_id(facet["facet_id"], "facet_id")
            label = facet["label"]
            if not isinstance(label, str) or not 1 <= len(label) <= 160:
                raise ValueError("benchmark_facets_invalid")
            facets.append((facet_id, label))
        qrels = _parse_qrels(raw["qrels"], {item[0] for item in facets})
        primary = _string_list(raw["acceptable_primary_domains"], 16, "primary_domains")
        groups = _groups(raw["known_independence_groups"])
        case_review = raw["review_state"]
        provenance = raw["review_provenance"]
        if case_review not in REVIEW_STATES or (provenance is not None and not isinstance(provenance, str)):
            raise ValueError("benchmark_review_invalid")
        if case_review == "REVIEWED" and (not qrels or not provenance):
            raise ValueError("benchmark_review_invalid")
        if case_review == "UNREVIEWED" and provenance is not None:
            raise ValueError("benchmark_review_invalid")
        cases.append(BenchmarkCaseV1(
            case_id, language, query, str(raw["category"]), tuple(facets), str(raw["freshness"]),
            _iso_datetime(raw["reference_time"], "reference_time"), str(raw["timezone"]), qrels,
            primary, groups, str(raw["expected_conflict"]), str(raw["sensitivity"]),
            str(case_review), provenance,
        ))
    counts = {language: sum(item.language == language for item in cases) for language in LANGUAGES}
    if counts != {"el": 50, "en": 50}:
        raise ValueError("benchmark_language_balance_invalid")
    root_review = str(value["review_state"])
    all_reviewed = all(item.review_state == "REVIEWED" for item in cases)
    if (root_review == "REVIEWED") != all_reviewed or (root_review == "REVIEWED" and not root_provenance):
        raise ValueError("benchmark_review_invalid")
    canonical_cases = sorted(cases, key=lambda item: item.case_id)
    qrels_payload = [
        {"case_id": item.case_id, "qrels": [qrel.__dict__ if hasattr(qrel, "__dict__") else {
            "qrel_id": qrel.qrel_id, "target_kind": qrel.target_kind, "target": qrel.target,
            "relevance": qrel.relevance, "primary": qrel.primary, "facet_ids": list(qrel.facet_ids),
        } for qrel in item.qrels]}
        for item in canonical_cases
    ]
    return BenchmarkV1(
        BENCHMARK_VERSION, root_review, root_provenance, tuple(canonical_cases),
        _sha256_json(value), _sha256_json(qrels_payload),
    )


def _parse_qrels(raw: object, facet_ids: set[str]) -> tuple[QrelV1, ...]:
    if not isinstance(raw, list) or len(raw) > 32:
        raise ValueError("benchmark_qrels_invalid")
    result: list[QrelV1] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != {
            "qrel_id", "target_kind", "target", "relevance", "primary", "facet_ids"
        }:
            raise ValueError("benchmark_qrels_invalid")
        qrel_id = _safe_id(item["qrel_id"], "qrel_id")
        target_kind, target = item["target_kind"], item["target"]
        if qrel_id in seen or target_kind not in {"URL", "DOMAIN", "SOURCE_IDENTITY"}:
            raise ValueError("benchmark_qrels_invalid")
        if not isinstance(target, str) or not 1 <= len(target) <= 2048:
            raise ValueError("benchmark_qrels_invalid")
        relevance = item["relevance"]
        primary = item["primary"]
        selected_facets = _string_list(item["facet_ids"], 6, "qrel_facets")
        if isinstance(relevance, bool) or relevance not in {1, 2} or not isinstance(primary, bool):
            raise ValueError("benchmark_qrels_invalid")
        if not set(selected_facets) <= facet_ids:
            raise ValueError("benchmark_qrels_invalid")
        seen.add(qrel_id)
        result.append(QrelV1(qrel_id, target_kind, target, relevance, primary, selected_facets))
    return tuple(result)


def _string_list(raw: object, maximum: int, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) > maximum or any(
        not isinstance(item, str) or not 1 <= len(item) <= 2048 for item in raw
    ):
        raise ValueError(field + "_invalid")
    if len(set(raw)) != len(raw):
        raise ValueError(field + "_invalid")
    return tuple(raw)


def _groups(raw: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw, list) or len(raw) > 16:
        raise ValueError("independence_groups_invalid")
    result = tuple(_string_list(item, 16, "independence_group") for item in raw)
    if any(len(item) < 2 for item in result):
        raise ValueError("independence_groups_invalid")
    return result


@dataclass(frozen=True, slots=True)
class RankedResultV1:
    url: str
    registrable_domain: str
    source_identity: str
    source_role: str


@dataclass(frozen=True, slots=True)
class CaseEvaluationV1:
    case_id: str
    language: str
    status: str
    ranked: tuple[RankedResultV1, ...]
    covered_facets: tuple[str, ...]
    latency_seconds: float
    compressed_bytes: int
    decompressed_bytes: int
    normalized_bytes: int
    retry_count: int


def _matches(qrel: QrelV1, item: RankedResultV1) -> bool:
    target = qrel.target.casefold()
    if qrel.target_kind == "URL":
        return item.url.casefold() == target
    if qrel.target_kind == "DOMAIN":
        return item.registrable_domain.casefold() == target
    return item.source_identity.casefold() == target


def compute_retrieval_metrics(
    benchmark: BenchmarkV1, evaluations: Sequence[CaseEvaluationV1]
) -> dict[str, object]:
    by_case = {item.case_id: item for item in evaluations}
    if len(by_case) != len(evaluations) or set(by_case) != {item.case_id for item in benchmark.cases}:
        raise ValueError("evaluation_case_set_invalid")
    per_language: dict[str, dict[str, float]] = {}
    rows: list[dict[str, float | str]] = []
    for case in benchmark.cases:
        evaluated = by_case[case.case_id]
        if evaluated.language != case.language or evaluated.status not in RESULT_STATUSES:
            raise ValueError("evaluation_case_invalid")
        relevant = case.qrels
        hits: list[int] = []
        primary_total = sum(item.primary for item in relevant)
        primary_hits = 0
        for qrel in relevant:
            ranks = [rank for rank, item in enumerate(evaluated.ranked, 1) if _matches(qrel, item)]
            if ranks:
                hits.append(min(ranks))
                if qrel.primary:
                    primary_hits += 1
        denominator = len(relevant)
        row: dict[str, float | str] = {"language": case.language}
        for cutoff in (1, 3, 5, 10):
            row[f"recall_at_{cutoff}"] = (
                sum(rank <= cutoff for rank in hits) / denominator if denominator else 0.0
            )
        row["mrr_at_10"] = max((1.0 / rank for rank in hits if rank <= 10), default=0.0)
        row["primary_source_recall"] = primary_hits / primary_total if primary_total else math.nan
        row["facet_coverage"] = len(set(evaluated.covered_facets)) / len(case.facets)
        rows.append(row)
    for language in sorted(LANGUAGES):
        per_language[language] = _aggregate_rows([row for row in rows if row["language"] == language])
    return {
        "overall_macro": _aggregate_rows(rows),
        "per_language": per_language,
        "case_count": len(rows),
    }


def _aggregate_rows(rows: Sequence[Mapping[str, float | str]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in (
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "mrr_at_10", "primary_source_recall", "facet_coverage",
    ):
        values = [float(row[name]) for row in rows if not math.isnan(float(row[name]))]
        result[name] = statistics.fmean(values) if values else 0.0
    return result


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile_input_invalid")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def performance_metrics(evaluations: Sequence[CaseEvaluationV1]) -> dict[str, object]:
    if not evaluations:
        raise ValueError("performance_evaluations_empty")
    latencies = [item.latency_seconds for item in evaluations]
    if any(not math.isfinite(item) or item < 0 for item in latencies):
        raise ValueError("performance_latency_invalid")
    if any(
        value < 0
        for item in evaluations
        for value in (item.compressed_bytes, item.decompressed_bytes, item.normalized_bytes)
    ):
        raise ValueError("performance_bytes_invalid")
    return {
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p90_latency_seconds": percentile(latencies, 0.90),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "maximum_latency_seconds": max(latencies),
        "maximum_compressed_bytes": max(item.compressed_bytes for item in evaluations),
        "maximum_decompressed_bytes": max(item.decompressed_bytes for item in evaluations),
        "maximum_normalized_bytes": max(item.normalized_bytes for item in evaluations),
        "maximum_retry_count": max(item.retry_count for item in evaluations),
        "within_frozen_bounds": all(
            item.latency_seconds <= 45.0
            and item.compressed_bytes <= 8 * 2 * 1024 * 1024
            and item.decompressed_bytes <= 8 * 2 * 1024 * 1024
            and item.normalized_bytes <= 512 * 1024
            and item.retry_count in {0, 1}
            for item in evaluations
        ),
    }


def verify_quote_locator(document: str, document_sha256: str, start: int, end: int, quote: str) -> bool:
    if _sha256_bytes(document.encode("utf-8")) != document_sha256:
        return False
    return (
        isinstance(start, int) and not isinstance(start, bool)
        and isinstance(end, int) and not isinstance(end, bool)
        and 0 <= start < end <= len(document)
        and document[start:end] == quote
    )


def classification_metrics(expected: Sequence[bool], actual: Sequence[bool]) -> dict[str, object]:
    if len(expected) != len(actual) or not expected:
        raise ValueError("classification_input_invalid")
    true_positive = sum(left and right for left, right in zip(expected, actual, strict=True))
    false_positive = sum(not left and right for left, right in zip(expected, actual, strict=True))
    false_negative = sum(left and not right for left, right in zip(expected, actual, strict=True))
    true_negative = len(expected) - true_positive - false_positive - false_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
    }


def query_privacy_result(
    *,
    approved_queries: Sequence[str],
    outgoing_queries: Sequence[str],
    canaries: Mapping[str, str],
) -> dict[str, object]:
    approved = set(approved_queries)
    if not approved or any(not isinstance(item, str) or not item for item in approved):
        raise ValueError("approved_query_set_invalid")
    leaked: list[str] = []
    for canary_id, value in canaries.items():
        _safe_id(canary_id, "canary_id")
        if not isinstance(value, str) or len(value) < 8:
            raise ValueError("canary_value_invalid")
        if any(value in query for query in outgoing_queries):
            leaked.append(canary_id)
    unapproved = sum(query not in approved for query in outgoing_queries)
    return {
        "status": "PASS" if not leaked and unapproved == 0 else "FAIL",
        "outgoing_query_count": len(outgoing_queries),
        "unapproved_query_count": unapproved,
        "leaked_canary_count": len(leaked),
        "leaked_canary_ids": sorted(leaked),
        "approved_query_set_sha256": _sha256_json(sorted(approved)),
    }


def verify_cancellation_snapshot(value: Mapping[str, object]) -> bool:
    required = {
        "tombstoned", "bundle_count", "document_count", "effect_count",
        "text_output_count", "speech_output_count",
    }
    return (
        set(value) == required
        and value.get("tombstoned") is True
        and all(value.get(field) == 0 for field in required - {"tombstoned"})
    )


def threshold_decisions(metrics: Mapping[str, object], thresholds: Mapping[str, float]) -> dict[str, bool]:
    if set(thresholds) != set(DEFAULT_THRESHOLDS):
        raise ValueError("threshold_manifest_invalid")
    overall = metrics.get("overall_macro")
    languages = metrics.get("per_language")
    if not isinstance(overall, Mapping) or not isinstance(languages, Mapping):
        raise ValueError("metric_contract_invalid")
    return {
        "overall_recall_at_5": float(overall["recall_at_5"]) >= thresholds["overall_recall_at_5"],
        "language_recall_at_5": all(float(languages[lang]["recall_at_5"]) >= thresholds["language_recall_at_5"] for lang in LANGUAGES),  # type: ignore[index]
        "overall_recall_at_10": float(overall["recall_at_10"]) >= thresholds["overall_recall_at_10"],
        "language_recall_at_10": all(float(languages[lang]["recall_at_10"]) >= thresholds["language_recall_at_10"] for lang in LANGUAGES),  # type: ignore[index]
        "overall_mrr_at_10": float(overall["mrr_at_10"]) >= thresholds["overall_mrr_at_10"],
        "language_mrr_at_10": all(float(languages[lang]["mrr_at_10"]) >= thresholds["language_mrr_at_10"] for lang in LANGUAGES),  # type: ignore[index]
        "primary_source_recall": float(overall["primary_source_recall"]) >= thresholds["primary_source_recall"],
    }


class OnlineRetrievalRunnerV1:
    """Execute a reviewed benchmark through the authenticated loopback API.

    This runner intentionally records only qrel matches, counts, timings and
    hashes.  It does not persist queries, result URLs, titles, snippets or
    documents in qualification evidence.
    """

    def __init__(
        self,
        benchmark: BenchmarkV1,
        *,
        api_base_url: str,
        bearer_token: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if benchmark.review_state != "REVIEWED":
            raise ValueError("benchmark_qrels_unreviewed")
        if not re.fullmatch(r"http://(?:127\.0\.0\.1|\[::1\]):[1-9][0-9]{0,4}/v1", api_base_url):
            raise ValueError("online_api_not_loopback")
        if not isinstance(bearer_token, str) or not 32 <= len(bearer_token.encode("utf-8")) <= 512:
            raise ValueError("online_api_token_invalid")
        self._benchmark = benchmark
        self._base = api_base_url
        self._token = bearer_token
        self._owned_client = client is None
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(50.0, connect=3.0),
        )

    async def run(self) -> tuple[tuple[CaseEvaluationV1, ...], tuple[dict[str, object], ...]]:
        evaluations: list[CaseEvaluationV1] = []
        summaries: list[dict[str, object]] = []
        try:
            for case in self._benchmark.cases:
                evaluated, summary = await self._run_case(case)
                evaluations.append(evaluated)
                summaries.append(summary)
        finally:
            if self._owned_client:
                await self._client.aclose()
        return tuple(evaluations), tuple(summaries)

    async def _run_case(self, case: BenchmarkCaseV1) -> tuple[CaseEvaluationV1, dict[str, object]]:
        body = {
            "task_id": "online-" + case.case_id,
            "conversation_id": "online-qualification",
            "epoch": 1,
            "plan": {
                "canonical_query": case.canonical_query,
                "facets": [
                    {"facet_id": facet_id, "label": label}
                    for facet_id, label in case.facets
                ],
                "subqueries": [case.canonical_query],
                "language": case.language,
                "freshness": case.freshness.casefold(),
                "timezone": case.timezone,
                "page_limit": 8,
                "deadline_seconds": 45,
                "browser_fallback_allowed": False,
                "version": "search-plan/1",
            },
            "deadline_seconds": 45,
            "max_documents": 8,
            "max_passages": 12,
            "max_claims": 6,
            "span_selection_output_tokens": 768,
        }
        raw = canonical_json(body).encode("utf-8")
        started = time.monotonic()
        status = "TECHNICAL_FAILURE"
        ranked: tuple[RankedResultV1, ...] = ()
        covered: tuple[str, ...] = ()
        normalized_bytes = 0
        retries = 0
        code = "request_failed"
        matched: list[dict[str, object]] = []
        response_hash = ""
        try:
            response = await self._client.post(
                self._base + "/research",
                headers={
                    "Authorization": "Bearer " + self._token,
                    "Content-Type": "application/json",
                    "Idempotency-Key": "online-" + case.case_id,
                },
                content=raw,
            )
            response_hash = _sha256_bytes(response.content)
            if response.status_code != 200:
                code = "api_http_" + str(response.status_code)
            else:
                corpus = self._parse_corpus(response.json())
                ranked = corpus[0]
                covered = corpus[1]
                normalized_bytes = corpus[2]
                retries = corpus[3]
                status = "SUCCESS"
                code = "ok"
                for qrel in case.qrels:
                    ranks = [rank for rank, item in enumerate(ranked, 1) if _matches(qrel, item)]
                    matched.append({"qrel_id": qrel.qrel_id, "rank": min(ranks) if ranks else None})
        except (httpx.HTTPError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            code = "api_contract_failure"
        latency = max(0.0, time.monotonic() - started)
        evaluation = CaseEvaluationV1(
            case.case_id, case.language, status, ranked, covered, latency,
            0, 0, normalized_bytes, retries,
        )
        summary = {
            "case_id": case.case_id,
            "language": case.language,
            "status": status,
            "code": code,
            "result_count": len(ranked),
            "covered_facet_count": len(covered),
            "latency_ms": int(latency * 1000),
            "normalized_bytes": normalized_bytes,
            "retry_count": retries,
            "qrel_matches": matched,
            "response_sha256": response_hash,
        }
        return evaluation, summary

    @staticmethod
    def _parse_corpus(value: object) -> tuple[tuple[RankedResultV1, ...], tuple[str, ...], int, int]:
        if not isinstance(value, dict) or set(value) != {"corpus"} or not isinstance(value["corpus"], dict):
            raise ValueError("online_corpus_schema_invalid")
        corpus = value["corpus"]
        if set(corpus) != {
            "plan_hash", "sources", "documents", "passages",
            "technical_failure_facets", "network_retry_count",
        }:
            raise ValueError("online_corpus_schema_invalid")
        sources, documents, passages = corpus["sources"], corpus["documents"], corpus["passages"]
        if not isinstance(sources, list) or len(sources) > 8 or not isinstance(documents, dict) or len(documents) != len(sources):
            raise ValueError("online_corpus_schema_invalid")
        if not isinstance(passages, list) or len(passages) > 12:
            raise ValueError("online_corpus_schema_invalid")
        ranked: list[RankedResultV1] = []
        normalized_bytes = 0
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("online_corpus_schema_invalid")
            source_id = source.get("source_id")
            document = documents.get(source_id) if isinstance(source_id, str) else None
            if not isinstance(document, dict) or set(document) != {
                "version", "text", "sha256", "character_length", "byte_length", "truncated"
            }:
                raise ValueError("online_corpus_schema_invalid")
            text = document["text"]
            document_hash = document["sha256"]
            byte_length = document["byte_length"]
            if (
                not isinstance(text, str) or not isinstance(document_hash, str)
                or _sha256_bytes(text.encode("utf-8")) != document_hash
                or byte_length != len(text.encode("utf-8"))
            ):
                raise ValueError("online_corpus_document_invalid")
            normalized_bytes += int(byte_length)
            ranked.append(RankedResultV1(
                str(source["accepted_canonical_url"]), str(source["registrable_domain"]),
                str(source_id), str(source["source_role"]),
            ))
        if normalized_bytes > 512 * 1024:
            raise ValueError("online_corpus_size_invalid")
        covered = tuple(sorted({
            str(item["facet_id"]) for item in passages
            if isinstance(item, dict) and isinstance(item.get("facet_id"), str)
        }))
        retries = corpus["network_retry_count"]
        if isinstance(retries, bool) or retries not in {0, 1}:
            raise ValueError("online_retry_count_invalid")
        return tuple(ranked), covered, normalized_bytes, retries


def rejected_evidence(
    benchmark: BenchmarkV1,
    *,
    image_status: str,
    blockers: Iterable[str],
    started_at: str,
    ended_at: str,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    blocker_list = sorted(set(blockers))
    payload: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "contract_version": ONLINE_GATE_VERSION,
        "gate_status": "REJECTED",
        "started_at": started_at,
        "ended_at": ended_at,
        "benchmark": {
            "version": benchmark.version,
            "review_state": benchmark.review_state,
            "sha256": benchmark.sha256,
            "qrels_sha256": benchmark.qrels_sha256,
            "case_count": len(benchmark.cases),
            "language_counts": {lang: sum(case.language == lang for case in benchmark.cases) for lang in sorted(LANGUAGES)},
        },
        "images": {"status": image_status},
        "thresholds": dict(sorted(thresholds.items())),
        "metrics": None,
        "integrity": {"status": "NOT_EVALUATED"},
        "freshness_conflicts": {"status": "NOT_EVALUATED"},
        "independence": {"status": "NOT_EVALUATED"},
        "performance": {"status": "NOT_EVALUATED"},
        "cancellation": {"status": "NOT_EVALUATED"},
        "privacy": {"status": "NOT_EVALUATED", "leaked_canary_count": 0},
        "cases": [],
        "decisions": {},
        "feature_state": {"research_enabled": False, "browser_qualification_sha256": ""},
        "blockers": blocker_list,
    }
    return {**payload, "evidence_sha256": _sha256_json(payload)}


def verify_evidence_digest(evidence: Mapping[str, object]) -> None:
    if set(evidence) != {
        "schema_version", "contract_version", "gate_status", "started_at", "ended_at",
        "benchmark", "images", "metrics", "integrity", "freshness_conflicts",
        "independence", "performance", "cancellation", "privacy", "feature_state",
        "thresholds", "cases", "decisions", "blockers", "evidence_sha256",
    }:
        raise ValueError("online_evidence_schema_invalid")
    digest = evidence.get("evidence_sha256")
    payload = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    if not isinstance(digest, str) or digest != _sha256_json(payload):
        raise ValueError("online_evidence_digest_invalid")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cyrus-research-online-qualify")
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8770/v1")
    parser.add_argument("--bearer-token-file", type=Path)
    args = parser.parse_args(argv)
    started = _now()
    blockers: list[str] = []
    image_status = "INVALID"
    try:
        benchmark = parse_benchmark(_load_json(args.benchmark))
        thresholds = parse_threshold_manifest(_load_json(args.thresholds))
        if benchmark.review_state != "REVIEWED":
            blockers.append("benchmark_qrels_unreviewed")
        observed_images: dict[str, object] = {}
        try:
            images = parse_image_manifest(_load_json(args.images))
            image_status, observed_images = inspect_runtime_images(images)
            if image_status != "VERIFIED":
                blockers.append("registry_images_not_runtime_verified")
        except ValueError as exc:
            blockers.append(str(exc))
        summaries: tuple[dict[str, object], ...] = ()
        metrics: dict[str, object] | None = None
        decisions: dict[str, bool] = {}
        if args.execute_live and benchmark.review_state == "REVIEWED" and image_status == "VERIFIED":
            if args.bearer_token_file is None:
                blockers.append("bearer_token_file_missing")
            else:
                try:
                    token = args.bearer_token_file.read_text(encoding="utf-8").strip()
                    evaluations, summaries = asyncio.run(OnlineRetrievalRunnerV1(
                        benchmark, api_base_url=args.api_base_url, bearer_token=token
                    ).run())
                    metrics = compute_retrieval_metrics(benchmark, evaluations)
                    decisions = threshold_decisions(metrics, thresholds)
                    if not all(decisions.values()):
                        blockers.append("retrieval_threshold_failed")
                except (OSError, ValueError, httpx.HTTPError) as exc:
                    blockers.append("live_retrieval_failed_" + type(exc).__name__)
        else:
            blockers.append("live_public_web_not_executed")
        blockers.extend((
            "quote_integrity_not_evaluated",
            "orchestrator_cancellation_not_evaluated",
            "privacy_gate_not_evaluated",
            "freshness_conflict_independence_not_evaluated",
            "bandwidth_not_observed",
        ))
        evidence = rejected_evidence(
            benchmark, image_status=image_status, blockers=blockers,
            started_at=started, ended_at=_now(), thresholds=thresholds,
        )
        evidence["images"] = {"status": image_status, "observed": observed_images}
        evidence["metrics"] = metrics
        evidence["cases"] = list(summaries)
        evidence["decisions"] = dict(sorted(decisions.items()))
        evidence["evidence_sha256"] = _sha256_json({
            key: value for key, value in evidence.items() if key != "evidence_sha256"
        })
        args.output.write_text(canonical_json(evidence) + "\n", encoding="utf-8")
        return 1
    except Exception as exc:
        fallback = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "contract_version": ONLINE_GATE_VERSION,
            "gate_status": "REJECTED",
            "error_code": type(exc).__name__,
        }
        args.output.write_text(canonical_json({**fallback, "evidence_sha256": _sha256_json(fallback)}) + "\n", encoding="utf-8")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
