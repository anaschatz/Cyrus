"""Human-review helper for the frozen bilingual retrieval benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jarvis_orchestrator.domain import canonical_json

from .online_qualification import _load_json, parse_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis-research-benchmark")
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--review-jsonl", type=Path)
    args = parser.parse_args(argv)
    benchmark = parse_benchmark(_load_json(args.benchmark))
    summary = {
        "version": benchmark.version,
        "review_state": benchmark.review_state,
        "case_count": len(benchmark.cases),
        "language_counts": {
            language: sum(case.language == language for case in benchmark.cases)
            for language in ("el", "en")
        },
        "benchmark_sha256": benchmark.sha256,
        "qrels_sha256": benchmark.qrels_sha256,
    }
    if args.review_jsonl is not None:
        rows = []
        for case in benchmark.cases:
            rows.append(canonical_json({
                "case_id": case.case_id,
                "language": case.language,
                "canonical_query": case.canonical_query,
                "facets": [
                    {"facet_id": facet_id, "label": label}
                    for facet_id, label in case.facets
                ],
                "qrels": [
                    {
                        "qrel_id": qrel.qrel_id,
                        "target_kind": qrel.target_kind,
                        "target": qrel.target,
                        "relevance": qrel.relevance,
                        "primary": qrel.primary,
                        "facet_ids": list(qrel.facet_ids),
                    }
                    for qrel in case.qrels
                ],
                "review_state": case.review_state,
                "review_provenance": case.review_provenance,
            }))
        args.review_jsonl.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(canonical_json(summary))
    return 0 if benchmark.review_state == "REVIEWED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
