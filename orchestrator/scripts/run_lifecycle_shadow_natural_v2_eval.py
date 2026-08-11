#!/usr/bin/env python3
"""Evaluate frozen deterministic gate v2 on its one-attempt natural corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from jarvis_orchestrator.anamnesis_shadow import (
    DETERMINISTIC_LIFECYCLE_GATE_V2_CONTRACT_SHA256,
    DeterministicLifecycleGateV2,
    LifecycleCandidate,
    LifecycleObservation,
)


PROTOCOL_SHA256 = "4b16ed22a8db5cf7496f14dc8e7c2e56cc83c295e5d8c19d57a676d3ec28f053"
DATASET_SHA256 = "c869d1ea3d52538dbb80673f632f585ee168aba7bebd1a1324fe0ca785c2651e"
MANIFEST_SHA256 = "92a19fb79cbecbc17da3001a08ac7eb9a121b260b414c9d6820da7eaa60e17a9"
GATE_SHA256 = "a5f522a1107fee9089a06a65c70eaadbe0d95a26894312b93259f482f0518915"
EXPECTED_KEYS = {"id", "category", "language", "text", "active", "expected"}


def _eval_root() -> Path:
    return Path(__file__).resolve().parents[1] / "eval"


def _load_cases() -> list[dict[str, object]]:
    root = _eval_root()
    protocol = root / "lifecycle_gate_v2.protocol.json"
    manifest = root / "lifecycle_shadow_natural_v2.manifest.json"
    dataset = root / "lifecycle_shadow_natural_v2.jsonl"
    for path, expected in (
        (protocol, PROTOCOL_SHA256),
        (manifest, MANIFEST_SHA256),
        (dataset, DATASET_SHA256),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen lifecycle artifact changed: {path.name}")
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if len(rows) != 64 or len({row.get("id") for row in rows}) != 64:
        raise RuntimeError("lifecycle v2 dataset identity mismatch")
    for row in rows:
        if not isinstance(row, dict) or set(row) != EXPECTED_KEYS:
            raise RuntimeError("lifecycle v2 case shape is invalid")
        if row["language"] not in {"en", "el"}:
            raise RuntimeError("lifecycle v2 language is invalid")
        if not isinstance(row["active"], list) or len(row["active"]) > 20:
            raise RuntimeError("lifecycle v2 active view is invalid")
    return rows


def _encoded_decision(decision) -> str:
    if decision is None:
        return "fallback"
    if decision.active_index is None:
        return decision.operation.value
    return f"{decision.operation.value}_{decision.active_index}"


def evaluate() -> dict[str, object]:
    if DETERMINISTIC_LIFECYCLE_GATE_V2_CONTRACT_SHA256 != GATE_SHA256:
        raise RuntimeError("lifecycle gate v2 contract changed")
    gate = DeterministicLifecycleGateV2()
    results = []
    for row in _load_cases():
        language = str(row["language"])
        active = tuple(
            LifecycleCandidate(index, str(text), language)
            for index, text in enumerate(row["active"])
        )
        actual = _encoded_decision(
            gate.decide(LifecycleObservation(str(row["text"]), language), active)
        )
        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "language": language,
                "expected": row["expected"],
                "actual": actual,
                "pass": actual == row["expected"],
            }
        )

    def score(items):
        selected = list(items)
        return {
            "passed": sum(bool(item["pass"]) for item in selected),
            "total": len(selected),
        }

    category_totals = Counter(str(result["category"]) for result in results)
    category_passed = Counter(
        str(result["category"]) for result in results if result["pass"]
    )
    language_totals = Counter(str(result["language"]) for result in results)
    language_passed = Counter(
        str(result["language"]) for result in results if result["pass"]
    )
    explicit = [
        result
        for result in results
        if result["expected"] == "create"
        or str(result["expected"]).startswith(("replace_", "cancel_"))
    ]
    ignore = [result for result in results if result["category"] == "ignore"]
    fallback = [result for result in results if result["category"] == "fallback"]
    gates = {
        "explicit_command_recall": score(explicit),
        "ignore_precision": score(ignore),
        "fallback_routing": score(fallback),
    }
    promotion_passed = (
        gates["explicit_command_recall"]["passed"]
        / gates["explicit_command_recall"]["total"]
        >= 0.9
        and gates["ignore_precision"]["passed"]
        == gates["ignore_precision"]["total"]
        and gates["fallback_routing"]["passed"]
        == gates["fallback_routing"]["total"]
    )
    return {
        "diagnostic": "jarvis.lifecycle-shadow-natural.v2",
        "run_date": "2026-08-10",
        "hypothesis_test_eligible": False,
        "authoritative_memory_enabled": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "gate_contract_sha256": GATE_SHA256,
        "model_calls": 0,
        "provider_api_cost_usd": 0.0,
        "passed": sum(bool(result["pass"]) for result in results),
        "total": len(results),
        "category_scores": {
            category: {
                "passed": category_passed[category],
                "total": category_totals[category],
            }
            for category in sorted(category_totals)
        },
        "language_scores": {
            language: {
                "passed": language_passed[language],
                "total": language_totals[language],
            }
            for language in sorted(language_totals)
        },
        "promotion_gates": gates,
        "failures": [result for result in results if not result["pass"]],
        "promotion_decision": (
            "shadow_pilot_candidate" if promotion_passed else "gate_v2_rejected"
        ),
    }


def main() -> int:
    result = evaluate()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if result["promotion_decision"] == "shadow_pilot_candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
