#!/usr/bin/env python3
"""Evaluate the frozen deterministic gate on natural lifecycle phrasing."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from jarvis_orchestrator.anamnesis_shadow import (
    DETERMINISTIC_LIFECYCLE_GATE_CONTRACT_SHA256,
    DeterministicLifecycleGate,
    LifecycleCandidate,
    LifecycleObservation,
)


DATASET_SHA256 = "8e652f4dab5da5d5fa7f35eb8a7996f8bc32e72ecc5e471b0456b9e27ce760c9"
MANIFEST_SHA256 = "7e0445bd3ea6c566d3ce9e74799d6e1d53a48ba6f3f1ca8bae9000ef40153ea3"
GATE_SHA256 = "51ff7b97ac3f8a04fcb584b88eb717f5de827e9dc4dde44a38564f505bda311c"
EXPECTED_KEYS = {"id", "category", "language", "text", "active", "expected"}
EXPECTED_VALUES = {"ignore", "create", "fallback"} | {
    f"{operation}_{index}"
    for operation in ("replace", "cancel")
    for index in range(20)
}


def _dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "eval/lifecycle_shadow_natural_v1.jsonl"


def _load_cases() -> list[dict[str, object]]:
    path = _dataset_path()
    manifest_path = path.with_suffix(".manifest.json")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError("lifecycle shadow manifest hash mismatch")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATASET_SHA256:
        raise RuntimeError("lifecycle shadow dataset hash mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(rows) != 48 or len({row.get("id") for row in rows}) != 48:
        raise RuntimeError("lifecycle shadow dataset identity mismatch")
    for row in rows:
        if not isinstance(row, dict) or set(row) != EXPECTED_KEYS:
            raise RuntimeError("lifecycle shadow case shape is invalid")
        if row["language"] not in {"en", "el"}:
            raise RuntimeError("lifecycle shadow case language is invalid")
        if row["expected"] not in EXPECTED_VALUES:
            raise RuntimeError("lifecycle shadow expected value is invalid")
        if not isinstance(row["active"], list) or len(row["active"]) > 20:
            raise RuntimeError("lifecycle shadow active view is invalid")
    return rows


def _encoded_decision(decision) -> str:
    if decision is None:
        return "fallback"
    if decision.active_index is None:
        return decision.operation.value
    return f"{decision.operation.value}_{decision.active_index}"


def evaluate() -> dict[str, object]:
    if DETERMINISTIC_LIFECYCLE_GATE_CONTRACT_SHA256 != GATE_SHA256:
        raise RuntimeError("lifecycle gate contract changed")
    gate = DeterministicLifecycleGate()
    results = []
    for row in _load_cases():
        language = str(row["language"])
        active = tuple(
            LifecycleCandidate(index, str(text), language)
            for index, text in enumerate(row["active"])
        )
        actual = _encoded_decision(
            gate.decide(
                LifecycleObservation(str(row["text"]), language),
                active,
            )
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
    category_totals = Counter(str(result["category"]) for result in results)
    category_passed = Counter(
        str(result["category"]) for result in results if result["pass"]
    )
    language_totals = Counter(str(result["language"]) for result in results)
    language_passed = Counter(
        str(result["language"]) for result in results if result["pass"]
    )
    passed = sum(bool(result["pass"]) for result in results)
    return {
        "diagnostic": "jarvis.lifecycle-shadow-natural.v1",
        "run_date": "2026-08-10",
        "hypothesis_test_eligible": False,
        "authoritative_memory_enabled": False,
        "dataset_sha256": DATASET_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "gate_contract_sha256": GATE_SHA256,
        "model_calls": 0,
        "provider_api_cost_usd": 0.0,
        "passed": passed,
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
        "failures": [result for result in results if not result["pass"]],
        "promotion_decision": "gate_v1_rejected_for_pilot_due_to_command_recall",
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
    return 0 if result["passed"] == result["total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
