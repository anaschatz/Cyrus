#!/usr/bin/env python3
"""Evaluate frozen gate v3 and its shadow-pilot promotion gates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from jarvis_orchestrator.anamnesis_shadow import (
    DETERMINISTIC_LIFECYCLE_GATE_V3_CONTRACT_SHA256,
    DeterministicLifecycleGateV3,
    LifecycleCandidate,
    LifecycleObservation,
)


PROTOCOL_SHA256 = "a63936f8d91332b9d887ed4a9a4df0814e19a0865aee7eeeba72fc722f058ece"
DATASET_SHA256 = "6bd4ec1643666335e3f5cbcaaf4cb44af6ab04006c26520274bc236db39812be"
MANIFEST_SHA256 = "2fb02eea130eaf76eb26e3b60931464e45033aa7bce0fe6942bfc1c4b7d77e99"
GATE_SHA256 = "18d80ac7e750625cb22990979d64d868b799d4fce59eda6e09c72fa0f8644a33"
EXPECTED_KEYS = {"id", "category", "language", "text", "active", "expected"}


def _load_cases() -> list[dict[str, object]]:
    root = Path(__file__).resolve().parents[1] / "eval"
    paths = (
        (root / "lifecycle_gate_v3.protocol.json", PROTOCOL_SHA256),
        (root / "lifecycle_shadow_natural_v3.manifest.json", MANIFEST_SHA256),
        (root / "lifecycle_shadow_natural_v3.jsonl", DATASET_SHA256),
    )
    for path, expected in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen lifecycle artifact changed: {path.name}")
    rows = [json.loads(line) for line in paths[-1][0].read_text().splitlines()]
    if len(rows) != 48 or len({row.get("id") for row in rows}) != 48:
        raise RuntimeError("lifecycle v3 dataset identity mismatch")
    for row in rows:
        if not isinstance(row, dict) or set(row) != EXPECTED_KEYS:
            raise RuntimeError("lifecycle v3 case shape is invalid")
        if row["language"] not in {"en", "el"}:
            raise RuntimeError("lifecycle v3 language is invalid")
        if not isinstance(row["active"], list) or len(row["active"]) > 20:
            raise RuntimeError("lifecycle v3 active view is invalid")
    return rows


def _encoded_decision(decision) -> str:
    if decision is None:
        return "fallback"
    if decision.active_index is None:
        return decision.operation.value
    return f"{decision.operation.value}_{decision.active_index}"


def evaluate() -> dict[str, object]:
    if DETERMINISTIC_LIFECYCLE_GATE_V3_CONTRACT_SHA256 != GATE_SHA256:
        raise RuntimeError("lifecycle gate v3 contract changed")
    gate = DeterministicLifecycleGateV3()
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

    def score(selected):
        items = list(selected)
        return {
            "passed": sum(bool(item["pass"]) for item in items),
            "total": len(items),
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
    ignored = [result for result in results if result["expected"] == "ignore"]
    fallback = [result for result in results if result["expected"] == "fallback"]
    gates = {
        "explicit_command_recall": score(explicit),
        "ignore_precision": score(ignored),
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
        "diagnostic": "jarvis.lifecycle-shadow-natural.v3",
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
            "shadow_pilot_candidate" if promotion_passed else "gate_v3_rejected"
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
