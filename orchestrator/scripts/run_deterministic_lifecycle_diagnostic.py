#!/usr/bin/env python3
"""Run the frozen bilingual lifecycle gate diagnostic without a model."""

from __future__ import annotations

import json

from jarvis_orchestrator.anamnesis_shadow import (
    DETERMINISTIC_LIFECYCLE_GATE_CONTRACT_SHA256,
    DeterministicLifecycleGate,
    LifecycleCandidate,
    LifecycleObservation,
)


CASES = (
    ("en_ignore", "What time is it?", "en", (), "ignore"),
    ("en_create", "Remember that I prefer tea.", "en", (), "create"),
    (
        "en_replace",
        "Correction: I prefer coffee instead.",
        "en",
        ("Remember that I prefer tea.",),
        "replace_0",
    ),
    (
        "en_cancel",
        "Forget my drink preference.",
        "en",
        ("Remember that I prefer coffee.",),
        "cancel_0",
    ),
    ("el_ignore", "Τι ώρα είναι;", "el", (), "ignore"),
    (
        "el_create",
        "Να θυμάσαι ότι προτιμώ τσάι.",
        "el",
        (),
        "create",
    ),
    (
        "el_replace",
        "Διόρθωση: προτιμώ καφέ.",
        "el",
        ("Να θυμάσαι ότι προτιμώ τσάι.",),
        "replace_0",
    ),
    (
        "el_cancel",
        "Ξέχασε την προτίμησή μου για ρόφημα.",
        "el",
        ("Να θυμάσαι ότι προτιμώ καφέ.",),
        "cancel_0",
    ),
    (
        "en_quoted_command_is_data",
        'Translate the phrase "remember that I prefer tea".',
        "en",
        ("Remember that I prefer coffee.",),
        "ignore",
    ),
    (
        "en_negated_forget_is_not_cancel",
        "I did not ask you to forget anything.",
        "en",
        ("Remember that I prefer coffee.",),
        "ignore",
    ),
    (
        "el_unaccented_create",
        "Να θυμασαι οτι προτιμω νερο.",
        "el",
        (),
        "create",
    ),
    (
        "el_unaccented_cancel",
        "Ξεχασε την προτιμηση μου.",
        "el",
        ("Να θυμασαι οτι προτιμω νερο.",),
        "cancel_0",
    ),
    (
        "en_replace_without_target",
        "Correction: I prefer coffee.",
        "en",
        (),
        "ignore",
    ),
    (
        "en_replace_with_multiple_targets",
        "Correction: I prefer coffee.",
        "en",
        ("Remember that I prefer tea.", "Remember that my timezone is UTC."),
        "fallback",
    ),
    (
        "el_cancel_with_multiple_targets",
        "Ξέχασε την προτίμησή μου.",
        "el",
        (
            "Να θυμάσαι ότι προτιμώ τσάι.",
            "Να θυμάσαι ότι μένω Αθήνα.",
        ),
        "fallback",
    ),
    (
        "en_incomplete_command",
        "Remember that ...",
        "en",
        (),
        "fallback",
    ),
)


def _encoded_decision(decision) -> str:
    if decision is None:
        return "fallback"
    if decision.active_index is None:
        return decision.operation.value
    return f"{decision.operation.value}_{decision.active_index}"


def main() -> int:
    gate = DeterministicLifecycleGate()
    results = []
    for case_id, text, language, active_texts, expected in CASES:
        active = tuple(
            LifecycleCandidate(index, active_text, language)
            for index, active_text in enumerate(active_texts)
        )
        actual = _encoded_decision(
            gate.decide(LifecycleObservation(text, language), active)
        )
        results.append(
            {
                "case": case_id,
                "expected": expected,
                "actual": actual,
                "pass": actual == expected,
            }
        )
    passed = sum(result["pass"] for result in results)
    output = {
        "diagnostic": "jarvis.deterministic-lifecycle-gate.v1",
        "run_date": "2026-08-10",
        "hypothesis_test_eligible": False,
        "authoritative_memory_enabled": False,
        "shadow_memory_enabled": False,
        "contract_sha256": DETERMINISTIC_LIFECYCLE_GATE_CONTRACT_SHA256,
        "model_calls": 0,
        "provider_api_cost_usd": 0.0,
        "passed": passed,
        "total": len(results),
        "results": results,
        "promotion_decision": "advance_to_larger_shadow_evaluation_not_production",
    }
    print(
        json.dumps(
            output,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
