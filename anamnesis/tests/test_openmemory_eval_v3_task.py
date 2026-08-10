"""Immediate-action prompt and Inspect task contract for OpenMemory v3."""

from __future__ import annotations

import hashlib

import pytest

import eval.anamnesis_openmemory_eval_v3 as openmemory_eval_v3
from anamnesis.openmemory_diagnostic import (
    build_openmemory_case_prompts,
    build_openmemory_immediate_decision_prompt,
    openmemory_immediate_decision_contract,
)


def test_v3_contract_is_exact_and_scoped_to_immediate_actions() -> None:
    contract = openmemory_immediate_decision_contract()

    assert hashlib.sha256(contract.encode()).hexdigest() == (
        openmemory_eval_v3.DECISION_CONTRACT_SHA256
    )
    assert "This is not the temporal-reminder firing path" in contract
    assert "explicitly instructs the assistant to act now" in contract
    assert "may only resolve a missing optional argument" in contract
    assert "cannot create, cancel, suppress, schedule, or prove an action" in contract
    assert "Set evidence_event_ids to exactly the current event ID" in contract
    assert "omd3_" not in contract
    assert "Northstar" not in contract


def test_v3_paired_prompts_change_only_recall_branch() -> None:
    artifact = openmemory_eval_v3._load_frozen_artifact()

    for case in artifact.cases:
        baseline, recall = build_openmemory_case_prompts(
            case,
            prompt_builder=build_openmemory_immediate_decision_prompt,
        )
        assert case.event.text in baseline and case.event.text in recall
        assert "(not provided in this arm)" in baseline
        assert "(not provided in this arm)" not in recall
        for hit in case.hits:
            assert hit.content not in baseline
            assert hit.content in recall
            assert hit.fixture_id not in recall
        before_baseline, after_baseline = baseline.split("Retrospective recall JSON:")
        before_recall, after_recall = recall.split("Retrospective recall JSON:")
        assert before_baseline == before_recall
        assert after_baseline.split("\n\n", 1)[1] == after_recall.split("\n\n", 1)[1]


def test_v3_task_freezes_transport_contract_and_artifact(monkeypatch) -> None:
    monkeypatch.setattr(openmemory_eval_v3, "_verify_git_state", lambda commit: None)
    monkeypatch.setattr(
        openmemory_eval_v3, "_require_ollama_models_dir", lambda value: value
    )
    monkeypatch.setattr(
        openmemory_eval_v3,
        "_verify_installed_w3_m2_model",
        lambda value: 6_600_000_000,
    )

    task = openmemory_eval_v3.local_openmemory_immediate_action_diagnostic_v3(
        ollama_models_dir="/frozen/models",
        source_commit="a" * 40,
        seed=101,
    )

    assert task.version == openmemory_eval_v3.TASK_VERSION
    assert task.config.temperature == 0.0
    assert task.config.seed == 101
    assert task.config.cache is False
    assert task.config.max_retries == 0
    assert task.config.max_connections == 1
    assert task.config.adaptive_connections is False
    assert task.config.extra_body == {"reasoning_effort": "none"}
    assert task.metadata["expected_model_calls"] == 16
    assert task.metadata["decision_contract_sha256"] == (
        openmemory_eval_v3.DECISION_CONTRACT_SHA256
    )
    assert task.metadata["fresh_case_version"] == "v3"
    assert task.metadata["intervention"] == (
        "dedicated_immediate_action_decision_contract"
    )


def test_v3_task_rejects_unfrozen_seed_or_missing_commit() -> None:
    with pytest.raises(ValueError, match="seed 101"):
        openmemory_eval_v3.local_openmemory_immediate_action_diagnostic_v3(
            ollama_models_dir="/unused", source_commit="a" * 40, seed=102
        )
    with pytest.raises(ValueError, match="source_commit"):
        openmemory_eval_v3.local_openmemory_immediate_action_diagnostic_v3(
            ollama_models_dir="/unused", seed=101
        )


def test_v1_v2_task_identities_remain_unchanged() -> None:
    import eval.anamnesis_openmemory_eval as v1
    import eval.anamnesis_openmemory_eval_v2 as v2

    assert v1.TASK_VERSION == "openmemory-decision-diagnostic.local.v0.1"
    assert v2.TASK_VERSION == "openmemory-decision-diagnostic.local.v0.2"
    assert v2.TRANSPORT_FIELD == "reasoning_effort=none"
