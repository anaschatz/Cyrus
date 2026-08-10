"""Inspect task contract for the frozen OpenMemory diagnostic v2."""

from __future__ import annotations

import pytest

import eval.anamnesis_openmemory_eval_v2 as openmemory_eval_v2


def test_v2_task_freezes_no_thinking_transport_and_artifact(monkeypatch) -> None:
    monkeypatch.setattr(openmemory_eval_v2, "_verify_git_state", lambda commit: None)
    monkeypatch.setattr(
        openmemory_eval_v2, "_require_ollama_models_dir", lambda value: value
    )
    monkeypatch.setattr(
        openmemory_eval_v2,
        "_verify_installed_w3_m2_model",
        lambda value: 6_600_000_000,
    )

    task = openmemory_eval_v2.local_openmemory_decision_diagnostic_v2(
        ollama_models_dir="/frozen/models",
        source_commit="a" * 40,
        seed=101,
    )

    assert task.version == openmemory_eval_v2.TASK_VERSION
    assert task.config.temperature == 0.0
    assert task.config.seed == 101
    assert task.config.cache is False
    assert task.config.max_retries == 0
    assert task.config.max_connections == 1
    assert task.config.adaptive_connections is False
    assert task.config.extra_body == {"reasoning_effort": "none"}
    assert task.metadata["expected_model_calls"] == 16
    assert task.metadata["call_order"] == "per_case_baseline_then_recall"
    assert task.metadata["transport_field"] == "reasoning_effort=none"
    assert task.metadata["fresh_case_version"] == "v2"
    assert task.metadata["artifact_raw_sha256"] == (
        openmemory_eval_v2.ARTIFACT_RAW_SHA256
    )
    assert task.metadata["artifact_canonical_sha256"] == (
        openmemory_eval_v2.ARTIFACT_CANONICAL_SHA256
    )


def test_v2_task_rejects_unfrozen_seed_or_missing_commit() -> None:
    with pytest.raises(ValueError, match="seed 101"):
        openmemory_eval_v2.local_openmemory_decision_diagnostic_v2(
            ollama_models_dir="/unused",
            source_commit="a" * 40,
            seed=102,
        )
    with pytest.raises(ValueError, match="source_commit"):
        openmemory_eval_v2.local_openmemory_decision_diagnostic_v2(
            ollama_models_dir="/unused",
            seed=101,
        )


def test_v2_loads_fresh_eight_case_artifact() -> None:
    artifact = openmemory_eval_v2._load_frozen_artifact()

    assert len(artifact.cases) == 8
    assert sum(len(case.hits) for case in artifact.cases) == 7
    assert all(case.id.startswith("omd2_") for case in artifact.cases)
    assert openmemory_eval_v2.LOCAL_W3_M2_OLLAMA_MODEL == ("ollama/qwen3.5:9b-q4_K_M")


def test_v1_task_contract_remains_reasoning_enabled() -> None:
    import eval.anamnesis_openmemory_eval as v1

    assert v1.TASK_VERSION == "openmemory-decision-diagnostic.local.v0.1"
    assert v1.ARTIFACT_PATH.name == "decision_diagnostic.v1.json"
