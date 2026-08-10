"""Inspect task contract for the frozen paired OpenMemory diagnostic."""

from __future__ import annotations

import pytest

import eval.anamnesis_openmemory_eval as openmemory_eval


def test_openmemory_task_freezes_model_calls_config_and_artifact(monkeypatch) -> None:
    monkeypatch.setattr(openmemory_eval, "_verify_git_state", lambda commit: None)
    monkeypatch.setattr(
        openmemory_eval,
        "_require_ollama_models_dir",
        lambda value: value,
    )
    monkeypatch.setattr(
        openmemory_eval,
        "_verify_installed_w3_m2_model",
        lambda value: 6_600_000_000,
    )

    task = openmemory_eval.local_openmemory_decision_diagnostic(
        ollama_models_dir="/frozen/models",
        source_commit="a" * 40,
        seed=101,
    )

    assert task.version == openmemory_eval.TASK_VERSION
    assert task.config.temperature == 0.0
    assert task.config.seed == 101
    assert task.config.cache is False
    assert task.config.max_retries == 0
    assert task.config.max_connections == 1
    assert task.config.adaptive_connections is False
    assert task.metadata["expected_model_calls"] == 16
    assert task.metadata["call_order"] == "per_case_baseline_then_recall"
    assert task.metadata["openmemory_online_writes"] is False
    assert task.metadata["openmemory_usage_complete"] is False
    assert task.metadata["artifact_raw_sha256"] == (openmemory_eval.ARTIFACT_RAW_SHA256)
    assert task.metadata["artifact_canonical_sha256"] == (
        openmemory_eval.ARTIFACT_CANONICAL_SHA256
    )


def test_openmemory_task_rejects_unfrozen_seed_or_missing_commit() -> None:
    with pytest.raises(ValueError, match="seed 101"):
        openmemory_eval.local_openmemory_decision_diagnostic(
            ollama_models_dir="/unused",
            source_commit="a" * 40,
            seed=102,
        )
    with pytest.raises(ValueError, match="source_commit"):
        openmemory_eval.local_openmemory_decision_diagnostic(
            ollama_models_dir="/unused",
            seed=101,
        )


def test_openmemory_task_loads_exact_eight_case_artifact() -> None:
    artifact = openmemory_eval._load_frozen_artifact()

    assert len(artifact.cases) == 8
    assert sum(len(case.hits) for case in artifact.cases) == 7
    assert openmemory_eval.LOCAL_W3_M2_OLLAMA_MODEL == ("ollama/qwen3.5:9b-q4_K_M")
