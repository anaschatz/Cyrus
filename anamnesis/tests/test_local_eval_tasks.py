from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
from inspect_ai.model import get_model_info

from anamnesis.local_experiment import LOCAL_PRICING_SHA256, LocalExperimentManifest
from anamnesis.local_runtime import LOCAL_OLLAMA_MODEL, LOCAL_ZERO_MODEL_COST
from anamnesis.oracle import ORACLE_COMPILER_VERSION, ORACLE_SYSTEM_NAME


def test_local_task_registry_names_are_isolated_and_frozen() -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")
    expected = {
        "local_model_preflight",
        "local_model_preflight_w2",
        "local_no_memory",
        "local_full_context",
        "local_vector_rag",
        "local_anamnesis",
        "local_anamnesis_writer_diagnostic",
        "local_anamnesis_writer_diagnostic_w2",
        "local_anamnesis_oracle_compiler",
    }

    assert {
        module[name].__registry_info__.name  # type: ignore[attr-defined]
        for name in expected
    } == expected
    task = module["local_model_preflight"]()
    model_info = get_model_info(LOCAL_OLLAMA_MODEL)
    assert model_info is not None
    assert model_info.cost == LOCAL_ZERO_MODEL_COST
    assert task.version == "local.0.1"
    assert task.config.temperature == 0.0
    assert task.config.seed == 101
    assert task.config.cache is False
    assert task.config.max_retries == 0
    assert task.config.max_connections == 1
    assert task.config.adaptive_connections is False
    assert task.metadata["hypothesis_test_eligible"] is False
    assert task.metadata["pricing_config_sha256"] == LOCAL_PRICING_SHA256
    w2_task = module["local_model_preflight_w2"]()
    assert w2_task.version == "local.w2.0.1"
    assert w2_task.config.seed == 101
    assert w2_task.metadata["preflight_fixture_sha256"] == (
        "3b82128bab1d801d073118488aa4f0a0a662603b98325f5c9d7dad497f026057"
    )
    with pytest.raises(ValueError, match="seed 101 exactly"):
        module["local_model_preflight_w2"](seed=202)


def test_local_scenario_tasks_fail_closed_without_frozen_manifest() -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")

    for name in (
        "local_no_memory",
        "local_full_context",
        "local_vector_rag",
        "local_anamnesis",
        "local_anamnesis_writer_diagnostic",
        "local_anamnesis_writer_diagnostic_w2",
        "local_anamnesis_oracle_compiler",
    ):
        with pytest.raises(ValueError, match="frozen local manifest"):
            module[name]()


def test_local_scenario_task_metadata_binds_ordered_smoke_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")
    raw = json.loads(
        Path("eval/local_experiment_manifest.template.json").read_text(encoding="utf-8")
    )
    raw["system_config_sha256"] = {
        "no_memory": "a" * 64,
        "full_context": "b" * 64,
        "vector_rag": "c" * 64,
        "anamnesis": "d" * 64,
    }
    manifest = LocalExperimentManifest.model_validate(raw)
    dataset_path = Path("eval/scenarios/smoke.jsonl").resolve()
    system_task = module["_system_task"]
    monkeypatch.setitem(
        system_task.__globals__,  # type: ignore[attr-defined]
        "_validated_local_manifest",
        lambda *args, **kwargs: (manifest, "e" * 64, dataset_path),
    )

    task = system_task(
        "no_memory",
        seed=101,
        repetition=1,
        manifest="ignored-by-test.json",
        ollama_models_dir="/ignored/by/test",
    )
    expected_ids = [
        json.loads(line)["id"]
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert task.metadata["dataset_scenario_count"] == 10
    assert task.metadata["dataset_sample_ids"] == expected_ids
    assert task.metadata["dataset_split"] == "smoke"
    assert task.metadata["claim_scope"] == "diagnostic_development_only"
    assert task.metadata["hypothesis_test_eligible"] is False
    assert task.metadata["provider_api_cost_usd"] == 0.0
    assert task.metadata["live_semantic_preflight_required"] is True


def test_local_writer_task_is_phase_isolated_and_reference_blind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")
    raw = json.loads(
        Path("eval/local_writer_experiment_manifest.template.json").read_text(
            encoding="utf-8"
        )
    )
    raw["system_config_sha256"] = {"anamnesis": "d" * 64}
    manifest = LocalExperimentManifest.model_validate(raw)
    dataset_path = Path("eval/scenarios/writer_diagnostic.v1.jsonl").resolve()
    system_task = module["_system_task"]

    def validated_manifest(*args: object, **kwargs: object):
        required_phase = kwargs.get("required_phase")
        if manifest.phase != required_phase:
            raise ValueError(f"local task requires phase {required_phase}")
        return manifest, "e" * 64, dataset_path

    monkeypatch.setitem(
        system_task.__globals__,  # type: ignore[attr-defined]
        "_validated_local_manifest",
        validated_manifest,
    )

    task = module["local_anamnesis_writer_diagnostic"](
        seed=101,
        repetition=1,
        manifest="ignored-by-test.json",
        ollama_models_dir="/ignored/by/test",
    )

    assert task.version == "local.0.1"
    assert task.metadata["system"] == "anamnesis"
    assert task.metadata["dataset"] == "eval/scenarios/writer_diagnostic.v1.jsonl"
    assert task.metadata["dataset_split"] == "writer_diagnostic"
    assert task.metadata["dataset_scenario_count"] == 10
    assert task.metadata["repetition"] == 1
    assert task.metadata["hypothesis_test_eligible"] is False
    assert "writer_reference" not in task.metadata
    assert "writer_reference_path" not in task.metadata
    assert "writer_reference_sha256" not in task.metadata

    reference = manifest.writer_reference
    assert reference is not None
    metadata_json = json.dumps(task.metadata, sort_keys=True)
    assert reference.path not in metadata_json
    assert reference.sha256 not in metadata_json
    solver_params = repr(getattr(task.solver, "__registry_params__", {}))
    assert reference.path not in solver_params
    assert reference.sha256 not in solver_params

    with pytest.raises(ValueError, match="requires phase smoke"):
        module["local_anamnesis"](
            seed=101,
            repetition=1,
            manifest="ignored-by-test.json",
            ollama_models_dir="/ignored/by/test",
        )


def test_local_writer_task_rejects_smoke_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")
    raw = json.loads(
        Path("eval/local_experiment_manifest.template.json").read_text(encoding="utf-8")
    )
    raw["system_config_sha256"] = {
        "no_memory": "a" * 64,
        "full_context": "b" * 64,
        "vector_rag": "c" * 64,
        "anamnesis": "d" * 64,
    }
    manifest = LocalExperimentManifest.model_validate(raw)
    dataset_path = Path("eval/scenarios/smoke.jsonl").resolve()
    system_task = module["_system_task"]

    def validated_manifest(*args: object, **kwargs: object):
        required_phase = kwargs.get("required_phase")
        if manifest.phase != required_phase:
            raise ValueError(f"local task requires phase {required_phase}")
        return manifest, "e" * 64, dataset_path

    monkeypatch.setitem(
        system_task.__globals__,  # type: ignore[attr-defined]
        "_validated_local_manifest",
        validated_manifest,
    )

    with pytest.raises(ValueError, match="requires phase writer_diagnostic"):
        module["local_anamnesis_writer_diagnostic"](
            seed=101,
            repetition=1,
            manifest="ignored-by-test.json",
            ollama_models_dir="/ignored/by/test",
        )


def test_local_writer_w2_task_binds_v3_and_stays_reference_blind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")
    raw = json.loads(
        Path("eval/local_writer_w2_experiment_manifest.template.json").read_text(
            encoding="utf-8"
        )
    )
    raw["system_config_sha256"] = {"anamnesis": "d" * 64}
    manifest = LocalExperimentManifest.model_validate(raw)
    dataset_path = Path("eval/scenarios/writer_diagnostic.v3.jsonl").resolve()
    system_task = module["_system_task"]
    monkeypatch.setitem(
        system_task.__globals__,  # type: ignore[attr-defined]
        "_validated_local_manifest",
        lambda *args, **kwargs: (manifest, "e" * 64, dataset_path),
    )

    task = module["local_anamnesis_writer_diagnostic_w2"](
        seed=101,
        repetition=1,
        manifest="ignored-by-test.json",
        ollama_models_dir="/ignored/by/test",
    )

    assert task.metadata["dataset"] == "eval/scenarios/writer_diagnostic.v3.jsonl"
    assert task.metadata["dataset_split"] == "writer_diagnostic_w2"
    assert task.metadata["dataset_scenario_count"] == 10
    assert task.metadata["memory_compiler_prompt_version"] == "local.v0.3"
    assert task.metadata["setup_preflight_model_calls"] == 4
    assert task.metadata["scenario_compiler_model_calls"] == 46
    assert task.metadata["setup_preflight_usage_in_headline"] is False
    metadata_json = json.dumps(task.metadata, sort_keys=True)
    assert manifest.writer_reference is not None
    assert manifest.writer_reference.path not in metadata_json
    assert manifest.writer_reference.sha256 not in metadata_json
    solver_params = json.dumps(task.solver.__registry_params__, sort_keys=True)  # type: ignore[attr-defined]
    assert manifest.writer_reference.path not in solver_params
    assert manifest.writer_reference.sha256 not in solver_params


def test_local_phase_mismatch_fails_before_runtime_or_artifact_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")
    raw = json.loads(
        Path("eval/local_writer_experiment_manifest.template.json").read_text(
            encoding="utf-8"
        )
    )
    raw.update(
        status="frozen",
        git_commit="a" * 40,
        decision_prompt_sha256="b" * 64,
        decision_schema_sha256="c" * 64,
        memory_compiler_prompt_sha256="d" * 64,
        memory_compiler_schema_sha256="e" * 64,
        system_config_sha256={"anamnesis": "f" * 64},
    )
    model = raw["model"]
    assert isinstance(model, dict)
    preflight = model["preflight"]
    assert isinstance(preflight, dict)
    preflight["sha256"] = "1" * 64
    path = tmp_path / "writer-manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    validator = module["_validated_local_manifest"]

    def forbidden_io(*args: object, **kwargs: object) -> None:
        raise AssertionError("phase mismatch reached runtime or artifact I/O")

    for name in (
        "require_local_only_environment",
        "verify_static_local_inputs",
        "_require_ollama_models_dir",
        "_verify_installed_model",
        "_repo_artifact_path",
        "validate_local_preflight_artifact",
        "_verify_git_state",
        "load_scenarios",
    ):
        monkeypatch.setitem(
            validator.__globals__,  # type: ignore[attr-defined]
            name,
            forbidden_io,
        )

    with pytest.raises(ValueError, match="requires phase smoke"):
        validator(
            str(path),
            system="anamnesis",
            seed=101,
            repetition=1,
            ollama_models_dir="/must/not/be/touched",
            required_phase="smoke",
        )


def test_local_oracle_task_binds_annotations_and_diagnostic_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")
    raw = json.loads(
        Path("eval/local_experiment_manifest.template.json").read_text(encoding="utf-8")
    )
    raw["phase"] = "oracle_smoke"
    raw["compiler_mode"] = "oracle"
    raw["systems"] = [ORACLE_SYSTEM_NAME]
    raw["oracle_annotations"] = {
        "path": "eval/oracle/smoke_memory_deltas.v1.json",
        "sha256": "a" * 64,
    }
    raw["system_config_sha256"] = {ORACLE_SYSTEM_NAME: "b" * 64}
    model = raw["model"]
    assert isinstance(model, dict)
    model["same_model_for_compiler_and_decision"] = False
    manifest = LocalExperimentManifest.model_validate(raw)
    dataset_path = Path("eval/scenarios/smoke.jsonl").resolve()
    system_task = module["_system_task"]
    monkeypatch.setitem(
        system_task.__globals__,  # type: ignore[attr-defined]
        "_validated_local_manifest",
        lambda *args, **kwargs: (manifest, "c" * 64, dataset_path),
    )
    monkeypatch.setitem(
        system_task.__globals__,  # type: ignore[attr-defined]
        "_require_oracle_annotations_path",
        lambda *args, **kwargs: dataset_path,
    )
    monkeypatch.setitem(
        system_task.__globals__,  # type: ignore[attr-defined]
        "load_oracle_artifact",
        lambda *args, **kwargs: object(),
    )

    task = system_task(
        ORACLE_SYSTEM_NAME,
        seed=101,
        repetition=1,
        manifest="ignored-by-test.json",
        ollama_models_dir="/ignored/by/test",
        oracle_annotations_path="eval/oracle/smoke_memory_deltas.v1.json",
    )

    assert task.metadata["system"] == ORACLE_SYSTEM_NAME
    assert task.metadata["dataset_split"] == "oracle_smoke"
    assert task.metadata["compiler_mode"] == "oracle"
    assert task.metadata["gold_assisted"] is True
    assert task.metadata["human_annotation_measured"] is False
    assert task.metadata["setup_preflight_compiler_used_in_scenarios"] is False
    assert task.metadata["oracle_compiler_version"] == ORACLE_COMPILER_VERSION
    assert task.metadata["oracle_annotations_path"] == (
        "eval/oracle/smoke_memory_deltas.v1.json"
    )
    assert task.metadata["oracle_annotations_sha256"] == "a" * 64
    assert task.metadata["oracle_token_scope"] == "decision_only_lower_bound"
    assert task.metadata["same_model_for_compiler_and_decision"] is False
    assert task.metadata["scenario_compiler_model_calls"] == 0
    assert task.metadata["setup_preflight_includes_llm_compiler_call"] is True
    assert task.metadata["hypothesis_test_eligible"] is False
