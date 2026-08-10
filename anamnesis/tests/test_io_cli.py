from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.model import ModelUsage

from anamnesis.cli import (
    InspectLogMetadata,
    InspectLogPolicy,
    _load_strict_eval_runs,
    _validate_checkpoint_timeline,
    _validate_hosted_warmup_attestation,
    _validate_inspect_log_policy,
    _validate_manifest_byte_binding,
    _validate_system_log_metadata,
    _verify_file_artifact,
    _verify_git_state,
    report_main,
    validate_main,
)
from anamnesis.experiment import ArtifactPin, ExperimentManifest
from anamnesis.inspect_adapter import (
    hosted_warmup_prompt_sha256,
    hosted_warmup_schema_sha256,
)
from anamnesis.io import (
    canonical_sha256,
    dataset_sha256,
    load_scenarios,
    require_preregistered_final_dataset,
    write_runs,
)
from anamnesis.schema import (
    CheckpointAudit,
    HostedWarmupAttestation,
    ScenarioRun,
    Usage,
)

TEMPLATE = Path("eval/experiment_manifest.template.json")


def valid_inspect_metadata() -> InspectLogMetadata:
    scenarios = load_scenarios("eval/scenarios/dev.jsonl")
    return InspectLogMetadata(
        status="success",
        invalidated=False,
        config_update_count=0,
        log_update_count=0,
        task="no_memory",
        task_file="eval/anamnesis_eval.py",
        task_args={
            "seed": 101,
            "repetition": 1,
            "dataset": "development",
            "manifest": str(TEMPLATE),
        },
        task_metadata={
            "manifest_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
            "hosted_warmup_prompt_sha256": hosted_warmup_prompt_sha256(),
            "hosted_warmup_schema_sha256": hosted_warmup_schema_sha256(),
        },
        model="provider/frozen-snapshot",
        model_base_url=None,
        model_args={"responses_api": False},
        generation_config={
            "max_retries": 0,
            "max_connections": 1,
            "adaptive_connections": False,
            "temperature": 0.0,
            "seed": 101,
            "cache": False,
        },
        dataset_name="anamnesis-development-v0",
        dataset_location="eval/scenarios/dev.jsonl",
        dataset_samples=35,
        dataset_sample_ids=tuple(scenario.id for scenario in scenarios),
        dataset_shuffled=False,
        temperature=0.0,
        seed=101,
        response_cache=False,
        max_retries=0,
        max_connections=1,
        adaptive_connections=False,
        max_samples=1,
        max_tasks=1,
        log_model_api=True,
        epochs=1,
        revision_commit="b" * 7,
        revision_dirty=False,
    )


def test_validate_prints_dataset_summary(capsys) -> None:
    assert validate_main(["eval/scenarios/smoke.jsonl"]) == 0
    output = capsys.readouterr().out
    assert '"scenarios": 10' in output
    assert '"events": 78' in output
    assert '"expected_actions": 8' in output


def test_report_writes_recomputable_csv_and_markdown(tmp_path: Path) -> None:
    scenario = load_scenarios("eval/scenarios/smoke.jsonl")[0]
    run = ScenarioRun(
        scenario_id=scenario.id,
        system="no_memory",
        repetition=1,
        model="fake/model",
        prompt_version="v0.2",
        scenario_sha256=canonical_sha256(scenario),
        prompt_sha256="0" * 64,
        system_config_sha256="1" * 64,
        usage=Usage(
            input_tokens=100,
            uncached_input_tokens=100,
            output_tokens=10,
        ),
        decision_usage=Usage(
            input_tokens=100,
            uncached_input_tokens=100,
            output_tokens=10,
        ),
        checkpoint_latency_ms=[1.0] * len(scenario.events),
    )
    runs_path = tmp_path / "runs.jsonl"
    csv_path = tmp_path / "table.csv"
    markdown_path = tmp_path / "table.md"
    write_runs(runs_path, [run])

    assert (
        report_main(
            [
                "--runs",
                str(runs_path),
                "--csv",
                str(csv_path),
                "--markdown",
                str(markdown_path),
                "--allow-incomplete",
            ]
        )
        == 0
    )

    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["system"] == "no_memory"
    assert rows[0]["fn"] == "1"
    assert rows[0]["input_tokens"] == "100"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| no_memory | fake/model | 1 |" in markdown
    assert "# Diagnostic incomplete results — not a hypothesis test" in markdown
    assert "Preregistered success gate" not in markdown


def test_manifest_artifact_verification_hashes_actual_bytes(tmp_path: Path) -> None:
    artifact_path = tmp_path / "contract.md"
    artifact_path.write_text("frozen contract\n", encoding="utf-8")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    _verify_file_artifact(
        "research_contract",
        ArtifactPin(path=str(artifact_path), sha256=digest),
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_file_artifact(
            "research_contract",
            ArtifactPin(path=str(artifact_path), sha256="0" * 64),
        )


def test_run_manifest_binding_rejects_byte_level_tampering() -> None:
    scenario = load_scenarios("eval/scenarios/smoke.jsonl")[0]
    run = ScenarioRun(
        scenario_id=scenario.id,
        system="no_memory",
        repetition=1,
        model="fake/model",
        prompt_version="v0.3",
        scenario_sha256=canonical_sha256(scenario),
        prompt_sha256="0" * 64,
        system_config_sha256="1" * 64,
        manifest_sha256="a" * 64,
    )

    _validate_manifest_byte_binding([run], "a" * 64)
    with pytest.raises(ValueError, match="manifest byte hash"):
        _validate_manifest_byte_binding([run], "b" * 64)


def test_strict_log_reconciles_one_warmup_outside_headline_usage() -> None:
    scenario = load_scenarios("eval/scenarios/smoke.jsonl")[0]
    warmup = HostedWarmupAttestation(
        model="provider/frozen-snapshot",
        prompt_sha256=hosted_warmup_prompt_sha256(),
        response_schema_sha256=hosted_warmup_schema_sha256(),
        raw_completion='{"actions":[]}',
        usage=Usage(
            input_tokens=4,
            uncached_input_tokens=4,
            output_tokens=1,
            cost_usd=0.001,
        ),
        usage_complete=True,
        cost_complete=True,
        parse_error=False,
        latency_ms=5.0,
    )
    base_run = ScenarioRun(
        scenario_id=scenario.id,
        system="no_memory",
        repetition=1,
        model=warmup.model,
        prompt_version="v0.3",
        scenario_sha256=canonical_sha256(scenario),
        prompt_sha256="0" * 64,
        system_config_sha256="1" * 64,
        usage=Usage(
            input_tokens=10,
            uncached_input_tokens=10,
            output_tokens=2,
            cost_usd=0.001,
        ),
        decision_usage=Usage(
            input_tokens=10,
            uncached_input_tokens=10,
            output_tokens=2,
            cost_usd=0.001,
        ),
        setup_latency_ms=5.0,
        hosted_warmup=warmup,
    )
    runs = [
        base_run,
        base_run.model_copy(
            update={"scenario_id": "second-scenario", "setup_latency_ms": 0.0}
        ),
    ]
    model_usage = ModelUsage(
        input_tokens=24,
        output_tokens=5,
        total_cost=0.003,
    )
    log = SimpleNamespace(
        eval=SimpleNamespace(model=warmup.model),
        stats=SimpleNamespace(model_usage={warmup.model: model_usage}),
    )

    _validate_hosted_warmup_attestation(log, runs)  # type: ignore[arg-type]

    log.stats.model_usage[warmup.model] = model_usage.model_copy(
        update={"input_tokens": 25}
    )
    with pytest.raises(ValueError, match="headline plus one warmup"):
        _validate_hosted_warmup_attestation(log, runs)  # type: ignore[arg-type]


def test_final_dataset_release_guard_fails_closed(tmp_path: Path) -> None:
    scenarios = load_scenarios("eval/scenarios/smoke.jsonl")
    dataset_path = tmp_path / "all.jsonl"
    dataset_path.write_text("placeholder\n", encoding="utf-8")
    release_path = tmp_path / "all.manifest.json"
    release = {
        "review_status": {"independent_human_review": "pending"},
        "preregistered_final_eligible": False,
        "canonical_dataset_sha256": dataset_sha256(scenarios),
    }
    release_path.write_text(json.dumps(release), encoding="utf-8")

    with pytest.raises(ValueError, match="human review"):
        require_preregistered_final_dataset(dataset_path, scenarios)

    release["review_status"]["independent_human_review"] = "passed"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ValueError, match="not eligible"):
        require_preregistered_final_dataset(dataset_path, scenarios)

    release["preregistered_final_eligible"] = True
    release_path.write_text(json.dumps(release), encoding="utf-8")
    require_preregistered_final_dataset(dataset_path, scenarios)


def test_git_verification_rejects_a_dirty_worktree() -> None:
    def fake_runner(command, **kwargs):
        output = "a" * 40 if "rev-parse" in command else " M src/example.py"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    with pytest.raises(ValueError, match="clean git worktree"):
        _verify_git_state("a" * 40, command_runner=fake_runner)


def test_checkpoint_timeline_requires_exact_ids_timestamps_and_order() -> None:
    scenario = load_scenarios("eval/scenarios/smoke.jsonl")[0]
    checkpoints = [
        CheckpointAudit(
            event_id=event.id,
            at=event.at,
            rendered_context_sha256="0" * 64,
            raw_decision_output="{}",
        )
        for event in scenario.events
    ]
    checkpoints[0] = checkpoints[0].model_copy(update={"event_id": "future-event"})
    run = ScenarioRun(
        scenario_id=scenario.id,
        system="no_memory",
        repetition=1,
        model="fake/model",
        prompt_version="v0.3",
        scenario_sha256=canonical_sha256(scenario),
        prompt_sha256="0" * 64,
        system_config_sha256="1" * 64,
        checkpoint_latency_ms=[0.0] * len(checkpoints),
        checkpoints=checkpoints,
    )

    with pytest.raises(ValueError, match="timeline mismatch"):
        _validate_checkpoint_timeline(run, scenario)


def test_inspect_log_policy_requires_provable_effective_execution_config() -> None:
    metadata = valid_inspect_metadata()
    policy = InspectLogPolicy(
        model="provider/frozen-snapshot",
        temperature=0.0,
        seed=101,
        response_cache=False,
        max_connections=1,
        max_samples=1,
        max_tasks=1,
        model_base_url=None,
        model_args={"responses_api": False},
        max_retries=0,
        log_model_api=True,
        git_commit="b" * 40,
    )

    _validate_inspect_log_policy(metadata, policy)

    for field, value, message in (
        ("status", "error", "status"),
        ("config_update_count", 1, "config updates"),
        ("response_cache", None, "cache policy"),
        ("max_retries", None, "max_retries"),
        ("model_args", {}, "model arguments"),
        (
            "generation_config",
            {
                **metadata.generation_config,
                "top_p": 0.9,
            },
            "unpinned generation",
        ),
        ("model_base_url", "https://proxy.invalid", "base URL"),
        ("log_model_api", False, "raw model API"),
        ("max_samples", None, "max_samples"),
        ("max_tasks", 2, "max_tasks"),
        ("max_connections", 2, "max_connections"),
        ("revision_commit", None, "revision is missing"),
        ("revision_dirty", None, "clean tree"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_inspect_log_policy(
                replace(metadata, **{field: value}),
                policy,
            )


def test_system_log_metadata_is_bound_to_manifest_task_args_and_dataset() -> None:
    manifest = ExperimentManifest.model_validate_json(TEMPLATE.read_text())
    manifest = manifest.model_copy(
        update={
            "model": manifest.model.model_copy(
                update={"snapshot": "provider/frozen-snapshot"}
            ),
            "git_commit": "b" * 40,
        }
    )
    scenarios = load_scenarios("eval/scenarios/dev.jsonl")
    metadata = valid_inspect_metadata()

    assert _validate_system_log_metadata(
        metadata,
        manifest=manifest,
        manifest_path=TEMPLATE,
        mode="baseline",
        scenarios_path=Path("eval/scenarios/dev.jsonl"),
        scenario_ids=[scenario.id for scenario in scenarios],
    ) == ("no_memory", 1)

    bad_args = dict(metadata.task_args)
    bad_args["seed"] = 202
    with pytest.raises(ValueError, match="seed"):
        _validate_system_log_metadata(
            replace(metadata, task_args=bad_args),
            manifest=manifest,
            manifest_path=TEMPLATE,
            mode="baseline",
            scenarios_path=Path("eval/scenarios/dev.jsonl"),
            scenario_ids=[scenario.id for scenario in scenarios],
        )


def test_strict_run_loader_rejects_raw_jsonl() -> None:
    manifest = ExperimentManifest.model_validate_json(TEMPLATE.read_text())

    with pytest.raises(ValueError, match=r"only Inspect \.eval"):
        _load_strict_eval_runs(
            [Path("results/raw-runs.jsonl")],
            manifest=manifest,
            manifest_path=TEMPLATE,
            mode="baseline",
            scenarios_path=Path("eval/scenarios/dev.jsonl"),
            scenario_ids=[],
        )
