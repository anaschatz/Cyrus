"""Small command-line utilities for dataset validation and result reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, effective_eval_config, read_eval_log

from anamnesis.experiment import ArtifactPin, ExperimentManifest
from anamnesis.io import (
    canonical_sha256,
    dataset_sha256,
    load_runs,
    load_scenarios,
    require_preregistered_final_dataset,
)
from anamnesis.preflight import (
    _validate_openai_chat_completion_log,
    validate_model_preflight_artifact,
)
from anamnesis.prompts import memory_compiler_contract, prompt_contract
from anamnesis.schema import Scenario, ScenarioRun, Usage
from anamnesis.scoring import (
    AggregateResult,
    SuccessGateResult,
    aggregate_results,
    evaluate_success_gate,
    score_scenario,
)

DEFAULT_SCENARIOS = Path("eval/scenarios/dev.jsonl")
REQUIRED_BASELINES = {"no_memory", "full_context", "vector_rag"}
REQUIRED_FINAL_SYSTEMS = REQUIRED_BASELINES | {"anamnesis"}
REPETITIONS_BY_MODE = {"baseline": {1}, "final": {1, 2, 3}}
SEEDS_BY_MODE = {
    "baseline": {1: 101},
    "final": {1: 101, 2: 202, 3: 303},
}
EXPECTED_SCENARIOS = {"baseline": 35, "final": 50}
EXPECTED_DATASET_PATHS = {
    "baseline": Path("eval/scenarios/dev.jsonl"),
    "final": Path("eval/scenarios/all.jsonl"),
}
EXPECTED_DATASET_NAMES = {
    "baseline": "anamnesis-development-v0",
    "final": "anamnesis-all-v0",
}
EXPECTED_DATASET_SPLITS = {"baseline": "development", "final": "all"}
INSPECT_TASK_FILE = Path("eval/anamnesis_eval.py")


@dataclass(frozen=True)
class InspectLogMetadata:
    """Small, testable projection of execution-critical Inspect log fields."""

    status: str
    invalidated: bool
    config_update_count: int
    log_update_count: int
    task: str
    task_file: str | None
    task_args: dict[str, Any]
    task_metadata: dict[str, Any]
    model: str
    model_base_url: str | None
    model_args: dict[str, Any]
    generation_config: dict[str, Any]
    dataset_name: str | None
    dataset_location: str | None
    dataset_samples: int | None
    dataset_sample_ids: tuple[str, ...] | None
    dataset_shuffled: bool | None
    temperature: float | None
    seed: int | None
    response_cache: object
    max_retries: int | None
    max_connections: int | None
    adaptive_connections: object
    max_samples: int | None
    max_tasks: int | None
    log_model_api: bool | None
    epochs: int | None
    revision_commit: str | None
    revision_dirty: bool | None


@dataclass(frozen=True)
class InspectLogPolicy:
    """Reusable execution policy for measured runs and pinned preflight logs."""

    model: str
    temperature: float
    seed: int
    response_cache: bool
    max_connections: int
    max_samples: int
    max_tasks: int
    model_base_url: str | None
    model_args: dict[str, Any]
    max_retries: int
    log_model_api: bool
    git_commit: str | None


def _validation_summary(scenarios: list[Scenario]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scenarios": len(scenarios),
        "events": sum(len(scenario.events) for scenario in scenarios),
        "expected_actions": sum(
            len(scenario.expected_actions) for scenario in scenarios
        ),
        "forbidden_actions": sum(
            len(scenario.forbidden_actions) for scenario in scenarios
        ),
        "dataset_sha256": dataset_sha256(scenarios),
    }


def validate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an Anamnesis dataset")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_SCENARIOS)
    args = parser.parse_args(argv)
    scenarios = load_scenarios(args.path)
    print(json.dumps(_validation_summary(scenarios), indent=2, sort_keys=True))
    return 0


def _inspect_log_metadata(log: EvalLog) -> InspectLogMetadata:
    """Project one Inspect log without retaining provider or sample payloads."""

    # Inspect applies the task/CLI plan config over the model-instance config.
    # Config updates are forbidden below, so the launch composition is also the
    # effective scientific generation configuration.
    generation = log.eval.model_generate_config.merge(log.plan.config)
    eval_config = effective_eval_config(log)
    revision = log.eval.revision
    dataset_ids = log.eval.dataset.sample_ids
    return InspectLogMetadata(
        status=log.status,
        invalidated=log.invalidated,
        config_update_count=len(log.config_updates or []),
        log_update_count=len(log.log_updates or []),
        task=log.eval.task_registry_name or log.eval.task,
        task_file=log.eval.task_file,
        task_args=dict(log.eval.task_args),
        task_metadata=dict(log.eval.metadata or {}),
        model=log.eval.model,
        model_base_url=log.eval.model_base_url,
        model_args=dict(log.eval.model_args),
        generation_config=generation.model_dump(mode="json", exclude_none=True),
        dataset_name=log.eval.dataset.name,
        dataset_location=log.eval.dataset.location,
        dataset_samples=log.eval.dataset.samples,
        dataset_sample_ids=(
            tuple(str(sample_id) for sample_id in dataset_ids)
            if dataset_ids is not None
            else None
        ),
        dataset_shuffled=log.eval.dataset.shuffled,
        temperature=generation.temperature,
        seed=generation.seed,
        response_cache=generation.cache,
        max_retries=generation.max_retries,
        max_connections=generation.max_connections,
        adaptive_connections=generation.adaptive_connections,
        max_samples=eval_config.max_samples,
        max_tasks=eval_config.max_tasks,
        log_model_api=eval_config.log_model_api,
        epochs=eval_config.epochs,
        revision_commit=revision.commit if revision is not None else None,
        revision_dirty=revision.dirty if revision is not None else None,
    )


def _validate_inspect_log_policy(
    metadata: InspectLogMetadata,
    policy: InspectLogPolicy,
) -> None:
    """Fail closed unless Inspect proves the declared measured-run policy."""

    if metadata.status != "success":
        raise ValueError("measured Inspect logs must have status='success'")
    if metadata.invalidated:
        raise ValueError("invalidated Inspect logs cannot be measured")
    if metadata.config_update_count:
        raise ValueError("measured Inspect logs cannot contain config updates")
    if metadata.log_update_count:
        raise ValueError("edited Inspect logs cannot be measured")
    if metadata.model != policy.model:
        raise ValueError("Inspect log model differs from the frozen manifest")
    if metadata.model_base_url != policy.model_base_url:
        raise ValueError("Inspect log does not prove the official provider base URL")
    if metadata.model_args != policy.model_args:
        raise ValueError("Inspect log model arguments differ from execution policy")
    expected_generation_config = {
        "max_retries": policy.max_retries,
        "max_connections": policy.max_connections,
        "adaptive_connections": False,
        "temperature": policy.temperature,
        "seed": policy.seed,
        "cache": policy.response_cache,
    }
    if metadata.generation_config != expected_generation_config:
        raise ValueError("Inspect log contains unpinned generation configuration")
    if metadata.temperature != policy.temperature:
        raise ValueError("Inspect log temperature differs from execution policy")
    if metadata.seed != policy.seed:
        raise ValueError("Inspect log seed differs from execution policy")
    if metadata.response_cache is not policy.response_cache:
        raise ValueError("Inspect response-generation cache policy is unverifiable")
    if metadata.max_retries != policy.max_retries:
        raise ValueError("Inspect max_retries differs from execution policy")
    if metadata.max_connections != policy.max_connections:
        raise ValueError("Inspect max_connections differs from execution policy")
    if metadata.adaptive_connections not in (None, False):
        raise ValueError("adaptive model connections are forbidden in measured runs")
    if metadata.max_samples != policy.max_samples:
        raise ValueError("Inspect max_samples differs from execution policy")
    if metadata.max_tasks != policy.max_tasks:
        raise ValueError("Inspect max_tasks differs from execution policy")
    if metadata.log_model_api is not policy.log_model_api:
        raise ValueError("Inspect log must retain raw model API calls")
    if metadata.epochs != 1:
        raise ValueError("measured Inspect logs require exactly one epoch")
    revision = metadata.revision_commit
    if revision is None or policy.git_commit is None:
        raise ValueError("Inspect log git revision is missing or unverifiable")
    if (
        len(revision) < 7
        or any(character not in "0123456789abcdef" for character in revision)
        or not policy.git_commit.startswith(revision)
    ):
        raise ValueError("Inspect log git revision differs from the manifest")
    if metadata.revision_dirty is not False:
        raise ValueError("Inspect log git revision does not prove a clean tree")


def _validate_system_log_metadata(
    metadata: InspectLogMetadata,
    *,
    manifest: ExperimentManifest,
    manifest_path: Path,
    mode: str,
    scenarios_path: Path,
    scenario_ids: list[str],
) -> tuple[str, int]:
    """Validate the task and dataset identity for one measured system log."""

    task_name = metadata.task.rsplit("@", maxsplit=1)[-1]
    if task_name not in manifest.systems:
        raise ValueError(f"unexpected measured Inspect task: {metadata.task}")
    if metadata.task_file is None or (
        Path(metadata.task_file).resolve() != INSPECT_TASK_FILE.resolve()
    ):
        raise ValueError("Inspect log task file is not eval/anamnesis_eval.py")

    task_args = metadata.task_args
    allowed_args = {"seed", "repetition", "dataset", "manifest"}
    if task_name == "vector_rag":
        allowed_args.update(
            {
                "top_k",
                "embedding_model",
                "embedding_repository",
                "embedding_revision",
                "embedding_snapshot_path",
            }
        )
    unknown_args = set(task_args) - allowed_args
    if unknown_args:
        raise ValueError(f"unexpected measured task arguments: {sorted(unknown_args)}")

    repetition = task_args.get("repetition")
    if isinstance(repetition, bool) or not isinstance(repetition, int):
        raise ValueError("Inspect task repetition is missing or invalid")
    if repetition < 1 or repetition > manifest.execution.repetitions:
        raise ValueError("Inspect task repetition is outside the manifest")
    expected_seed = manifest.execution.seeds[repetition - 1]
    if task_args.get("seed") != expected_seed:
        raise ValueError("Inspect task seed differs from its repetition")
    if task_args.get("dataset") != EXPECTED_DATASET_SPLITS[mode]:
        raise ValueError("Inspect task dataset split differs from report mode")
    task_manifest = task_args.get("manifest")
    if not isinstance(task_manifest, str) or (
        Path(task_manifest).resolve() != manifest_path.resolve()
    ):
        raise ValueError("Inspect task was not bound to this frozen manifest")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if metadata.task_metadata.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Inspect task metadata has a stale manifest byte hash")
    from anamnesis.inspect_adapter import (
        hosted_warmup_prompt_sha256,
        hosted_warmup_schema_sha256,
    )

    if (
        metadata.task_metadata.get("hosted_warmup_prompt_sha256")
        != hosted_warmup_prompt_sha256()
        or metadata.task_metadata.get("hosted_warmup_schema_sha256")
        != hosted_warmup_schema_sha256()
    ):
        raise ValueError("Inspect task warmup contract differs from runtime")

    if task_name == "vector_rag":
        expected_embedding_args = {
            "top_k": manifest.embedding.top_k,
            "embedding_model": manifest.embedding.model,
            "embedding_repository": manifest.embedding.repository,
            "embedding_revision": manifest.embedding.revision,
        }
        for name, expected in expected_embedding_args.items():
            if task_args.get(name) != expected:
                raise ValueError(f"Inspect vector task {name} differs from manifest")
        _validate_vector_snapshot_task_arg(task_args)

    _validate_inspect_log_policy(
        metadata,
        InspectLogPolicy(
            model=manifest.model.snapshot or "",
            temperature=manifest.execution.temperature,
            seed=expected_seed,
            response_cache=manifest.execution.response_cache,
            max_connections=manifest.execution.concurrency,
            max_samples=manifest.execution.max_samples,
            # v0 has one shared concurrency pin; bind it to both task-level
            # and model-connection concurrency until those knobs are split.
            max_tasks=manifest.execution.concurrency,
            model_base_url=None,
            model_args=manifest.model.provider_args.model_dump(mode="python"),
            max_retries=manifest.execution.max_retries,
            log_model_api=manifest.execution.log_model_api,
            git_commit=manifest.git_commit,
        ),
    )

    if metadata.dataset_name != EXPECTED_DATASET_NAMES[mode]:
        raise ValueError("Inspect log dataset name differs from report mode")
    if metadata.dataset_location is None or (
        Path(metadata.dataset_location).resolve() != scenarios_path.resolve()
    ):
        raise ValueError("Inspect log dataset location differs from --scenarios")
    if metadata.dataset_samples != len(scenario_ids):
        raise ValueError("Inspect log dataset size differs from --scenarios")
    if metadata.dataset_sample_ids != tuple(scenario_ids):
        raise ValueError("Inspect log sample IDs/order differ from --scenarios")
    if metadata.dataset_shuffled is not False:
        raise ValueError("measured Inspect datasets must be unshuffled")
    return task_name, repetition


def _validate_vector_snapshot_task_arg(task_args: dict[str, Any]) -> None:
    """Require proof that the measured vector task used an explicit local path."""

    snapshot_path = task_args.get("embedding_snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path.strip():
        raise ValueError(
            "Inspect vector task requires an explicit embedding_snapshot_path"
        )
    if not Path(snapshot_path).is_absolute():
        raise ValueError(
            "Inspect vector task embedding_snapshot_path must be an absolute local path"
        )


def _scenario_runs_from_eval_log(log: EvalLog) -> list[ScenarioRun]:
    """Extract only runner-owned, schema-validated records from an Inspect log."""

    from anamnesis.inspect_adapter import SCENARIO_RUN_STORE_KEY

    if log.samples is None:
        raise ValueError("Inspect log does not contain sample records")
    runs: list[ScenarioRun] = []
    for sample in log.samples:
        if sample.epoch != 1:
            raise ValueError("Inspect sample epoch differs from measured policy")
        if sample.error is not None or sample.invalidation is not None:
            raise ValueError(f"Inspect sample {sample.id} is errored or invalidated")
        if sample.error_retries:
            raise ValueError(f"Inspect sample {sample.id} contains retry history")
        raw_run = sample.store.get(SCENARIO_RUN_STORE_KEY)
        if raw_run is None:
            raise ValueError(f"Inspect sample {sample.id} has no Anamnesis ScenarioRun")
        run = ScenarioRun.model_validate(raw_run)
        if str(sample.id) != run.scenario_id:
            raise ValueError("Inspect sample ID differs from its ScenarioRun")
        runs.append(run)
    return runs


def _validate_hosted_warmup_attestation(
    log: EvalLog,
    runs: list[ScenarioRun],
) -> None:
    """Prove one setup call occurred and was excluded from headline usage."""

    if not runs or any(run.hosted_warmup is None for run in runs):
        raise ValueError("measured ScenarioRuns are missing hosted warmup attestation")
    warmups = [run.hosted_warmup for run in runs]
    serialized = {warmup.model_dump_json() for warmup in warmups if warmup is not None}
    if len(serialized) != 1:
        raise ValueError("ScenarioRuns disagree about the task-level hosted warmup")
    warmup = warmups[0]
    assert warmup is not None

    from anamnesis.inspect_adapter import (
        hosted_warmup_prompt_sha256,
        hosted_warmup_schema_sha256,
    )

    if (
        warmup.model != log.eval.model
        or warmup.prompt_sha256 != hosted_warmup_prompt_sha256()
        or warmup.response_schema_sha256 != hosted_warmup_schema_sha256()
        or warmup.parse_error
        or not warmup.usage_complete
        or not warmup.cost_complete
    ):
        raise ValueError("hosted warmup attestation is invalid or stale")
    if runs[0].setup_latency_ms < warmup.latency_ms or any(
        not math.isclose(run.setup_latency_ms, 0.0, abs_tol=1e-12) for run in runs[1:]
    ):
        raise ValueError("hosted warmup was not recorded exactly once as setup latency")

    if set(log.stats.model_usage) != {warmup.model}:
        raise ValueError("Inspect log model usage does not match the warmup model")
    actual = Usage()
    for model_usage in log.stats.model_usage.values():
        if model_usage.total_cost is None:
            raise ValueError("Inspect log cannot attest complete warmup cost")
        actual = actual.plus(
            Usage(
                input_tokens=(
                    model_usage.input_tokens
                    + (model_usage.input_tokens_cache_read or 0)
                    + (model_usage.input_tokens_cache_write or 0)
                ),
                uncached_input_tokens=model_usage.input_tokens,
                cache_read_input_tokens=model_usage.input_tokens_cache_read or 0,
                cache_write_input_tokens=model_usage.input_tokens_cache_write or 0,
                output_tokens=model_usage.output_tokens,
                cost_usd=model_usage.total_cost,
            )
        )
    expected = warmup.usage
    for run in runs:
        expected = expected.plus(run.usage)
    for field_name in (
        "input_tokens",
        "uncached_input_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
    ):
        if getattr(actual, field_name) != getattr(expected, field_name):
            raise ValueError("Inspect usage does not equal headline plus one warmup")
    if (
        actual.cost_usd is None
        or expected.cost_usd is None
        or not math.isclose(
            actual.cost_usd,
            expected.cost_usd,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Inspect cost does not equal headline plus one warmup")


def _load_strict_eval_runs(
    paths: list[Path],
    *,
    manifest: ExperimentManifest,
    manifest_path: Path,
    mode: str,
    scenarios_path: Path,
    scenario_ids: list[str],
) -> list[ScenarioRun]:
    runs: list[ScenarioRun] = []
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for path in paths:
        if path.suffix != ".eval":
            raise ValueError("strict reports accept only Inspect .eval logs")
        log = read_eval_log(path)
        system, repetition = _validate_system_log_metadata(
            _inspect_log_metadata(log),
            manifest=manifest,
            manifest_path=manifest_path,
            mode=mode,
            scenarios_path=scenarios_path,
            scenario_ids=scenario_ids,
        )
        log_runs = _scenario_runs_from_eval_log(log)
        _validate_openai_chat_completion_log(
            log,
            model_name=manifest.model.snapshot or "",
            temperature=manifest.execution.temperature,
            seed=manifest.execution.seeds[repetition - 1],
        )
        _validate_hosted_warmup_attestation(log, log_runs)
        for run in log_runs:
            if (
                run.system != system
                or run.repetition != repetition
                or run.model != manifest.model.snapshot
                or run.manifest_sha256 != manifest_sha256
            ):
                raise ValueError("ScenarioRun identity differs from its Inspect log")
        runs.extend(log_runs)
    return runs


def _load_diagnostic_runs(paths: list[Path]) -> list[ScenarioRun]:
    """Diagnostic mode retains raw JSONL support and may also read .eval logs."""

    runs: list[ScenarioRun] = []
    for path in paths:
        if path.suffix == ".eval":
            runs.extend(_scenario_runs_from_eval_log(read_eval_log(path)))
        else:
            runs.extend(load_runs(path))
    return runs


def _result_row(result: AggregateResult) -> dict[str, object]:
    row = result.model_dump()
    row["precision"] = round(result.precision, 6)
    row["recall"] = round(result.recall, 6)
    row["f1"] = round(result.f1, 6)
    for key in (
        "false_alarm_rate",
        "obsolete_trap_rate",
        "provenance_exact_accuracy",
        "input_token_reduction_vs_full_context",
    ):
        value = row[key]
        row[key] = "" if value is None else round(float(value), 6)
    row["cost_usd"] = "" if result.cost_usd is None else result.cost_usd
    return row


def _write_csv(path: Path, results: list[AggregateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(result) for result in results]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _validate_frozen_manifest(
    *,
    path: Path,
    mode: str,
    scenarios_path: Path,
    scenarios: list[Scenario],
    runs: list[ScenarioRun],
) -> ExperimentManifest:
    """Bind a strict report to its preregistered frozen experiment."""

    manifest_bytes = path.read_bytes()
    manifest = ExperimentManifest.model_validate_json(manifest_bytes)
    if manifest.status != "frozen":
        raise ValueError("measured reports require a frozen experiment manifest")
    if manifest.phase != mode:
        raise ValueError(
            f"manifest phase {manifest.phase!r} differs from report mode {mode!r}"
        )
    if manifest.scenario_count != len(scenarios):
        raise ValueError("manifest scenario_count differs from the dataset")
    if Path(manifest.dataset.path).resolve() != scenarios_path.resolve():
        raise ValueError("manifest dataset path differs from --scenarios")
    canonical_dataset_hash = dataset_sha256(scenarios)
    if manifest.dataset.sha256 != canonical_dataset_hash:
        raise ValueError("manifest dataset hash differs from --scenarios")
    if mode == "final":
        require_preregistered_final_dataset(scenarios_path, scenarios)
    for name, artifact in (
        ("model.pricing", manifest.model.pricing),
        ("model.preflight", manifest.model.preflight),
        ("dependency_lock", manifest.dependency_lock),
        ("research_contract", manifest.research_contract),
        ("architecture_contract", manifest.architecture_contract),
    ):
        _verify_file_artifact(name, artifact)
    validate_model_preflight_artifact(
        manifest.model.preflight,
        model_name=manifest.model.snapshot or "",
        pricing=manifest.model.pricing,
    )
    _verify_git_state(manifest.git_commit)
    models = {run.model for run in runs}
    if models != {manifest.model.snapshot}:
        raise ValueError("run model differs from the frozen model snapshot")
    _validate_manifest_byte_binding(
        runs,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )
    pricing_hashes = {run.pricing_config_sha256 for run in runs}
    if pricing_hashes != {manifest.model.pricing.sha256}:
        raise ValueError("run pricing configuration differs from the manifest")
    prompt_hashes = {run.prompt_sha256 for run in runs}
    if prompt_hashes != {manifest.decision_prompt_sha256}:
        raise ValueError("run prompt hash differs from the frozen manifest")
    current_prompt_hash = hashlib.sha256(prompt_contract().encode()).hexdigest()
    if manifest.decision_prompt_sha256 != current_prompt_hash:
        raise ValueError("current decision prompt differs from the manifest")
    if mode == "final":
        compiler_hash = hashlib.sha256(memory_compiler_contract().encode()).hexdigest()
        if manifest.memory_compiler_sha256 != compiler_hash:
            raise ValueError("memory compiler contract differs from the manifest")
    run_systems = {run.system for run in runs}
    if run_systems != set(manifest.systems):
        raise ValueError("run systems differ from the frozen manifest")
    for system in manifest.systems:
        hashes = {run.system_config_sha256 for run in runs if run.system == system}
        if hashes != {manifest.system_config_sha256[system]}:
            raise ValueError(
                f"run configuration for {system} differs from the manifest"
            )
    return manifest


def _validate_manifest_byte_binding(
    runs: list[ScenarioRun], expected_sha256: str
) -> None:
    """Bind every raw run to the exact manifest bytes used at task creation."""

    hashes = {run.manifest_sha256 for run in runs}
    if hashes != {expected_sha256}:
        raise ValueError("ScenarioRun manifest byte hash differs from --manifest")


def _verify_file_artifact(name: str, artifact: ArtifactPin) -> None:
    path = Path(artifact.path)
    if not path.is_file():
        raise ValueError(f"manifest artifact {name} does not exist: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if artifact.sha256 != digest:
        raise ValueError(f"manifest artifact hash mismatch: {name}")


def _verify_git_state(expected_commit: str | None, command_runner=None) -> None:
    runner = command_runner or subprocess.run
    current_commit = runner(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if expected_commit != current_commit:
        raise ValueError("manifest git_commit differs from current HEAD")
    dirty = runner(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("measured reports require a clean git worktree")


def _validate_checkpoint_timeline(run: ScenarioRun, scenario: Scenario) -> None:
    """Require the trace to match authored checkpoint IDs, times, and order."""

    actual = [(checkpoint.event_id, checkpoint.at) for checkpoint in run.checkpoints]
    expected = [(event.id, event.at) for event in scenario.events]
    if actual != expected:
        raise ValueError(
            f"checkpoint timeline mismatch for {run.scenario_id}/{run.system}"
        )


def _markdown_table(
    results: list[AggregateResult],
    *,
    title: str | None = None,
    gates: list[SuccessGateResult] | None = None,
) -> str:
    header = (
        "| System | Model | Rep | TP | FP | FN | Precision | Recall | F1 | "
        "False reminders | FAR | Obsolete | Provenance | Input tokens | "
        "Reduction vs full | Cost USD | Latency p50/p95 ms | Setup ms |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---:|"
    )
    rows = []
    for result in results:
        cost = "N/A" if result.cost_usd is None else f"{result.cost_usd:.6f}"
        rows.append(
            "| "
            + " | ".join(
                [
                    result.system,
                    result.model,
                    str(result.repetition),
                    str(result.tp),
                    str(result.fp),
                    str(result.fn),
                    _format_percent(result.precision),
                    _format_percent(result.recall),
                    _format_percent(result.f1),
                    str(result.false_reminders),
                    _format_percent(result.false_alarm_rate),
                    str(result.obsolete_errors),
                    _format_percent(result.provenance_exact_accuracy),
                    str(result.input_tokens),
                    _format_percent(result.input_token_reduction_vs_full_context),
                    cost,
                    f"{result.latency_p50_ms:.1f}/{result.latency_p95_ms:.1f}",
                    f"{result.setup_latency_ms:.1f}",
                ]
            )
            + " |"
        )
    prefix = f"# {title}\n\n" if title else ""
    table = prefix + header + "\n" + "\n".join(rows) + "\n"
    if gates is None:
        return table
    gate_header = (
        "\n## Preregistered success gate\n\n"
        "| Model | Rep | Comparator | F1 gain | Token reduction | "
        "False alarms (Anamnesis/comparator) | Supported |\n"
        "|---|---:|---|---:|---:|---:|---|"
    )
    gate_rows = [
        "| "
        + " | ".join(
            [
                gate.model,
                str(gate.repetition),
                gate.comparator,
                f"{gate.f1_gain:.1%}",
                f"{gate.input_token_reduction:.1%}",
                (
                    f"{gate.anamnesis_false_alarm_checkpoints}/"
                    f"{gate.comparator_false_alarm_checkpoints}"
                ),
                "yes" if gate.supported else "no",
            ]
        )
        + " |"
        for gate in gates
    ]
    if not gates:
        raise ValueError("a final success-gate section requires repetitions")
    if all(gate.supported for gate in gates):
        conclusion = (
            "**Overall preregistered conclusion: hypothesis supported in all "
            "repetitions.**"
        )
    else:
        conclusion = (
            "**Overall preregistered conclusion: hypothesis not supported by "
            "the frozen criteria.**"
        )
    return (
        table + gate_header + "\n" + "\n".join(gate_rows) + "\n\n" + conclusion + "\n"
    )


def report_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score raw Anamnesis runs")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Frozen experiment manifest (required for a strict report)",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "final"),
        default="baseline",
        help="Strict experiment matrix to validate before reporting",
    )
    parser.add_argument("--csv", type=Path, default=Path("results/table.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("results/table.md"))
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a report without every scenario in each run group",
    )
    args = parser.parse_args(argv)

    scenarios = load_scenarios(args.scenarios)
    by_id = {scenario.id: scenario for scenario in scenarios}
    strict_manifest: ExperimentManifest | None = None
    if args.allow_incomplete:
        runs = _load_diagnostic_runs(args.runs)
    else:
        if args.manifest is None:
            raise ValueError("strict reports require --manifest")
        expected_dataset_path = EXPECTED_DATASET_PATHS[args.mode].resolve()
        if args.scenarios.resolve() != expected_dataset_path:
            raise ValueError(
                f"strict {args.mode} reports require {expected_dataset_path}"
            )
        strict_manifest = ExperimentManifest.model_validate_json(
            args.manifest.read_text(encoding="utf-8")
        )
        if strict_manifest.status != "frozen" or strict_manifest.phase != args.mode:
            raise ValueError(f"strict {args.mode} reports require a frozen manifest")
        runs = _load_strict_eval_runs(
            args.runs,
            manifest=strict_manifest,
            manifest_path=args.manifest,
            mode=args.mode,
            scenarios_path=args.scenarios,
            scenario_ids=list(by_id),
        )
    run_keys = [
        (run.scenario_id, run.system, run.repetition, run.model) for run in runs
    ]
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("duplicate scenario/system/repetition/model run records")
    prompt_hashes = {run.prompt_sha256 for run in runs}
    prompt_versions = {run.prompt_version for run in runs}
    if len(prompt_hashes) != 1 or len(prompt_versions) != 1:
        raise ValueError("all compared runs must use one prompt contract")
    if not args.allow_incomplete:
        assert args.manifest is not None
        assert strict_manifest is not None
        _validate_frozen_manifest(
            path=args.manifest,
            mode=args.mode,
            scenarios_path=args.scenarios,
            scenarios=scenarios,
            runs=runs,
        )
        expected_scenario_count = EXPECTED_SCENARIOS[args.mode]
        if len(scenarios) != expected_scenario_count:
            raise ValueError(
                f"{args.mode} mode requires exactly {expected_scenario_count} scenarios"
            )
        expected_ids = set(by_id)
        grouped_ids: dict[tuple[str, int, str], set[str]] = {}
        for run in runs:
            key = (run.system, run.repetition, run.model)
            grouped_ids.setdefault(key, set()).add(run.scenario_id)
        for key, actual_ids in grouped_ids.items():
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)
                extra = sorted(actual_ids - expected_ids)
                raise ValueError(
                    f"incomplete run group {key}: missing={missing}, extra={extra}"
                )
        models = {run.model for run in runs}
        required_systems = (
            REQUIRED_BASELINES if args.mode == "baseline" else REQUIRED_FINAL_SYSTEMS
        )
        required_repetitions = REPETITIONS_BY_MODE[args.mode]
        required_seeds = SEEDS_BY_MODE[args.mode]
        for model in models:
            actual_systems = {run.system for run in runs if run.model == model}
            if actual_systems != required_systems:
                raise ValueError(
                    f"{model}/{args.mode} requires systems "
                    f"{sorted(required_systems)} exactly; got "
                    f"{sorted(actual_systems)}"
                )
            for system in required_systems:
                available_repetitions = {
                    run.repetition
                    for run in runs
                    if run.model == model and run.system == system
                }
                if available_repetitions != required_repetitions:
                    raise ValueError(
                        f"{model}/{system} must use repetitions "
                        f"{sorted(required_repetitions)} exactly; got "
                        f"{sorted(available_repetitions)}"
                    )
            for repetition in required_repetitions:
                seeds = {
                    run.seed
                    for run in runs
                    if run.model == model
                    and run.repetition == repetition
                    and run.system in required_systems
                }
                expected_seed = required_seeds[repetition]
                if seeds != {expected_seed}:
                    raise ValueError(
                        f"{model} repetition {repetition} must use seed "
                        f"{expected_seed} across all systems; got "
                        f"{sorted(str(seed) for seed in seeds)}"
                    )
            for system in required_systems:
                config_hashes = {
                    run.system_config_sha256
                    for run in runs
                    if run.model == model and run.system == system
                }
                if len(config_hashes) != 1:
                    raise ValueError(
                        f"{model}/{system} used multiple system configurations"
                    )
            incomplete_usage = [
                run.scenario_id for run in runs if not run.usage_complete
            ]
            if incomplete_usage:
                raise ValueError(
                    f"measured runs have incomplete usage: {sorted(incomplete_usage)}"
                )
            incomplete_cost = [run.scenario_id for run in runs if not run.cost_complete]
            if incomplete_cost:
                raise ValueError(
                    f"measured runs have incomplete pricing: {sorted(incomplete_cost)}"
                )
    scored = []
    for run in runs:
        scenario = by_id.get(run.scenario_id)
        if scenario is None:
            raise ValueError(f"run references unknown scenario: {run.scenario_id}")
        expected_hash = canonical_sha256(scenario)
        if run.scenario_sha256 != expected_hash:
            raise ValueError(f"scenario hash mismatch for {run.scenario_id}")
        if len(run.checkpoint_latency_ms) != len(scenario.events):
            raise ValueError(
                f"checkpoint count mismatch for {run.scenario_id}: "
                f"expected {len(scenario.events)}, got "
                f"{len(run.checkpoint_latency_ms)}"
            )
        if (not args.allow_incomplete or run.checkpoints) and len(
            run.checkpoints
        ) != len(scenario.events):
            raise ValueError(
                f"checkpoint audit count mismatch for {run.scenario_id}: "
                f"expected {len(scenario.events)}, got {len(run.checkpoints)}"
            )
        if not args.allow_incomplete:
            _validate_checkpoint_timeline(run, scenario)
            if run.decision_usage.input_tokens <= 0:
                raise ValueError(
                    f"{run.scenario_id}/{run.system} has zero decision input tokens"
                )
            expected_compiler_calls = [
                run.system == "anamnesis" and event.kind != "clock_tick"
                for event in scenario.events
            ]
            actual_compiler_calls = [
                checkpoint.compiler_called for checkpoint in run.checkpoints
            ]
            if actual_compiler_calls != expected_compiler_calls:
                raise ValueError(
                    f"compiler call policy mismatch for {run.scenario_id}/{run.system}"
                )
            for checkpoint in run.checkpoints:
                if checkpoint.decision_usage.input_tokens <= 0:
                    raise ValueError(
                        f"checkpoint {checkpoint.event_id} has zero decision usage"
                    )
                if (
                    checkpoint.compiler_called
                    and checkpoint.compiler_usage.input_tokens <= 0
                ):
                    raise ValueError(
                        f"checkpoint {checkpoint.event_id} has zero compiler usage"
                    )
        scored.append((score_scenario(scenario, run), run))
    results = aggregate_results(scored)
    if not results:
        raise ValueError("no aggregate results were produced")
    gates = (
        evaluate_success_gate(results)
        if args.mode == "final" and not args.allow_incomplete
        else None
    )

    if args.allow_incomplete:
        title = "Diagnostic incomplete results — not a hypothesis test"
    elif args.mode == "baseline":
        title = "Development baseline — not a final hypothesis test"
    else:
        title = "Final preregistered results"

    _write_csv(args.csv, results)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    rendered = _markdown_table(results, title=title, gates=gates)
    args.markdown.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0
