"""Strict reporting for the local oracle-compiler smoke ceiling.

The oracle compiler is a frozen human-authored diagnostic input, not an
evaluated memory compiler.  This reporter therefore proves that scenario-time
compiler records came from the pinned oracle artifact, that no scenario
compiler model call occurred, and that only decision-model usage enters the
headline row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, read_eval_log

from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.local_experiment import (
    LOCAL_MODEL_ID,
    LocalExperimentManifest,
    validate_zero_api_pricing,
    verify_static_local_inputs,
)
from anamnesis.local_preflight import validate_local_preflight_artifact
from anamnesis.local_report import (
    LOCAL_TASK_FILE,
    REPO_ROOT,
    _extract_scenario_run,
    _logged_dataset_candidates,
    _parse_decision_event,
    _resolve_logged_repo_path,
    _sha256_file,
    _sha256_text,
    _validate_effective_log_policy,
    _validate_live_preflight,
    _validate_log_usage,
    _validate_path_task_arg,
    _verify_current_git_state,
)
from anamnesis.local_runtime import (
    LOCAL_DECISION_VERSION,
    LOCAL_PREFLIGHT_STORE_KEY,
    LOCAL_SCENARIO_TASK_VERSION,
    LocalModelPreflightResult,
    local_decision_contract,
    local_decision_prompt_contract,
    local_decision_schema_contract,
    local_system_config_sha256,
)
from anamnesis.oracle import (
    ORACLE_ANNOTATION_POLICY,
    ORACLE_ARTIFACT_PURPOSE,
    ORACLE_COMPILER_VERSION,
    ORACLE_SYSTEM_NAME,
    OracleCompilerArtifact,
    load_oracle_artifact,
    oracle_artifact_sha256,
)
from anamnesis.schema import PredictedAction, Scenario, ScenarioRun, Usage
from anamnesis.scoring import AggregateResult, aggregate_results, score_scenario

ORACLE_CEILING_TITLE = "Local oracle-compiler ceiling — diagnostic only"
ORACLE_TASK_NAME = "local_anamnesis_oracle_compiler"
ORACLE_DATASET_NAME = "anamnesis-local-oracle_smoke-v0"
ORACLE_PROVENANCE_PATH = Path("results/local_oracle_smoke.provenance.json")
ORACLE_CSV_PATH = Path("results/local_oracle_smoke.csv")
ORACLE_MARKDOWN_PATH = Path("results/local_oracle_smoke.md")


def _repo_relative(
    path: Path,
    *,
    label: str,
    must_exist: bool = True,
) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"{label} must be inside the repository")
    if must_exist and not resolved.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")
    return resolved.relative_to(REPO_ROOT).as_posix()


def _repo_file(relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} path must be repository-relative")
    resolved = (REPO_ROOT / candidate).resolve()
    if not resolved.is_relative_to(REPO_ROOT) or not resolved.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return resolved


def _require_pinned_file(path: Path, expected_sha256: str | None, *, label: str) -> str:
    if expected_sha256 is None:
        raise ValueError(f"frozen oracle manifest is missing {label} SHA-256")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} bytes differ from the frozen manifest")
    return actual


def _expected_oracle_system_hash(manifest: LocalExperimentManifest) -> str:
    annotations = manifest.oracle_annotations
    if annotations is None or annotations.sha256 is None:
        raise ValueError("frozen oracle manifest is missing oracle annotations")
    return local_system_config_sha256(
        system=ORACLE_SYSTEM_NAME,
        top_k=manifest.embedding.top_k,
        embedding_model=manifest.embedding.model,
        embedding_repository=manifest.embedding.repository,
        embedding_revision=manifest.embedding.revision,
        pricing_config_sha256=manifest.model.pricing.sha256,
        oracle_annotations_sha256=annotations.sha256,
    )


def _validate_oracle_manifest_identity(manifest: LocalExperimentManifest) -> None:
    if manifest.status != "frozen" or manifest.phase != "oracle_smoke":
        raise ValueError("oracle ceiling requires a frozen oracle_smoke manifest")
    if manifest.compiler_mode != "oracle":
        raise ValueError("oracle ceiling manifest requires compiler_mode=oracle")
    if manifest.hypothesis_test_eligible is not False:
        raise ValueError("oracle ceiling cannot be hypothesis-test eligible")
    if manifest.systems != [ORACLE_SYSTEM_NAME]:
        raise ValueError("oracle ceiling manifest requires exactly the oracle system")
    if manifest.scenario_count != 10:
        raise ValueError("oracle ceiling requires exactly 10 smoke scenarios")
    if manifest.execution.repetitions != 1 or manifest.execution.seeds != [101]:
        raise ValueError("oracle ceiling requires one repetition with seed 101")
    if manifest.model.same_model_for_compiler_and_decision:
        raise ValueError("oracle ceiling cannot claim an LLM scenario compiler")
    if (
        manifest.memory_compiler_prompt_sha256 is not None
        or manifest.memory_compiler_schema_sha256 is not None
    ):
        raise ValueError("oracle ceiling cannot pin an LLM compiler contract")
    if manifest.oracle_annotations is None:
        raise ValueError("oracle ceiling manifest is missing oracle annotations")
    if set(manifest.system_config_sha256) != {ORACLE_SYSTEM_NAME}:
        raise ValueError("oracle ceiling manifest has the wrong system hash matrix")


def _validate_frozen_oracle_manifest(
    *,
    manifest_path: Path,
    scenarios_path: Path,
    oracle_path: Path,
    command_runner: Callable[..., Any] | None = None,
) -> tuple[
    LocalExperimentManifest,
    list[Scenario],
    OracleCompilerArtifact,
    str,
]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = LocalExperimentManifest.model_validate_json(manifest_bytes)
    _validate_oracle_manifest_identity(manifest)

    expected_dataset = _repo_file(manifest.dataset.path, label="scenario dataset")
    if scenarios_path.resolve() != expected_dataset:
        raise ValueError("--scenarios differs from the frozen oracle dataset")
    annotations = manifest.oracle_annotations
    assert annotations is not None
    expected_oracle = _repo_file(annotations.path, label="oracle annotation artifact")
    if oracle_path.resolve() != expected_oracle:
        raise ValueError("--oracle-artifact differs from the frozen manifest")

    verify_static_local_inputs(manifest, repo_root=REPO_ROOT)
    _require_pinned_file(
        scenarios_path,
        manifest.dataset.sha256,
        label="scenario dataset",
    )
    _require_pinned_file(
        oracle_path,
        annotations.sha256,
        label="oracle annotation artifact",
    )

    scenarios = load_scenarios(scenarios_path)
    if len(scenarios) != 10:
        raise ValueError("oracle ceiling dataset must contain exactly 10 scenarios")
    artifact = load_oracle_artifact(oracle_path, scenarios)

    pricing_path = _repo_file(manifest.model.pricing.path, label="pricing config")
    pricing_sha256 = validate_zero_api_pricing(
        pricing_path,
        manifest.model.snapshot,
    )
    if pricing_sha256 != manifest.model.pricing.sha256:
        raise ValueError("oracle pricing bytes differ from the frozen manifest")

    preflight_path = _repo_file(manifest.model.preflight.path, label="model preflight")
    if manifest.git_commit is None:
        raise ValueError("frozen oracle manifest is missing git_commit")
    validate_local_preflight_artifact(
        manifest.model.preflight.model_copy(update={"path": str(preflight_path)}),
        expected_git_commit=manifest.git_commit,
        expected_pricing_sha256=pricing_sha256,
        seed=101,
    )
    if command_runner is None:
        _verify_current_git_state(manifest.git_commit)
    else:
        _verify_current_git_state(manifest.git_commit, command_runner=command_runner)

    if manifest.decision_prompt_sha256 != _sha256_text(
        local_decision_prompt_contract()
    ):
        raise ValueError("oracle manifest decision prompt differs from runtime")
    if manifest.decision_schema_sha256 != _sha256_text(
        local_decision_schema_contract()
    ):
        raise ValueError("oracle manifest decision schema differs from runtime")
    if manifest.system_config_sha256 != {
        ORACLE_SYSTEM_NAME: _expected_oracle_system_hash(manifest)
    }:
        raise ValueError("oracle manifest system hash differs from runtime")
    return (
        manifest,
        scenarios,
        artifact,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _validate_oracle_task_and_dataset(
    log: EvalLog,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    oracle_path: Path,
    scenarios: list[Scenario],
) -> None:
    spec = log.eval
    if spec.task_registry_name != ORACLE_TASK_NAME:
        raise ValueError("unexpected oracle ceiling task")
    if spec.task_version != LOCAL_SCENARIO_TASK_VERSION:
        raise ValueError("oracle task version differs from runtime")
    if (
        spec.task_file is None
        or _resolve_logged_repo_path(spec.task_file)
        != (REPO_ROOT / LOCAL_TASK_FILE).resolve()
    ):
        raise ValueError("oracle task file differs from the frozen task")

    args = dict(spec.task_args or {})
    expected_args = {
        "seed",
        "repetition",
        "manifest",
        "ollama_models_dir",
        "oracle_annotations_path",
    }
    if set(args) != expected_args:
        raise ValueError("oracle task arguments are missing or contain extras")
    if args.get("seed") != 101 or args.get("repetition") != 1:
        raise ValueError("oracle task seed/repetition differs from the manifest")
    task_manifest = args.get("manifest")
    if (
        not isinstance(task_manifest, str)
        or _resolve_logged_repo_path(task_manifest) != manifest_path.resolve()
    ):
        raise ValueError("oracle task is not bound to this frozen manifest")
    _validate_path_task_arg(args, "ollama_models_dir")
    annotations_arg = args.get("oracle_annotations_path")
    if (
        not isinstance(annotations_arg, str)
        or _resolve_logged_repo_path(annotations_arg) != oracle_path.resolve()
    ):
        raise ValueError("oracle task is not bound to the pinned annotation artifact")

    annotations = manifest.oracle_annotations
    assert annotations is not None and annotations.sha256 is not None
    ids = [scenario.id for scenario in scenarios]
    required_metadata = {
        "track": "local_zero_api_cost",
        "claim_scope": "diagnostic_development_only",
        "hypothesis_test_eligible": False,
        "system": ORACLE_SYSTEM_NAME,
        "dataset": manifest.dataset.path,
        "dataset_split": "oracle_smoke",
        "dataset_scenario_count": 10,
        "dataset_sample_ids": ids,
        "canonical_dataset_sha256": dataset_sha256(scenarios),
        "repetition": 1,
        "manifest_sha256": manifest_sha256,
        "live_semantic_preflight_required": True,
        "provider_api_cost_usd": 0.0,
        "pricing_config_sha256": manifest.model.pricing.sha256,
        "electricity_measured": False,
        "decision_prompt_version": LOCAL_DECISION_VERSION,
        "compiler_mode": "oracle",
        "gold_assisted": True,
        "human_annotation_measured": False,
        "oracle_artifact_purpose": ORACLE_ARTIFACT_PURPOSE,
        "oracle_annotation_policy": ORACLE_ANNOTATION_POLICY,
        "oracle_compiler_version": ORACLE_COMPILER_VERSION,
        "oracle_annotations_path": annotations.path,
        "oracle_annotations_sha256": annotations.sha256,
        "oracle_token_scope": "decision_only_lower_bound",
        "same_model_for_compiler_and_decision": False,
        "scenario_compiler_model_calls": 0,
        "setup_preflight_includes_llm_compiler_call": True,
        "setup_preflight_compiler_used_in_scenarios": False,
    }
    metadata = dict(spec.metadata or {})
    if metadata != required_metadata:
        changed = sorted(
            key
            for key in set(metadata) | set(required_metadata)
            if metadata.get(key) != required_metadata.get(key)
        )
        raise ValueError(f"oracle task metadata differs: {changed}")

    dataset = spec.dataset
    if dataset.name != ORACLE_DATASET_NAME:
        raise ValueError("oracle log dataset name differs from protocol")
    if dataset.location is None or scenarios_path.resolve() not in (
        _logged_dataset_candidates(dataset.location)
    ):
        raise ValueError("oracle log dataset location differs from --scenarios")
    if dataset.samples != 10:
        raise ValueError("oracle log dataset must contain exactly 10 samples")
    if tuple(str(item) for item in (dataset.sample_ids or [])) != tuple(ids):
        raise ValueError("oracle log sample IDs/order differ from the dataset")
    if dataset.shuffled is not False:
        raise ValueError("oracle smoke dataset must be unshuffled")


def _oracle_zero_usage() -> Usage:
    return Usage(cost_usd=0.0)


def _validate_oracle_checkpoint(
    *,
    checkpoint: Any,
    authored_event: Any,
    frozen_delta_json: str | None,
) -> None:
    if checkpoint.event_id != authored_event.id or checkpoint.at != authored_event.at:
        raise ValueError("oracle checkpoint differs from the authored event")
    if checkpoint.state_sha256 is None:
        raise ValueError("oracle checkpoint is missing a deterministic state hash")
    if checkpoint.decision_parse_error:
        raise ValueError("oracle decision checkpoint has a parse error")

    if authored_event.kind == "clock_tick":
        if (
            checkpoint.compiler_called
            or checkpoint.raw_compiler_output is not None
            or checkpoint.memory_delta_json is not None
            or checkpoint.memory_delta_accepted is not None
            or checkpoint.compiler_parse_error
            or checkpoint.compiler_usage != Usage()
            or not math.isclose(checkpoint.compiler_latency_ms, 0.0, abs_tol=1e-12)
        ):
            raise ValueError("clock checkpoint contains oracle compiler work")
        return

    if not checkpoint.compiler_called:
        raise ValueError("non-clock checkpoint omitted the oracle compiler record")
    if checkpoint.compiler_parse_error:
        raise ValueError("oracle compiler record has a parse error")
    if checkpoint.memory_delta_accepted is not True:
        raise ValueError("frozen oracle delta was rejected")
    if checkpoint.compiler_usage != _oracle_zero_usage():
        raise ValueError("oracle compiler usage must be exact zero-token zero-cost")
    if frozen_delta_json is None:
        raise ValueError("oracle annotation record is missing")
    if checkpoint.raw_compiler_output != frozen_delta_json:
        raise ValueError("raw oracle compiler output differs from frozen annotations")
    if checkpoint.memory_delta_json != frozen_delta_json:
        raise ValueError("accepted memory delta differs from frozen annotations")


def _validate_oracle_run_and_events(
    *,
    sample: Any,
    scenario: Scenario,
    run: ScenarioRun,
    manifest: LocalExperimentManifest,
    manifest_sha256: str,
    artifact: OracleCompilerArtifact,
    first_sample: bool,
    expected_preflight: LocalModelPreflightResult,
) -> tuple[Usage, Usage]:
    if (
        run.system != ORACLE_SYSTEM_NAME
        or run.repetition != 1
        or run.seed != 101
        or run.model != manifest.model.snapshot
        or run.prompt_version != LOCAL_DECISION_VERSION
        or run.scenario_sha256 != canonical_sha256(scenario)
        or run.prompt_sha256 != _sha256_text(local_decision_contract())
        or run.system_config_sha256 != manifest.system_config_sha256[ORACLE_SYSTEM_NAME]
        or run.manifest_sha256 != manifest_sha256
        or run.pricing_config_sha256 != manifest.model.pricing.sha256
    ):
        raise ValueError("oracle ScenarioRun identity or contract binding differs")
    if run.hosted_warmup is not None:
        raise ValueError("local oracle run cannot contain hosted warmup evidence")
    if not run.usage_complete or not run.cost_complete:
        raise ValueError("oracle ScenarioRun has incomplete accounting")
    if run.decision_parse_errors or run.compiler_parse_errors or run.parse_errors:
        raise ValueError("oracle ScenarioRun contains parse errors")
    if run.compiler_usage != _oracle_zero_usage():
        raise ValueError("oracle ScenarioRun compiler usage is not exact zero")
    if run.decision_usage.input_tokens <= 0 or run.decision_usage.output_tokens <= 0:
        raise ValueError("oracle ScenarioRun has no measured decision usage")
    if run.decision_usage.cost_usd != 0.0:
        raise ValueError("oracle decision usage is not zero provider cost")
    if run.usage != run.decision_usage.plus(run.compiler_usage):
        raise ValueError("oracle headline usage differs from decision plus oracle zero")
    if run.usage.embedding_inputs or run.usage.embedding_characters:
        raise ValueError("oracle run cannot contain embedding usage")

    if len(run.checkpoints) != len(scenario.events):
        raise ValueError("oracle checkpoint count differs from the dataset")
    if len(run.checkpoint_latency_ms) != len(scenario.events):
        raise ValueError("oracle checkpoint latency count differs")

    records = iter(artifact.records_for(scenario.to_runtime()))
    for authored_event, checkpoint in zip(
        scenario.events,
        run.checkpoints,
        strict=True,
    ):
        frozen_delta_json = None
        if authored_event.kind != "clock_tick":
            record = next(records)
            if record.event_id != authored_event.id:
                raise ValueError("oracle record order differs from the checkpoint")
            frozen_delta_json = record.delta.model_dump_json()
        _validate_oracle_checkpoint(
            checkpoint=checkpoint,
            authored_event=authored_event,
            frozen_delta_json=frozen_delta_json,
        )
    try:
        next(records)
    except StopIteration:
        pass
    else:
        raise ValueError("oracle artifact contains unconsumed scenario records")

    raw_events = [event for event in sample.events if isinstance(event, ModelEvent)]
    expected_model_events = len(scenario.events) + (2 if first_sample else 0)
    if len(raw_events) != expected_model_events:
        raise ValueError(
            "oracle sample must contain two first-sample setup ModelEvents and "
            "exactly one decision ModelEvent per checkpoint"
        )

    cursor = 0
    raw_usage = _oracle_zero_usage()
    if first_sample:
        raw_usage = raw_usage.plus(
            _validate_live_preflight(raw_events[:2], expected_preflight)
        )
        cursor = 2
        if run.setup_latency_ms < expected_preflight.setup_latency_ms:
            raise ValueError("first oracle run omits setup preflight latency")
    elif not math.isclose(run.setup_latency_ms, 0.0, abs_tol=1e-12):
        raise ValueError("oracle setup latency is allowed only on the first sample")

    raw_predictions: list[PredictedAction] = []
    decision_usage = _oracle_zero_usage()
    for authored_event, checkpoint in zip(
        scenario.events,
        run.checkpoints,
        strict=True,
    ):
        usage, predictions, _ = _parse_decision_event(
            raw_events[cursor],
            checkpoint=checkpoint,
            authored_event=authored_event,
        )
        cursor += 1
        raw_usage = raw_usage.plus(usage)
        decision_usage = decision_usage.plus(usage)
        raw_predictions.extend(predictions)
    if cursor != len(raw_events):
        raise ValueError("oracle sample contains unaccounted ModelEvents")
    if decision_usage != run.decision_usage:
        raise ValueError("oracle decision usage differs from raw ModelEvents")
    if raw_predictions != run.predictions:
        raise ValueError("oracle predictions differ from raw decision outputs")
    return raw_usage, run.usage


def _validate_oracle_log(
    log: EvalLog,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    oracle_path: Path,
    scenarios: list[Scenario],
    artifact: OracleCompilerArtifact,
) -> list[ScenarioRun]:
    _validate_effective_log_policy(log, manifest)
    _validate_oracle_task_and_dataset(
        log,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        scenarios_path=scenarios_path,
        oracle_path=oracle_path,
        scenarios=scenarios,
    )
    if log.samples is None or len(log.samples) != 10:
        raise ValueError("oracle ceiling requires exactly 10 sample records")
    scenario_ids = [scenario.id for scenario in scenarios]
    if [str(sample.id) for sample in log.samples] != scenario_ids:
        raise ValueError("oracle sample records differ from dataset ID/order")

    serialized_preflights = {
        json.dumps(sample.store.get(LOCAL_PREFLIGHT_STORE_KEY), sort_keys=True)
        for sample in log.samples
    }
    if len(serialized_preflights) != 1 or "null" in serialized_preflights:
        raise ValueError("oracle samples disagree about setup preflight evidence")
    preflight = LocalModelPreflightResult.model_validate_json(
        next(iter(serialized_preflights))
    )

    by_id = {scenario.id: scenario for scenario in scenarios}
    raw_log_usage = _oracle_zero_usage()
    headline_usage = _oracle_zero_usage()
    runs: list[ScenarioRun] = []
    for index, sample in enumerate(log.samples):
        run = _extract_scenario_run(sample)
        scenario = by_id.get(run.scenario_id)
        if scenario is None:
            raise ValueError("oracle ScenarioRun does not match the smoke dataset")
        raw_usage, measured_usage = _validate_oracle_run_and_events(
            sample=sample,
            scenario=scenario,
            run=run,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            artifact=artifact,
            first_sample=index == 0,
            expected_preflight=preflight,
        )
        raw_log_usage = raw_log_usage.plus(raw_usage)
        headline_usage = headline_usage.plus(measured_usage)
        runs.append(run)

    expected_raw = headline_usage.plus(
        preflight.compiler_usage.plus(preflight.decision_usage)
    )
    if raw_log_usage != expected_raw:
        raise ValueError("oracle raw calls differ from headline plus setup preflight")
    _validate_log_usage(log, raw_log_usage)
    keys = [(run.scenario_id, run.system, run.repetition, run.model) for run in runs]
    if len(keys) != 10 or len(keys) != len(set(keys)):
        raise ValueError("oracle ceiling requires 10 unique ScenarioRuns")
    return runs


def _load_oracle_runs(
    path: Path,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    oracle_path: Path,
    scenarios: list[Scenario],
    artifact: OracleCompilerArtifact,
) -> list[ScenarioRun]:
    if path.suffix != ".eval" or not path.is_file():
        raise ValueError("oracle ceiling requires exactly one Inspect .eval file")
    log = read_eval_log(path, resolve_attachments=True)
    return _validate_oracle_log(
        log,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        scenarios_path=scenarios_path,
        oracle_path=oracle_path,
        scenarios=scenarios,
        artifact=artifact,
    )


ORACLE_CSV_FIELDS = (
    "title",
    "hypothesis_test_eligible",
    "compiler_mode",
    "gold_assisted",
    "human_annotation_measured",
    "human_annotation_effort_measured",
    "oracle_token_scope",
    "system",
    "repetition",
    "model",
    "scenarios",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "false_reminders",
    "false_alarm_checkpoints",
    "false_alarm_rate",
    "obsolete_errors",
    "provenance_exact_accuracy",
    "invalid_outputs",
    "input_tokens",
    "output_tokens",
    "decision_input_tokens",
    "oracle_compiler_input_tokens",
    "oracle_compiler_output_tokens",
    "oracle_compiler_provider_api_cost_usd",
    "provider_api_cost_usd",
    "usage_complete",
    "cost_complete",
    "latency_p50_ms",
    "latency_p95_ms",
    "decision_latency_ms",
    "oracle_replay_latency_ms",
    "local_latency_ms",
    "setup_latency_ms",
)


def _result_row(result: AggregateResult) -> dict[str, object]:
    return {
        "title": ORACLE_CEILING_TITLE,
        "hypothesis_test_eligible": False,
        "compiler_mode": "oracle",
        "gold_assisted": True,
        "human_annotation_measured": False,
        "human_annotation_effort_measured": False,
        "oracle_token_scope": "decision_only_lower_bound",
        "system": result.system,
        "repetition": result.repetition,
        "model": result.model,
        "scenarios": result.scenarios,
        "tp": result.tp,
        "fp": result.fp,
        "fn": result.fn,
        "precision": round(result.precision, 6),
        "recall": round(result.recall, 6),
        "f1": round(result.f1, 6),
        "false_reminders": result.false_reminders,
        "false_alarm_checkpoints": result.false_alarm_checkpoints,
        "false_alarm_rate": (
            "" if result.false_alarm_rate is None else round(result.false_alarm_rate, 6)
        ),
        "obsolete_errors": result.obsolete_errors,
        "provenance_exact_accuracy": (
            ""
            if result.provenance_exact_accuracy is None
            else round(result.provenance_exact_accuracy, 6)
        ),
        "invalid_outputs": result.invalid_outputs,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "decision_input_tokens": result.decision_input_tokens,
        "oracle_compiler_input_tokens": 0,
        "oracle_compiler_output_tokens": 0,
        "oracle_compiler_provider_api_cost_usd": "0.000000",
        "provider_api_cost_usd": f"{result.cost_usd:.6f}",
        "usage_complete": result.usage_complete,
        "cost_complete": result.cost_complete,
        "latency_p50_ms": round(result.latency_p50_ms, 3),
        "latency_p95_ms": round(result.latency_p95_ms, 3),
        "decision_latency_ms": round(result.decision_latency_ms, 3),
        "oracle_replay_latency_ms": round(result.compiler_latency_ms, 3),
        "local_latency_ms": round(result.local_latency_ms, 3),
        "setup_latency_ms": round(result.setup_latency_ms, 3),
    }


def _write_csv(path: Path, result: AggregateResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ORACLE_CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(_result_row(result))


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _render_markdown(result: AggregateResult) -> str:
    if result.cost_usd != 0.0 or result.compiler_input_tokens != 0:
        raise ValueError("oracle aggregate violates zero-cost compiler accounting")
    return (
        f"# {ORACLE_CEILING_TITLE}\n\n"
        "This is a gold-assisted diagnostic ceiling for the frozen oracle "
        "compiler. It is not a headline Anamnesis result, is not "
        "hypothesis-test eligible, and has "
        "no success gate or baseline comparison.\n\n"
        "| System | TP | FP | FN | Precision | Recall | F1 | False reminders | "
        "FAR | Obsolete | Provenance | Decision input tokens | Oracle compiler "
        "tokens in/out | Provider API cost USD | Latency p50/p95 ms | Setup ms |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|\n"
        f"| {result.system} | {result.tp} | {result.fp} | {result.fn} | "
        f"{_percent(result.precision)} | {_percent(result.recall)} | "
        f"{_percent(result.f1)} | {result.false_reminders} | "
        f"{_percent(result.false_alarm_rate)} | {result.obsolete_errors} | "
        f"{_percent(result.provenance_exact_accuracy)} | "
        f"{result.decision_input_tokens} | 0/0 | {result.cost_usd:.6f} | "
        f"{result.latency_p50_ms:.1f}/{result.latency_p95_ms:.1f} | "
        f"{result.setup_latency_ms:.1f} |\n\n"
        "Scenario compiler tokens and provider API cost are exactly zero because "
        "the compiler replays frozen oracle annotations locally. Human annotation "
        "effort is unmeasured, so the reported token scope is a decision-only "
        "lower bound. Electricity and hardware cost are also unmeasured. "
        "The two setup preflight model calls are excluded from headline usage; "
        "setup latency is reported separately.\n"
    )


def _validate_provenance_locations(
    *,
    manifest_path: Path,
    scenarios_path: Path,
    oracle_path: Path,
    run_path: Path,
    csv_path: Path,
    markdown_path: Path,
    provenance_path: Path,
) -> None:
    sources = [manifest_path, scenarios_path, oracle_path, run_path]
    targets = [csv_path, markdown_path, provenance_path]
    for index, source in enumerate(sources, start=1):
        _repo_relative(source, label=f"oracle provenance source {index}")
    for index, target in enumerate(targets, start=1):
        _repo_relative(
            target,
            label=f"oracle result artifact {index}",
            must_exist=False,
        )
    resolved_targets = [path.resolve() for path in targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("oracle result artifact paths must be distinct")
    if {path.resolve() for path in sources}.intersection(resolved_targets):
        raise ValueError("oracle result artifacts cannot overwrite input artifacts")


def _write_provenance(
    path: Path,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    oracle_path: Path,
    artifact: OracleCompilerArtifact,
    run_path: Path,
    expected_run_sha256: str,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    _validate_provenance_locations(
        manifest_path=manifest_path,
        scenarios_path=scenarios_path,
        oracle_path=oracle_path,
        run_path=run_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
        provenance_path=path,
    )
    if manifest.git_commit is None or manifest.oracle_annotations is None:
        raise ValueError("frozen oracle manifest is incomplete")
    if _sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("frozen oracle manifest changed while reporting")
    dataset_digest = _require_pinned_file(
        scenarios_path,
        manifest.dataset.sha256,
        label="scenario dataset",
    )
    oracle_digest = _require_pinned_file(
        oracle_path,
        manifest.oracle_annotations.sha256,
        label="oracle annotation artifact",
    )
    run_digest = _sha256_file(run_path)
    if run_digest != expected_run_sha256:
        raise ValueError("oracle .eval log changed while reporting")

    pricing_path = _repo_file(manifest.model.pricing.path, label="pricing config")
    pricing_digest = _require_pinned_file(
        pricing_path,
        manifest.model.pricing.sha256,
        label="pricing config",
    )
    preflight_path = _repo_file(manifest.model.preflight.path, label="model preflight")
    preflight_digest = _require_pinned_file(
        preflight_path,
        manifest.model.preflight.sha256,
        label="model preflight",
    )
    payload = {
        "schema_version": 1,
        "artifact": "anamnesis_local_oracle_ceiling_result_provenance",
        "title": ORACLE_CEILING_TITLE,
        "hypothesis_test_eligible": False,
        "compiler_mode": "oracle",
        "gold_assisted": True,
        "human_annotation_measured": False,
        "human_annotation_effort_measured": False,
        "oracle_token_scope": "decision_only_lower_bound",
        "source_git_commit": manifest.git_commit,
        "frozen_manifest": {
            "path": _repo_relative(manifest_path, label="frozen oracle manifest"),
            "sha256": manifest_sha256,
        },
        "scenario_dataset": {
            "path": _repo_relative(scenarios_path, label="scenario dataset"),
            "sha256": dataset_digest,
            "canonical_sha256": dataset_sha256(load_scenarios(scenarios_path)),
        },
        "oracle_annotations": {
            "path": _repo_relative(oracle_path, label="oracle annotations"),
            "sha256": oracle_digest,
            "canonical_semantics_sha256": oracle_artifact_sha256(artifact),
            "annotation_policy": ORACLE_ANNOTATION_POLICY,
        },
        "model_preflight": {
            "path": _repo_relative(preflight_path, label="model preflight"),
            "sha256": preflight_digest,
        },
        "pricing_config": {
            "path": _repo_relative(pricing_path, label="pricing config"),
            "sha256": pricing_digest,
        },
        "input_eval_log": {
            "path": _repo_relative(run_path, label="oracle input .eval log"),
            "sha256": run_digest,
        },
        "accounting_scope": {
            "setup_model_events": 2,
            "scenario_compiler_model_events": 0,
            "oracle_compiler_input_tokens": 0,
            "oracle_compiler_output_tokens": 0,
            "oracle_compiler_provider_api_cost_usd": 0.0,
            "human_annotation_effort_measured": False,
            "oracle_token_scope": "decision_only_lower_bound",
            "setup_usage_in_headline": False,
        },
        "outputs": {
            "csv": {
                "path": _repo_relative(csv_path, label="oracle CSV output"),
                "sha256": _sha256_file(csv_path),
            },
            "markdown": {
                "path": _repo_relative(markdown_path, label="oracle Markdown output"),
                "sha256": _sha256_file(markdown_path),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def oracle_report_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and score the strict local oracle-compiler ceiling"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("eval/scenarios/smoke.jsonl"),
    )
    parser.add_argument(
        "--oracle-artifact",
        type=Path,
        default=Path("eval/oracle/smoke_memory_deltas.v1.json"),
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=ORACLE_CSV_PATH)
    parser.add_argument("--markdown", type=Path, default=ORACLE_MARKDOWN_PATH)
    parser.add_argument("--provenance", type=Path, default=ORACLE_PROVENANCE_PATH)
    args = parser.parse_args(argv)

    _validate_provenance_locations(
        manifest_path=args.manifest,
        scenarios_path=args.scenarios,
        oracle_path=args.oracle_artifact,
        run_path=args.run,
        csv_path=args.csv,
        markdown_path=args.markdown,
        provenance_path=args.provenance,
    )
    run_sha256 = _sha256_file(args.run)
    manifest, scenarios, artifact, manifest_sha256 = _validate_frozen_oracle_manifest(
        manifest_path=args.manifest,
        scenarios_path=args.scenarios,
        oracle_path=args.oracle_artifact,
    )
    runs = _load_oracle_runs(
        args.run,
        manifest=manifest,
        manifest_path=args.manifest,
        manifest_sha256=manifest_sha256,
        scenarios_path=args.scenarios,
        oracle_path=args.oracle_artifact,
        scenarios=scenarios,
        artifact=artifact,
    )
    by_id = {scenario.id: scenario for scenario in scenarios}
    scored = [(score_scenario(by_id[run.scenario_id], run), run) for run in runs]
    results = aggregate_results(scored)
    if len(results) != 1:
        raise ValueError("oracle ceiling must aggregate to exactly one row")
    result = results[0]
    if (
        result.system != ORACLE_SYSTEM_NAME
        or result.scenarios != 10
        or result.repetition != 1
        or result.model != LOCAL_MODEL_ID
        or result.compiler_input_tokens != 0
        or result.embedding_inputs != 0
        or result.embedding_characters != 0
        or not result.usage_complete
        or not result.cost_complete
        or result.cost_usd != 0.0
    ):
        raise ValueError("oracle aggregate violates the diagnostic matrix")

    _write_csv(args.csv, result)
    rendered = _render_markdown(result)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(rendered, encoding="utf-8")
    _write_provenance(
        args.provenance,
        manifest=manifest,
        manifest_path=args.manifest,
        manifest_sha256=manifest_sha256,
        scenarios_path=args.scenarios,
        oracle_path=args.oracle_artifact,
        artifact=artifact,
        run_path=args.run,
        expected_run_sha256=run_sha256,
        csv_path=args.csv,
        markdown_path=args.markdown,
    )
    print(rendered, end="")
    return 0


__all__ = [
    "ORACLE_CEILING_TITLE",
    "ORACLE_PROVENANCE_PATH",
    "oracle_report_main",
]
