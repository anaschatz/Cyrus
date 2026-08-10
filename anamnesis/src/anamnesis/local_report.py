"""Strict reporting for the isolated 10-scenario local smoke diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, effective_eval_config, read_eval_log
from inspect_ai.model import ChatMessageUser, ModelUsage, ResponseSchema
from pydantic import ValidationError

from anamnesis.inspect_adapter import SCENARIO_RUN_STORE_KEY
from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.local_experiment import (
    LOCAL_BASE_URL,
    LOCAL_MODEL_ID,
    LocalExperimentManifest,
    validate_zero_api_pricing,
    verify_static_local_inputs,
)
from anamnesis.local_preflight import (
    _validate_local_model_event,
    local_preflight_prompts,
    validate_local_preflight_artifact,
)
from anamnesis.local_runtime import (
    LOCAL_DECISION_VERSION,
    LOCAL_PREFLIGHT_STORE_KEY,
    LOCAL_SCENARIO_TASK_VERSION,
    LocalDecisionWire,
    LocalModelPreflightResult,
    _compiler_preflight_semantics,
    _local_decision_schema,
    _local_memory_delta_schema,
    local_decision_contract,
    local_decision_prompt_contract,
    local_decision_schema_contract,
    local_memory_compiler_prompt_contract,
    local_memory_compiler_schema_contract,
    local_system_config_sha256,
)
from anamnesis.local_wire import LocalMemoryDeltaWire
from anamnesis.memory import CompilerCall
from anamnesis.schema import Decision, PredictedAction, Scenario, ScenarioRun, Usage
from anamnesis.scoring import AggregateResult, aggregate_results, score_scenario

LOCAL_SMOKE_TITLE = "Local smoke diagnostic — not a hypothesis test"
LOCAL_SMOKE_PROVENANCE_PATH = Path("results/local_smoke.provenance.json")
LOCAL_TASK_FILE = Path("eval/anamnesis_local_eval.py")
LOCAL_SMOKE_DATASET_NAME = "anamnesis-local-smoke-v0"
LOCAL_SYSTEM_TASKS = {
    "no_memory": "local_no_memory",
    "full_context": "local_full_context",
    "vector_rag": "local_vector_rag",
    "anamnesis": "local_anamnesis",
}


def _discover_repo_root(start: Path | None = None) -> Path:
    """Find the checkout root even when this module is installed non-editably."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "RESEARCH.md").is_file()
            and (candidate / "eval").is_dir()
        ):
            return candidate
    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / "pyproject.toml").is_file() and (
        source_candidate / "RESEARCH.md"
    ).is_file():
        return source_candidate
    raise RuntimeError("run anamnesis-local-report from inside the repository")


REPO_ROOT = _discover_repo_root()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_path(
    path: Path,
    *,
    label: str,
    must_exist: bool = True,
) -> str:
    """Return a non-identifying repository path or reject external artifacts."""

    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(
            f"{label} must be inside the repository; external paths are not "
            "recorded in the provenance sidecar"
        )
    if must_exist and not resolved.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")
    return resolved.relative_to(REPO_ROOT).as_posix()


def _repo_artifact(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("local manifest artifact paths must be repo-relative")
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT) or not resolved.is_file():
        raise ValueError(f"local manifest artifact does not exist: {relative}")
    return resolved


def _resolve_logged_repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _logged_dataset_candidates(value: str) -> set[Path]:
    """Resolve both repository-relative and Inspect task-relative locations."""

    path = Path(value)
    if path.is_absolute():
        return {path.resolve()}
    if ".." in path.parts:
        raise ValueError("local log dataset location cannot escape the repository")
    candidates = {
        (REPO_ROOT / path).resolve(),
        ((REPO_ROOT / LOCAL_TASK_FILE).parent / path).resolve(),
    }
    if any(not candidate.is_relative_to(REPO_ROOT) for candidate in candidates):
        raise ValueError("local log dataset location cannot escape the repository")
    return candidates


def _verify_current_git_state(
    expected_commit: str | None,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
) -> None:
    if expected_commit is None:
        raise ValueError("frozen local manifest is missing git_commit")
    current = command_runner(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != expected_commit:
        raise ValueError("local manifest git_commit differs from current HEAD")
    dirty = command_runner(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("strict local reports require a clean git worktree")


def _expected_system_hashes(
    manifest: LocalExperimentManifest,
) -> dict[str, str]:
    return {
        system: local_system_config_sha256(
            system=system,  # type: ignore[arg-type]
            top_k=manifest.embedding.top_k,
            embedding_model=manifest.embedding.model,
            embedding_repository=manifest.embedding.repository,
            embedding_revision=manifest.embedding.revision,
            embedding_artifact_sha256=(
                manifest.embedding.artifact_sha256 if system == "vector_rag" else None
            ),
            pricing_config_sha256=manifest.model.pricing.sha256,
        )
        for system in LOCAL_SYSTEM_TASKS
    }


def _validate_frozen_local_manifest(
    *,
    manifest_path: Path,
    scenarios_path: Path,
) -> tuple[LocalExperimentManifest, list[Scenario], str]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = LocalExperimentManifest.model_validate_json(manifest_bytes)
    if manifest.status != "frozen" or manifest.phase != "smoke":
        raise ValueError("local smoke reports require a frozen smoke manifest")
    if manifest.hypothesis_test_eligible is not False:
        raise ValueError("local smoke manifest cannot be hypothesis-test eligible")
    if set(manifest.systems) != set(LOCAL_SYSTEM_TASKS):
        raise ValueError("local smoke manifest must contain all four systems exactly")

    expected_dataset_path = _repo_artifact(manifest.dataset.path)
    if scenarios_path.resolve() != expected_dataset_path:
        raise ValueError("--scenarios differs from the frozen local dataset")
    scenarios = load_scenarios(scenarios_path)
    if len(scenarios) != 10 or manifest.scenario_count != 10:
        raise ValueError("local smoke reports require exactly 10 scenarios")

    verify_static_local_inputs(manifest, repo_root=REPO_ROOT)
    pricing_path = _repo_artifact(manifest.model.pricing.path)
    pricing_sha256 = validate_zero_api_pricing(pricing_path, manifest.model.snapshot)
    if pricing_sha256 != manifest.model.pricing.sha256:
        raise ValueError("local zero-pricing bytes differ from the frozen manifest")

    preflight_path = _repo_artifact(manifest.model.preflight.path)
    if manifest.git_commit is None:
        raise ValueError("frozen local manifest is missing git_commit")
    validate_local_preflight_artifact(
        manifest.model.preflight.model_copy(update={"path": str(preflight_path)}),
        expected_git_commit=manifest.git_commit,
        expected_pricing_sha256=pricing_sha256,
        seed=manifest.execution.seeds[0],
    )
    _verify_current_git_state(manifest.git_commit)

    current_contracts = {
        "decision_prompt_sha256": _sha256_text(local_decision_prompt_contract()),
        "decision_schema_sha256": _sha256_text(local_decision_schema_contract()),
        "memory_compiler_prompt_sha256": _sha256_text(
            local_memory_compiler_prompt_contract()
        ),
        "memory_compiler_schema_sha256": _sha256_text(
            local_memory_compiler_schema_contract()
        ),
    }
    for field_name, expected in current_contracts.items():
        if getattr(manifest, field_name) != expected:
            raise ValueError(f"local manifest {field_name} differs from runtime")
    if manifest.system_config_sha256 != _expected_system_hashes(manifest):
        raise ValueError("local manifest system hashes differ from runtime")
    return manifest, scenarios, hashlib.sha256(manifest_bytes).hexdigest()


def _validate_effective_log_policy(
    log: EvalLog, manifest: LocalExperimentManifest
) -> None:
    if log.status != "success":
        raise ValueError("local smoke Inspect logs must have status='success'")
    if log.invalidated:
        raise ValueError("invalidated local Inspect logs cannot be reported")
    if log.config_updates:
        raise ValueError("local smoke logs cannot contain config updates")
    if log.log_updates:
        raise ValueError("edited local smoke logs cannot be reported")
    spec = log.eval
    if spec.model != manifest.model.snapshot:
        raise ValueError("local log model differs from the frozen manifest")
    if str(spec.model_base_url).rstrip("/") != LOCAL_BASE_URL:
        raise ValueError("local log model base URL differs from the loopback pin")
    if spec.model_args not in ({}, None):
        raise ValueError("local log contains unpinned model arguments")

    generation = spec.model_generate_config.merge(log.plan.config)
    effective_generation = generation.model_dump(mode="json", exclude_none=True)
    expected_generation = {
        "max_retries": 0,
        "max_connections": 1,
        "adaptive_connections": False,
        "temperature": 0.0,
        "seed": 101,
        "cache": False,
    }
    if effective_generation != expected_generation:
        raise ValueError("local log contains unpinned generation configuration")
    config = effective_eval_config(log)
    if (
        config.max_samples != 1
        or config.max_tasks != 1
        or config.epochs != 1
        or config.log_model_api is not True
    ):
        raise ValueError("local log has an invalid sample/task/logging policy")
    revision = spec.revision
    if (
        revision is None
        or len(revision.commit or "") < 7
        or any(
            character not in "0123456789abcdef" for character in (revision.commit or "")
        )
        or manifest.git_commit is None
        or not manifest.git_commit.startswith(revision.commit or "")
        or revision.dirty is not False
    ):
        raise ValueError("local log was not produced from the frozen clean commit")


def _validate_path_task_arg(task_args: dict[str, Any], name: str) -> None:
    value = task_args.get(name)
    if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
        raise ValueError(f"local task requires an absolute {name}")


def _validate_task_and_dataset(
    log: EvalLog,
    *,
    system: str,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    scenarios: list[Scenario],
) -> None:
    spec = log.eval
    expected_task = LOCAL_SYSTEM_TASKS[system]
    if spec.task_registry_name != expected_task:
        raise ValueError(f"unexpected local task: {spec.task_registry_name}")
    if spec.task_version != LOCAL_SCENARIO_TASK_VERSION:
        raise ValueError("local task version differs from the runtime contract")
    if (
        spec.task_file is None
        or _resolve_logged_repo_path(spec.task_file)
        != (REPO_ROOT / LOCAL_TASK_FILE).resolve()
    ):
        raise ValueError("local log task file is not eval/anamnesis_local_eval.py")

    task_args = dict(spec.task_args or {})
    allowed = {"seed", "repetition", "manifest", "ollama_models_dir"}
    if system == "vector_rag":
        allowed.add("embedding_snapshot_path")
    if set(task_args) != allowed:
        raise ValueError("local task arguments are missing or contain extra fields")
    if task_args.get("seed") != 101 or task_args.get("repetition") != 1:
        raise ValueError("local task seed/repetition differs from the smoke matrix")
    task_manifest = task_args.get("manifest")
    if (
        not isinstance(task_manifest, str)
        or _resolve_logged_repo_path(task_manifest) != manifest_path.resolve()
    ):
        raise ValueError("local task was not bound to this frozen manifest")
    _validate_path_task_arg(task_args, "ollama_models_dir")
    if system == "vector_rag":
        _validate_path_task_arg(task_args, "embedding_snapshot_path")

    ids = [scenario.id for scenario in scenarios]
    required_metadata = {
        "track": "local_zero_api_cost",
        "claim_scope": "diagnostic_development_only",
        "hypothesis_test_eligible": False,
        "system": system,
        "dataset": manifest.dataset.path,
        "dataset_split": "smoke",
        "dataset_scenario_count": 10,
        "dataset_sample_ids": ids,
        "canonical_dataset_sha256": dataset_sha256(scenarios),
        "repetition": 1,
        "manifest_sha256": manifest_sha256,
        "live_semantic_preflight_required": True,
        "provider_api_cost_usd": 0.0,
        "electricity_measured": False,
        "decision_prompt_version": LOCAL_DECISION_VERSION,
        "pricing_config_sha256": manifest.model.pricing.sha256,
    }
    metadata = dict(spec.metadata or {})
    for name, expected in required_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(f"local task metadata differs for {name}")

    dataset = spec.dataset
    if dataset.name != LOCAL_SMOKE_DATASET_NAME:
        raise ValueError("local log dataset name differs from the smoke protocol")
    if dataset.location is None or scenarios_path.resolve() not in (
        _logged_dataset_candidates(dataset.location)
    ):
        raise ValueError("local log dataset location differs from --scenarios")
    if dataset.samples != 10:
        raise ValueError("local log dataset does not contain exactly 10 samples")
    if tuple(str(item) for item in (dataset.sample_ids or [])) != tuple(ids):
        raise ValueError("local log dataset IDs/order differ from --scenarios")
    if dataset.shuffled is not False:
        raise ValueError("local smoke dataset must be unshuffled")


def _usage_from_model_usage(model_usage: ModelUsage) -> Usage:
    if model_usage.input_tokens <= 0 or model_usage.output_tokens <= 0:
        raise ValueError("local ModelEvent must report positive token usage")
    if model_usage.total_cost != 0.0:
        raise ValueError("local ModelEvent must report complete zero provider cost")
    cache_read = model_usage.input_tokens_cache_read or 0
    cache_write = model_usage.input_tokens_cache_write or 0
    return Usage(
        input_tokens=model_usage.input_tokens + cache_read + cache_write,
        uncached_input_tokens=model_usage.input_tokens,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        output_tokens=model_usage.output_tokens,
        cost_usd=0.0,
    )


def _validate_model_event(
    event: ModelEvent,
    *,
    schema: ResponseSchema,
    seed: int,
    expected_prompt_sha256: str | None = None,
) -> tuple[str, Usage]:
    if len(event.input) != 1 or not isinstance(event.input[0], ChatMessageUser):
        raise ValueError("local ModelEvent has no single user prompt")
    prompt = event.input[0].content
    if not isinstance(prompt, str):
        raise ValueError("local ModelEvent prompt is not plain text")
    if expected_prompt_sha256 is not None and _sha256_text(prompt) != (
        expected_prompt_sha256
    ):
        raise ValueError("raw local prompt differs from checkpoint context hash")
    _validate_local_model_event(
        event,
        prompt=prompt,
        schema=schema,
        seed=seed,
    )
    if event.output.usage is None:
        raise ValueError("local ModelEvent is missing usage")
    return prompt, _usage_from_model_usage(event.output.usage)


def _validate_live_preflight(
    events: list[ModelEvent], result: LocalModelPreflightResult
) -> Usage:
    if len(events) != 2:
        raise ValueError("live local preflight requires exactly two model calls")
    if (
        not result.passed
        or result.loaded_model is None
        or not result.same_model_for_compiler_and_decision
        or result.compiler_parse_error
        or result.decision_parse_error
        or not result.compiler_semantic_valid
        or not result.decision_semantic_valid
        or not result.compiler_usage_complete
        or not result.decision_usage_complete
        or not result.compiler_cost_complete
        or not result.decision_cost_complete
    ):
        raise ValueError("live local semantic preflight did not fully pass")
    compiler_prompt, decision_prompt = local_preflight_prompts()
    prompt, compiler_usage = _validate_model_event(
        events[0],
        schema=_local_memory_delta_schema(LOCAL_MODEL_ID),
        seed=101,
    )
    if prompt != compiler_prompt or events[0].output.completion is None:
        raise ValueError("live compiler preflight prompt/output differs from protocol")
    prompt, decision_usage = _validate_model_event(
        events[1],
        schema=_local_decision_schema(LOCAL_MODEL_ID),
        seed=101,
    )
    if prompt != decision_prompt or events[1].output.completion is None:
        raise ValueError("live decision preflight prompt/output differs from protocol")
    try:
        delta = LocalMemoryDeltaWire.model_validate_json(
            events[0].output.completion
        ).to_domain()
    except (ValidationError, ValueError) as error:
        raise ValueError("live compiler preflight output is invalid") from error
    if not _compiler_preflight_semantics(
        CompilerCall(
            delta=delta,
            usage=compiler_usage,
            raw_completion=events[0].output.completion,
        )
    ):
        raise ValueError("live compiler preflight semantics are invalid")
    try:
        decision = LocalDecisionWire.model_validate_json(
            events[1].output.completion
        ).to_domain()
    except (ValidationError, ValueError) as error:
        raise ValueError("live decision preflight output is invalid") from error
    if decision.actions:
        raise ValueError("live decision preflight emitted a false reminder")
    if (
        result.compiler_usage != compiler_usage
        or result.decision_usage != decision_usage
    ):
        raise ValueError("live preflight result usage differs from raw events")
    return compiler_usage.plus(decision_usage)


def _parse_decision_event(
    event: ModelEvent,
    *,
    checkpoint: Any,
    authored_event: Any,
) -> tuple[Usage, list[PredictedAction], Decision]:
    _, usage = _validate_model_event(
        event,
        schema=_local_decision_schema(LOCAL_MODEL_ID),
        seed=101,
        expected_prompt_sha256=checkpoint.rendered_context_sha256,
    )
    if event.output.completion != checkpoint.raw_decision_output:
        raise ValueError("raw decision output differs from checkpoint audit")
    if usage != checkpoint.decision_usage:
        raise ValueError("raw decision usage differs from checkpoint audit")
    parse_error = False
    decision = Decision()
    try:
        decision = LocalDecisionWire.model_validate_json(
            event.output.completion
        ).to_domain()
    except (ValidationError, ValueError):
        parse_error = True
    if parse_error != checkpoint.decision_parse_error:
        raise ValueError("decision parse status differs from raw output")
    predictions = [
        PredictedAction(
            **action.model_dump(),
            emitted_at=authored_event.at,
            decision_event_id=authored_event.id,
        )
        for action in decision.actions
    ]
    return usage, predictions, decision


def _expected_vector_retrieval_usage(
    scenario: Scenario, decisions: list[Decision]
) -> tuple[int, int]:
    if len(decisions) != len(scenario.events):
        raise ValueError("vector retrieval decisions differ from checkpoint count")
    embedding_inputs = 0
    embedding_characters = 0
    for index, (event, decision) in enumerate(
        zip(scenario.events, decisions, strict=True)
    ):
        if event.kind != "clock_tick":
            embedding_inputs += 1
            embedding_characters += len(event.text)
        if index > 0:
            query_text = (
                f"Current time: {event.at.isoformat()}. "
                "Retrieve prior facts and intentions relevant to deciding whether "
                f"an action is due now. Current event: {event.text}"
            )
            embedding_inputs += 1
            embedding_characters += len(query_text)
        decision_text = f"Assistant decision output: {decision.model_dump_json()}"
        embedding_inputs += 1
        embedding_characters += len(decision_text)
    return embedding_inputs, embedding_characters


def _parse_compiler_event(
    event: ModelEvent,
    *,
    checkpoint: Any,
    authored_event: Any,
) -> Usage:
    prompt, usage = _validate_model_event(
        event,
        schema=_local_memory_delta_schema(LOCAL_MODEL_ID),
        seed=101,
    )
    event_line = (
        f"Current event: [{authored_event.id}] {authored_event.at.isoformat()} | "
        f"{authored_event.kind} | {authored_event.text}"
    )
    if event_line not in prompt:
        raise ValueError("compiler prompt does not contain the exact observable event")
    if event.output.completion != checkpoint.raw_compiler_output:
        raise ValueError("raw compiler output differs from checkpoint audit")
    if usage != checkpoint.compiler_usage:
        raise ValueError("raw compiler usage differs from checkpoint audit")
    parse_error = False
    try:
        LocalMemoryDeltaWire.model_validate_json(event.output.completion).to_domain()
    except (ValidationError, ValueError):
        parse_error = True
    if parse_error != checkpoint.compiler_parse_error:
        raise ValueError("compiler parse status differs from raw output")
    return usage


def _extract_scenario_run(sample: Any) -> ScenarioRun:
    if (
        sample.epoch != 1
        or sample.error is not None
        or sample.invalidation is not None
        or sample.error_retries not in (None, 0, [])
        or sample.output is None
    ):
        raise ValueError(f"local sample {sample.id} failed or was retried")
    raw_run = sample.store.get(SCENARIO_RUN_STORE_KEY)
    if raw_run is None:
        raise ValueError(f"local sample {sample.id} has no ScenarioRun")
    stored = ScenarioRun.model_validate(raw_run)
    output = ScenarioRun.model_validate_json(sample.output.completion)
    if stored != output:
        raise ValueError("sample output and stored ScenarioRun differ")
    if str(sample.id) != stored.scenario_id:
        raise ValueError("sample ID and ScenarioRun scenario_id differ")
    return stored


def _validate_run_and_events(
    *,
    sample: Any,
    scenario: Scenario,
    run: ScenarioRun,
    system: str,
    manifest: LocalExperimentManifest,
    manifest_sha256: str,
    first_sample: bool,
    expected_preflight: LocalModelPreflightResult,
) -> tuple[Usage, Usage]:
    if (
        run.system != system
        or run.repetition != 1
        or run.seed != 101
        or run.model != manifest.model.snapshot
        or run.prompt_version != LOCAL_DECISION_VERSION
        or run.scenario_sha256 != canonical_sha256(scenario)
        or run.prompt_sha256 != _sha256_text(local_decision_contract())
        or run.system_config_sha256 != manifest.system_config_sha256[system]
        or run.manifest_sha256 != manifest_sha256
        or run.pricing_config_sha256 != manifest.model.pricing.sha256
    ):
        raise ValueError("ScenarioRun identity or frozen contract binding differs")
    if run.hosted_warmup is not None:
        raise ValueError("local ScenarioRun cannot contain hosted warmup evidence")
    if not run.usage_complete or not run.cost_complete:
        raise ValueError("local ScenarioRun has incomplete usage or cost")
    if run.usage.cost_usd != 0.0 or run.decision_usage.cost_usd != 0.0:
        raise ValueError("local ScenarioRun does not have exact zero API cost")
    compiler_was_called = any(item.compiler_called for item in run.checkpoints)
    if compiler_was_called:
        if run.compiler_usage.cost_usd != 0.0:
            raise ValueError("local compiler usage does not have exact zero API cost")
    elif run.compiler_usage != Usage():
        raise ValueError("baseline without compiler calls has non-empty compiler usage")
    if run.decision_usage.input_tokens <= 0:
        raise ValueError("local ScenarioRun has no decision input usage")

    expected_timeline = [(event.id, event.at) for event in scenario.events]
    actual_timeline = [(item.event_id, item.at) for item in run.checkpoints]
    if actual_timeline != expected_timeline:
        raise ValueError("local ScenarioRun checkpoint timeline differs from dataset")
    if len(run.checkpoint_latency_ms) != len(scenario.events):
        raise ValueError("local ScenarioRun checkpoint latency count differs")
    expected_compilers = [
        system == "anamnesis" and event.kind != "clock_tick"
        for event in scenario.events
    ]
    if [item.compiler_called for item in run.checkpoints] != expected_compilers:
        raise ValueError("local ScenarioRun compiler-call policy differs")

    raw_events = [event for event in sample.events if isinstance(event, ModelEvent)]
    cursor = 0
    raw_usage = Usage(cost_usd=0.0)
    if first_sample:
        live_usage = _validate_live_preflight(raw_events[:2], expected_preflight)
        raw_usage = raw_usage.plus(live_usage)
        cursor = 2
        if run.setup_latency_ms < expected_preflight.setup_latency_ms:
            raise ValueError("first local run omits live preflight setup latency")
    elif not math.isclose(run.setup_latency_ms, 0.0, abs_tol=1e-12):
        raise ValueError("local setup latency must be recorded only on first sample")

    raw_predictions: list[PredictedAction] = []
    raw_decisions: list[Decision] = []
    for authored_event, checkpoint, compiler_called in zip(
        scenario.events,
        run.checkpoints,
        expected_compilers,
        strict=True,
    ):
        if compiler_called:
            if cursor >= len(raw_events):
                raise ValueError("local sample is missing a compiler ModelEvent")
            raw_usage = raw_usage.plus(
                _parse_compiler_event(
                    raw_events[cursor],
                    checkpoint=checkpoint,
                    authored_event=authored_event,
                )
            )
            cursor += 1
        if cursor >= len(raw_events):
            raise ValueError("local sample is missing a decision ModelEvent")
        decision_usage, predictions, decision = _parse_decision_event(
            raw_events[cursor],
            checkpoint=checkpoint,
            authored_event=authored_event,
        )
        raw_usage = raw_usage.plus(decision_usage)
        raw_predictions.extend(predictions)
        raw_decisions.append(decision)
        cursor += 1
    if cursor != len(raw_events):
        raise ValueError("local sample contains unaccounted ModelEvents")
    if raw_predictions != run.predictions:
        raise ValueError("ScenarioRun predictions differ from raw decision outputs")
    if system == "vector_rag":
        expected_retrieval = _expected_vector_retrieval_usage(scenario, raw_decisions)
        actual_retrieval = (
            run.usage.embedding_inputs,
            run.usage.embedding_characters,
        )
        if actual_retrieval != expected_retrieval:
            raise ValueError("local vector retrieval usage differs from raw decisions")
    elif run.usage.embedding_inputs or run.usage.embedding_characters:
        raise ValueError("non-vector local run contains retrieval usage")
    return raw_usage, run.decision_usage.plus(run.compiler_usage)


def _validate_log_usage(log: EvalLog, raw_usage: Usage) -> None:
    if log.stats is None or set(log.stats.model_usage) != {LOCAL_MODEL_ID}:
        raise ValueError("local log stats do not identify one pinned model")
    model_usage = log.stats.model_usage[LOCAL_MODEL_ID]
    stats_usage = _usage_from_model_usage(model_usage)
    if stats_usage != raw_usage:
        raise ValueError("local log stats differ from raw ModelEvent accounting")


def _load_local_smoke_runs(
    paths: list[Path],
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    scenarios: list[Scenario],
) -> list[ScenarioRun]:
    if len(paths) != 4 or len(set(path.resolve() for path in paths)) != 4:
        raise ValueError("local smoke report requires exactly four distinct .eval logs")
    scenario_ids = [scenario.id for scenario in scenarios]
    by_id = {scenario.id: scenario for scenario in scenarios}
    logs_by_system: dict[str, EvalLog] = {}
    all_runs: list[ScenarioRun] = []

    for path in paths:
        if path.suffix != ".eval":
            raise ValueError("local smoke report accepts only Inspect .eval logs")
        log = read_eval_log(path, resolve_attachments=True)
        registry_name = log.eval.task_registry_name
        systems = [
            system
            for system, task_name in LOCAL_SYSTEM_TASKS.items()
            if task_name == registry_name
        ]
        if len(systems) != 1:
            raise ValueError(f"unexpected local smoke task: {registry_name}")
        system = systems[0]
        if system in logs_by_system:
            raise ValueError(f"duplicate local smoke log for {system}")
        logs_by_system[system] = log
        _validate_effective_log_policy(log, manifest)
        _validate_task_and_dataset(
            log,
            system=system,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            scenarios_path=scenarios_path,
            scenarios=scenarios,
        )
        if log.samples is None or len(log.samples) != 10:
            raise ValueError("each local smoke log must contain exactly 10 samples")
        if [str(sample.id) for sample in log.samples] != scenario_ids:
            raise ValueError("local log sample records differ from dataset ID/order")

        preflight_serializations = {
            json.dumps(sample.store.get(LOCAL_PREFLIGHT_STORE_KEY), sort_keys=True)
            for sample in log.samples
        }
        if len(preflight_serializations) != 1 or "null" in preflight_serializations:
            raise ValueError("local samples disagree about live preflight evidence")
        preflight = LocalModelPreflightResult.model_validate_json(
            next(iter(preflight_serializations))
        )

        raw_log_usage = Usage(cost_usd=0.0)
        headline_usage = Usage(cost_usd=0.0)
        for index, sample in enumerate(log.samples):
            run = _extract_scenario_run(sample)
            raw_usage, run_usage = _validate_run_and_events(
                sample=sample,
                scenario=by_id[run.scenario_id],
                run=run,
                system=system,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                first_sample=index == 0,
                expected_preflight=preflight,
            )
            raw_log_usage = raw_log_usage.plus(raw_usage)
            headline_usage = headline_usage.plus(run_usage)
            all_runs.append(run)
        expected_raw = headline_usage.plus(
            preflight.compiler_usage.plus(preflight.decision_usage)
        )
        if raw_log_usage != expected_raw:
            raise ValueError(
                "raw local usage differs from headline plus live preflight"
            )
        _validate_log_usage(log, raw_log_usage)

    if set(logs_by_system) != set(LOCAL_SYSTEM_TASKS):
        raise ValueError("local smoke logs do not cover all four systems exactly")
    return all_runs


def _result_row(result: AggregateResult) -> dict[str, object]:
    row = result.model_dump()
    for name in ("precision", "recall", "f1"):
        row[name] = round(float(row[name]), 6)
    for name in (
        "false_alarm_rate",
        "obsolete_trap_rate",
        "provenance_exact_accuracy",
        "input_token_reduction_vs_full_context",
    ):
        value = row[name]
        row[name] = "" if value is None else round(float(value), 6)
    return row


def _write_csv(path: Path, results: list[AggregateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(result) for result in results]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _render_markdown(results: list[AggregateResult]) -> str:
    header = (
        "| System | TP | FP | FN | Precision | Recall | F1 | False reminders | "
        "FAR | Obsolete | Provenance | Input tokens | Reduction vs full | "
        "Provider API cost USD | Latency p50/p95 ms | Setup ms |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|"
    )
    rows = []
    for result in results:
        if result.cost_usd != 0.0:
            raise ValueError("local aggregate does not have exact zero provider cost")
        rows.append(
            "| "
            + " | ".join(
                [
                    result.system,
                    str(result.tp),
                    str(result.fp),
                    str(result.fn),
                    _percent(result.precision),
                    _percent(result.recall),
                    _percent(result.f1),
                    str(result.false_reminders),
                    _percent(result.false_alarm_rate),
                    str(result.obsolete_errors),
                    _percent(result.provenance_exact_accuracy),
                    str(result.input_tokens),
                    _percent(result.input_token_reduction_vs_full_context),
                    f"{result.cost_usd:.6f}",
                    f"{result.latency_p50_ms:.1f}/{result.latency_p95_ms:.1f}",
                    f"{result.setup_latency_ms:.1f}",
                ]
            )
            + " |"
        )
    return (
        f"# {LOCAL_SMOKE_TITLE}\n\n"
        + header
        + "\n"
        + "\n".join(rows)
        + "\n\n"
        + "Provider API cost is zero. Electricity and hardware cost are "
        "unmeasured. Setup latency is reported separately.\n"
    )


def _validate_provenance_locations(
    *,
    manifest_path: Path,
    scenarios_path: Path,
    run_paths: Sequence[Path],
    csv_path: Path,
    markdown_path: Path,
    provenance_path: Path,
) -> None:
    """Fail before reporting if a sidecar would expose or overwrite a path."""

    source_paths = [manifest_path, scenarios_path, *run_paths]
    for index, path in enumerate(source_paths):
        _repo_relative_path(path, label=f"provenance source {index + 1}")

    target_paths = [csv_path, markdown_path, provenance_path]
    for index, path in enumerate(target_paths):
        _repo_relative_path(
            path,
            label=f"result artifact {index + 1}",
            must_exist=False,
        )

    resolved_sources = {path.resolve() for path in source_paths}
    resolved_targets = [path.resolve() for path in target_paths]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("CSV, Markdown, and provenance paths must be distinct")
    if resolved_sources.intersection(resolved_targets):
        raise ValueError("result artifacts cannot overwrite provenance sources")


def _write_result_provenance(
    path: Path,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    run_paths: Sequence[Path],
    expected_run_sha256: Sequence[str],
    csv_path: Path,
    markdown_path: Path,
) -> None:
    """Write the last artifact, binding validated inputs to rendered outputs."""

    _validate_provenance_locations(
        manifest_path=manifest_path,
        scenarios_path=scenarios_path,
        run_paths=run_paths,
        csv_path=csv_path,
        markdown_path=markdown_path,
        provenance_path=path,
    )
    if manifest.git_commit is None:
        raise ValueError("frozen local manifest is missing git_commit")
    if _sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("frozen local manifest changed while reporting")

    dataset_digest = _sha256_file(scenarios_path)
    if dataset_digest != manifest.dataset.sha256:
        raise ValueError("scenario dataset differs from the frozen manifest")

    if len(run_paths) != 4 or len(expected_run_sha256) != 4:
        raise ValueError("provenance requires exactly four input .eval logs")
    run_records: list[dict[str, str]] = []
    for index, (run_path, expected_digest) in enumerate(
        zip(run_paths, expected_run_sha256, strict=True),
        start=1,
    ):
        actual_digest = _sha256_file(run_path)
        if actual_digest != expected_digest:
            raise ValueError(f"input .eval log {index} changed while reporting")
        run_records.append(
            {
                "path": _repo_relative_path(
                    run_path,
                    label=f"input .eval log {index}",
                ),
                "sha256": actual_digest,
            }
        )

    payload = {
        "schema_version": 1,
        "artifact": "anamnesis_local_smoke_result_provenance",
        "title": LOCAL_SMOKE_TITLE,
        "hypothesis_test_eligible": False,
        "source_git_commit": manifest.git_commit,
        "frozen_manifest": {
            "path": _repo_relative_path(
                manifest_path,
                label="frozen manifest",
            ),
            "sha256": manifest_sha256,
        },
        "scenario_dataset": {
            "path": _repo_relative_path(
                scenarios_path,
                label="scenario dataset",
            ),
            "sha256": dataset_digest,
        },
        "input_eval_logs": run_records,
        "outputs": {
            "csv": {
                "path": _repo_relative_path(csv_path, label="CSV output"),
                "sha256": _sha256_file(csv_path),
            },
            "markdown": {
                "path": _repo_relative_path(
                    markdown_path,
                    label="Markdown output",
                ),
                "sha256": _sha256_file(markdown_path),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def local_report_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and score the strict local smoke diagnostic"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("eval/scenarios/smoke.jsonl"),
    )
    parser.add_argument("--runs", type=Path, nargs=4, required=True)
    parser.add_argument("--csv", type=Path, default=Path("results/local_smoke.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("results/local_smoke.md"))
    parser.add_argument(
        "--provenance",
        type=Path,
        default=LOCAL_SMOKE_PROVENANCE_PATH,
    )
    args = parser.parse_args(argv)

    _validate_provenance_locations(
        manifest_path=args.manifest,
        scenarios_path=args.scenarios,
        run_paths=args.runs,
        csv_path=args.csv,
        markdown_path=args.markdown,
        provenance_path=args.provenance,
    )
    run_sha256 = [_sha256_file(path) for path in args.runs]
    manifest, scenarios, manifest_sha256 = _validate_frozen_local_manifest(
        manifest_path=args.manifest,
        scenarios_path=args.scenarios,
    )
    runs = _load_local_smoke_runs(
        args.runs,
        manifest=manifest,
        manifest_path=args.manifest,
        manifest_sha256=manifest_sha256,
        scenarios_path=args.scenarios,
        scenarios=scenarios,
    )
    run_keys = [
        (run.scenario_id, run.system, run.repetition, run.model) for run in runs
    ]
    if len(run_keys) != 40 or len(run_keys) != len(set(run_keys)):
        raise ValueError("local smoke matrix must contain 40 unique ScenarioRuns")

    scored = [
        (score_scenario(scenario, run), run)
        for run in runs
        for scenario in scenarios
        if scenario.id == run.scenario_id
    ]
    if len(scored) != 40:
        raise ValueError("local smoke ScenarioRuns do not match the 10 scenarios")
    results = aggregate_results(scored)
    if len(results) != 4 or {result.system for result in results} != set(
        LOCAL_SYSTEM_TASKS
    ):
        raise ValueError("local smoke aggregation did not produce four systems")
    if any(
        result.scenarios != 10
        or result.repetition != 1
        or result.model != LOCAL_MODEL_ID
        or not result.usage_complete
        or not result.cost_complete
        or result.cost_usd != 0.0
        for result in results
    ):
        raise ValueError("local smoke aggregates violate the frozen matrix")

    _write_csv(args.csv, results)
    rendered = _render_markdown(results)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(rendered, encoding="utf-8")
    _write_result_provenance(
        args.provenance,
        manifest=manifest,
        manifest_path=args.manifest,
        manifest_sha256=manifest_sha256,
        scenarios_path=args.scenarios,
        run_paths=args.runs,
        expected_run_sha256=run_sha256,
        csv_path=args.csv,
        markdown_path=args.markdown,
    )
    print(rendered, end="")
    return 0


__all__ = [
    "LOCAL_SMOKE_PROVENANCE_PATH",
    "LOCAL_SMOKE_TITLE",
    "local_report_main",
]
