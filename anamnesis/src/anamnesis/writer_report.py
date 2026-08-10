"""Strict reporting for the frozen local W1 memory-writer diagnostic.

The evaluated LLM writer never receives the gold writer reference.  This
module validates the complete Inspect log first, and only then opens the
reference to compare the deterministic candidates produced by measured and
gold deltas.  Decision outputs are reported as a diagnostic, but cannot affect
the writer gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeAlias

from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ChatMessageUser
from pydantic import ValidationError

from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.local_experiment import (
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
    _repo_relative_path,
    _resolve_logged_repo_path,
    _sha256_file,
    _sha256_text,
    _validate_effective_log_policy,
    _validate_log_usage,
    _validate_path_task_arg,
    _validate_run_and_events,
    _verify_current_git_state,
)
from anamnesis.local_runtime import (
    LOCAL_DECISION_VERSION,
    LOCAL_PREFLIGHT_STORE_KEY,
    LOCAL_SCENARIO_TASK_VERSION,
    LocalModelPreflightResult,
    local_decision_prompt_contract,
    local_decision_schema_contract,
    local_memory_compiler_prompt_contract,
    local_memory_compiler_schema_contract,
    local_system_config_sha256,
)
from anamnesis.local_wire import (
    LocalMemoryDeltaWire,
    build_local_memory_compiler_prompt,
)
from anamnesis.memory import InMemoryAnamnesis, MemoryDelta
from anamnesis.oracle import (
    OracleCompilerArtifact,
    load_oracle_artifact,
    oracle_artifact_sha256,
)
from anamnesis.schema import (
    Decision,
    ProposedAction,
    Scenario,
    ScenarioRun,
    Usage,
)
from anamnesis.scoring import aggregate_results, score_scenario

WRITER_TITLE = "Local W1 writer diagnostic — not a hypothesis test"
WRITER_TASK_NAME = "local_anamnesis_writer_diagnostic"
WRITER_DATASET_NAME = "anamnesis-local-writer_diagnostic-v0"
WRITER_SYSTEM = "anamnesis"
WRITER_COMPILER_CALLS = 45
WRITER_CSV_PATH = Path("results/local_writer_w1.csv")
WRITER_MARKDOWN_PATH = Path("results/local_writer_w1.md")
WRITER_PROVENANCE_PATH = Path("results/local_writer_w1.provenance.json")

CandidateKey: TypeAlias = tuple[
    str,  # checkpoint event ID
    str,  # stable action key
    str,  # due_at ISO-8601
    str,  # action kind
    str,  # canonical payload JSON
    str,  # summary
    tuple[str, ...],  # sorted evidence event IDs
]


def _repo_file(relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} path must be repository-relative")
    resolved = (REPO_ROOT / candidate).resolve()
    if not resolved.is_relative_to(REPO_ROOT) or not resolved.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return resolved


def _require_hash(path: Path, expected: str | None, *, label: str) -> str:
    if expected is None:
        raise ValueError(f"frozen writer manifest is missing {label} SHA-256")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} bytes differ from the frozen manifest")
    return actual


def _validate_writer_manifest_identity(manifest: LocalExperimentManifest) -> None:
    if manifest.status != "frozen" or manifest.phase != "writer_diagnostic":
        raise ValueError("writer report requires a frozen writer_diagnostic manifest")
    if manifest.hypothesis_test_eligible is not False:
        raise ValueError("writer diagnostic cannot be hypothesis-test eligible")
    if manifest.compiler_mode != "llm":
        raise ValueError("writer diagnostic requires compiler_mode=llm")
    if manifest.systems != [WRITER_SYSTEM]:
        raise ValueError("writer diagnostic requires exactly the anamnesis system")
    if manifest.scenario_count != 10:
        raise ValueError("writer diagnostic requires exactly 10 scenarios")
    if manifest.execution.repetitions != 1 or manifest.execution.seeds != [101]:
        raise ValueError("writer diagnostic requires one repetition with seed 101")
    if not manifest.model.same_model_for_compiler_and_decision:
        raise ValueError("writer diagnostic requires the same compiler/decision model")
    if manifest.writer_reference is None:
        raise ValueError("writer diagnostic is missing writer_reference")
    if manifest.oracle_annotations is not None:
        raise ValueError("writer diagnostic cannot expose oracle_annotations")
    if set(manifest.system_config_sha256) != {WRITER_SYSTEM}:
        raise ValueError("writer manifest has the wrong system hash matrix")


def _validate_frozen_writer_manifest(
    *,
    manifest_path: Path,
    scenarios_path: Path,
    command_runner: Callable[..., Any] | None = None,
) -> tuple[LocalExperimentManifest, list[Scenario], str]:
    """Validate measured inputs without resolving the gold writer reference."""

    manifest_bytes = manifest_path.read_bytes()
    manifest = LocalExperimentManifest.model_validate_json(manifest_bytes)
    _validate_writer_manifest_identity(manifest)

    expected_dataset = _repo_file(manifest.dataset.path, label="writer dataset")
    if scenarios_path.resolve() != expected_dataset:
        raise ValueError("--scenarios differs from the frozen writer dataset")
    verify_static_local_inputs(manifest, repo_root=REPO_ROOT)
    _require_hash(scenarios_path, manifest.dataset.sha256, label="writer dataset")
    scenarios = load_scenarios(scenarios_path)
    if len(scenarios) != 10:
        raise ValueError("writer diagnostic dataset must contain 10 scenarios")

    pricing_path = _repo_file(manifest.model.pricing.path, label="pricing config")
    pricing_sha256 = validate_zero_api_pricing(pricing_path, manifest.model.snapshot)
    if pricing_sha256 != manifest.model.pricing.sha256:
        raise ValueError("writer pricing bytes differ from the frozen manifest")

    preflight_path = _repo_file(manifest.model.preflight.path, label="model preflight")
    if manifest.git_commit is None:
        raise ValueError("frozen writer manifest is missing git_commit")
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

    contracts = {
        "decision_prompt_sha256": _sha256_text(local_decision_prompt_contract()),
        "decision_schema_sha256": _sha256_text(local_decision_schema_contract()),
        "memory_compiler_prompt_sha256": _sha256_text(
            local_memory_compiler_prompt_contract()
        ),
        "memory_compiler_schema_sha256": _sha256_text(
            local_memory_compiler_schema_contract()
        ),
    }
    for field, expected in contracts.items():
        if getattr(manifest, field) != expected:
            raise ValueError(f"writer manifest {field} differs from runtime")
    expected_system_hash = local_system_config_sha256(
        system=WRITER_SYSTEM,
        top_k=manifest.embedding.top_k,
        embedding_model=manifest.embedding.model,
        embedding_repository=manifest.embedding.repository,
        embedding_revision=manifest.embedding.revision,
        pricing_config_sha256=manifest.model.pricing.sha256,
    )
    if manifest.system_config_sha256 != {WRITER_SYSTEM: expected_system_hash}:
        raise ValueError("writer manifest system hash differs from runtime")
    return manifest, scenarios, hashlib.sha256(manifest_bytes).hexdigest()


def _validate_writer_task_and_dataset(
    log: EvalLog,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    scenarios: list[Scenario],
) -> None:
    spec = log.eval
    if spec.task_registry_name != WRITER_TASK_NAME:
        raise ValueError("unexpected writer diagnostic task")
    if spec.task_version != LOCAL_SCENARIO_TASK_VERSION:
        raise ValueError("writer task version differs from the runtime contract")
    if (
        spec.task_file is None
        or _resolve_logged_repo_path(spec.task_file)
        != (REPO_ROOT / LOCAL_TASK_FILE).resolve()
    ):
        raise ValueError("writer log task file differs from the frozen task")

    task_args = dict(spec.task_args or {})
    if set(task_args) != {"seed", "repetition", "manifest", "ollama_models_dir"}:
        raise ValueError("writer task arguments are missing or contain extras")
    if task_args.get("seed") != 101 or task_args.get("repetition") != 1:
        raise ValueError("writer task seed/repetition differs from the matrix")
    logged_manifest = task_args.get("manifest")
    if (
        not isinstance(logged_manifest, str)
        or _resolve_logged_repo_path(logged_manifest) != manifest_path.resolve()
    ):
        raise ValueError("writer task is not bound to this frozen manifest")
    _validate_path_task_arg(task_args, "ollama_models_dir")

    ids = [scenario.id for scenario in scenarios]
    required_metadata = {
        "track": "local_zero_api_cost",
        "claim_scope": "diagnostic_development_only",
        "hypothesis_test_eligible": False,
        "system": WRITER_SYSTEM,
        "dataset": manifest.dataset.path,
        "dataset_split": "writer_diagnostic",
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
    }
    metadata = dict(spec.metadata or {})
    if metadata != required_metadata:
        changed = sorted(
            key
            for key in set(metadata) | set(required_metadata)
            if metadata.get(key) != required_metadata.get(key)
        )
        raise ValueError(f"writer task metadata differs: {changed}")

    dataset = spec.dataset
    if dataset.name != WRITER_DATASET_NAME:
        raise ValueError("writer log dataset name differs from the protocol")
    if dataset.location is None or scenarios_path.resolve() not in (
        _logged_dataset_candidates(dataset.location)
    ):
        raise ValueError("writer log dataset location differs from --scenarios")
    if dataset.samples != 10:
        raise ValueError("writer log dataset must contain exactly 10 samples")
    if tuple(str(item) for item in (dataset.sample_ids or [])) != tuple(ids):
        raise ValueError("writer log dataset IDs/order differ from --scenarios")
    if dataset.shuffled is not False:
        raise ValueError("writer diagnostic dataset must be unshuffled")


def _zero_usage() -> Usage:
    return Usage(cost_usd=0.0)


def _validate_checkpoint_delta_binding(checkpoint: Any) -> None:
    """Bind raw local-wire output to the exact replayed domain delta."""

    if not checkpoint.compiler_called:
        return
    raw = checkpoint.raw_compiler_output
    raw_delta: MemoryDelta | None = None
    try:
        if raw is None:
            raise ValueError("compiler checkpoint has no raw output")
        raw_delta = LocalMemoryDeltaWire.model_validate_json(raw).to_domain()
    except (ValidationError, ValueError, TypeError):
        if not checkpoint.compiler_parse_error:
            raise ValueError(
                "parse-valid checkpoint has invalid raw compiler output"
            ) from None
        if checkpoint.memory_delta_json is not None:
            raise ValueError(
                "invalid raw compiler output has an audit MemoryDelta"
            ) from None
        if checkpoint.memory_delta_accepted is not False:
            raise ValueError("invalid raw compiler output was not rejected") from None
        return

    if checkpoint.compiler_parse_error:
        raise ValueError("valid raw compiler output is marked as a parse error")
    if checkpoint.memory_delta_json is None:
        raise ValueError("parse-valid writer checkpoint has no audit MemoryDelta")
    audit_delta = _memory_delta_from_audit_json(checkpoint.memory_delta_json)
    if canonical_sha256(raw_delta) != canonical_sha256(audit_delta):
        raise ValueError("raw compiler output differs from audit MemoryDelta semantics")
    if checkpoint.memory_delta_accepted is None:
        raise ValueError("parse-valid writer checkpoint has no acceptance result")


def _validate_exact_compiler_prompts(
    *,
    sample: Any,
    scenario: Scenario,
    run: ScenarioRun,
    first_sample: bool,
) -> None:
    """Replay the measured prefix and bind every compiler prompt byte-for-byte."""

    raw_events = [event for event in sample.events if isinstance(event, ModelEvent)]
    cursor = 2 if first_sample else 0
    memory = InMemoryAnamnesis()
    for event, checkpoint in zip(
        scenario.to_runtime().events, run.checkpoints, strict=True
    ):
        delta: MemoryDelta | None = None
        if event.kind != "clock_tick":
            if cursor >= len(raw_events):
                raise ValueError("writer sample is missing a compiler ModelEvent")
            compiler_event = raw_events[cursor]
            expected_prompt = build_local_memory_compiler_prompt(
                event=event,
                active_state=memory.compiler_state(),
            )
            if (
                len(compiler_event.input) != 1
                or not isinstance(compiler_event.input[0], ChatMessageUser)
                or compiler_event.input[0].content != expected_prompt
            ):
                raise ValueError(
                    "writer compiler prompt differs from sanitized event/current state"
                )
            if not checkpoint.compiler_parse_error:
                if checkpoint.memory_delta_json is None:
                    raise ValueError(
                        "parse-valid writer checkpoint has no audit MemoryDelta"
                    )
                delta = _memory_delta_from_audit_json(checkpoint.memory_delta_json)
            cursor += 1
        applied = memory.ingest(event, delta)
        if event.kind != "clock_tick" and (
            applied.accepted != checkpoint.memory_delta_accepted
        ):
            raise ValueError("prompt replay acceptance differs from checkpoint")
        memory.select(event)
        if cursor >= len(raw_events):
            raise ValueError("writer sample is missing a decision ModelEvent")
        cursor += 1
        # Compiler state contains current facts and active intentions only. A
        # no-action commit advances lifecycle without importing decision text.
        memory.commit(event, Decision())
    if cursor != len(raw_events):
        raise ValueError("writer prompt replay found unaccounted ModelEvents")


def _load_writer_run(
    path: Path,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    scenarios: list[Scenario],
) -> list[ScenarioRun]:
    if path.suffix != ".eval" or not path.is_file():
        raise ValueError("writer report requires exactly one Inspect .eval log")
    log = read_eval_log(path, resolve_attachments=True)
    _validate_effective_log_policy(log, manifest)
    _validate_writer_task_and_dataset(
        log,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        scenarios_path=scenarios_path,
        scenarios=scenarios,
    )
    if log.samples is None or len(log.samples) != 10:
        raise ValueError("writer log requires exactly 10 ordered samples")
    ids = [scenario.id for scenario in scenarios]
    if [str(sample.id) for sample in log.samples] != ids:
        raise ValueError("writer sample records differ from dataset ID/order")

    serialized_preflights = {
        json.dumps(sample.store.get(LOCAL_PREFLIGHT_STORE_KEY), sort_keys=True)
        for sample in log.samples
    }
    if len(serialized_preflights) != 1 or "null" in serialized_preflights:
        raise ValueError("writer samples disagree about setup preflight evidence")
    preflight = LocalModelPreflightResult.model_validate_json(
        next(iter(serialized_preflights))
    )

    by_id = {scenario.id: scenario for scenario in scenarios}
    raw_log_usage = _zero_usage()
    headline_usage = _zero_usage()
    runs: list[ScenarioRun] = []
    compiler_calls = 0
    for index, sample in enumerate(log.samples):
        run = _extract_scenario_run(sample)
        scenario = by_id.get(run.scenario_id)
        if scenario is None:
            raise ValueError("writer ScenarioRun does not match the dataset")
        raw_usage, measured_usage = _validate_run_and_events(
            sample=sample,
            scenario=scenario,
            run=run,
            system=WRITER_SYSTEM,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            first_sample=index == 0,
            expected_preflight=preflight,
        )
        for checkpoint in run.checkpoints:
            _validate_checkpoint_delta_binding(checkpoint)
        _validate_exact_compiler_prompts(
            sample=sample,
            scenario=scenario,
            run=run,
            first_sample=index == 0,
        )
        raw_log_usage = raw_log_usage.plus(raw_usage)
        headline_usage = headline_usage.plus(measured_usage)
        compiler_calls += sum(
            checkpoint.compiler_called for checkpoint in run.checkpoints
        )
        runs.append(run)
    if compiler_calls != WRITER_COMPILER_CALLS:
        raise ValueError("writer run must contain exactly 45 scenario compiler calls")
    expected_raw = headline_usage.plus(
        preflight.compiler_usage.plus(preflight.decision_usage)
    )
    if raw_log_usage != expected_raw:
        raise ValueError("writer raw calls differ from headline plus setup preflight")
    _validate_log_usage(log, raw_log_usage)
    keys = [(run.scenario_id, run.system, run.repetition, run.model) for run in runs]
    if len(keys) != 10 or len(keys) != len(set(keys)):
        raise ValueError("writer report requires 10 unique ScenarioRuns")
    return runs


def _candidate_key(checkpoint: str, candidate: Any) -> CandidateKey:
    payload = json.dumps(
        dict(candidate.action_template.payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        checkpoint,
        candidate.action_key,
        candidate.due_at.isoformat(),
        candidate.action_template.kind,
        payload,
        candidate.action_template.summary,
        tuple(sorted(candidate.evidence_event_ids)),
    )


def _memory_delta_from_audit_json(value: str) -> MemoryDelta:
    """Round-trip the domain audit form, omitting unset update fields.

    ``UpdateIntent`` deliberately distinguishes an omitted field from an
    explicit null. Pydantic's default domain serialization includes its unset
    optional fields as null, so reconstruct that distinction before replay.
    """

    raw = json.loads(value)
    if not isinstance(raw, dict) or not isinstance(raw.get("mutations"), list):
        raise ValueError("checkpoint memory_delta_json is not a MemoryDelta object")
    for mutation in raw["mutations"]:
        if isinstance(mutation, dict) and mutation.get("op") == "update_intent":
            for name in (
                "trigger",
                "required_conditions",
                "blockers",
                "action_template",
            ):
                if mutation.get(name) is None:
                    mutation.pop(name, None)
    return MemoryDelta.model_validate(raw)


def _commit_all_due(memory: InMemoryAnamnesis, event: Any, candidates: Any) -> None:
    actions = [
        ProposedAction(
            kind=candidate.action_template.kind,
            action_key=candidate.action_key,
            payload=dict(candidate.action_template.payload),
            summary=candidate.action_template.summary,
            evidence_event_ids=list(candidate.evidence_event_ids),
        )
        for candidate in candidates
    ]
    memory.commit(event, Decision(actions=actions))


def _replay_candidate_counters(
    scenarios: list[Scenario],
    runs: list[ScenarioRun],
    reference: OracleCompilerArtifact,
) -> tuple[Counter[CandidateKey], Counter[CandidateKey], int, int, int]:
    """Replay measured and reference deltas through independent fresh stores."""

    by_run = {run.scenario_id: run for run in runs}
    measured_candidates: Counter[CandidateKey] = Counter()
    reference_candidates: Counter[CandidateKey] = Counter()
    parse_invalid = 0
    semantic_invalid = 0
    accepted = 0

    for scenario in scenarios:
        run = by_run.get(scenario.id)
        if run is None:
            raise ValueError(f"writer run is missing scenario {scenario.id}")
        runtime = scenario.to_runtime()
        measured_memory = InMemoryAnamnesis()
        reference_memory = InMemoryAnamnesis()
        reference_records = iter(reference.records_for(runtime))

        for event, checkpoint in zip(runtime.events, run.checkpoints, strict=True):
            measured_delta: MemoryDelta | None = None
            reference_delta: MemoryDelta | None = None
            if event.kind != "clock_tick":
                record = next(reference_records)
                if record.event_id != event.id:
                    raise ValueError("writer reference order differs from dataset")
                reference_delta = record.delta
                if checkpoint.compiler_parse_error:
                    parse_invalid += 1
                elif checkpoint.memory_delta_json is None:
                    raise ValueError("parse-valid writer checkpoint has no MemoryDelta")
                else:
                    measured_delta = _memory_delta_from_audit_json(
                        checkpoint.memory_delta_json
                    )

            measured_apply = measured_memory.ingest(event, measured_delta)
            reference_apply = reference_memory.ingest(event, reference_delta)
            if not reference_apply.accepted:
                raise ValueError("frozen writer reference contains a rejected delta")
            if measured_apply.accepted != checkpoint.memory_delta_accepted and (
                event.kind != "clock_tick"
            ):
                raise ValueError("measured replay acceptance differs from checkpoint")
            if event.kind != "clock_tick":
                if measured_apply.accepted:
                    accepted += 1
                elif not checkpoint.compiler_parse_error:
                    semantic_invalid += 1
            measured_selection = measured_memory.select(event)
            reference_selection = reference_memory.select(event)
            if list(measured_selection.due_candidate_ids) != (
                checkpoint.due_candidate_ids
            ):
                raise ValueError("measured replay due IDs differ from checkpoint")
            measured_candidates.update(
                _candidate_key(event.id, candidate)
                for candidate in measured_selection.due_candidates
            )
            reference_candidates.update(
                _candidate_key(event.id, candidate)
                for candidate in reference_selection.due_candidates
            )
            _commit_all_due(measured_memory, event, measured_selection.due_candidates)
            _commit_all_due(reference_memory, event, reference_selection.due_candidates)
        try:
            next(reference_records)
        except StopIteration:
            pass
        else:
            raise ValueError("writer reference contains unconsumed event records")

    return (
        measured_candidates,
        reference_candidates,
        parse_invalid,
        semantic_invalid,
        accepted,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _candidate_confusion(
    measured: Counter[CandidateKey], gold: Counter[CandidateKey]
) -> tuple[int, int, int]:
    return (
        sum((measured & gold).values()),
        sum((measured - gold).values()),
        sum((gold - measured).values()),
    )


def _metrics(
    scenarios: list[Scenario],
    runs: list[ScenarioRun],
    reference: OracleCompilerArtifact,
) -> dict[str, object]:
    measured, gold, parse_invalid, semantic_invalid, accepted = (
        _replay_candidate_counters(scenarios, runs, reference)
    )
    true_positive, false_positive, false_negative = _candidate_confusion(measured, gold)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)

    action_results = aggregate_results(
        [
            (score_scenario(scenario, run), run)
            for scenario, run in zip(scenarios, runs, strict=True)
        ]
    )
    if len(action_results) != 1:
        raise ValueError("writer final-action diagnostic must aggregate to one row")
    actions = action_results[0]
    compiler_usage = _zero_usage()
    decision_usage = _zero_usage()
    for run in runs:
        compiler_usage = compiler_usage.plus(run.compiler_usage)
        decision_usage = decision_usage.plus(run.decision_usage)
    total_usage = compiler_usage.plus(decision_usage)
    if total_usage.cost_usd != 0.0:
        raise ValueError("writer diagnostic must have exact zero provider API cost")

    gate = {
        "invalid_zero": parse_invalid == 0 and semantic_invalid == 0,
        "accepted_all": accepted == WRITER_COMPILER_CALLS,
        "candidate_fp_zero": false_positive == 0,
        "candidate_fn_zero": false_negative == 0,
    }
    return {
        "title": WRITER_TITLE,
        "hypothesis_test_eligible": False,
        "compiler_calls": WRITER_COMPILER_CALLS,
        "compiler_parse_invalid": parse_invalid,
        "compiler_semantic_invalid": semantic_invalid,
        "compiler_accepted": accepted,
        "candidate_tp": true_positive,
        "candidate_fp": false_positive,
        "candidate_fn": false_negative,
        "candidate_precision": precision,
        "candidate_recall": recall,
        "candidate_f1": f1,
        "compiler_input_tokens": compiler_usage.input_tokens,
        "compiler_output_tokens": compiler_usage.output_tokens,
        "decision_input_tokens": decision_usage.input_tokens,
        "decision_output_tokens": decision_usage.output_tokens,
        "total_input_tokens": total_usage.input_tokens,
        "total_output_tokens": total_usage.output_tokens,
        "compiler_latency_ms": sum(run.compiler_latency_ms for run in runs),
        "decision_latency_ms": sum(run.decision_latency_ms for run in runs),
        "local_latency_ms": sum(run.local_latency_ms for run in runs),
        "total_latency_ms": sum(
            run.compiler_latency_ms + run.decision_latency_ms + run.local_latency_ms
            for run in runs
        ),
        "setup_latency_ms": sum(run.setup_latency_ms for run in runs),
        "provider_api_cost_usd": 0.0,
        "final_action_tp_diagnostic": actions.tp,
        "final_action_fp_diagnostic": actions.fp,
        "final_action_fn_diagnostic": actions.fn,
        "final_action_precision_diagnostic": actions.precision,
        "final_action_recall_diagnostic": actions.recall,
        "final_action_f1_diagnostic": actions.f1,
        "gate_invalid_zero": gate["invalid_zero"],
        "gate_accepted_all": gate["accepted_all"],
        "gate_candidate_fp_zero": gate["candidate_fp_zero"],
        "gate_candidate_fn_zero": gate["candidate_fn_zero"],
        "gate_passed": all(gate.values()),
    }


def _write_csv(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in metrics.items()
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def _percent(value: object) -> str:
    return f"{float(value):.1%}"


def _render_markdown(metrics: dict[str, object]) -> str:
    return (
        f"# {WRITER_TITLE}\n\n"
        "This development-only diagnostic evaluates the LLM memory writer. "
        "It is not hypothesis-test evidence. The gate is computed only from "
        "deterministically replayed due candidates; final decision actions are "
        "reported separately and cannot change the gate.\n\n"
        "| Calls | Parse invalid | Semantic invalid | Accepted | Candidate TP | "
        "FP | FN | Precision | Recall | F1 | Gate |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n"
        f"| {metrics['compiler_calls']} | {metrics['compiler_parse_invalid']} | "
        f"{metrics['compiler_semantic_invalid']} | {metrics['compiler_accepted']} | "
        f"{metrics['candidate_tp']} | {metrics['candidate_fp']} | "
        f"{metrics['candidate_fn']} | {_percent(metrics['candidate_precision'])} | "
        f"{_percent(metrics['candidate_recall'])} | "
        f"{_percent(metrics['candidate_f1'])} | "
        f"{'PASS' if metrics['gate_passed'] else 'FAIL'} |\n\n"
        "| Compiler tokens in/out | Decision tokens in/out | Total tokens in/out | "
        "Compiler ms | Decision ms | Local ms | Total ms | Setup ms | API cost USD |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| {metrics['compiler_input_tokens']}/{metrics['compiler_output_tokens']} | "
        f"{metrics['decision_input_tokens']}/{metrics['decision_output_tokens']} | "
        f"{metrics['total_input_tokens']}/{metrics['total_output_tokens']} | "
        f"{float(metrics['compiler_latency_ms']):.1f} | "
        f"{float(metrics['decision_latency_ms']):.1f} | "
        f"{float(metrics['local_latency_ms']):.1f} | "
        f"{float(metrics['total_latency_ms']):.1f} | "
        f"{float(metrics['setup_latency_ms']):.1f} | "
        f"{float(metrics['provider_api_cost_usd']):.6f} |\n\n"
        "Final-action diagnostic (excluded from the gate): "
        f"TP={metrics['final_action_tp_diagnostic']}, "
        f"FP={metrics['final_action_fp_diagnostic']}, "
        f"FN={metrics['final_action_fn_diagnostic']}, "
        f"F1={_percent(metrics['final_action_f1_diagnostic'])}.\n\n"
        "Gate: zero parse/semantic invalid deltas, 45/45 accepted deltas, "
        "zero candidate false positives, and zero candidate false negatives. "
        "Provider API cost is exactly zero; electricity and hardware cost are "
        "unmeasured.\n"
    )


def _validate_output_locations(
    *,
    sources: Sequence[Path],
    csv_path: Path,
    markdown_path: Path,
    provenance_path: Path,
) -> None:
    for index, source in enumerate(sources, start=1):
        _repo_relative_path(source, label=f"writer provenance source {index}")
    targets = [csv_path, markdown_path, provenance_path]
    results_root = (REPO_ROOT / "results").resolve()
    raw_runs_root = (results_root / "runs").resolve()
    for index, target in enumerate(targets, start=1):
        _repo_relative_path(
            target,
            label=f"writer result artifact {index}",
            must_exist=False,
        )
        resolved_target = target.resolve()
        if resolved_target == results_root or not resolved_target.is_relative_to(
            results_root
        ):
            raise ValueError("writer outputs must be files under results/")
        if resolved_target == raw_runs_root or resolved_target.is_relative_to(
            raw_runs_root
        ):
            raise ValueError("writer outputs cannot be written under results/runs/")
    resolved = [target.resolve() for target in targets]
    if len(set(resolved)) != len(resolved):
        raise ValueError("writer result artifact paths must be distinct")
    if {source.resolve() for source in sources}.intersection(resolved):
        raise ValueError("writer outputs cannot overwrite source artifacts")


def _all_writer_source_paths(
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    scenarios_path: Path,
    reference_path: Path,
    run_path: Path,
) -> dict[str, Path]:
    """Resolve every immutable report input before any output is written."""

    model_artifact_path = _repo_file(
        manifest.model.artifact.path, label="model artifact pin"
    )
    dependency_lock_path = _repo_file(
        manifest.dependency_lock.path, label="dependency lock"
    )
    research_contract_path = _repo_file(
        manifest.research_contract.path, label="research contract"
    )
    architecture_contract_path = _repo_file(
        manifest.architecture_contract.path, label="architecture contract"
    )
    preflight_path = _repo_file(manifest.model.preflight.path, label="model preflight")
    pricing_path = _repo_file(manifest.model.pricing.path, label="pricing config")
    reporter_path = Path(__file__).resolve()
    task_path = (REPO_ROOT / LOCAL_TASK_FILE).resolve()
    if not reporter_path.is_file() or not task_path.is_file():
        raise ValueError("writer reporter/task source files are missing")
    return {
        "manifest": manifest_path,
        "dataset": scenarios_path,
        "reference": reference_path,
        "run": run_path,
        "model_artifact": model_artifact_path,
        "dependency_lock": dependency_lock_path,
        "research_contract": research_contract_path,
        "architecture_contract": architecture_contract_path,
        "reporter": reporter_path,
        "task": task_path,
        "preflight": preflight_path,
        "pricing": pricing_path,
    }


def _write_provenance(
    path: Path,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    reference_path: Path,
    reference: OracleCompilerArtifact,
    run_path: Path,
    run_sha256: str,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    sources = _all_writer_source_paths(
        manifest=manifest,
        manifest_path=manifest_path,
        scenarios_path=scenarios_path,
        reference_path=reference_path,
        run_path=run_path,
    )
    _validate_output_locations(
        sources=list(sources.values()),
        csv_path=csv_path,
        markdown_path=markdown_path,
        provenance_path=path,
    )
    if manifest.git_commit is None or manifest.writer_reference is None:
        raise ValueError("frozen writer manifest is incomplete")
    if _sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("writer manifest changed while reporting")
    dataset_digest = _require_hash(
        scenarios_path, manifest.dataset.sha256, label="writer dataset"
    )
    reference_digest = _require_hash(
        reference_path,
        manifest.writer_reference.sha256,
        label="writer reference",
    )
    if _sha256_file(run_path) != run_sha256:
        raise ValueError("writer .eval log changed while reporting")
    preflight_path = sources["preflight"]
    pricing_path = sources["pricing"]
    preflight_digest = _require_hash(
        preflight_path, manifest.model.preflight.sha256, label="model preflight"
    )
    pricing_digest = _require_hash(
        pricing_path, manifest.model.pricing.sha256, label="pricing config"
    )
    reporter_path = sources["reporter"]
    task_path = sources["task"]
    payload = {
        "schema_version": 1,
        "artifact": "anamnesis_local_w1_writer_result_provenance",
        "title": WRITER_TITLE,
        "hypothesis_test_eligible": False,
        "source_git_commit": manifest.git_commit,
        "source": {
            "reporter": {
                "path": _repo_relative_path(reporter_path, label="writer reporter"),
                "sha256": _sha256_file(reporter_path),
            },
            "task": {
                "path": _repo_relative_path(task_path, label="writer task"),
                "sha256": _sha256_file(task_path),
            },
        },
        "frozen_manifest": {
            "path": _repo_relative_path(manifest_path, label="writer manifest"),
            "sha256": manifest_sha256,
        },
        "scenario_dataset": {
            "path": _repo_relative_path(scenarios_path, label="writer dataset"),
            "sha256": dataset_digest,
            "canonical_sha256": dataset_sha256(load_scenarios(scenarios_path)),
        },
        "writer_reference": {
            "path": _repo_relative_path(reference_path, label="writer reference"),
            "sha256": reference_digest,
            "canonical_semantics_sha256": oracle_artifact_sha256(reference),
            "opened_only_after_run_validation": True,
        },
        "model_preflight": {
            "path": _repo_relative_path(preflight_path, label="model preflight"),
            "sha256": preflight_digest,
        },
        "pricing_config": {
            "path": _repo_relative_path(pricing_path, label="pricing config"),
            "sha256": pricing_digest,
        },
        "input_eval_log": {
            "path": _repo_relative_path(run_path, label="writer input .eval log"),
            "sha256": run_sha256,
        },
        "outputs": {
            "csv": {
                "path": _repo_relative_path(csv_path, label="writer CSV"),
                "sha256": _sha256_file(csv_path),
            },
            "markdown": {
                "path": _repo_relative_path(markdown_path, label="writer Markdown"),
                "sha256": _sha256_file(markdown_path),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def writer_report_main(argv: Sequence[str] | None = None) -> int:
    # The existing CLI remains the single entrypoint. Select isolated W2/W3
    # reporters only from the manifest phase; task names or output paths cannot
    # silently switch protocols.
    phase_probe = argparse.ArgumentParser(add_help=False)
    phase_probe.add_argument("--manifest", type=Path)
    phase_args, _ = phase_probe.parse_known_args(argv)
    if phase_args.manifest is not None:
        try:
            raw_manifest = json.loads(phase_args.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_manifest = None
        phase = raw_manifest.get("phase") if isinstance(raw_manifest, dict) else None
        if phase == "writer_diagnostic_w3":
            from anamnesis.writer_report_w3 import writer_report_w3_main

            return writer_report_w3_main(argv)
        if phase == "writer_diagnostic_w2":
            from anamnesis.writer_report_w2 import writer_report_w2_main

            return writer_report_w2_main(argv)

    parser = argparse.ArgumentParser(
        description="Validate and score the strict local W1 writer diagnostic"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("eval/scenarios/writer_diagnostic.v1.jsonl"),
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=WRITER_CSV_PATH)
    parser.add_argument("--markdown", type=Path, default=WRITER_MARKDOWN_PATH)
    parser.add_argument("--provenance", type=Path, default=WRITER_PROVENANCE_PATH)
    args = parser.parse_args(argv)

    # Deliberately validate the measured run before opening the gold reference.
    sources_without_reference = [args.manifest, args.scenarios, args.run]
    _validate_output_locations(
        sources=sources_without_reference,
        csv_path=args.csv,
        markdown_path=args.markdown,
        provenance_path=args.provenance,
    )
    run_sha256 = _sha256_file(args.run)
    manifest, scenarios, manifest_sha256 = _validate_frozen_writer_manifest(
        manifest_path=args.manifest,
        scenarios_path=args.scenarios,
    )
    runs = _load_writer_run(
        args.run,
        manifest=manifest,
        manifest_path=args.manifest,
        manifest_sha256=manifest_sha256,
        scenarios_path=args.scenarios,
        scenarios=scenarios,
    )

    reference_pin = manifest.writer_reference
    assert reference_pin is not None
    reference_path = _repo_file(reference_pin.path, label="writer reference")
    all_sources = _all_writer_source_paths(
        manifest=manifest,
        manifest_path=args.manifest,
        scenarios_path=args.scenarios,
        reference_path=reference_path,
        run_path=args.run,
    )
    _validate_output_locations(
        sources=list(all_sources.values()),
        csv_path=args.csv,
        markdown_path=args.markdown,
        provenance_path=args.provenance,
    )
    _require_hash(reference_path, reference_pin.sha256, label="writer reference")
    reference = load_oracle_artifact(reference_path, scenarios)
    metrics = _metrics(scenarios, runs, reference)

    rendered = _render_markdown(metrics)
    _write_csv(args.csv, metrics)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(rendered, encoding="utf-8")
    _write_provenance(
        args.provenance,
        manifest=manifest,
        manifest_path=args.manifest,
        manifest_sha256=manifest_sha256,
        scenarios_path=args.scenarios,
        reference_path=reference_path,
        reference=reference,
        run_path=args.run,
        run_sha256=run_sha256,
        csv_path=args.csv,
        markdown_path=args.markdown,
    )
    print(rendered, end="")
    return 0 if metrics["gate_passed"] else 2


__all__ = [
    "WRITER_CSV_PATH",
    "WRITER_MARKDOWN_PATH",
    "WRITER_PROVENANCE_PATH",
    "WRITER_TITLE",
    "writer_report_main",
]
