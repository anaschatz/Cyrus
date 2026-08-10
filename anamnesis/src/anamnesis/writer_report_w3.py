"""Strict reporting for the frozen local W3 memory-writer diagnostic.

This reporter is deliberately isolated from the W1/W2 reporters.  It first
validates the complete v4 measured log, including the contemporaneous
C1-C8,D1 setup gate and every raw model call.  Only after that succeeds does
it resolve and open the reporter-only oracle.  The writer gate compares
deterministically replayed due-candidate multisets; decision actions are a
diagnostic and never affect the gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeAlias

from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ChatMessageUser

from anamnesis.inspect_adapter import SCENARIO_RUN_STORE_KEY
from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.local_experiment import (
    LocalExperimentManifest,
    validate_zero_api_pricing,
    verify_static_local_inputs,
)
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
    LOCAL_PREFLIGHT_W3_STORE_KEY,
    LOCAL_SCENARIO_W3_TASK_VERSION,
    LocalDecisionWire,
    LocalModelPreflightW3Result,
    build_local_decision_prompt,
    load_local_w3_preflight_fixture,
    load_local_w3_preflight_protocol,
    local_decision_prompt_contract,
    local_decision_schema_contract,
    local_memory_compiler_schema_contract,
    local_memory_compiler_w3_prompt_contract,
    local_system_config_sha256,
)
from anamnesis.local_wire import (
    LOCAL_MEMORY_COMPILER_W3_VERSION,
    build_local_memory_compiler_w3_prompt,
)
from anamnesis.memory import InMemoryAnamnesis, MemoryDelta
from anamnesis.oracle import (
    OracleCompilerArtifact,
    load_oracle_artifact,
    oracle_artifact_sha256,
)
from anamnesis.schema import Decision, ProposedAction, Scenario, ScenarioRun, Usage
from anamnesis.scoring import aggregate_results, score_scenario
from anamnesis.writer_report import (
    _memory_delta_from_audit_json,
    _require_hash,
    _validate_checkpoint_delta_binding,
    _validate_output_locations,
)

WRITER_W3_TITLE = "Local writer W3 diagnostic — not a hypothesis test"
WRITER_W3_PHASE = "writer_diagnostic_w3"
WRITER_W3_TASK_NAME = "local_anamnesis_writer_diagnostic_w3"
WRITER_W3_DATASET_NAME = "anamnesis-local-writer_diagnostic_w3-v0"
WRITER_W3_DATASET_SPLIT = "writer_diagnostic_w3"
WRITER_W3_SYSTEM = "anamnesis"
WRITER_W3_SCENARIOS = 10
WRITER_W3_CHECKPOINTS = 62
WRITER_W3_COMPILER_CALLS = 39
WRITER_W3_SETUP_COMPILER_CALLS = 8
WRITER_W3_SETUP_DECISION_CALLS = 1
WRITER_W3_SETUP_CALLS = 9
WRITER_W3_CSV_PATH = Path("results/local_writer_w3.csv")
WRITER_W3_MARKDOWN_PATH = Path("results/local_writer_w3.md")
WRITER_W3_PROVENANCE_PATH = Path("results/local_writer_w3.provenance.json")
WRITER_W3_SCENARIOS_PATH = Path("eval/scenarios/writer_diagnostic.v4.jsonl")
WRITER_W3_REFERENCE_PATH = Path("eval/oracle/writer_diagnostic_memory_deltas.v4.json")
WRITER_W3_FIXTURE_PATH = Path("eval/preflight/local_writer_w3.v1.json")
WRITER_W3_PROTOCOL_PATH = Path("eval/preflight/local_writer_w3.protocol.v1.json")

# These public preregistration pins are repeated here intentionally.  The
# manifest and runtime must independently agree with them.
WRITER_W3_DATASET_SHA256 = (
    "6b2530cb9f3426c792500f07e854d7f31ad84081ac77104cb8032737234ff91c"
)
WRITER_W3_REFERENCE_SHA256 = (
    "72308bb34bda758cc72dc651e3f0fd2fd2bd1bff820479e2cf0774ee8d66cf5c"
)
WRITER_W3_FIXTURE_SHA256 = (
    "5628c3c1d7f8e1a5da43d6e567d55ac8e4fbabd8b9c4054325de6f4def1da30c"
)
WRITER_W3_PROTOCOL_SHA256 = (
    "7f63c156a8af74ced2d5e5530b3e8083da95c7f46e14e1afafaaf864b3ce1915"
)
WRITER_W3_PROMPT_SHA256 = (
    "412a63d6b42ea6b5e294401cabbcbacf5a6b7facddbd8fe04ca7b91914c141e5"
)
WRITER_W3_SCHEMA_SHA256 = (
    "8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030"
)

W3CandidateKey: TypeAlias = tuple[
    str,  # checkpoint event ID
    str,  # stable action key
    str,  # due_at ISO-8601
    str,  # action kind
    str,  # canonical payload JSON
    tuple[str, ...],  # lexically sorted evidence event IDs
]


def _repo_file(relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} path must be repository-relative")
    resolved = (REPO_ROOT / candidate).resolve()
    if not resolved.is_relative_to(REPO_ROOT) or not resolved.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return resolved


def _validate_w3_manifest_identity(manifest: LocalExperimentManifest) -> None:
    if manifest.status != "frozen" or manifest.phase != WRITER_W3_PHASE:
        raise ValueError(
            "W3 writer report requires a frozen writer_diagnostic_w3 manifest"
        )
    if manifest.hypothesis_test_eligible is not False:
        raise ValueError("W3 writer diagnostic cannot be hypothesis-test eligible")
    if manifest.compiler_mode != "llm":
        raise ValueError("W3 writer diagnostic requires compiler_mode=llm")
    if manifest.systems != [WRITER_W3_SYSTEM]:
        raise ValueError("W3 writer diagnostic requires exactly the anamnesis system")
    if manifest.scenario_count != WRITER_W3_SCENARIOS:
        raise ValueError("W3 writer diagnostic requires exactly 10 scenarios")
    if manifest.execution.repetitions != 1 or manifest.execution.seeds != [101]:
        raise ValueError("W3 writer diagnostic requires one repetition with seed 101")
    if not manifest.model.same_model_for_compiler_and_decision:
        raise ValueError("W3 writer diagnostic requires one compiler/decision model")
    if manifest.writer_reference is None:
        raise ValueError("W3 writer diagnostic is missing writer_reference")
    if manifest.preflight_fixture is None:
        raise ValueError("W3 writer diagnostic is missing preflight_fixture")
    if manifest.preflight_protocol is None:
        raise ValueError("W3 writer diagnostic is missing preflight_protocol")
    if manifest.oracle_annotations is not None:
        raise ValueError("W3 writer diagnostic cannot expose oracle_annotations")
    if set(manifest.system_config_sha256) != {WRITER_W3_SYSTEM}:
        raise ValueError("W3 writer manifest has the wrong system hash matrix")


def _validate_frozen_w3_manifest(
    *,
    manifest_path: Path,
    scenarios_path: Path,
    command_runner: Callable[..., Any] | None = None,
) -> tuple[
    LocalExperimentManifest,
    list[Scenario],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    """Validate every measured input without resolving the gold reference."""

    manifest_bytes = manifest_path.read_bytes()
    manifest = LocalExperimentManifest.model_validate_json(manifest_bytes)
    _validate_w3_manifest_identity(manifest)

    expected_dataset = _repo_file(manifest.dataset.path, label="W3 writer dataset")
    if scenarios_path.resolve() != expected_dataset:
        raise ValueError("--scenarios differs from the frozen W3 writer dataset")
    verify_static_local_inputs(manifest, repo_root=REPO_ROOT)
    dataset_digest = _require_hash(
        scenarios_path, manifest.dataset.sha256, label="W3 writer dataset"
    )
    if dataset_digest != WRITER_W3_DATASET_SHA256:
        raise ValueError("W3 writer dataset differs from the public v4 pin")
    scenarios = load_scenarios(scenarios_path)
    if len(scenarios) != WRITER_W3_SCENARIOS:
        raise ValueError("W3 writer dataset must contain exactly 10 scenarios")
    if sum(len(scenario.events) for scenario in scenarios) != WRITER_W3_CHECKPOINTS:
        raise ValueError("W3 writer dataset must contain exactly 62 checkpoints")
    compiler_events = sum(
        event.kind != "clock_tick"
        for scenario in scenarios
        for event in scenario.events
    )
    if compiler_events != WRITER_W3_COMPILER_CALLS:
        raise ValueError("W3 writer dataset must contain exactly 39 compiler events")

    pricing_path = _repo_file(manifest.model.pricing.path, label="pricing config")
    pricing_sha256 = validate_zero_api_pricing(pricing_path, manifest.model.snapshot)
    if pricing_sha256 != manifest.model.pricing.sha256:
        raise ValueError("W3 writer pricing differs from the frozen manifest")

    fixture_pin = manifest.preflight_fixture
    protocol_pin = manifest.preflight_protocol
    assert fixture_pin is not None and protocol_pin is not None
    fixture_path = _repo_file(fixture_pin.path, label="W3 preflight fixture")
    protocol_path = _repo_file(protocol_pin.path, label="W3 preflight protocol")
    if (
        fixture_path != (REPO_ROOT / WRITER_W3_FIXTURE_PATH).resolve()
        or protocol_path != (REPO_ROOT / WRITER_W3_PROTOCOL_PATH).resolve()
    ):
        raise ValueError("W3 preflight paths differ from the public protocol")
    fixture_digest = _require_hash(
        fixture_path,
        fixture_pin.sha256,
        label="W3 preflight fixture",
    )
    if fixture_digest != WRITER_W3_FIXTURE_SHA256:
        raise ValueError("W3 preflight fixture differs from its public pin")
    protocol_digest = _require_hash(
        protocol_path,
        protocol_pin.sha256,
        label="W3 preflight protocol",
    )
    if protocol_digest != WRITER_W3_PROTOCOL_SHA256:
        raise ValueError("W3 preflight protocol differs from its public pin")
    fixture = load_local_w3_preflight_fixture(fixture_path)
    protocol = load_local_w3_preflight_protocol(protocol_path)

    if manifest.git_commit is None:
        raise ValueError("frozen W3 writer manifest is missing git_commit")
    preflight_path = _repo_file(manifest.model.preflight.path, label="W3 preflight log")
    _require_hash(
        preflight_path,
        manifest.model.preflight.sha256,
        label="W3 preflight log",
    )
    from anamnesis.local_preflight import validate_local_w3_preflight_artifact

    validate_local_w3_preflight_artifact(
        manifest.model.preflight.model_copy(update={"path": str(preflight_path)}),
        fixture_artifact=fixture_pin.model_copy(update={"path": str(fixture_path)}),
        protocol_artifact=protocol_pin.model_copy(update={"path": str(protocol_path)}),
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
            local_memory_compiler_w3_prompt_contract()
        ),
        "memory_compiler_schema_sha256": _sha256_text(
            local_memory_compiler_schema_contract()
        ),
    }
    for field, expected in contracts.items():
        if getattr(manifest, field) != expected:
            raise ValueError(f"W3 writer manifest {field} differs from runtime")
    if contracts["memory_compiler_prompt_sha256"] != WRITER_W3_PROMPT_SHA256:
        raise ValueError("W3 compiler prompt differs from the preregistered pin")
    if contracts["memory_compiler_schema_sha256"] != WRITER_W3_SCHEMA_SHA256:
        raise ValueError("W3 compiler schema differs from the preregistered pin")
    expected_system_hash = local_system_config_sha256(
        system=WRITER_W3_SYSTEM,
        top_k=manifest.embedding.top_k,
        embedding_model=manifest.embedding.model,
        embedding_repository=manifest.embedding.repository,
        embedding_revision=manifest.embedding.revision,
        embedding_artifact_sha256=manifest.embedding.artifact_sha256,
        pricing_config_sha256=manifest.model.pricing.sha256,
        compiler_prompt_variant="w3",
        w3_preflight_fixture_sha256=manifest.preflight_fixture.sha256,
        w3_preflight_protocol_sha256=manifest.preflight_protocol.sha256,
        w3_dataset_sha256=manifest.dataset.sha256,
        w3_reference_sha256=manifest.writer_reference.sha256,
    )
    if manifest.system_config_sha256 != {WRITER_W3_SYSTEM: expected_system_hash}:
        raise ValueError("W3 writer manifest system hash differs from runtime")
    return (
        manifest,
        scenarios,
        fixture,
        protocol,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _contains_raw_artifact(value: Any, artifact: dict[str, Any]) -> bool:
    """Detect a complete reporter/preflight artifact object or JSON string."""

    if value == artifact:
        return True
    frozen = json.dumps(
        artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    if isinstance(value, str):
        if value == frozen:
            return True
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return False
        return decoded == artifact
    if isinstance(value, dict):
        return any(_contains_raw_artifact(item, artifact) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_raw_artifact(item, artifact) for item in value)
    return False


def _reject_raw_preflight_leak(
    log: EvalLog, fixture: dict[str, Any], protocol: dict[str, Any]
) -> None:
    """Measured scenario state may contain results, never raw gate material."""

    spec = log.eval
    surfaces: list[Any] = [spec.task_args or {}, spec.metadata or {}]
    if log.samples is not None:
        surfaces.extend(sample.store for sample in log.samples)
    if any(
        _contains_raw_artifact(surface, artifact)
        for surface in surfaces
        for artifact in (fixture, protocol)
    ):
        raise ValueError("raw W3 preflight fixture/protocol leaked into measured state")


def _validate_exact_w3_prompts(
    *,
    sample: Any,
    scenario: Scenario,
    run: ScenarioRun,
    first_sample: bool,
) -> None:
    """Bind every scenario prompt to sanitized inputs and deterministic state."""

    raw_events = [event for event in sample.events if isinstance(event, ModelEvent)]
    cursor = WRITER_W3_SETUP_CALLS if first_sample else 0
    memory = InMemoryAnamnesis()
    for event, checkpoint in zip(
        scenario.to_runtime().events, run.checkpoints, strict=True
    ):
        delta: MemoryDelta | None = None
        if event.kind != "clock_tick":
            if cursor >= len(raw_events):
                raise ValueError("W3 writer sample is missing a compiler ModelEvent")
            compiler_event = raw_events[cursor]
            expected_prompt = build_local_memory_compiler_w3_prompt(
                event=event,
                active_state=memory.compiler_state(),
            )
            if (
                len(compiler_event.input) != 1
                or not isinstance(compiler_event.input[0], ChatMessageUser)
                or compiler_event.input[0].content != expected_prompt
            ):
                raise ValueError(
                    "W3 compiler prompt differs from sanitized event/current state"
                )
            if not checkpoint.compiler_parse_error:
                if checkpoint.memory_delta_json is None:
                    raise ValueError(
                        "parse-valid W3 writer checkpoint has no audit MemoryDelta"
                    )
                delta = _memory_delta_from_audit_json(checkpoint.memory_delta_json)
            cursor += 1
        applied = memory.ingest(event, delta)
        if event.kind != "clock_tick" and (
            applied.accepted != checkpoint.memory_delta_accepted
        ):
            raise ValueError("W3 prompt replay acceptance differs from checkpoint")
        selection = memory.select(event)
        if cursor >= len(raw_events):
            raise ValueError("W3 writer sample is missing a decision ModelEvent")
        decision_event = raw_events[cursor]
        expected_decision_prompt = build_local_decision_prompt(
            now=event.at.isoformat(),
            current_event_id=event.id,
            context_events=[event],
            decision_history=[],
            memory_view=selection.view,
        )
        if (
            len(decision_event.input) != 1
            or not isinstance(decision_event.input[0], ChatMessageUser)
            or decision_event.input[0].content != expected_decision_prompt
        ):
            raise ValueError("W3 decision prompt differs from deterministic replay")
        decision = Decision()
        try:
            decision = LocalDecisionWire.model_validate_json(
                decision_event.output.completion
            ).to_domain()
        except (ValueError, TypeError):
            if not checkpoint.decision_parse_error:
                raise ValueError(
                    "parse-valid W3 checkpoint has invalid raw decision output"
                ) from None
        cursor += 1
        memory.commit(event, decision)
    if cursor != len(raw_events):
        raise ValueError("W3 prompt replay found unaccounted ModelEvents")


def _validate_w3_task_and_dataset(
    log: EvalLog,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    scenarios: list[Scenario],
) -> None:
    spec = log.eval
    if spec.task_registry_name != WRITER_W3_TASK_NAME:
        raise ValueError("unexpected W3 writer diagnostic task")
    if spec.task_version != LOCAL_SCENARIO_W3_TASK_VERSION:
        raise ValueError("W3 writer task version differs from the runtime contract")
    if (
        spec.task_file is None
        or _resolve_logged_repo_path(spec.task_file)
        != (REPO_ROOT / LOCAL_TASK_FILE).resolve()
    ):
        raise ValueError("W3 writer log task file differs from the frozen task")

    task_args = dict(spec.task_args or {})
    if set(task_args) != {"seed", "repetition", "manifest", "ollama_models_dir"}:
        raise ValueError("W3 writer task arguments are missing or contain extras")
    if task_args.get("seed") != 101 or task_args.get("repetition") != 1:
        raise ValueError("W3 writer task seed/repetition differs from the matrix")
    logged_manifest = task_args.get("manifest")
    if (
        not isinstance(logged_manifest, str)
        or _resolve_logged_repo_path(logged_manifest) != manifest_path.resolve()
    ):
        raise ValueError("W3 writer task is not bound to this frozen manifest")
    _validate_path_task_arg(task_args, "ollama_models_dir")

    ids = [scenario.id for scenario in scenarios]
    fixture_sha = (
        manifest.preflight_fixture.sha256
        if manifest.preflight_fixture is not None
        else None
    )
    protocol_sha = (
        manifest.preflight_protocol.sha256
        if manifest.preflight_protocol is not None
        else None
    )
    required_metadata = {
        "track": "local_zero_api_cost",
        "claim_scope": "diagnostic_development_only",
        "hypothesis_test_eligible": False,
        "system": WRITER_W3_SYSTEM,
        "dataset": manifest.dataset.path,
        "dataset_split": WRITER_W3_DATASET_SPLIT,
        "dataset_scenario_count": WRITER_W3_SCENARIOS,
        "dataset_sample_ids": ids,
        "canonical_dataset_sha256": dataset_sha256(scenarios),
        "repetition": 1,
        "manifest_sha256": manifest_sha256,
        "live_semantic_preflight_required": True,
        "provider_api_cost_usd": 0.0,
        "pricing_config_sha256": manifest.model.pricing.sha256,
        "electricity_measured": False,
        "decision_prompt_version": LOCAL_DECISION_VERSION,
        "compiler_mode": "llm",
        "memory_compiler_prompt_version": LOCAL_MEMORY_COMPILER_W3_VERSION,
        "preflight_fixture_sha256": fixture_sha,
        "preflight_protocol_sha256": protocol_sha,
        "setup_preflight_task": "local_model_preflight_w3",
        "setup_preflight_model_calls": WRITER_W3_SETUP_CALLS,
        "setup_preflight_compiler_calls": WRITER_W3_SETUP_COMPILER_CALLS,
        "setup_preflight_decision_calls": WRITER_W3_SETUP_DECISION_CALLS,
        "setup_preflight_usage_in_headline": False,
        "same_model_for_compiler_and_decision": True,
        "scenario_compiler_model_calls": WRITER_W3_COMPILER_CALLS,
        "dataset_checkpoint_count": WRITER_W3_CHECKPOINTS,
        "intervention": "bundled-repair",
    }
    metadata = dict(spec.metadata or {})
    if metadata != required_metadata:
        changed = sorted(
            key
            for key in set(metadata) | set(required_metadata)
            if metadata.get(key) != required_metadata.get(key)
        )
        raise ValueError(f"W3 writer task metadata differs: {changed}")

    dataset = spec.dataset
    if dataset.name != WRITER_W3_DATASET_NAME:
        raise ValueError("W3 writer log dataset name differs from the protocol")
    if dataset.location is None or scenarios_path.resolve() not in (
        _logged_dataset_candidates(dataset.location)
    ):
        raise ValueError("W3 writer log dataset location differs from --scenarios")
    if dataset.samples != WRITER_W3_SCENARIOS:
        raise ValueError("W3 writer log dataset must contain exactly 10 samples")
    if tuple(str(item) for item in (dataset.sample_ids or [])) != tuple(ids):
        raise ValueError("W3 writer log dataset IDs/order differ from --scenarios")
    if dataset.shuffled is not False:
        raise ValueError("W3 writer diagnostic dataset must be unshuffled")


def _validate_w3_setup_latency(recorded_ms: float, expected_ms: float) -> None:
    """The first run must contain exactly the nine-call gate latency."""

    if not math.isclose(recorded_ms, expected_ms, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("first W3 run setup latency differs from the exact preflight")


def _load_w3_writer_run(
    path: Path,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    scenarios: list[Scenario],
    fixture: dict[str, Any],
    protocol: dict[str, Any],
) -> list[ScenarioRun]:
    """Validate one complete W3 measured log without reading its reference."""

    if path.suffix != ".eval" or not path.is_file():
        raise ValueError("W3 writer report requires exactly one Inspect .eval log")
    log = read_eval_log(path, resolve_attachments=True)
    _validate_effective_log_policy(log, manifest)
    _validate_w3_task_and_dataset(
        log,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        scenarios_path=scenarios_path,
        scenarios=scenarios,
    )
    if log.samples is None or len(log.samples) != WRITER_W3_SCENARIOS:
        raise ValueError("W3 writer log requires exactly 10 ordered samples")
    ids = [scenario.id for scenario in scenarios]
    if [str(sample.id) for sample in log.samples] != ids:
        raise ValueError("W3 writer samples differ from dataset ID/order")
    _reject_raw_preflight_leak(log, fixture, protocol)
    required_store_keys = {
        LOCAL_PREFLIGHT_W3_STORE_KEY,
        SCENARIO_RUN_STORE_KEY,
    }
    if any(
        not isinstance(sample.store, dict) or set(sample.store) != required_store_keys
        for sample in log.samples
    ):
        raise ValueError("W3 writer sample store keys differ from the protocol")

    serialized_preflights = {
        json.dumps(sample.store.get(LOCAL_PREFLIGHT_W3_STORE_KEY), sort_keys=True)
        for sample in log.samples
    }
    if len(serialized_preflights) != 1 or "null" in serialized_preflights:
        raise ValueError("W3 writer samples disagree about setup preflight evidence")
    preflight = LocalModelPreflightW3Result.model_validate_json(
        next(iter(serialized_preflights))
    )

    from anamnesis.local_preflight import validate_local_w3_preflight_model_events

    by_id = {scenario.id: scenario for scenario in scenarios}
    raw_log_usage = Usage(cost_usd=0.0)
    headline_usage = Usage(cost_usd=0.0)
    runs: list[ScenarioRun] = []
    compiler_calls = 0
    decision_calls = 0
    for index, sample in enumerate(log.samples):
        run = _extract_scenario_run(sample)
        scenario = by_id.get(run.scenario_id)
        if scenario is None:
            raise ValueError("W3 writer ScenarioRun does not match the dataset")
        required_sample_metadata = {
            "scenario": scenario.to_runtime().model_dump(mode="json"),
            "scenario_sha256": canonical_sha256(scenario),
            LOCAL_PREFLIGHT_W3_STORE_KEY: preflight.model_dump(mode="json"),
        }
        if sample.metadata != required_sample_metadata:
            raise ValueError("W3 writer sample metadata differs from measured inputs")

        raw_events = [event for event in sample.events if isinstance(event, ModelEvent)]
        setup_count = WRITER_W3_SETUP_CALLS if index == 0 else 0
        if index == 0:
            setup_usage = validate_local_w3_preflight_model_events(
                raw_events[:setup_count],
                fixture=fixture,
                result=preflight,
                seed=101,
            )
            raw_log_usage = raw_log_usage.plus(setup_usage)
            _validate_w3_setup_latency(run.setup_latency_ms, preflight.setup_latency_ms)
        elif not math.isclose(run.setup_latency_ms, 0.0, abs_tol=1e-12):
            raise ValueError("W3 setup latency must be recorded only on first sample")

        scenario_sample = SimpleNamespace(events=raw_events[setup_count:])
        validation_run = run.model_copy(update={"setup_latency_ms": 0.0})
        raw_usage, measured_usage = _validate_run_and_events(
            sample=scenario_sample,
            scenario=scenario,
            run=validation_run,
            system=WRITER_W3_SYSTEM,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            first_sample=False,
            expected_preflight=preflight,  # ignored when first_sample=False
        )
        for checkpoint in run.checkpoints:
            _validate_checkpoint_delta_binding(checkpoint)
        _validate_exact_w3_prompts(
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
        decision_calls += len(run.checkpoints)
        runs.append(run)

    if compiler_calls != WRITER_W3_COMPILER_CALLS:
        raise ValueError("W3 run must contain exactly 39 scenario compiler calls")
    if decision_calls != WRITER_W3_CHECKPOINTS:
        raise ValueError("W3 run must contain exactly 62 scenario decision calls")
    expected_raw = headline_usage.plus(preflight.usage)
    if raw_log_usage != expected_raw:
        raise ValueError("W3 raw calls differ from headline plus exact setup preflight")
    _validate_log_usage(log, raw_log_usage)
    keys = [(run.scenario_id, run.system, run.repetition, run.model) for run in runs]
    if len(keys) != WRITER_W3_SCENARIOS or len(keys) != len(set(keys)):
        raise ValueError("W3 writer report requires 10 unique ScenarioRuns")
    return runs


def _w3_candidate_key(checkpoint: str, candidate: Any) -> W3CandidateKey:
    """Public six-field W3 key; UX text and runtime-local IDs are excluded."""

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
        tuple(sorted(candidate.evidence_event_ids)),
    )


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


def _replay_w3_candidate_counters(
    scenarios: list[Scenario],
    runs: list[ScenarioRun],
    reference: OracleCompilerArtifact,
) -> tuple[Counter[W3CandidateKey], Counter[W3CandidateKey], int, int, int]:
    """Replay measured/reference deltas through independent fresh stores."""

    by_run = {run.scenario_id: run for run in runs}
    if len(by_run) != len(runs):
        raise ValueError("W3 writer runs contain duplicate scenario IDs")
    measured_candidates: Counter[W3CandidateKey] = Counter()
    reference_candidates: Counter[W3CandidateKey] = Counter()
    parse_invalid = 0
    semantic_invalid = 0
    accepted = 0

    for scenario in scenarios:
        run = by_run.get(scenario.id)
        if run is None:
            raise ValueError(f"W3 writer run is missing scenario {scenario.id}")
        runtime = scenario.to_runtime()
        measured_memory = InMemoryAnamnesis()
        reference_memory = InMemoryAnamnesis()
        reference_records = iter(reference.records_for(runtime))

        for event, checkpoint in zip(runtime.events, run.checkpoints, strict=True):
            measured_delta: MemoryDelta | None = None
            reference_delta: MemoryDelta | None = None
            if event.kind != "clock_tick":
                try:
                    record = next(reference_records)
                except StopIteration:
                    raise ValueError(
                        "W3 writer reference ended before the dataset"
                    ) from None
                if record.event_id != event.id:
                    raise ValueError("W3 writer reference order differs from dataset")
                reference_delta = record.delta
                if checkpoint.compiler_parse_error:
                    parse_invalid += 1
                elif checkpoint.memory_delta_json is None:
                    raise ValueError(
                        "parse-valid W3 writer checkpoint has no MemoryDelta"
                    )
                else:
                    measured_delta = _memory_delta_from_audit_json(
                        checkpoint.memory_delta_json
                    )

            measured_apply = measured_memory.ingest(event, measured_delta)
            reference_apply = reference_memory.ingest(event, reference_delta)
            if not reference_apply.accepted:
                raise ValueError("frozen W3 writer reference contains a rejected delta")
            if event.kind != "clock_tick":
                if measured_apply.accepted != checkpoint.memory_delta_accepted:
                    raise ValueError(
                        "measured W3 replay acceptance differs from checkpoint"
                    )
                if measured_apply.accepted:
                    accepted += 1
                elif not checkpoint.compiler_parse_error:
                    semantic_invalid += 1

            measured_selection = measured_memory.select(event)
            reference_selection = reference_memory.select(event)
            if list(measured_selection.due_candidate_ids) != (
                checkpoint.due_candidate_ids
            ):
                raise ValueError("measured W3 replay due IDs differ from checkpoint")
            measured_candidates.update(
                _w3_candidate_key(event.id, candidate)
                for candidate in measured_selection.due_candidates
            )
            reference_candidates.update(
                _w3_candidate_key(event.id, candidate)
                for candidate in reference_selection.due_candidates
            )
            _commit_all_due(measured_memory, event, measured_selection.due_candidates)
            _commit_all_due(reference_memory, event, reference_selection.due_candidates)

        try:
            next(reference_records)
        except StopIteration:
            pass
        else:
            raise ValueError("W3 writer reference contains unconsumed event records")

    return (
        measured_candidates,
        reference_candidates,
        parse_invalid,
        semantic_invalid,
        accepted,
    )


def _candidate_confusion(
    measured: Counter[W3CandidateKey], gold: Counter[W3CandidateKey]
) -> tuple[int, int, int]:
    """Return commutative multiset TP/FP/FN without collapsing duplicates."""

    return (
        sum((measured & gold).values()),
        sum((measured - gold).values()),
        sum((gold - measured).values()),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _w3_metrics(
    scenarios: list[Scenario],
    runs: list[ScenarioRun],
    reference: OracleCompilerArtifact,
) -> dict[str, object]:
    measured, gold, parse_invalid, semantic_invalid, accepted = (
        _replay_w3_candidate_counters(scenarios, runs, reference)
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
        raise ValueError("W3 final-action diagnostic must aggregate to one row")
    actions = action_results[0]

    compiler_usage = Usage(cost_usd=0.0)
    decision_usage = Usage(cost_usd=0.0)
    for run in runs:
        if not run.usage_complete or not run.cost_complete:
            raise ValueError("W3 writer run has incomplete usage or cost")
        compiler_usage = compiler_usage.plus(run.compiler_usage)
        decision_usage = decision_usage.plus(run.decision_usage)
    total_usage = compiler_usage.plus(decision_usage)
    if total_usage.cost_usd != 0.0:
        raise ValueError("W3 writer diagnostic must have exact zero provider API cost")

    gate = {
        "invalid_zero": parse_invalid == 0 and semantic_invalid == 0,
        "accepted_all": accepted == WRITER_W3_COMPILER_CALLS,
        "candidate_fp_zero": false_positive == 0,
        "candidate_fn_zero": false_negative == 0,
    }
    return {
        "title": WRITER_W3_TITLE,
        "hypothesis_test_eligible": False,
        "compiler_calls": WRITER_W3_COMPILER_CALLS,
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


def _render_w3_markdown(metrics: dict[str, object]) -> str:
    return (
        f"# {WRITER_W3_TITLE}\n\n"
        "This development-only diagnostic evaluates the bundled W3 LLM memory "
        "writer. It is not hypothesis-test evidence and is not a single-factor "
        "ablation. The gate is computed only from independently replayed "
        "due-candidate multisets. Summary text and runtime-local "
        "intent/occurrence IDs are excluded; final decision actions are "
        "diagnostic-only.\n\n"
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
        "Gate: zero parse/semantic invalid deltas, 39/39 accepted deltas, "
        "zero candidate false positives, and zero candidate false negatives. "
        "Provider API cost is exactly zero; electricity and hardware cost are "
        "unmeasured.\n"
    )


def _all_w3_source_paths(
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    scenarios_path: Path,
    reference_path: Path,
    run_path: Path,
) -> dict[str, Path]:
    """Resolve every immutable W3 input/source before creating outputs."""

    if manifest.preflight_fixture is None or manifest.preflight_protocol is None:
        raise ValueError("frozen W3 writer manifest is missing preflight pins")
    source_root = REPO_ROOT / "src/anamnesis"
    sources = {
        "manifest": manifest_path,
        "dataset": scenarios_path,
        "reference": reference_path,
        "run": run_path,
        "model_artifact": _repo_file(
            manifest.model.artifact.path, label="model artifact pin"
        ),
        "dependency_lock": _repo_file(
            manifest.dependency_lock.path, label="dependency lock"
        ),
        "research_contract": _repo_file(
            manifest.research_contract.path, label="research contract"
        ),
        "architecture_contract": _repo_file(
            manifest.architecture_contract.path, label="architecture contract"
        ),
        "reporter": Path(__file__).resolve(),
        "dispatcher": (source_root / "writer_report.py").resolve(),
        "task": (REPO_ROOT / LOCAL_TASK_FILE).resolve(),
        "local_runtime": (source_root / "local_runtime.py").resolve(),
        "local_preflight": (source_root / "local_preflight.py").resolve(),
        "local_wire": (source_root / "local_wire.py").resolve(),
        "local_experiment": (source_root / "local_experiment.py").resolve(),
        "local_report": (source_root / "local_report.py").resolve(),
        "memory": (source_root / "memory.py").resolve(),
        "oracle": (source_root / "oracle.py").resolve(),
        "schema": (source_root / "schema.py").resolve(),
        "scoring": (source_root / "scoring.py").resolve(),
        "io": (source_root / "io.py").resolve(),
        "model_preflight": _repo_file(
            manifest.model.preflight.path, label="W3 model preflight"
        ),
        "preflight_fixture": _repo_file(
            manifest.preflight_fixture.path, label="W3 preflight fixture"
        ),
        "preflight_protocol": _repo_file(
            manifest.preflight_protocol.path, label="W3 preflight protocol"
        ),
        "pricing": _repo_file(manifest.model.pricing.path, label="pricing config"),
    }
    if any(not path.is_file() for path in sources.values()):
        raise ValueError("one or more W3 reporter source files are missing")
    return sources


def _validate_w3_provenance_inputs(
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    manifest_sha256: str,
    scenarios_path: Path,
    reference_path: Path,
    run_path: Path,
    run_sha256: str,
    sources: dict[str, Path],
) -> dict[str, str]:
    """Finish every source-integrity check before any result file is written."""

    if (
        manifest.writer_reference is None
        or manifest.preflight_fixture is None
        or manifest.preflight_protocol is None
    ):
        raise ValueError("frozen W3 writer manifest is incomplete")
    if _sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("W3 writer manifest changed while reporting")
    dataset_digest = _require_hash(
        scenarios_path, manifest.dataset.sha256, label="W3 writer dataset"
    )
    reference_digest = _require_hash(
        reference_path,
        manifest.writer_reference.sha256,
        label="W3 writer reference",
    )
    if dataset_digest != WRITER_W3_DATASET_SHA256:
        raise ValueError("W3 writer dataset no longer matches the v4 pin")
    if reference_digest != WRITER_W3_REFERENCE_SHA256:
        raise ValueError("W3 writer reference no longer matches the v4 pin")
    if _sha256_file(run_path) != run_sha256:
        raise ValueError("W3 writer .eval log changed while reporting")
    _require_hash(
        sources["model_preflight"],
        manifest.model.preflight.sha256,
        label="W3 model preflight",
    )
    fixture_digest = _require_hash(
        sources["preflight_fixture"],
        manifest.preflight_fixture.sha256,
        label="W3 preflight fixture",
    )
    protocol_digest = _require_hash(
        sources["preflight_protocol"],
        manifest.preflight_protocol.sha256,
        label="W3 preflight protocol",
    )
    if fixture_digest != WRITER_W3_FIXTURE_SHA256:
        raise ValueError("W3 fixture differs from the public pin")
    if protocol_digest != WRITER_W3_PROTOCOL_SHA256:
        raise ValueError("W3 protocol differs from the public pin")
    _require_hash(
        sources["pricing"], manifest.model.pricing.sha256, label="pricing config"
    )
    return {name: _sha256_file(path) for name, path in sources.items()}


def _assert_source_snapshot(
    sources: dict[str, Path], expected_sha256: dict[str, str]
) -> None:
    if set(sources) != set(expected_sha256):
        raise ValueError("W3 source snapshot is incomplete")
    changed = [
        name
        for name, path in sources.items()
        if _sha256_file(path) != expected_sha256[name]
    ]
    if changed:
        raise ValueError(f"W3 immutable inputs changed while reporting: {changed}")


def _write_w3_provenance(
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
    sources: dict[str, Path],
    source_sha256: dict[str, str],
) -> None:
    _validate_output_locations(
        sources=list(sources.values()),
        csv_path=csv_path,
        markdown_path=markdown_path,
        provenance_path=path,
    )
    _assert_source_snapshot(sources, source_sha256)
    if (
        manifest.git_commit is None
        or manifest.writer_reference is None
        or manifest.preflight_fixture is None
        or manifest.preflight_protocol is None
    ):
        raise ValueError("frozen W3 writer manifest is incomplete")

    def pin(name: str, label: str) -> dict[str, str]:
        return {
            "path": _repo_relative_path(sources[name], label=label),
            "sha256": source_sha256[name],
        }

    source_names = (
        "reporter",
        "dispatcher",
        "task",
        "local_runtime",
        "local_preflight",
        "local_wire",
        "local_experiment",
        "local_report",
        "memory",
        "oracle",
        "schema",
        "scoring",
        "io",
    )
    payload = {
        "schema_version": 1,
        "artifact": "anamnesis_local_w3_writer_result_provenance",
        "title": WRITER_W3_TITLE,
        "hypothesis_test_eligible": False,
        "intervention": "bundled-repair",
        "causal_interpretation": "multi_factor_not_single_ablation",
        "source_git_commit": manifest.git_commit,
        "candidate_matching": {
            "version": "w3.candidate-key.v1",
            "comparison": "commutative_multiset",
            "key_fields": [
                "checkpoint",
                "action_key",
                "due_at",
                "kind",
                "canonical_payload",
                "sorted_evidence",
            ],
            "excluded": ["summary", "intent_id", "occurrence_id"],
        },
        "protocol_contracts": {
            "compiler_prompt_version": LOCAL_MEMORY_COMPILER_W3_VERSION,
            "compiler_prompt_sha256": manifest.memory_compiler_prompt_sha256,
            "compiler_schema_sha256": manifest.memory_compiler_schema_sha256,
            "decision_prompt_version": LOCAL_DECISION_VERSION,
            "decision_prompt_sha256": manifest.decision_prompt_sha256,
            "decision_schema_sha256": manifest.decision_schema_sha256,
            "system_config_sha256": manifest.system_config_sha256,
        },
        "source": {name: pin(name, f"W3 source {name}") for name in source_names},
        "frozen_manifest": {
            "path": _repo_relative_path(manifest_path, label="W3 writer manifest"),
            "sha256": manifest_sha256,
        },
        "scenario_dataset": {
            "path": _repo_relative_path(scenarios_path, label="W3 writer dataset"),
            "sha256": source_sha256["dataset"],
            "canonical_sha256": dataset_sha256(load_scenarios(scenarios_path)),
            "scenario_count": WRITER_W3_SCENARIOS,
            "checkpoint_count": WRITER_W3_CHECKPOINTS,
            "compiler_event_count": WRITER_W3_COMPILER_CALLS,
        },
        "writer_reference": {
            "path": _repo_relative_path(reference_path, label="W3 writer reference"),
            "sha256": source_sha256["reference"],
            "canonical_semantics_sha256": oracle_artifact_sha256(reference),
            "opened_only_after_run_validation": True,
        },
        "model_preflight": {
            **pin("model_preflight", "W3 model preflight"),
            "setup_calls_in_measured_log": [
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "C7",
                "C8",
                "D1",
            ],
            "headline_usage_excludes_setup": True,
        },
        "preflight_fixture": {
            **pin("preflight_fixture", "W3 preflight fixture"),
            "raw_fixture_visible_to_scenario_writer": False,
        },
        "preflight_protocol": {
            **pin("preflight_protocol", "W3 preflight protocol"),
            "raw_protocol_visible_to_scenario_writer": False,
        },
        "pinned_inputs": {
            "model_artifact": pin("model_artifact", "model artifact pin"),
            "dependency_lock": pin("dependency_lock", "dependency lock"),
            "research_contract": pin("research_contract", "research contract"),
            "architecture_contract": pin(
                "architecture_contract", "architecture contract"
            ),
            "pricing_config": pin("pricing", "pricing config"),
        },
        "input_eval_log": {
            "path": _repo_relative_path(run_path, label="W3 writer input .eval log"),
            "sha256": run_sha256,
        },
        "outputs": {
            "csv": {
                "path": _repo_relative_path(csv_path, label="W3 writer CSV"),
                "sha256": _sha256_file(csv_path),
            },
            "markdown": {
                "path": _repo_relative_path(markdown_path, label="W3 writer Markdown"),
                "sha256": _sha256_file(markdown_path),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def writer_report_w3_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and score the strict local W3 writer diagnostic"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, default=WRITER_W3_SCENARIOS_PATH)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=WRITER_W3_CSV_PATH)
    parser.add_argument("--markdown", type=Path, default=WRITER_W3_MARKDOWN_PATH)
    parser.add_argument("--provenance", type=Path, default=WRITER_W3_PROVENANCE_PATH)
    args = parser.parse_args(argv)

    # The reference path is deliberately not resolved before measured-log
    # validation.  Integrity failures therefore cannot observe gold or create
    # any output artifact.
    _validate_output_locations(
        sources=[args.manifest, args.scenarios, args.run],
        csv_path=args.csv,
        markdown_path=args.markdown,
        provenance_path=args.provenance,
    )
    run_sha256 = _sha256_file(args.run)
    manifest, scenarios, fixture, protocol, manifest_sha256 = (
        _validate_frozen_w3_manifest(
            manifest_path=args.manifest,
            scenarios_path=args.scenarios,
        )
    )
    runs = _load_w3_writer_run(
        args.run,
        manifest=manifest,
        manifest_path=args.manifest,
        manifest_sha256=manifest_sha256,
        scenarios_path=args.scenarios,
        scenarios=scenarios,
        fixture=fixture,
        protocol=protocol,
    )

    reference_pin = manifest.writer_reference
    assert reference_pin is not None
    reference_path = _repo_file(reference_pin.path, label="W3 writer reference")
    if reference_path != (REPO_ROOT / WRITER_W3_REFERENCE_PATH).resolve():
        raise ValueError("W3 writer reference differs from the public v4 path")
    all_sources = _all_w3_source_paths(
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
    source_sha256 = _validate_w3_provenance_inputs(
        manifest=manifest,
        manifest_path=args.manifest,
        manifest_sha256=manifest_sha256,
        scenarios_path=args.scenarios,
        reference_path=reference_path,
        run_path=args.run,
        run_sha256=run_sha256,
        sources=all_sources,
    )
    reference = load_oracle_artifact(reference_path, scenarios)
    metrics = _w3_metrics(scenarios, runs, reference)
    rendered = _render_w3_markdown(metrics)

    # Recheck every immutable input immediately before first output creation.
    _assert_source_snapshot(all_sources, source_sha256)
    _write_csv(args.csv, metrics)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(rendered, encoding="utf-8")
    _write_w3_provenance(
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
        sources=all_sources,
        source_sha256=source_sha256,
    )
    print(rendered, end="")
    return 0 if metrics["gate_passed"] else 2


__all__ = [
    "WRITER_W3_CSV_PATH",
    "WRITER_W3_MARKDOWN_PATH",
    "WRITER_W3_PROVENANCE_PATH",
    "WRITER_W3_TITLE",
    "writer_report_w3_main",
]
