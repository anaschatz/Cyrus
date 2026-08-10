from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.model import ChatMessageUser

import anamnesis.writer_report as writer_report
from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.local_experiment import LocalExperimentManifest
from anamnesis.local_runtime import LOCAL_DECISION_VERSION, LOCAL_SCENARIO_TASK_VERSION
from anamnesis.local_wire import build_local_memory_compiler_prompt
from anamnesis.memory import CancelIntent, InMemoryAnamnesis, MemoryDelta
from anamnesis.oracle import OracleCompilerArtifact, load_oracle_artifact
from anamnesis.schema import (
    CheckpointAudit,
    Decision,
    ProposedAction,
    Scenario,
    ScenarioRun,
    Usage,
)
from anamnesis.writer_report import (
    WRITER_TITLE,
    _candidate_confusion,
    _memory_delta_from_audit_json,
    _metrics,
    _render_markdown,
    _replay_candidate_counters,
    _require_hash,
    _validate_checkpoint_delta_binding,
    _validate_exact_compiler_prompts,
    _validate_output_locations,
    _validate_writer_manifest_identity,
    _validate_writer_task_and_dataset,
    writer_report_main,
)

DATASET = Path("eval/scenarios/writer_diagnostic.v1.jsonl")
REFERENCE = Path("eval/oracle/writer_diagnostic_memory_deltas.v1.json")
TEMPLATE = Path("eval/local_writer_experiment_manifest.template.json")
ZERO_SHA256 = "0" * 64


def _commit_all(memory: InMemoryAnamnesis, event: object, candidates: object) -> None:
    actions = [
        ProposedAction(
            kind=candidate.action_template.kind,
            action_key=candidate.action_key,
            payload=dict(candidate.action_template.payload),
            summary=candidate.action_template.summary,
            evidence_event_ids=list(candidate.evidence_event_ids),
        )
        for candidate in candidates  # type: ignore[union-attr]
    ]
    memory.commit(event, Decision(actions=actions))  # type: ignore[arg-type]


def _reference_runs(
    scenarios: list[Scenario], reference: OracleCompilerArtifact
) -> list[ScenarioRun]:
    runs: list[ScenarioRun] = []
    zero = Usage(cost_usd=0.0)
    for scenario in scenarios:
        runtime = scenario.to_runtime()
        records = iter(reference.records_for(runtime))
        memory = InMemoryAnamnesis()
        checkpoints: list[CheckpointAudit] = []
        for event in runtime.events:
            delta = None if event.kind == "clock_tick" else next(records).delta
            applied = memory.ingest(event, delta)
            assert applied.accepted
            selection = memory.select(event)
            _commit_all(memory, event, selection.due_candidates)
            checkpoints.append(
                CheckpointAudit(
                    event_id=event.id,
                    at=event.at,
                    compiler_called=event.kind != "clock_tick",
                    raw_compiler_output=(
                        delta.model_dump_json() if delta is not None else None
                    ),
                    memory_delta_json=(
                        delta.model_dump_json() if delta is not None else None
                    ),
                    memory_delta_accepted=(
                        True if event.kind != "clock_tick" else None
                    ),
                    state_sha256=memory.state_hash(),
                    due_candidate_ids=list(selection.due_candidate_ids),
                    rendered_context_sha256=ZERO_SHA256,
                    raw_decision_output=Decision().model_dump_json(),
                    compiler_usage=zero,
                    decision_usage=zero,
                )
            )
        runs.append(
            ScenarioRun(
                scenario_id=scenario.id,
                system="anamnesis",
                repetition=1,
                model="ollama/qwen3:4b-instruct",
                prompt_version="test",
                scenario_sha256=canonical_sha256(scenario),
                prompt_sha256=ZERO_SHA256,
                system_config_sha256=ZERO_SHA256,
                predictions=[],
                usage=zero,
                decision_usage=zero,
                compiler_usage=zero,
                usage_complete=True,
                cost_complete=True,
                checkpoint_latency_ms=[0.0] * len(checkpoints),
                checkpoints=checkpoints,
            )
        )
    return runs


@pytest.fixture(scope="module")
def writer_fixture() -> tuple[
    list[Scenario], OracleCompilerArtifact, list[ScenarioRun]
]:
    scenarios = load_scenarios(DATASET)
    reference = load_oracle_artifact(REFERENCE, scenarios)
    return scenarios, reference, _reference_runs(scenarios, reference)


def test_valid_writer_fixture_passes_gate_and_decisions_are_excluded(
    writer_fixture: tuple[list[Scenario], OracleCompilerArtifact, list[ScenarioRun]],
) -> None:
    scenarios, reference, runs = writer_fixture
    metrics = _metrics(scenarios, runs, reference)

    assert metrics["compiler_calls"] == 45
    assert metrics["compiler_accepted"] == 45
    assert metrics["compiler_parse_invalid"] == 0
    assert metrics["compiler_semantic_invalid"] == 0
    assert metrics["candidate_fp"] == 0
    assert metrics["candidate_fn"] == 0
    assert metrics["gate_passed"] is True
    # These fixture runs intentionally emit no final actions. Their failure is
    # diagnostic and cannot alter the candidate-only writer gate.
    assert metrics["final_action_fn_diagnostic"] == 8
    assert metrics["final_action_f1_diagnostic"] == 0.0

    rendered = _render_markdown(metrics)
    assert rendered.startswith(f"# {WRITER_TITLE}\n")
    assert "Final-action diagnostic (excluded from the gate)" in rendered


def test_replay_counts_parse_and_semantic_invalid_deltas(
    writer_fixture: tuple[list[Scenario], OracleCompilerArtifact, list[ScenarioRun]],
) -> None:
    scenarios, reference, runs = writer_fixture
    index = next(
        i for i, scenario in enumerate(scenarios) if scenario.id.startswith("wd07")
    )
    scenario = scenarios[index]
    run = runs[index]

    parse_checkpoint = run.checkpoints[0].model_copy(
        update={
            "raw_compiler_output": "not-json",
            "memory_delta_json": None,
            "memory_delta_accepted": False,
            "compiler_parse_error": True,
        }
    )
    parse_run = run.model_copy(
        update={"checkpoints": [parse_checkpoint, *run.checkpoints[1:]]}
    )
    _, _, parse_invalid, semantic_invalid, accepted = _replay_candidate_counters(
        [scenario], [parse_run], reference
    )
    assert (parse_invalid, semantic_invalid, accepted) == (1, 0, 2)

    rejected_delta = MemoryDelta(mutations=(CancelIntent(intent_id="never_created"),))
    semantic_checkpoint = run.checkpoints[0].model_copy(
        update={
            "raw_compiler_output": rejected_delta.model_dump_json(),
            "memory_delta_json": rejected_delta.model_dump_json(),
            "memory_delta_accepted": False,
        }
    )
    semantic_run = run.model_copy(
        update={"checkpoints": [semantic_checkpoint, *run.checkpoints[1:]]}
    )
    _, _, parse_invalid, semantic_invalid, accepted = _replay_candidate_counters(
        [scenario], [semantic_run], reference
    )
    assert (parse_invalid, semantic_invalid, accepted) == (0, 1, 2)


def test_candidate_counter_reports_fp_and_fn_without_runtime_ids() -> None:
    left = (
        "checkpoint",
        "action-key",
        "2026-09-18T12:00:00+03:00",
        "reminder",
        '{"subject":"alpha"}',
        "Alpha",
        ("e1", "e2"),
    )
    right = (*left[:4], '{"subject":"beta"}', *left[5:])
    measured = Counter({left: 2, right: 1})
    gold = Counter({left: 1})

    assert _candidate_confusion(measured, gold) == (1, 2, 0)
    assert _candidate_confusion(Counter(), gold) == (0, 0, 1)
    assert len(left) == 7  # no intent_id or occurrence_id in the canonical key


def test_update_intent_audit_json_round_trips_unset_null_fields(
    writer_fixture: tuple[list[Scenario], OracleCompilerArtifact, list[ScenarioRun]],
) -> None:
    _, _, runs = writer_fixture
    serialized = next(
        checkpoint.memory_delta_json
        for run in runs
        for checkpoint in run.checkpoints
        if checkpoint.memory_delta_json
        and '"op":"update_intent"' in checkpoint.memory_delta_json
    )
    assert serialized is not None
    assert _memory_delta_from_audit_json(serialized).mutations


def test_raw_local_wire_output_is_bound_to_audit_delta_semantics() -> None:
    raw_empty = json.dumps(
        {
            "fact_assertions": [],
            "intent_creates": [],
            "intent_updates": [],
            "intent_cancellations": [],
        }
    )
    empty_delta = MemoryDelta()
    checkpoint = CheckpointAudit(
        event_id="event-1",
        at="2026-09-14T10:00:00+03:00",
        compiler_called=True,
        raw_compiler_output=raw_empty,
        memory_delta_json=empty_delta.model_dump_json(),
        memory_delta_accepted=True,
        rendered_context_sha256=ZERO_SHA256,
        raw_decision_output=Decision().model_dump_json(),
    )
    _validate_checkpoint_delta_binding(checkpoint)

    tampered = MemoryDelta(mutations=(CancelIntent(intent_id="different_semantics"),))
    with pytest.raises(ValueError, match="differs from audit"):
        _validate_checkpoint_delta_binding(
            checkpoint.model_copy(
                update={"memory_delta_json": tampered.model_dump_json()}
            )
        )

    invalid = checkpoint.model_copy(
        update={
            "raw_compiler_output": "not-json",
            "memory_delta_json": None,
            "memory_delta_accepted": False,
            "compiler_parse_error": True,
        }
    )
    _validate_checkpoint_delta_binding(invalid)
    with pytest.raises(ValueError, match="was not rejected"):
        _validate_checkpoint_delta_binding(
            invalid.model_copy(update={"memory_delta_accepted": True})
        )


def test_compiler_prompt_is_bound_to_exact_replayed_active_state(
    writer_fixture: tuple[list[Scenario], OracleCompilerArtifact, list[ScenarioRun]],
) -> None:
    scenarios, _, runs = writer_fixture
    scenario = scenarios[0]
    run = runs[0]
    memory = InMemoryAnamnesis()
    model_events: list[ModelEvent] = []
    for event, checkpoint in zip(
        scenario.to_runtime().events, run.checkpoints, strict=True
    ):
        delta = None
        if event.kind != "clock_tick":
            prompt = build_local_memory_compiler_prompt(
                event=event,
                active_state=memory.compiler_state(),
            )
            model_events.append(
                ModelEvent.model_construct(
                    event="model",
                    input=[ChatMessageUser(content=prompt)],
                )
            )
            assert checkpoint.memory_delta_json is not None
            delta = _memory_delta_from_audit_json(checkpoint.memory_delta_json)
        memory.ingest(event, delta)
        memory.select(event)
        model_events.append(
            ModelEvent.model_construct(
                event="model",
                input=[ChatMessageUser(content="decision")],
            )
        )
        memory.commit(event, Decision())

    sample = SimpleNamespace(events=model_events)
    _validate_exact_compiler_prompts(
        sample=sample,
        scenario=scenario,
        run=run,
        first_sample=False,
    )

    second_event = scenario.to_runtime().events[1]
    tampered_prompt = build_local_memory_compiler_prompt(
        event=second_event,
        active_state='{"facts":[],"intents":[]}',
    )
    model_events[2] = ModelEvent.model_construct(
        event="model",
        input=[ChatMessageUser(content=tampered_prompt)],
    )
    with pytest.raises(ValueError, match="current state"):
        _validate_exact_compiler_prompts(
            sample=sample,
            scenario=scenario,
            run=run,
            first_sample=False,
        )


def test_writer_manifest_identity_rejects_phase_and_missing_reference() -> None:
    draft = LocalExperimentManifest.model_validate_json(TEMPLATE.read_text())
    frozen = draft.model_copy(
        update={
            "status": "frozen",
            "system_config_sha256": {"anamnesis": ZERO_SHA256},
        }
    )
    _validate_writer_manifest_identity(frozen)

    with pytest.raises(ValueError, match="writer_diagnostic"):
        _validate_writer_manifest_identity(frozen.model_copy(update={"phase": "smoke"}))
    with pytest.raises(ValueError, match="writer_reference"):
        _validate_writer_manifest_identity(
            frozen.model_copy(update={"writer_reference": None})
        )


def test_reference_hash_tamper_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text("reference", encoding="utf-8")
    actual = __import__("hashlib").sha256(path.read_bytes()).hexdigest()

    assert _require_hash(path, actual, label="writer reference") == actual
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ"):
        _require_hash(path, actual, label="writer reference")


def test_writer_task_identity_config_and_dataset_are_exact() -> None:
    manifest = LocalExperimentManifest.model_validate_json(TEMPLATE.read_text())
    scenarios = load_scenarios(DATASET)
    ids = [scenario.id for scenario in scenarios]
    manifest_sha256 = "a" * 64
    metadata = {
        "track": "local_zero_api_cost",
        "claim_scope": "diagnostic_development_only",
        "hypothesis_test_eligible": False,
        "system": "anamnesis",
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
    dataset = SimpleNamespace(
        name="anamnesis-local-writer_diagnostic-v0",
        location=str(DATASET.resolve()),
        samples=10,
        sample_ids=ids,
        shuffled=False,
    )
    spec = SimpleNamespace(
        task_registry_name="local_anamnesis_writer_diagnostic",
        task_version=LOCAL_SCENARIO_TASK_VERSION,
        task_file="eval/anamnesis_local_eval.py",
        task_args={
            "seed": 101,
            "repetition": 1,
            "manifest": str(TEMPLATE.resolve()),
            "ollama_models_dir": "/absolute/ollama/models",
        },
        metadata=metadata,
        dataset=dataset,
    )
    log = SimpleNamespace(eval=spec)
    kwargs = {
        "manifest": manifest,
        "manifest_path": TEMPLATE,
        "manifest_sha256": manifest_sha256,
        "scenarios_path": DATASET,
        "scenarios": scenarios,
    }
    _validate_writer_task_and_dataset(log, **kwargs)  # type: ignore[arg-type]

    spec.task_registry_name = "local_anamnesis"
    with pytest.raises(ValueError, match="unexpected writer"):
        _validate_writer_task_and_dataset(log, **kwargs)  # type: ignore[arg-type]
    spec.task_registry_name = "local_anamnesis_writer_diagnostic"
    spec.task_args = {**spec.task_args, "seed": 202}
    with pytest.raises(ValueError, match="seed/repetition"):
        _validate_writer_task_and_dataset(log, **kwargs)  # type: ignore[arg-type]
    spec.task_args = {**spec.task_args, "seed": 101}
    dataset.name = "anamnesis-local-smoke-v0"
    with pytest.raises(ValueError, match="dataset name"):
        _validate_writer_task_and_dataset(log, **kwargs)  # type: ignore[arg-type]


def _failed_gate_metrics() -> dict[str, object]:
    return {
        "title": WRITER_TITLE,
        "hypothesis_test_eligible": False,
        "compiler_calls": 45,
        "compiler_parse_invalid": 0,
        "compiler_semantic_invalid": 0,
        "compiler_accepted": 45,
        "candidate_tp": 7,
        "candidate_fp": 0,
        "candidate_fn": 1,
        "candidate_precision": 1.0,
        "candidate_recall": 0.875,
        "candidate_f1": 0.933333,
        "compiler_input_tokens": 100,
        "compiler_output_tokens": 20,
        "decision_input_tokens": 200,
        "decision_output_tokens": 30,
        "total_input_tokens": 300,
        "total_output_tokens": 50,
        "compiler_latency_ms": 10.0,
        "decision_latency_ms": 20.0,
        "local_latency_ms": 1.0,
        "total_latency_ms": 31.0,
        "setup_latency_ms": 2.0,
        "provider_api_cost_usd": 0.0,
        "final_action_tp_diagnostic": 0,
        "final_action_fp_diagnostic": 0,
        "final_action_fn_diagnostic": 8,
        "final_action_precision_diagnostic": 0.0,
        "final_action_recall_diagnostic": 0.0,
        "final_action_f1_diagnostic": 0.0,
        "gate_invalid_zero": True,
        "gate_accepted_all": True,
        "gate_candidate_fp_zero": True,
        "gate_candidate_fn_zero": False,
        "gate_passed": False,
    }


def test_valid_gate_failure_writes_report_and_returns_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    scenarios_path = tmp_path / "scenarios.jsonl"
    run_path = tmp_path / "writer.eval"
    reference_path = tmp_path / "reference.json"
    for path in (manifest_path, scenarios_path, run_path, reference_path):
        path.write_text("fixture", encoding="utf-8")
    csv_path = tmp_path / "writer.csv"
    markdown_path = tmp_path / "writer.md"
    provenance_path = tmp_path / "writer.provenance.json"
    reference_pin = SimpleNamespace(path="reference.json", sha256="a" * 64)
    manifest = SimpleNamespace(writer_reference=reference_pin)

    monkeypatch.setattr(writer_report, "_validate_output_locations", lambda **_: None)
    monkeypatch.setattr(
        writer_report,
        "_validate_frozen_writer_manifest",
        lambda **_: (manifest, [object()], "b" * 64),
    )
    monkeypatch.setattr(writer_report, "_load_writer_run", lambda *_, **__: [object()])
    monkeypatch.setattr(writer_report, "_repo_file", lambda *_, **__: reference_path)
    monkeypatch.setattr(writer_report, "_all_writer_source_paths", lambda **_: {})
    monkeypatch.setattr(writer_report, "_require_hash", lambda *_, **__: "a" * 64)
    monkeypatch.setattr(
        writer_report, "load_oracle_artifact", lambda *_, **__: object()
    )
    monkeypatch.setattr(
        writer_report, "_metrics", lambda *_, **__: _failed_gate_metrics()
    )

    def write_provenance(path: Path, **_: object) -> None:
        path.write_text('{"gate":"failed"}\n', encoding="utf-8")

    monkeypatch.setattr(writer_report, "_write_provenance", write_provenance)
    exit_code = writer_report_main(
        [
            "--manifest",
            str(manifest_path),
            "--scenarios",
            str(scenarios_path),
            "--run",
            str(run_path),
            "--csv",
            str(csv_path),
            "--markdown",
            str(markdown_path),
            "--provenance",
            str(provenance_path),
        ]
    )

    assert exit_code == 2
    assert csv_path.is_file()
    assert "gate_passed" in csv_path.read_text(encoding="utf-8")
    assert "| FAIL |" in markdown_path.read_text(encoding="utf-8")
    assert provenance_path.is_file()


@pytest.mark.parametrize(
    "collision_name",
    ["reference", "research_contract", "model_artifact"],
)
def test_source_collision_is_rejected_before_any_output_write(
    collision_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    scenarios_path = tmp_path / "scenarios.jsonl"
    run_path = tmp_path / "writer.eval"
    reference_path = tmp_path / "results/reference.json"
    model_artifact_path = tmp_path / "results/model.pin.json"
    dependency_lock_path = tmp_path / "uv.lock"
    research_contract_path = tmp_path / "results/RESEARCH.md"
    architecture_contract_path = tmp_path / "ARCHITECTURE.md"
    reporter_path = tmp_path / "src/anamnesis/writer_report.py"
    task_path = tmp_path / "eval/anamnesis_local_eval.py"
    preflight_path = tmp_path / "preflight.eval"
    pricing_path = tmp_path / "pricing.json"
    for path in (
        manifest_path,
        scenarios_path,
        run_path,
        reference_path,
        model_artifact_path,
        dependency_lock_path,
        research_contract_path,
        architecture_contract_path,
        reporter_path,
        task_path,
        preflight_path,
        pricing_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable", encoding="utf-8")
    collision_path = {
        "reference": reference_path,
        "research_contract": research_contract_path,
        "model_artifact": model_artifact_path,
    }[collision_name]
    original_source = collision_path.read_bytes()
    manifest = SimpleNamespace(
        writer_reference=SimpleNamespace(
            path="results/reference.json", sha256="a" * 64
        ),
        model=SimpleNamespace(
            artifact=SimpleNamespace(path="results/model.pin.json"),
            preflight=SimpleNamespace(path="preflight.eval"),
            pricing=SimpleNamespace(path="pricing.json"),
        ),
        dependency_lock=SimpleNamespace(path="uv.lock"),
        research_contract=SimpleNamespace(path="results/RESEARCH.md"),
        architecture_contract=SimpleNamespace(path="ARCHITECTURE.md"),
    )

    monkeypatch.setattr(writer_report, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(writer_report, "__file__", str(reporter_path))
    monkeypatch.setattr(
        writer_report,
        "_repo_relative_path",
        lambda path, **_: Path(path).resolve().relative_to(tmp_path).as_posix(),
    )
    monkeypatch.setattr(
        writer_report,
        "_validate_frozen_writer_manifest",
        lambda **_: (manifest, [object()], "b" * 64),
    )
    monkeypatch.setattr(writer_report, "_load_writer_run", lambda *_, **__: [object()])

    markdown_path = tmp_path / "results/writer.md"
    provenance_path = tmp_path / "results/writer.provenance.json"
    with pytest.raises(ValueError, match="cannot overwrite source"):
        writer_report_main(
            [
                "--manifest",
                str(manifest_path),
                "--scenarios",
                str(scenarios_path),
                "--run",
                str(run_path),
                "--csv",
                str(collision_path),
                "--markdown",
                str(markdown_path),
                "--provenance",
                str(provenance_path),
            ]
        )

    assert collision_path.read_bytes() == original_source
    assert not markdown_path.exists()
    assert not provenance_path.exists()


@pytest.mark.parametrize(
    ("csv_relative", "message"),
    [
        ("outside.csv", "under results"),
        ("results/runs/writer.csv", "results/runs"),
    ],
)
def test_writer_outputs_are_confined_away_from_sources_and_raw_runs(
    csv_relative: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(writer_report, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        writer_report,
        "_repo_relative_path",
        lambda path, **_: Path(path).resolve().relative_to(tmp_path).as_posix(),
    )
    with pytest.raises(ValueError, match=message):
        _validate_output_locations(
            sources=[],
            csv_path=tmp_path / csv_relative,
            markdown_path=tmp_path / "results/writer.md",
            provenance_path=tmp_path / "results/writer.provenance.json",
        )


def test_reporter_source_declares_strict_task_dataset_and_sidecar_contract() -> None:
    source = Path("src/anamnesis/writer_report.py").read_text(encoding="utf-8")
    assert 'WRITER_TASK_NAME = "local_anamnesis_writer_diagnostic"' in source
    assert 'WRITER_DATASET_NAME = "anamnesis-local-writer_diagnostic-v0"' in source
    assert '"writer_reference": {' in source
    assert '"model_preflight": {' in source
    assert '"pricing_config": {' in source
    assert '"input_eval_log": {' in source
    assert '"outputs": {' in source
    assert "opened_only_after_run_validation" in source
    assert "WRITER_COMPILER_CALLS = 45" in source


def test_csv_payload_is_json_scalar_compatible(
    writer_fixture: tuple[list[Scenario], OracleCompilerArtifact, list[ScenarioRun]],
) -> None:
    scenarios, reference, runs = writer_fixture
    metrics = _metrics(scenarios, runs, reference)
    assert json.loads(json.dumps(metrics))["title"] == WRITER_TITLE
