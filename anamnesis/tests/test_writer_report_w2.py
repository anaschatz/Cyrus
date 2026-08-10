from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.model import ChatMessageUser, ModelOutput

import anamnesis.writer_report as writer_report_dispatcher
import anamnesis.writer_report_w2 as writer_report_w2
from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.local_experiment import LocalExperimentManifest
from anamnesis.local_runtime import (
    LOCAL_DECISION_VERSION,
    LOCAL_SCENARIO_TASK_VERSION,
    build_local_decision_prompt,
)
from anamnesis.local_wire import build_local_memory_compiler_w2_prompt
from anamnesis.memory import InMemoryAnamnesis
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
    _memory_delta_from_audit_json,
    _validate_writer_manifest_identity,
)
from anamnesis.writer_report_w2 import (
    WRITER_W2_COMPILER_CALLS,
    WRITER_W2_TITLE,
    _candidate_confusion,
    _contains_raw_fixture,
    _render_w2_markdown,
    _validate_exact_w2_prompts,
    _validate_w2_manifest_identity,
    _validate_w2_setup_latency,
    _validate_w2_task_and_dataset,
    _w2_candidate_key,
    _w2_metrics,
    writer_report_w2_main,
)

DATASET = Path("eval/scenarios/writer_diagnostic.v3.jsonl")
REFERENCE = Path("eval/oracle/writer_diagnostic_memory_deltas.v3.json")
W1_TEMPLATE = Path("eval/local_writer_experiment_manifest.template.json")
W2_TEMPLATE = Path("eval/local_writer_w2_experiment_manifest.template.json")
ZERO_SHA256 = "0" * 64


def _candidate(
    *,
    summary: str = "UX text",
    payload: dict[str, object] | None = None,
    evidence: list[str] | None = None,
    intent_id: str = "intent-local-a",
    occurrence_id: str = "occurrence-local-a",
) -> SimpleNamespace:
    return SimpleNamespace(
        intent_id=intent_id,
        occurrence_id=occurrence_id,
        action_key="source-event",
        due_at=datetime.fromisoformat("2027-04-12T12:00:00+03:00"),
        action_template=SimpleNamespace(
            kind="reminder",
            payload=payload or {"subject": "send note"},
            summary=summary,
        ),
        evidence_event_ids=evidence or ["e2", "e1"],
    )


def test_w2_candidate_key_excludes_summary_and_runtime_local_ids() -> None:
    left = _w2_candidate_key("checkpoint", _candidate())
    right = _w2_candidate_key(
        "checkpoint",
        _candidate(
            summary="Entirely different UX text",
            intent_id="intent-local-b",
            occurrence_id="occurrence-local-b",
        ),
    )

    assert left == right
    assert len(left) == 6
    assert "UX text" not in left
    assert "intent-local-a" not in left
    assert "occurrence-local-a" not in left
    assert _candidate_confusion(Counter({left: 1}), Counter({right: 1})) == (
        1,
        0,
        0,
    )


def test_w2_candidate_key_payload_or_evidence_drift_is_fp_and_fn() -> None:
    gold = _w2_candidate_key("checkpoint", _candidate())
    payload_drift = _w2_candidate_key(
        "checkpoint", _candidate(payload={"subject": "send another note"})
    )
    evidence_drift = _w2_candidate_key("checkpoint", _candidate(evidence=["e1", "e3"]))

    assert _candidate_confusion(Counter({payload_drift: 1}), Counter({gold: 1})) == (
        0,
        1,
        1,
    )
    assert _candidate_confusion(Counter({evidence_drift: 1}), Counter({gold: 1})) == (
        0,
        1,
        1,
    )


def test_w2_candidate_comparison_preserves_duplicate_multiset_counts() -> None:
    key = _w2_candidate_key("checkpoint", _candidate())
    assert _candidate_confusion(Counter({key: 3}), Counter({key: 2})) == (2, 1, 0)


def test_w2_setup_latency_requires_exact_four_call_gate_total() -> None:
    _validate_w2_setup_latency(1234.567, 1234.567)
    with pytest.raises(ValueError, match="differs from the exact preflight"):
        _validate_w2_setup_latency(1234.568, 1234.567)
    with pytest.raises(ValueError, match="differs from the exact preflight"):
        _validate_w2_setup_latency(1234.566, 1234.567)


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
            actions = [
                ProposedAction(
                    kind=candidate.action_template.kind,
                    action_key=candidate.action_key,
                    payload=dict(candidate.action_template.payload),
                    summary=candidate.action_template.summary,
                    evidence_event_ids=list(candidate.evidence_event_ids),
                )
                for candidate in selection.due_candidates
            ]
            memory.commit(event, Decision(actions=actions))
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


def test_v3_reference_fixture_passes_exact_w2_candidate_gate() -> None:
    scenarios = load_scenarios(DATASET)
    reference = load_oracle_artifact(REFERENCE, scenarios)
    metrics = _w2_metrics(scenarios, _reference_runs(scenarios, reference), reference)

    assert sum(len(scenario.events) for scenario in scenarios) == 69
    assert metrics["compiler_calls"] == WRITER_W2_COMPILER_CALLS == 46
    assert metrics["compiler_accepted"] == 46
    assert metrics["compiler_parse_invalid"] == 0
    assert metrics["compiler_semantic_invalid"] == 0
    assert metrics["candidate_tp"] == 8
    assert metrics["candidate_fp"] == 0
    assert metrics["candidate_fn"] == 0
    assert metrics["gate_passed"] is True
    assert metrics["final_action_f1_diagnostic"] == 0.0

    rendered = _render_w2_markdown(metrics)
    assert rendered.startswith(f"# {WRITER_W2_TITLE}\n")
    assert "Summary text" in rendered
    assert "46/46 accepted" in rendered


def test_w2_metrics_reject_incomplete_or_nonzero_cost_accounting() -> None:
    scenarios = load_scenarios(DATASET)
    reference = load_oracle_artifact(REFERENCE, scenarios)
    runs = _reference_runs(scenarios, reference)

    incomplete = runs[0].model_copy(update={"cost_complete": False})
    with pytest.raises(ValueError, match="incomplete usage or cost"):
        _w2_metrics(scenarios, [incomplete, *runs[1:]], reference)

    paid_usage = Usage(cost_usd=0.01)
    paid = runs[0].model_copy(
        update={
            "compiler_usage": paid_usage,
            "usage": paid_usage,
        }
    )
    with pytest.raises(ValueError, match="zero provider API cost"):
        _w2_metrics(scenarios, [paid, *runs[1:]], reference)


def test_w2_scenario_compiler_and_decision_prompts_are_exactly_replayed() -> None:
    scenarios = load_scenarios(DATASET)
    reference = load_oracle_artifact(REFERENCE, scenarios)
    scenario = scenarios[0]
    run = _reference_runs([scenario], reference)[0]
    memory = InMemoryAnamnesis()
    model_events: list[ModelEvent] = []
    no_action = '{"mode":"no_action","actions":[]}'

    for event, checkpoint in zip(
        scenario.to_runtime().events, run.checkpoints, strict=True
    ):
        delta = None
        if event.kind != "clock_tick":
            model_events.append(
                ModelEvent.model_construct(
                    event="model",
                    input=[
                        ChatMessageUser(
                            content=build_local_memory_compiler_w2_prompt(
                                event=event,
                                active_state=memory.compiler_state(),
                            )
                        )
                    ],
                )
            )
            assert checkpoint.memory_delta_json is not None
            delta = _memory_delta_from_audit_json(checkpoint.memory_delta_json)
        memory.ingest(event, delta)
        selection = memory.select(event)
        model_events.append(
            ModelEvent.model_construct(
                event="model",
                input=[
                    ChatMessageUser(
                        content=build_local_decision_prompt(
                            now=event.at.isoformat(),
                            current_event_id=event.id,
                            context_events=[event],
                            decision_history=[],
                            memory_view=selection.view,
                        )
                    )
                ],
                output=ModelOutput.from_content(
                    model="ollama/qwen3:4b-instruct", content=no_action
                ),
            )
        )
        memory.commit(event, Decision())

    sample = SimpleNamespace(events=model_events)
    _validate_exact_w2_prompts(
        sample=sample,
        scenario=scenario,
        run=run,
        first_sample=False,
    )

    original = model_events[1]
    model_events[1] = ModelEvent.model_construct(
        event="model",
        input=[ChatMessageUser(content="tampered decision prompt")],
        output=original.output,
    )
    with pytest.raises(ValueError, match="decision prompt"):
        _validate_exact_w2_prompts(
            sample=sample,
            scenario=scenario,
            run=run,
            first_sample=False,
        )


def test_w1_and_w2_manifest_validators_cross_reject() -> None:
    w1 = LocalExperimentManifest.model_validate_json(W1_TEMPLATE.read_text())
    w2 = LocalExperimentManifest.model_validate_json(W2_TEMPLATE.read_text())
    w1_frozen = w1.model_copy(
        update={"status": "frozen", "system_config_sha256": {"anamnesis": ZERO_SHA256}}
    )
    w2_frozen = w2.model_copy(
        update={"status": "frozen", "system_config_sha256": {"anamnesis": ZERO_SHA256}}
    )

    _validate_writer_manifest_identity(w1_frozen)
    _validate_w2_manifest_identity(w2_frozen)
    with pytest.raises(ValueError, match="writer_diagnostic"):
        _validate_writer_manifest_identity(w2_frozen)
    with pytest.raises(ValueError, match="writer_diagnostic_w2"):
        _validate_w2_manifest_identity(w1_frozen)


def test_raw_fixture_body_leak_detection_rejects_only_complete_fixture() -> None:
    fixture = json.loads(
        Path("eval/preflight/local_writer_w2.v1.json").read_text(encoding="utf-8")
    )
    frozen = json.dumps(
        fixture, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )

    assert _contains_raw_fixture({"nested": fixture}, fixture)
    assert _contains_raw_fixture({"nested": frozen}, fixture)
    assert not _contains_raw_fixture(
        {"case_ids": ["C1", "C2", "C3", "D1"], "passed": True}, fixture
    )

    leaked_log = SimpleNamespace(
        eval=SimpleNamespace(task_args={}, metadata={}),
        samples=[SimpleNamespace(store={"raw_fixture": fixture})],
    )
    with pytest.raises(ValueError, match="fixture leaked"):
        writer_report_w2._reject_raw_fixture_leak(leaked_log, fixture)


def test_w2_task_metadata_dataset_and_identity_are_exact() -> None:
    manifest = LocalExperimentManifest.model_validate_json(W2_TEMPLATE.read_text())
    scenarios = load_scenarios(DATASET)
    ids = [scenario.id for scenario in scenarios]
    manifest_sha256 = "a" * 64
    metadata = {
        "track": "local_zero_api_cost",
        "claim_scope": "diagnostic_development_only",
        "hypothesis_test_eligible": False,
        "system": "anamnesis",
        "dataset": manifest.dataset.path,
        "dataset_split": "writer_diagnostic_w2",
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
        "compiler_mode": "llm",
        "memory_compiler_prompt_version": "local.v0.3",
        "preflight_fixture_sha256": manifest.preflight_fixture.sha256,
        "setup_preflight_task": "local_model_preflight_w2",
        "setup_preflight_model_calls": 4,
        "setup_preflight_compiler_calls": 3,
        "setup_preflight_decision_calls": 1,
        "setup_preflight_usage_in_headline": False,
        "same_model_for_compiler_and_decision": True,
        "scenario_compiler_model_calls": 46,
    }
    dataset = SimpleNamespace(
        name="anamnesis-local-writer_diagnostic_w2-v0",
        location=str(DATASET.resolve()),
        samples=10,
        sample_ids=ids,
        shuffled=False,
    )
    spec = SimpleNamespace(
        task_registry_name="local_anamnesis_writer_diagnostic_w2",
        task_version=LOCAL_SCENARIO_TASK_VERSION,
        task_file="eval/anamnesis_local_eval.py",
        task_args={
            "seed": 101,
            "repetition": 1,
            "manifest": str(W2_TEMPLATE.resolve()),
            "ollama_models_dir": "/absolute/ollama/models",
        },
        metadata=metadata,
        dataset=dataset,
    )
    kwargs = {
        "manifest": manifest,
        "manifest_path": W2_TEMPLATE,
        "manifest_sha256": manifest_sha256,
        "scenarios_path": DATASET,
        "scenarios": scenarios,
    }
    _validate_w2_task_and_dataset(SimpleNamespace(eval=spec), **kwargs)

    spec.task_registry_name = "local_anamnesis_writer_diagnostic"
    with pytest.raises(ValueError, match="unexpected W2"):
        _validate_w2_task_and_dataset(SimpleNamespace(eval=spec), **kwargs)
    spec.task_registry_name = "local_anamnesis_writer_diagnostic_w2"
    metadata["scenario_compiler_model_calls"] = 45
    with pytest.raises(ValueError, match="metadata differs"):
        _validate_w2_task_and_dataset(SimpleNamespace(eval=spec), **kwargs)


def _failed_gate_metrics() -> dict[str, object]:
    return {
        "title": WRITER_W2_TITLE,
        "hypothesis_test_eligible": False,
        "compiler_calls": 46,
        "compiler_parse_invalid": 0,
        "compiler_semantic_invalid": 0,
        "compiler_accepted": 46,
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


def test_valid_w2_gate_failure_writes_all_outputs_and_returns_two(
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
    manifest = SimpleNamespace(
        writer_reference=SimpleNamespace(path="reference.json", sha256="a" * 64)
    )

    monkeypatch.setattr(
        writer_report_w2, "_validate_output_locations", lambda **_: None
    )
    monkeypatch.setattr(
        writer_report_w2,
        "_validate_frozen_w2_manifest",
        lambda **_: (manifest, [object()], {}, "b" * 64),
    )
    monkeypatch.setattr(
        writer_report_w2, "_load_w2_writer_run", lambda *_, **__: [object()]
    )
    monkeypatch.setattr(writer_report_w2, "_repo_file", lambda *_, **__: reference_path)
    monkeypatch.setattr(writer_report_w2, "_all_w2_source_paths", lambda **_: {})
    monkeypatch.setattr(
        writer_report_w2, "_validate_w2_provenance_inputs", lambda **_: None
    )
    monkeypatch.setattr(
        writer_report_w2, "load_oracle_artifact", lambda *_, **__: object()
    )
    monkeypatch.setattr(
        writer_report_w2, "_w2_metrics", lambda *_, **__: _failed_gate_metrics()
    )

    def write_provenance(path: Path, **_: object) -> None:
        path.write_text('{"gate":"failed"}\n', encoding="utf-8")

    monkeypatch.setattr(writer_report_w2, "_write_w2_provenance", write_provenance)
    exit_code = writer_report_w2_main(
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
    assert "| FAIL |" in markdown_path.read_text(encoding="utf-8")
    assert provenance_path.is_file()


def test_w2_integrity_failure_happens_before_reference_or_output_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    scenarios_path = tmp_path / "scenarios.jsonl"
    run_path = tmp_path / "writer.eval"
    for path in (manifest_path, scenarios_path, run_path):
        path.write_text("fixture", encoding="utf-8")
    csv_path = tmp_path / "writer.csv"
    markdown_path = tmp_path / "writer.md"
    provenance_path = tmp_path / "writer.provenance.json"
    manifest = SimpleNamespace(
        writer_reference=SimpleNamespace(path="reference.json", sha256="a" * 64)
    )
    reference_opened = False

    monkeypatch.setattr(
        writer_report_w2, "_validate_output_locations", lambda **_: None
    )
    monkeypatch.setattr(
        writer_report_w2,
        "_validate_frozen_w2_manifest",
        lambda **_: (manifest, [object()], {}, "b" * 64),
    )

    def reject_measured_log(*_: object, **__: object) -> list[ScenarioRun]:
        raise ValueError("measured log integrity failure")

    def observe_reference(*_: object, **__: object) -> Path:
        nonlocal reference_opened
        reference_opened = True
        return tmp_path / "reference.json"

    monkeypatch.setattr(writer_report_w2, "_load_w2_writer_run", reject_measured_log)
    monkeypatch.setattr(writer_report_w2, "_repo_file", observe_reference)

    with pytest.raises(ValueError, match="measured log integrity"):
        writer_report_w2_main(
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

    assert reference_opened is False
    assert not csv_path.exists()
    assert not markdown_path.exists()
    assert not provenance_path.exists()


def test_shared_writer_entrypoint_infers_w2_only_from_manifest_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"phase":"writer_diagnostic_w2"}\n', encoding="utf-8")
    monkeypatch.setattr(writer_report_w2, "writer_report_w2_main", lambda argv: 7)

    assert (
        writer_report_dispatcher.writer_report_main(
            ["--manifest", str(manifest_path), "--run", "unused.eval"]
        )
        == 7
    )


def test_w2_reporter_source_pins_protocol_and_all_provenance_inputs() -> None:
    source = Path("src/anamnesis/writer_report_w2.py").read_text(encoding="utf-8")
    assert (
        'WRITER_W2_TITLE = "Local writer W2 diagnostic — not a hypothesis test"'
        in source
    )
    assert "WRITER_W2_COMPILER_CALLS = 46" in source
    assert '"excluded": ["summary", "intent_id", "occurrence_id"]' in source
    assert '"protocol_contracts": {' in source
    assert '"model_preflight": {' in source
    assert '"preflight_fixture": {' in source
    assert '"input_eval_log": {' in source
    assert '"outputs": {' in source
    assert '"opened_only_after_run_validation": True' in source
