"""Adversarial checks for the strict oracle-compiler ceiling reporter."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalPlan,
    EvalRevision,
    EvalSample,
    EvalSpec,
    EvalStats,
)
from inspect_ai.model import GenerateConfig, ModelOutput, ModelUsage

from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.local_experiment import LocalExperimentManifest
from anamnesis.local_runtime import (
    LOCAL_DECISION_VERSION,
    LOCAL_OLLAMA_BASE_URL,
    LOCAL_OLLAMA_MODEL,
    LOCAL_PREFLIGHT_STORE_KEY,
    LOCAL_SCENARIO_TASK_VERSION,
    _local_decision_schema,
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
)
from anamnesis.oracle_report import (
    ORACLE_CEILING_TITLE,
    ORACLE_DATASET_NAME,
    ORACLE_TASK_NAME,
    _load_oracle_runs,
    _render_markdown,
    _require_pinned_file,
    _result_row,
    _validate_oracle_log,
    _validate_oracle_manifest_identity,
    _write_csv,
    _write_provenance,
)
from anamnesis.schema import CheckpointAudit, Scenario, ScenarioRun, Usage
from anamnesis.scoring import aggregate_results, score_scenario
from tests.test_local_report import (
    GIT_COMMIT,
    MANIFEST_SHA256,
    NO_ACTION_COMPLETION,
    _model_event,
    _preflight_events,
    _preflight_result,
    _sha256_file,
    _sha256_text,
    _usage,
)

TEMPLATE = Path("eval/local_experiment_manifest.template.json")
SCENARIOS_PATH = Path("eval/scenarios/smoke.jsonl")
ORACLE_PATH = Path("eval/oracle/smoke_memory_deltas.v1.json")


def _oracle_manifest() -> LocalExperimentManifest:
    raw = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    raw.update(
        status="frozen",
        phase="oracle_smoke",
        compiler_mode="oracle",
        systems=[ORACLE_SYSTEM_NAME],
        git_commit=GIT_COMMIT,
        oracle_annotations={
            "path": ORACLE_PATH.as_posix(),
            "sha256": _sha256_file(ORACLE_PATH),
        },
        decision_prompt_sha256=_sha256_text(local_decision_prompt_contract()),
        decision_schema_sha256=_sha256_text(local_decision_schema_contract()),
        memory_compiler_prompt_sha256=None,
        memory_compiler_schema_sha256=None,
        system_config_sha256={ORACLE_SYSTEM_NAME: "c" * 64},
    )
    model = raw["model"]
    assert isinstance(model, dict)
    model["same_model_for_compiler_and_decision"] = False
    model["preflight"]["sha256"] = "d" * 64
    provisional = LocalExperimentManifest.model_validate(raw)
    annotations = provisional.oracle_annotations
    assert annotations is not None and annotations.sha256 is not None
    raw["system_config_sha256"] = {
        ORACLE_SYSTEM_NAME: local_system_config_sha256(
            system=ORACLE_SYSTEM_NAME,
            top_k=provisional.embedding.top_k,
            embedding_model=provisional.embedding.model,
            embedding_repository=provisional.embedding.repository,
            embedding_revision=provisional.embedding.revision,
            pricing_config_sha256=provisional.model.pricing.sha256,
            oracle_annotations_sha256=annotations.sha256,
        )
    }
    return LocalExperimentManifest.model_validate(raw)


def _oracle_sample(
    scenario: Scenario,
    *,
    artifact: OracleCompilerArtifact,
    manifest: LocalExperimentManifest,
    first: bool,
) -> EvalSample:
    preflight = _preflight_result()
    model_events: list[ModelEvent] = _preflight_events() if first else []
    records = iter(artifact.records_for(scenario.to_runtime()))
    checkpoints: list[CheckpointAudit] = []
    compiler_calls = 0

    for index, authored in enumerate(scenario.events, start=1):
        compiler_called = authored.kind != "clock_tick"
        raw_compiler_output = None
        memory_delta_json = None
        compiler_usage = Usage()
        if compiler_called:
            compiler_calls += 1
            record = next(records)
            assert record.event_id == authored.id
            raw_compiler_output = record.delta.model_dump_json()
            memory_delta_json = raw_compiler_output
            compiler_usage = Usage(cost_usd=0.0)

        decision_prompt = f"oracle decision {scenario.id}/{authored.id}"
        model_events.append(
            _model_event(
                decision_prompt,
                _local_decision_schema(LOCAL_OLLAMA_MODEL),
                NO_ACTION_COMPLETION,
            )
        )
        checkpoints.append(
            CheckpointAudit(
                event_id=authored.id,
                at=authored.at,
                compiler_called=compiler_called,
                raw_compiler_output=raw_compiler_output,
                memory_delta_json=memory_delta_json,
                memory_delta_accepted=True if compiler_called else None,
                state_sha256=f"{index:064x}",
                rendered_context_sha256=hashlib.sha256(
                    decision_prompt.encode()
                ).hexdigest(),
                raw_decision_output=NO_ACTION_COMPLETION,
                compiler_usage=compiler_usage,
                decision_usage=_usage(),
                compiler_latency_ms=0.1 if compiler_called else 0.0,
                decision_latency_ms=1.0,
            )
        )

    decision_usage = _usage(len(scenario.events))
    compiler_usage = Usage(cost_usd=0.0)
    run = ScenarioRun(
        scenario_id=scenario.id,
        system=ORACLE_SYSTEM_NAME,
        repetition=1,
        model=LOCAL_OLLAMA_MODEL,
        prompt_version=LOCAL_DECISION_VERSION,
        scenario_sha256=canonical_sha256(scenario),
        prompt_sha256=_sha256_text(local_decision_contract()),
        system_config_sha256=manifest.system_config_sha256[ORACLE_SYSTEM_NAME],
        manifest_sha256=MANIFEST_SHA256,
        pricing_config_sha256=manifest.model.pricing.sha256,
        seed=101,
        usage=decision_usage.plus(compiler_usage),
        decision_usage=decision_usage,
        compiler_usage=compiler_usage,
        usage_complete=True,
        cost_complete=True,
        decision_latency_ms=float(len(scenario.events)),
        compiler_latency_ms=0.1 * compiler_calls,
        setup_latency_ms=preflight.setup_latency_ms if first else 0.0,
        checkpoint_latency_ms=[
            1.0 + (0.1 if checkpoint.compiler_called else 0.0)
            for checkpoint in checkpoints
        ],
        checkpoints=checkpoints,
    )
    return EvalSample.model_construct(
        id=scenario.id,
        epoch=1,
        events=model_events,
        store={
            "anamnesis.scenario_run": run.model_dump(mode="json"),
            LOCAL_PREFLIGHT_STORE_KEY: preflight.model_dump(mode="json"),
        },
        error=None,
        invalidation=None,
        error_retries=[],
        output=ModelOutput.from_content(
            model=LOCAL_OLLAMA_MODEL,
            content=run.model_dump_json(),
        ),
    )


def _oracle_log(
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    scenarios: list[Scenario],
    artifact: OracleCompilerArtifact,
) -> EvalLog:
    samples = [
        _oracle_sample(
            scenario,
            artifact=artifact,
            manifest=manifest,
            first=index == 0,
        )
        for index, scenario in enumerate(scenarios)
    ]
    model_event_count = sum(
        isinstance(event, ModelEvent) for sample in samples for event in sample.events
    )
    annotations = manifest.oracle_annotations
    assert annotations is not None and annotations.sha256 is not None
    scenario_ids = [scenario.id for scenario in scenarios]
    metadata = {
        "track": "local_zero_api_cost",
        "claim_scope": "diagnostic_development_only",
        "hypothesis_test_eligible": False,
        "system": ORACLE_SYSTEM_NAME,
        "dataset": manifest.dataset.path,
        "dataset_split": "oracle_smoke",
        "dataset_scenario_count": 10,
        "dataset_sample_ids": scenario_ids,
        "canonical_dataset_sha256": dataset_sha256(scenarios),
        "repetition": 1,
        "manifest_sha256": MANIFEST_SHA256,
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
    spec = EvalSpec.model_construct(
        task=ORACLE_TASK_NAME,
        task_registry_name=ORACLE_TASK_NAME,
        task_version=LOCAL_SCENARIO_TASK_VERSION,
        task_file="eval/anamnesis_local_eval.py",
        task_args={
            "seed": 101,
            "repetition": 1,
            "manifest": str(manifest_path),
            "ollama_models_dir": "/absolute/ollama/models",
            "oracle_annotations_path": ORACLE_PATH.as_posix(),
        },
        metadata=metadata,
        model=LOCAL_OLLAMA_MODEL,
        model_base_url=LOCAL_OLLAMA_BASE_URL,
        model_args={},
        model_generate_config=GenerateConfig(
            temperature=0.0,
            seed=101,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
        ),
        config=EvalConfig(
            max_samples=1,
            max_tasks=1,
            epochs=1,
            log_model_api=True,
        ),
        revision=EvalRevision(
            type="git",
            origin="test",
            commit=GIT_COMMIT[:7],
            dirty=False,
        ),
        dataset=EvalDataset(
            name=ORACLE_DATASET_NAME,
            location=SCENARIOS_PATH.as_posix(),
            samples=10,
            sample_ids=scenario_ids,
            shuffled=False,
        ),
    )
    return EvalLog.model_construct(
        status="success",
        invalidated=False,
        config_updates=None,
        log_updates=None,
        eval=spec,
        plan=EvalPlan.model_construct(
            config=GenerateConfig(
                temperature=0.0,
                seed=101,
                cache=False,
                max_retries=0,
                max_connections=1,
                adaptive_connections=False,
            )
        ),
        samples=samples,
        stats=EvalStats(
            model_usage={
                LOCAL_OLLAMA_MODEL: ModelUsage(
                    input_tokens=10 * model_event_count,
                    output_tokens=2 * model_event_count,
                    total_tokens=12 * model_event_count,
                    total_cost=0.0,
                )
            }
        ),
    )


@pytest.fixture
def oracle_case(
    tmp_path: Path,
) -> tuple[
    LocalExperimentManifest,
    Path,
    list[Scenario],
    OracleCompilerArtifact,
    EvalLog,
]:
    manifest = _oracle_manifest()
    manifest_path = tmp_path / "oracle-smoke.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    scenarios = load_scenarios(SCENARIOS_PATH)
    artifact = load_oracle_artifact(ORACLE_PATH, scenarios)
    log = _oracle_log(
        manifest=manifest,
        manifest_path=manifest_path,
        scenarios=scenarios,
        artifact=artifact,
    )
    return manifest, manifest_path, scenarios, artifact, log


def _replace_run(sample: EvalSample, run: ScenarioRun) -> None:
    sample.store["anamnesis.scenario_run"] = run.model_dump(mode="json")
    sample.output = ModelOutput.from_content(
        model=LOCAL_OLLAMA_MODEL,
        content=run.model_dump_json(),
    )


def _validate_case(
    case: tuple[
        LocalExperimentManifest,
        Path,
        list[Scenario],
        OracleCompilerArtifact,
        EvalLog,
    ],
) -> list[ScenarioRun]:
    manifest, manifest_path, scenarios, artifact, log = case
    return _validate_oracle_log(
        log,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=MANIFEST_SHA256,
        scenarios_path=SCENARIOS_PATH,
        oracle_path=ORACLE_PATH,
        scenarios=scenarios,
        artifact=artifact,
    )


def test_strict_oracle_log_reconciles_setup_raw_calls_and_headline_usage(
    oracle_case,
) -> None:
    runs = _validate_case(oracle_case)

    assert len(runs) == 10
    assert all(run.compiler_usage == Usage(cost_usd=0.0) for run in runs)
    assert all(run.usage == run.decision_usage for run in runs)
    first_events = [
        event
        for event in oracle_case[-1].samples[0].events
        if isinstance(event, ModelEvent)
    ]
    assert len(first_events) == 2 + len(oracle_case[2][0].events)
    for sample, scenario in zip(
        oracle_case[-1].samples[1:],
        oracle_case[2][1:],
        strict=True,
    ):
        assert sum(isinstance(event, ModelEvent) for event in sample.events) == len(
            scenario.events
        )


@pytest.mark.parametrize("mutation", ["extra", "missing", "setup_on_second"])
def test_oracle_report_rejects_extra_or_missing_model_events(
    oracle_case,
    mutation: str,
) -> None:
    log = oracle_case[-1]
    if mutation == "extra":
        log.samples[0].events.append(log.samples[0].events[-1])
    elif mutation == "missing":
        log.samples[0].events.pop()
    else:
        log.samples[1].events.insert(0, _preflight_events()[0])

    with pytest.raises(ValueError, match="exactly one decision ModelEvent"):
        _validate_case(oracle_case)


@pytest.mark.parametrize("mutation", ["raw_delta", "accepted", "usage"])
def test_oracle_report_rejects_delta_drift_rejection_or_nonzero_usage(
    oracle_case,
    mutation: str,
) -> None:
    sample = oracle_case[-1].samples[0]
    run = ScenarioRun.model_validate(sample.store["anamnesis.scenario_run"])
    checkpoint = run.checkpoints[0]
    if mutation == "raw_delta":
        checkpoint = checkpoint.model_copy(update={"raw_compiler_output": '{"x":1}'})
    elif mutation == "accepted":
        checkpoint = checkpoint.model_copy(update={"memory_delta_accepted": False})
    else:
        checkpoint = checkpoint.model_copy(update={"compiler_usage": _usage()})
    updates: dict[str, object] = {"checkpoints": [checkpoint, *run.checkpoints[1:]]}
    if mutation == "usage":
        updates["compiler_usage"] = _usage()
        updates["usage"] = run.decision_usage.plus(_usage())
    run = run.model_copy(update=updates)
    _replace_run(sample, run)

    expected = {
        "raw_delta": "differs from frozen annotations",
        "accepted": "was rejected",
        "usage": "exact zero",
    }[mutation]
    with pytest.raises(ValueError, match=expected):
        _validate_case(oracle_case)


def test_oracle_report_rejects_wrong_task_system_phase_and_artifact_hash(
    oracle_case,
) -> None:
    manifest, _, _, _, log = oracle_case
    wrong_phase = manifest.model_copy(update={"phase": "smoke"})
    with pytest.raises(ValueError, match="oracle_smoke"):
        _validate_oracle_manifest_identity(wrong_phase)

    log.eval = log.eval.model_copy(update={"task_registry_name": "local_anamnesis"})
    with pytest.raises(ValueError, match="unexpected oracle ceiling task"):
        _validate_case(oracle_case)

    log = _oracle_log(
        manifest=manifest,
        manifest_path=oracle_case[1],
        scenarios=oracle_case[2],
        artifact=oracle_case[3],
    )
    sample = log.samples[0]
    run = ScenarioRun.model_validate(sample.store["anamnesis.scenario_run"])
    _replace_run(sample, run.model_copy(update={"system": "anamnesis"}))
    changed = (*oracle_case[:-1], log)
    with pytest.raises(ValueError, match="identity or contract binding"):
        _validate_case(changed)

    with pytest.raises(ValueError, match="bytes differ"):
        _require_pinned_file(ORACLE_PATH, "0" * 64, label="oracle artifact")


def test_oracle_report_rejects_wrong_logged_annotation_hash_and_stats(
    oracle_case,
) -> None:
    log = oracle_case[-1]
    metadata = dict(log.eval.metadata)
    metadata["oracle_annotations_sha256"] = "0" * 64
    log.eval = log.eval.model_copy(update={"metadata": metadata})
    with pytest.raises(ValueError, match="oracle task metadata differs"):
        _validate_case(oracle_case)

    log = _oracle_log(
        manifest=oracle_case[0],
        manifest_path=oracle_case[1],
        scenarios=oracle_case[2],
        artifact=oracle_case[3],
    )
    usage = log.stats.model_usage[LOCAL_OLLAMA_MODEL]
    log.stats.model_usage[LOCAL_OLLAMA_MODEL] = usage.model_copy(
        update={"input_tokens": usage.input_tokens + 1}
    )
    changed = (*oracle_case[:-1], log)
    with pytest.raises(ValueError, match="stats differ from raw"):
        _validate_case(changed)


def test_oracle_report_loader_accepts_one_eval_only(
    oracle_case,
    tmp_path: Path,
) -> None:
    manifest, manifest_path, scenarios, artifact, _ = oracle_case
    with pytest.raises(ValueError, match="exactly one Inspect .eval"):
        _load_oracle_runs(
            tmp_path / "not-a-log.json",
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=MANIFEST_SHA256,
            scenarios_path=SCENARIOS_PATH,
            oracle_path=ORACLE_PATH,
            scenarios=scenarios,
            artifact=artifact,
        )


def test_oracle_output_is_one_diagnostic_row_without_success_comparison(
    oracle_case,
    tmp_path: Path,
) -> None:
    runs = _validate_case(oracle_case)
    scenarios = {scenario.id: scenario for scenario in oracle_case[2]}
    result = aggregate_results(
        (score_scenario(scenarios[run.scenario_id], run), run) for run in runs
    )[0]
    csv_path = tmp_path / "oracle.csv"
    _write_csv(csv_path, result)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    markdown = _render_markdown(result)

    assert len(rows) == 1
    assert rows[0]["title"] == ORACLE_CEILING_TITLE
    assert rows[0]["oracle_compiler_input_tokens"] == "0"
    assert rows[0]["human_annotation_effort_measured"] == "False"
    assert markdown.startswith(f"# {ORACLE_CEILING_TITLE}")
    assert "Human annotation effort is unmeasured" in markdown
    assert "no success gate or baseline comparison" in markdown
    assert "Reduction vs full" not in markdown
    assert _result_row(result)["hypothesis_test_eligible"] is False


def test_oracle_provenance_binds_repo_relative_inputs_outputs_and_pins(
    oracle_case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("anamnesis.oracle_report.REPO_ROOT", tmp_path)
    manifest = oracle_case[0]

    scenario_path = tmp_path / "eval/scenarios/smoke.jsonl"
    oracle_path = tmp_path / "eval/oracle/smoke_memory_deltas.v1.json"
    pricing_path = tmp_path / manifest.model.pricing.path
    preflight_path = tmp_path / manifest.model.preflight.path
    for path in (scenario_path, oracle_path, pricing_path, preflight_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_bytes(SCENARIOS_PATH.read_bytes())
    oracle_path.write_bytes(ORACLE_PATH.read_bytes())
    pricing_path.write_bytes(Path(manifest.model.pricing.path).read_bytes())
    preflight_path.write_bytes(Path(manifest.model.preflight.path).read_bytes())

    raw = json.loads(manifest.model_dump_json())
    raw["dataset"]["sha256"] = _sha256_file(scenario_path)
    raw["oracle_annotations"]["sha256"] = _sha256_file(oracle_path)
    raw["model"]["pricing"]["sha256"] = _sha256_file(pricing_path)
    raw["model"]["preflight"]["sha256"] = _sha256_file(preflight_path)
    manifest = LocalExperimentManifest.model_validate(raw)

    manifest_path = tmp_path / "results/runs/local/oracle-smoke.json"
    run_path = tmp_path / "results/runs/local/oracle.eval"
    csv_path = tmp_path / "results/local_oracle_smoke.csv"
    markdown_path = tmp_path / "results/local_oracle_smoke.md"
    provenance_path = tmp_path / "results/local_oracle_smoke.provenance.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    run_path.write_bytes(b"synthetic-eval")
    csv_path.write_text("header\nvalue\n", encoding="utf-8")
    markdown_path.write_text("# result\n", encoding="utf-8")

    _write_provenance(
        provenance_path,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        scenarios_path=scenario_path,
        oracle_path=oracle_path,
        artifact=oracle_case[3],
        run_path=run_path,
        expected_run_sha256=_sha256_file(run_path),
        csv_path=csv_path,
        markdown_path=markdown_path,
    )
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert payload["hypothesis_test_eligible"] is False
    assert payload["gold_assisted"] is True
    assert payload["oracle_token_scope"] == "decision_only_lower_bound"
    assert payload["input_eval_log"]["path"] == ("results/runs/local/oracle.eval")
    assert payload["oracle_annotations"]["sha256"] == _sha256_file(oracle_path)
    assert payload["model_preflight"]["sha256"] == _sha256_file(preflight_path)
    assert payload["pricing_config"]["sha256"] == _sha256_file(pricing_path)
    assert payload["outputs"]["csv"]["sha256"] == _sha256_file(csv_path)
    assert not any(
        value.startswith("/")
        for record in (
            payload["frozen_manifest"],
            payload["scenario_dataset"],
            payload["oracle_annotations"],
            payload["model_preflight"],
            payload["pricing_config"],
            payload["input_eval_log"],
            *payload["outputs"].values(),
        )
        for key, value in record.items()
        if key == "path"
    )

    preflight_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="model preflight bytes differ"):
        _write_provenance(
            tmp_path / "results/changed.provenance.json",
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=_sha256_file(manifest_path),
            scenarios_path=scenario_path,
            oracle_path=oracle_path,
            artifact=oracle_case[3],
            run_path=run_path,
            expected_run_sha256=_sha256_file(run_path),
            csv_path=csv_path,
            markdown_path=markdown_path,
        )


def test_pyproject_declares_oracle_report_entrypoint() -> None:
    content = Path("pyproject.toml").read_text(encoding="utf-8")
    assert (
        'anamnesis-oracle-report = "anamnesis.oracle_report:oracle_report_main"'
        in content
    )
