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
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelCall,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._openai import openai_completion_params

from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.local_experiment import LocalExperimentManifest
from anamnesis.local_report import (
    LOCAL_SMOKE_PROVENANCE_PATH,
    LOCAL_SMOKE_TITLE,
    LOCAL_SYSTEM_TASKS,
    _discover_repo_root,
    _expected_system_hashes,
    _expected_vector_retrieval_usage,
    _load_local_smoke_runs,
    _logged_dataset_candidates,
    _render_markdown,
    _repo_relative_path,
    _sha256_file,
    _sha256_text,
    _validate_frozen_local_manifest,
    _write_csv,
    _write_result_provenance,
)
from anamnesis.local_runtime import (
    LOCAL_DECISION_VERSION,
    LOCAL_OLLAMA_BASE_URL,
    LOCAL_OLLAMA_CONTEXT_LENGTH,
    LOCAL_OLLAMA_FAMILY,
    LOCAL_OLLAMA_MANIFEST_SHA256,
    LOCAL_OLLAMA_MODEL,
    LOCAL_OLLAMA_PARAMETER_SIZE,
    LOCAL_OLLAMA_QUANTIZATION,
    LOCAL_OLLAMA_SERVICE_MODEL,
    LOCAL_PREFLIGHT_STORE_KEY,
    LOCAL_SCENARIO_TASK_VERSION,
    LocalLoadedModelAttestation,
    LocalModelPreflightResult,
    LocalOllamaRuntimeAttestation,
    _local_decision_schema,
    _local_memory_delta_schema,
    local_decision_contract,
    local_decision_prompt_contract,
    local_decision_schema_contract,
    local_memory_compiler_prompt_contract,
    local_memory_compiler_schema_contract,
)
from anamnesis.local_wire import build_local_memory_compiler_prompt
from anamnesis.schema import CheckpointAudit, Decision, Scenario, ScenarioRun, Usage
from anamnesis.scoring import aggregate_results, score_scenario

TEMPLATE = Path("eval/local_experiment_manifest.template.json")
SCENARIOS_PATH = Path("eval/scenarios/smoke.jsonl")
GIT_COMMIT = "a" * 40
MANIFEST_SHA256 = "b" * 64
PREFLIGHT_COMPILER_COMPLETION = json.dumps(
    {
        "fact_assertions": [],
        "intent_creates": [
            {
                "intent_id": "compatibility-check",
                "trigger": {"type": "at", "at": "2026-01-05T17:00:00Z"},
                "required_conditions": [],
                "blockers": [],
                "action_template": {
                    "payload": {"subject": "perform compatibility check"},
                    "summary": "Perform compatibility check.",
                },
            }
        ],
        "intent_updates": [],
        "intent_cancellations": [],
    }
)
EMPTY_COMPILER_COMPLETION = json.dumps(
    {
        "fact_assertions": [],
        "intent_creates": [],
        "intent_updates": [],
        "intent_cancellations": [],
    }
)
NO_ACTION_COMPLETION = '{"mode":"no_action","actions":[]}'


def test_local_report_discovers_checkout_from_a_subdirectory() -> None:
    assert _discover_repo_root(Path("eval/scenarios")) == Path.cwd().resolve()


def test_logged_dataset_location_accepts_real_inspect_task_relative_shape() -> None:
    assert SCENARIOS_PATH.resolve() in _logged_dataset_candidates(
        "scenarios/smoke.jsonl"
    )


def _usage(calls: int = 1) -> Usage:
    return Usage(
        input_tokens=10 * calls,
        uncached_input_tokens=10 * calls,
        output_tokens=2 * calls,
        cost_usd=0.0,
    )


def _preflight_result() -> LocalModelPreflightResult:
    return LocalModelPreflightResult(
        model=LOCAL_OLLAMA_MODEL,
        runtime=LocalOllamaRuntimeAttestation(
            model=LOCAL_OLLAMA_MODEL,
            base_url=LOCAL_OLLAMA_BASE_URL,
            no_cloud="1",
            context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
            host="127.0.0.1:11434",
            num_parallel=1,
            max_loaded_models=1,
        ),
        loaded_model=LocalLoadedModelAttestation(
            model=LOCAL_OLLAMA_SERVICE_MODEL,
            digest=LOCAL_OLLAMA_MANIFEST_SHA256,
            family=LOCAL_OLLAMA_FAMILY,
            parameter_size=LOCAL_OLLAMA_PARAMETER_SIZE,
            quantization_level=LOCAL_OLLAMA_QUANTIZATION,
            context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
            size_vram=3_000_000_000,
            ollama_version="0.31.1",
        ),
        same_model_for_compiler_and_decision=True,
        compiler_parse_error=False,
        decision_parse_error=False,
        compiler_semantic_valid=True,
        decision_semantic_valid=True,
        compiler_usage=_usage(),
        decision_usage=_usage(),
        compiler_usage_complete=True,
        decision_usage_complete=True,
        compiler_cost_complete=True,
        decision_cost_complete=True,
        compiler_latency_ms=1.0,
        decision_latency_ms=1.0,
        residency_probe_latency_ms=1.0,
        passed=True,
    )


def _model_event(prompt: str, schema, completion: str) -> ModelEvent:
    config = GenerateConfig(
        temperature=0.0,
        seed=101,
        cache=False,
        max_retries=0,
        max_connections=1,
        adaptive_connections=False,
        response_schema=schema,
    )
    request = openai_completion_params(
        LOCAL_OLLAMA_SERVICE_MODEL,
        config,
        tools=False,
    )
    request.update(
        messages=[{"role": "user", "content": prompt}],
        tools=None,
        tool_choice=None,
        extra_headers={"x-irid": "synthetic-local-request"},
    )
    return ModelEvent.model_construct(
        event="model",
        model=LOCAL_OLLAMA_MODEL,
        input=[ChatMessageUser(content=prompt)],
        config=config,
        output=ModelOutput(
            model=LOCAL_OLLAMA_SERVICE_MODEL,
            completion=completion,
            usage=ModelUsage(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
                total_cost=0.0,
            ),
        ),
        call=ModelCall(
            request=request,
            response={"model": LOCAL_OLLAMA_SERVICE_MODEL},
        ),
        cache=None,
        error=None,
        retries=None,
    )


def _preflight_events() -> list[ModelEvent]:
    from anamnesis.local_preflight import local_preflight_prompts

    compiler_prompt, decision_prompt = local_preflight_prompts()
    return [
        _model_event(
            compiler_prompt,
            _local_memory_delta_schema(LOCAL_OLLAMA_MODEL),
            PREFLIGHT_COMPILER_COMPLETION,
        ),
        _model_event(
            decision_prompt,
            _local_decision_schema(LOCAL_OLLAMA_MODEL),
            NO_ACTION_COMPLETION,
        ),
    ]


def _frozen_manifest() -> LocalExperimentManifest:
    raw = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    raw.update(
        status="frozen",
        git_commit=GIT_COMMIT,
        decision_prompt_sha256=_sha256_text(local_decision_prompt_contract()),
        decision_schema_sha256=_sha256_text(local_decision_schema_contract()),
        memory_compiler_prompt_sha256=_sha256_text(
            local_memory_compiler_prompt_contract()
        ),
        memory_compiler_schema_sha256=_sha256_text(
            local_memory_compiler_schema_contract()
        ),
        system_config_sha256={system: "c" * 64 for system in LOCAL_SYSTEM_TASKS},
    )
    raw["model"]["preflight"]["sha256"] = "d" * 64
    provisional = LocalExperimentManifest.model_validate(raw)
    raw["system_config_sha256"] = _expected_system_hashes(provisional)
    return LocalExperimentManifest.model_validate(raw)


def _sample(
    scenario: Scenario,
    *,
    system: str,
    manifest: LocalExperimentManifest,
    first: bool,
) -> EvalSample:
    preflight = _preflight_result()
    events: list[ModelEvent] = _preflight_events() if first else []
    checkpoints: list[CheckpointAudit] = []
    decision_calls = len(scenario.events)
    compiler_calls = 0
    local_latency = 0.1 if system == "vector_rag" else 0.0

    for authored in scenario.events:
        observable = authored.to_observable()
        compiler_called = system == "anamnesis" and authored.kind != "clock_tick"
        compiler_usage = Usage()
        compiler_output = None
        if compiler_called:
            compiler_calls += 1
            compiler_output = EMPTY_COMPILER_COMPLETION
            compiler_usage = _usage()
            events.append(
                _model_event(
                    build_local_memory_compiler_prompt(
                        event=observable,
                        active_state='{"facts":[],"intents":[]}',
                    ),
                    _local_memory_delta_schema(LOCAL_OLLAMA_MODEL),
                    compiler_output,
                )
            )
        decision_prompt = f"local decision {scenario.id}/{authored.id}"
        events.append(
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
                raw_compiler_output=compiler_output,
                memory_delta_accepted=True if compiler_called else None,
                rendered_context_sha256=hashlib.sha256(
                    decision_prompt.encode()
                ).hexdigest(),
                raw_decision_output=NO_ACTION_COMPLETION,
                compiler_usage=compiler_usage,
                decision_usage=_usage(),
                compiler_latency_ms=1.0 if compiler_called else 0.0,
                decision_latency_ms=1.0,
                local_latency_ms=local_latency,
            )
        )

    decision_usage = _usage(decision_calls)
    compiler_usage = _usage(compiler_calls) if compiler_calls else Usage()
    total_usage = decision_usage.plus(compiler_usage)
    if system == "vector_rag":
        embedding_inputs, embedding_characters = _expected_vector_retrieval_usage(
            scenario, [Decision() for _ in scenario.events]
        )
        total_usage = total_usage.plus(
            Usage(
                embedding_inputs=embedding_inputs,
                embedding_characters=embedding_characters,
                cost_usd=0.0,
            )
        )
    run = ScenarioRun(
        scenario_id=scenario.id,
        system=system,
        repetition=1,
        model=LOCAL_OLLAMA_MODEL,
        prompt_version=LOCAL_DECISION_VERSION,
        scenario_sha256=canonical_sha256(scenario),
        prompt_sha256=_sha256_text(local_decision_contract()),
        system_config_sha256=manifest.system_config_sha256[system],
        manifest_sha256=MANIFEST_SHA256,
        pricing_config_sha256=manifest.model.pricing.sha256,
        seed=101,
        usage=total_usage,
        decision_usage=decision_usage,
        compiler_usage=compiler_usage,
        usage_complete=True,
        cost_complete=True,
        decision_latency_ms=float(decision_calls),
        compiler_latency_ms=float(compiler_calls),
        local_latency_ms=local_latency * decision_calls,
        setup_latency_ms=preflight.setup_latency_ms if first else 0.0,
        checkpoint_latency_ms=[
            1.0 + (1.0 if checkpoint.compiler_called else 0.0) + local_latency
            for checkpoint in checkpoints
        ],
        checkpoints=checkpoints,
    )
    store = {
        "anamnesis.scenario_run": run.model_dump(mode="json"),
        LOCAL_PREFLIGHT_STORE_KEY: preflight.model_dump(mode="json"),
    }
    return EvalSample.model_construct(
        id=scenario.id,
        epoch=1,
        events=events,
        store=store,
        error=None,
        invalidation=None,
        error_retries=[],
        output=ModelOutput.from_content(
            model=LOCAL_OLLAMA_MODEL,
            content=run.model_dump_json(),
        ),
    )


def _log(
    system: str,
    *,
    manifest: LocalExperimentManifest,
    manifest_path: Path,
    scenarios: list[Scenario],
) -> EvalLog:
    samples = [
        _sample(scenario, system=system, manifest=manifest, first=index == 0)
        for index, scenario in enumerate(scenarios)
    ]
    model_events = [
        event
        for sample in samples
        for event in sample.events
        if isinstance(event, ModelEvent)
    ]
    task_args = {
        "seed": 101,
        "repetition": 1,
        "manifest": str(manifest_path),
        "ollama_models_dir": "/absolute/ollama/models",
    }
    if system == "vector_rag":
        task_args["embedding_snapshot_path"] = "/absolute/fastembed/snapshot"
    scenario_ids = [scenario.id for scenario in scenarios]
    metadata = {
        "track": "local_zero_api_cost",
        "claim_scope": "diagnostic_development_only",
        "hypothesis_test_eligible": False,
        "system": system,
        "dataset": manifest.dataset.path,
        "dataset_split": "smoke",
        "dataset_scenario_count": 10,
        "dataset_sample_ids": scenario_ids,
        "canonical_dataset_sha256": dataset_sha256(scenarios),
        "repetition": 1,
        "manifest_sha256": MANIFEST_SHA256,
        "live_semantic_preflight_required": True,
        "provider_api_cost_usd": 0.0,
        "electricity_measured": False,
        "decision_prompt_version": LOCAL_DECISION_VERSION,
        "pricing_config_sha256": manifest.model.pricing.sha256,
    }
    spec = EvalSpec.model_construct(
        task=LOCAL_SYSTEM_TASKS[system],
        task_registry_name=LOCAL_SYSTEM_TASKS[system],
        task_version=LOCAL_SCENARIO_TASK_VERSION,
        task_file="eval/anamnesis_local_eval.py",
        task_args=task_args,
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
            name="anamnesis-local-smoke-v0",
            location=str(SCENARIOS_PATH),
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
                    input_tokens=10 * len(model_events),
                    output_tokens=2 * len(model_events),
                    total_tokens=12 * len(model_events),
                    total_cost=0.0,
                )
            }
        ),
    )


def _logs(
    tmp_path: Path,
) -> tuple[LocalExperimentManifest, Path, list[Scenario], dict[str, EvalLog]]:
    manifest = _frozen_manifest()
    manifest_path = tmp_path / "frozen-local-smoke.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    scenarios = load_scenarios(SCENARIOS_PATH)
    logs = {
        f"{system}.eval": _log(
            system,
            manifest=manifest,
            manifest_path=manifest_path,
            scenarios=scenarios,
        )
        for system in LOCAL_SYSTEM_TASKS
    }
    return manifest, manifest_path, scenarios, logs


def test_strict_local_loader_validates_four_complete_raw_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_path, scenarios, logs = _logs(tmp_path)
    resolved_attachments: list[bool] = []

    def read_log(path: Path, *, resolve_attachments: bool) -> EvalLog:
        resolved_attachments.append(resolve_attachments)
        return logs[Path(path).name]

    monkeypatch.setattr(
        "anamnesis.local_report.read_eval_log",
        read_log,
    )

    runs = _load_local_smoke_runs(
        [tmp_path / name for name in reversed(logs)],
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=MANIFEST_SHA256,
        scenarios_path=SCENARIOS_PATH,
        scenarios=scenarios,
    )

    assert len(runs) == 40
    assert {run.system for run in runs} == set(LOCAL_SYSTEM_TASKS)
    assert all(run.cost_complete and run.usage.cost_usd == 0.0 for run in runs)
    assert resolved_attachments == [True, True, True, True]


@pytest.mark.parametrize(
    ("field", "value"),
    [("temperature", 0.1), ("seed", 202), ("cache", True)],
)
def test_strict_local_loader_rejects_generation_seed_and_cache_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    manifest, manifest_path, scenarios, logs = _logs(tmp_path)
    setattr(logs["no_memory.eval"].plan.config, field, value)
    monkeypatch.setattr(
        "anamnesis.local_report.read_eval_log",
        lambda path, **_: logs[Path(path).name],
    )

    with pytest.raises(ValueError, match="generation configuration"):
        _load_local_smoke_runs(
            [tmp_path / name for name in logs],
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=MANIFEST_SHA256,
            scenarios_path=SCENARIOS_PATH,
            scenarios=scenarios,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_strict_local_loader_rejects_missing_or_extra_model_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    manifest, manifest_path, scenarios, logs = _logs(tmp_path)
    events = logs["no_memory.eval"].samples[1].events
    if mutation == "missing":
        events.pop()
        expected = "missing a decision ModelEvent"
    else:
        events.append(
            _model_event(
                "unaccounted",
                _local_decision_schema(LOCAL_OLLAMA_MODEL),
                NO_ACTION_COMPLETION,
            )
        )
        expected = "unaccounted ModelEvents"
    monkeypatch.setattr(
        "anamnesis.local_report.read_eval_log",
        lambda path, **_: logs[Path(path).name],
    )

    with pytest.raises(ValueError, match=expected):
        _load_local_smoke_runs(
            [tmp_path / name for name in logs],
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=MANIFEST_SHA256,
            scenarios_path=SCENARIOS_PATH,
            scenarios=scenarios,
        )


def test_strict_local_loader_rejects_missing_raw_call_and_unknown_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_path, scenarios, logs = _logs(tmp_path)
    broken = logs["no_memory.eval"]
    first_model_event = next(
        event for event in broken.samples[0].events if isinstance(event, ModelEvent)
    )
    assert first_model_event.output.usage is not None
    first_model_event.output.usage.total_cost = None
    monkeypatch.setattr(
        "anamnesis.local_report.read_eval_log",
        lambda path, **_: logs[Path(path).name],
    )

    with pytest.raises(ValueError, match="complete zero provider cost"):
        _load_local_smoke_runs(
            [tmp_path / name for name in logs],
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=MANIFEST_SHA256,
            scenarios_path=SCENARIOS_PATH,
            scenarios=scenarios,
        )


def test_local_manifest_report_gate_binds_current_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _frozen_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    preflight = tmp_path / "preflight.eval"
    preflight.write_bytes(b"synthetic")

    def artifact(relative: str) -> Path:
        if relative == manifest.dataset.path:
            return SCENARIOS_PATH.resolve()
        if relative == manifest.model.pricing.path:
            return Path("eval/local_model_costs.json").resolve()
        if relative == manifest.model.preflight.path:
            return preflight
        raise AssertionError(relative)

    monkeypatch.setattr("anamnesis.local_report._repo_artifact", artifact)
    monkeypatch.setattr(
        "anamnesis.local_report.verify_static_local_inputs", lambda *_, **__: None
    )
    monkeypatch.setattr(
        "anamnesis.local_report.validate_local_preflight_artifact",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "anamnesis.local_report._verify_current_git_state", lambda *_, **__: None
    )

    frozen, scenarios, digest = _validate_frozen_local_manifest(
        manifest_path=manifest_path,
        scenarios_path=SCENARIOS_PATH.resolve(),
    )

    assert frozen == manifest
    assert len(scenarios) == 10
    assert digest == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_local_outputs_have_exact_diagnostic_title_and_no_success_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_path, scenarios, logs = _logs(tmp_path)
    monkeypatch.setattr(
        "anamnesis.local_report.read_eval_log",
        lambda path, **_: logs[Path(path).name],
    )
    runs = _load_local_smoke_runs(
        [tmp_path / name for name in logs],
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=MANIFEST_SHA256,
        scenarios_path=SCENARIOS_PATH,
        scenarios=scenarios,
    )
    by_id = {scenario.id: scenario for scenario in scenarios}
    results = aggregate_results(
        (score_scenario(by_id[run.scenario_id], run), run) for run in runs
    )
    csv_path = tmp_path / "local.csv"
    _write_csv(csv_path, results)
    markdown = _render_markdown(results)

    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert b"\r\n" not in csv_path.read_bytes()
    assert len(rows) == 4
    assert {row["system"] for row in rows} == set(LOCAL_SYSTEM_TASKS)
    assert all(row["cost_usd"] == "0.0" for row in rows)
    assert markdown.startswith(f"# {LOCAL_SMOKE_TITLE}\n")
    assert "Preregistered success gate" not in markdown
    assert "Electricity and hardware cost are unmeasured" in markdown


def _provenance_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    LocalExperimentManifest,
    Path,
    Path,
    list[Path],
    list[str],
    Path,
    Path,
    Path,
]:
    repo = tmp_path / "repo"
    (repo / "eval/scenarios").mkdir(parents=True)
    (repo / "results/runs/local").mkdir(parents=True)
    monkeypatch.setattr("anamnesis.local_report.REPO_ROOT", repo)

    scenarios_path = repo / "eval/scenarios/smoke.jsonl"
    scenarios_path.write_text('{"id":"provenance-fixture"}\n', encoding="utf-8")
    manifest = _frozen_manifest()
    manifest = manifest.model_copy(
        update={
            "dataset": manifest.dataset.model_copy(
                update={
                    "path": "eval/scenarios/smoke.jsonl",
                    "sha256": _sha256_file(scenarios_path),
                }
            )
        }
    )
    manifest_path = repo / "results/runs/local/local_smoke_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    run_paths = []
    for system in LOCAL_SYSTEM_TASKS:
        run_path = repo / f"results/runs/local/{system}.eval"
        run_path.write_bytes(f"validated {system} log".encode())
        run_paths.append(run_path)
    run_sha256 = [_sha256_file(path) for path in run_paths]

    csv_path = repo / "results/local_smoke.csv"
    csv_path.write_text("system,f1\nno_memory,0.0\n", encoding="utf-8")
    markdown_path = repo / "results/local_smoke.md"
    markdown_path.write_text(f"# {LOCAL_SMOKE_TITLE}\n", encoding="utf-8")
    sidecar_path = repo / LOCAL_SMOKE_PROVENANCE_PATH
    return (
        manifest,
        manifest_path,
        scenarios_path,
        run_paths,
        run_sha256,
        csv_path,
        markdown_path,
        sidecar_path,
    )


def test_result_provenance_sidecar_binds_exact_inputs_and_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        manifest_path,
        scenarios_path,
        run_paths,
        run_sha256,
        csv_path,
        markdown_path,
        sidecar_path,
    ) = _provenance_inputs(tmp_path, monkeypatch)
    manifest_sha256 = _sha256_file(manifest_path)

    _write_result_provenance(
        sidecar_path,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        scenarios_path=scenarios_path,
        run_paths=run_paths,
        expected_run_sha256=run_sha256,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )

    raw = sidecar_path.read_text(encoding="utf-8")
    sidecar = json.loads(raw)
    assert sidecar == {
        "schema_version": 1,
        "artifact": "anamnesis_local_smoke_result_provenance",
        "title": LOCAL_SMOKE_TITLE,
        "hypothesis_test_eligible": False,
        "source_git_commit": manifest.git_commit,
        "frozen_manifest": {
            "path": "results/runs/local/local_smoke_manifest.json",
            "sha256": manifest_sha256,
        },
        "scenario_dataset": {
            "path": "eval/scenarios/smoke.jsonl",
            "sha256": _sha256_file(scenarios_path),
        },
        "input_eval_logs": [
            {
                "path": f"results/runs/local/{system}.eval",
                "sha256": digest,
            }
            for system, digest in zip(
                LOCAL_SYSTEM_TASKS,
                run_sha256,
                strict=True,
            )
        ],
        "outputs": {
            "csv": {
                "path": "results/local_smoke.csv",
                "sha256": _sha256_file(csv_path),
            },
            "markdown": {
                "path": "results/local_smoke.md",
                "sha256": _sha256_file(markdown_path),
            },
        },
    }
    assert str(tmp_path) not in raw


def test_result_provenance_is_last_and_rejects_changed_or_external_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        manifest_path,
        scenarios_path,
        run_paths,
        run_sha256,
        csv_path,
        markdown_path,
        sidecar_path,
    ) = _provenance_inputs(tmp_path, monkeypatch)
    manifest_sha256 = _sha256_file(manifest_path)
    markdown_path.unlink()

    with pytest.raises(ValueError, match="Markdown output does not exist"):
        _write_result_provenance(
            sidecar_path,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            scenarios_path=scenarios_path,
            run_paths=run_paths,
            expected_run_sha256=run_sha256,
            csv_path=csv_path,
            markdown_path=markdown_path,
        )
    assert not sidecar_path.exists()

    markdown_path.write_text(f"# {LOCAL_SMOKE_TITLE}\n", encoding="utf-8")
    run_paths[0].write_bytes(b"changed after validation")
    with pytest.raises(ValueError, match=r"input \.eval log 1 changed"):
        _write_result_provenance(
            sidecar_path,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            scenarios_path=scenarios_path,
            run_paths=run_paths,
            expected_run_sha256=run_sha256,
            csv_path=csv_path,
            markdown_path=markdown_path,
        )
    assert not sidecar_path.exists()

    outside = tmp_path / "outside.eval"
    outside.write_bytes(b"external")
    with pytest.raises(ValueError, match="must be inside the repository"):
        _repo_relative_path(outside, label="input .eval log")


def test_local_provenance_default_is_the_tracked_result_sidecar() -> None:
    assert Path("results/local_smoke.provenance.json") == LOCAL_SMOKE_PROVENANCE_PATH
