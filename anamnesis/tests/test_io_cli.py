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


@pytest.mark.parametrize("body", ["[]", "null", "[" * 1100 + "]" * 1100])
def test_final_dataset_release_rejects_invalid_manifest_shapes(
    tmp_path: Path,
    body: str,
) -> None:
    dataset_path = tmp_path / "all.jsonl"
    dataset_path.with_suffix(".manifest.json").write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid final dataset release manifest"):
        require_preregistered_final_dataset(dataset_path, [])


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


@pytest.mark.parametrize(
    "system,tamper",
    [
        (system, tamper)
        for system in ("no_memory", "anamnesis")
        for tamper in [
            "none",
            "action",
            "parse-failure",
            "predictions",
            "stored-output",
            "checkpoint-output",
            "prompt-hash",
            "parse-flag",
            "missing-event",
            "extra-event",
            "event-order",
            "warmup-content",
            "warmup-prompt",
            "checkpoint-usage",
            "compiler-output",
            "compiler-delta",
        ]
        if system == "anamnesis" or not tamper.startswith("compiler-")
    ],
)
def test_hosted_loader_binds_producer_run_to_model_events(
    system: str,
    tamper: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from inspect_ai.log import EvalStats
    from inspect_ai.model import ModelOutput
    from anamnesis.baselines import AnamnesisMemoryStrategy, NoPersistentMemory
    from anamnesis.inspect_adapter import (
        InspectDecisionModel,
        InspectMemoryCompiler,
        _run_hosted_model_warmup,
    )
    from anamnesis.runner import run_scenario
    from anamnesis.wire import DecisionWire, PayloadWire
    from test_preflight import MODEL, _model_event, _valid_preflight_log

    scenario = load_scenarios("eval/scenarios/smoke.jsonl")[0]
    action_json = json.dumps(
        {
            "actions": [
                {
                    "kind": "reminder",
                    "action_key": scenario.events[0].id,
                    "payload": dict(
                        dict.fromkeys(PayloadWire.model_fields),
                        subject="send assignment",
                    ),
                    "summary": "Send assignment",
                    "evidence_event_ids": [scenario.events[0].id],
                }
            ]
        }
    )
    events = []

    async def generate(state, *, response_schema, **kwargs):
        completion = '{"actions":[]}'
        if response_schema.name == "anamnesis_memory_delta":
            completion = json.dumps(
                dict.fromkeys(
                    (
                        "fact_assertions",
                        "intent_creates",
                        "intent_updates",
                        "intent_cancellations",
                    ),
                    [],
                )
            )
        elif events and tamper in ("action", "parse-failure"):
            completion = action_json if tamper == "action" else "invalid JSON"
        event = _model_event(state.messages[0].content, response_schema, completion)
        events.append(event)
        state.output = event.output
        return state

    monkeypatch.setattr(
        "anamnesis.inspect_adapter.get_model",
        lambda: SimpleNamespace(
            api=SimpleNamespace(
                client=SimpleNamespace(base_url="https://api.openai.com/v1")
            ),
        ),
    )
    manifest = ExperimentManifest.model_validate_json(TEMPLATE.read_bytes())
    manifest.model.snapshot = MODEL
    manifest_hash = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()

    async def produce():
        model = InspectDecisionModel(SimpleNamespace(model=MODEL), generate)
        warmup = await _run_hosted_model_warmup(model)
        strategy = (
            AnamnesisMemoryStrategy(InspectMemoryCompiler(model))
            if system == "anamnesis"
            else NoPersistentMemory()
        )
        return await run_scenario(
            scenario=scenario,
            strategy=strategy,
            model=model,
            seed=101,
            manifest_sha256=manifest_hash,
            hosted_warmup=warmup,
            setup_latency_ms=warmup.latency_ms,
        )

    run = asyncio.run(produce())
    raw_run = run.model_dump(mode="json")
    checkpoint = raw_run["checkpoints"][0]
    if tamper == "predictions":
        raw_run["predictions"] = [
            dict(
                DecisionWire.model_validate_json(action_json)
                .to_domain()
                .actions[0]
                .model_dump(),
                emitted_at=checkpoint["at"],
                decision_event_id=checkpoint["event_id"],
            )
        ]
    elif tamper == "checkpoint-output":
        checkpoint["raw_decision_output"] = action_json
    elif tamper == "prompt-hash":
        checkpoint["rendered_context_sha256"] = "f" * 64
    elif tamper == "parse-flag":
        checkpoint["decision_parse_error"] = True
        raw_run["decision_parse_errors"] = raw_run["parse_errors"] = 1
    elif tamper == "missing-event":
        events.pop()
    elif tamper == "extra-event":
        events.append(events[-1])
    elif tamper == "event-order":
        events[1], events[2] = events[2], events[1]
    elif tamper == "warmup-content":
        raw_run["hosted_warmup"]["raw_completion"] = action_json
    elif tamper == "warmup-prompt":
        events[0].input[0].content = "different warmup"
        events[0].call.request["messages"][0]["content"] = "different warmup"
    elif tamper == "checkpoint-usage":
        events[-1].output.usage.input_tokens += 1
        events[-1].output.usage.total_tokens += 1
        events[-1].call.response["usage"]["prompt_tokens"] += 1
        events[-1].call.response["usage"]["total_tokens"] += 1
    elif tamper == "compiler-output":
        checkpoint["raw_compiler_output"] = "invalid JSON"
    elif tamper == "compiler-delta":
        checkpoint["memory_delta_json"] = "invalid JSON"
    run = ScenarioRun.model_validate(raw_run)
    log = _valid_preflight_log()
    sample = log.samples[0]
    sample.id = run.scenario_id
    sample.events = events
    sample.store = {"anamnesis.scenario_run": run.model_dump(mode="json")}
    sample.output = ModelOutput.from_content(
        model=MODEL,
        content=run.model_dump_json(),
    )
    if tamper == "stored-output":
        sample.output.completion = run.model_copy(
            update={"prompt_sha256": "f" * 64}
        ).model_dump_json()
    usage = run.usage.plus(run.hosted_warmup.usage)
    log.stats = EvalStats(
        model_usage={
            MODEL: ModelUsage(
                input_tokens=usage.uncached_input_tokens,
                output_tokens=usage.output_tokens,
                total_cost=usage.cost_usd,
            )
        }
    )

    def read_log(path, *, resolve_attachments=False):
        assert resolve_attachments is True
        return log

    monkeypatch.setattr("anamnesis.cli.read_eval_log", read_log)
    monkeypatch.setattr(
        "anamnesis.cli._validate_system_log_metadata", lambda *a, **k: (system, 1)
    )
    kwargs = dict(
        manifest=manifest,
        manifest_path=TEMPLATE,
        mode="baseline",
        scenarios_path=Path("eval/scenarios/smoke.jsonl"),
        scenario_ids=[scenario.id],
    )
    if tamper in ("none", "action", "parse-failure"):
        assert _load_strict_eval_runs([Path("synthetic.eval")], **kwargs) == [run]
    else:
        with pytest.raises(ValueError):
            _load_strict_eval_runs([Path("synthetic.eval")], **kwargs)
