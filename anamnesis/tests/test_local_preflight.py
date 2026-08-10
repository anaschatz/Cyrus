from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
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
    write_eval_log,
)
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelCall,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._openai import openai_completion_params

from anamnesis.experiment import ArtifactPin
from anamnesis.local_experiment import LOCAL_PRICING_SHA256
from anamnesis.local_preflight import (
    LOCAL_PREFLIGHT_SAMPLE_ID,
    local_preflight_prompts,
    validate_local_preflight_artifact,
    validate_local_preflight_log,
    validate_local_w2_preflight_artifact,
    validate_local_w2_preflight_log,
)
from anamnesis.local_runtime import (
    LOCAL_MODEL_PREFLIGHT_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_TASK_VERSION,
    LOCAL_MODEL_PREFLIGHT_W2_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W2_SAMPLE_ID,
    LOCAL_MODEL_PREFLIGHT_W2_TASK_VERSION,
    LOCAL_OLLAMA_BASE_URL,
    LOCAL_OLLAMA_CONTEXT_LENGTH,
    LOCAL_OLLAMA_FAMILY,
    LOCAL_OLLAMA_MANIFEST_SHA256,
    LOCAL_OLLAMA_MODEL,
    LOCAL_OLLAMA_PARAMETER_SIZE,
    LOCAL_OLLAMA_QUANTIZATION,
    LOCAL_OLLAMA_SERVICE_MODEL,
    LocalLoadedModelAttestation,
    LocalModelPreflightResult,
    LocalModelPreflightW2CaseResult,
    LocalModelPreflightW2Result,
    LocalOllamaRuntimeAttestation,
    _local_decision_schema,
    _local_memory_delta_schema,
    load_local_w2_preflight_fixture,
    local_w2_preflight_prompts,
)
from anamnesis.schema import Usage

COMMIT = "a" * 40


def _compiler_completion() -> str:
    return json.dumps(
        {
            "fact_assertions": [],
            "intent_creates": [
                {
                    "intent_id": "remind-compatibility-check-1700",
                    "trigger": {
                        "type": "at",
                        "at": "2026-01-05T17:00:00Z",
                    },
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


def _model_event(prompt: str, schema, completion: str) -> ModelEvent:
    usage = ModelUsage(
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        total_cost=0.0,
    )
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
        extra_headers={"x-irid": "local-test-request"},
    )
    return ModelEvent.model_construct(
        event="model",
        model=LOCAL_OLLAMA_MODEL,
        input=[ChatMessageUser(content=prompt)],
        tools=[],
        tool_choice="none",
        config=config,
        output=ModelOutput(
            model=LOCAL_OLLAMA_SERVICE_MODEL,
            completion=completion,
            usage=usage,
        ),
        call=ModelCall(
            request=request,
            response={"model": LOCAL_OLLAMA_SERVICE_MODEL},
        ),
        cache=None,
        error=None,
        retries=None,
        timestamp=datetime(2026, 1, 5, 9, 0, tzinfo=UTC),
        working_start=0.0,
    )


def _valid_log() -> EvalLog:
    compiler_prompt, decision_prompt = local_preflight_prompts()
    events = [
        _model_event(
            compiler_prompt,
            _local_memory_delta_schema(LOCAL_OLLAMA_MODEL),
            _compiler_completion(),
        ),
        _model_event(
            decision_prompt,
            _local_decision_schema(LOCAL_OLLAMA_MODEL),
            '{"mode":"no_action","actions":[]}',
        ),
    ]
    compiler_usage = Usage(
        input_tokens=100,
        uncached_input_tokens=100,
        output_tokens=10,
        cost_usd=0.0,
    )
    result = LocalModelPreflightResult(
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
            size_vram=3_169_761_361,
            ollama_version="0.31.1",
        ),
        same_model_for_compiler_and_decision=True,
        compiler_parse_error=False,
        decision_parse_error=False,
        compiler_semantic_valid=True,
        decision_semantic_valid=True,
        compiler_usage=compiler_usage,
        decision_usage=compiler_usage,
        compiler_usage_complete=True,
        decision_usage_complete=True,
        compiler_cost_complete=True,
        decision_cost_complete=True,
        compiler_latency_ms=1.0,
        decision_latency_ms=1.0,
        residency_probe_latency_ms=1.0,
        passed=True,
    )
    sample = EvalSample.model_construct(
        id=LOCAL_PREFLIGHT_SAMPLE_ID,
        epoch=1,
        input="Check local compatibility.",
        target="pass",
        events=events,
        error=None,
        invalidation=None,
        error_retries=[],
        output=ModelOutput.from_content(
            model=LOCAL_OLLAMA_MODEL,
            content=result.model_dump_json(),
        ),
    )
    spec = EvalSpec.model_construct(
        created="2026-01-05T09:00:00+00:00",
        task="local_model_preflight",
        task_registry_name="local_model_preflight",
        task_version=LOCAL_MODEL_PREFLIGHT_TASK_VERSION,
        metadata={
            "purpose": LOCAL_MODEL_PREFLIGHT_PURPOSE,
            "pricing_config_sha256": LOCAL_PRICING_SHA256,
        },
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
            commit=COMMIT[:7],
            dirty=False,
        ),
        dataset=EvalDataset(),
    )
    return EvalLog.model_construct(
        status="success",
        invalidated=False,
        config_updates=None,
        log_updates=None,
        eval=spec,
        plan=EvalPlan.model_construct(config=GenerateConfig(cache=False)),
        samples=[sample],
    )


def _valid_w2_log() -> EvalLog:
    fixture = load_local_w2_preflight_fixture("eval/preflight/local_writer_w2.v1.json")
    prompts = local_w2_preflight_prompts(fixture)
    compiler_cases = fixture["compiler_cases"]
    decision_cases = fixture["decision_cases"]
    assert isinstance(compiler_cases, list)
    assert isinstance(decision_cases, list)
    completions = [
        json.dumps(case["valid_wire_example"]) for case in compiler_cases
    ] + [json.dumps(decision_cases[0]["valid_wire_example"])]
    schemas = [
        _local_memory_delta_schema(LOCAL_OLLAMA_MODEL),
        _local_memory_delta_schema(LOCAL_OLLAMA_MODEL),
        _local_memory_delta_schema(LOCAL_OLLAMA_MODEL),
        _local_decision_schema(LOCAL_OLLAMA_MODEL),
    ]
    events = [
        _model_event(prompt, schema, completion)
        for prompt, schema, completion in zip(
            prompts,
            schemas,
            completions,
            strict=True,
        )
    ]
    case_usage = Usage(
        input_tokens=100,
        uncached_input_tokens=100,
        output_tokens=10,
        cost_usd=0.0,
    )
    result = LocalModelPreflightW2Result(
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
            size_vram=3_169_761_361,
            ollama_version="0.31.1",
        ),
        same_model_for_compiler_and_decision=True,
        cases=[
            LocalModelPreflightW2CaseResult(
                case_id=case_id,  # type: ignore[arg-type]
                role="compiler" if case_id.startswith("C") else "decision",
                parse_error=False,
                semantic_valid=True,
                usage=case_usage,
                usage_complete=True,
                cost_complete=True,
                latency_ms=1.0,
            )
            for case_id in ("C1", "C2", "C3", "D1")
        ],
        residency_probe_latency_ms=1.0,
        fixture_sha256=(
            "3b82128bab1d801d073118488aa4f0a0a662603b98325f5c9d7dad497f026057"
        ),
        passed=True,
    )
    sample = EvalSample.model_construct(
        id=LOCAL_MODEL_PREFLIGHT_W2_SAMPLE_ID,
        epoch=1,
        input="Check the frozen local W2 compiler and decision protocol.",
        target="pass",
        events=events,
        metadata={"anamnesis.local_preflight_w2": result.model_dump(mode="json")},
        store={"anamnesis.local_preflight_w2": result.model_dump(mode="json")},
        error=None,
        invalidation=None,
        error_retries=[],
        output=ModelOutput.from_content(
            model=LOCAL_OLLAMA_MODEL,
            content=result.model_dump_json(),
        ),
    )
    spec = EvalSpec.model_construct(
        created="2026-01-05T09:00:00+00:00",
        task="local_model_preflight_w2",
        task_registry_name="local_model_preflight_w2",
        task_version=LOCAL_MODEL_PREFLIGHT_W2_TASK_VERSION,
        metadata={
            "purpose": LOCAL_MODEL_PREFLIGHT_W2_PURPOSE,
            "track": "local_zero_api_cost",
            "hypothesis_test_eligible": False,
            "pricing_config_sha256": LOCAL_PRICING_SHA256,
            "preflight_fixture_sha256": (
                "3b82128bab1d801d073118488aa4f0a0a662603b98325f5c9d7dad497f026057"
            ),
        },
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
            commit=COMMIT[:7],
            dirty=False,
        ),
        dataset=EvalDataset(),
    )
    return EvalLog.model_construct(
        status="success",
        invalidated=False,
        config_updates=None,
        log_updates=None,
        eval=spec,
        plan=EvalPlan.model_construct(config=GenerateConfig(cache=False)),
        samples=[sample],
    )


def test_local_preflight_log_accepts_exact_semantic_two_call_gate() -> None:
    validate_local_preflight_log(
        _valid_log(),
        expected_git_commit=COMMIT,
        expected_pricing_sha256=LOCAL_PRICING_SHA256,
    )


@pytest.mark.parametrize(
    ("tamper", "expected"),
    [
        ("dirty", "frozen clean commit"),
        ("commit", "frozen clean commit"),
        ("route", "provider route"),
        ("pricing", "pricing hash"),
        ("retry", "cached, retried, or failed"),
        ("response", "raw local response model"),
        ("early_action", "false reminder"),
        ("wrong_time", "compiler preflight semantics"),
    ],
)
def test_local_preflight_log_rejects_drift(tamper: str, expected: str) -> None:
    log = _valid_log()
    if tamper == "dirty":
        log.eval.revision.dirty = True
    elif tamper == "commit":
        log.eval.revision.commit = "b" * 7
    elif tamper == "route":
        log.eval.model_base_url = "https://proxy.example/v1"
    elif tamper == "pricing":
        log.eval.metadata["pricing_config_sha256"] = "0" * 64
    elif tamper == "retry":
        log.samples[0].events[0].retries = 1
    elif tamper == "response":
        log.samples[0].events[0].call.response = None
    elif tamper == "early_action":
        log.samples[0].events[1].output.completion = json.dumps(
            {
                "mode": "emit",
                "actions": [
                    {
                        "action_key": "local-preflight-event",
                        "payload": {"subject": "perform compatibility check"},
                        "summary": "Perform compatibility check.",
                        "evidence_event_ids": ["local-preflight-event"],
                    }
                ],
            }
        )
    else:
        completion = json.loads(log.samples[0].events[0].output.completion)
        completion["intent_creates"][0]["trigger"]["at"] = "2026-01-05T09:00:00Z"
        log.samples[0].events[0].output.completion = json.dumps(completion)

    with pytest.raises(ValueError, match=expected):
        validate_local_preflight_log(
            log,
            expected_git_commit=COMMIT,
            expected_pricing_sha256=LOCAL_PRICING_SHA256,
        )


def test_local_preflight_artifact_binds_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model_preflight.eval"
    path.write_bytes(b"pinned local Inspect log")
    artifact = ArtifactPin(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "anamnesis.local_preflight.read_eval_log",
        lambda *args, **kwargs: _valid_log(),
    )

    validate_local_preflight_artifact(
        artifact,
        expected_git_commit=COMMIT,
        expected_pricing_sha256=LOCAL_PRICING_SHA256,
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_local_preflight_artifact(
            artifact.model_copy(update={"sha256": "0" * 64}),
            expected_git_commit=COMMIT,
            expected_pricing_sha256=LOCAL_PRICING_SHA256,
        )


def test_serialized_eval_roundtrip_uses_attachment_resolution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model_preflight.eval"
    write_eval_log(_valid_log(), path, format="eval")
    artifact = ArtifactPin(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    validate_local_preflight_artifact(
        artifact,
        expected_git_commit=COMMIT,
        expected_pricing_sha256=LOCAL_PRICING_SHA256,
    )


def test_local_w2_preflight_log_accepts_exact_four_call_fixture_gate() -> None:
    fixture = load_local_w2_preflight_fixture("eval/preflight/local_writer_w2.v1.json")

    result = validate_local_w2_preflight_log(
        _valid_w2_log(),
        fixture=fixture,
        expected_git_commit=COMMIT,
        expected_pricing_sha256=LOCAL_PRICING_SHA256,
    )

    assert result.passed
    assert result.usage.input_tokens == 400
    assert [case.case_id for case in result.cases] == ["C1", "C2", "C3", "D1"]


@pytest.mark.parametrize(
    ("tamper", "expected"),
    [
        ("order", "prompt differs"),
        ("fifth_call", "exactly four"),
        ("schema", "unpinned response schema"),
        ("result", "result flags"),
        ("metadata", "task identity"),
    ],
)
def test_local_w2_preflight_log_rejects_raw_or_result_drift(
    tamper: str,
    expected: str,
) -> None:
    fixture = load_local_w2_preflight_fixture("eval/preflight/local_writer_w2.v1.json")
    log = _valid_w2_log()
    if tamper == "order":
        log.samples[0].events[0], log.samples[0].events[1] = (
            log.samples[0].events[1],
            log.samples[0].events[0],
        )
    elif tamper == "fifth_call":
        log.samples[0].events.append(log.samples[0].events[0])
    elif tamper == "schema":
        log.samples[0].events[0].config.response_schema = _local_decision_schema(
            LOCAL_OLLAMA_MODEL
        )
    elif tamper == "result":
        raw = json.loads(log.samples[0].output.completion)
        raw["cases"][0]["semantic_valid"] = False
        log.samples[0].output.completion = json.dumps(raw)
        log.samples[0].metadata = {"anamnesis.local_preflight_w2": raw}
        log.samples[0].store = {"anamnesis.local_preflight_w2": raw}
    else:
        log.eval.metadata["preflight_fixture_sha256"] = "0" * 64

    with pytest.raises(ValueError, match=expected):
        validate_local_w2_preflight_log(
            log,
            fixture=fixture,
            expected_git_commit=COMMIT,
            expected_pricing_sha256=LOCAL_PRICING_SHA256,
        )


def test_local_w2_preflight_artifact_binds_fixture_and_log_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_source = Path("eval/preflight/local_writer_w2.v1.json")
    fixture_path = tmp_path / "local_writer_w2.v1.json"
    fixture_path.write_bytes(fixture_source.read_bytes())
    fixture_artifact = ArtifactPin(
        path=str(fixture_path),
        sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
    )
    log_path = tmp_path / "model_preflight.eval"
    log_path.write_bytes(b"pinned W2 Inspect log")
    log_artifact = ArtifactPin(
        path=str(log_path),
        sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "anamnesis.local_preflight.read_eval_log",
        lambda *args, **kwargs: _valid_w2_log(),
    )

    assert validate_local_w2_preflight_artifact(
        log_artifact,
        fixture_artifact=fixture_artifact,
        expected_git_commit=COMMIT,
        expected_pricing_sha256=LOCAL_PRICING_SHA256,
    ).passed
    with pytest.raises(ValueError, match="fixture artifact hash mismatch"):
        validate_local_w2_preflight_artifact(
            log_artifact,
            fixture_artifact=fixture_artifact.model_copy(update={"sha256": "0" * 64}),
            expected_git_commit=COMMIT,
            expected_pricing_sha256=LOCAL_PRICING_SHA256,
        )
