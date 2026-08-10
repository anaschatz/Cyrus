from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
from pydantic import ValidationError

import anamnesis.writer_report as writer_report_dispatcher
import anamnesis.writer_report_w3 as writer_report_w3
from anamnesis.experiment import ArtifactPin
from anamnesis.local_experiment import (
    LOCAL_WRITER_W3_DATASET_SHA256,
    LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
    LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
    LOCAL_WRITER_W3_REFERENCE_SHA256,
    LocalExperimentManifest,
)
from anamnesis.local_preflight import validate_local_w3_preflight_artifact
from anamnesis.local_runtime import (
    LOCAL_MODEL_PREFLIGHT_W3_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W3_SAMPLE_ID,
    LOCAL_MODEL_PREFLIGHT_W3_TASK_VERSION,
    LOCAL_OLLAMA_BASE_URL,
    LOCAL_OLLAMA_CONTEXT_LENGTH,
    LOCAL_OLLAMA_FAMILY,
    LOCAL_OLLAMA_MANIFEST_SHA256,
    LOCAL_OLLAMA_MODEL,
    LOCAL_OLLAMA_PARAMETER_SIZE,
    LOCAL_OLLAMA_QUANTIZATION,
    LOCAL_OLLAMA_SERVICE_MODEL,
    LocalLoadedModelAttestation,
    LocalModelPreflightW3CaseResult,
    LocalModelPreflightW3Result,
    LocalOllamaRuntimeAttestation,
    _local_decision_schema,
    _local_memory_delta_schema,
    load_local_w3_preflight_fixture,
    local_memory_compiler_schema_contract,
    local_memory_compiler_transport_contract,
    local_memory_compiler_w2_transport_contract,
    local_memory_compiler_w3_prompt_contract,
    local_memory_compiler_w3_transport_contract,
    local_system_config_sha256,
    local_w3_preflight_prompts,
    run_local_model_preflight_w3,
)
from anamnesis.local_wire import build_local_memory_compiler_w3_prompt
from anamnesis.runner import DecisionCall, DecisionRequest
from anamnesis.schema import Decision, ObservableEvent, Usage
from anamnesis.writer_report_w3 import (
    WRITER_W3_TITLE,
    _candidate_confusion,
    _validate_w3_setup_latency,
    _w3_candidate_key,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "eval" / "preflight" / "local_writer_w3.v1.json"
TEMPLATE_PATH = ROOT / "eval" / "local_writer_w3_experiment_manifest.template.json"
PRICING_SHA256 = "c185e2fad06d6bd2abaaf0be81a1720fc245555fa2a477c1b1bea558b28c2f74"


def _model_usage(input_tokens: int, output_tokens: int) -> ModelUsage:
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost=0.0,
    )


def _loaded_model() -> LocalLoadedModelAttestation:
    return LocalLoadedModelAttestation(
        model=LOCAL_OLLAMA_SERVICE_MODEL,
        digest=LOCAL_OLLAMA_MANIFEST_SHA256,
        family=LOCAL_OLLAMA_FAMILY,
        parameter_size=LOCAL_OLLAMA_PARAMETER_SIZE,
        quantization_level=LOCAL_OLLAMA_QUANTIZATION,
        context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
        size_vram=3_169_761_361,
        ollama_version="0.31.1",
    )


def test_w3_preflight_runs_exact_input_only_c1_to_c8_then_d1() -> None:
    fixture = load_local_w3_preflight_fixture(FIXTURE_PATH)
    compiler_cases = fixture["compiler_cases"]
    assert isinstance(compiler_cases, list)
    completions = [json.dumps(case["valid_wire_example"]) for case in compiler_cases]

    class FakeW3Model:
        name = LOCAL_OLLAMA_MODEL
        runtime_attestation = LocalOllamaRuntimeAttestation(
            model=LOCAL_OLLAMA_MODEL,
            base_url=LOCAL_OLLAMA_BASE_URL,
            no_cloud="1",
            context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
            host="127.0.0.1:11434",
            num_parallel=1,
            max_loaded_models=1,
        )

        def __init__(self) -> None:
            self.compiler_prompts: list[str] = []
            self.decision_prompts: list[str] = []

        async def complete_structured(self, *, prompt, response_schema):
            assert response_schema.name == "anamnesis_local_memory_delta"
            index = len(self.compiler_prompts)
            self.compiler_prompts.append(prompt)
            return (
                ModelOutput(
                    model=LOCAL_OLLAMA_SERVICE_MODEL,
                    completion=completions[index],
                    usage=_model_usage(100 + index, 10 + index),
                ),
                2.0,
            )

        async def decide(self, request: DecisionRequest) -> DecisionCall:
            self.decision_prompts.append(request.prompt)
            return DecisionCall(
                decision=Decision(),
                usage=Usage(
                    input_tokens=150,
                    uncached_input_tokens=150,
                    output_tokens=4,
                    cost_usd=0.0,
                ),
                latency_ms=3.0,
                raw_completion='{"mode":"no_action","actions":[]}',
                usage_complete=True,
                cost_complete=True,
            )

    model = FakeW3Model()
    result = asyncio.run(
        run_local_model_preflight_w3(  # type: ignore[arg-type]
            model,
            fixture=fixture,
            residency_probe=_loaded_model,
        )
    )

    assert isinstance(result, LocalModelPreflightW3Result)
    assert result.passed
    assert [(case.case_id, case.role) for case in result.cases] == [
        *((f"C{index}", "compiler") for index in range(1, 9)),
        ("D1", "decision"),
    ]
    assert all(case.semantic_valid and not case.parse_error for case in result.cases)
    assert len(model.compiler_prompts) == 8
    assert len(model.decision_prompts) == 1
    for prompt, case in zip(model.compiler_prompts, compiler_cases, strict=True):
        case_input = case["input"]
        assert prompt == build_local_memory_compiler_w3_prompt(
            event=ObservableEvent.model_validate(case_input["event"]),
            active_state=case_input["active_state"],
        )
        for forbidden in (
            str(case["category"]),
            "valid_wire_example",
            "valid_domain_example",
            "acceptance",
            "custodian_audit",
        ):
            assert forbidden not in prompt


def test_w3_prompt_hashes_stay_frozen_while_current_system_uses_runtime_v2() -> None:
    assert hashlib.sha256(
        local_memory_compiler_w3_prompt_contract().encode()
    ).hexdigest() == (
        "412a63d6b42ea6b5e294401cabbcbacf5a6b7facddbd8fe04ca7b91914c141e5"
    )
    assert hashlib.sha256(
        local_memory_compiler_schema_contract().encode()
    ).hexdigest() == (
        "8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030"
    )
    assert hashlib.sha256(
        local_memory_compiler_w3_transport_contract().encode()
    ).hexdigest() == (
        "57d4c0a6152c5319fcd1adab4071ad010d107f9e65d987c1740fa47adaca1bcc"
    )
    assert hashlib.sha256(
        local_memory_compiler_transport_contract().encode()
    ).hexdigest() == (
        "d90471077b929d65737e0d098dbd0c2a12c67f6c75ff31209fca6a63782a7067"
    )
    assert hashlib.sha256(
        local_memory_compiler_w2_transport_contract().encode()
    ).hexdigest() == (
        "5187889e0b2bb998d73857f9ad6c0b252141cf07b3c387567fa9e25bfb7f9a89"
    )
    assert (
        local_system_config_sha256(
            system="anamnesis",
            pricing_config_sha256=PRICING_SHA256,
            compiler_prompt_variant="w3",
            w3_preflight_fixture_sha256=LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
            w3_preflight_protocol_sha256=LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
            w3_dataset_sha256=LOCAL_WRITER_W3_DATASET_SHA256,
            w3_reference_sha256=LOCAL_WRITER_W3_REFERENCE_SHA256,
        )
        == "db2d51d9e8a70a9456a6f01808e453603a55ae8df7a1e5a1097583119c817c97"
    )


def _serialized_model_event(prompt: str, schema, completion: str) -> ModelEvent:
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
        extra_headers={"x-irid": "local-w3-test-request"},
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
        timestamp=datetime.fromisoformat("2034-01-01T00:00:00+00:00"),
        working_start=0.0,
    )


def test_w3_preflight_serialized_eval_round_trip_is_strictly_valid(
    tmp_path: Path,
) -> None:
    fixture = load_local_w3_preflight_fixture(FIXTURE_PATH)
    compiler_cases = fixture["compiler_cases"]
    decision_cases = fixture["decision_cases"]
    assert isinstance(compiler_cases, list)
    assert isinstance(decision_cases, list)
    prompts = local_w3_preflight_prompts(fixture)
    completions = [
        *(json.dumps(case["valid_wire_example"]) for case in compiler_cases),
        json.dumps(decision_cases[0]["valid_wire_example"]),
    ]
    schemas = [
        *(_local_memory_delta_schema(LOCAL_OLLAMA_MODEL) for _ in range(8)),
        _local_decision_schema(LOCAL_OLLAMA_MODEL),
    ]
    events = [
        _serialized_model_event(prompt, schema, completion)
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
    result = LocalModelPreflightW3Result(
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
        loaded_model=_loaded_model(),
        same_model_for_compiler_and_decision=True,
        cases=[
            LocalModelPreflightW3CaseResult(
                case_id=case_id,  # type: ignore[arg-type]
                role="compiler" if case_id.startswith("C") else "decision",
                parse_error=False,
                semantic_valid=True,
                usage=case_usage,
                usage_complete=True,
                cost_complete=True,
                latency_ms=1.0,
            )
            for case_id in (*[f"C{index}" for index in range(1, 9)], "D1")
        ],
        residency_probe_latency_ms=1.0,
        fixture_sha256=LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
        protocol_sha256=LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
        passed=True,
    )
    serialized_result = result.model_dump(mode="json")
    sample = EvalSample.model_construct(
        id=LOCAL_MODEL_PREFLIGHT_W3_SAMPLE_ID,
        epoch=1,
        input="Check the frozen local W3 compiler and decision protocol.",
        target="pass",
        events=events,
        metadata={"anamnesis.local_preflight_w3": serialized_result},
        store={"anamnesis.local_preflight_w3": serialized_result},
        error=None,
        invalidation=None,
        error_retries=[],
        output=ModelOutput.from_content(
            model=LOCAL_OLLAMA_MODEL,
            content=result.model_dump_json(),
        ),
    )
    commit = "a" * 40
    spec = EvalSpec.model_construct(
        created="2034-01-01T00:00:00+00:00",
        task="local_model_preflight_w3",
        task_registry_name="local_model_preflight_w3",
        task_version=LOCAL_MODEL_PREFLIGHT_W3_TASK_VERSION,
        metadata={
            "purpose": LOCAL_MODEL_PREFLIGHT_W3_PURPOSE,
            "track": "local_zero_api_cost",
            "hypothesis_test_eligible": False,
            "pricing_config_sha256": PRICING_SHA256,
            "preflight_fixture_sha256": LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
            "preflight_protocol_sha256": LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
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
            commit=commit[:7],
            dirty=False,
        ),
        dataset=EvalDataset(),
    )
    log = EvalLog.model_construct(
        status="success",
        invalidated=False,
        config_updates=None,
        log_updates=None,
        eval=spec,
        plan=EvalPlan.model_construct(config=GenerateConfig(cache=False)),
        samples=[sample],
    )
    path = tmp_path / "w3-preflight.eval"
    write_eval_log(log, path, format="eval")
    protocol_path = ROOT / "eval" / "preflight" / "local_writer_w3.protocol.v1.json"
    validated = validate_local_w3_preflight_artifact(
        ArtifactPin(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
        fixture_artifact=ArtifactPin(
            path=str(FIXTURE_PATH),
            sha256=LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
        ),
        protocol_artifact=ArtifactPin(
            path=str(protocol_path),
            sha256=LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
        ),
        expected_git_commit=commit,
        expected_pricing_sha256=PRICING_SHA256,
    )
    assert validated == result


def test_w3_manifest_template_locks_phase_matrix_and_all_public_pins() -> None:
    raw = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    manifest = LocalExperimentManifest.model_validate(raw)

    assert manifest.phase == "writer_diagnostic_w3"
    assert manifest.systems == ["anamnesis"]
    assert manifest.scenario_count == 10
    assert manifest.dataset.sha256 == LOCAL_WRITER_W3_DATASET_SHA256
    assert manifest.writer_reference is not None
    assert manifest.writer_reference.sha256 == LOCAL_WRITER_W3_REFERENCE_SHA256
    assert manifest.preflight_fixture is not None
    assert manifest.preflight_fixture.sha256 == LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256
    assert manifest.preflight_protocol is not None
    assert (
        manifest.preflight_protocol.sha256 == LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256
    )
    assert manifest.execution.warmup_policy == ("frozen_w3_semantic_gate_c1_to_c8_d1")

    wrong_warmup = json.loads(json.dumps(raw))
    wrong_warmup["execution"]["warmup_policy"] = "frozen_w2_semantic_gate_c1_c2_c3_d1"
    with pytest.raises(ValidationError, match="requires warmup_policy"):
        LocalExperimentManifest.model_validate(wrong_warmup)

    missing_protocol = json.loads(json.dumps(raw))
    missing_protocol.pop("preflight_protocol")
    with pytest.raises(ValidationError, match="requires preflight_protocol"):
        LocalExperimentManifest.model_validate(missing_protocol)

    wrong_reference = json.loads(json.dumps(raw))
    wrong_reference["writer_reference"]["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="writer_reference.sha256"):
        LocalExperimentManifest.model_validate(wrong_reference)


def _candidate(*, summary: str, evidence: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        intent_id="runtime-intent",
        occurrence_id="runtime-occurrence",
        action_key="source-event",
        due_at=datetime.fromisoformat("2033-04-12T12:00:00+03:00"),
        action_template=SimpleNamespace(
            kind="reminder",
            payload={"subject": "send note", "item": "Lumen folio"},
            summary=summary,
        ),
        evidence_event_ids=evidence,
    )


def test_w3_candidate_gate_uses_frozen_six_field_multiset_key() -> None:
    left = _w3_candidate_key(
        "checkpoint",
        _candidate(summary="First UX summary", evidence=["e2", "e1"]),
    )
    right = _w3_candidate_key(
        "checkpoint",
        _candidate(summary="Different UX summary", evidence=["e1", "e2"]),
    )

    assert left == right
    assert len(left) == 6
    assert _candidate_confusion(Counter({left: 2}), Counter({right: 1})) == (
        1,
        1,
        0,
    )


def test_w3_setup_latency_is_exact_and_diagnostic_title_is_unmistakable() -> None:
    _validate_w3_setup_latency(1234.567, 1234.567)
    with pytest.raises(ValueError, match="differs from the exact preflight"):
        _validate_w3_setup_latency(1234.568, 1234.567)
    with pytest.raises(ValueError, match="differs from the exact preflight"):
        _validate_w3_setup_latency(1234.566, 1234.567)

    assert WRITER_W3_TITLE == "Local writer W3 diagnostic — not a hypothesis test"


def test_shared_writer_report_dispatches_w3_only_from_manifest_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "w3.json"
    manifest.write_text('{"phase":"writer_diagnostic_w3"}\n', encoding="utf-8")
    monkeypatch.setattr(writer_report_w3, "writer_report_w3_main", lambda argv: 17)

    assert (
        writer_report_dispatcher.writer_report_main(["--manifest", str(manifest)]) == 17
    )
