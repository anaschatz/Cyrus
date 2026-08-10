from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.model import ModelInfo, ModelOutput, ModelUsage
from pydantic import ValidationError

from anamnesis.baselines import NoPersistentMemory
from anamnesis.io import load_scenarios
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
    LOCAL_STRUCTURED_MEMORY_PRECEDENCE,
    LOCAL_ZERO_MODEL_COST,
    LocalDecisionWire,
    LocalLoadedModelAttestation,
    LocalModelPreflightW2Result,
    LocalOllamaRuntimeAttestation,
    _loaded_model_from_ps,
    _local_memory_delta_schema,
    _local_usage_from_output,
    _validate_oracle_scenario_run,
    _verify_effective_zero_model_cost,
    _verify_local_ollama_runtime,
    build_local_decision_prompt,
    load_local_w2_preflight_fixture,
    local_decision_contract,
    local_decision_prompt_contract,
    local_decision_schema_contract,
    local_memory_compiler_prompt_contract,
    local_memory_compiler_schema_contract,
    local_memory_compiler_transport_contract,
    local_memory_compiler_w2_prompt_contract,
    local_memory_compiler_w2_transport_contract,
    local_scenario_solver,
    local_system_config_sha256,
    run_local_model_preflight,
    run_local_model_preflight_w2,
)
from anamnesis.local_wire import (
    LOCAL_MEMORY_COMPILER_INSTRUCTIONS,
    LOCAL_MEMORY_COMPILER_VERSION,
    build_local_memory_compiler_prompt,
    build_local_memory_compiler_w2_prompt,
)
from anamnesis.memory import CancelIntent, MemoryDelta
from anamnesis.oracle import (
    ORACLE_COMPILER_VERSION,
    ORACLE_SYSTEM_NAME,
    OracleAnamnesisMemoryStrategy,
    OracleCompiler,
    load_oracle_artifact,
)
from anamnesis.runner import DecisionCall, DecisionRequest, run_scenario
from anamnesis.schema import (
    Decision,
    MemoryView,
    MemoryViewBlock,
    ObservableEvent,
    RuntimeScenario,
    Usage,
)


def _environment() -> dict[str, str]:
    return {
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_CONTEXT_LENGTH": "4096",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }


def _active_model(base_url: str = LOCAL_OLLAMA_BASE_URL) -> object:
    return SimpleNamespace(
        api=SimpleNamespace(
            service="Ollama",
            base_url=base_url,
            client=SimpleNamespace(base_url=f"{base_url}/"),
        )
    )


def _model_usage(input_tokens: int = 10, output_tokens: int = 2) -> ModelUsage:
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


def test_local_decision_discriminator_rejects_inconsistent_actions() -> None:
    assert (
        LocalDecisionWire.model_validate(
            {"mode": "no_action", "actions": []}
        ).to_domain()
        == Decision()
    )

    with pytest.raises(ValidationError, match="no_action"):
        LocalDecisionWire.model_validate(
            {
                "mode": "no_action",
                "actions": [
                    {
                        "action_key": "event-1",
                        "payload": {"subject": "send the assignment"},
                        "summary": "Send it.",
                        "evidence_event_ids": ["event-1"],
                    }
                ],
            }
        )
    with pytest.raises(ValidationError, match="emit"):
        LocalDecisionWire.model_validate({"mode": "emit", "actions": []})


def test_local_decision_prompt_uses_local_wire_and_shared_d1_contract() -> None:
    event = ObservableEvent(
        id="event-1",
        at="2026-01-05T09:00:00+00:00",
        kind="user_message",
        text="At 17:00 remind me to send the assignment.",
    )
    prompt = build_local_decision_prompt(
        now=event.at.isoformat(),
        current_event_id=event.id,
        context_events=[event],
        decision_history=[],
        memory_view=None,
    )

    assert '"mode":"no_action"' in prompt
    assert 'Return {"actions": []}' not in prompt
    assert "Use null for every unused wire slot" not in prompt
    assert "Omit every unused optional payload slot" in prompt
    assert event.text in prompt
    assert "supersedes" not in prompt
    assert LOCAL_STRUCTURED_MEMORY_PRECEDENCE in prompt
    assert "- (not provided by this system)" in prompt
    assert local_decision_contract() == local_decision_contract()


def test_local_decision_d1_rules_are_exact_for_absent_and_empty_views() -> None:
    expected_rules = (
        "Structured-memory precedence (D1):\n"
        "- When Structured memory view is provided by this system, it is "
        "authoritative.\n"
        "- If it contains zero DUE_CANDIDATE blocks, set mode=no_action with an "
        "empty actions array regardless of wording in Available context.\n"
        "- For each DUE_CANDIDATE block, emit exactly one action and copy kind, "
        "action_key, payload, and summary value-for-value from that block's JSON.\n"
        "- Set evidence_event_ids to exactly the block's evidence IDs in displayed "
        "order, followed by Current decision event if it is not already present; "
        "include no other IDs.\n"
        "- A prior EXECUTION suppresses only a DUE_CANDIDATE with the same "
        "occurrence_id. A different occurrence_id or date is a distinct recurring "
        "occurrence even when action_key is the same.\n"
        "- When Structured memory view is not provided by this system, use the "
        "general Rules above.\n"
    )
    assert expected_rules == LOCAL_STRUCTURED_MEMORY_PRECEDENCE

    event = ObservableEvent(
        id="checkpoint-1",
        at="2026-01-05T17:00:00+00:00",
        kind="clock_tick",
        text="Decision checkpoint.",
    )
    common = {
        "now": event.at.isoformat(),
        "current_event_id": event.id,
        "context_events": [event],
        "decision_history": [],
    }
    absent_prompt = build_local_decision_prompt(memory_view=None, **common)
    empty_prompt = build_local_decision_prompt(memory_view=MemoryView(), **common)

    assert expected_rules in absent_prompt
    assert expected_rules in empty_prompt
    assert "- (not provided by this system)" in absent_prompt
    assert "- (empty; no structured candidate is due)" in empty_prompt


def test_local_decision_d1_renders_due_and_execution_occurrences_exactly() -> None:
    event = ObservableEvent(
        id="checkpoint-2",
        at="2026-01-06T17:00:00+00:00",
        kind="clock_tick",
        text="Decision checkpoint.",
    )
    due_content = (
        '{"action_key":"request-1","due_at":"2026-01-06T17:00:00+00:00",'
        '"intent_id":"daily-request","kind":"reminder",'
        '"occurrence_id":"daily-request:2026-01-06",'
        '"payload":{"date":"2026-01-06","subject":"check the report"},'
        '"summary":"Check the report."}'
    )
    execution_content = (
        '{"emitted_at":"2026-01-05T17:00:00+00:00",'
        '"occurrence_id":"daily-request:2026-01-05",'
        '"payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
    )
    memory_view = MemoryView(
        blocks=[
            MemoryViewBlock(
                kind="due_candidate",
                title="Due reminder: Check the report.",
                content=due_content,
                evidence_event_ids=["request-1", "update-1"],
            ),
            MemoryViewBlock(
                kind="execution",
                title="Prior execution: request-1",
                content=execution_content,
                evidence_event_ids=["request-1", "checkpoint-1"],
            ),
        ]
    )

    prompt = build_local_decision_prompt(
        now=event.at.isoformat(),
        current_event_id=event.id,
        context_events=[event],
        decision_history=[],
        memory_view=memory_view,
    )

    assert (
        "- DUE_CANDIDATE | Due reminder: Check the report. | "
        f'{due_content} | evidence=["request-1","update-1"]'
    ) in prompt
    assert (
        "- EXECUTION | Prior execution: request-1 | "
        f'{execution_content} | evidence=["request-1","checkpoint-1"]'
    ) in prompt
    assert (
        "A prior EXECUTION suppresses only a DUE_CANDIDATE with the same occurrence_id."
    ) in prompt
    assert (
        "followed by Current decision event if it is not already present; "
        "include no other IDs."
    ) in prompt


def test_local_decision_d1_version_and_hash_drift_preserve_schema() -> None:
    prompt_sha256 = hashlib.sha256(
        local_decision_prompt_contract().encode()
    ).hexdigest()
    schema_sha256 = hashlib.sha256(
        local_decision_schema_contract().encode()
    ).hexdigest()
    contract_sha256 = hashlib.sha256(local_decision_contract().encode()).hexdigest()

    assert LOCAL_DECISION_VERSION == "ollama.decision.v0.2"
    assert prompt_sha256 == (
        "871fe15e3160e66abe7480cbde15dfb943dec2d0ff89bb01a03849ad35defd8d"
    )
    assert prompt_sha256 != (
        "0a258709387f47fdcc80e9ab701983939df3961695ba7396e6b0be38524991ec"
    )
    assert schema_sha256 == (
        "1b7c38d3f4bf150523ecc1e468ad3fb1f94753611f190d70f93abbf5ec582426"
    )
    assert contract_sha256 == (
        "2f2a701b57f9a6002920d58f9073bb96eea128ad9c830759dc11175007c4d29f"
    )


def test_actual_local_compiler_schema_contains_closed_trigger_variants() -> None:
    schema = _local_memory_delta_schema(LOCAL_OLLAMA_MODEL).json_schema
    assert schema.properties is not None
    creates = schema.properties["intent_creates"].items
    assert creates is not None and creates.properties is not None
    trigger = creates.properties["trigger"]

    assert [
        variant.properties["type"].enum  # type: ignore[index]
        for variant in trigger.anyOf or []
    ] == [["at"], ["recurring"], ["condition_transition"]]
    assert all(variant.additionalProperties is False for variant in trigger.anyOf or [])


def test_local_compiler_w1_rules_are_frozen_and_data_blind() -> None:
    required_rules = (
        "Treat the current event as the only source of new information.",
        "Never emit facts, triggers, conditions, or action templates solely "
        "because they appear in active state.",
        "An explicit same-value reaffirmation is a valid new fact assertion even "
        "when active state already contains that value.",
        "A factual schedule, observation, possibility, brainstorming statement, "
        "or explicit instruction not to remind never creates an intent.",
        "Never replace a requested future trigger with the current event timestamp.",
        "For at or recurring, if the exact instant or range is not unambiguously "
        "resolvable, omit the mutation rather than using the event time.",
        "When no window is stated, set active_from to the current event timestamp "
        "and active_until to exactly seven calendar days later at the same local "
        "time and UTC offset.",
        "Preserve every explicit AND conjunct in required_conditions; blockers "
        "suppress when any blocker is true.",
        "Encode an explicit completed or already-done state as a blocker, not as "
        "a required synthetic negative fact.",
        "Never use spaces or uppercase characters.",
        "For a recurring per-occurrence date use {date}, never a weekday word.",
        "Omit an uncertain mutation instead of guessing.",
    )
    for rule in required_rules:
        assert LOCAL_MEMORY_COMPILER_INSTRUCTIONS.count(rule) == 1

    event = ObservableEvent(
        id="<event-id>",
        at="2000-01-01T00:00:00+00:00",
        kind="user_message",
        text="<event-text>",
    )
    active_state = '{"facts":[],"intents":[]}'
    prompt = build_local_memory_compiler_prompt(
        event=event,
        active_state=active_state,
    )

    assert prompt.startswith(f"{LOCAL_MEMORY_COMPILER_INSTRUCTIONS}\nCurrent event: ")
    assert "Current event: [<event-id>] 2000-01-01T00:00:00+00:00" in prompt
    assert f"Active compact state (canonical JSON):\n{active_state}\n" in prompt
    assert "for example" not in prompt.casefold()
    assert "e.g." not in prompt.casefold()
    for hidden_field in (
        "supersedes",
        "expected_actions",
        "acceptable_evidence_sets",
        "forbidden_actions",
        "future_events",
        "gold_labels",
    ):
        assert hidden_field not in prompt


def test_local_compiler_w1_version_and_hash_drift_preserve_schema() -> None:
    prompt_sha256 = hashlib.sha256(
        local_memory_compiler_prompt_contract().encode()
    ).hexdigest()
    schema_sha256 = hashlib.sha256(
        local_memory_compiler_schema_contract().encode()
    ).hexdigest()
    transport_sha256 = hashlib.sha256(
        local_memory_compiler_transport_contract().encode()
    ).hexdigest()

    assert LOCAL_MEMORY_COMPILER_VERSION == "local.v0.2"
    assert prompt_sha256 == (
        "4a1f6ece3a1a72e98b54f91433039b6d41ff78e766969852a5498916909d1f60"
    )
    assert prompt_sha256 != (
        "b5d910ee7a96e358ef6b1cb45f99627610aeddf9f4a212161bb8fc1f2b452821"
    )
    assert schema_sha256 == (
        "8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030"
    )
    assert transport_sha256 == (
        "d90471077b929d65737e0d098dbd0c2a12c67f6c75ff31209fca6a63782a7067"
    )


def test_local_runtime_accepts_only_exact_loopback_and_environment() -> None:
    attestation = _verify_local_ollama_runtime(
        _active_model(),
        LOCAL_OLLAMA_MODEL,
        environ=_environment(),
    )
    assert attestation.base_url == LOCAL_OLLAMA_BASE_URL
    assert attestation.no_cloud == "1"
    assert attestation.num_parallel == 1

    with pytest.raises(ValueError, match="localhost route"):
        _verify_local_ollama_runtime(
            _active_model("https://proxy.example/v1"),
            LOCAL_OLLAMA_MODEL,
            environ=_environment(),
        )
    invalid = _environment()
    invalid["OLLAMA_NO_CLOUD"] = "true"
    with pytest.raises(ValueError, match="OLLAMA_NO_CLOUD=1"):
        _verify_local_ollama_runtime(
            _active_model(),
            LOCAL_OLLAMA_MODEL,
            environ=invalid,
        )


def test_local_usage_is_complete_zero_cost_and_rejects_nonzero_cost() -> None:
    output = ModelOutput(
        model=LOCAL_OLLAMA_SERVICE_MODEL,
        completion="{}",
        usage=ModelUsage(
            input_tokens=100,
            input_tokens_cache_read=20,
            output_tokens=10,
            total_cost=None,
        ),
    )
    usage = _local_usage_from_output(output)
    assert usage.input_tokens == 120
    assert usage.cost_usd == 0.0

    output.usage.total_cost = 0.01
    with pytest.raises(ValueError, match="non-zero API cost"):
        _local_usage_from_output(output)


def test_effective_local_pricing_must_equal_the_tracked_zero_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "anamnesis.local_runtime.get_model_info",
        lambda _: ModelInfo(cost=LOCAL_ZERO_MODEL_COST),
    )
    _verify_effective_zero_model_cost(object())

    monkeypatch.setattr(
        "anamnesis.local_runtime.get_model_info",
        lambda _: None,
    )
    with pytest.raises(ValueError, match="all-zero local pricing"):
        _verify_effective_zero_model_cost(object())


def test_oracle_runtime_hash_binds_annotations_and_shared_decision_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="oracle_annotations_sha256"):
        local_system_config_sha256(
            system=ORACLE_SYSTEM_NAME,
            pricing_config_sha256="1" * 64,
        )

    first = local_system_config_sha256(
        system=ORACLE_SYSTEM_NAME,
        pricing_config_sha256="1" * 64,
        oracle_annotations_sha256="2" * 64,
    )
    second = local_system_config_sha256(
        system=ORACLE_SYSTEM_NAME,
        pricing_config_sha256="1" * 64,
        oracle_annotations_sha256="3" * 64,
    )
    monkeypatch.setattr(
        "anamnesis.local_runtime.local_decision_contract",
        lambda: "different shared decision contract",
    )
    changed_decision = local_system_config_sha256(
        system=ORACLE_SYSTEM_NAME,
        pricing_config_sha256="1" * 64,
        oracle_annotations_sha256="2" * 64,
    )

    assert first != second
    assert first != changed_decision
    assert ORACLE_COMPILER_VERSION == "oracle.v1"


def test_oracle_solver_fails_closed_without_pinned_annotations() -> None:
    with pytest.raises(ValueError, match="pinned oracle artifact"):
        local_scenario_solver(ORACLE_SYSTEM_NAME)


def test_oracle_solver_plan_binds_path_and_hash_not_full_annotations() -> None:
    path = Path("eval/oracle/smoke_memory_deltas.v1.json").resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    solver = local_scenario_solver(
        ORACLE_SYSTEM_NAME,
        oracle_annotations_path=str(path),
        oracle_annotations_sha256=digest,
    )

    params = solver.__registry_params__  # type: ignore[attr-defined]
    assert params["oracle_annotations_path"] == str(path)
    assert params["oracle_annotations_sha256"] == digest
    assert "oracle_artifact" not in params
    assert "fact_assertions" not in json.dumps(params)


def test_oracle_strategy_runner_records_zero_complete_compiler_work() -> None:
    scenarios = load_scenarios(Path("eval/scenarios/smoke.jsonl"))
    artifact = load_oracle_artifact(
        Path("eval/oracle/smoke_memory_deltas.v1.json"), scenarios
    )
    scenario = scenarios[0].to_runtime()
    compiler = OracleCompiler(artifact, scenario)

    class CountingDecisionModel:
        name = LOCAL_OLLAMA_MODEL

        def __init__(self) -> None:
            self.calls = 0

        async def decide(self, request: DecisionRequest) -> DecisionCall:
            self.calls += 1
            return DecisionCall(
                decision=Decision(),
                usage=Usage(cost_usd=0.0),
                raw_completion='{"mode":"no_action","actions":[]}',
                usage_complete=True,
                cost_complete=True,
            )

    model = CountingDecisionModel()
    run = asyncio.run(
        run_scenario(
            scenario=scenario,
            strategy=OracleAnamnesisMemoryStrategy(compiler),
            model=model,
            decision_prompt_builder=build_local_decision_prompt,
            decision_prompt_contract=local_decision_contract(),
            decision_prompt_version=LOCAL_DECISION_VERSION,
        )
    )
    compiler.assert_complete()

    assert run.system == ORACLE_SYSTEM_NAME
    assert model.calls == len(scenario.events)
    assert len(compiler.requests) == sum(
        event.kind != "clock_tick" for event in scenario.events
    )
    assert run.compiler_usage == Usage(cost_usd=0.0)
    assert run.usage.cost_usd == 0.0
    assert run.usage_complete and run.cost_complete
    for event, checkpoint in zip(scenario.events, run.checkpoints, strict=True):
        assert checkpoint.compiler_called is (event.kind != "clock_tick")
        if checkpoint.compiler_called:
            assert checkpoint.compiler_usage == Usage(cost_usd=0.0)
            assert checkpoint.raw_compiler_output is not None
    _validate_oracle_scenario_run(scenario, run)

    checkpoints = list(run.checkpoints)
    rejected_index = next(
        index
        for index, checkpoint in enumerate(checkpoints)
        if checkpoint.compiler_called
    )
    checkpoints[rejected_index] = checkpoints[rejected_index].model_copy(
        update={"memory_delta_accepted": False}
    )
    rejected = run.model_copy(update={"checkpoints": checkpoints})
    with pytest.raises(ValueError, match="memory delta was not accepted"):
        _validate_oracle_scenario_run(scenario, rejected)


def test_oracle_run_rejects_a_schema_valid_but_semantically_invalid_delta() -> None:
    scenarios = load_scenarios(Path("eval/scenarios/smoke.jsonl"))
    artifact = load_oracle_artifact(
        Path("eval/oracle/smoke_memory_deltas.v1.json"), scenarios
    )
    scenario_records = list(artifact.scenarios)
    event_records = list(scenario_records[0].events)
    event_records[0] = event_records[0].model_copy(
        update={
            "delta": MemoryDelta(mutations=(CancelIntent(intent_id="never-created"),))
        }
    )
    scenario_records[0] = scenario_records[0].model_copy(
        update={"events": tuple(event_records)}
    )
    invalid_artifact = artifact.model_copy(
        update={"scenarios": tuple(scenario_records)}
    )
    runtime_scenario = scenarios[0].to_runtime()
    compiler = OracleCompiler(invalid_artifact, runtime_scenario)

    class NoActionDecisionModel:
        name = LOCAL_OLLAMA_MODEL

        async def decide(self, request: DecisionRequest) -> DecisionCall:
            return DecisionCall(
                decision=Decision(),
                usage=Usage(cost_usd=0.0),
                cost_complete=True,
            )

    run = asyncio.run(
        run_scenario(
            scenario=runtime_scenario,
            strategy=OracleAnamnesisMemoryStrategy(compiler),
            model=NoActionDecisionModel(),
            decision_prompt_builder=build_local_decision_prompt,
            decision_prompt_contract=local_decision_contract(),
            decision_prompt_version=LOCAL_DECISION_VERSION,
        )
    )

    assert run.checkpoints[0].memory_delta_accepted is False
    with pytest.raises(ValueError, match="memory delta was not accepted"):
        _validate_oracle_scenario_run(runtime_scenario, run)


@pytest.mark.parametrize(
    "compiler_usage",
    [
        Usage(input_tokens=1, uncached_input_tokens=1, cost_usd=0.0),
        Usage(cost_usd=0.01),
    ],
)
def test_oracle_run_rejects_nonzero_compiler_tokens_or_cost(
    compiler_usage: Usage,
) -> None:
    scenarios = load_scenarios(Path("eval/scenarios/smoke.jsonl"))
    scenario = scenarios[0].to_runtime()
    checkpoint = SimpleNamespace()
    run = SimpleNamespace(
        system=ORACLE_SYSTEM_NAME,
        checkpoints=[checkpoint] * len(scenario.events),
        compiler_parse_errors=0,
        usage_complete=True,
        cost_complete=True,
        compiler_usage=compiler_usage,
        usage=Usage(cost_usd=0.0),
        decision_usage=Usage(cost_usd=0.0),
    )

    with pytest.raises(ValueError, match="compiler usage must be exactly zero"):
        _validate_oracle_scenario_run(scenario, run)  # type: ignore[arg-type]


def test_loaded_model_probe_payload_is_bound_to_the_artifact_pin() -> None:
    payload = {
        "models": [
            {
                "name": LOCAL_OLLAMA_SERVICE_MODEL,
                "model": LOCAL_OLLAMA_SERVICE_MODEL,
                "digest": LOCAL_OLLAMA_MANIFEST_SHA256,
                "size_vram": 3_169_761_361,
                "context_length": LOCAL_OLLAMA_CONTEXT_LENGTH,
                "details": {
                    "family": LOCAL_OLLAMA_FAMILY,
                    "parameter_size": LOCAL_OLLAMA_PARAMETER_SIZE,
                    "quantization_level": LOCAL_OLLAMA_QUANTIZATION,
                },
            }
        ]
    }
    assert _loaded_model_from_ps(payload) == _loaded_model()

    payload["models"][0]["digest"] = "0" * 64
    with pytest.raises(ValidationError, match="digest"):
        _loaded_model_from_ps(payload)


def test_semantic_preflight_requires_17h_intent_and_no_current_action() -> None:
    compiler_completion = json.dumps(
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

    class FakeModel:
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

        async def complete_structured(self, *, prompt, response_schema):
            return (
                ModelOutput(
                    model=LOCAL_OLLAMA_SERVICE_MODEL,
                    completion=compiler_completion,
                    usage=_model_usage(474, 155),
                ),
                15.0,
            )

        async def decide(self, request: DecisionRequest) -> DecisionCall:
            return DecisionCall(
                decision=Decision(),
                usage=Usage(
                    input_tokens=534,
                    uncached_input_tokens=534,
                    output_tokens=11,
                    cost_usd=0.0,
                ),
                latency_ms=7.0,
                raw_completion='{"mode":"no_action","actions":[]}',
                usage_complete=True,
                cost_complete=True,
            )

    result = asyncio.run(
        run_local_model_preflight(  # type: ignore[arg-type]
            FakeModel(),
            residency_probe=_loaded_model,
        )
    )

    assert result.passed
    assert result.compiler_semantic_valid
    assert result.decision_semantic_valid
    assert result.compiler_usage.cost_usd == 0.0
    assert result.loaded_model == _loaded_model()


def test_w2_preflight_sends_only_ordered_inputs_and_accepts_all_four_cases() -> None:
    fixture = load_local_w2_preflight_fixture("eval/preflight/local_writer_w2.v1.json")
    compiler_cases = fixture["compiler_cases"]
    assert isinstance(compiler_cases, list)
    completions = [json.dumps(case["valid_wire_example"]) for case in compiler_cases]

    class FakeW2Model:
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

        async def complete_structured(self, *, prompt, response_schema):
            index = len(self.compiler_prompts)
            self.compiler_prompts.append(prompt)
            return (
                ModelOutput(
                    model=LOCAL_OLLAMA_SERVICE_MODEL,
                    completion=completions[index],
                    usage=_model_usage(100 + index, 10),
                ),
                2.0,
            )

        async def decide(self, request: DecisionRequest) -> DecisionCall:
            return DecisionCall(
                decision=Decision(),
                usage=Usage(
                    input_tokens=120,
                    uncached_input_tokens=120,
                    output_tokens=4,
                    cost_usd=0.0,
                ),
                latency_ms=3.0,
                raw_completion='{"mode":"no_action","actions":[]}',
                usage_complete=True,
                cost_complete=True,
            )

    model = FakeW2Model()
    result = asyncio.run(
        run_local_model_preflight_w2(  # type: ignore[arg-type]
            model,
            fixture=fixture,
            residency_probe=_loaded_model,
        )
    )

    assert isinstance(result, LocalModelPreflightW2Result)
    assert result.passed
    assert [case.case_id for case in result.cases] == ["C1", "C2", "C3", "D1"]
    assert all(case.semantic_valid for case in result.cases)
    assert len(model.compiler_prompts) == 3
    for prompt, case in zip(model.compiler_prompts, compiler_cases, strict=True):
        assert prompt == build_local_memory_compiler_w2_prompt(
            event=ObservableEvent.model_validate(case["input"]["event"]),
            active_state=case["input"]["active_state"],
        )
        assert str(case["category"]) not in prompt
        assert "valid_wire_example" not in prompt
        assert "acceptance" not in prompt


def test_w2_contract_and_system_hash_are_distinct_without_changing_w1() -> None:
    w1_transport = local_memory_compiler_transport_contract()
    w2_transport = local_memory_compiler_w2_transport_contract()

    assert local_memory_compiler_prompt_contract() != (
        local_memory_compiler_w2_prompt_contract()
    )
    assert w1_transport != w2_transport
    w1_hash = local_system_config_sha256(
        system="anamnesis",
        pricing_config_sha256="1" * 64,
    )
    w2_hash = local_system_config_sha256(
        system="anamnesis",
        pricing_config_sha256="1" * 64,
        compiler_prompt_variant="w2",
    )
    assert w1_hash != w2_hash
    with pytest.raises(ValueError, match="requires system=anamnesis"):
        local_system_config_sha256(
            system="full_context",
            compiler_prompt_variant="w2",
        )


def test_runner_hashes_and_labels_the_actual_local_prompt() -> None:
    scenario = RuntimeScenario(
        id="local-prompt-audit",
        events=[
            ObservableEvent(
                id="event-1",
                at="2026-01-05T09:00:00+00:00",
                kind="clock_tick",
                text="Current simulated time is 09:00.",
            )
        ],
    )

    class CapturingModel:
        name = LOCAL_OLLAMA_MODEL

        def __init__(self) -> None:
            self.prompt = ""

        async def decide(self, request: DecisionRequest) -> DecisionCall:
            self.prompt = request.prompt
            return DecisionCall(
                decision=Decision(),
                usage=Usage(
                    input_tokens=1,
                    uncached_input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.0,
                ),
                cost_complete=True,
            )

    model = CapturingModel()
    contract = local_decision_contract()
    run = asyncio.run(
        run_scenario(
            scenario=scenario,
            strategy=NoPersistentMemory(),
            model=model,
            decision_prompt_builder=build_local_decision_prompt,
            decision_prompt_contract=contract,
            decision_prompt_version=LOCAL_DECISION_VERSION,
        )
    )

    assert run.prompt_version == LOCAL_DECISION_VERSION
    assert run.prompt_sha256 == hashlib.sha256(contract.encode()).hexdigest()
    assert (
        run.checkpoints[0].rendered_context_sha256
        == hashlib.sha256(model.prompt.encode()).hexdigest()
    )
    assert '"mode":"no_action"' in model.prompt
