from __future__ import annotations

import asyncio

import numpy as np
import pytest
from pydantic import ValidationError

from anamnesis.baselines import (
    AnamnesisMemoryStrategy,
    FullContextMemory,
    NoPersistentMemory,
    VectorRAGMemory,
)
from anamnesis.io import canonical_sha256, load_scenarios
from anamnesis.memory import (
    ActionTemplate,
    AtTrigger,
    CompilerCall,
    CompilerRequest,
    Condition,
    CreateIntent,
    DeterministicCompiler,
    FactKey,
    MemoryDelta,
    SetFact,
    UpdateIntent,
)
from anamnesis.runner import DecisionCall, DecisionRequest, run_scenario
from anamnesis.schema import Decision, ProposedAction, Usage
from anamnesis.scoring import score_scenario


class RecordingModel:
    name = "fake/model"

    def __init__(self, action_event_id: str | None = None) -> None:
        self.action_event_id = action_event_id
        self.requests: list[DecisionRequest] = []

    async def decide(self, request: DecisionRequest) -> DecisionCall:
        self.requests.append(request)
        actions = []
        if request.event.id == self.action_event_id:
            actions = [
                ProposedAction(
                    action_key="s01-e01",
                    payload={"subject": "send statistics assignment"},
                    summary="Send the statistics assignment",
                    evidence_event_ids=["s01-e01", "s01-e05", "s01-e07"],
                )
            ]
        return DecisionCall(
            decision=Decision(actions=actions),
            usage=Usage(
                input_tokens=10,
                uncached_input_tokens=10,
                output_tokens=2,
                cost_usd=0.001,
            ),
            latency_ms=3.0,
            cost_complete=True,
        )


def scenario_one():
    return load_scenarios("eval/scenarios/smoke.jsonl")[0]


def test_observable_event_cannot_be_mutated_after_ingest_boundary() -> None:
    event = scenario_one().events[0].to_observable()

    with pytest.raises(ValidationError, match="frozen"):
        event.text = "replace the observed event"  # type: ignore[misc]


def test_runner_calls_model_once_per_event_and_accumulates_usage() -> None:
    scenario = scenario_one()
    model = RecordingModel(action_event_id="s01-e07")
    run = asyncio.run(
        run_scenario(
            scenario=scenario,
            strategy=FullContextMemory(),
            model=model,
            repetition=2,
            seed=17,
        )
    )

    assert len(model.requests) == len(scenario.events)
    assert run.scenario_sha256 == canonical_sha256(scenario)
    assert run.repetition == 2
    assert run.seed == 17
    assert len(run.predictions) == 1
    assert run.predictions[0].emitted_at == scenario.events[6].at
    assert run.usage.input_tokens == len(scenario.events) * 10
    assert run.usage.output_tokens == len(scenario.events) * 2
    assert run.usage.cost_usd == len(scenario.events) * 0.001
    assert run.decision_latency_ms == len(scenario.events) * 3.0
    assert len(run.checkpoint_latency_ms) == len(scenario.events)


def test_no_memory_prompt_never_contains_prior_or_future_events() -> None:
    scenario = scenario_one()
    model = RecordingModel()
    asyncio.run(
        run_scenario(
            scenario=scenario,
            strategy=NoPersistentMemory(),
            model=model,
        )
    )

    for request, event in zip(model.requests, scenario.events, strict=True):
        assert event.id in request.prompt
        for other in scenario.events:
            if other.id != event.id:
                assert f"[{other.id}]" not in request.prompt


def test_full_context_contains_typed_prior_decisions() -> None:
    scenario = scenario_one()
    model = RecordingModel(action_event_id="s01-e07")
    asyncio.run(
        run_scenario(
            scenario=scenario,
            strategy=FullContextMemory(),
            model=model,
        )
    )

    assert "[decision:s01-e01]" in model.requests[1].prompt
    assert '"actions":[]' in model.requests[1].prompt
    assert "[decision:s01-e07]" in model.requests[7].prompt
    assert "statistics assignment" in model.requests[7].prompt


def test_parse_errors_are_recorded_without_synthetic_actions() -> None:
    class InvalidModel(RecordingModel):
        async def decide(self, request: DecisionRequest) -> DecisionCall:
            return DecisionCall(decision=Decision(), parse_error=True)

    scenario = scenario_one()
    run = asyncio.run(
        run_scenario(
            scenario=scenario,
            strategy=NoPersistentMemory(),
            model=InvalidModel(),
        )
    )

    assert run.parse_errors == len(scenario.events)
    assert run.predictions == []


def test_vector_runner_meters_event_decision_and_query_embeddings() -> None:
    class ConstantVectorizer:
        def embed_documents(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 3), dtype=np.float32)

        def embed_query(self, text: str) -> np.ndarray:
            return np.ones(3, dtype=np.float32)

    scenario = scenario_one()
    run = asyncio.run(
        run_scenario(
            scenario=scenario,
            strategy=VectorRAGMemory(ConstantVectorizer(), top_k=5),
            model=RecordingModel(),
        )
    )

    authored_documents = sum(event.kind != "clock_tick" for event in scenario.events)
    decision_documents = len(scenario.events)
    retrieval_queries = len(scenario.events) - 1
    assert run.usage.embedding_inputs == (
        authored_documents + decision_documents + retrieval_queries
    )
    assert run.usage.embedding_characters > 0


def test_anamnesis_compiles_only_non_clock_events_and_accounts_all_calls() -> None:
    class MeteredCompiler:
        name = "fake/model"

        def __init__(self) -> None:
            self.requests: list[CompilerRequest] = []

        async def compile(self, request: CompilerRequest) -> CompilerCall:
            self.requests.append(request)
            delta = MemoryDelta()
            return CompilerCall(
                delta=delta,
                usage=Usage(
                    input_tokens=5,
                    uncached_input_tokens=5,
                    output_tokens=1,
                    cost_usd=0.0002,
                ),
                latency_ms=2.0,
                raw_completion=delta.model_dump_json(),
                cost_complete=True,
            )

    scenario = scenario_one()
    compiler = MeteredCompiler()
    model = RecordingModel()
    run = asyncio.run(
        run_scenario(
            scenario=scenario,
            strategy=AnamnesisMemoryStrategy(compiler),
            model=model,
        )
    )

    compiler_events = [event for event in scenario.events if event.kind != "clock_tick"]
    assert [request.event.id for request in compiler.requests] == [
        event.id for event in compiler_events
    ]
    assert len(model.requests) == len(scenario.events)
    assert run.compiler_usage.input_tokens == len(compiler_events) * 5
    assert run.decision_usage.input_tokens == len(scenario.events) * 10
    assert run.usage.input_tokens == (
        run.compiler_usage.input_tokens + run.decision_usage.input_tokens
    )
    assert run.usage.cost_usd == pytest.approx(
        len(compiler_events) * 0.0002 + len(scenario.events) * 0.001
    )
    assert run.compiler_latency_ms == len(compiler_events) * 2.0
    assert run.cost_complete
    assert len(run.checkpoints) == len(scenario.events)
    assert [audit.compiler_called for audit in run.checkpoints] == [
        event.kind != "clock_tick" for event in scenario.events
    ]
    assert all(audit.state_sha256 for audit in run.checkpoints)
    assert "Structured memory view:" in model.requests[0].prompt


def test_runner_records_rejected_memory_delta_reason() -> None:
    scenario = scenario_one()
    compiler = DeterministicCompiler(
        {
            "s01-e01": MemoryDelta(
                mutations=(
                    UpdateIntent(
                        intent_id="missing",
                        action_template=ActionTemplate(
                            payload={"subject": "check missing intent"},
                            summary="Check missing intent",
                        ),
                    ),
                )
            )
        }
    )

    run = asyncio.run(
        run_scenario(
            scenario=scenario,
            strategy=AnamnesisMemoryStrategy(compiler),
            model=RecordingModel(),
        )
    )

    first = run.checkpoints[0]
    assert first.memory_delta_accepted is False
    assert (
        first.memory_delta_error == "cannot update missing or inactive intent: missing"
    )


def test_anamnesis_end_to_end_with_deterministic_fake_compiler() -> None:
    scenario = scenario_one()
    sent = FactKey(entity="statistics_assignment", attribute="sent")
    compiler = DeterministicCompiler(
        {
            "s01-e01": MemoryDelta(
                mutations=(
                    CreateIntent(
                        intent_id="statistics_assignment_reminder",
                        trigger=AtTrigger(at=scenario.events[6].at),
                        blockers=(Condition(key=sent, operator="eq", value=True),),
                        action_template=ActionTemplate(
                            payload={"subject": "send statistics assignment"},
                            summary="Send the statistics assignment",
                        ),
                    ),
                )
            ),
            "s01-e05": MemoryDelta(mutations=(SetFact(key=sent, value=False),)),
        }
    )
    strategy = AnamnesisMemoryStrategy(compiler)
    model = RecordingModel(action_event_id="s01-e07")
    run = asyncio.run(
        run_scenario(
            scenario=scenario,
            strategy=strategy,
            model=model,
        )
    )

    score = score_scenario(scenario, run)
    assert (score.tp, score.fp, score.fn) == (1, 0, 0)
    assert score.provenance_exact == 1
    assert len(strategy.memory.executions) == 1
    assert strategy.memory.executions[0].action_key == "s01-e01"
    due_prompt = model.requests[6].prompt
    assert "DUE_CANDIDATE" in due_prompt
    assert '"action_key":"s01-e01"' in due_prompt
    assert 'evidence=["s01-e01","s01-e05","s01-e07"]' in due_prompt
