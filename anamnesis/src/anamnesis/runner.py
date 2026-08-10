"""Provider-neutral execution of a memory strategy over a scenario timeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from anamnesis.baselines import MemoryStrategy, RetrievalUsage, StrategyWork
from anamnesis.io import canonical_sha256
from anamnesis.prompts import PROMPT_VERSION, build_decision_prompt, prompt_contract
from anamnesis.runtime_contract import anamnesis_runtime_contract
from anamnesis.schema import (
    CheckpointAudit,
    Decision,
    HostedWarmupAttestation,
    ObservableEvent,
    PredictedAction,
    RuntimeScenario,
    Scenario,
    ScenarioRun,
    Usage,
)


@dataclass(frozen=True)
class DecisionRequest:
    prompt: str
    event: ObservableEvent


@dataclass(frozen=True)
class DecisionCall:
    decision: Decision
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    parse_error: bool = False
    raw_completion: str | None = None
    usage_complete: bool = True
    cost_complete: bool = False


class DecisionModel(Protocol):
    """An LLM adapter used identically by every memory strategy."""

    name: str

    async def decide(self, request: DecisionRequest) -> DecisionCall: ...


async def run_scenario(
    *,
    scenario: Scenario | RuntimeScenario,
    strategy: MemoryStrategy,
    model: DecisionModel,
    repetition: int = 1,
    seed: int | None = None,
    scenario_sha256_override: str | None = None,
    system_config_sha256: str | None = None,
    manifest_sha256: str | None = None,
    pricing_config_sha256: str | None = None,
    setup_latency_ms: float = 0.0,
    hosted_warmup: HostedWarmupAttestation | None = None,
    decision_prompt_builder: Callable[..., str] = build_decision_prompt,
    decision_prompt_contract: str | None = None,
    decision_prompt_version: str = PROMPT_VERSION,
) -> ScenarioRun:
    """Advance the logical clock and record every emitted action."""

    strategy.reset()
    predictions: list[PredictedAction] = []
    decision_usage = Usage()
    compiler_usage = Usage()
    retrieval_usage = RetrievalUsage()
    decision_latency_ms = 0.0
    compiler_latency_ms = 0.0
    decision_parse_errors = 0
    compiler_parse_errors = 0
    usage_complete = True
    cost_complete = True
    checkpoint_latency_ms: list[float] = []
    checkpoint_audits: list[CheckpointAudit] = []

    for authored_event in scenario.events:
        checkpoint_started = perf_counter()
        # This constructor is the structural anti-leakage boundary. Strategies
        # cannot receive supersedes, tags, hidden gold, or future events.
        event = (
            authored_event.to_observable()
            if not isinstance(authored_event, ObservableEvent)
            else ObservableEvent.model_validate(authored_event.model_dump())
        )
        ingest_work = await strategy.ingest(event)
        selection = strategy.select(event)
        prompt_kwargs: dict[str, object] = {
            "now": event.at.isoformat(),
            "current_event_id": event.id,
            "context_events": selection.events,
            "decision_history": selection.decisions,
            "memory_view": selection.memory_view,
        }
        if selection.retrospective_recall is not None:
            prompt_kwargs["retrospective_recall"] = selection.retrospective_recall
        request = DecisionRequest(
            event=event,
            prompt=decision_prompt_builder(**prompt_kwargs),
        )
        call = await model.decide(request)
        raw_completion = call.raw_completion
        if raw_completion is None:
            raw_completion = call.decision.model_dump_json()
        commit_work = strategy.commit(event, call.decision)
        selection_work = StrategyWork(local_usage=selection.usage)
        checkpoint_work = ingest_work.plus(selection_work).plus(commit_work)
        checkpoint_retrieval = checkpoint_work.local_usage
        retrieval_usage = retrieval_usage.plus(checkpoint_retrieval)
        decision_usage = decision_usage.plus(call.usage)
        compiler_usage = compiler_usage.plus(checkpoint_work.compiler_usage)
        decision_latency_ms += call.latency_ms
        compiler_latency_ms += checkpoint_work.compiler_latency_ms
        decision_parse_errors += int(call.parse_error)
        compiler_parse_errors += int(checkpoint_work.compiler_parse_error)
        usage_complete = (
            usage_complete and call.usage_complete and checkpoint_work.usage_complete
        )
        cost_complete = (
            cost_complete and call.cost_complete and checkpoint_work.cost_complete
        )
        predictions.extend(
            PredictedAction(
                **action.model_dump(),
                emitted_at=event.at,
                decision_event_id=event.id,
            )
            for action in call.decision.actions
        )
        checkpoint_elapsed_ms = (perf_counter() - checkpoint_started) * 1000
        checkpoint_latency_ms.append(checkpoint_elapsed_ms)
        checkpoint_audits.append(
            CheckpointAudit(
                event_id=event.id,
                at=event.at,
                compiler_called=checkpoint_work.compiler_called,
                raw_compiler_output=checkpoint_work.raw_compiler_output,
                memory_delta_json=checkpoint_work.memory_delta_json,
                memory_delta_accepted=checkpoint_work.memory_delta_accepted,
                memory_delta_error=checkpoint_work.memory_delta_error,
                state_sha256=(checkpoint_work.state_sha256 or selection.state_sha256),
                due_candidate_ids=(
                    checkpoint_work.due_candidate_ids or selection.due_candidate_ids
                ),
                rendered_context_sha256=hashlib.sha256(
                    request.prompt.encode()
                ).hexdigest(),
                raw_decision_output=raw_completion,
                compiler_usage=checkpoint_work.compiler_usage,
                decision_usage=call.usage,
                compiler_latency_ms=checkpoint_work.compiler_latency_ms,
                decision_latency_ms=call.latency_ms,
                local_latency_ms=checkpoint_work.local_usage.latency_ms,
                compiler_parse_error=checkpoint_work.compiler_parse_error,
                decision_parse_error=call.parse_error,
            )
        )

    total_usage = decision_usage.plus(compiler_usage).plus(
        Usage(
            embedding_inputs=retrieval_usage.embedding_inputs,
            embedding_characters=retrieval_usage.embedding_characters,
        )
    )
    if not cost_complete:
        total_usage = total_usage.model_copy(update={"cost_usd": None})
    active_prompt_contract = decision_prompt_contract or prompt_contract()
    if system_config_sha256 is None:
        fallback_config: dict[str, object] = {
            "model": model.name,
            "pricing_config_sha256": pricing_config_sha256,
            "prompt_sha256": hashlib.sha256(
                active_prompt_contract.encode()
            ).hexdigest(),
            "system": strategy.name,
        }
        if strategy.name == "anamnesis":
            fallback_config["deterministic_memory"] = anamnesis_runtime_contract()
        strategy_contract = getattr(strategy, "strategy_contract", None)
        if callable(strategy_contract):
            fallback_config["strategy_contract"] = strategy_contract()
        serialized_config = json.dumps(
            fallback_config,
            sort_keys=True,
            separators=(",", ":"),
        )
        system_config_sha256 = hashlib.sha256(serialized_config.encode()).hexdigest()
    return ScenarioRun(
        scenario_id=scenario.id,
        system=strategy.name,
        repetition=repetition,
        model=model.name,
        prompt_version=decision_prompt_version,
        scenario_sha256=scenario_sha256_override or canonical_sha256(scenario),
        prompt_sha256=hashlib.sha256(active_prompt_contract.encode()).hexdigest(),
        system_config_sha256=system_config_sha256,
        manifest_sha256=manifest_sha256,
        pricing_config_sha256=pricing_config_sha256,
        seed=seed,
        predictions=predictions,
        usage=total_usage,
        decision_usage=decision_usage,
        compiler_usage=compiler_usage,
        usage_complete=usage_complete,
        cost_complete=cost_complete,
        decision_latency_ms=decision_latency_ms,
        compiler_latency_ms=compiler_latency_ms,
        local_latency_ms=retrieval_usage.latency_ms,
        setup_latency_ms=setup_latency_ms,
        hosted_warmup=hosted_warmup,
        checkpoint_latency_ms=checkpoint_latency_ms,
        decision_parse_errors=decision_parse_errors,
        compiler_parse_errors=compiler_parse_errors,
        parse_errors=decision_parse_errors + compiler_parse_errors,
        checkpoints=checkpoint_audits,
    )
