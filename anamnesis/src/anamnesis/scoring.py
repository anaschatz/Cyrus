"""Deterministic, action-level scoring for prospective-memory evaluations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import inf
from statistics import median

import numpy as np
from pydantic import Field

from anamnesis.schema import (
    ExpectedAction,
    ForbiddenAction,
    PredictedAction,
    Scenario,
    ScenarioRun,
    StrictModel,
)


class ScenarioScore(StrictModel):
    """Raw recomputable counts for one scenario run."""

    scenario_id: str
    system: str
    repetition: int
    model: str
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    false_reminders: int = Field(ge=0)
    negative_checkpoints: int = Field(ge=0)
    false_alarm_checkpoints: int = Field(ge=0)
    obsolete_errors: int = Field(ge=0)
    obsolete_traps: int = Field(ge=0)
    obsolete_traps_triggered: int = Field(ge=0)
    duplicate_errors: int = Field(ge=0)
    provenance_exact: int = Field(ge=0)
    provenance_covered: int = Field(ge=0)
    invalid_outputs: int = Field(ge=0)
    matched_pairs: list[tuple[str, int]] = Field(default_factory=list)
    unmatched_prediction_indices: list[int] = Field(default_factory=list)
    unmatched_expected_ids: list[str] = Field(default_factory=list)


class AggregateResult(StrictModel):
    """Headline micro metrics for one system and repetition."""

    system: str
    repetition: int
    model: str
    scenarios: int = Field(ge=1)
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    false_reminders: int = Field(ge=0)
    negative_checkpoints: int = Field(ge=0)
    false_alarm_checkpoints: int = Field(ge=0)
    false_alarm_rate: float | None = Field(default=None, ge=0, le=1)
    obsolete_errors: int = Field(ge=0)
    obsolete_trap_rate: float | None = Field(default=None, ge=0, le=1)
    provenance_exact_accuracy: float | None = Field(default=None, ge=0, le=1)
    invalid_outputs: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    decision_input_tokens: int = Field(ge=0)
    compiler_input_tokens: int = Field(ge=0)
    embedding_inputs: int = Field(ge=0)
    embedding_characters: int = Field(ge=0)
    input_token_reduction_vs_full_context: float | None = None
    cost_usd: float | None = Field(default=None, ge=0)
    usage_complete: bool
    cost_complete: bool
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    decision_latency_ms: float = Field(ge=0)
    compiler_latency_ms: float = Field(ge=0)
    local_latency_ms: float = Field(ge=0)
    setup_latency_ms: float = Field(ge=0)


class SuccessGateResult(StrictModel):
    """Preregistered Anamnesis success checks for one model/repetition."""

    repetition: int = Field(ge=1)
    model: str
    comparator: str
    f1_gain: float
    input_token_reduction: float
    anamnesis_false_alarm_checkpoints: int = Field(ge=0)
    comparator_false_alarm_checkpoints: int = Field(ge=0)
    f1_pass: bool
    token_pass: bool
    false_alarm_pass: bool
    supported: bool


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int


def _add_edge(
    graph: list[list[_FlowEdge]], source: int, target: int, capacity: int, cost: int
) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity, cost)
    backward = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(backward)
    return forward


def _eligible(prediction: PredictedAction, expected: ExpectedAction) -> bool:
    return (
        prediction.kind == expected.kind
        and prediction.action_key == expected.action_key
        and _normalized_payload(prediction.payload)
        == _normalized_payload(expected.payload)
        and expected.window_start <= prediction.emitted_at <= expected.window_end
    )


def _normalized_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: " ".join(value.casefold().split()) if isinstance(value, str) else value
        for key, value in payload.items()
    }


def _minimum_cost_maximum_matching(
    predictions: list[PredictedAction], expected_actions: list[ExpectedAction]
) -> list[tuple[int, int]]:
    """Return (expected index, prediction index) pairs with deterministic ties."""

    gold_count = len(expected_actions)
    prediction_count = len(predictions)
    source = 0
    first_gold = 1
    first_prediction = first_gold + gold_count
    sink = first_prediction + prediction_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]

    for gold_index in range(gold_count):
        _add_edge(graph, source, first_gold + gold_index, 1, 0)
    for prediction_index in range(prediction_count):
        _add_edge(graph, first_prediction + prediction_index, sink, 1, 0)

    candidate_edges: dict[tuple[int, int], _FlowEdge] = {}
    gold_ranks = {
        gold_index: rank
        for rank, gold_index in enumerate(
            sorted(
                range(gold_count),
                key=lambda index: expected_actions[index].id,
            )
        )
    }
    max_matches = min(gold_count, prediction_count)
    max_tie_per_edge = max(0, gold_count * max(1, prediction_count) - 1)
    tie_span = max(1, max_matches * max_tie_per_edge + 1)
    for gold_index, expected in enumerate(expected_actions):
        for prediction_index, prediction in enumerate(predictions):
            if not _eligible(prediction, expected):
                continue
            distance_ms = round(
                abs((prediction.emitted_at - expected.window_start).total_seconds())
                * 1000
            )
            tie_break = (
                gold_ranks[gold_index] * max(1, prediction_count) + prediction_index
            )
            edge = _add_edge(
                graph,
                first_gold + gold_index,
                first_prediction + prediction_index,
                1,
                distance_ms * tie_span + tie_break,
            )
            candidate_edges[(gold_index, prediction_index)] = edge

    while True:
        distances = [inf] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = 0

        for _ in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distances[node] == inf:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate = distances[node] + edge.cost
                    if candidate < distances[edge.to]:
                        distances[edge.to] = candidate
                        previous[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break

        if previous[sink] is None:
            break
        node = sink
        while node != source:
            parent, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent

    return sorted(pair for pair, edge in candidate_edges.items() if edge.capacity == 0)


def _matches_forbidden(prediction: PredictedAction, forbidden: ForbiddenAction) -> bool:
    return (
        prediction.kind == forbidden.kind
        and prediction.action_key == forbidden.action_key
        and _normalized_payload(prediction.payload)
        == _normalized_payload(forbidden.payload)
        and forbidden.window_start <= prediction.emitted_at <= forbidden.window_end
    )


def _provenance_is_exact(
    *,
    prediction: PredictedAction,
    expected: ExpectedAction,
    scenario: Scenario,
) -> bool:
    if len(prediction.evidence_event_ids) != len(set(prediction.evidence_event_ids)):
        return False
    cited = set(prediction.evidence_event_ids)
    visible_ids = {
        event.id for event in scenario.events if event.at <= prediction.emitted_at
    }
    if not cited <= visible_ids:
        return False
    return any(
        cited == set(acceptable) for acceptable in expected.acceptable_evidence_sets
    )


def score_scenario(scenario: Scenario, run: ScenarioRun) -> ScenarioScore:
    """Score executed actions; malformed decisions are recorded but never executed."""

    if run.scenario_id != scenario.id:
        raise ValueError("scenario and run IDs differ")
    events_by_id = {event.id: event for event in scenario.events}
    for prediction in run.predictions:
        decision_event = events_by_id.get(prediction.decision_event_id)
        if decision_event is None:
            raise ValueError(
                "prediction references an unknown decision_event_id: "
                f"{prediction.decision_event_id}"
            )
        if prediction.emitted_at != decision_event.at:
            raise ValueError(
                "prediction emitted_at differs from its decision event timestamp"
            )

    matched = _minimum_cost_maximum_matching(run.predictions, scenario.expected_actions)
    matched_expected = {gold_index for gold_index, _ in matched}
    matched_predictions = {prediction_index for _, prediction_index in matched}
    unmatched_predictions = [
        index
        for index in range(len(run.predictions))
        if index not in matched_predictions
    ]
    unmatched_expected = [
        index
        for index in range(len(scenario.expected_actions))
        if index not in matched_expected
    ]

    obsolete_trap_ids = {
        forbidden.id
        for forbidden in scenario.forbidden_actions
        if forbidden.reason == "obsolete"
    }
    triggered_obsolete_traps: set[str] = set()
    obsolete_errors = 0
    duplicate_errors = 0
    for prediction_index in unmatched_predictions:
        prediction = run.predictions[prediction_index]
        obsolete_matches = [
            forbidden
            for forbidden in scenario.forbidden_actions
            if forbidden.reason == "obsolete"
            and _matches_forbidden(prediction, forbidden)
        ]
        if obsolete_matches:
            obsolete_errors += 1
            triggered_obsolete_traps.update(item.id for item in obsolete_matches)
            continue
        if any(
            forbidden.reason == "duplicate"
            and _matches_forbidden(prediction, forbidden)
            for forbidden in scenario.forbidden_actions
        ):
            duplicate_errors += 1
            continue
        if any(
            prediction.kind == expected.kind
            and prediction.action_key == expected.action_key
            and _normalized_payload(prediction.payload)
            == _normalized_payload(expected.payload)
            and expected.window_start <= prediction.emitted_at <= expected.window_end
            and gold_index in matched_expected
            for gold_index, expected in enumerate(scenario.expected_actions)
        ):
            duplicate_errors += 1

    positive_checkpoint_ids = {
        event.id
        for event in scenario.events
        if any(
            expected.window_start <= event.at <= expected.window_end
            for expected in scenario.expected_actions
        )
    }
    negative_checkpoint_ids = {
        event.id for event in scenario.events if event.id not in positive_checkpoint_ids
    }
    false_alarm_checkpoints = {
        run.predictions[index].decision_event_id
        for index in unmatched_predictions
        if run.predictions[index].decision_event_id in negative_checkpoint_ids
    }

    provenance_exact = 0
    provenance_covered = 0
    matched_pairs: list[tuple[str, int]] = []
    for gold_index, prediction_index in matched:
        expected = scenario.expected_actions[gold_index]
        prediction = run.predictions[prediction_index]
        matched_pairs.append((expected.id, prediction_index))
        provenance_covered += int(bool(prediction.evidence_event_ids))
        provenance_exact += int(
            _provenance_is_exact(
                prediction=prediction,
                expected=expected,
                scenario=scenario,
            )
        )

    return ScenarioScore(
        scenario_id=scenario.id,
        system=run.system,
        repetition=run.repetition,
        model=run.model,
        tp=len(matched),
        fp=len(unmatched_predictions),
        fn=len(unmatched_expected),
        false_reminders=len(unmatched_predictions),
        negative_checkpoints=len(negative_checkpoint_ids),
        false_alarm_checkpoints=len(false_alarm_checkpoints),
        obsolete_errors=obsolete_errors,
        obsolete_traps=len(obsolete_trap_ids),
        obsolete_traps_triggered=len(triggered_obsolete_traps),
        duplicate_errors=duplicate_errors,
        provenance_exact=provenance_exact,
        provenance_covered=provenance_covered,
        invalid_outputs=run.parse_errors,
        matched_pairs=matched_pairs,
        unmatched_prediction_indices=unmatched_predictions,
        unmatched_expected_ids=[
            scenario.expected_actions[index].id for index in unmatched_expected
        ],
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_results(
    scored_runs: Iterable[tuple[ScenarioScore, ScenarioRun]],
) -> list[AggregateResult]:
    """Compute action-level micro metrics grouped by system and repetition."""

    grouped: dict[tuple[str, int, str], list[tuple[ScenarioScore, ScenarioRun]]] = (
        defaultdict(list)
    )
    for score, run in scored_runs:
        grouped[(score.system, score.repetition, score.model)].append((score, run))

    results: list[AggregateResult] = []
    for (system, repetition, model), items in sorted(grouped.items()):
        scores = [item[0] for item in items]
        runs = [item[1] for item in items]
        tp = sum(score.tp for score in scores)
        fp = sum(score.fp for score in scores)
        fn = sum(score.fn for score in scores)
        precision = _safe_ratio(tp, tp + fp) or 0.0
        recall = _safe_ratio(tp, tp + fn) or 0.0
        f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn) or 0.0
        negative_checkpoints = sum(score.negative_checkpoints for score in scores)
        obsolete_traps = sum(score.obsolete_traps for score in scores)
        provenance_total = tp
        checkpoint_latencies = [
            latency for run in runs for latency in run.checkpoint_latency_ms
        ]
        if checkpoint_latencies:
            p50 = float(median(checkpoint_latencies))
            p95 = float(np.percentile(checkpoint_latencies, 95))
        else:
            p50 = 0.0
            p95 = 0.0
        costs = [run.usage.cost_usd for run in runs]
        cost = None if any(value is None for value in costs) else sum(costs)  # type: ignore[arg-type]

        results.append(
            AggregateResult(
                system=system,
                repetition=repetition,
                model=model,
                scenarios=len(items),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                false_reminders=sum(score.false_reminders for score in scores),
                negative_checkpoints=negative_checkpoints,
                false_alarm_checkpoints=sum(
                    score.false_alarm_checkpoints for score in scores
                ),
                false_alarm_rate=_safe_ratio(
                    sum(score.false_alarm_checkpoints for score in scores),
                    negative_checkpoints,
                ),
                obsolete_errors=sum(score.obsolete_errors for score in scores),
                obsolete_trap_rate=_safe_ratio(
                    sum(score.obsolete_traps_triggered for score in scores),
                    obsolete_traps,
                ),
                provenance_exact_accuracy=_safe_ratio(
                    sum(score.provenance_exact for score in scores),
                    provenance_total,
                ),
                invalid_outputs=sum(score.invalid_outputs for score in scores),
                input_tokens=sum(run.usage.input_tokens for run in runs),
                output_tokens=sum(run.usage.output_tokens for run in runs),
                decision_input_tokens=sum(
                    run.decision_usage.input_tokens for run in runs
                ),
                compiler_input_tokens=sum(
                    run.compiler_usage.input_tokens for run in runs
                ),
                embedding_inputs=sum(run.usage.embedding_inputs for run in runs),
                embedding_characters=sum(
                    run.usage.embedding_characters for run in runs
                ),
                cost_usd=cost,
                usage_complete=all(run.usage_complete for run in runs),
                cost_complete=all(run.cost_complete for run in runs),
                latency_p50_ms=p50,
                latency_p95_ms=p95,
                total_latency_ms=sum(sum(run.checkpoint_latency_ms) for run in runs),
                decision_latency_ms=sum(run.decision_latency_ms for run in runs),
                compiler_latency_ms=sum(run.compiler_latency_ms for run in runs),
                local_latency_ms=sum(run.local_latency_ms for run in runs),
                setup_latency_ms=sum(run.setup_latency_ms for run in runs),
            )
        )
    full_context_tokens = {
        (result.repetition, result.model): result.input_tokens
        for result in results
        if result.system == "full_context" and result.input_tokens > 0
    }
    return [
        result.model_copy(
            update={
                "input_token_reduction_vs_full_context": (
                    1
                    - result.input_tokens
                    / full_context_tokens[(result.repetition, result.model)]
                    if (result.repetition, result.model) in full_context_tokens
                    else None
                )
            }
        )
        for result in results
    ]


def evaluate_success_gate(results: list[AggregateResult]) -> list[SuccessGateResult]:
    """Apply the fixed success rule independently to every repetition."""

    grouped: dict[tuple[int, str], list[AggregateResult]] = defaultdict(list)
    for result in results:
        grouped[(result.repetition, result.model)].append(result)

    gates: list[SuccessGateResult] = []
    simple_systems = {"no_memory", "full_context", "vector_rag"}
    for (repetition, model), group in sorted(grouped.items()):
        by_system = {result.system: result for result in group}
        missing = (simple_systems | {"anamnesis"}) - set(by_system)
        if missing:
            raise ValueError(
                f"cannot evaluate success gate for {model}/repetition "
                f"{repetition}; missing systems: {sorted(missing)}"
            )
        simple = [by_system[name] for name in sorted(simple_systems)]
        # Comparator selection is frozen in RESEARCH.md: maximum F1, then fewer
        # false-alarm checkpoints, then fewer input tokens, then lexical name.
        comparator = min(
            simple,
            key=lambda item: (
                -item.f1,
                item.false_alarm_checkpoints,
                item.input_tokens,
                item.system,
            ),
        )
        anamnesis = by_system["anamnesis"]
        full_context = by_system["full_context"]
        if full_context.input_tokens <= 0:
            raise ValueError("full_context must report positive input tokens")
        reduction = 1 - anamnesis.input_tokens / full_context.input_tokens
        f1_gain = anamnesis.f1 - comparator.f1
        f1_pass = f1_gain >= 0.05
        token_pass = reduction >= 0.30
        false_alarm_pass = (
            anamnesis.false_alarm_checkpoints <= comparator.false_alarm_checkpoints
        )
        gates.append(
            SuccessGateResult(
                repetition=repetition,
                model=model,
                comparator=comparator.system,
                f1_gain=f1_gain,
                input_token_reduction=reduction,
                anamnesis_false_alarm_checkpoints=(anamnesis.false_alarm_checkpoints),
                comparator_false_alarm_checkpoints=(comparator.false_alarm_checkpoints),
                f1_pass=f1_pass,
                token_pass=token_pass,
                false_alarm_pass=false_alarm_pass,
                supported=f1_pass and token_pass and false_alarm_pass,
            )
        )
    return gates
