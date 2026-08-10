from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from anamnesis.schema import (
    ExpectedAction,
    ForbiddenAction,
    PredictedAction,
    Scenario,
    ScenarioEvent,
    ScenarioRun,
    Usage,
)
from anamnesis.scoring import (
    aggregate_results,
    evaluate_success_gate,
    score_scenario,
)

BASE_TIME = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
SCENARIO_HASH = "0" * 64
PROMPT_HASH = "1" * 64


def _at(day: int) -> datetime:
    return BASE_TIME + timedelta(days=day)


def _event(
    event_id: str,
    day: int,
    *,
    kind: str = "clock_tick",
    supersedes: list[str] | None = None,
) -> ScenarioEvent:
    return ScenarioEvent(
        id=event_id,
        at=_at(day),
        kind=kind,
        text=f"Event {event_id}",
        supersedes=supersedes or [],
    )


def _expected_action(
    *,
    action_id: str = "expected-reminder",
    day: int = 4,
    action_key: str = "intent",
    payload: dict[str, str] | None = None,
    acceptable_evidence_sets: list[list[str]] | None = None,
) -> ExpectedAction:
    return ExpectedAction(
        id=action_id,
        action_key=action_key,
        payload=payload or {"subject": "submit the assignment"},
        window_start=_at(day),
        window_end=_at(day),
        acceptable_evidence_sets=acceptable_evidence_sets or [["intent"]],
    )


def _scenario(
    scenario_id: str,
    *,
    events: list[ScenarioEvent],
    expected_actions: list[ExpectedAction] | None = None,
    forbidden_actions: list[ForbiddenAction] | None = None,
) -> Scenario:
    return Scenario(
        id=scenario_id,
        title=f"Scenario {scenario_id}",
        description="A deterministic scorer fixture.",
        timezone="UTC",
        start_at=BASE_TIME,
        end_at=_at(7),
        events=events,
        expected_actions=expected_actions or [],
        forbidden_actions=forbidden_actions or [],
    )


def _prediction(
    *,
    day: int,
    decision_event_id: str,
    action_key: str = "intent",
    payload: dict[str, str] | None = None,
    evidence_event_ids: list[str] | None = None,
) -> PredictedAction:
    return PredictedAction(
        action_key=action_key,
        payload=payload or {"subject": "submit the assignment"},
        summary="Remind the user.",
        evidence_event_ids=(
            evidence_event_ids if evidence_event_ids is not None else ["intent"]
        ),
        emitted_at=_at(day),
        decision_event_id=decision_event_id,
    )


def _run(
    scenario: Scenario,
    predictions: list[PredictedAction],
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float | None = None,
    system: str = "full_context",
) -> ScenarioRun:
    return ScenarioRun(
        scenario_id=scenario.id,
        system=system,
        repetition=1,
        model="test-model",
        prompt_version="test-v1",
        scenario_sha256=SCENARIO_HASH,
        prompt_sha256=PROMPT_HASH,
        system_config_sha256="2" * 64,
        predictions=predictions,
        usage=Usage(
            input_tokens=input_tokens,
            uncached_input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ),
        decision_usage=Usage(
            input_tokens=input_tokens,
            uncached_input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ),
    )


def _standard_scenario(scenario_id: str = "standard") -> Scenario:
    return _scenario(
        scenario_id,
        events=[
            _event("intent", 0, kind="user_message"),
            _event("due", 4),
        ],
        expected_actions=[_expected_action()],
    )


def test_perfect_run_scores_one_true_positive_with_exact_provenance() -> None:
    scenario = _standard_scenario()
    run = _run(
        scenario,
        [_prediction(day=4, decision_event_id="due")],
    )

    score = score_scenario(scenario, run)

    assert (score.tp, score.fp, score.fn) == (1, 0, 0)
    assert score.false_reminders == 0
    assert score.false_alarm_checkpoints == 0
    assert score.provenance_covered == 1
    assert score.provenance_exact == 1
    assert score.matched_pairs == [("expected-reminder", 0)]
    assert score.unmatched_prediction_indices == []
    assert score.unmatched_expected_ids == []


def test_duplicate_action_is_one_false_positive_and_duplicate_error() -> None:
    scenario = _standard_scenario("duplicate")
    prediction = _prediction(day=4, decision_event_id="due")
    run = _run(scenario, [prediction, prediction.model_copy()])

    score = score_scenario(scenario, run)

    assert (score.tp, score.fp, score.fn) == (1, 1, 0)
    assert score.false_reminders == 1
    assert score.duplicate_errors == 1
    assert score.false_alarm_checkpoints == 0
    assert score.matched_pairs == [("expected-reminder", 0)]
    assert score.unmatched_prediction_indices == [1]


def test_obsolete_action_is_an_fp_and_triggers_the_obsolete_trap() -> None:
    scenario = _scenario(
        "obsolete",
        events=[
            _event("intent", 0, kind="user_message"),
            _event("deadline-update", 2, kind="user_message", supersedes=["intent"]),
            _event("old-due", 4),
            _event("new-due", 6),
        ],
        expected_actions=[
            _expected_action(
                day=6,
                payload={
                    "subject": "submit the assignment",
                    "date": "2026-01-11",
                },
                acceptable_evidence_sets=[["intent", "deadline-update"]],
            )
        ],
        forbidden_actions=[
            ForbiddenAction(
                id="old-deadline-trap",
                action_key="intent",
                payload={
                    "subject": "submit the assignment",
                    "date": "2026-01-09",
                },
                window_start=_at(4),
                window_end=_at(4),
                reason="obsolete",
                related_event_ids=["intent", "deadline-update"],
            )
        ],
    )
    run = _run(
        scenario,
        [
            _prediction(
                day=4,
                decision_event_id="old-due",
                payload={
                    "subject": "submit the assignment",
                    "date": "2026-01-09",
                },
            )
        ],
    )

    score = score_scenario(scenario, run)

    assert (score.tp, score.fp, score.fn) == (0, 1, 1)
    assert score.obsolete_errors == 1
    assert score.obsolete_traps == 1
    assert score.obsolete_traps_triggered == 1
    assert score.duplicate_errors == 0
    assert score.false_alarm_checkpoints == 1


def test_action_on_negative_scenario_is_a_false_alarm() -> None:
    scenario = _scenario(
        "negative",
        events=[_event("nothing-due", 3)],
    )
    run = _run(
        scenario,
        [
            _prediction(
                day=3,
                decision_event_id="nothing-due",
                action_key="unrequested-reminder",
                evidence_event_ids=["nothing-due"],
            )
        ],
    )

    score = score_scenario(scenario, run)
    aggregate = aggregate_results([(score, run)])[0]

    assert (score.tp, score.fp, score.fn) == (0, 1, 0)
    assert score.negative_checkpoints == 1
    assert score.false_alarm_checkpoints == 1
    assert score.false_reminders == 1
    assert aggregate.false_alarm_rate == 1.0


@pytest.mark.parametrize("citation", ["irrelevant", "future"])
def test_incorrect_or_future_provenance_does_not_receive_credit(
    citation: str,
) -> None:
    scenario = _scenario(
        f"bad-provenance-{citation}",
        events=[
            _event("intent", 0, kind="user_message"),
            _event("irrelevant", 1),
            _event("due", 4),
            _event("future", 5),
        ],
        expected_actions=[_expected_action()],
    )
    run = _run(
        scenario,
        [
            _prediction(
                day=4,
                decision_event_id="due",
                evidence_event_ids=[citation],
            )
        ],
    )

    score = score_scenario(scenario, run)

    assert (score.tp, score.fp, score.fn) == (1, 0, 0)
    assert score.provenance_covered == 1
    assert score.provenance_exact == 0


def test_duplicate_provenance_citation_is_not_exact() -> None:
    scenario = _standard_scenario("duplicate-provenance")
    run = _run(
        scenario,
        [
            _prediction(
                day=4,
                decision_event_id="due",
                evidence_event_ids=["intent", "due", "due"],
            )
        ],
    )

    score = score_scenario(scenario, run)

    assert (score.tp, score.fp, score.fn) == (1, 0, 0)
    assert score.provenance_covered == 1
    assert score.provenance_exact == 0


def test_missing_provenance_does_not_invalidate_a_correct_action() -> None:
    scenario = _standard_scenario("missing-provenance")
    run = _run(
        scenario,
        [
            _prediction(
                day=4,
                decision_event_id="due",
                evidence_event_ids=[],
            )
        ],
    )

    score = score_scenario(scenario, run)

    assert (score.tp, score.fp, score.fn) == (1, 0, 0)
    assert score.provenance_covered == 0
    assert score.provenance_exact == 0


def test_scorer_rejects_tampered_prediction_checkpoint() -> None:
    scenario = _standard_scenario("tampered-checkpoint")
    unknown_checkpoint = _prediction(day=4, decision_event_id="unknown")
    wrong_timestamp = _prediction(
        day=3,
        decision_event_id="due",
    )

    with pytest.raises(ValueError, match="unknown decision_event_id"):
        score_scenario(scenario, _run(scenario, [unknown_checkpoint]))
    with pytest.raises(ValueError, match="emitted_at"):
        score_scenario(scenario, _run(scenario, [wrong_timestamp]))


def test_aggregation_uses_micro_counts_for_precision_recall_and_f1() -> None:
    perfect_scenario = _standard_scenario("aggregate-perfect")
    perfect_run = _run(
        perfect_scenario,
        [_prediction(day=4, decision_event_id="due")],
        input_tokens=10,
        output_tokens=2,
        cost_usd=1.0,
    )
    imperfect_scenario = _standard_scenario("aggregate-imperfect")
    imperfect_run = _run(
        imperfect_scenario,
        [
            _prediction(
                day=4,
                decision_event_id="due",
                action_key=f"wrong-{index}",
            )
            for index in range(3)
        ],
        input_tokens=30,
        output_tokens=6,
        cost_usd=2.0,
    )

    aggregate = aggregate_results(
        [
            (score_scenario(perfect_scenario, perfect_run), perfect_run),
            (score_scenario(imperfect_scenario, imperfect_run), imperfect_run),
        ]
    )[0]

    assert aggregate.scenarios == 2
    assert (aggregate.tp, aggregate.fp, aggregate.fn) == (1, 3, 1)
    assert aggregate.precision == pytest.approx(0.25)
    assert aggregate.recall == pytest.approx(0.5)
    assert aggregate.f1 == pytest.approx(1 / 3)
    assert aggregate.input_tokens == 40
    assert aggregate.output_tokens == 8
    assert aggregate.cost_usd == pytest.approx(3.0)


def test_success_gate_uses_best_simple_f1_and_all_fixed_thresholds() -> None:
    scenario = _standard_scenario("success-gate")
    perfect_prediction = _prediction(day=4, decision_event_id="due")
    inputs = {
        "no_memory": ([], 80),
        "full_context": ([], 100),
        "vector_rag": ([], 70),
        "anamnesis": ([perfect_prediction], 60),
    }
    scored_runs = []
    for system, (predictions, input_tokens) in inputs.items():
        run = _run(
            scenario,
            predictions,
            system=system,
            input_tokens=input_tokens,
        )
        scored_runs.append((score_scenario(scenario, run), run))

    gate = evaluate_success_gate(aggregate_results(scored_runs))[0]

    # All simple systems tie at F1=0 and false alarms, so lower tokens wins.
    assert gate.comparator == "vector_rag"
    assert gate.f1_gain == 1.0
    assert gate.input_token_reduction == pytest.approx(0.4)
    assert gate.false_alarm_pass
    assert gate.supported
