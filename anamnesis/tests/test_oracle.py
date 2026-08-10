"""Isolation and ceiling checks for the manually annotated oracle compiler."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from anamnesis.baselines import create_strategy
from anamnesis.io import canonical_sha256, load_scenarios
from anamnesis.memory import CompilerRequest, InMemoryAnamnesis
from anamnesis.oracle import (
    ORACLE_ANNOTATION_POLICY,
    ORACLE_ARTIFACT_PURPOSE,
    ORACLE_SYSTEM_NAME,
    OracleAnamnesisMemoryStrategy,
    OracleCompiler,
    OracleCompilerArtifact,
    load_oracle_artifact,
    oracle_artifact_sha256,
)
from anamnesis.schema import (
    Decision,
    PredictedAction,
    ProposedAction,
    Scenario,
    ScenarioRun,
    Usage,
)
from anamnesis.scoring import score_scenario

ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = ROOT / "eval" / "scenarios" / "smoke.jsonl"
ORACLE_PATH = ROOT / "eval" / "oracle" / "smoke_memory_deltas.v1.json"
ZERO_SHA256 = "0" * 64


def _load() -> tuple[list[Scenario], OracleCompilerArtifact]:
    scenarios = load_scenarios(SMOKE_PATH)
    return scenarios, load_oracle_artifact(ORACLE_PATH, scenarios)


def test_oracle_artifact_covers_every_non_clock_event_exactly() -> None:
    scenarios, artifact = _load()

    assert artifact.purpose == ORACLE_ARTIFACT_PURPOSE
    assert artifact.annotation_policy == ORACLE_ANNOTATION_POLICY
    assert artifact.hypothesis_test_eligible is False
    assert [record.scenario_id for record in artifact.scenarios] == [
        scenario.id for scenario in scenarios
    ]
    assert sum(len(record.events) for record in artifact.scenarios) == 53
    for scenario in scenarios:
        records = artifact.records_for(scenario.to_runtime())
        assert [record.event_id for record in records] == [
            event.id for event in scenario.events if event.kind != "clock_tick"
        ]


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("root", "expected_actions", []),
        ("root", "tags", ["gold"]),
        ("event", "supersedes", ["future-event"]),
        ("event", "acceptable_evidence_sets", [["gold-event"]]),
    ],
)
def test_oracle_artifact_schema_rejects_author_only_fields(
    location: str,
    field: str,
    value: object,
) -> None:
    raw = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    target = raw if location == "root" else raw["scenarios"][0]["events"][0]
    target[field] = value

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OracleCompilerArtifact.model_validate(raw)


def test_oracle_artifact_fails_closed_on_missing_record_or_event_change() -> None:
    scenarios, _ = _load()
    raw = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    raw["scenarios"][0]["events"].pop()
    incomplete = OracleCompilerArtifact.model_validate(raw)
    with pytest.raises(ValueError, match="coverage/order differs"):
        incomplete.records_for(scenarios[0].to_runtime())

    raw = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    raw["scenarios"][0]["events"][0]["observable_event_sha256"] = ZERO_SHA256
    changed = OracleCompilerArtifact.model_validate(raw)
    with pytest.raises(ValueError, match="observable-event hash differs"):
        changed.records_for(scenarios[0].to_runtime())


def test_oracle_artifact_hash_is_independent_of_json_formatting() -> None:
    scenarios, artifact = _load()
    compact = OracleCompilerArtifact.model_validate_json(
        json.dumps(json.loads(ORACLE_PATH.read_text(encoding="utf-8")))
    )
    assert oracle_artifact_sha256(compact) == oracle_artifact_sha256(artifact)


def test_oracle_compiler_has_complete_zero_accounting_and_strict_order() -> None:
    scenarios, artifact = _load()
    runtime = scenarios[0].to_runtime()
    compiler = OracleCompiler(artifact, runtime)
    first = next(event for event in runtime.events if event.kind != "clock_tick")
    call = asyncio.run(
        compiler.compile(
            CompilerRequest(
                event=first,
                active_state='{"facts":[],"intents":[]}',
            )
        )
    )

    assert call.delta is not None
    assert call.raw_completion == call.delta.model_dump_json()
    assert call.usage == Usage(cost_usd=0.0)
    assert call.usage_complete and call.cost_complete
    assert not call.parse_error
    assert call.latency_ms >= 0
    assert compiler.requests[0].event == first

    later = next(
        event
        for event in runtime.events
        if event.kind != "clock_tick" and event.id != first.id
    )
    compiler.reset()
    with pytest.raises(ValueError, match="event order differs"):
        asyncio.run(
            compiler.compile(
                CompilerRequest(event=later, active_state='{"facts":[],"intents":[]}')
            )
        )
    with pytest.raises(ValueError, match="did not consume all"):
        compiler.assert_complete()


def test_oracle_strategy_name_is_isolated_from_headline_factory() -> None:
    scenarios, artifact = _load()
    compiler = OracleCompiler(artifact, scenarios[0].to_runtime())
    strategy = OracleAnamnesisMemoryStrategy(compiler)

    assert strategy.name == ORACLE_SYSTEM_NAME
    assert strategy.name != "anamnesis"
    with pytest.raises(ValueError, match="unknown baseline"):
        create_strategy(ORACLE_SYSTEM_NAME)


async def _replay_with_due_candidate_copier(
    scenario: Scenario,
    artifact: OracleCompilerArtifact,
) -> ScenarioRun:
    """Execute only sanitized events; the copier has no scenario gold input."""

    runtime = scenario.to_runtime()
    compiler = OracleCompiler(artifact, runtime)
    memory = InMemoryAnamnesis()
    predictions: list[PredictedAction] = []

    for event in runtime.events:
        delta = None
        if event.kind != "clock_tick":
            call = await compiler.compile(
                CompilerRequest(event=event, active_state=memory.compiler_state())
            )
            assert call.usage == Usage(cost_usd=0.0)
            delta = call.delta
        applied = memory.ingest(event, delta)
        assert applied.accepted, (scenario.id, event.id, applied.error)
        selection = memory.select(event)
        actions: list[ProposedAction] = []
        for candidate in selection.due_candidates:
            evidence = list(candidate.evidence_event_ids)
            if event.id not in evidence:
                evidence.append(event.id)
            action = ProposedAction(
                kind=candidate.action_template.kind,
                action_key=candidate.action_key,
                payload=dict(candidate.action_template.payload),
                summary=candidate.action_template.summary,
                evidence_event_ids=evidence,
            )
            actions.append(action)
            predictions.append(
                PredictedAction(
                    **action.model_dump(),
                    emitted_at=event.at,
                    decision_event_id=event.id,
                )
            )
        memory.commit(event, Decision(actions=actions))

    compiler.assert_complete()
    zero_usage = Usage(cost_usd=0.0)
    return ScenarioRun(
        scenario_id=scenario.id,
        system=ORACLE_SYSTEM_NAME,
        repetition=1,
        model="deterministic/oracle-action-copier",
        prompt_version="offline.oracle.v1",
        scenario_sha256=canonical_sha256(scenario),
        prompt_sha256=ZERO_SHA256,
        system_config_sha256="1" * 64,
        predictions=predictions,
        usage=zero_usage,
        decision_usage=zero_usage,
        compiler_usage=zero_usage,
        usage_complete=True,
        cost_complete=True,
    )


def test_oracle_annotations_reach_exact_smoke_ceiling_offline() -> None:
    scenarios, artifact = _load()

    async def replay_all() -> list[ScenarioRun]:
        return [
            await _replay_with_due_candidate_copier(scenario, artifact)
            for scenario in scenarios
        ]

    runs = asyncio.run(replay_all())
    scores = [
        score_scenario(scenario, run)
        for scenario, run in zip(scenarios, runs, strict=True)
    ]

    assert sum(score.tp for score in scores) == 8
    assert sum(score.fp for score in scores) == 0
    assert sum(score.fn for score in scores) == 0
    assert sum(score.provenance_exact for score in scores) == 8
    assert sum(score.obsolete_errors for score in scores) == 0
    assert all(score.invalid_outputs == 0 for score in scores)
