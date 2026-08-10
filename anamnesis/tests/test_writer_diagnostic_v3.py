"""Frozen-byte, blind-isolation, and oracle-ceiling checks for writer v3."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.memory import CompilerRequest, DueCandidate, InMemoryAnamnesis
from anamnesis.oracle import (
    ORACLE_ANNOTATION_POLICY,
    ORACLE_ARTIFACT_PURPOSE,
    ORACLE_SYSTEM_NAME,
    OracleCompiler,
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
SCENARIO_DIR = ROOT / "eval" / "scenarios"
DATASET_PATH = SCENARIO_DIR / "writer_diagnostic.v3.jsonl"
MANIFEST_PATH = SCENARIO_DIR / "writer_diagnostic.v3.manifest.json"
ORACLE_PATH = ROOT / "eval" / "oracle" / "writer_diagnostic_memory_deltas.v3.json"
V2_DATASET_PATH = SCENARIO_DIR / "writer_diagnostic.v2.jsonl"
V2_MANIFEST_PATH = SCENARIO_DIR / "writer_diagnostic.v2.manifest.json"
V2_ORACLE_PATH = ROOT / "eval" / "oracle" / "writer_diagnostic_memory_deltas.v2.json"
COMPARISON_PATHS = (
    SCENARIO_DIR / "smoke.jsonl",
    SCENARIO_DIR / "dev.jsonl",
    SCENARIO_DIR / "sealed.jsonl",
    SCENARIO_DIR / "all.jsonl",
    SCENARIO_DIR / "writer_diagnostic.v1.jsonl",
)

DATASET_FILE_SHA256 = "34e2e8751bf32a3a2e29ac75d727f2b5cf73aaba13ccc9ba1d9fdf00bf7eaf4f"
DATASET_CANONICAL_SHA256 = (
    "37a62b643ed920bf115aa0ea2495f00fc12bc349fabc4664ad430c4e9eb71115"
)
MANIFEST_FILE_SHA256 = (
    "18236aa26da2e9957560e2300c36b219380e801d80953ec490217046debd1b25"
)
ORACLE_FILE_SHA256 = "7adb64eda15daf5351260933fbd0625fbc13c6899361735a9bf0ce13c063f857"
ORACLE_CANONICAL_SHA256 = (
    "9e46d95b308cdd5cd6f995f8521c7600ef4a041a0ad9913e6e8eb8ce808c01f3"
)
ZERO_SHA256 = "0" * 64

EXPECTED_FAMILIES = {
    "basic_deadline",
    "cancellation",
    "conjunctive_trigger",
    "deadline_update",
    "entity_grounding",
    "fact_update",
    "negative_control",
    "recurring_intention",
    "reversible_completion",
    "threshold_trigger",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> tuple[list[Scenario], dict[str, object]]:
    scenarios = load_scenarios(DATASET_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return scenarios, manifest


def _authored_surfaces(scenario: Scenario) -> set[str]:
    return {
        scenario.title,
        scenario.description,
        *(event.text for event in scenario.events),
    }


def _normalize_v3_lineage(value: object) -> object:
    """Reverse only the declared namespace and three v3 corrections."""

    if isinstance(value, dict):
        return {key: _normalize_v3_lineage(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_v3_lineage(item) for item in value]
    if not isinstance(value, str):
        return value
    replacements = {
        "Orion Dome": "orion dome",
        "Cedar incubator": "cedar incubator",
        (
            "If I have not sent the theater sponsorship letter to the theater "
            "sponsor by Friday at 17:00, remind me to send it."
        ): (
            "If I have not sent the theater sponsorship letter by Friday at "
            "17:00, remind me to send it."
        ),
    }
    value = replacements.get(value, value)
    return value.replace("wd3_", "wd2_").replace(
        "writer_diagnostic_v3", "writer_diagnostic_v2"
    )


def _jsonl_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _canonical_payload(candidate: DueCandidate) -> str:
    return json.dumps(
        dict(candidate.action_template.payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _w2_candidate_key(
    checkpoint: str,
    candidate: DueCandidate,
) -> tuple[object, ...]:
    return (
        checkpoint,
        candidate.action_key,
        candidate.due_at.isoformat(),
        candidate.action_template.kind,
        _canonical_payload(candidate),
        tuple(sorted(candidate.evidence_event_ids)),
    )


def _legacy_candidate_key(
    checkpoint: str,
    candidate: DueCandidate,
) -> tuple[object, ...]:
    return (
        checkpoint,
        candidate.action_key,
        candidate.due_at.isoformat(),
        candidate.action_template.kind,
        _canonical_payload(candidate),
        candidate.action_template.summary,
        tuple(sorted(candidate.evidence_event_ids)),
    )


def test_writer_diagnostic_v3_exact_bytes_and_hashes_are_frozen() -> None:
    scenarios, manifest = _load()
    oracle = load_oracle_artifact(ORACLE_PATH, scenarios)

    assert _sha256(DATASET_PATH) == DATASET_FILE_SHA256
    assert _sha256(MANIFEST_PATH) == MANIFEST_FILE_SHA256
    assert _sha256(ORACLE_PATH) == ORACLE_FILE_SHA256
    assert all(
        b"\r\n" not in path.read_bytes()
        for path in (DATASET_PATH, MANIFEST_PATH, ORACLE_PATH)
    )

    assert dataset_sha256(scenarios) == DATASET_CANONICAL_SHA256
    assert oracle_artifact_sha256(oracle) == ORACLE_CANONICAL_SHA256
    assert manifest["file_sha256"] == DATASET_FILE_SHA256
    assert manifest["canonical_dataset_sha256"] == DATASET_CANONICAL_SHA256
    assert manifest["record_sha256"] == {
        scenario.id: canonical_sha256(scenario) for scenario in scenarios
    }
    assert manifest["oracle_reference"] == {
        "path": "eval/oracle/writer_diagnostic_memory_deltas.v3.json",
        "file_sha256": ORACLE_FILE_SHA256,
        "canonical_artifact_sha256": ORACLE_CANONICAL_SHA256,
        "annotation_policy": ORACLE_ANNOTATION_POLICY,
        "event_record_count": 46,
        "visible_to_evaluated_writer": False,
        "human_annotation_measured": False,
        "offline_replay_ceiling": {
            "tp": 8,
            "fp": 0,
            "fn": 0,
            "provenance_exact": 8,
            "obsolete_errors": 0,
            "invalid_outputs": 0,
        },
    }


def test_writer_diagnostic_v3_scope_counts_families_and_origin() -> None:
    scenarios, manifest = _load()
    origins = manifest["scenario_origins"]
    assert isinstance(origins, list)
    family_counts = Counter(item["family"] for item in origins)

    assert len(scenarios) == manifest["scenario_count"] == 10
    assert sum(len(scenario.events) for scenario in scenarios) == 69
    assert (
        sum(
            event.kind != "clock_tick"
            for scenario in scenarios
            for event in scenario.events
        )
        == 46
    )
    assert sum(len(scenario.expected_actions) for scenario in scenarios) == 8
    assert sum(len(scenario.forbidden_actions) for scenario in scenarios) == 18
    assert sum(not scenario.expected_actions for scenario in scenarios) == 2
    assert (
        sum(
            any(action.reason == "obsolete" for action in scenario.forbidden_actions)
            for scenario in scenarios
        )
        == 3
    )
    assert set(family_counts) == EXPECTED_FAMILIES
    assert set(family_counts.values()) == {1}
    assert manifest["family_counts"] == dict(sorted(family_counts.items()))

    assert manifest["status"] == "frozen-diagnostic"
    assert manifest["claim_scope"] == "diagnostic_development_only"
    assert manifest["hypothesis_evidence"] is False
    assert manifest["preregistered_final_eligible"] is False
    assert manifest["member_of_development_35"] is False
    assert manifest["member_of_sealed_set"] is False
    assert manifest["writer_prompt_status_at_freeze"] == "not-authored"
    assert manifest["review_status"] == {
        "automated_integrity": "passed",
        "gold_provenance_policy": "minimal-causal-evidence-v1",
        "independent_human_review": "pending",
    }
    assert manifest["origin"] == {
        "type": "locally-authored",
        "longmemeval_items": 0,
        "triggerbench_items": 0,
        "external_dataset_items": 0,
    }
    assert all(
        item["origin"] == "locally-authored"
        and item["review_status"]
        == "automated-integrity-passed; independent-review-pending"
        for item in origins
    )


def test_writer_diagnostic_v3_is_exactly_the_declared_v2_correction() -> None:
    scenarios, manifest = _load()
    v2_records = _jsonl_records(V2_DATASET_PATH)
    v3_records = _jsonl_records(DATASET_PATH)

    assert [_normalize_v3_lineage(record) for record in v3_records] == v2_records

    by_id = {scenario.id: scenario for scenario in scenarios}
    orion = by_id["wd3_01_planetarium_deadline"]
    orion_payloads = [
        action.payload for action in (*orion.expected_actions, *orion.forbidden_actions)
    ]
    assert len(orion_payloads) == 3
    assert all(payload["project"] == "Orion Dome" for payload in orion_payloads)
    assert "Orion Dome" in orion.events[0].text

    cedar = by_id["wd3_05_incubator_grounding"]
    cedar_payloads = [
        action.payload for action in (*cedar.expected_actions, *cedar.forbidden_actions)
    ]
    assert len(cedar_payloads) == 3
    assert all(payload["item"] == "Cedar incubator" for payload in cedar_payloads)
    assert "Cedar incubator" in cedar.events[0].text

    letter = by_id["wd3_09_letter_reversal"]
    assert letter.events[0].text == (
        "If I have not sent the theater sponsorship letter to the theater sponsor "
        "by Friday at 17:00, remind me to send it."
    )
    letter_payloads = [
        action.payload
        for action in (*letter.expected_actions, *letter.forbidden_actions)
    ]
    assert len(letter_payloads) == 3
    assert all(payload["recipient"] == "theater sponsor" for payload in letter_payloads)

    v2_oracle = json.loads(V2_ORACLE_PATH.read_text(encoding="utf-8"))
    v3_oracle = _normalize_v3_lineage(
        json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    )
    assert isinstance(v3_oracle, dict)
    v3_oracle["canonical_dataset_sha256"] = v2_oracle["canonical_dataset_sha256"]
    for corrected_scenario, predecessor_scenario in zip(
        v3_oracle["scenarios"],
        v2_oracle["scenarios"],
        strict=True,
    ):
        for corrected_event, predecessor_event in zip(
            corrected_scenario["events"],
            predecessor_scenario["events"],
            strict=True,
        ):
            corrected_event["observable_event_sha256"] = predecessor_event[
                "observable_event_sha256"
            ]
    assert v3_oracle == v2_oracle

    assert _sha256(V2_DATASET_PATH) == manifest["lineage"]["derived_from_file_sha256"]
    assert (
        _sha256(V2_MANIFEST_PATH) == manifest["lineage"]["derived_from_manifest_sha256"]
    )
    assert _sha256(V2_ORACLE_PATH) == manifest["lineage"]["derived_from_oracle_sha256"]
    assert manifest["lineage"] == {
        "derived_from": "eval/scenarios/writer_diagnostic.v2.jsonl",
        "derived_from_git_commit": ("0a8b6714dea57eb9a0c9dded8535569c380be9b1"),
        "derived_from_file_sha256": (
            "76e4caaac216c61f66df1af6457b2e138c034dc5db87a3a6e5bd40336d87c5ab"
        ),
        "derived_from_manifest_sha256": (
            "a3ec4c7e8c911c6d10421b1efc2e11e33b5ca0edd0429529981c01023029adbf"
        ),
        "derived_from_oracle_sha256": (
            "49a7ad07bc7e441b28d7bfeed6d76cbe2d870d1b60b38a855c9c270ff385eec3"
        ),
        "predecessor_disposition": "rejected-before-prompt",
        "independent_audit_findings": [
            (
                "wd2_01 optional project payload casing did not exactly match "
                "observable source casing"
            ),
            (
                "wd2_05 optional item payload casing did not exactly match "
                "observable source casing"
            ),
            (
                "wd2_09 recipient payload was not explicitly supported by the "
                "creating observable event"
            ),
        ],
        "w2_prompt_existed_at_rejection": False,
        "w2_model_calls_before_rejection": 0,
        "id_namespace_change": "wd2_ to wd3_",
        "allowed_corrections": [
            "Preserve exact source casing Orion Dome in wd3_01 project payload",
            ("Preserve exact source casing Cedar incubator in wd3_05 item payload"),
            ("Add explicit to the theater sponsor wording to wd3_09 creating event"),
        ],
        "other_causal_or_gold_changes": 0,
    }


def test_writer_diagnostic_v3_public_w2_candidate_key_excludes_summary() -> None:
    scenarios, manifest = _load()
    artifact = load_oracle_artifact(ORACLE_PATH, scenarios)

    async def collect_due_candidates() -> list[tuple[str, DueCandidate]]:
        collected: list[tuple[str, DueCandidate]] = []
        for scenario in scenarios:
            runtime = scenario.to_runtime()
            compiler = OracleCompiler(artifact, runtime)
            memory = InMemoryAnamnesis()
            for event in runtime.events:
                delta = None
                if event.kind != "clock_tick":
                    delta = (
                        await compiler.compile(
                            CompilerRequest(
                                event=event,
                                active_state=memory.compiler_state(),
                            )
                        )
                    ).delta
                assert memory.ingest(event, delta).accepted
                selection = memory.select(event)
                collected.extend(
                    (event.id, candidate) for candidate in selection.due_candidates
                )
                actions = [
                    ProposedAction(
                        kind=candidate.action_template.kind,
                        action_key=candidate.action_key,
                        payload=dict(candidate.action_template.payload),
                        summary=candidate.action_template.summary,
                        evidence_event_ids=list(candidate.evidence_event_ids),
                    )
                    for candidate in selection.due_candidates
                ]
                memory.commit(event, Decision(actions=actions))
            compiler.assert_complete()
        return collected

    due = asyncio.run(collect_due_candidates())
    w2_gate = Counter(_w2_candidate_key(checkpoint, item) for checkpoint, item in due)
    legacy_diagnostic = Counter(
        _legacy_candidate_key(checkpoint, item) for checkpoint, item in due
    )

    assert sum(w2_gate.values()) == sum(legacy_diagnostic.values()) == 8
    assert len(w2_gate) == len(legacy_diagnostic) == 8
    assert all(len(key) == 6 for key in w2_gate)
    assert all(len(key) == 7 for key in legacy_diagnostic)

    summary_variants: list[tuple[str, DueCandidate]] = []
    for index, (checkpoint, candidate) in enumerate(due):
        template = candidate.action_template.model_copy(
            update={"summary": f"UX rendering variant {index + 1}"}
        )
        summary_variants.append(
            (
                checkpoint,
                candidate.model_copy(update={"action_template": template}),
            )
        )
    assert (
        Counter(
            _w2_candidate_key(checkpoint, item) for checkpoint, item in summary_variants
        )
        == w2_gate
    )
    assert (
        Counter(
            _legacy_candidate_key(checkpoint, item)
            for checkpoint, item in summary_variants
        )
        != legacy_diagnostic
    )

    payloads = [json.loads(str(key[4])) for key in w2_gate]
    assert any(payload.get("project") == "Orion Dome" for payload in payloads)
    assert any(payload.get("item") == "Cedar incubator" for payload in payloads)
    assert any(payload.get("recipient") == "theater sponsor" for payload in payloads)

    assert manifest["w2_candidate_matching_protocol"] == {
        "version": "w2.candidate-key.v1",
        "gate_key_fields": [
            "checkpoint",
            "action_key",
            "due_at",
            "kind",
            "canonical_payload",
            "sorted_evidence",
        ],
        "excluded_from_gate": ["summary", "intent_id", "occurrence_id"],
        "summary_role": "noncanonical UX text",
        "hidden_exact_summary_gate": False,
        "legacy_exact_key_diagnostic_only": True,
        "legacy_exact_key_fields": [
            "checkpoint",
            "action_key",
            "due_at",
            "kind",
            "canonical_payload",
            "summary",
            "sorted_evidence",
        ],
    }


def test_writer_diagnostic_v3_anti_overlap_attestation_is_recomputed() -> None:
    scenarios, manifest = _load()
    comparison = [
        scenario for path in COMPARISON_PATHS for scenario in load_scenarios(path)
    ]

    new_scenario_ids = {scenario.id for scenario in scenarios}
    old_scenario_ids = {scenario.id for scenario in comparison}
    new_event_ids = {event.id for scenario in scenarios for event in scenario.events}
    old_event_ids = {event.id for scenario in comparison for event in scenario.events}
    new_hashes = {canonical_sha256(scenario) for scenario in scenarios}
    old_hashes = {canonical_sha256(scenario) for scenario in comparison}
    new_surfaces = {
        surface for scenario in scenarios for surface in _authored_surfaces(scenario)
    }
    old_surfaces = {
        surface for scenario in comparison for surface in _authored_surfaces(scenario)
    }

    assert new_scenario_ids.isdisjoint(old_scenario_ids)
    assert new_event_ids.isdisjoint(old_event_ids)
    assert new_hashes.isdisjoint(old_hashes)
    assert new_surfaces.isdisjoint(old_surfaces)
    assert manifest["anti_overlap_attestation"] == {
        "comparison_sets": [str(path.relative_to(ROOT)) for path in COMPARISON_PATHS],
        "scenario_ids_disjoint": True,
        "event_ids_disjoint": True,
        "canonical_record_hashes_disjoint": True,
        "exact_titles_descriptions_event_texts_disjoint": True,
        "automated_test": "tests/test_writer_diagnostic_v3.py",
        "manual_source_content_reviewed_for_authorship": False,
        "scope": (
            "pre-v2 datasets; v2 excluded because v3 is an audited correction release"
        ),
        "derived_predecessor": "eval/scenarios/writer_diagnostic.v2.jsonl",
        "derived_predecessor_exact_overlap_expected": True,
    }


def test_writer_diagnostic_v3_timelines_gold_and_provenance_are_well_formed() -> None:
    scenarios, _ = _load()

    for scenario in scenarios:
        local_zone = ZoneInfo(scenario.timezone)
        start = scenario.start_at.astimezone(local_zone)
        end = scenario.end_at.astimezone(local_zone)
        events_by_id = {event.id: event for event in scenario.events}

        assert scenario.timezone == "Europe/Athens"
        assert start.weekday() == 0
        assert end.weekday() == 6
        assert (end.date() - start.date()).days + 1 == 7
        assert all(event.kind != "assistant_decision" for event in scenario.events)

        observed: set[str] = set()
        for event in scenario.events:
            assert set(event.supersedes) <= observed
            observed.add(event.id)

        for action in scenario.expected_actions:
            checkpoint_ids = {
                event.id
                for event in scenario.events
                if action.window_start <= event.at <= action.window_end
            }
            assert action.window_start == action.window_end
            assert len(checkpoint_ids) == 1
            assert action.action_key in events_by_id
            assert events_by_id[action.action_key].kind == "user_message"
            assert len(action.acceptable_evidence_sets) == len(
                {tuple(evidence) for evidence in action.acceptable_evidence_sets}
            )
            for evidence in action.acceptable_evidence_sets:
                assert action.action_key in evidence
                assert checkpoint_ids.intersection(evidence)
                assert set(evidence) <= events_by_id.keys()
                assert all(
                    events_by_id[event_id].at <= action.window_end
                    for event_id in evidence
                )
                assert not any(
                    set(other) < set(evidence)
                    for other in action.acceptable_evidence_sets
                )
            if occurrence_date := action.payload.get("date"):
                assert date.fromisoformat(str(occurrence_date)).isoformat() == (
                    occurrence_date
                )
                assert occurrence_date == action.window_start.date().isoformat()

        for action in scenario.forbidden_actions:
            assert action.action_key in events_by_id
            assert events_by_id[action.action_key].kind == "user_message"
            assert set(action.related_event_ids) <= events_by_id.keys()
            assert all(
                events_by_id[event_id].at <= action.window_end
                for event_id in action.related_event_ids
            )
            assert any(
                action.window_start <= event.at <= action.window_end
                for event in scenario.events
            )


def test_writer_diagnostic_v3_runtime_boundary_removes_author_fields() -> None:
    scenarios, _ = _load()
    hidden_names = {
        "acceptable_evidence_sets",
        "expected_actions",
        "forbidden_actions",
        "related_event_ids",
        "supersedes",
        "tags",
    }

    for scenario in scenarios:
        runtime = scenario.to_runtime()
        assert [event.id for event in runtime.events] == [
            event.id for event in scenario.events
        ]
        for index, event in enumerate(runtime.events):
            assert set(type(event).model_fields) == {"id", "at", "kind", "text"}
            serialized = event.model_dump_json()
            assert all(name not in serialized for name in hidden_names)
            assert scenario.title not in serialized
            assert scenario.description not in serialized
            assert all(
                future.text not in serialized for future in runtime.events[index + 1 :]
            )


def test_writer_diagnostic_v3_oracle_covers_only_non_clock_observables() -> None:
    scenarios, manifest = _load()
    artifact = load_oracle_artifact(ORACLE_PATH, scenarios)
    raw = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))

    assert artifact.purpose == ORACLE_ARTIFACT_PURPOSE
    assert artifact.claim_scope == "diagnostic_development_only"
    assert artifact.hypothesis_test_eligible is False
    assert artifact.annotation_policy == ORACLE_ANNOTATION_POLICY
    assert artifact.canonical_dataset_sha256 == DATASET_CANONICAL_SHA256
    assert [record.scenario_id for record in artifact.scenarios] == [
        scenario.id for scenario in scenarios
    ]
    assert sum(len(record.events) for record in artifact.scenarios) == 46
    assert manifest["oracle_reference"]["event_record_count"] == 46

    for scenario in scenarios:
        records = artifact.records_for(scenario.to_runtime())
        expected = [
            event
            for event in scenario.to_runtime().events
            if event.kind != "clock_tick"
        ]
        assert [record.event_id for record in records] == [
            event.id for event in expected
        ]
        assert [record.observable_event_sha256 for record in records] == [
            canonical_sha256(event) for event in expected
        ]

    hidden_names = {
        "acceptable_evidence_sets",
        "expected_actions",
        "forbidden_actions",
        "related_event_ids",
        "supersedes",
        "tags",
        "future_events",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert hidden_names.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(raw)


def test_writer_diagnostic_v3_oracle_real_store_reaches_exact_ceiling() -> None:
    scenarios, manifest = _load()
    artifact = load_oracle_artifact(ORACLE_PATH, scenarios)

    async def replay_one(scenario: Scenario) -> ScenarioRun:
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
                assert call.usage_complete and call.cost_complete
                assert not call.parse_error
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
            model="deterministic/writer-diagnostic-v3-oracle",
            prompt_version="offline.writer-oracle.v3",
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

    runs = [asyncio.run(replay_one(scenario)) for scenario in scenarios]
    scores = [
        score_scenario(scenario, run)
        for scenario, run in zip(scenarios, runs, strict=True)
    ]
    actual = {
        "tp": sum(score.tp for score in scores),
        "fp": sum(score.fp for score in scores),
        "fn": sum(score.fn for score in scores),
        "provenance_exact": sum(score.provenance_exact for score in scores),
        "obsolete_errors": sum(score.obsolete_errors for score in scores),
        "invalid_outputs": sum(score.invalid_outputs for score in scores),
    }

    assert actual == manifest["oracle_reference"]["offline_replay_ceiling"]
    assert actual == {
        "tp": 8,
        "fp": 0,
        "fn": 0,
        "provenance_exact": 8,
        "obsolete_errors": 0,
        "invalid_outputs": 0,
    }
