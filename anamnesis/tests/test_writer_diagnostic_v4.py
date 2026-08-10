"""Blind v4 freeze, prefix-derivability, and real-store ceiling checks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.memory import (
    AtTrigger,
    CancelIntent,
    CompilerRequest,
    ConditionTransitionTrigger,
    CreateIntent,
    DueCandidate,
    InMemoryAnamnesis,
    RecurringTrigger,
    SetFact,
    TruthValue,
    UpdateIntent,
)
from anamnesis.oracle import (
    ORACLE_ANNOTATION_POLICY,
    ORACLE_ARTIFACT_PURPOSE,
    ORACLE_SYSTEM_NAME,
    OracleCompiler,
    load_oracle_artifact,
    oracle_artifact_sha256,
)
from anamnesis.schema import (
    OPTIONAL_PAYLOAD_KEYS,
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
DATASET_PATH = SCENARIO_DIR / "writer_diagnostic.v4.jsonl"
MANIFEST_PATH = SCENARIO_DIR / "writer_diagnostic.v4.manifest.json"
ORACLE_PATH = ROOT / "eval" / "oracle" / "writer_diagnostic_memory_deltas.v4.json"
COMPARISON_PATHS = (
    SCENARIO_DIR / "smoke.jsonl",
    SCENARIO_DIR / "dev.jsonl",
    SCENARIO_DIR / "sealed.jsonl",
    SCENARIO_DIR / "all.jsonl",
    SCENARIO_DIR / "writer_diagnostic.v1.jsonl",
    SCENARIO_DIR / "writer_diagnostic.v2.jsonl",
    SCENARIO_DIR / "writer_diagnostic.v3.jsonl",
)

DATASET_FILE_SHA256 = "6b2530cb9f3426c792500f07e854d7f31ad84081ac77104cb8032737234ff91c"
DATASET_CANONICAL_SHA256 = (
    "ee80a55874ac6d6cfd5ee32484d91113bff78d829d66c9ff46bcb646456eb598"
)
MANIFEST_FILE_SHA256 = (
    "9cb287cc2271ff136c59618d6d3a6c07255a65bc5576c4c7a9af8f5de8a63f16"
)
ORACLE_FILE_SHA256 = "72308bb34bda758cc72dc651e3f0fd2fd2bd1bff820479e2cf0774ee8d66cf5c"
ORACLE_CANONICAL_SHA256 = (
    "b877bcd6fe15767d9f1bb42a5840a799d2ef5a4a3691eb6a59ae2f9f7d40813b"
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

FACT_GROUNDING_TOKENS = {
    "wd4_03_e2": ("Fenwick tide gauge", "calibrated"),
    "wd4_03_e3": ("Fenwick rain gauge", "calibrated"),
    "wd4_03_e4": ("coral survey permit", "countersigned"),
    "wd4_09_e2": ("Indigo Harp prospectus", "was dispatched"),
    "wd4_09_e4": ("Indigo Harp", "no longer complete"),
    "wd4_10_e2": ("Borealis cistern", "68", "kilopascals"),
    "wd4_10_e3": ("Aurora cistern", "75", "kilopascals"),
    "wd4_10_e4": ("Borealis cistern", "71", "kilopascals"),
    "wd4_10_e5": ("Borealis cistern", "72", "kilopascals"),
}

WEEKDAY_DATE_PATTERN = re.compile(
    r"\b(?P<weekday>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) "
    r"(?P<date>\d{4}-\d{2}-\d{2})\b"
)

ACTIVE_INTENT_REFERENCE_TOKENS = {
    "wd4_02_e3": "Kestrel kiln",
    "wd4_04_e3": "Velvet Finch crate",
    "wd4_06_e3": "quill-sample reminder",
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


def _optional_payload_values(scenarios: list[Scenario]) -> set[str]:
    return {
        str(value).casefold()
        for scenario in scenarios
        for action in (*scenario.expected_actions, *scenario.forbidden_actions)
        for key, value in action.payload.items()
        if key in OPTIONAL_PAYLOAD_KEYS and key not in {"date", "quantity"}
    }


def _calendar_dates(scenarios: list[Scenario]) -> set[object]:
    values: set[object] = set()
    for scenario in scenarios:
        values.update({scenario.start_at.date(), scenario.end_at.date()})
        values.update(event.at.date() for event in scenario.events)
        for action in (*scenario.expected_actions, *scenario.forbidden_actions):
            values.update({action.window_start.date(), action.window_end.date()})
    return values


def _canonical_payload(candidate: DueCandidate) -> str:
    return json.dumps(
        dict(candidate.action_template.payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _w3_candidate_key(
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


def _assert_trigger_grounded(
    trigger: AtTrigger | RecurringTrigger | ConditionTransitionTrigger,
    event_at: datetime,
    text: str,
    scenario_timezone: str,
) -> None:
    if isinstance(trigger, AtTrigger):
        assert trigger.at.date().isoformat() in text
        assert trigger.at.strftime("%H:%M") in text
        return
    if isinstance(trigger, RecurringTrigger):
        assert trigger.start_date.isoformat() in text
        assert trigger.end_date.isoformat() in text
        assert trigger.local_time.strftime("%H:%M") in text
        assert trigger.timezone == scenario_timezone
        assert trigger.timezone in text
        assert all(
            weekday.casefold() in text.casefold() for weekday in trigger.weekdays
        )
        cursor = trigger.start_date
        matching_dates: list[date] = []
        while cursor <= trigger.end_date:
            if cursor.strftime("%A").casefold() in trigger.weekdays:
                matching_dates.append(cursor)
            cursor += timedelta(days=1)
        assert matching_dates
        assert all(
            item.strftime("%A").casefold() in trigger.weekdays
            for item in matching_dates
        )
        return
    assert trigger.active_from == event_at
    assert trigger.active_until.date().isoformat() in text
    assert trigger.active_until.strftime("%H:%M") in text
    assert trigger.active_until.isoformat()[-6:] in text


def _assert_named_weekday_date_pairs(text: str) -> int:
    matches = list(WEEKDAY_DATE_PATTERN.finditer(text))
    for match in matches:
        parsed = date.fromisoformat(match.group("date"))
        assert parsed.strftime("%A") == match.group("weekday")
    return len(matches)


def test_writer_diagnostic_v4_exact_bytes_and_hashes_are_frozen() -> None:
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
        "path": "eval/oracle/writer_diagnostic_memory_deltas.v4.json",
        "file_sha256": ORACLE_FILE_SHA256,
        "canonical_artifact_sha256": ORACLE_CANONICAL_SHA256,
        "annotation_policy": ORACLE_ANNOTATION_POLICY,
        "consumer_scope": "reporter-only-offline-replay",
        "event_record_count": 39,
        "visible_to_evaluated_writer": False,
        "writer_input_eligible": False,
        "human_annotation_measured": False,
        "nonempty_delta_event_count": 21,
        "mutation_count": 21,
        "mutation_counts": {
            "set_fact": 9,
            "create_intent": 9,
            "update_intent": 2,
            "cancel_intent": 1,
        },
        "offline_replay_ceiling": {
            "tp": 8,
            "fp": 0,
            "fn": 0,
            "provenance_exact": 8,
            "obsolete_errors": 0,
            "invalid_outputs": 0,
        },
    }


def test_writer_diagnostic_v4_scope_counts_families_and_blind_order() -> None:
    scenarios, manifest = _load()
    origins = manifest["scenario_origins"]
    assert isinstance(origins, list)
    scenario_order = [scenario.id for scenario in scenarios]
    origin_order = [item["scenario_id"] for item in origins]
    scenario_family_order: list[str] = []
    for scenario in scenarios:
        tagged_families = EXPECTED_FAMILIES.intersection(scenario.tags)
        assert len(tagged_families) == 1
        scenario_family_order.append(next(iter(tagged_families)))
    origin_family_order = [item["family"] for item in origins]
    family_counts = Counter(scenario_family_order)

    assert len(scenarios) == len(origins) == manifest["scenario_count"] == 10
    assert (
        sum(len(scenario.events) for scenario in scenarios)
        == manifest["event_count"]
        == 62
    )
    assert (
        sum(
            event.kind != "clock_tick"
            for scenario in scenarios
            for event in scenario.events
        )
        == manifest["non_clock_event_count"]
        == 39
    )
    assert (
        sum(len(scenario.expected_actions) for scenario in scenarios)
        == manifest["expected_action_count"]
        == 8
    )
    assert (
        sum(len(scenario.forbidden_actions) for scenario in scenarios)
        == manifest["forbidden_action_count"]
        == 13
    )
    assert (
        sum(not scenario.expected_actions for scenario in scenarios)
        == manifest["no_action_scenario_count"]
        == 2
    )
    assert (
        sum(
            any(action.reason == "obsolete" for action in scenario.forbidden_actions)
            for scenario in scenarios
        )
        == manifest["obsolete_trap_scenario_count"]
        == 3
    )
    assert (
        sum(
            key in OPTIONAL_PAYLOAD_KEYS
            for scenario in scenarios
            for action in (*scenario.expected_actions, *scenario.forbidden_actions)
            for key in action.payload
        )
        == manifest["optional_payload_slot_count"]
        == 34
    )
    assert scenario_order == origin_order
    assert scenario_family_order == origin_family_order
    assert len(origin_order) == len(set(origin_order))
    assert set(family_counts) == EXPECTED_FAMILIES
    assert set(family_counts.values()) == {1}
    assert manifest["family_counts"] == dict(sorted(family_counts.items()))
    assert manifest["status"] == "frozen-diagnostic"
    assert manifest["claim_scope"] == "diagnostic_development_only"
    assert manifest["hypothesis_evidence"] is False
    assert manifest["preregistered_final_eligible"] is False
    assert manifest["member_of_development_35"] is False
    assert manifest["member_of_sealed_set"] is False
    assert manifest["freeze_order"] == {
        "dataset_authored_before_w3_prompt": True,
        "oracle_authored_before_w3_prompt": True,
        "w3_prompt_existed_at_freeze": False,
        "w3_prompt_status": "not-authored",
        "w3_model_calls_before_freeze": 0,
    }
    assert manifest["review_status"] == {
        "automated_integrity": "passed",
        "gold_provenance_policy": "minimal-causal-evidence-v1",
        "independent_human_review": "pending",
        "independent_agent_review": "passed",
        "independent_agent_review_date": "2026-08-08",
        "independent_agent_review_note": (
            "Passed after two independent agent audits, correction of all findings, "
            "and a final clean post-fix audit."
        ),
    }
    assert manifest["origin"] == {
        "type": "locally-authored",
        "authorship": "brand-new-independent-v4",
        "derived_from_prior_writer_diagnostic": False,
        "longmemeval_items": 0,
        "triggerbench_items": 0,
        "external_dataset_items": 0,
    }
    assert all(
        item["origin"] == "locally-authored"
        and item["review_status"]
        == (
            "automated-integrity-passed; independent-agent-review-passed; "
            "independent-human-review-pending"
        )
        for item in origins
    )

    subjects = {
        scenario.id: {
            action.payload["subject"]
            for action in (*scenario.expected_actions, *scenario.forbidden_actions)
        }
        for scenario in scenarios
    }
    assert subjects == {
        scenario_id: {subject}
        for scenario_id, subject in manifest["canonical_action_subjects"][
            "by_scenario"
        ].items()
    }
    assert all(
        not {"a", "an", "the"}.intersection(subject.split())
        for values in subjects.values()
        for subject in values
    )
    assert manifest["canonical_action_subjects"]["policy"] == (
        "trimmed-lowercase-article-free-verb-object-v1"
    )


def test_writer_diagnostic_v4_has_new_ids_surfaces_entities_and_dates() -> None:
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
    assert {value.casefold() for value in new_surfaces}.isdisjoint(
        {value.casefold() for value in old_surfaces}
    )
    assert _optional_payload_values(scenarios).isdisjoint(
        _optional_payload_values(comparison)
    )
    assert _calendar_dates(scenarios).isdisjoint(_calendar_dates(comparison))
    assert manifest["anti_overlap_attestation"] == {
        "comparison_sets": [str(path.relative_to(ROOT)) for path in COMPARISON_PATHS],
        "scenario_ids_disjoint": True,
        "event_ids_disjoint": True,
        "canonical_record_hashes_disjoint": True,
        "exact_titles_descriptions_event_texts_disjoint": True,
        "casefolded_titles_descriptions_event_texts_disjoint": True,
        "optional_payload_values_disjoint_casefolded": True,
        "calendar_dates_disjoint": True,
        "automated_test": "tests/test_writer_diagnostic_v4.py",
        "old_contents_displayed_by_test": False,
        "manual_source_content_reviewed_for_authorship": False,
        "scope": "all pre-v4 core and writer-diagnostic scenario datasets",
    }


def test_writer_diagnostic_v4_timelines_gold_and_slot_grounding() -> None:
    scenarios, manifest = _load()
    optional_count = 0
    weekday_date_pair_count = 0

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
            weekday_date_pair_count += _assert_named_weekday_date_pairs(event.text)
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
            assert events_by_id[action.action_key].kind == "user_message"
            for evidence in action.acceptable_evidence_sets:
                assert action.action_key in evidence
                assert checkpoint_ids.intersection(evidence)
                evidence_text = "\n".join(events_by_id[item].text for item in evidence)
                for key, value in action.payload.items():
                    if key in OPTIONAL_PAYLOAD_KEYS:
                        optional_count += 1
                        assert str(value) in evidence_text

        for action in scenario.forbidden_actions:
            assert events_by_id[action.action_key].kind == "user_message"
            grounding_ids = {action.action_key, *action.related_event_ids}
            grounding_text = "\n".join(
                events_by_id[item].text for item in sorted(grounding_ids)
            )
            for key, value in action.payload.items():
                if key in OPTIONAL_PAYLOAD_KEYS:
                    optional_count += 1
                    assert str(value) in grounding_text

    assert optional_count == manifest["optional_payload_slot_count"] == 34
    assert (
        weekday_date_pair_count
        == manifest["causal_derivability"]["named_weekday_date_pair_count"]
        == 7
    )
    assert manifest["optional_payload_grounding"] == {
        "policy": "exact-source-cased-observable-text-v1",
        "all_optional_slots_explicitly_grounded": True,
        "automated_test": "tests/test_writer_diagnostic_v4.py",
    }


def test_writer_diagnostic_v4_mutations_are_prefix_derivable() -> None:
    scenarios, manifest = _load()
    artifact = load_oracle_artifact(ORACLE_PATH, scenarios)
    seen_fact_events: set[str] = set()
    seen_reference_events: set[str] = set()
    inherited_update_leaves: set[tuple[str, str, str]] = set()

    for scenario in scenarios:
        runtime = scenario.to_runtime()
        records = {record.event_id: record for record in artifact.records_for(runtime)}
        memory = InMemoryAnamnesis()

        for event in runtime.events:
            delta = records[event.id].delta if event.kind != "clock_tick" else None
            for mutation in delta.mutations if delta is not None else ():
                if isinstance(mutation, CreateIntent):
                    assert event.kind == "user_message"
                    _assert_trigger_grounded(
                        mutation.trigger,
                        event.at,
                        event.text,
                        scenario.timezone,
                    )
                    for condition in (
                        *mutation.required_conditions,
                        *mutation.blockers,
                    ):
                        assert condition.key.entity.replace("_", " ") in (
                            event.text.casefold()
                        )
                        assert condition.key.attribute in event.text.casefold()
                        if not isinstance(condition.value, bool):
                            assert str(condition.value) in event.text
                        if condition.unit is not None:
                            assert condition.unit in event.text
                    for key, value in mutation.action_template.payload.items():
                        if key in OPTIONAL_PAYLOAD_KEYS:
                            assert str(value) in event.text
                elif isinstance(mutation, SetFact):
                    tokens = FACT_GROUNDING_TOKENS[records[event.id].event_id]
                    assert all(
                        token.casefold() in event.text.casefold() for token in tokens
                    )
                    seen_fact_events.add(event.id)
                else:
                    assert isinstance(mutation, (UpdateIntent, CancelIntent))
                    active = [
                        intent
                        for intent in memory.current_intents
                        if intent.status == "active"
                    ]
                    assert len(active) == 1
                    assert active[0].intent_id == mutation.intent_id
                    reference = ACTIVE_INTENT_REFERENCE_TOKENS[event.id]
                    assert reference.casefold() in event.text.casefold()
                    seen_reference_events.add(event.id)
                    if isinstance(mutation, UpdateIntent):
                        if mutation.trigger is not None:
                            _assert_trigger_grounded(
                                mutation.trigger,
                                event.at,
                                event.text,
                                scenario.timezone,
                            )
                        if mutation.action_template is not None:
                            for key, value in mutation.action_template.payload.items():
                                if (
                                    key in OPTIONAL_PAYLOAD_KEYS
                                    and str(value) not in event.text
                                ):
                                    assert (
                                        active[0].action_template.payload.get(key)
                                        == value
                                    )
                                    inherited_update_leaves.add(
                                        (event.id, key, str(value))
                                    )

            applied = memory.ingest(event, delta)
            assert applied.accepted, (scenario.id, event.id, applied.error)
            memory.select(event)
            memory.commit(event, Decision())

    assert seen_fact_events == set(FACT_GROUNDING_TOKENS)
    assert seen_reference_events == set(ACTIVE_INTENT_REFERENCE_TOKENS)
    assert inherited_update_leaves == {("wd4_06_e3", "item", "quill samples")}
    assert manifest["causal_derivability"] == {
        "policy": "current-observable-plus-prior-active-state-v1",
        "all_gold_mutations_prefix_derivable": True,
        "future_or_gold_annotation_dependency": False,
        "recurrence_iana_timezone_explicit_in_observable": True,
        "mutation_weekday_date_pairs_explicit": True,
        "named_weekday_date_pairs_validated_against_iso_calendar": True,
        "named_weekday_date_pair_count": 7,
        "condition_active_until_explicit": True,
        "update_cancel_reference_one_active_intent": True,
        "optional_payload_values_exact_source_cased": True,
        "unchanged_update_leaves_may_derive_from_prior_active_state": True,
        "conditional_future_scope_not_asserted_as_current_fact": True,
        "summary_is_noncanonical": True,
        "automated_test": "tests/test_writer_diagnostic_v4.py",
    }


def test_writer_diagnostic_v4_conditional_future_scope_is_not_a_current_fact() -> None:
    scenarios, _ = _load()
    scenario = next(
        item for item in scenarios if item.id == "wd4_09_indigo_harp_reversal"
    )
    artifact = load_oracle_artifact(ORACLE_PATH, scenarios)
    runtime = scenario.to_runtime()
    records = {record.event_id: record for record in artifact.records_for(runtime)}
    first_delta = records["wd4_09_e1"].delta

    assert [mutation.op for mutation in first_delta.mutations] == ["create_intent"]
    create = first_delta.mutations[0]
    assert isinstance(create, CreateIntent)
    assert len(create.blockers) == 1

    memory = InMemoryAnamnesis()
    due_by_event: dict[str, int] = {}
    for event in runtime.events:
        delta = records[event.id].delta if event.kind != "clock_tick" else None
        applied = memory.ingest(event, delta)
        assert applied.accepted, (event.id, applied.error)
        if event.id == "wd4_09_e1":
            assert memory.current_facts == ()
            assert memory.evaluate_condition(create.blockers[0]) == TruthValue.UNKNOWN
        selection = memory.select(event)
        due_by_event[event.id] = len(selection.due_candidates)
        memory.commit(event, Decision())

    assert due_by_event["wd4_09_e3"] == 0
    assert due_by_event["wd4_09_e6"] == 1


def test_writer_diagnostic_v4_runtime_boundary_removes_author_fields() -> None:
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


async def _collect_due_candidates(
    scenarios: list[Scenario],
) -> list[tuple[str, DueCandidate]]:
    artifact = load_oracle_artifact(ORACLE_PATH, scenarios)
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


def test_writer_diagnostic_v4_public_w3_candidate_key_excludes_summary() -> None:
    scenarios, manifest = _load()
    due = asyncio.run(_collect_due_candidates(scenarios))
    w3_gate = Counter(_w3_candidate_key(checkpoint, item) for checkpoint, item in due)
    legacy = Counter(
        _legacy_candidate_key(checkpoint, item) for checkpoint, item in due
    )

    expected_count = manifest["expected_action_count"]
    assert sum(w3_gate.values()) == sum(legacy.values()) == expected_count == 8
    assert len(w3_gate) == len(legacy) == expected_count
    assert all(len(key) == 6 for key in w3_gate)
    assert all(len(key) == 7 for key in legacy)

    summary_variants: list[tuple[str, DueCandidate]] = []
    for index, (checkpoint, candidate) in enumerate(due):
        template = candidate.action_template.model_copy(
            update={"summary": f"Noncanonical rendering variant {index + 1}"}
        )
        summary_variants.append(
            (
                checkpoint,
                candidate.model_copy(update={"action_template": template}),
            )
        )
    assert (
        Counter(
            _w3_candidate_key(checkpoint, item) for checkpoint, item in summary_variants
        )
        == w3_gate
    )
    assert (
        Counter(
            _legacy_candidate_key(checkpoint, item)
            for checkpoint, item in summary_variants
        )
        != legacy
    )

    assert manifest["w3_candidate_matching_protocol"] == {
        "version": "w3.candidate-key.v1",
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


def test_writer_diagnostic_v4_oracle_is_reporter_only_and_complete() -> None:
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
    assert (
        sum(len(record.events) for record in artifact.scenarios)
        == manifest["oracle_reference"]["event_record_count"]
        == manifest["non_clock_event_count"]
        == 39
    )
    assert manifest["oracle_reference"]["consumer_scope"] == (
        "reporter-only-offline-replay"
    )
    assert manifest["oracle_reference"]["visible_to_evaluated_writer"] is False
    assert manifest["oracle_reference"]["writer_input_eligible"] is False

    mutations = [
        mutation
        for scenario in artifact.scenarios
        for record in scenario.events
        for mutation in record.delta.mutations
    ]
    assert (
        sum(
            bool(record.delta.mutations)
            for item in artifact.scenarios
            for record in item.events
        )
        == manifest["oracle_reference"]["nonempty_delta_event_count"]
        == 21
    )
    assert len(mutations) == manifest["oracle_reference"]["mutation_count"] == 21
    mutation_counts = Counter(mutation.op for mutation in mutations)
    assert mutation_counts == manifest["oracle_reference"]["mutation_counts"]
    assert mutation_counts == {
        "set_fact": 9,
        "create_intent": 9,
        "update_intent": 2,
        "cancel_intent": 1,
    }

    oracle_subjects: dict[str, set[str]] = {}
    for item in artifact.scenarios:
        for record in item.events:
            for mutation in record.delta.mutations:
                if isinstance(mutation, (CreateIntent, UpdateIntent)):
                    template = mutation.action_template
                    if template is not None:
                        oracle_subjects.setdefault(item.scenario_id, set()).add(
                            str(template.payload["subject"])
                        )
    expected_subjects = manifest["canonical_action_subjects"]["by_scenario"]
    assert oracle_subjects == {
        scenario_id: {subject}
        for scenario_id, subject in expected_subjects.items()
        if scenario_id != "wd4_07_sunken_orchard_quote"
    }

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


async def _replay_one(
    scenario: Scenario,
    scenarios: list[Scenario],
) -> ScenarioRun:
    artifact = load_oracle_artifact(ORACLE_PATH, scenarios)
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
        model="deterministic/writer-diagnostic-v4-oracle",
        prompt_version="offline.writer-oracle.v4",
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


def test_writer_diagnostic_v4_oracle_real_store_reaches_exact_ceiling() -> None:
    scenarios, manifest = _load()
    runs = [asyncio.run(_replay_one(scenario, scenarios)) for scenario in scenarios]
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
