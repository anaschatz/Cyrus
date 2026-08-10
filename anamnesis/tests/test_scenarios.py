"""Integrity, provenance, split-isolation, and leakage checks for v0 data."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from anamnesis.baselines import FullContextMemory
from anamnesis.io import canonical_sha256, dataset_sha256, load_scenarios
from anamnesis.prompts import build_decision_prompt
from anamnesis.schema import Scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "eval" / "scenarios"
SMOKE_PATH = SCENARIO_DIR / "smoke.jsonl"
DEV_PATH = SCENARIO_DIR / "dev.jsonl"
SEALED_PATH = SCENARIO_DIR / "sealed.jsonl"
ALL_PATH = SCENARIO_DIR / "all.jsonl"

FAMILIES = {
    "basic_deadline",
    "deadline_update",
    "fact_update",
    "conditional_trigger",
    "completion_cancellation",
    "recurring_intention",
    "negative_control",
}

PAYLOAD_KEYS = {
    "subject",
    "address",
    "build",
    "date",
    "flight",
    "greenhouse",
    "item",
    "project",
    "quantity",
    "recipient",
    "room",
    "shipment",
    "tank",
    "trip",
}

IMPERATIVE_SUBJECT_VERBS = {
    "back",
    "book",
    "call",
    "check",
    "collect",
    "confirm",
    "deliver",
    "email",
    "file",
    "inspect",
    "issue",
    "label",
    "notify",
    "order",
    "pack",
    "page",
    "pay",
    "print",
    "publish",
    "record",
    "refrigerate",
    "renew",
    "rent",
    "return",
    "review",
    "rotate",
    "send",
    "sign",
    "start",
    "submit",
    "tag",
    "take",
    "turn",
    "upload",
    "water",
}

EXPECTED_FAMILY_COUNTS = {
    "development": {
        "basic_deadline": 6,
        "deadline_update": 5,
        "fact_update": 5,
        "conditional_trigger": 6,
        "completion_cancellation": 4,
        "recurring_intention": 4,
        "negative_control": 5,
    },
    "sealed": {
        "basic_deadline": 2,
        "deadline_update": 3,
        "fact_update": 3,
        "conditional_trigger": 2,
        "completion_cancellation": 2,
        "recurring_intention": 2,
        "negative_control": 1,
    },
    "all": {
        "basic_deadline": 8,
        "deadline_update": 8,
        "fact_update": 8,
        "conditional_trigger": 8,
        "completion_cancellation": 6,
        "recurring_intention": 6,
        "negative_control": 6,
    },
}

EXPECTED_TAGS_BY_SMOKE_SCENARIO = {
    "s01_unmet_deadline": {"deadline", "distractors", "one_shot"},
    "s02_completion_suppresses_reminder": {
        "completion",
        "deadline",
        "no_action",
    },
    "s03_deadline_rescheduled": {
        "deadline_update",
        "stale_deadline",
        "supersession",
    },
    "s04_updated_action_parameter": {
        "contradictory_update",
        "provenance",
        "stale_parameter",
    },
    "s05_recurring_conditional": {
        "conditional",
        "mixed_action_no_action",
        "recurring",
    },
    "s06_conjunctive_trigger": {
        "conditional",
        "conjunction",
        "entity_grounding",
        "provenance",
    },
    "s07_negative_brainstorming": {
        "distractors",
        "false_alarm",
        "negative_control",
        "no_action",
    },
    "s08_threshold_trigger": {
        "conditional",
        "distractor_entity",
        "provenance",
        "threshold",
    },
    "s09_explicit_cancellation": {
        "cancellation",
        "entity_distractor",
        "no_action",
        "obsolete_memory",
    },
    "s10_similar_entities": {
        "deadline",
        "distractors",
        "entity_grounding",
        "negative_sub_intent",
    },
}


@pytest.fixture(scope="module")
def datasets() -> dict[str, list[Scenario]]:
    return {
        "smoke": load_scenarios(SMOKE_PATH),
        "development": load_scenarios(DEV_PATH),
        "sealed": load_scenarios(SEALED_PATH),
        "all": load_scenarios(ALL_PATH),
    }


def _manifest(split: str) -> dict[str, object]:
    name = "manifest.json" if split == "smoke" else f"{split}.manifest.json"
    return json.loads((SCENARIO_DIR / name).read_text(encoding="utf-8"))


def _is_timezone_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def test_partition_sizes_order_and_disjointness(
    datasets: dict[str, list[Scenario]],
) -> None:
    smoke = datasets["smoke"]
    development = datasets["development"]
    sealed = datasets["sealed"]
    all_scenarios = datasets["all"]

    assert len(smoke) == 10
    assert len(development) == 35
    assert len(sealed) == 15
    assert len(all_scenarios) == 50

    smoke_ids = [scenario.id for scenario in smoke]
    development_ids = [scenario.id for scenario in development]
    sealed_ids = [scenario.id for scenario in sealed]
    all_ids = [scenario.id for scenario in all_scenarios]

    assert development_ids[:10] == smoke_ids
    assert set(development_ids).isdisjoint(sealed_ids)
    assert all_ids == development_ids + sealed_ids
    assert [item.model_dump() for item in all_scenarios] == [
        *[item.model_dump() for item in development],
        *[item.model_dump() for item in sealed],
    ]


def test_family_allocation_matches_preregistered_plan(
    datasets: dict[str, list[Scenario]],
) -> None:
    for split in ("development", "sealed", "all"):
        manifest = _manifest(split if split != "development" else "dev")
        scenario_origins = manifest["scenario_origins"]
        counts = Counter(item["family"] for item in scenario_origins)

        assert set(counts) == FAMILIES
        assert dict(counts) == EXPECTED_FAMILY_COUNTS[split]
        assert manifest["family_counts"] == EXPECTED_FAMILY_COUNTS[split]
        assert len(scenario_origins) == len(datasets[split])


def test_dataset_meets_no_action_and_obsolete_trap_minima(
    datasets: dict[str, list[Scenario]],
) -> None:
    scenarios = datasets["all"]
    no_action_count = sum(not scenario.expected_actions for scenario in scenarios)
    obsolete_trap_count = sum(
        any(item.reason == "obsolete" for item in scenario.forbidden_actions)
        for scenario in scenarios
    )

    assert no_action_count == 22
    assert no_action_count >= 15
    assert obsolete_trap_count == 20
    assert obsolete_trap_count >= 20


def test_manifests_freeze_hashes_counts_origin_and_review_status(
    datasets: dict[str, list[Scenario]],
) -> None:
    for dataset_key, manifest_key in (
        ("development", "dev"),
        ("sealed", "sealed"),
        ("all", "all"),
    ):
        scenarios = datasets[dataset_key]
        manifest = _manifest(manifest_key)

        assert manifest["status"] == "frozen-candidate-pending-independent-review"
        assert manifest["dataset_version"] == "v0.1.2"
        assert manifest["review_status"] == {
            "automated_integrity": "passed",
            "gold_provenance_policy": "minimal-causal-evidence-v1",
            "independent_human_review": "pending",
        }
        assert manifest["origin"]["type"] == "locally-authored"
        assert manifest["origin"]["longmemeval_items"] == 0
        assert manifest["origin"]["triggerbench_items"] == 0
        assert manifest["scenario_count"] == len(scenarios)
        assert manifest["event_count"] == sum(
            len(scenario.events) for scenario in scenarios
        )
        assert manifest["expected_action_count"] == sum(
            len(scenario.expected_actions) for scenario in scenarios
        )
        assert manifest["forbidden_action_count"] == sum(
            len(scenario.forbidden_actions) for scenario in scenarios
        )
        assert manifest["canonical_dataset_sha256"] == dataset_sha256(scenarios)
        assert manifest["record_sha256"] == {
            scenario.id: canonical_sha256(scenario) for scenario in scenarios
        }
        assert all(
            item["origin"] == "locally-authored"
            and item["review_status"]
            == "automated-integrity-passed; independent-review-pending"
            for item in manifest["scenario_origins"]
        )

    all_manifest = _manifest("all")
    assert all_manifest["component_datasets"] == {
        "development": {
            "path": "dev.jsonl",
            "canonical_dataset_sha256": _manifest("dev")["canonical_dataset_sha256"],
        },
        "sealed": {
            "path": "sealed.jsonl",
            "canonical_dataset_sha256": _manifest("sealed")["canonical_dataset_sha256"],
        },
    }


def test_smoke_manifest_and_existing_fixture_contract(
    datasets: dict[str, list[Scenario]],
) -> None:
    smoke = datasets["smoke"]
    manifest = _manifest("smoke")

    assert sum(len(scenario.events) for scenario in smoke) == 78
    assert sum(len(scenario.expected_actions) for scenario in smoke) == 8
    assert sum(len(scenario.forbidden_actions) for scenario in smoke) == 19
    assert sum(not scenario.expected_actions for scenario in smoke) == 3
    assert manifest["status"] == "development-smoke-only"
    assert manifest["hypothesis_evidence"] is False
    assert manifest["canonical_dataset_sha256"] == dataset_sha256(smoke)
    assert {scenario.id: set(scenario.tags) for scenario in smoke} == (
        EXPECTED_TAGS_BY_SMOKE_SCENARIO
    )


def test_s01_s03_explicit_non_completion_provenance_is_consistent(
    datasets: dict[str, list[Scenario]],
) -> None:
    by_id = {scenario.id: scenario for scenario in datasets["smoke"]}
    s01 = by_id["s01_unmet_deadline"].expected_actions[0]
    s03 = by_id["s03_deadline_rescheduled"].expected_actions[0]

    assert s01.acceptable_evidence_sets == [["s01-e01", "s01-e05", "s01-e07"]]
    assert s03.acceptable_evidence_sets == [
        ["s03-e01", "s03-e02", "s03-e03", "s03-e07"]
    ]


def test_explicit_facts_are_evidence_but_unknown_blockers_are_not(
    datasets: dict[str, list[Scenario]],
) -> None:
    by_id = {scenario.id: scenario for scenario in datasets["all"]}

    assert by_id["s10_similar_entities"].expected_actions[
        0
    ].acceptable_evidence_sets == [["s10-e01", "s10-e04", "s10-e06"]]
    assert by_id["s36_deliver_costume_deposit"].expected_actions[
        0
    ].acceptable_evidence_sets == [["s36-e01", "s36-e03", "s36-e05"]]

    # No event asserts the Tuesday blocker in s05, or the due-occurrence
    # blockers in s29/s48, so the minimal trace contains no invented source.
    assert by_id["s05_recurring_conditional"].expected_actions[
        0
    ].acceptable_evidence_sets == [["s05-e01", "s05-e05"]]
    assert by_id["s29_alternating_plant_watering"].expected_actions[
        0
    ].acceptable_evidence_sets == [["s29-e01", "s29-e04"]]
    assert by_id["s48_recurring_freezer_check"].expected_actions[
        0
    ].acceptable_evidence_sets == [["s48-e01", "s48-e04"]]


def test_payloads_follow_the_closed_canonical_grammar(
    datasets: dict[str, list[Scenario]],
) -> None:
    for scenario in datasets["all"]:
        local_zone = ZoneInfo(scenario.timezone)
        for action in [
            *scenario.expected_actions,
            *scenario.forbidden_actions,
        ]:
            assert set(action.payload) <= PAYLOAD_KEYS
            assert "subject" in action.payload

            subject = action.payload["subject"]
            assert isinstance(subject, str)
            assert subject == subject.strip() == subject.lower()
            words = subject.split()
            assert len(words) >= 2
            assert words[0] in IMPERATIVE_SUBJECT_VERBS

            if occurrence_date := action.payload.get("date"):
                assert isinstance(occurrence_date, str)
                parsed_date = date.fromisoformat(occurrence_date)
                assert parsed_date.isoformat() == occurrence_date
                assert (
                    occurrence_date
                    == action.window_start.astimezone(local_zone).date().isoformat()
                )


def test_every_recurring_occurrence_uses_an_iso_date(
    datasets: dict[str, list[Scenario]],
) -> None:
    recurring_scenarios = [
        scenario for scenario in datasets["all"] if "recurring" in scenario.tags
    ]
    assert len(recurring_scenarios) == 6

    for scenario in recurring_scenarios:
        for action in [
            *scenario.expected_actions,
            *scenario.forbidden_actions,
        ]:
            assert "date" in action.payload
            assert "weekday" not in action.payload


def test_forbidden_signatures_are_unique_and_taxonomy_is_consistent(
    datasets: dict[str, list[Scenario]],
) -> None:
    by_id = {scenario.id: scenario for scenario in datasets["all"]}

    for scenario in datasets["all"]:
        signatures = [
            (
                action.action_key,
                action.kind,
                json.dumps(action.payload, sort_keys=True),
                action.window_start,
                action.window_end,
            )
            for action in scenario.forbidden_actions
        ]
        assert len(signatures) == len(set(signatures))

        events_by_id = {event.id: event for event in scenario.events}
        for action in scenario.forbidden_actions:
            if action.reason != "obsolete":
                continue
            related_ids = {action.action_key, *action.related_event_ids}
            assert any(
                set(events_by_id[event_id].supersedes) & related_ids
                for event_id in action.related_event_ids
            )

    s02 = by_id["s02_completion_suppresses_reminder"]
    assert next(event for event in s02.events if event.id == "s02-e03").supersedes == []
    assert [action.reason for action in s02.forbidden_actions] == [
        "condition_satisfied"
    ]

    s28 = by_id["s28_completion_reaffirmed"]
    assert len(s28.forbidden_actions) == 1
    assert s28.forbidden_actions[0].reason == "condition_satisfied"

    s47 = by_id["s47_completed_at_deadline"]
    assert {action.reason for action in s47.forbidden_actions} == {
        "condition_satisfied"
    }


def test_scenarios_cover_exactly_seven_local_calendar_days(
    datasets: dict[str, list[Scenario]],
) -> None:
    for scenario in datasets["all"]:
        local_zone = ZoneInfo(scenario.timezone)
        start = scenario.start_at.astimezone(local_zone)
        end = scenario.end_at.astimezone(local_zone)

        assert scenario.timezone == "Europe/Athens"
        assert _is_timezone_aware(scenario.start_at)
        assert _is_timezone_aware(scenario.end_at)
        assert start.weekday() == 0
        assert end.weekday() == 6
        assert (end.date() - start.date()).days + 1 == 7

        event_timestamps = [event.at for event in scenario.events]
        assert len(event_timestamps) == len(set(event_timestamps))
        assert all(event.kind != "assistant_decision" for event in scenario.events)

        timestamps = [event.at for event in scenario.events]
        timestamps.extend(
            timestamp
            for action in scenario.expected_actions
            for timestamp in (action.window_start, action.window_end)
        )
        timestamps.extend(
            timestamp
            for action in scenario.forbidden_actions
            for timestamp in (action.window_start, action.window_end)
        )
        for timestamp in timestamps:
            assert _is_timezone_aware(timestamp)
            assert scenario.start_at <= timestamp <= scenario.end_at
            assert timestamp.utcoffset() == timestamp.astimezone(local_zone).utcoffset()


def test_all_fixture_ids_are_globally_unique(
    datasets: dict[str, list[Scenario]],
) -> None:
    scenarios = datasets["all"]
    scenario_ids = [scenario.id for scenario in scenarios]
    event_ids = [event.id for scenario in scenarios for event in scenario.events]
    expected_ids = [
        action.id for scenario in scenarios for action in scenario.expected_actions
    ]
    forbidden_ids = [
        action.id for scenario in scenarios for action in scenario.forbidden_actions
    ]

    assert len(scenario_ids) == len(set(scenario_ids))
    assert len(event_ids) == len(set(event_ids))
    assert len(expected_ids) == len(set(expected_ids))
    assert len(forbidden_ids) == len(set(forbidden_ids))
    assert len(event_ids + expected_ids + forbidden_ids) == len(
        set(event_ids + expected_ids + forbidden_ids)
    )


def test_action_and_provenance_references_are_minimal_and_checkpoint_exact(
    datasets: dict[str, list[Scenario]],
) -> None:
    alternate_evidence_actions = 0

    for scenario in datasets["all"]:
        events_by_id = {event.id: event for event in scenario.events}
        for action in scenario.expected_actions:
            trigger_ids = {
                event.id
                for event in scenario.events
                if action.window_start <= event.at <= action.window_end
            }
            assert action.window_start == action.window_end
            assert len(trigger_ids) == 1
            assert action.action_key in events_by_id
            assert events_by_id[action.action_key].kind == "user_message"
            assert len(action.acceptable_evidence_sets) == len(
                {tuple(item) for item in action.acceptable_evidence_sets}
            )
            alternate_evidence_actions += len(action.acceptable_evidence_sets) > 1

            for evidence_set in action.acceptable_evidence_sets:
                assert action.action_key in evidence_set
                assert trigger_ids.intersection(evidence_set)
                assert set(evidence_set) <= events_by_id.keys()
                assert all(
                    events_by_id[evidence_id].at <= action.window_end
                    for evidence_id in evidence_set
                )
                assert not any(
                    set(other) < set(evidence_set)
                    for other in action.acceptable_evidence_sets
                )

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

    assert alternate_evidence_actions >= 1


def test_supersedes_only_references_already_observed_events(
    datasets: dict[str, list[Scenario]],
) -> None:
    assert any(
        event.supersedes for scenario in datasets["all"] for event in scenario.events
    )
    for scenario in datasets["all"]:
        observed: set[str] = set()
        for event in scenario.events:
            assert set(event.supersedes) <= observed
            observed.add(event.id)


def test_sanitized_events_and_incremental_prompts_never_leak_gold(
    datasets: dict[str, list[Scenario]],
) -> None:
    hidden_field_names = {
        "acceptable_evidence_sets",
        "expected_actions",
        "forbidden_actions",
        "related_event_ids",
        "supersedes",
        "tags",
    }

    for scenario in datasets["all"]:
        memory = FullContextMemory()
        memory.reset()

        for index, authored_event in enumerate(scenario.events):
            event = authored_event.to_observable()
            assert set(type(event).model_fields) == {"id", "at", "kind", "text"}
            asyncio.run(memory.ingest(event))
            selection = memory.select(event)
            prompt = build_decision_prompt(
                now=event.at.isoformat(),
                current_event_id=event.id,
                context_events=selection.events,
                decision_history=selection.decisions,
                memory_view=selection.memory_view,
            )

            assert [selected.id for selected in selection.events] == [
                observed.id for observed in scenario.events[: index + 1]
            ]
            assert f"Current decision event: {event.id}" in prompt
            for observed in scenario.events[: index + 1]:
                assert f"[{observed.id}]" in prompt
                assert observed.text in prompt
            for future in scenario.events[index + 1 :]:
                assert f"[{future.id}]" not in prompt
                assert future.text not in prompt

            lowered_prompt = prompt.lower()
            assert all(name not in lowered_prompt for name in hidden_field_names)
            assert scenario.title not in prompt
            assert scenario.description not in prompt
            for action in [
                *scenario.expected_actions,
                *scenario.forbidden_actions,
            ]:
                assert action.id not in prompt


def test_development_task_default_cannot_select_sealed_or_all_dataset() -> None:
    task_source = (ROOT / "eval" / "anamnesis_eval.py").read_text(encoding="utf-8")
    dataset_defaults = re.findall(
        r'dataset:\s*DatasetSplit\s*=\s*"([^\"]+)"',
        task_source,
    )
    assert dataset_defaults
    assert set(dataset_defaults) == {"development"}
    assert '"development": ("dev.jsonl"' in task_source
    assert '"sealed":' not in task_source
