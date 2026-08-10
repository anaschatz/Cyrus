from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from anamnesis.io import dataset_sha256, load_scenarios
from anamnesis.local_preflight import local_preflight_event
from anamnesis.local_runtime import (
    LOCAL_DECISION_VERSION,
    LocalDecisionWire,
    build_local_decision_prompt,
    local_decision_prompt_contract,
    local_decision_schema_contract,
    local_memory_compiler_schema_contract,
)
from anamnesis.local_wire import (
    LOCAL_MEMORY_COMPILER_W3_ADDENDUM,
    LOCAL_MEMORY_COMPILER_W3_INSTRUCTIONS,
    LOCAL_MEMORY_COMPILER_W3_VERSION,
    LocalMemoryDeltaWire,
    build_local_memory_compiler_w3_prompt,
    local_memory_compiler_w3_contract,
)
from anamnesis.oracle import load_oracle_artifact, oracle_artifact_sha256
from anamnesis.schema import OPTIONAL_PAYLOAD_KEYS, MemoryView, ObservableEvent

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "eval" / "preflight" / "local_writer_w3.v1.json"
PROTOCOL_PATH = ROOT / "eval" / "preflight" / "local_writer_w3.protocol.v1.json"
W2_FIXTURE_PATH = ROOT / "eval" / "preflight" / "local_writer_w2.v1.json"
V4_DATASET_PATH = ROOT / "eval" / "scenarios" / "writer_diagnostic.v4.jsonl"
V4_ORACLE_PATH = ROOT / "eval" / "oracle" / "writer_diagnostic_memory_deltas.v4.json"
FIXTURE_SHA256 = "5628c3c1d7f8e1a5da43d6e567d55ac8e4fbabd8b9c4054325de6f4def1da30c"
PROTOCOL_SHA256 = "7f63c156a8af74ced2d5e5530b3e8083da95c7f46e14e1afafaaf864b3ce1915"
PROMPT_COMMIT = "a9fb1602158c4545f6791c296dfc05d8decc7d90"
EMPTY_ACTIVE_STATE = '{"facts":[],"intents":[]}'
V4_DATASET_RAW_SHA256 = (
    "6b2530cb9f3426c792500f07e854d7f31ad84081ac77104cb8032737234ff91c"
)
V4_DATASET_CANONICAL_SHA256 = (
    "ee80a55874ac6d6cfd5ee32484d91113bff78d829d66c9ff46bcb646456eb598"
)
V4_ORACLE_RAW_SHA256 = (
    "72308bb34bda758cc72dc651e3f0fd2fd2bd1bff820479e2cf0774ee8d66cf5c"
)
V4_ORACLE_CANONICAL_SHA256 = (
    "b877bcd6fe15767d9f1bb42a5840a799d2ef5a4a3691eb6a59ae2f9f7d40813b"
)

ORDERED_CASES = (
    ("C1", "compiler", "normalization_fact"),
    ("C2", "compiler", "bare_weekday_at"),
    ("C3", "compiler", "condition_transition_and"),
    ("C4", "compiler", "recurrence_iana_range"),
    ("C5", "compiler", "stable_id_trigger_update"),
    ("C6", "compiler", "full_action_template_update"),
    ("C7", "compiler", "complete_sparse_payload_including_zero"),
    ("C8", "compiler", "ambiguous_empty"),
    ("D1", "decision", "no_action"),
)

FRESH_MARKERS = (
    "sapphire observatory",
    "aurora spectrometer",
    "north vault",
    "cobalt beacon",
    "lunar archive",
    "zephyr antenna",
    "polaris lens",
    "meridian capsule",
    "aster relay",
    "silver comet",
    "quasar room",
    "prism samples",
    "basalt ridge",
)


def _load(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compiler_cases() -> list[dict[str, Any]]:
    cases = _load()["compiler_cases"]
    assert isinstance(cases, list)
    return cases


def _compiler_case(case_id: str) -> dict[str, Any]:
    return next(case for case in _compiler_cases() if case["id"] == case_id)


def _wire_example(case: dict[str, Any]) -> dict[str, Any]:
    example = case["valid_wire_example"]
    assert isinstance(example, dict)
    return example


def _single_create(candidate: dict[str, Any]) -> dict[str, Any]:
    creates = candidate["intent_creates"]
    assert isinstance(creates, list) and len(creates) == 1
    create = creates[0]
    assert isinstance(create, dict)
    return create


def _single_update(candidate: dict[str, Any]) -> dict[str, Any]:
    updates = candidate["intent_updates"]
    assert isinstance(updates, list) and len(updates) == 1
    update = updates[0]
    assert isinstance(update, dict)
    return update


def _action_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    creates = candidate["intent_creates"]
    updates = candidate["intent_updates"]
    if creates:
        mutation = _single_create(candidate)
    else:
        assert updates
        mutation = _single_update(candidate)
    template = mutation["action_template"]
    assert isinstance(template, dict)
    payload = template["payload"]
    assert isinstance(payload, dict)
    return payload


def _compiler_prompt(case: dict[str, Any]) -> str:
    case_input = case["input"]
    assert set(case_input) == {"event", "active_state"}
    return build_local_memory_compiler_w3_prompt(
        event=ObservableEvent.model_validate(case_input["event"]),
        active_state=case_input["active_state"],
    )


def _decision_prompt(case: dict[str, Any]) -> str:
    case_input = case["input"]
    assert set(case_input) == {
        "now",
        "current_event_id",
        "context_events",
        "decision_history",
        "memory_view",
    }
    return build_local_decision_prompt(
        now=case_input["now"],
        current_event_id=case_input["current_event_id"],
        context_events=[
            ObservableEvent.model_validate(event)
            for event in case_input["context_events"]
        ],
        decision_history=case_input["decision_history"],
        memory_view=MemoryView.model_validate(case_input["memory_view"]),
    )


def _canonical_multiset(values: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values
    )


def _assert_trigger_projection(
    actual: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    expected = acceptance["trigger"]
    assert isinstance(expected, dict)
    if acceptance.get("weekdays_match") != "set":
        assert actual == expected, "trigger"
        return

    actual_without_weekdays = {
        key: value for key, value in actual.items() if key != "weekdays"
    }
    expected_without_weekdays = {
        key: value for key, value in expected.items() if key != "weekdays"
    }
    assert actual_without_weekdays == expected_without_weekdays, "trigger"
    assert set(actual["weekdays"]) == set(expected["weekdays"]), "trigger"


def _validate_compiler_projection(
    case: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    parsed = LocalMemoryDeltaWire.model_validate(candidate)
    delta = parsed.to_domain()
    acceptance = case["acceptance"]
    assert isinstance(acceptance, dict)
    mutations = delta.mutations
    assert len(mutations) == acceptance["mutation_count"], "mutation_count"

    if acceptance["mutation_type"] == "empty_delta":
        assert not mutations, "empty_delta"
        return delta.model_dump(mode="json")

    mutation = mutations[0]
    dumped = mutation.model_dump(mode="json")
    assert dumped["op"] == acceptance["mutation_type"], "mutation_type"

    if acceptance["mutation_type"] == "set_fact":
        assert dumped["key"] == acceptance["key"], "key"
        assert dumped["value"] == acceptance["value"], "value"
        assert type(dumped["value"]) is type(acceptance["value"]), "value_type"
        assert dumped["unit"] == acceptance["unit"], "unit"
        return delta.model_dump(mode="json")

    if acceptance["mutation_type"] == "create_intent":
        _assert_trigger_projection(dumped["trigger"], acceptance)
        if acceptance.get("required_conditions_match") == "canonical_multiset":
            assert _canonical_multiset(dumped["required_conditions"]) == (
                _canonical_multiset(acceptance["required_conditions"])
            ), "required_conditions"
        else:
            assert dumped["required_conditions"] == acceptance["required_conditions"], (
                "required_conditions"
            )
        assert dumped["blockers"] == acceptance["blockers"], "blockers"
        template = dumped["action_template"]
        assert template["kind"] == acceptance["action_template"]["kind"], "kind"
        assert template["payload"] == acceptance["action_template"]["payload"], (
            "payload"
        )
        assert acceptance["intent_id"] == "structural_only"
        assert acceptance["summary"] == "structural_only"
        assert isinstance(dumped["intent_id"], str) and dumped["intent_id"]
        assert isinstance(template["summary"], str) and template["summary"]
        return delta.model_dump(mode="json")

    assert acceptance["mutation_type"] == "update_intent"
    assert dumped["intent_id"] == acceptance["intent_id"], "intent_id"
    assert mutation.model_fields_set == {
        "intent_id",
        *acceptance["changed_fields"],
    }, "changed_fields"
    if acceptance["changed_fields"] == ["trigger"]:
        assert dumped["trigger"] == acceptance["trigger"], "trigger"
    else:
        assert acceptance["changed_fields"] == ["action_template"]
        assert dumped["action_template"] == acceptance["action_template"], (
            "action_template"
        )
    return delta.model_dump(mode="json")


def _with_all_unused_payload_slots_null(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    changed = copy.deepcopy(candidate)
    payload = _action_payload(changed)
    for key in OPTIONAL_PAYLOAD_KEYS - payload.keys():
        payload[key] = None
    return changed


def _with_alternative_structural_create(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    changed = copy.deepcopy(candidate)
    create = _single_create(changed)
    create["intent_id"] = "alternate_fixture_action"
    template = create["action_template"]
    assert isinstance(template, dict)
    template["summary"] = "A schema-valid alternative summary."
    return changed


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _collect_ids(value: Any) -> set[str]:
    collected: set[str] = set()

    def visit(current: Any, key: str | None = None) -> None:
        if isinstance(current, dict):
            for child_key, child in current.items():
                visit(child, child_key)
            return
        if isinstance(current, list):
            for child in current:
                visit(child, key)
            return
        if not isinstance(current, str) or key is None:
            return
        if (
            key == "id"
            or key.endswith("_id")
            or key
            in {
                "action_key",
                "evidence_event_ids",
                "acceptable_evidence_sets",
                "related_event_ids",
            }
        ):
            collected.add(current)

    visit(value)
    return collected


def _collect_dates(value: Any) -> set[str]:
    collected: set[str] = set()

    def visit(current: Any) -> None:
        if isinstance(current, dict):
            for child in current.values():
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)
        elif isinstance(current, str):
            collected.update(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", current))

    visit(value)
    return collected


def _collect_actions(value: Any) -> set[str]:
    collected: set[str] = set()

    def visit(current: Any) -> None:
        if isinstance(current, dict):
            payload = current.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("subject"), str):
                collected.add(_canonical_json(payload))
                summary = current.get("summary")
                if isinstance(summary, str):
                    collected.add(
                        _canonical_json(
                            {
                                "kind": current.get("kind", "reminder"),
                                "payload": payload,
                                "summary": summary,
                            }
                        )
                    )
            for child in current.values():
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return collected


def _fixture_custodian_material(fixture: dict[str, Any]) -> list[Any]:
    material: list[Any] = []
    for case in fixture["compiler_cases"]:
        material.extend((case["input"], case["valid_wire_example"]))
        material.append(json.loads(case["input"]["active_state"]))
    for case in fixture["decision_cases"]:
        material.extend((case["input"], case["valid_wire_example"]))
    return material


def _fixture_full_inputs(fixture: dict[str, Any]) -> set[str]:
    inputs: set[str] = set()
    for case in fixture["compiler_cases"]:
        inputs.add(_canonical_json(case["input"]))
        inputs.add(_canonical_json(case["input"]["event"]))
        inputs.add(case["input"]["event"]["text"])
    for case in fixture["decision_cases"]:
        inputs.add(_canonical_json(case["input"]))
        for event in case["input"]["context_events"]:
            inputs.add(_canonical_json(event))
            inputs.add(event["text"])
    return inputs


def _v4_full_inputs(scenarios: list[Any]) -> set[str]:
    inputs: set[str] = set()
    for scenario in scenarios:
        runtime_events: list[dict[str, Any]] = []
        for event in scenario.events:
            if event.kind == "assistant_decision":
                continue
            observable = event.to_observable().model_dump(mode="json")
            runtime_events.append(observable)
            inputs.add(_canonical_json(observable))
            inputs.add(observable["text"])
        inputs.add(_canonical_json({"id": scenario.id, "events": runtime_events}))
    return inputs


def _require_disjoint(left: set[str], right: set[str], dimension: str) -> None:
    if not left.isdisjoint(right):
        pytest.fail(
            f"non-disclosing custodian {dimension} disjointness audit failed",
            pytrace=False,
        )


def test_local_w3_fixture_bytes_are_frozen() -> None:
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_fixture_was_authored_after_the_frozen_prompt_commit() -> None:
    fixture = _load()
    authorship = fixture["authorship"]
    assert authorship == {
        "author_role": "separate_v4_blind_fixture_author",
        "prompt_commit": PROMPT_COMMIT,
        "protocol_id": "local_writer_w3.protocol.v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "fixture_authored_after_prompt_commit": True,
        "case_material_source": "frozen_protocol_only",
        "v4_scenario_and_oracle_blind": True,
        "model_boundary": "case_input_only",
    }
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == PROTOCOL_SHA256

    commit_exists = subprocess.run(
        ["git", "cat-file", "-e", f"{PROMPT_COMMIT}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert commit_exists.returncode == 0
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROMPT_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert ancestor.returncode == 0
    fixture_at_prompt_commit = subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            f"{PROMPT_COMMIT}:eval/preflight/local_writer_w3.v1.json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert fixture_at_prompt_commit.returncode != 0


def test_completed_custodian_audit_is_hash_pinned_and_disjoint() -> None:
    fixture = _load()
    audit = fixture["custodian_audit"]
    assert audit == {
        "status": "passed",
        "method": "non_disclosing_hash_and_disjointness_assertions",
        "dataset_version": "v4",
        "dataset": {
            "raw_sha256": V4_DATASET_RAW_SHA256,
            "canonical_sha256": V4_DATASET_CANONICAL_SHA256,
        },
        "oracle": {
            "raw_sha256": V4_ORACLE_RAW_SHA256,
            "canonical_sha256": V4_ORACLE_CANONICAL_SHA256,
        },
        "disjoint_dimensions": ["full_inputs", "ids", "dates", "actions"],
        "model_boundary_unchanged": True,
    }

    assert hashlib.sha256(V4_DATASET_PATH.read_bytes()).hexdigest() == (
        V4_DATASET_RAW_SHA256
    )
    scenarios = load_scenarios(V4_DATASET_PATH)
    assert dataset_sha256(scenarios) == V4_DATASET_CANONICAL_SHA256
    assert hashlib.sha256(V4_ORACLE_PATH.read_bytes()).hexdigest() == (
        V4_ORACLE_RAW_SHA256
    )
    oracle = load_oracle_artifact(V4_ORACLE_PATH, scenarios)
    assert oracle_artifact_sha256(oracle) == V4_ORACLE_CANONICAL_SHA256

    fixture_material = _fixture_custodian_material(fixture)
    v4_material = [
        *(scenario.model_dump(mode="json") for scenario in scenarios),
        oracle.model_dump(mode="json"),
    ]
    _require_disjoint(
        _fixture_full_inputs(fixture),
        _v4_full_inputs(scenarios),
        "full_inputs",
    )
    _require_disjoint(
        _collect_ids(fixture_material),
        _collect_ids(v4_material),
        "ids",
    )
    _require_disjoint(
        _collect_dates(fixture_material),
        _collect_dates(v4_material),
        "dates",
    )
    _require_disjoint(
        _collect_actions(fixture_material),
        _collect_actions(v4_material),
        "actions",
    )


def test_fixture_scope_order_and_projections_match_the_frozen_protocol() -> None:
    fixture = _load()
    protocol = _load(PROTOCOL_PATH)
    assert set(fixture) == {
        "schema_version",
        "fixture_id",
        "purpose",
        "hypothesis_test_eligible",
        "authorship",
        "custodian_audit",
        "contracts",
        "compiler_cases",
        "decision_cases",
    }
    assert fixture["schema_version"] == 1
    assert fixture["fixture_id"] == "local_writer_w3.v1"
    assert fixture["purpose"] == "diagnostic"
    assert fixture["hypothesis_test_eligible"] is False

    cases = [*fixture["compiler_cases"], *fixture["decision_cases"]]
    protocol_cases = protocol["preflight"]["ordered_categories"]
    assert [(case["id"], case["role"], case["category"]) for case in cases] == list(
        ORDERED_CASES
    )
    assert len(fixture["compiler_cases"]) == 8
    assert len(fixture["decision_cases"]) == 1
    for case, protocol_case in zip(cases, protocol_cases, strict=True):
        assert case["id"] == protocol_case["id"]
        assert case["role"] == protocol_case["role"]
        assert case["category"] == protocol_case["category"]
        assert case["acceptance_projection"] == protocol_case["acceptance_projection"]
        assert set(case) == {
            "id",
            "role",
            "category",
            "input",
            "valid_wire_example",
            "valid_domain_example",
            "acceptance_projection",
            "acceptance",
        }


def test_fixture_contract_hashes_match_w3_and_the_unchanged_decision_contract() -> None:
    fixture = _load()
    protocol = _load(PROTOCOL_PATH)
    contracts = fixture["contracts"]
    assert contracts == {
        "compiler": protocol["contracts"]["compiler"],
        "decision": protocol["contracts"]["decision"],
    }

    sentinel = ObservableEvent(
        id="<event-id>",
        at="2000-01-01T00:00:00+00:00",
        kind="user_message",
        text="<event-text>",
    )
    compiler_prompt = build_local_memory_compiler_w3_prompt(
        event=sentinel,
        active_state=EMPTY_ACTIVE_STATE,
    )
    model_schema = json.dumps(
        LocalMemoryDeltaWire.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    )
    assert contracts["compiler"] == {
        "prompt_version": LOCAL_MEMORY_COMPILER_W3_VERSION,
        "addendum_sha256": hashlib.sha256(
            LOCAL_MEMORY_COMPILER_W3_ADDENDUM.encode()
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(compiler_prompt.encode()).hexdigest(),
        "local_wire_contract_sha256": hashlib.sha256(
            local_memory_compiler_w3_contract().encode()
        ).hexdigest(),
        "local_wire_model_schema_sha256": hashlib.sha256(
            model_schema.encode()
        ).hexdigest(),
        "inspect_response_schema_sha256": hashlib.sha256(
            local_memory_compiler_schema_contract().encode()
        ).hexdigest(),
        "inspect_response_schema_unchanged_from_w2": True,
    }
    assert contracts["decision"] == {
        "prompt_version": LOCAL_DECISION_VERSION,
        "prompt_sha256": hashlib.sha256(
            local_decision_prompt_contract().encode()
        ).hexdigest(),
        "schema_sha256": hashlib.sha256(
            local_decision_schema_contract().encode()
        ).hexdigest(),
        "unchanged_from_w2": True,
    }


def test_fixture_material_is_fresh_neutral_and_2034_only() -> None:
    fixture = _load()
    w2_fixture = _load(W2_FIXTURE_PATH)
    compiler_events = [case["input"]["event"] for case in fixture["compiler_cases"]]
    decision_events = [
        event
        for case in fixture["decision_cases"]
        for event in case["input"]["context_events"]
    ]
    events = [*compiler_events, *decision_events]
    parsed_events = [ObservableEvent.model_validate(event) for event in events]
    assert len({event.id for event in parsed_events}) == 9
    assert all(event.at.year == 2034 for event in parsed_events)
    assert all(event.id.startswith("w3-preflight-") for event in parsed_events)

    public_w1_w2_texts = {local_preflight_event().text}
    public_w1_w2_texts.update(
        case["input"]["event"]["text"] for case in w2_fixture["compiler_cases"]
    )
    public_w1_w2_texts.update(
        event["text"]
        for case in w2_fixture["decision_cases"]
        for event in case["input"]["context_events"]
    )
    assert {event.text for event in parsed_events}.isdisjoint(public_w1_w2_texts)
    assert all(
        event.text not in LOCAL_MEMORY_COMPILER_W3_INSTRUCTIONS
        for event in parsed_events
    )

    prior_and_prompt_surface = (
        "\n".join(sorted(public_w1_w2_texts))
        + "\n"
        + LOCAL_MEMORY_COMPILER_W3_INSTRUCTIONS
    ).casefold()
    for marker in FRESH_MARKERS:
        assert marker not in prior_and_prompt_surface


def test_active_states_are_canonical_and_minimal_for_each_case() -> None:
    expected_intent_counts = {
        "C1": 0,
        "C2": 0,
        "C3": 0,
        "C4": 0,
        "C5": 1,
        "C6": 1,
        "C7": 0,
        "C8": 2,
    }
    states: dict[str, dict[str, Any]] = {}
    for case in _compiler_cases():
        raw = case["input"]["active_state"]
        state = json.loads(raw)
        states[case["id"]] = state
        assert raw == json.dumps(state, sort_keys=True, separators=(",", ":"))
        assert set(state) == {"facts", "intents"}
        assert state["facts"] == []
        assert len(state["intents"]) == expected_intent_counts[case["id"]]
        assert all(intent["status"] == "active" for intent in state["intents"])

    assert [intent["intent_id"] for intent in states["C5"]["intents"]] == [
        "align_polaris_lens"
    ]
    assert [intent["intent_id"] for intent in states["C6"]["intents"]] == [
        "dispatch_meridian_capsule"
    ]
    assert [intent["intent_id"] for intent in states["C8"]["intents"]] == [
        "catalog_prism_samples_north",
        "catalog_prism_samples_south",
    ]


def test_model_boundary_is_derived_only_from_each_case_input() -> None:
    for case in _compiler_cases():
        prompt = _compiler_prompt(case)
        assert case["input"]["event"]["text"] in prompt
        assert case["input"]["active_state"] in prompt
        mutated_metadata = copy.deepcopy(case)
        sentinels: list[str] = []
        for field in set(case) - {"input"}:
            sentinel = f"{field}_metadata_must_not_leak"
            sentinels.append(sentinel)
            mutated_metadata[field] = sentinel
        assert _compiler_prompt(mutated_metadata) == prompt
        for sentinel in sentinels:
            assert sentinel not in prompt

    decision_case = _load()["decision_cases"][0]
    prompt = _decision_prompt(decision_case)
    mutated_metadata = copy.deepcopy(decision_case)
    sentinels = []
    for field in set(decision_case) - {"input"}:
        sentinel = f"{field}_metadata_must_not_leak"
        sentinels.append(sentinel)
        mutated_metadata[field] = sentinel
    assert _decision_prompt(mutated_metadata) == prompt
    for sentinel in sentinels:
        assert sentinel not in prompt


def test_all_valid_wire_examples_round_trip_to_their_domain_examples() -> None:
    fixture = _load()
    for case in fixture["compiler_cases"]:
        example = _wire_example(case)
        parsed = LocalMemoryDeltaWire.model_validate(example)
        assert (
            parsed.to_domain().model_dump(mode="json") == case["valid_domain_example"]
        )
        _validate_compiler_projection(case, example)

    decision_case = fixture["decision_cases"][0]
    decision = LocalDecisionWire.model_validate(decision_case["valid_wire_example"])
    assert (
        decision.to_domain().model_dump(mode="json")
        == decision_case["valid_domain_example"]
    )
    assert decision.mode == decision_case["acceptance"]["mode"]
    assert decision.actions == decision_case["acceptance"]["actions"]


@pytest.mark.parametrize("case_id", ["C2", "C3", "C4", "C6", "C7"])
def test_payload_acceptance_allows_omitted_or_null_unused_slots(case_id: str) -> None:
    case = _compiler_case(case_id)
    example = _wire_example(case)
    omitted = _validate_compiler_projection(case, example)
    explicit_null = _validate_compiler_projection(
        case,
        _with_all_unused_payload_slots_null(example),
    )
    assert explicit_null == omitted


@pytest.mark.parametrize("case_id", ["C2", "C3", "C4", "C7"])
def test_create_ids_and_summaries_are_structural_only(case_id: str) -> None:
    case = _compiler_case(case_id)
    _validate_compiler_projection(
        case,
        _with_alternative_structural_create(_wire_example(case)),
    )


@pytest.mark.parametrize("case_id", ["C5", "C6"])
def test_update_acceptance_allows_omitted_or_null_unchanged_fields(
    case_id: str,
) -> None:
    case = _compiler_case(case_id)
    example = _wire_example(case)
    candidate = copy.deepcopy(example)
    update = _single_update(candidate)
    for field in case["acceptance"]["unchanged_top_level_fields"]:
        update[field] = None
    assert _validate_compiler_projection(case, candidate) == (
        _validate_compiler_projection(case, example)
    )


def test_absent_units_allow_wire_omission_or_null() -> None:
    c1 = _compiler_case("C1")
    c1_omitted = copy.deepcopy(_wire_example(c1))
    c1_omitted["fact_assertions"][0].pop("unit")
    assert _validate_compiler_projection(c1, c1_omitted) == (
        _validate_compiler_projection(c1, _wire_example(c1))
    )

    c3 = _compiler_case("C3")
    c3_omitted = copy.deepcopy(_wire_example(c3))
    _single_create(c3_omitted)["required_conditions"][1].pop("unit")
    assert _validate_compiler_projection(c3, c3_omitted) == (
        _validate_compiler_projection(c3, _wire_example(c3))
    )


def test_c1_requires_the_exact_normalized_typed_unitless_fact() -> None:
    case = _compiler_case("C1")
    acceptance = case["acceptance"]
    assert acceptance["key"] == {
        "entity": "sapphire_observatory",
        "attribute": "backup_battery_level",
    }
    assert acceptance["value"] == 73
    assert type(acceptance["value"]) is int
    assert acceptance["unit"] is None

    wrong_key = copy.deepcopy(_wire_example(case))
    wrong_key["fact_assertions"][0]["entity"] = "sapphire_observatory_annex"
    with pytest.raises(AssertionError, match="key"):
        _validate_compiler_projection(case, wrong_key)

    wrong_type = copy.deepcopy(_wire_example(case))
    wrong_type["fact_assertions"][0]["value"] = "73"
    with pytest.raises(AssertionError, match="value"):
        _validate_compiler_projection(case, wrong_type)

    invented_unit = copy.deepcopy(_wire_example(case))
    invented_unit["fact_assertions"][0]["unit"] = "percent"
    with pytest.raises(AssertionError, match="unit"):
        _validate_compiler_projection(case, invented_unit)


def test_c2_bare_weekday_is_the_first_strictly_future_local_occurrence() -> None:
    case = _compiler_case("C2")
    event_at = datetime.fromisoformat(case["input"]["event"]["at"])
    trigger_at = datetime.fromisoformat(case["acceptance"]["trigger"]["at"])
    assert event_at.strftime("%A").casefold() == "monday"
    assert trigger_at.strftime("%A").casefold() == "monday"
    assert trigger_at > event_at
    assert (trigger_at.date() - event_at.date()).days == 7
    assert trigger_at.utcoffset() == event_at.utcoffset()
    assert "item Aurora Spectrometer" in case["input"]["event"]["text"]
    assert case["acceptance"]["action_template"]["payload"] == {
        "subject": "calibrate equipment",
        "item": "Aurora Spectrometer",
    }


def test_c3_requires_both_exact_and_conditions_and_no_synthetic_negative() -> None:
    case = _compiler_case("C3")
    example = _wire_example(case)
    assert len(case["acceptance"]["required_conditions"]) == 2
    assert case["acceptance"]["blockers"] == []

    missing_conjunct = copy.deepcopy(example)
    _single_create(missing_conjunct)["required_conditions"].pop()
    with pytest.raises(AssertionError, match="required_conditions"):
        _validate_compiler_projection(case, missing_conjunct)

    synthetic_negative = copy.deepcopy(example)
    _single_create(synthetic_negative)["blockers"].append(
        {
            "entity": "cobalt_beacon",
            "attribute": "status",
            "operator": "eq",
            "value": "not_ready",
        }
    )
    with pytest.raises(AssertionError, match="blockers"):
        _validate_compiler_projection(case, synthetic_negative)


def test_c3_condition_acceptance_is_a_canonical_multiset() -> None:
    case = _compiler_case("C3")
    assert case["acceptance"]["required_conditions_match"] == ("canonical_multiset")
    reversed_conditions = copy.deepcopy(_wire_example(case))
    _single_create(reversed_conditions)["required_conditions"].reverse()
    _validate_compiler_projection(case, reversed_conditions)


def test_c4_requires_the_exact_recurrence_range_and_iana_timezone() -> None:
    case = _compiler_case("C4")
    trigger = case["acceptance"]["trigger"]
    assert {key: value for key, value in trigger.items() if key != "weekdays"} == {
        "type": "recurring",
        "local_time": "06:35:00",
        "start_date": "2034-11-01",
        "end_date": "2034-12-15",
        "timezone": "America/Toronto",
    }
    assert set(trigger["weekdays"]) == {"tuesday", "saturday"}
    assert case["acceptance"]["weekdays_match"] == "set"
    assert "item Zephyr Antenna" in case["input"]["event"]["text"]
    assert case["acceptance"]["action_template"]["payload"] == {
        "subject": "inspect equipment",
        "item": "Zephyr Antenna",
    }
    assert "date" not in case["acceptance"]["action_template"]["payload"]


def test_c4_weekday_acceptance_is_a_set() -> None:
    case = _compiler_case("C4")
    reversed_weekdays = copy.deepcopy(_wire_example(case))
    _single_create(reversed_weekdays)["trigger"]["weekdays"].reverse()
    _validate_compiler_projection(case, reversed_weekdays)


@pytest.mark.parametrize(
    ("case_id", "field", "wrong_value"),
    [
        ("C2", "at", "2034-08-21T09:10:00+10:00"),
        ("C3", "active_until", "2034-09-19T18:00:00+02:00"),
        ("C4", "timezone", "America/Vancouver"),
        ("C5", "at", "2034-11-15T16:45:00+01:00"),
        ("C7", "at", "2034-04-10T10:30:00+05:30"),
    ],
)
def test_wrong_dates_or_trigger_fields_are_rejected(
    case_id: str,
    field: str,
    wrong_value: str,
) -> None:
    case = _compiler_case(case_id)
    candidate = copy.deepcopy(_wire_example(case))
    if candidate["intent_creates"]:
        trigger = _single_create(candidate)["trigger"]
    else:
        trigger = _single_update(candidate)["trigger"]
    trigger[field] = wrong_value
    with pytest.raises(AssertionError, match="trigger"):
        _validate_compiler_projection(case, candidate)


@pytest.mark.parametrize("case_id", ["C5", "C6"])
def test_update_ids_must_be_copied_exactly_from_active_state(case_id: str) -> None:
    case = _compiler_case(case_id)
    candidate = copy.deepcopy(_wire_example(case))
    _single_update(candidate)["intent_id"] = "plausible_but_wrong_intent"
    with pytest.raises(AssertionError, match="intent_id"):
        _validate_compiler_projection(case, candidate)


@pytest.mark.parametrize("missing_leaf", ["project", "room"])
def test_c6_full_template_update_rejects_missing_preserved_leaves(
    missing_leaf: str,
) -> None:
    case = _compiler_case("C6")
    candidate = copy.deepcopy(_wire_example(case))
    _action_payload(candidate).pop(missing_leaf)
    with pytest.raises(AssertionError, match="action_template"):
        _validate_compiler_projection(case, candidate)


def test_c6_changes_only_the_licensed_recipient_leaf() -> None:
    case = _compiler_case("C6")
    acceptance = case["acceptance"]
    assert acceptance["preserved_payload_leaves"] == {
        "subject": "dispatch meridian capsule",
        "project": "Helios Ledger",
        "room": "Orchid Bay 7",
    }
    assert acceptance["changed_payload_leaves"] == {"recipient": "Dr. Rowan Keir"}
    wrong_preserved_value = copy.deepcopy(_wire_example(case))
    _action_payload(wrong_preserved_value)["project"] = "Different Project"
    with pytest.raises(AssertionError, match="action_template"):
        _validate_compiler_projection(case, wrong_preserved_value)


@pytest.mark.parametrize(
    ("key", "wrong_value"),
    [
        ("item", None),
        ("quantity", 1),
        ("recipient", None),
        ("room", None),
        ("project", None),
        ("date", "2034-04-09"),
        ("address", ""),
    ],
)
def test_c7_rejects_missing_values_wrong_zero_or_filler(
    key: str,
    wrong_value: Any,
) -> None:
    case = _compiler_case("C7")
    candidate = copy.deepcopy(_wire_example(case))
    payload = _action_payload(candidate)
    if wrong_value is None:
        payload.pop(key)
    else:
        payload[key] = wrong_value
    with pytest.raises(AssertionError, match="payload"):
        _validate_compiler_projection(case, candidate)


def test_c7_preserves_proper_casing_and_numeric_zero() -> None:
    payload = _compiler_case("C7")["acceptance"]["action_template"]["payload"]
    assert payload == {
        "subject": "stage supply handoff",
        "item": "Aster Relay",
        "project": "Silver Comet",
        "quantity": 0,
        "recipient": "Captain Nia Sol",
        "room": "Quasar Room 12",
    }
    assert type(payload["quantity"]) is int
    assert "date" not in payload


def test_non_null_unused_payload_filler_is_rejected() -> None:
    case = _compiler_case("C2")
    candidate = copy.deepcopy(_wire_example(case))
    _action_payload(candidate)["room"] = ""
    with pytest.raises(AssertionError, match="payload"):
        _validate_compiler_projection(case, candidate)


@pytest.mark.parametrize("case_id", [case[0] for case in ORDERED_CASES[:8]])
def test_every_compiler_case_rejects_an_extra_independent_mutation(
    case_id: str,
) -> None:
    case = _compiler_case(case_id)
    candidate = copy.deepcopy(_wire_example(case))
    candidate["fact_assertions"].append(
        {
            "entity": "extraneous_fixture",
            "attribute": "flag",
            "value": True,
        }
    )
    with pytest.raises(AssertionError, match="mutation_count"):
        _validate_compiler_projection(case, candidate)


def test_c8_has_two_plausible_active_targets_and_rejects_a_guessed_id() -> None:
    case = _compiler_case("C8")
    state = json.loads(case["input"]["active_state"])
    assert len(state["intents"]) == 2
    assert {
        intent["action_template"]["payload"]["subject"] for intent in state["intents"]
    } == {"catalog prism samples"}
    assert _validate_compiler_projection(case, _wire_example(case)) == {"mutations": []}

    guessed = copy.deepcopy(_wire_example(case))
    guessed["intent_cancellations"].append({"intent_id": "catalog_prism_samples_north"})
    with pytest.raises(AssertionError, match="mutation_count"):
        _validate_compiler_projection(case, guessed)


def test_d1_requires_wire_and_domain_no_action() -> None:
    case = _load()["decision_cases"][0]
    case_input = case["input"]
    event = ObservableEvent.model_validate(case_input["context_events"][0])
    assert case_input["current_event_id"] == event.id
    assert datetime.fromisoformat(case_input["now"]) == event.at
    assert case_input["memory_view"] == {"blocks": []}
    assert case_input["decision_history"] == []

    unwanted_emit = {
        "mode": "emit",
        "actions": [
            {
                "action_key": "unwanted_fixture_action",
                "payload": {"subject": "raise unwanted flag"},
                "summary": "An unwanted action.",
                "evidence_event_ids": [event.id],
            }
        ],
    }
    parsed = LocalDecisionWire.model_validate(unwanted_emit)
    assert parsed.to_domain().actions
    with pytest.raises(AssertionError):
        assert parsed.mode == case["acceptance"]["mode"]
