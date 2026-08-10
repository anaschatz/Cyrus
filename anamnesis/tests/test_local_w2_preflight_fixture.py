from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from anamnesis.local_runtime import (
    LOCAL_DECISION_VERSION,
    LocalDecisionWire,
    local_decision_prompt_contract,
    local_decision_schema_contract,
    local_memory_compiler_schema_contract,
)
from anamnesis.local_wire import (
    LOCAL_MEMORY_COMPILER_W2_VERSION,
    LocalMemoryDeltaWire,
    build_local_memory_compiler_w2_prompt,
)
from anamnesis.schema import OPTIONAL_PAYLOAD_KEYS, ObservableEvent

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "eval" / "preflight" / "local_writer_w2.v1.json"
FIXTURE_SHA256 = "3b82128bab1d801d073118488aa4f0a0a662603b98325f5c9d7dad497f026057"
EMPTY_ACTIVE_STATE = '{"facts":[],"intents":[]}'

COMPILER_CASES = (
    ("C1", "trivial_explicit_same_day_at_subject_only"),
    ("C2", "trivial_explicit_next_day_at_address_only"),
    ("C3", "irrelevant_observation_empty_memory_delta"),
)
DECISION_CASES = (("D1", "structured_memory_empty_irrelevant_raw_event_no_action"),)


def _load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _wire_payload(wire: dict[str, object]) -> dict[str, object]:
    creates = wire["intent_creates"]
    assert isinstance(creates, list) and len(creates) == 1
    create = creates[0]
    assert isinstance(create, dict)
    action_template = create["action_template"]
    assert isinstance(action_template, dict)
    payload = action_template["payload"]
    assert isinstance(payload, dict)
    return payload


def _compiler_example(case: dict[str, object]) -> dict[str, object]:
    example = case["valid_wire_example"]
    assert isinstance(example, dict)
    return example


def _validate_compiler_projection(
    case: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    """Validate only the preregistered semantics, not example surface text."""

    parsed = LocalMemoryDeltaWire.model_validate(candidate)
    domain = parsed.to_domain().model_dump(mode="json")
    acceptance = case["acceptance"]
    assert isinstance(acceptance, dict)
    mutations = domain["mutations"]
    assert isinstance(mutations, list)

    if acceptance["mutation_type"] == "empty_delta":
        assert mutations == [], "mutation_type"
        return domain

    assert acceptance["mutation_type"] == "create_intent"
    assert len(mutations) == 1, "mutation_count"
    mutation = mutations[0]
    assert isinstance(mutation, dict)
    assert mutation["op"] == acceptance["mutation_type"], "mutation_type"
    assert mutation["trigger"] == acceptance["trigger"], "trigger"
    assert mutation["required_conditions"] == acceptance["required_conditions"], (
        "required_conditions"
    )
    assert mutation["blockers"] == acceptance["blockers"], "blockers"

    action_template = mutation["action_template"]
    assert isinstance(action_template, dict)
    assert action_template["kind"] == acceptance["kind"], "kind"
    assert action_template["payload"] == acceptance["payload"], "payload"

    assert acceptance["intent_id"] == "structural_only"
    assert acceptance["summary"] == "structural_only"
    assert isinstance(mutation["intent_id"], str) and mutation["intent_id"]
    assert isinstance(action_template["summary"], str) and action_template["summary"]
    return domain


def _with_unused_payload_slots_null(
    wire: dict[str, object],
) -> dict[str, object]:
    candidate = copy.deepcopy(wire)
    payload = _wire_payload(candidate)
    for key in OPTIONAL_PAYLOAD_KEYS - payload.keys():
        payload[key] = None
    return candidate


def _with_alternative_structural_fields(
    wire: dict[str, object],
) -> dict[str, object]:
    candidate = copy.deepcopy(wire)
    creates = candidate["intent_creates"]
    assert isinstance(creates, list) and len(creates) == 1
    create = creates[0]
    assert isinstance(create, dict)
    create["intent_id"] = "alternate-reminder-id"
    action_template = create["action_template"]
    assert isinstance(action_template, dict)
    action_template["summary"] = "An alternative valid reminder summary."
    return candidate


def _w2_prompt_contract() -> str:
    sentinel = ObservableEvent(
        id="<event-id>",
        at="2000-01-01T00:00:00+00:00",
        kind="user_message",
        text="<event-text>",
    )
    return build_local_memory_compiler_w2_prompt(
        event=sentinel,
        active_state=EMPTY_ACTIVE_STATE,
    )


def test_local_w2_preflight_fixture_bytes_are_frozen() -> None:
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_local_w2_preflight_fixture_scope_and_case_order_are_exact() -> None:
    fixture = _load_fixture()

    assert fixture["schema_version"] == 1
    assert fixture["purpose"] == "diagnostic"
    assert fixture["hypothesis_test_eligible"] is False
    compiler_cases = fixture["compiler_cases"]
    decision_cases = fixture["decision_cases"]
    assert isinstance(compiler_cases, list)
    assert isinstance(decision_cases, list)
    assert [(case["id"], case["category"]) for case in compiler_cases] == list(
        COMPILER_CASES
    )
    assert [(case["id"], case["category"]) for case in decision_cases] == list(
        DECISION_CASES
    )
    assert all("acceptance" in case for case in compiler_cases)
    assert all("acceptance" in case for case in decision_cases)


def test_local_w2_preflight_contract_hashes_match_runtime_contracts() -> None:
    fixture = _load_fixture()
    contracts = fixture["contracts"]
    assert isinstance(contracts, dict)
    compiler = contracts["compiler"]
    decision = contracts["decision"]
    assert isinstance(compiler, dict)
    assert isinstance(decision, dict)

    assert compiler == {
        "prompt_version": LOCAL_MEMORY_COMPILER_W2_VERSION,
        "prompt_sha256": hashlib.sha256(_w2_prompt_contract().encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(
            local_memory_compiler_schema_contract().encode()
        ).hexdigest(),
    }
    assert decision == {
        "prompt_version": LOCAL_DECISION_VERSION,
        "prompt_sha256": hashlib.sha256(
            local_decision_prompt_contract().encode()
        ).hexdigest(),
        "schema_sha256": hashlib.sha256(
            local_decision_schema_contract().encode()
        ).hexdigest(),
    }


def test_compiler_inputs_are_exact_observable_events_with_empty_active_state() -> None:
    fixture = _load_fixture()
    compiler_cases = fixture["compiler_cases"]
    assert isinstance(compiler_cases, list)

    events: list[ObservableEvent] = []
    for case in compiler_cases:
        case_input = case["input"]
        assert isinstance(case_input, dict)
        assert case_input["active_state"] == EMPTY_ACTIVE_STATE
        events.append(ObservableEvent.model_validate(case_input["event"]))

    assert len({event.id for event in events}) == 3
    assert len({event.at for event in events}) == 3
    assert events[0].kind == "user_message"
    assert events[1].kind == "user_message"
    assert events[2].kind == "observation"


def test_valid_compiler_examples_round_trip_to_their_domain_examples() -> None:
    fixture = _load_fixture()
    compiler_cases = fixture["compiler_cases"]
    assert isinstance(compiler_cases, list)

    for case in compiler_cases:
        example = _compiler_example(case)
        domain_example = case["valid_domain_example"]
        parsed = LocalMemoryDeltaWire.model_validate(example)
        assert parsed.to_domain().model_dump(mode="json") == domain_example
        _validate_compiler_projection(case, example)


@pytest.mark.parametrize("case_index", [0, 1])
def test_create_acceptance_allows_omitted_or_null_unused_slots(
    case_index: int,
) -> None:
    fixture = _load_fixture()
    case = fixture["compiler_cases"][case_index]
    example = _compiler_example(case)

    omitted_domain = _validate_compiler_projection(case, example)
    null_domain = _validate_compiler_projection(
        case,
        _with_unused_payload_slots_null(example),
    )
    assert null_domain == omitted_domain


@pytest.mark.parametrize("case_index", [0, 1])
def test_create_acceptance_allows_alternative_valid_id_and_summary(
    case_index: int,
) -> None:
    fixture = _load_fixture()
    case = fixture["compiler_cases"][case_index]
    alternative = _with_alternative_structural_fields(_compiler_example(case))

    _validate_compiler_projection(case, alternative)


@pytest.mark.parametrize(
    ("case_index", "extra_key", "extra_value"),
    [
        (0, "room", ""),
        (1, "date", "2027-04-14"),
    ],
)
def test_create_acceptance_rejects_non_null_filler_or_extra_payload(
    case_index: int,
    extra_key: str,
    extra_value: str,
) -> None:
    fixture = _load_fixture()
    case = fixture["compiler_cases"][case_index]
    candidate = copy.deepcopy(_compiler_example(case))
    _wire_payload(candidate)[extra_key] = extra_value

    with pytest.raises(AssertionError, match="payload"):
        _validate_compiler_projection(case, candidate)


def test_c1_is_same_day_at_with_subject_only_acceptance() -> None:
    fixture = _load_fixture()
    case = fixture["compiler_cases"][0]
    acceptance = case["acceptance"]
    event_at = datetime.fromisoformat(case["input"]["event"]["at"])
    trigger_at = datetime.fromisoformat(acceptance["trigger"]["at"])

    assert trigger_at.date() == event_at.date()
    assert trigger_at > event_at
    assert acceptance["payload"] == {"subject": "water basil"}
    assert OPTIONAL_PAYLOAD_KEYS.isdisjoint(acceptance["payload"])


def test_c2_is_next_day_at_with_only_proper_cased_address_acceptance() -> None:
    fixture = _load_fixture()
    case = fixture["compiler_cases"][1]
    acceptance = case["acceptance"]
    event_at = datetime.fromisoformat(case["input"]["event"]["at"])
    trigger_at = datetime.fromisoformat(acceptance["trigger"]["at"])
    payload = acceptance["payload"]

    assert (trigger_at.date() - event_at.date()).days == 1
    assert payload == {
        "subject": "collect parcel",
        "address": "42 Harbor Avenue",
    }
    assert set(payload) & OPTIONAL_PAYLOAD_KEYS == {"address"}
    assert payload["address"] == "42 Harbor Avenue"


def test_c3_is_an_irrelevant_observation_with_empty_delta_acceptance() -> None:
    fixture = _load_fixture()
    case = fixture["compiler_cases"][2]

    assert case["input"]["event"]["kind"] == "observation"
    assert case["acceptance"] == {"mutation_type": "empty_delta"}
    assert _validate_compiler_projection(case, _compiler_example(case)) == {
        "mutations": []
    }


def test_d1_empty_structured_memory_and_irrelevant_event_is_no_action() -> None:
    fixture = _load_fixture()
    case = fixture["decision_cases"][0]
    case_input = case["input"]
    assert case_input["memory_view"] == {"blocks": []}
    assert case_input["decision_history"] == []
    assert len(case_input["context_events"]) == 1
    event = ObservableEvent.model_validate(case_input["context_events"][0])
    assert event.kind == "observation"
    assert case_input["current_event_id"] == event.id
    assert datetime.fromisoformat(case_input["now"]) == event.at

    example = case["valid_wire_example"]
    parsed = LocalDecisionWire.model_validate(example)
    assert case["acceptance"] == {"mode": "no_action", "actions": []}
    assert parsed.mode == case["acceptance"]["mode"]
    assert parsed.actions == case["acceptance"]["actions"]
    assert parsed.to_domain().model_dump(mode="json") == case["valid_domain_example"]
