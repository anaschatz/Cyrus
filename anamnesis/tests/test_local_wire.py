from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from anamnesis.local_wire import (
    LOCAL_MEMORY_COMPILER_INSTRUCTIONS,
    LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS,
    LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT,
    LOCAL_MEMORY_COMPILER_W3_ADDENDUM,
    LOCAL_MEMORY_COMPILER_W3_INSTRUCTIONS,
    LOCAL_MEMORY_COMPILER_W3_VERSION,
    LocalMemoryDeltaWire,
    build_local_memory_compiler_prompt,
    build_local_memory_compiler_w2_prompt,
    build_local_memory_compiler_w3_prompt,
    local_memory_compiler_contract,
    local_memory_compiler_w2_contract,
    local_memory_compiler_w3_contract,
)
from anamnesis.memory import (
    AtTrigger,
    ConditionTransitionTrigger,
    CreateIntent,
    RecurringTrigger,
    UpdateIntent,
)
from anamnesis.schema import ObservableEvent

W3_PROTOCOL_PATH = Path("eval/preflight/local_writer_w3.protocol.v1.json")


def _delta_with_trigger(trigger: dict[str, object]) -> dict[str, object]:
    return {
        "fact_assertions": [],
        "intent_creates": [
            {
                "intent_id": "assignment.reminder",
                "trigger": trigger,
                "required_conditions": [],
                "blockers": [],
                "action_template": {
                    "payload": {"subject": "send the assignment"},
                    "summary": "Send the assignment.",
                },
            }
        ],
        "intent_updates": [],
        "intent_cancellations": [],
    }


@pytest.mark.parametrize(
    ("trigger", "domain_type"),
    [
        (
            {"type": "at", "at": "2026-03-06T17:00:00+02:00"},
            AtTrigger,
        ),
        (
            {
                "type": "recurring",
                "local_time": "09:30:00",
                "weekdays": ["monday", "friday"],
                "start_date": "2026-03-02",
                "end_date": "2026-03-06",
                "timezone": "Europe/Athens",
            },
            RecurringTrigger,
        ),
        (
            {
                "type": "condition_transition",
                "active_from": "2026-03-02T09:00:00+02:00",
                "active_until": "2026-03-06T17:00:00+02:00",
            },
            ConditionTransitionTrigger,
        ),
    ],
)
def test_local_trigger_variants_convert_to_domain(
    trigger: dict[str, object],
    domain_type: type[object],
) -> None:
    record = _delta_with_trigger(trigger)
    if trigger["type"] == "condition_transition":
        create = record["intent_creates"][0]
        create["required_conditions"] = [
            {
                "entity": "assignment",
                "attribute": "submitted",
                "operator": "eq",
                "value": False,
            }
        ]

    delta = LocalMemoryDeltaWire.model_validate(record).to_domain()

    mutation = delta.mutations[0]
    assert isinstance(mutation, CreateIntent)
    assert isinstance(mutation.trigger, domain_type)
    assert mutation.action_template.payload == {"subject": "send the assignment"}


def test_local_at_trigger_rejects_fields_from_other_variants() -> None:
    record = _delta_with_trigger(
        {
            "type": "at",
            "at": "2026-03-06T17:00:00+02:00",
            "active_from": "2026-03-02T09:00:00+02:00",
        }
    )

    with pytest.raises(ValidationError, match="active_from"):
        LocalMemoryDeltaWire.model_validate(record)


def test_local_trigger_schema_is_closed_and_discriminated() -> None:
    schema = LocalMemoryDeltaWire.model_json_schema()
    trigger_schema = schema["$defs"]["LocalIntentCreateWire"]["properties"]["trigger"]

    assert trigger_schema["discriminator"] == {
        "mapping": {
            "at": "#/$defs/LocalAtTriggerWire",
            "condition_transition": ("#/$defs/LocalConditionTransitionTriggerWire"),
            "recurring": "#/$defs/LocalRecurringTriggerWire",
        },
        "propertyName": "type",
    }
    assert len(trigger_schema["oneOf"]) == 3
    for name in (
        "LocalAtTriggerWire",
        "LocalRecurringTriggerWire",
        "LocalConditionTransitionTriggerWire",
    ):
        assert schema["$defs"][name]["additionalProperties"] is False


def test_local_update_allows_omitted_unchanged_fields() -> None:
    wire = LocalMemoryDeltaWire.model_validate(
        {
            "fact_assertions": [],
            "intent_creates": [],
            "intent_updates": [
                {
                    "intent_id": "assignment.reminder",
                    "trigger": {
                        "type": "at",
                        "at": "2026-03-07T17:00:00+02:00",
                    },
                }
            ],
            "intent_cancellations": [],
        }
    )

    mutation = wire.to_domain().mutations[0]

    assert isinstance(mutation, UpdateIntent)
    assert mutation.model_fields_set == {"intent_id", "trigger"}
    assert mutation.required_conditions is None


def test_local_update_rejects_noop() -> None:
    with pytest.raises(ValidationError, match="requires a changed field"):
        LocalMemoryDeltaWire.model_validate(
            {
                "fact_assertions": [],
                "intent_creates": [],
                "intent_updates": [{"intent_id": "assignment.reminder"}],
                "intent_cancellations": [],
            }
        )


def test_local_compiler_prompt_has_only_observable_event_and_active_state() -> None:
    event = ObservableEvent(
        id="event-1",
        at="2026-03-02T09:00:00+02:00",
        kind="user_message",
        text="Remind me Friday to send the assignment.",
    )

    prompt = build_local_memory_compiler_prompt(
        event=event,
        active_state='{"facts":[],"intents":[]}',
    )

    assert "event-1" in prompt
    assert event.text in prompt
    assert '"facts":[],"intents":[]' in prompt
    assert "supersedes" not in prompt
    assert "gold" not in prompt.casefold()
    assert "exactly one of these shapes" in prompt


def test_local_compiler_contract_is_deterministic_and_schema_bound() -> None:
    first = local_memory_compiler_contract()
    second = local_memory_compiler_contract()

    assert first == second
    assert (
        hashlib.sha256(first.encode()).hexdigest()
        == hashlib.sha256(second.encode()).hexdigest()
    )
    assert '"discriminator"' in first
    assert first.startswith("local.v0.2\n")


def test_local_w1_contract_hash_remains_frozen() -> None:
    assert (
        hashlib.sha256(local_memory_compiler_contract().encode()).hexdigest()
        == "1ac94e36a5db89ef03798b091424494b9cf50f52ac8e7aaa70e8cfcfc3b0ebd8"
    )


def test_local_w2_contract_hash_is_frozen() -> None:
    contract = local_memory_compiler_w2_contract()

    assert contract.startswith("local.v0.3\n")
    assert (
        hashlib.sha256(contract.encode()).hexdigest()
        == "cb46570bfb1a101bff51008315ba121e07cea38a93de38fe6c79693d746f72c9"
    )


def test_local_w2_sparse_optional_payload_rule_is_complete() -> None:
    rule = LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT

    assert "explicitly sourced by the current event" in rule
    assert "legitimately preserved within an action_template" in rule
    assert "current event actually updates" in rule
    assert "Omit an unused payload key (preferred), or use JSON null" in rule
    for forbidden_filler in (
        "empty string",
        "false",
        "empty collection",
        "placeholder",
        "zero filler",
    ):
        assert forbidden_filler in rule
    assert "explicitly sourced quantity of zero remains valid" in rule
    assert "Before returning, remove filler values" in rule
    assert '{"subject":"check permit"}' in rule
    assert "Add optional keys only when sourced" in rule


def test_local_w2_uses_the_unchanged_w1_wire_schema() -> None:
    w1_schema = local_memory_compiler_contract().rsplit("\n", 1)[-1]
    w2_schema = local_memory_compiler_w2_contract().rsplit("\n", 1)[-1]

    assert w2_schema == w1_schema


def test_local_w2_prompt_delta_is_only_the_sparse_payload_invariant() -> None:
    event = ObservableEvent(
        id="event-1",
        at="2026-03-02T09:00:00+02:00",
        kind="user_message",
        text="Current event only.",
    )
    active_state = '{"facts":[],"intents":[]}'
    w1_prompt = build_local_memory_compiler_prompt(
        event=event,
        active_state=active_state,
    )
    w2_prompt = build_local_memory_compiler_w2_prompt(
        event=event,
        active_state=active_state,
    )

    assert LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS == (
        LOCAL_MEMORY_COMPILER_INSTRUCTIONS + LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT
    )
    assert w2_prompt == w1_prompt.replace(
        LOCAL_MEMORY_COMPILER_INSTRUCTIONS,
        LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS,
        1,
    )
    assert w2_prompt.count(LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT) == 1
    for leak_marker in ("writer_diagnostic", "oracle", "gold", "scenario"):
        assert leak_marker not in LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT.casefold()


def test_local_w3_contract_hashes_and_version_are_frozen() -> None:
    event = ObservableEvent(
        id="<event-id>",
        at="2000-01-01T00:00:00+00:00",
        kind="user_message",
        text="<event-text>",
    )
    prompt = build_local_memory_compiler_w3_prompt(
        event=event,
        active_state='{"facts":[],"intents":[]}',
    )
    contract = local_memory_compiler_w3_contract()

    assert LOCAL_MEMORY_COMPILER_W3_VERSION == "local.v0.4"
    assert contract.startswith("local.v0.4\n")
    assert hashlib.sha256(LOCAL_MEMORY_COMPILER_W3_ADDENDUM.encode()).hexdigest() == (
        "84897bc8493dc4c89272aacd9ec6aaf869de92e63b1e225b954d97af84877793"
    )
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "412a63d6b42ea6b5e294401cabbcbacf5a6b7facddbd8fe04ca7b91914c141e5"
    )
    assert hashlib.sha256(contract.encode()).hexdigest() == (
        "b90298df967f81c91cd6aed6289190768b1f4fe28af4743fb118920d11f8ec51"
    )


def test_local_w3_uses_the_unchanged_w1_w2_wire_schema() -> None:
    w1_schema = local_memory_compiler_contract().rsplit("\n", 1)[-1]
    w2_schema = local_memory_compiler_w2_contract().rsplit("\n", 1)[-1]
    w3_schema = local_memory_compiler_w3_contract().rsplit("\n", 1)[-1]

    assert w1_schema == w2_schema == w3_schema
    assert hashlib.sha256(w3_schema.encode()).hexdigest() == (
        "f0e0ab9c3aef10f9b99ca5055d1ee1f2e6d7f091be666ee95035040e564302ec"
    )


def test_local_w3_prompt_delta_is_only_the_frozen_addendum() -> None:
    event = ObservableEvent(
        id="event-1",
        at="2026-03-02T09:00:00+02:00",
        kind="user_message",
        text="Current event only.",
    )
    active_state = '{"facts":[],"intents":[]}'
    w2_prompt = build_local_memory_compiler_w2_prompt(
        event=event,
        active_state=active_state,
    )
    w3_prompt = build_local_memory_compiler_w3_prompt(
        event=event,
        active_state=active_state,
    )

    assert LOCAL_MEMORY_COMPILER_W3_INSTRUCTIONS == (
        LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS + LOCAL_MEMORY_COMPILER_W3_ADDENDUM
    )
    assert w3_prompt == w2_prompt.replace(
        LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS,
        LOCAL_MEMORY_COMPILER_W3_INSTRUCTIONS,
        1,
    )
    assert w3_prompt.count(LOCAL_MEMORY_COMPILER_W3_ADDENDUM) == 1
    assert LOCAL_MEMORY_COMPILER_W3_ADDENDUM not in w2_prompt


def test_local_w3_addendum_freezes_all_bundled_repair_boundaries() -> None:
    addendum = LOCAL_MEMORY_COMPILER_W3_ADDENDUM

    for required_rule in (
        "replace each maximal run outside [a-z0-9] with one underscore",
        "never use an empty string",
        "copy its intent_id character-for-character from Active compact state",
        "compute (target weekday index - current weekday index) modulo seven",
        "Use condition_transition",
        "required IANA timezone",
        "including an explicitly sourced zero",
        "retain every unchanged sourced leaf",
        "every explicit AND conjunct is preserved",
        "If any required check is uncertain, omit that mutation",
    ):
        assert required_rule in addendum


def test_local_w3_addendum_contains_no_benchmark_or_acceptance_leakage() -> None:
    normalized = LOCAL_MEMORY_COMPILER_W3_ADDENDUM.casefold()

    for leak_marker in (
        "writer_diagnostic",
        "writer diagnostic",
        "scenario",
        "oracle",
        "gold",
        "wd3_",
        "wd4_",
        "v3.json",
        "v4.json",
        "acceptance_projection",
        "candidate false",
        "c1",
        "d1",
    ):
        assert leak_marker not in normalized


def test_local_w3_machine_protocol_is_frozen_and_contains_no_case_material() -> None:
    protocol_bytes = W3_PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    ordered = [
        (case["id"], case["role"], case["category"])
        for case in protocol["preflight"]["ordered_categories"]
    ]

    assert hashlib.sha256(protocol_bytes).hexdigest() == (
        "7f63c156a8af74ced2d5e5530b3e8083da95c7f46e14e1afafaaf864b3ce1915"
    )
    assert protocol["intervention"] == {
        "label": "bundled-repair",
        "causal_interpretation": "multi_factor_not_single_ablation",
        "prompt_only": True,
        "base_prompt_version": "local.v0.3",
        "prompt_version": "local.v0.4",
        "wire_schema_changed": False,
        "decision_contract_changed": False,
        "reducer_changed": False,
    }
    compiler_contract = protocol["contracts"]["compiler"]
    assert compiler_contract["prompt_version"] == LOCAL_MEMORY_COMPILER_W3_VERSION
    assert (
        compiler_contract["addendum_sha256"]
        == hashlib.sha256(LOCAL_MEMORY_COMPILER_W3_ADDENDUM.encode()).hexdigest()
    )
    assert (
        compiler_contract["prompt_sha256"]
        == hashlib.sha256(
            build_local_memory_compiler_w3_prompt(
                event=ObservableEvent(
                    id="<event-id>",
                    at="2000-01-01T00:00:00+00:00",
                    kind="user_message",
                    text="<event-text>",
                ),
                active_state='{"facts":[],"intents":[]}',
            ).encode()
        ).hexdigest()
    )
    assert (
        compiler_contract["local_wire_contract_sha256"]
        == hashlib.sha256(local_memory_compiler_w3_contract().encode()).hexdigest()
    )
    assert (
        compiler_contract["local_wire_model_schema_sha256"]
        == hashlib.sha256(
            local_memory_compiler_w3_contract().rsplit("\n", 1)[-1].encode()
        ).hexdigest()
    )
    assert compiler_contract["inspect_response_schema_sha256"] == (
        "8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030"
    )
    assert compiler_contract["inspect_response_schema_unchanged_from_w2"] is True
    assert protocol["contracts"]["decision"] == {
        "prompt_version": "ollama.decision.v0.2",
        "prompt_sha256": (
            "871fe15e3160e66abe7480cbde15dfb943dec2d0ff89bb01a03849ad35defd8d"
        ),
        "schema_sha256": (
            "1b7c38d3f4bf150523ecc1e468ad3fb1f94753611f190d70f93abbf5ec582426"
        ),
        "unchanged_from_w2": True,
    }
    assert ordered == [
        ("C1", "compiler", "normalization_fact"),
        ("C2", "compiler", "bare_weekday_at"),
        ("C3", "compiler", "condition_transition_and"),
        ("C4", "compiler", "recurrence_iana_range"),
        ("C5", "compiler", "stable_id_trigger_update"),
        ("C6", "compiler", "full_action_template_update"),
        ("C7", "compiler", "complete_sparse_payload_including_zero"),
        ("C8", "compiler", "ambiguous_empty"),
        ("D1", "decision", "no_action"),
    ]
    assert protocol["preflight"]["case_material_status"] == (
        "unwritten_at_protocol_freeze"
    )
    assert protocol["preflight"]["compiler_calls"] == 8
    assert protocol["preflight"]["decision_calls"] == 1
    assert protocol["preflight"]["model_calls"] == 9
    for case in protocol["preflight"]["ordered_categories"]:
        assert set(case) == {"id", "role", "category", "acceptance_projection"}
        assert case["acceptance_projection"]

    prohibited_case_keys = {
        "input",
        "event",
        "events",
        "text",
        "active_state",
        "valid_wire_example",
        "valid_domain_example",
        "example",
        "examples",
    }

    def assert_no_case_material(value: object) -> None:
        if isinstance(value, dict):
            assert prohibited_case_keys.isdisjoint(value)
            for child in value.values():
                assert_no_case_material(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_case_material(child)

    assert_no_case_material(protocol)


def test_local_w3_protocol_freezes_execution_gate_and_stopping_rule() -> None:
    protocol = json.loads(W3_PROTOCOL_PATH.read_bytes())

    assert protocol["model"] == {
        "snapshot": "ollama/qwen3:4b-instruct",
        "seed": 101,
        "temperature": 0.0,
        "provider_api_cost_usd": 0.0,
        "same_model_for_compiler_and_decision": True,
    }
    assert protocol["execution"] == {
        "standalone_preflight_attempts": 1,
        "measured_task_attempts": 1,
        "measured_task_setup_replays_same_preflight_once": True,
        "response_cache": False,
        "max_retries": 0,
        "structured_output_repair_calls": 0,
        "concurrency": 1,
        "raw_model_calls_logged": True,
        "setup_usage_excluded_from_scenario_headline": True,
    }
    assert protocol["scenario_gate"] == {
        "compiler_calls": 39,
        "parse_or_domain_invalid": 0,
        "semantic_or_store_invalid": 0,
        "accepted_deltas": 39,
        "due_candidate_false_positives": 0,
        "due_candidate_false_negatives": 0,
        "measured_and_reference_candidate_multisets_must_match": True,
        "decision_actions_excluded_from_writer_gate": True,
    }
    stopping = protocol["stopping_rule"]
    assert stopping["no_second_w3_run_on_v4"] is True
    assert stopping["no_w4_on_v4"] is True
    assert stopping["next_prompt_revision_requires_fresh_dataset"] is True
