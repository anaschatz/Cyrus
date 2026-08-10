from __future__ import annotations

from anamnesis.memory import AtTrigger, CreateIntent
from anamnesis.wire import DecisionWire, MemoryDeltaWire


def _payload(**updates):
    payload = {
        "subject": "send the assignment",
        "address": None,
        "build": None,
        "date": None,
        "flight": None,
        "greenhouse": None,
        "item": None,
        "project": None,
        "quantity": None,
        "recipient": None,
        "room": None,
        "shipment": None,
        "tank": None,
        "trip": None,
    }
    payload.update(updates)
    return payload


def _at_trigger():
    return {
        "type": "at",
        "at": "2026-03-06T17:00:00+02:00",
        "local_time": None,
        "weekdays": None,
        "start_date": None,
        "end_date": None,
        "timezone": None,
        "active_from": None,
        "active_until": None,
    }


def test_decision_wire_drops_null_payload_slots() -> None:
    wire = DecisionWire.model_validate(
        {
            "actions": [
                {
                    "kind": "reminder",
                    "action_key": "event-1",
                    "payload": _payload(recipient="Dr. Ada"),
                    "summary": "Send the assignment to Dr. Ada.",
                    "evidence_event_ids": ["event-1"],
                }
            ]
        }
    )

    decision = wire.to_domain()

    assert decision.actions[0].payload == {
        "subject": "send the assignment",
        "recipient": "Dr. Ada",
    }


def test_memory_delta_wire_converts_flat_trigger_to_domain_union() -> None:
    wire = MemoryDeltaWire.model_validate(
        {
            "fact_assertions": [],
            "intent_creates": [
                {
                    "intent_id": "assignment.reminder",
                    "trigger": _at_trigger(),
                    "required_conditions": [],
                    "blockers": [],
                    "action_template": {
                        "kind": "reminder",
                        "payload": _payload(),
                        "summary": "Send the assignment.",
                    },
                }
            ],
            "intent_updates": [],
            "intent_cancellations": [],
        }
    )

    delta = wire.to_domain()

    mutation = delta.mutations[0]
    assert isinstance(mutation, CreateIntent)
    assert isinstance(mutation.trigger, AtTrigger)
    assert mutation.action_template.payload == {"subject": "send the assignment"}
