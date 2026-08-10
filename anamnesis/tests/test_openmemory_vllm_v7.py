import hashlib

from anamnesis.action_canonicalizer import canonicalize_immediate_decision
from anamnesis.openmemory_vllm_v6 import _correct
from anamnesis.openmemory_vllm_v7 import (
    CANONICALIZER_PATH,
    FIXTURE_PATH,
    V7Fixture,
    _load_inputs,
)
from anamnesis.schema import Decision, ProposedAction


def test_v7_frozen_inputs_and_no_v6_ids() -> None:
    pin, fixture, runtime = _load_inputs()
    assert len(fixture.cases) == 6
    assert sum(case.helpful_opportunity for case in fixture.cases) == 3
    assert runtime.served_model == "anamnesis-openmemory-v7"
    assert hashlib.sha256(CANONICALIZER_PATH.read_bytes()).hexdigest() == (
        pin.canonicalizer_source_sha256
    )
    assert "omr1_" not in FIXTURE_PATH.read_text()


def test_v7_expected_helpful_actions_are_reachable_by_frozen_rules() -> None:
    fixture = V7Fixture.model_validate_json(FIXTURE_PATH.read_text())
    raw_payloads = (
        {
            "subject": "send foundry inspection form",
            "recipient": "Helix Safety Desk",
            "room": "73 Moss Lane",
        },
        {
            "subject": "track shipment",
            "item": "geology cores",
            "shipment": "CORE-991",
        },
        {"subject": "upload report", "project": "Emerald Current"},
    )
    for case, payload in zip(fixture.cases[:3], raw_payloads, strict=True):
        raw = Decision(
            actions=[
                ProposedAction(
                    kind="reminder",
                    action_key=case.event.id,
                    payload=payload,
                    summary="Execute action",
                    evidence_event_ids=[case.event.id],
                )
            ]
        )
        normalized = canonicalize_immediate_decision(
            event=case.event,
            retrospective_recall=tuple(item.content for item in case.memories[:1]),
            decision=raw,
        )
        assert _correct(case.expected, normalized.decision)


def test_v7_controls_do_not_require_canonicalizer_changes() -> None:
    fixture = V7Fixture.model_validate_json(FIXTURE_PATH.read_text())
    no_action = Decision(actions=[])
    for case in (fixture.cases[4],):
        normalized = canonicalize_immediate_decision(
            event=case.event,
            retrospective_recall=tuple(item.content for item in case.memories),
            decision=no_action,
        )
        assert normalized.decision == no_action
        assert not normalized.changes
