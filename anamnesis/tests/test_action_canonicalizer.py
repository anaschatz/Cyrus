from anamnesis.action_canonicalizer import canonicalize_immediate_decision
from anamnesis.schema import Decision, ObservableEvent, ProposedAction


def _event(text: str) -> ObservableEvent:
    return ObservableEvent(
        id="event_1",
        at="2042-01-01T09:00:00+02:00",
        kind="user_message",
        text=text,
    )


def _decision(payload: dict[str, str]) -> Decision:
    return Decision(
        actions=[
            ProposedAction(
                kind="reminder",
                action_key="event_1",
                payload=payload,
                summary="Execute requested action",
                evidence_event_ids=["event_1"],
            )
        ]
    )


def test_moves_grounded_address_from_wrong_optional_slot() -> None:
    event = _event("Act now: send the permit to my usual agent.")
    result = canonicalize_immediate_decision(
        event=event,
        retrospective_recall=("The agent is at 44 Amber Crescent.",),
        decision=_decision(
            {
                "subject": "send permit",
                "recipient": "Lumen Office",
                "room": "44 Amber Crescent",
            }
        ),
    )
    assert result.decision.actions[0].payload == {
        "subject": "send permit",
        "recipient": "Lumen Office",
        "address": "44 Amber Crescent",
    }


def test_does_not_move_ungrounded_address_like_value() -> None:
    event = _event("Act now: send the permit.")
    result = canonicalize_immediate_decision(
        event=event,
        retrospective_recall=None,
        decision=_decision({"subject": "send permit", "room": "44 Amber Crescent"}),
    )
    assert result.decision.actions[0].payload["room"] == "44 Amber Crescent"


def test_track_shipment_uses_source_grounded_item_as_subject_object() -> None:
    event = _event("Act now: track the shipment for my prism lenses.")
    result = canonicalize_immediate_decision(
        event=event,
        retrospective_recall=("The shipment identifier is PRISM-72.",),
        decision=_decision(
            {
                "subject": "track shipment",
                "item": "prism lenses",
                "shipment": "PRISM-72",
                "recipient": "my",
            }
        ),
    )
    assert result.decision.actions[0].payload == {
        "subject": "track prism lenses",
        "shipment": "PRISM-72",
    }


def test_upload_report_rewrites_only_from_explicit_event_shape() -> None:
    event = _event("Act now: upload the latest report for my river sediment study.")
    result = canonicalize_immediate_decision(
        event=event,
        retrospective_recall=("The study is Delta Silt Review.",),
        decision=_decision(
            {"subject": "upload report", "project": "Delta Silt Review"}
        ),
    )
    assert result.decision.actions[0].payload == {
        "subject": "upload river sediment report",
        "project": "Delta Silt Review",
    }


def test_no_action_and_wrong_provenance_are_unchanged() -> None:
    event = _event("Act now: track the shipment for my prism lenses.")
    assert not canonicalize_immediate_decision(
        event=event, retrospective_recall=(), decision=Decision(actions=[])
    ).changes
    wrong = _decision({"subject": "track shipment", "item": "prism lenses"})
    wrong.actions[0].evidence_event_ids = ["other"]
    result = canonicalize_immediate_decision(
        event=event, retrospective_recall=(), decision=wrong
    )
    assert result.decision == wrong
