from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from anamnesis.local_wire import (
    LocalActionTemplateWire,
    LocalConditionWire,
    LocalFactAssertionWire,
)
from anamnesis.memory import (
    ActionTemplate,
    AtTrigger,
    CancelIntent,
    CompilerRequest,
    Condition,
    ConditionTransitionTrigger,
    CreateIntent,
    DeterministicCompiler,
    FactKey,
    FactKeyTemplate,
    InMemoryAnamnesis,
    MemoryDelta,
    RecurringTrigger,
    SetFact,
    TruthValue,
    UpdateIntent,
)
from anamnesis.schema import Decision, ObservableEvent, ProposedAction


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def event(
    event_id: str,
    at: str,
    *,
    kind: str = "user_message",
    text: str | None = None,
) -> ObservableEvent:
    return ObservableEvent(
        id=event_id,
        at=dt(at),
        kind=kind,
        text=text or event_id,
    )


def empty_checkpoint(
    memory: InMemoryAnamnesis,
    current: ObservableEvent,
) -> None:
    result = memory.ingest(
        current,
        None if current.kind == "clock_tick" else MemoryDelta(),
    )
    assert result.accepted
    assert not memory.select(current).due_candidates
    memory.commit(current, Decision())


def create_at_intent(
    *,
    intent_id: str = "assignment",
    at: str = "2026-01-02T09:00:00+00:00",
    required: tuple[Condition, ...] = (),
    blockers: tuple[Condition, ...] = (),
    payload: dict[str, str] | None = None,
) -> CreateIntent:
    return CreateIntent(
        intent_id=intent_id,
        trigger=AtTrigger(at=dt(at)),
        required_conditions=required,
        blockers=blockers,
        action_template=ActionTemplate(
            payload=payload or {"subject": "submit the assignment"},
            summary="Submit assignment",
        ),
    )


def proposed(candidate, evidence: list[str] | None = None) -> ProposedAction:
    return ProposedAction(
        kind=candidate.action_template.kind,
        action_key=candidate.action_key,
        payload=candidate.action_template.payload,
        summary=candidate.action_template.summary,
        evidence_event_ids=evidence or list(candidate.evidence_event_ids),
    )


def test_observable_event_is_strict_and_has_no_author_annotations() -> None:
    with pytest.raises(ValidationError):
        ObservableEvent.model_validate(
            {
                "id": "e1",
                "at": "2026-01-01T09:00:00+00:00",
                "kind": "user_message",
                "text": "hello",
                "supersedes": ["hidden-gold"],
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"subject": "Assignment"},
        {"subject": "assignment"},
        {"subject": "submit the assignment", "weekday": "Monday"},
        {"subject": "submit the {unknown}"},
    ],
)
def test_action_template_rejects_noncanonical_payloads(
    payload: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        ActionTemplate(payload=payload, summary="Invalid compiler payload")


def test_fact_reaffirmation_creates_revision_and_hash_is_deterministic() -> None:
    def replay() -> InMemoryAnamnesis:
        memory = InMemoryAnamnesis()
        key = FactKey(entity="assignment", attribute="submitted")
        first = event("e1", "2026-01-01T09:00:00+00:00")
        assert memory.ingest(
            first,
            MemoryDelta(mutations=(SetFact(key=key, value=False),)),
        ).accepted
        memory.select(first)
        memory.commit(first, Decision())
        second = event("e2", "2026-01-01T10:00:00+00:00")
        assert memory.ingest(
            second,
            MemoryDelta(mutations=(SetFact(key=key, value=False),)),
        ).accepted
        memory.select(second)
        memory.commit(second, Decision())
        return memory

    memory = replay()
    first, second = memory.fact_revisions
    assert first.valid_to == dt("2026-01-01T10:00:00+00:00")
    assert second.previous_revision_id == first.revision_id
    assert second.source_event_id == "e2"
    assert second.revision == 2
    assert memory.state_hash() == replay().state_hash()


def test_semantic_delta_failure_is_atomic_but_audited() -> None:
    memory = InMemoryAnamnesis()
    current = event("e1", "2026-01-01T09:00:00+00:00")
    result = memory.ingest(
        current,
        MemoryDelta(
            mutations=(
                SetFact(
                    key=FactKey(entity="assignment", attribute="submitted"),
                    value=True,
                ),
                UpdateIntent(
                    intent_id="missing",
                    action_template=ActionTemplate(
                        payload={"subject": "update the reminder"},
                        summary="changed",
                    ),
                ),
            )
        ),
    )
    assert not result.accepted
    assert "missing or inactive" in (result.error or "")
    assert memory.events == (current,)
    assert not memory.fact_revisions
    assert not memory.intent_revisions
    assert memory.delta_audit[-1].accepted is False


def test_validate_delta_is_a_side_effect_free_exact_reducer_dry_run() -> None:
    memory = InMemoryAnamnesis()
    current = event("create", "2026-01-01T09:00:00+00:00")
    delta = MemoryDelta(mutations=(create_at_intent(),))
    before_hash = memory.state_hash()

    memory.validate_delta(current, delta)

    assert memory.state_hash() == before_hash
    assert not memory.events
    assert not memory.intent_revisions
    with pytest.raises(ValueError, match="missing or inactive"):
        memory.validate_delta(
            current,
            MemoryDelta(
                mutations=(
                    UpdateIntent(
                        intent_id="missing",
                        action_template=create_at_intent().action_template,
                    ),
                )
            ),
        )
    assert memory.state_hash() == before_hash


def test_compiler_state_exposes_semantics_without_revision_internals() -> None:
    memory = InMemoryAnamnesis()
    current = event("create", "2026-01-01T09:00:00+00:00")
    result = memory.ingest(
        current,
        MemoryDelta(
            mutations=(
                SetFact(
                    key=FactKey(entity="assignment", attribute="submitted"),
                    value=False,
                ),
                create_at_intent(),
            )
        ),
    )
    assert result.accepted

    state = json.loads(memory.compiler_state())

    assert state == {
        "facts": [
            {
                "attribute": "submitted",
                "entity": "assignment",
                "unit": None,
                "value": False,
            }
        ],
        "intents": [
            {
                "action_template": {
                    "kind": "reminder",
                    "payload": {"subject": "submit the assignment"},
                    "summary": "Submit assignment",
                },
                "blockers": [],
                "intent_id": "assignment",
                "required_conditions": [],
                "trigger": {
                    "at": "2026-01-02T09:00:00Z",
                    "type": "at",
                },
            }
        ],
    }
    serialized = memory.compiler_state()
    for forbidden in (
        "action_key",
        "field_provenance",
        "previous_revision_id",
        "revision_id",
        "source_event_id",
        "status",
        "valid_from",
        "valid_to",
    ):
        assert forbidden not in serialized


def test_fact_key_canonical_identity_cannot_collide_across_dotted_fields() -> None:
    left = FactKey(entity="a.b", attribute="c")
    right = FactKey(entity="a", attribute="b.c")

    assert left.display == right.display == "a.b.c"
    assert left.canonical != right.canonical

    memory = InMemoryAnamnesis()
    current = event("facts", "2026-01-01T09:00:00+00:00")
    assert memory.ingest(
        current,
        MemoryDelta(
            mutations=(
                SetFact(key=left, value="left"),
                SetFact(key=right, value="right"),
            )
        ),
    ).accepted
    assert len(memory.current_facts) == 2
    assert (
        memory.evaluate_condition(Condition(key=left, operator="eq", value="left"))
        == TruthValue.TRUE
    )
    assert (
        memory.evaluate_condition(Condition(key=right, operator="eq", value="right"))
        == TruthValue.TRUE
    )


def test_compiler_state_fact_and_condition_shapes_are_local_wire_aligned() -> None:
    memory = InMemoryAnamnesis()
    key = FactKey(entity="weather.station", attribute="temperature")
    current = event("create", "2026-01-01T09:00:00+00:00")
    assert memory.ingest(
        current,
        MemoryDelta(
            mutations=(
                SetFact(key=key, value=18, unit="celsius"),
                create_at_intent(
                    required=(
                        Condition(
                            key=key,
                            operator="gte",
                            value=15,
                            unit="celsius",
                        ),
                    ),
                ),
            )
        ),
    ).accepted

    state = json.loads(memory.compiler_state())
    fact = LocalFactAssertionWire.model_validate(state["facts"][0])
    condition = LocalConditionWire.model_validate(
        state["intents"][0]["required_conditions"][0]
    )
    template = LocalActionTemplateWire.model_validate(
        state["intents"][0]["action_template"]
    )

    assert fact.to_domain() == SetFact(key=key, value=18, unit="celsius")
    assert condition.to_domain() == Condition(
        key=key,
        operator="gte",
        value=15,
        unit="celsius",
    )
    assert template.to_domain() == create_at_intent().action_template


@pytest.mark.parametrize(
    "trigger_at",
    ["2026-01-01T08:59:59+00:00", "2026-01-01T09:00:00+00:00"],
)
def test_create_rejects_nonfuture_at_trigger(trigger_at: str) -> None:
    memory = InMemoryAnamnesis()
    current = event("create", "2026-01-01T09:00:00+00:00")

    result = memory.ingest(
        current,
        MemoryDelta(mutations=(create_at_intent(at=trigger_at),)),
    )

    assert not result.accepted
    assert result.error == "at trigger must be strictly after its source event"
    assert not memory.intent_revisions


def test_update_rejects_nonfuture_at_trigger_atomically() -> None:
    memory = InMemoryAnamnesis()
    created = event("create", "2026-01-01T09:00:00+00:00")
    assert memory.ingest(
        created,
        MemoryDelta(mutations=(create_at_intent(),)),
    ).accepted
    memory.select(created)
    memory.commit(created, Decision())
    updated = event("update", "2026-01-01T10:00:00+00:00")

    result = memory.ingest(
        updated,
        MemoryDelta(
            mutations=(
                UpdateIntent(
                    intent_id="assignment",
                    trigger=AtTrigger(at=updated.at),
                ),
            )
        ),
    )

    assert not result.accepted
    assert result.error == "at trigger must be strictly after its source event"
    assert memory.current_intents[0].revision == 1


def test_update_rejects_semantic_noop_without_new_revision() -> None:
    memory = InMemoryAnamnesis()
    created = event("create", "2026-01-01T09:00:00+00:00")
    original = create_at_intent()
    assert memory.ingest(
        created,
        MemoryDelta(mutations=(original,)),
    ).accepted
    memory.select(created)
    memory.commit(created, Decision())
    updated = event("update", "2026-01-01T10:00:00+00:00")

    result = memory.ingest(
        updated,
        MemoryDelta(
            mutations=(
                UpdateIntent(
                    intent_id=original.intent_id,
                    action_template=original.action_template,
                ),
            )
        ),
    )

    assert not result.accepted
    assert result.error == "intent update does not change active semantics"
    assert len(memory.intent_revisions) == 1
    assert memory.current_intents[0].revision == 1


def test_recurring_trigger_rejects_ranges_larger_than_one_year() -> None:
    with pytest.raises(ValidationError, match="cannot exceed 366 days"):
        RecurringTrigger(
            local_time="09:00:00",
            weekdays=("monday",),
            start_date="2026-01-01",
            end_date="2027-01-03",
            timezone="Europe/Athens",
        )


def test_create_rejects_recurring_contract_without_future_occurrence() -> None:
    memory = InMemoryAnamnesis()
    current = event("create", "2026-01-05T12:00:00+02:00")
    recurrence = RecurringTrigger(
        local_time="09:00:00",
        weekdays=("monday",),
        start_date="2026-01-05",
        end_date="2026-01-05",
        timezone="Europe/Athens",
    )

    result = memory.ingest(
        current,
        MemoryDelta(
            mutations=(
                CreateIntent(
                    intent_id="past-recurrence",
                    trigger=recurrence,
                    action_template=ActionTemplate(
                        payload={"subject": "check recurrence"},
                        summary="Check recurrence",
                    ),
                ),
            )
        ),
    )

    assert not result.accepted
    assert result.error == (
        "recurring trigger has no valid occurrence after its source event"
    )


def test_create_rejects_condition_window_that_has_already_closed() -> None:
    memory = InMemoryAnamnesis()
    current = event("create", "2026-01-05T12:00:00+00:00")
    key = FactKey(entity="permit", attribute="approved")
    transition = ConditionTransitionTrigger(
        active_from=dt("2026-01-01T09:00:00+00:00"),
        active_until=current.at,
    )

    result = memory.ingest(
        current,
        MemoryDelta(
            mutations=(
                CreateIntent(
                    intent_id="closed-window",
                    trigger=transition,
                    required_conditions=(
                        Condition(key=key, operator="eq", value=True),
                    ),
                    action_template=ActionTemplate(
                        payload={"subject": "check permit"},
                        summary="Check permit",
                    ),
                ),
            )
        ),
    )

    assert not result.accepted
    assert result.error == (
        "condition trigger window must extend past its source event"
    )


def test_duplicate_targets_are_rejected_before_reducer() -> None:
    key = FactKey(entity="weather", attribute="temperature")
    with pytest.raises(ValidationError, match="duplicate fact mutation"):
        MemoryDelta(
            mutations=(
                SetFact(key=key, value=20, unit="celsius"),
                SetFact(key=key, value=21, unit="celsius"),
            )
        )


def test_invalid_combined_intent_update_is_rejected_atomically() -> None:
    memory = InMemoryAnamnesis()
    created = event("create", "2026-03-02T09:00:00+02:00")
    template_condition = Condition(
        key=FactKeyTemplate(
            entity="lab_notes.{date}",
            attribute="uploaded",
        ),
        operator="eq",
        value=True,
    )
    memory.ingest(
        created,
        MemoryDelta(
            mutations=(
                CreateIntent(
                    intent_id="lab-notes",
                    trigger=RecurringTrigger(
                        local_time="18:00:00",
                        weekdays=("monday",),
                        start_date="2026-03-02",
                        end_date="2026-03-02",
                        timezone="Europe/Athens",
                    ),
                    blockers=(template_condition,),
                    action_template=ActionTemplate(
                        payload={"subject": "upload lab notes"},
                        summary="Upload lab notes",
                    ),
                ),
            )
        ),
    )
    memory.select(created)
    memory.commit(created, Decision())
    original = memory.current_intents[0]

    invalid_update = event("update", "2026-03-02T10:00:00+02:00")
    result = memory.ingest(
        invalid_update,
        MemoryDelta(
            mutations=(
                UpdateIntent(
                    intent_id="lab-notes",
                    trigger=AtTrigger(at=dt("2026-03-02T18:00:00+02:00")),
                ),
            )
        ),
    )
    assert not result.accepted
    assert memory.current_intents == (original,)
    assert len(memory.intent_revisions) == 1


def test_condition_evaluation_is_three_valued_and_type_safe() -> None:
    memory = InMemoryAnamnesis()
    key = FactKey(entity="weather", attribute="temperature")
    threshold = Condition(key=key, operator="gte", value=30, unit="celsius")
    assert memory.evaluate_condition(threshold) == TruthValue.UNKNOWN

    observed = event("e1", "2026-01-01T09:00:00+00:00", kind="observation")
    assert memory.ingest(
        observed,
        MemoryDelta(mutations=(SetFact(key=key, value=31, unit="celsius"),)),
    ).accepted
    assert memory.evaluate_condition(threshold) == TruthValue.TRUE
    assert (
        memory.evaluate_condition(
            Condition(key=key, operator="gte", value=30, unit="fahrenheit")
        )
        == TruthValue.UNKNOWN
    )
    memory.select(observed)
    memory.commit(observed, Decision())

    bool_event = event("e2", "2026-01-01T10:00:00+00:00")
    bool_key = FactKey(entity="assignment", attribute="submitted")
    memory.ingest(
        bool_event,
        MemoryDelta(mutations=(SetFact(key=bool_key, value=True),)),
    )
    assert (
        memory.evaluate_condition(Condition(key=bool_key, operator="eq", value=1))
        == TruthValue.FALSE
    )


def test_deadline_update_keeps_action_key_and_changes_only_field_provenance() -> None:
    memory = InMemoryAnamnesis()
    created = event("create", "2026-01-01T09:00:00+00:00")
    memory.ingest(
        created,
        MemoryDelta(mutations=(create_at_intent(),)),
    )
    memory.select(created)
    memory.commit(created, Decision())

    updated = event("update", "2026-01-01T12:00:00+00:00")
    new_trigger = AtTrigger(at=dt("2026-01-03T09:00:00+00:00"))
    assert memory.ingest(
        updated,
        MemoryDelta(
            mutations=(UpdateIntent(intent_id="assignment", trigger=new_trigger),)
        ),
    ).accepted
    memory.select(updated)
    memory.commit(updated, Decision())

    old, new = memory.intent_revisions
    assert old.valid_to == updated.at
    assert new.action_key == "create"
    assert new.previous_revision_id == old.revision_id
    assert new.field_provenance["trigger.at"] == ("update",)
    assert new.field_provenance["trigger.type"] == ("create",)
    assert new.field_provenance["action_template.payload.subject"] == ("create",)
    assert new.field_provenance["action_template.summary"] == ("create",)
    assert new.field_provenance["action_key"] == ("create",)

    old_deadline = event(
        "old-deadline",
        "2026-01-02T09:00:00+00:00",
        kind="clock_tick",
    )
    empty_checkpoint(memory, old_deadline)
    due_event = event(
        "new-deadline",
        "2026-01-03T09:00:00+00:00",
        kind="clock_tick",
    )
    memory.ingest(due_event, None)
    selection = memory.select(due_event)
    assert len(selection.due_candidates) == 1
    candidate = selection.due_candidates[0]
    assert candidate.action_key == "create"
    assert candidate.evidence_event_ids == ("create", "update", "new-deadline")


def test_successive_action_template_updates_retain_independent_leaf_sources() -> None:
    memory = InMemoryAnamnesis()
    created = event("e1", "2026-01-01T09:00:00+00:00")
    memory.ingest(
        created,
        MemoryDelta(
            mutations=(
                create_at_intent(
                    payload={
                        "subject": "send the assignment",
                        "recipient": "Professor Ada",
                        "address": "Old Campus",
                    }
                ),
            )
        ),
    )
    memory.select(created)
    memory.commit(created, Decision())

    recipient_update = event("e2", "2026-01-01T10:00:00+00:00")
    memory.ingest(
        recipient_update,
        MemoryDelta(
            mutations=(
                UpdateIntent(
                    intent_id="assignment",
                    action_template=ActionTemplate(
                        payload={
                            "subject": "send the assignment",
                            "recipient": "Professor Grace",
                            "address": "Old Campus",
                        },
                        summary="Submit assignment",
                    ),
                ),
            )
        ),
    )
    memory.select(recipient_update)
    memory.commit(recipient_update, Decision())

    address_update = event("e3", "2026-01-01T11:00:00+00:00")
    memory.ingest(
        address_update,
        MemoryDelta(
            mutations=(
                UpdateIntent(
                    intent_id="assignment",
                    action_template=ActionTemplate(
                        payload={
                            "subject": "send the assignment",
                            "recipient": "Professor Grace",
                            "address": "New Campus",
                        },
                        summary="Submit assignment",
                    ),
                ),
            )
        ),
    )
    memory.select(address_update)
    memory.commit(address_update, Decision())

    current = memory.current_intents[0]
    assert current.revision == 3
    assert current.field_provenance["action_template.payload.subject"] == ("e1",)
    assert current.field_provenance["action_template.payload.recipient"] == ("e2",)
    assert current.field_provenance["action_template.payload.address"] == ("e3",)
    assert current.field_provenance["action_template.summary"] == ("e1",)
    assert current.field_provenance["trigger.at"] == ("e1",)

    due = event("due", "2026-01-02T09:00:00+00:00", kind="clock_tick")
    memory.ingest(due, None)
    selection = memory.select(due)
    assert len(selection.due_candidates) == 1
    assert selection.due_candidates[0].evidence_event_ids == (
        "e1",
        "e2",
        "e3",
        "due",
    )


def test_removed_action_leaf_remains_causal_as_an_active_tombstone() -> None:
    memory = InMemoryAnamnesis()
    created = event("e1", "2026-01-01T09:00:00+00:00")
    memory.ingest(
        created,
        MemoryDelta(
            mutations=(
                create_at_intent(
                    payload={
                        "subject": "send the assignment",
                        "recipient": "Professor Ada",
                    }
                ),
            )
        ),
    )
    memory.select(created)
    memory.commit(created, Decision())

    updated = event("e2", "2026-01-01T10:00:00+00:00")
    memory.ingest(
        updated,
        MemoryDelta(
            mutations=(
                UpdateIntent(
                    intent_id="assignment",
                    action_template=ActionTemplate(
                        payload={"subject": "send the assignment"},
                        summary="Submit assignment",
                    ),
                ),
            )
        ),
    )
    memory.select(updated)
    memory.commit(updated, Decision())

    current = memory.current_intents[0]
    removed_path = "action_template.payload.recipient.__removed__"
    assert current.field_provenance[removed_path] == ("e2",)
    due = event("due", "2026-01-02T09:00:00+00:00", kind="clock_tick")
    memory.ingest(due, None)
    selection = memory.select(due)
    assert selection.due_candidates[0].evidence_event_ids == ("e1", "e2", "due")


def test_stored_payload_and_provenance_are_deeply_immutable() -> None:
    memory = InMemoryAnamnesis()
    created = event("e1", "2026-01-01T09:00:00+00:00")
    memory.ingest(created, MemoryDelta(mutations=(create_at_intent(),)))
    memory.select(created)
    memory.commit(created, Decision())
    state_before = memory.state_hash()

    intent = memory.current_intents[0]
    with pytest.raises(TypeError):
        intent.action_template.payload["subject"] = "change the assignment"
    with pytest.raises(TypeError):
        intent.field_provenance["action_template.payload.subject"] = ("forged",)

    copied_template = intent.action_template.model_copy(
        update={"payload": dict(intent.action_template.payload)}
    )
    with pytest.raises(TypeError):
        copied_template.payload["subject"] = "change the assignment"
    copied_intent = intent.model_copy(
        update={"field_provenance": dict(intent.field_provenance)}
    )
    with pytest.raises(TypeError):
        copied_intent.field_provenance["action_template.payload.subject"] = ("forged",)
    assert memory.state_hash() == state_before


def test_cancellation_prevents_future_occurrence() -> None:
    memory = InMemoryAnamnesis()
    created = event("create", "2026-01-01T09:00:00+00:00")
    memory.ingest(created, MemoryDelta(mutations=(create_at_intent(),)))
    memory.select(created)
    memory.commit(created, Decision())
    cancelled = event("cancel", "2026-01-01T10:00:00+00:00")
    memory.ingest(
        cancelled,
        MemoryDelta(mutations=(CancelIntent(intent_id="assignment"),)),
    )
    memory.select(cancelled)
    memory.commit(cancelled, Decision())

    assert memory.current_intents[0].status == "cancelled"
    due = event("due", "2026-01-02T09:00:00+00:00", kind="clock_tick")
    empty_checkpoint(memory, due)
    assert not memory.occurrence_revisions


def test_required_unknown_suppresses_but_unknown_blocker_does_not() -> None:
    missing = FactKey(entity="assignment", attribute="submitted")
    required_memory = InMemoryAnamnesis()
    create_required = event("required", "2026-01-01T09:00:00+00:00")
    required_memory.ingest(
        create_required,
        MemoryDelta(
            mutations=(
                create_at_intent(
                    required=(Condition(key=missing, operator="eq", value=True),)
                ),
            )
        ),
    )
    required_memory.select(create_required)
    required_memory.commit(create_required, Decision())
    due = event("due", "2026-01-02T09:00:00+00:00", kind="clock_tick")
    required_memory.ingest(due, None)
    assert not required_memory.select(due).due_candidates
    assert required_memory.occurrence_revisions[-1].status == "suppressed"

    blocker_memory = InMemoryAnamnesis()
    create_blocker = event("blocker", "2026-01-01T09:00:00+00:00")
    blocker_memory.ingest(
        create_blocker,
        MemoryDelta(
            mutations=(
                create_at_intent(
                    blockers=(Condition(key=missing, operator="eq", value=True),)
                ),
            )
        ),
    )
    blocker_memory.select(create_blocker)
    blocker_memory.commit(create_blocker, Decision())
    blocker_due = event(
        "blocker-due",
        "2026-01-02T09:00:00+00:00",
        kind="clock_tick",
    )
    blocker_memory.ingest(blocker_due, None)
    assert len(blocker_memory.select(blocker_due).due_candidates) == 1


def test_commit_executes_exact_candidate_then_prevents_duplicate() -> None:
    memory = InMemoryAnamnesis()
    created = event("create", "2026-01-01T09:00:00+00:00")
    memory.ingest(created, MemoryDelta(mutations=(create_at_intent(),)))
    memory.select(created)
    memory.commit(created, Decision())
    due = event("due", "2026-01-02T09:00:00+00:00", kind="clock_tick")
    memory.ingest(due, None)
    candidate = memory.select(due).due_candidates[0]
    committed = memory.commit(
        due,
        Decision(actions=[proposed(candidate, ["create", "invented-future"])]),
    )
    assert committed.executed_occurrence_ids == (candidate.occurrence_id,)
    assert memory.executions[0].evidence_event_ids == ("create", "due")
    assert memory.occurrence_revisions[-1].status == "executed"

    later = event("later", "2026-01-02T10:00:00+00:00", kind="clock_tick")
    empty_checkpoint(memory, later)
    assert len(memory.executions) == 1


def test_unitless_condition_does_not_match_a_unit_bearing_fact() -> None:
    memory = InMemoryAnamnesis()
    key = FactKey(entity="tank", attribute="temperature")
    current = event("temperature", "2026-01-01T09:00:00+00:00")
    assert memory.ingest(
        current,
        MemoryDelta(mutations=(SetFact(key=key, value=20, unit="celsius"),)),
    ).accepted

    assert (
        memory.evaluate_condition(Condition(key=key, operator="gte", value=18))
        == TruthValue.UNKNOWN
    )
    assert (
        memory.evaluate_condition(
            Condition(key=key, operator="gte", value=18, unit="celsius")
        )
        == TruthValue.TRUE
    )


def test_wrong_payload_never_marks_due_occurrence_executed() -> None:
    memory = InMemoryAnamnesis()
    created = event("create", "2026-01-01T09:00:00+00:00")
    memory.ingest(created, MemoryDelta(mutations=(create_at_intent(),)))
    memory.select(created)
    memory.commit(created, Decision())
    due = event("due", "2026-01-02T09:00:00+00:00", kind="clock_tick")
    memory.ingest(due, None)
    candidate = memory.select(due).due_candidates[0]
    wrong = ProposedAction(
        kind="reminder",
        action_key=candidate.action_key,
        payload={"subject": "send wrong payload"},
        summary="Wrong payload but already emitted",
        evidence_event_ids=["create"],
    )
    result = memory.commit(due, Decision(actions=[wrong]))
    assert not result.executed_occurrence_ids
    assert result.expired_occurrence_ids == (candidate.occurrence_id,)
    assert memory.occurrence_revisions[-1].status == "expired"
    assert not memory.executions


def test_unemitted_due_occurrence_expires() -> None:
    memory = InMemoryAnamnesis()
    created = event("create", "2026-01-01T09:00:00+00:00")
    memory.ingest(created, MemoryDelta(mutations=(create_at_intent(),)))
    memory.select(created)
    memory.commit(created, Decision())
    due = event("due", "2026-01-02T09:00:00+00:00", kind="clock_tick")
    memory.ingest(due, None)
    candidate = memory.select(due).due_candidates[0]
    result = memory.commit(due, Decision())
    assert result.expired_occurrence_ids == (candidate.occurrence_id,)
    assert memory.occurrence_revisions[-1].status == "expired"


def test_recurring_occurrences_are_independent_and_resolve_placeholders() -> None:
    memory = InMemoryAnamnesis()
    created = event("create", "2026-01-04T10:00:00+02:00")
    recurrence = RecurringTrigger(
        local_time="09:00:00",
        weekdays=("monday", "tuesday"),
        start_date="2026-01-05",
        end_date="2026-01-06",
        timezone="Europe/Athens",
    )
    memory.ingest(
        created,
        MemoryDelta(
            mutations=(
                CreateIntent(
                    intent_id="medicine",
                    trigger=recurrence,
                    action_template=ActionTemplate(
                        payload={
                            "subject": "take the {weekday} medicine",
                            "date": "{date}",
                        },
                        summary="Take medicine on {weekday}",
                    ),
                ),
            )
        ),
    )
    memory.select(created)
    memory.commit(created, Decision())

    monday = event(
        "monday",
        "2026-01-05T09:00:00+02:00",
        kind="clock_tick",
    )
    memory.ingest(monday, None)
    first = memory.select(monday).due_candidates[0]
    assert first.action_template.payload["subject"] == "take the monday medicine"
    assert first.action_template.payload["date"] == "2026-01-05"
    assert first.action_template.summary == "Take medicine on Monday"
    memory.commit(monday, Decision(actions=[proposed(first)]))

    tuesday = event(
        "tuesday",
        "2026-01-06T09:00:00+02:00",
        kind="clock_tick",
    )
    memory.ingest(tuesday, None)
    second_selection = memory.select(tuesday)
    second = second_selection.due_candidates[0]
    assert second.occurrence_id != first.occurrence_id
    assert second.action_template.payload["subject"] == "take the tuesday medicine"
    assert second.action_template.summary == "Take medicine on Tuesday"
    assert {block.kind for block in second_selection.view.blocks} == {
        "due_candidate",
        "execution",
    }
    memory.commit(tuesday, Decision(actions=[proposed(second)]))
    assert len(memory.executions) == 2


def test_recurring_fact_key_template_is_occurrence_local() -> None:
    def replay(*, tuesday_completed: bool):
        memory = InMemoryAnamnesis()
        created = event("create", "2026-03-02T09:00:00+02:00")
        blocker = Condition(
            key=FactKeyTemplate(
                entity="lab_notes.{date}",
                attribute="uploaded",
            ),
            operator="eq",
            value=True,
        )
        memory.ingest(
            created,
            MemoryDelta(
                mutations=(
                    CreateIntent(
                        intent_id="lab-notes",
                        trigger=RecurringTrigger(
                            local_time="18:00:00",
                            weekdays=("monday", "tuesday"),
                            start_date="2026-03-02",
                            end_date="2026-03-03",
                            timezone="Europe/Athens",
                        ),
                        blockers=(blocker,),
                        action_template=ActionTemplate(
                            payload={
                                "subject": "upload the lab notes",
                                "date": "{date}",
                            },
                            summary="Upload {weekday} lab notes",
                        ),
                    ),
                )
            ),
        )
        memory.select(created)
        memory.commit(created, Decision())
        monday_done = event(
            "monday-done",
            "2026-03-02T17:00:00+02:00",
            kind="observation",
        )
        memory.ingest(
            monday_done,
            MemoryDelta(
                mutations=(
                    SetFact(
                        key=FactKey(
                            entity="lab_notes.2026-03-02",
                            attribute="uploaded",
                        ),
                        value=True,
                    ),
                )
            ),
        )
        memory.select(monday_done)
        memory.commit(monday_done, Decision())
        monday = event(
            "monday",
            "2026-03-02T18:00:00+02:00",
            kind="clock_tick",
        )
        memory.ingest(monday, None)
        assert not memory.select(monday).due_candidates
        assert memory.occurrence_revisions[-1].status == "suppressed"
        memory.commit(monday, Decision())

        if tuesday_completed:
            tuesday_done = event(
                "tuesday-done",
                "2026-03-03T17:00:00+02:00",
                kind="observation",
            )
            memory.ingest(
                tuesday_done,
                MemoryDelta(
                    mutations=(
                        SetFact(
                            key=FactKey(
                                entity="lab_notes.2026-03-03",
                                attribute="uploaded",
                            ),
                            value=True,
                        ),
                    )
                ),
            )
            memory.select(tuesday_done)
            memory.commit(tuesday_done, Decision())
        tuesday = event(
            "tuesday",
            "2026-03-03T18:00:00+02:00",
            kind="clock_tick",
        )
        memory.ingest(tuesday, None)
        return memory, tuesday, memory.select(tuesday)

    open_memory, open_tuesday, selection = replay(tuesday_completed=False)
    assert len(selection.due_candidates) == 1
    assert selection.due_candidates[0].action_template.payload == {
        "subject": "upload the lab notes",
        "date": "2026-03-03",
    }
    open_memory.commit(open_tuesday, Decision())

    blocked_memory, blocked_tuesday, selection = replay(tuesday_completed=True)
    assert not selection.due_candidates
    assert blocked_memory.occurrence_revisions[-1].status == "suppressed"
    assert blocked_memory.occurrence_revisions[-1].evidence_event_ids == (
        "create",
        "tuesday-done",
        "tuesday",
    )
    blocked_memory.commit(blocked_tuesday, Decision())


def test_nonexistent_dst_wall_time_is_skipped() -> None:
    memory = InMemoryAnamnesis()
    created = event("create", "2026-03-28T09:00:00+02:00")
    memory.ingest(
        created,
        MemoryDelta(
            mutations=(
                CreateIntent(
                    intent_id="dst-check",
                    trigger=RecurringTrigger(
                        local_time="03:30:00",
                        weekdays=("sunday",),
                        start_date="2026-03-29",
                        end_date="2026-03-29",
                        timezone="Europe/Athens",
                    ),
                    action_template=ActionTemplate(
                        payload={"subject": "check the clock"},
                        summary="DST reminder",
                    ),
                ),
            )
        ),
    )
    memory.select(created)
    memory.commit(created, Decision())
    after_jump = event(
        "after-jump",
        "2026-03-29T04:30:00+03:00",
        kind="clock_tick",
    )
    empty_checkpoint(memory, after_jump)
    assert not memory.occurrence_revisions


def test_condition_transition_has_no_creation_ack_and_is_one_shot() -> None:
    memory = InMemoryAnamnesis()
    key = FactKey(entity="weather", attribute="temperature")
    transition = ConditionTransitionTrigger(
        active_from=dt("2026-01-01T09:00:00+00:00"),
        active_until=dt("2026-01-07T09:00:00+00:00"),
    )
    created = event("create", "2026-01-01T09:00:00+00:00")
    memory.ingest(
        created,
        MemoryDelta(
            mutations=(
                SetFact(key=key, value=30, unit="celsius"),
                CreateIntent(
                    intent_id="heat-alert",
                    trigger=transition,
                    required_conditions=(
                        Condition(
                            key=key,
                            operator="gte",
                            value=30,
                            unit="celsius",
                        ),
                    ),
                    action_template=ActionTemplate(
                        payload={"subject": "check the heat"},
                        summary="Heat alert",
                    ),
                ),
            )
        ),
    )
    assert not memory.select(created).due_candidates
    memory.commit(created, Decision())

    cool = event("cool", "2026-01-02T09:00:00+00:00", kind="observation")
    memory.ingest(
        cool,
        MemoryDelta(mutations=(SetFact(key=key, value=20, unit="celsius"),)),
    )
    assert not memory.select(cool).due_candidates
    memory.commit(cool, Decision())

    hot = event("hot", "2026-01-03T09:00:00+00:00", kind="observation")
    memory.ingest(
        hot,
        MemoryDelta(mutations=(SetFact(key=key, value=31, unit="celsius"),)),
    )
    candidate = memory.select(hot).due_candidates[0]
    memory.commit(hot, Decision(actions=[proposed(candidate)]))

    cool_again = event(
        "cool-again",
        "2026-01-04T09:00:00+00:00",
        kind="observation",
    )
    memory.ingest(
        cool_again,
        MemoryDelta(mutations=(SetFact(key=key, value=20, unit="celsius"),)),
    )
    assert not memory.select(cool_again).due_candidates
    memory.commit(cool_again, Decision())
    hot_again = event(
        "hot-again",
        "2026-01-05T09:00:00+00:00",
        kind="observation",
    )
    memory.ingest(
        hot_again,
        MemoryDelta(mutations=(SetFact(key=key, value=32, unit="celsius"),)),
    )
    assert not memory.select(hot_again).due_candidates
    memory.commit(hot_again, Decision())
    assert len(memory.executions) == 1


def test_condition_true_before_window_does_not_fire_on_window_entry() -> None:
    memory = InMemoryAnamnesis()
    key = FactKey(entity="permit", attribute="approved")
    created = event("create", "2026-01-01T09:00:00+00:00")
    transition = ConditionTransitionTrigger(
        active_from=dt("2026-01-02T09:00:00+00:00"),
        active_until=dt("2026-01-03T09:00:00+00:00"),
    )
    assert memory.ingest(
        created,
        MemoryDelta(
            mutations=(
                CreateIntent(
                    intent_id="permit-alert",
                    trigger=transition,
                    required_conditions=(
                        Condition(key=key, operator="eq", value=True),
                    ),
                    action_template=ActionTemplate(
                        payload={"subject": "check permit approval"},
                        summary="Check permit approval",
                    ),
                ),
            )
        ),
    ).accepted
    assert not memory.select(created).due_candidates
    memory.commit(created, Decision())

    approved = event("approved", "2026-01-01T12:00:00+00:00", kind="observation")
    assert memory.ingest(
        approved,
        MemoryDelta(mutations=(SetFact(key=key, value=True),)),
    ).accepted
    assert not memory.select(approved).due_candidates
    memory.commit(approved, Decision())

    window_entry = event(
        "window-entry",
        "2026-01-02T09:00:00+00:00",
        kind="clock_tick",
    )
    memory.ingest(window_entry, None)
    assert not memory.select(window_entry).due_candidates


def test_condition_contract_update_establishes_baseline_before_transition() -> None:
    memory = InMemoryAnamnesis()
    old_key = FactKey(entity="permit", attribute="approved")
    new_key = FactKey(entity="inspection", attribute="passed")
    transition = ConditionTransitionTrigger(
        active_from=dt("2026-01-01T09:00:00+00:00"),
        active_until=dt("2026-01-07T09:00:00+00:00"),
    )
    created = event("create", "2026-01-01T09:00:00+00:00")
    assert memory.ingest(
        created,
        MemoryDelta(
            mutations=(
                SetFact(key=old_key, value=False),
                SetFact(key=new_key, value=True),
                CreateIntent(
                    intent_id="permit-alert",
                    trigger=transition,
                    required_conditions=(
                        Condition(key=old_key, operator="eq", value=True),
                    ),
                    action_template=ActionTemplate(
                        payload={"subject": "check permit approval"},
                        summary="Check permit approval",
                    ),
                ),
            )
        ),
    ).accepted
    assert not memory.select(created).due_candidates
    memory.commit(created, Decision())

    updated = event("update", "2026-01-02T09:00:00+00:00")
    assert memory.ingest(
        updated,
        MemoryDelta(
            mutations=(
                UpdateIntent(
                    intent_id="permit-alert",
                    required_conditions=(
                        Condition(key=new_key, operator="eq", value=True),
                    ),
                ),
            )
        ),
    ).accepted
    assert not memory.select(updated).due_candidates
    memory.commit(updated, Decision())

    false_event = event("false", "2026-01-03T09:00:00+00:00", kind="observation")
    memory.ingest(
        false_event,
        MemoryDelta(mutations=(SetFact(key=new_key, value=False),)),
    )
    assert not memory.select(false_event).due_candidates
    memory.commit(false_event, Decision())
    true_event = event("true", "2026-01-04T09:00:00+00:00", kind="observation")
    memory.ingest(
        true_event,
        MemoryDelta(mutations=(SetFact(key=new_key, value=True),)),
    )
    assert len(memory.select(true_event).due_candidates) == 1


def test_action_template_update_does_not_hide_same_event_condition_transition() -> None:
    memory = InMemoryAnamnesis()
    key = FactKey(entity="job", attribute="ready")
    transition = ConditionTransitionTrigger(
        active_from=dt("2026-01-01T09:00:00+00:00"),
        active_until=dt("2026-01-07T09:00:00+00:00"),
    )
    created = event("create", "2026-01-01T09:00:00+00:00")
    assert memory.ingest(
        created,
        MemoryDelta(
            mutations=(
                SetFact(key=key, value=False),
                CreateIntent(
                    intent_id="job-alert",
                    trigger=transition,
                    required_conditions=(
                        Condition(key=key, operator="eq", value=True),
                    ),
                    action_template=ActionTemplate(
                        payload={"subject": "check job"},
                        summary="Check job",
                    ),
                ),
            )
        ),
    ).accepted
    assert not memory.select(created).due_candidates
    memory.commit(created, Decision())

    transitioned = event("transition", "2026-01-02T09:00:00+00:00")
    assert memory.ingest(
        transitioned,
        MemoryDelta(
            mutations=(
                SetFact(key=key, value=True),
                UpdateIntent(
                    intent_id="job-alert",
                    action_template=ActionTemplate(
                        payload={"subject": "notify job owner"},
                        summary="Notify job owner",
                    ),
                ),
            )
        ),
    ).accepted
    selection = memory.select(transitioned)
    assert len(selection.due_candidates) == 1
    assert selection.due_candidates[0].action_template.payload == {
        "subject": "notify job owner"
    }


def test_compact_view_contains_only_due_evidence_and_current_facts() -> None:
    memory = InMemoryAnamnesis()
    submitted = FactKey(entity="assignment", attribute="submitted")
    initial = event("status", "2026-01-01T08:00:00+00:00", kind="observation")
    memory.ingest(
        initial,
        MemoryDelta(mutations=(SetFact(key=submitted, value=False),)),
    )
    memory.select(initial)
    memory.commit(initial, Decision())
    created = event("create", "2026-01-01T09:00:00+00:00")
    memory.ingest(
        created,
        MemoryDelta(
            mutations=(
                create_at_intent(
                    blockers=(Condition(key=submitted, operator="eq", value=True),)
                ),
            )
        ),
    )
    memory.select(created)
    memory.commit(created, Decision())
    due = event("due", "2026-01-02T09:00:00+00:00", kind="clock_tick")
    memory.ingest(due, None)
    selection = memory.select(due)
    candidate = selection.due_candidates[0]
    assert candidate.evidence_event_ids == ("status", "create", "due")
    assert [block.kind for block in selection.view.blocks] == [
        "due_candidate",
        "fact",
    ]
    assert "assignment.submitted" in selection.view.blocks[1].content
    assert candidate.evidence_event_ids[-1] == "due"


def test_deterministic_compiler_exposes_only_request_and_defaults_to_noop() -> None:
    current = event("e1", "2026-01-01T09:00:00+00:00")
    delta = MemoryDelta(
        mutations=(
            SetFact(
                key=FactKey(entity="profile", attribute="timezone"),
                value="Europe/Athens",
            ),
        )
    )
    compiler = DeterministicCompiler({"e1": delta})
    request = CompilerRequest(event=current, active_state='{"facts":[],"intents":[]}')
    call = asyncio.run(compiler.compile(request))
    assert call.delta == delta
    assert call.cost_complete is False
    assert compiler.requests == [request]

    other = event("e2", "2026-01-01T10:00:00+00:00")
    noop = asyncio.run(
        compiler.compile(
            CompilerRequest(event=other, active_state='{"facts":[],"intents":[]}')
        )
    )
    assert noop.delta == MemoryDelta()
