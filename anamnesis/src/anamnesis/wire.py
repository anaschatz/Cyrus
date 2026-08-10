"""Provider-strict wire schemas converted into the domain evaluation models.

Inspect's generic Pydantic converter cannot preserve discriminated unions and
OpenAI strict schemas require every object field to be required. These flat,
required-but-nullable transport models keep the provider contract closed while
the domain models retain their ergonomic union shapes.
"""

from __future__ import annotations

from datetime import date, datetime, time
from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from anamnesis.memory import (
    ActionTemplate,
    AtTrigger,
    CancelIntent,
    Condition,
    ConditionTransitionTrigger,
    CreateIntent,
    FactKey,
    FactKeyTemplate,
    MemoryDelta,
    RecurringTrigger,
    SetFact,
    UpdateIntent,
    Weekday,
)
from anamnesis.schema import (
    ActionValue,
    Decision,
    ProposedAction,
    StrictModel,
)


class PayloadWire(StrictModel):
    """Closed action payload; null optional slots disappear in domain JSON."""

    subject: str
    address: str | None
    build: str | None
    date: str | None
    flight: str | None
    greenhouse: str | None
    item: str | None
    project: str | None
    quantity: int | float | None
    recipient: str | None
    room: str | None
    shipment: str | None
    tank: str | None
    trip: str | None

    def to_payload(self) -> dict[str, ActionValue]:
        return self.model_dump(exclude_none=True)


class ReminderKind(StrEnum):
    REMINDER = "reminder"


class ProposedActionWire(StrictModel):
    kind: ReminderKind
    action_key: str
    payload: PayloadWire
    summary: str
    evidence_event_ids: list[str]

    def to_domain(self) -> ProposedAction:
        return ProposedAction(
            kind=self.kind.value,
            action_key=self.action_key,
            payload=self.payload.to_payload(),
            summary=self.summary,
            evidence_event_ids=self.evidence_event_ids,
        )


class DecisionWire(StrictModel):
    actions: list[ProposedActionWire]

    def to_domain(self) -> Decision:
        return Decision(actions=[action.to_domain() for action in self.actions])


class TriggerWire(StrictModel):
    """One flat trigger with mode-inapplicable fields explicitly null."""

    type: Literal["at", "recurring", "condition_transition"]
    at: datetime | None
    local_time: time | None
    weekdays: list[Weekday] | None
    start_date: date | None
    end_date: date | None
    timezone: str | None
    active_from: datetime | None
    active_until: datetime | None

    @model_validator(mode="after")
    def validate_mode_fields(self) -> Self:
        required_by_type = {
            "at": {"at"},
            "recurring": {
                "local_time",
                "weekdays",
                "start_date",
                "end_date",
                "timezone",
            },
            "condition_transition": {"active_from", "active_until"},
        }
        all_fields = {
            "at",
            "local_time",
            "weekdays",
            "start_date",
            "end_date",
            "timezone",
            "active_from",
            "active_until",
        }
        required = required_by_type[self.type]
        missing = {name for name in required if getattr(self, name) is None}
        non_null_extras = {
            name for name in all_fields - required if getattr(self, name) is not None
        }
        if missing or non_null_extras:
            raise ValueError(
                f"invalid {self.type} trigger fields: missing={sorted(missing)}, "
                f"non_null_extras={sorted(non_null_extras)}"
            )
        return self

    def to_domain(
        self,
    ) -> AtTrigger | RecurringTrigger | ConditionTransitionTrigger:
        if self.type == "at":
            assert self.at is not None
            return AtTrigger(at=self.at)
        if self.type == "recurring":
            assert self.local_time is not None
            assert self.weekdays is not None
            assert self.start_date is not None
            assert self.end_date is not None
            assert self.timezone is not None
            return RecurringTrigger(
                local_time=self.local_time,
                weekdays=tuple(self.weekdays),
                start_date=self.start_date,
                end_date=self.end_date,
                timezone=self.timezone,
            )
        assert self.active_from is not None
        assert self.active_until is not None
        return ConditionTransitionTrigger(
            active_from=self.active_from,
            active_until=self.active_until,
        )


class ConditionWire(StrictModel):
    entity: str
    attribute: str
    key_template: bool
    operator: Literal["eq", "gte", "lte"]
    value: ActionValue
    unit: str | None

    def to_domain(self) -> Condition:
        key = (
            FactKeyTemplate(entity=self.entity, attribute=self.attribute)
            if self.key_template
            else FactKey(entity=self.entity, attribute=self.attribute)
        )
        return Condition(
            key=key,
            operator=self.operator,
            value=self.value,
            unit=self.unit,
        )


class ActionTemplateWire(StrictModel):
    kind: ReminderKind
    payload: PayloadWire
    summary: str

    def to_domain(self) -> ActionTemplate:
        return ActionTemplate(
            kind=self.kind.value,
            payload=self.payload.to_payload(),
            summary=self.summary,
        )


class FactAssertionWire(StrictModel):
    entity: str
    attribute: str
    value: ActionValue
    unit: str | None

    def to_domain(self) -> SetFact:
        return SetFact(
            key=FactKey(entity=self.entity, attribute=self.attribute),
            value=self.value,
            unit=self.unit,
        )


class IntentCreateWire(StrictModel):
    intent_id: str
    trigger: TriggerWire
    required_conditions: list[ConditionWire]
    blockers: list[ConditionWire]
    action_template: ActionTemplateWire

    def to_domain(self) -> CreateIntent:
        return CreateIntent(
            intent_id=self.intent_id,
            trigger=self.trigger.to_domain(),
            required_conditions=tuple(
                condition.to_domain() for condition in self.required_conditions
            ),
            blockers=tuple(condition.to_domain() for condition in self.blockers),
            action_template=self.action_template.to_domain(),
        )


class IntentUpdateWire(StrictModel):
    intent_id: str
    trigger: TriggerWire | None
    required_conditions: list[ConditionWire] | None
    blockers: list[ConditionWire] | None
    action_template: ActionTemplateWire | None

    def to_domain(self) -> UpdateIntent:
        updates: dict[str, object] = {"intent_id": self.intent_id}
        if self.trigger is not None:
            updates["trigger"] = self.trigger.to_domain()
        if self.required_conditions is not None:
            updates["required_conditions"] = tuple(
                condition.to_domain() for condition in self.required_conditions
            )
        if self.blockers is not None:
            updates["blockers"] = tuple(
                condition.to_domain() for condition in self.blockers
            )
        if self.action_template is not None:
            updates["action_template"] = self.action_template.to_domain()
        return UpdateIntent.model_validate(updates)


class IntentCancellationWire(StrictModel):
    intent_id: str

    def to_domain(self) -> CancelIntent:
        return CancelIntent(intent_id=self.intent_id)


class MemoryDeltaWire(StrictModel):
    fact_assertions: list[FactAssertionWire]
    intent_creates: list[IntentCreateWire]
    intent_updates: list[IntentUpdateWire]
    intent_cancellations: list[IntentCancellationWire]

    def to_domain(self) -> MemoryDelta:
        mutations = [item.to_domain() for item in self.fact_assertions]
        mutations.extend(item.to_domain() for item in self.intent_creates)
        mutations.extend(item.to_domain() for item in self.intent_updates)
        mutations.extend(item.to_domain() for item in self.intent_cancellations)
        return MemoryDelta(mutations=tuple(mutations))


__all__ = ["DecisionWire", "MemoryDeltaWire", "PayloadWire"]
