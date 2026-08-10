"""Local-model compiler contract with type-specific, closed trigger shapes.

The hosted experiment intentionally keeps :mod:`anamnesis.wire` frozen around
the flat, required-but-nullable schema accepted by OpenAI strict structured
outputs.  Smaller local models have no reason to emit fields belonging to a
different trigger variant, so this module exposes a separate discriminated
transport contract.  Both contracts reduce to the same provider-neutral
``MemoryDelta`` domain model.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from typing import Annotated, Literal

from pydantic import Field, model_validator

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
from anamnesis.schema import ActionValue, ObservableEvent, StrictModel

LOCAL_MEMORY_COMPILER_VERSION = "local.v0.2"
LOCAL_MEMORY_COMPILER_W2_VERSION = "local.v0.3"
LOCAL_MEMORY_COMPILER_W3_VERSION = "local.v0.4"


class LocalPayloadWire(StrictModel):
    """Closed payload whose unused local wire slots may be omitted."""

    subject: str
    address: str | None = None
    build: str | None = None
    date: str | None = None
    flight: str | None = None
    greenhouse: str | None = None
    item: str | None = None
    project: str | None = None
    quantity: int | float | None = None
    recipient: str | None = None
    room: str | None = None
    shipment: str | None = None
    tank: str | None = None
    trip: str | None = None

    def to_payload(self) -> dict[str, ActionValue]:
        return self.model_dump(exclude_none=True)


class LocalAtTriggerWire(StrictModel):
    type: Literal["at"]
    at: datetime

    def to_domain(self) -> AtTrigger:
        return AtTrigger(at=self.at)


class LocalRecurringTriggerWire(StrictModel):
    type: Literal["recurring"]
    local_time: time
    weekdays: list[Weekday]
    start_date: date
    end_date: date
    timezone: str

    def to_domain(self) -> RecurringTrigger:
        return RecurringTrigger(
            local_time=self.local_time,
            weekdays=tuple(self.weekdays),
            start_date=self.start_date,
            end_date=self.end_date,
            timezone=self.timezone,
        )


class LocalConditionTransitionTriggerWire(StrictModel):
    type: Literal["condition_transition"]
    active_from: datetime
    active_until: datetime

    def to_domain(self) -> ConditionTransitionTrigger:
        return ConditionTransitionTrigger(
            active_from=self.active_from,
            active_until=self.active_until,
        )


LocalTriggerWire = Annotated[
    LocalAtTriggerWire
    | LocalRecurringTriggerWire
    | LocalConditionTransitionTriggerWire,
    Field(discriminator="type"),
]


class LocalConditionWire(StrictModel):
    entity: str
    attribute: str
    key_template: bool = False
    operator: Literal["eq", "gte", "lte"]
    value: ActionValue
    unit: str | None = None

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


class LocalActionTemplateWire(StrictModel):
    kind: Literal["reminder"] = "reminder"
    payload: LocalPayloadWire
    summary: str

    def to_domain(self) -> ActionTemplate:
        return ActionTemplate(
            kind=self.kind,
            payload=self.payload.to_payload(),
            summary=self.summary,
        )


class LocalFactAssertionWire(StrictModel):
    entity: str
    attribute: str
    value: ActionValue
    unit: str | None = None

    def to_domain(self) -> SetFact:
        return SetFact(
            key=FactKey(entity=self.entity, attribute=self.attribute),
            value=self.value,
            unit=self.unit,
        )


class LocalIntentCreateWire(StrictModel):
    intent_id: str
    trigger: LocalTriggerWire
    required_conditions: list[LocalConditionWire]
    blockers: list[LocalConditionWire]
    action_template: LocalActionTemplateWire

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


class LocalIntentUpdateWire(StrictModel):
    """A partial update; unchanged compound fields are omitted or null."""

    intent_id: str
    trigger: LocalTriggerWire | None = None
    required_conditions: list[LocalConditionWire] | None = None
    blockers: list[LocalConditionWire] | None = None
    action_template: LocalActionTemplateWire | None = None

    @model_validator(mode="after")
    def validate_changed_field(self) -> LocalIntentUpdateWire:
        if all(
            value is None
            for value in (
                self.trigger,
                self.required_conditions,
                self.blockers,
                self.action_template,
            )
        ):
            raise ValueError("local intent update requires a changed field")
        return self

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


class LocalIntentCancellationWire(StrictModel):
    intent_id: str

    def to_domain(self) -> CancelIntent:
        return CancelIntent(intent_id=self.intent_id)


class LocalMemoryDeltaWire(StrictModel):
    """Local structured-output envelope reduced atomically by the same store."""

    fact_assertions: list[LocalFactAssertionWire]
    intent_creates: list[LocalIntentCreateWire]
    intent_updates: list[LocalIntentUpdateWire]
    intent_cancellations: list[LocalIntentCancellationWire]

    def to_domain(self) -> MemoryDelta:
        mutations = [item.to_domain() for item in self.fact_assertions]
        mutations.extend(item.to_domain() for item in self.intent_creates)
        mutations.extend(item.to_domain() for item in self.intent_updates)
        mutations.extend(item.to_domain() for item in self.intent_cancellations)
        return MemoryDelta(mutations=tuple(mutations))


LOCAL_MEMORY_COMPILER_INSTRUCTIONS = (
    "You are the memory compiler for a simulated personal assistant. Convert "
    "only the current observable event into one strict local MemoryDelta.\n\n"
    "Writer ablation W1 rules:\n"
    "- Return all four arrays: fact_assertions, intent_creates, "
    "intent_updates, and intent_cancellations. Use [] when an array has no "
    "mutation.\n"
    "- Treat the current event as the only source of new information. Use active "
    "compact state only to resolve an exact active intent_id or reconstruct a "
    "compound field that this event truly changes. Never emit facts, triggers, "
    "conditions, or action templates solely because they appear in active "
    "state.\n"
    "- If the event is irrelevant, ambiguous, or licenses no mutation, return "
    "all four arrays empty. Omit an uncertain mutation instead of guessing.\n"
    "- Create an intent only when the current event explicitly asks the "
    "assistant to perform a reminder action in the future. A factual schedule, "
    "observation, possibility, brainstorming statement, or explicit instruction "
    "not to remind never creates an intent. Choose one stable normalized "
    "intent_id; the deterministic store supplies action_key and provenance.\n"
    "- Use a fact assertion only for a current-world value, completion, or "
    "correction explicitly stated by the current event. Preserve the exact "
    "entity, typed value, and unit. Keep numeric magnitudes numeric instead of "
    "collapsing them to booleans, and never transfer a fact between similar "
    "entities. An explicit same-value reaffirmation is a valid new fact "
    "assertion even when active state already contains that value.\n"
    "- Entity and attribute identifiers use lowercase ASCII letters, digits, "
    "dots, underscores, or hyphens; each dotted segment starts with a letter or "
    "digit. An intent_id uses the same characters but each dotted segment starts "
    "with a letter. Never use spaces or uppercase characters. Reuse exactly the "
    "same entity, attribute, value type, and unit in a condition and its fact "
    "assertions.\n"
    "- Update or cancel only an exact intent_id present in active state and "
    "unambiguously referenced by the current event. In an update, omit every "
    "unchanged top-level field; when a compound field changes, return its full "
    "current value while preserving unchanged leaves. An unrelated event or its "
    "timestamp never moves a trigger or rewrites an action.\n"
    "- Choose the trigger by meaning: at is one explicit scheduled instant or "
    "deadline; recurring is a repeated schedule; condition_transition fires "
    "when explicit state or threshold conditions become true. Never replace a "
    "requested future trigger with the current event timestamp.\n"
    "- A trigger must have exactly one of these shapes and no fields from the "
    "other shapes:\n"
    '  at: {"type":"at","at":"ISO datetime with UTC offset"}\n'
    '  recurring: {"type":"recurring","local_time":"HH:MM:SS",'
    '"weekdays":["monday"],"start_date":"YYYY-MM-DD",'
    '"end_date":"YYYY-MM-DD","timezone":"IANA timezone"}\n'
    '  condition transition: {"type":"condition_transition",'
    '"active_from":"ISO datetime with UTC offset",'
    '"active_until":"ISO datetime with UTC offset"}\n'
    "- Resolve explicit dates, weekdays, and times against the current event's "
    "local calendar and UTC offset, preserving the requested local time. For at "
    "or recurring, if the exact instant or range is not unambiguously "
    "resolvable, omit the mutation rather than using the event time.\n"
    "- For condition_transition, an explicit active window overrides defaults. "
    "When no window is stated, set active_from to the current event timestamp "
    "and active_until to exactly seven calendar days later at the same local "
    "time and UTC offset. Every condition_transition needs at least one required "
    "condition.\n"
    "- Conditions use only eq, gte, or lte. Preserve every explicit AND "
    "conjunct in required_conditions; blockers suppress when any blocker is "
    "true. Encode an explicit completed or already-done state as a blocker, not "
    "as a required synthetic negative fact. Do not invent operators, hidden "
    "facts, unsupported Boolean logic, or provenance IDs.\n"
    "- Set key_template=true only for a recurring condition whose entity or "
    "attribute contains the allowed {date} or {weekday} placeholder; otherwise "
    "set it to false. These are the only exceptions to ordinary identifier "
    "characters.\n"
    "- Make payload.subject a minimal lowercase imperative verb plus direct "
    "object, without personal pronouns, articles, or parameter details that fit "
    "a dedicated optional slot. Put address, build, date, flight, greenhouse, "
    "item, project, quantity, recipient, room, shipment, tank, and trip values "
    "in their matching slots, preserving source casing, and omit unused slots. "
    "For a recurring per-occurrence date use {date}, never a weekday word.\n"
    "- Return only schema-matching JSON and no prose.\n"
)

LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT = (
    "\nWriter ablation W2 sparse optional-payload serialization invariant:\n"
    "- An optional payload slot may contain a value only when that value is "
    "explicitly sourced by the current event, or when the value is legitimately "
    "preserved within an action_template that the current event actually "
    "updates. Otherwise the slot is unused.\n"
    "- Omit an unused payload key (preferred), or use JSON null. Never fill an "
    "unused key with an empty string, false, an empty collection, a placeholder, "
    "or zero filler. An explicitly sourced quantity of zero remains valid. "
    "Before returning, remove filler values.\n"
    '- Minimal payload example: {"subject":"check permit"}. Add optional keys '
    "only when sourced.\n"
)

LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS = (
    LOCAL_MEMORY_COMPILER_INSTRUCTIONS + LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT
)

LOCAL_MEMORY_COMPILER_W3_ADDENDUM = (
    "\nWriter intervention W3 semantic validation procedure:\n"
    "- Apply these checks silently and in order: license the mutation from the "
    "current event; resolve its target; choose the trigger type; resolve calendar "
    "values; preserve conditions; assemble the complete sourced payload; "
    "validate; then serialize. Return JSON only, without reasoning.\n"
    "- Normalize identifiers before serialization. For an ordinary source "
    "phrase, lowercase ASCII, replace each maximal run outside [a-z0-9] with one "
    "underscore, and trim leading or trailing underscores. Preserve a dot only "
    "when an explicit hierarchy already exists. Every fact entity and attribute "
    "segment must match [a-z0-9][a-z0-9_-]*; every intent_id segment must match "
    "[a-z][a-z0-9_-]*. Use exactly the same normalized entity and attribute in a "
    "condition and its corresponding fact. A unit is either an explicitly "
    "sourced non-empty string or null/omitted; never use an empty string. If a "
    "unique normalized identity cannot be formed, omit that mutation.\n"
    "- A newly created intent_id names the enduring requested action, not a "
    "mutable date, weekday, time, threshold, state, or value. For update or "
    "cancellation, match exactly one active intent by the requested action and "
    "copy its intent_id character-for-character from Active compact state. Never "
    "construct, rename, or infer an update or cancellation ID from revised "
    "details. If zero or multiple active intents match, omit that mutation.\n"
    "- Select the trigger before filling its fields. Use recurring for an "
    "every/each repeated schedule. Use condition_transition for a request that "
    "fires when explicit facts or thresholds become true and has no independent "
    "firing instant; a limiting deadline belongs in its active window. Use at for "
    "one explicit future instant or deadline. A condition attached to an explicit "
    "at or recurring schedule remains in required_conditions. Never substitute "
    "the current event timestamp for a missing trigger.\n"
    "- Resolve dates and times on the current event's local calendar. today is "
    "the current local date and tomorrow is one calendar day later. For a bare "
    "weekday, choose the first occurrence whose requested local datetime is "
    "strictly later than the event: compute (target weekday index - current "
    "weekday index) modulo seven, adding seven days when the result is zero and "
    "the requested time is not later. this weekday means that weekday in the "
    "current ISO week; next weekday means that weekday in the following ISO week. "
    "Verify that the resulting date has the named weekday and preserve the "
    "required UTC offset. If wording, date, time, offset, recurrence range, or "
    "required IANA timezone is not uniquely supplied by the current event, or "
    "preserved unchanged from the matched active intent during an update, omit "
    "the mutation.\n"
    "- For a create or an action_template update, first form the minimal subject "
    "and then inspect every closed optional slot: address, build, date, flight, "
    "greenhouse, item, project, quantity, recipient, room, shipment, tank, and "
    "trip. Include every current-event value that is an argument of the requested "
    "action and fits exactly one slot, exactly once. Preserve source casing and "
    "numeric type, including an explicitly sourced zero. Do not omit a sourced "
    "value merely to stay sparse, and do not copy a trigger-only date or time into "
    "the payload. Omit or null every unsourced slot and never use filler. A "
    "recurring per-occurrence action date uses {date}.\n"
    "- On an action_template update, begin with the matched active template, apply "
    "only changes licensed by the current event, retain every unchanged sourced "
    "leaf, and return the full current action_template. On a trigger, conditions, "
    "or blockers update, likewise return the full changed compound field. Omit "
    "every unchanged top-level update field.\n"
    "- Before returning a mutation, confirm that its target is unique, its "
    "identifiers and units satisfy both transport and domain constraints, its "
    "trigger has the correct type and complete fields, its temporal value is "
    "uniquely resolved, every explicit AND conjunct is preserved, and every "
    "sourced payload value is present without filler. If any required check is "
    "uncertain, omit that mutation; keep any independent mutation that remains "
    "fully supported.\n"
)

LOCAL_MEMORY_COMPILER_W3_INSTRUCTIONS = (
    LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS + LOCAL_MEMORY_COMPILER_W3_ADDENDUM
)


def build_local_memory_compiler_prompt(
    *,
    event: ObservableEvent,
    active_state: str,
) -> str:
    """Render the local compiler prompt without raw history or hidden gold."""

    return (
        f"{LOCAL_MEMORY_COMPILER_INSTRUCTIONS}\n"
        f"Current event: [{event.id}] {event.at.isoformat()} | "
        f"{event.kind} | {event.text}\n\n"
        "Active compact state (canonical JSON):\n"
        f"{active_state}\n"
    )


def build_local_memory_compiler_w2_prompt(
    *,
    event: ObservableEvent,
    active_state: str,
) -> str:
    """Render the W2 local compiler prompt with sparse payload serialization."""

    return (
        f"{LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS}\n"
        f"Current event: [{event.id}] {event.at.isoformat()} | "
        f"{event.kind} | {event.text}\n\n"
        "Active compact state (canonical JSON):\n"
        f"{active_state}\n"
    )


def build_local_memory_compiler_w3_prompt(
    *,
    event: ObservableEvent,
    active_state: str,
) -> str:
    """Render the W3 local compiler prompt with bundled semantic validation."""

    return (
        f"{LOCAL_MEMORY_COMPILER_W3_INSTRUCTIONS}\n"
        f"Current event: [{event.id}] {event.at.isoformat()} | "
        f"{event.kind} | {event.text}\n\n"
        "Active compact state (canonical JSON):\n"
        f"{active_state}\n"
    )


def local_memory_compiler_contract() -> str:
    """Return the local compiler prompt and schema for reproducibility hashes."""

    sentinel = ObservableEvent(
        id="<event-id>",
        at=datetime.fromisoformat("2000-01-01T00:00:00+00:00"),
        kind="user_message",
        text="<event-text>",
    )
    rendered = build_local_memory_compiler_prompt(
        event=sentinel,
        active_state='{"facts":[],"intents":[]}',
    )
    schema = json.dumps(
        LocalMemoryDeltaWire.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{LOCAL_MEMORY_COMPILER_VERSION}\n{rendered}\n{schema}"


def local_memory_compiler_w2_contract() -> str:
    """Return the W2 local compiler prompt and unchanged wire schema."""

    sentinel = ObservableEvent(
        id="<event-id>",
        at=datetime.fromisoformat("2000-01-01T00:00:00+00:00"),
        kind="user_message",
        text="<event-text>",
    )
    rendered = build_local_memory_compiler_w2_prompt(
        event=sentinel,
        active_state='{"facts":[],"intents":[]}',
    )
    schema = json.dumps(
        LocalMemoryDeltaWire.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{LOCAL_MEMORY_COMPILER_W2_VERSION}\n{rendered}\n{schema}"


def local_memory_compiler_w3_contract() -> str:
    """Return the W3 local compiler prompt and unchanged wire schema."""

    sentinel = ObservableEvent(
        id="<event-id>",
        at=datetime.fromisoformat("2000-01-01T00:00:00+00:00"),
        kind="user_message",
        text="<event-text>",
    )
    rendered = build_local_memory_compiler_w3_prompt(
        event=sentinel,
        active_state='{"facts":[],"intents":[]}',
    )
    schema = json.dumps(
        LocalMemoryDeltaWire.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{LOCAL_MEMORY_COMPILER_W3_VERSION}\n{rendered}\n{schema}"


__all__ = [
    "LOCAL_MEMORY_COMPILER_VERSION",
    "LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS",
    "LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT",
    "LOCAL_MEMORY_COMPILER_W2_VERSION",
    "LOCAL_MEMORY_COMPILER_W3_ADDENDUM",
    "LOCAL_MEMORY_COMPILER_W3_INSTRUCTIONS",
    "LOCAL_MEMORY_COMPILER_W3_VERSION",
    "LocalAtTriggerWire",
    "LocalConditionTransitionTriggerWire",
    "LocalMemoryDeltaWire",
    "LocalRecurringTriggerWire",
    "LocalTriggerWire",
    "build_local_memory_compiler_prompt",
    "build_local_memory_compiler_w2_prompt",
    "build_local_memory_compiler_w3_prompt",
    "local_memory_compiler_contract",
    "local_memory_compiler_w2_contract",
    "local_memory_compiler_w3_contract",
]
