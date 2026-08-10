"""Deterministic temporal and prospective memory primitives for Anamnesis v0.

The module deliberately contains no provider, database, embedding, or agent
framework integration.  A compiler proposes a validated :class:`MemoryDelta`;
the store owns provenance, versioning, trigger evaluation, and execution state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Protocol, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    model_validator,
)

from anamnesis.schema import (
    ActionValue,
    Decision,
    MemoryView,
    MemoryViewBlock,
    ObservableEvent,
    Usage,
    validate_action_payload,
)

_NORMALIZED_NAME = r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$"
_FACT_NAME = r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)*$"
Weekday = Literal[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
_WEEKDAY_INDEX: dict[Weekday, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
IntentField = Literal[
    "action_key",
    "trigger",
    "required_conditions",
    "blockers",
    "action_template",
    "status",
]
IntentStatus = Literal["active", "cancelled"]
OccurrenceStatus = Literal[
    "pending",
    "executed",
    "suppressed",
    "expired",
    "cancelled",
]


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")


def _validate_finite(value: ActionValue, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    value = _json_compatible(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return value


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


_EMPTY_MAPPING = object()
_EMPTY_SEQUENCE = object()
_REMOVED_LEAF_SUFFIX = ".__removed__"


def _semantic_leaves(prefix: str, value: object) -> dict[str, object]:
    """Flatten one semantic value into stable dotted leaf paths.

    Empty containers are leaves in their own right. This makes changing an
    empty condition collection into a populated one (or vice versa) a
    provenance-bearing semantic change instead of an invisible shape change.
    """

    leaves: dict[str, object] = {}

    def visit(path: str, item: object) -> None:
        if isinstance(item, BaseModel):
            fields = type(item).model_fields
            if not fields:
                leaves[path] = _EMPTY_MAPPING
                return
            for field_name in fields:
                visit(f"{path}.{field_name}", getattr(item, field_name))
            return
        if isinstance(item, Mapping):
            if not item:
                leaves[path] = _EMPTY_MAPPING
                return
            for key in sorted(item, key=str):
                visit(f"{path}.{key}", item[key])
            return
        if isinstance(item, (tuple, list)):
            if not item:
                leaves[path] = _EMPTY_SEQUENCE
                return
            for index, child in enumerate(item):
                visit(f"{path}.{index}", child)
            return
        leaves[path] = item

    visit(prefix, value)
    return leaves


def _same_semantic_leaf(left: object, right: object) -> bool:
    """Compare leaves without conflating values such as ``True`` and ``1``."""

    return type(left) is type(right) and left == right


def _revise_leaf_provenance(
    previous: Mapping[str, tuple[str, ...]],
    previous_leaves: Mapping[str, object],
    current_leaves: Mapping[str, object],
    source_event_id: str,
) -> dict[str, tuple[str, ...]]:
    """Carry unchanged leaf sources and mark only semantic changes.

    A removed leaf remains part of the current semantic state as an explicit
    dotted tombstone. Its provenance therefore continues to explain why that
    field is absent until a later revision adds it again.
    """

    provenance: dict[str, tuple[str, ...]] = {}
    for path, value in current_leaves.items():
        if path in previous_leaves and _same_semantic_leaf(
            previous_leaves[path], value
        ):
            provenance[path] = previous[path]
        else:
            provenance[path] = (source_event_id,)

    for path, sources in previous.items():
        if not path.endswith(_REMOVED_LEAF_SUFFIX):
            continue
        removed_path = path.removesuffix(_REMOVED_LEAF_SUFFIX)
        if removed_path not in current_leaves:
            provenance[path] = sources

    for path in previous_leaves.keys() - current_leaves.keys():
        provenance[f"{path}{_REMOVED_LEAF_SUFFIX}"] = (source_event_id,)
    return provenance


class MemoryModel(BaseModel):
    """Immutable model that rejects unknown compiler or store fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_copy(
        self,
        *,
        update: dict[str, object] | None = None,
        deep: bool = False,
    ) -> Self:
        """Copy through validation so updates cannot bypass deep freezing."""

        if deep:
            values = self.model_dump(mode="python", round_trip=True)
        else:
            values = {
                field_name: getattr(self, field_name)
                for field_name in type(self).model_fields
            }
        if update:
            values.update(update)
        return type(self).model_validate(values)


class FactKey(MemoryModel):
    """Normalized identity for one current-world value."""

    entity: str = Field(min_length=1, pattern=_FACT_NAME)
    attribute: str = Field(min_length=1, pattern=_FACT_NAME)

    @property
    def canonical(self) -> str:
        return f"{len(self.entity)}:{self.entity}{self.attribute}"

    @property
    def display(self) -> str:
        return f"{self.entity}.{self.attribute}"


class FactKeyTemplate(MemoryModel):
    """Occurrence-local fact identity using only the closed v0 placeholders."""

    entity: str = Field(min_length=1)
    attribute: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_template(self) -> Self:
        combined = f"{self.entity}.{self.attribute}"
        if "{date}" not in combined and "{weekday}" not in combined:
            raise ValueError("fact key template requires {date} or {weekday}")
        for field_name, value in (
            ("entity", self.entity),
            ("attribute", self.attribute),
        ):
            normalized = value.replace("{date}", "2000-01-01").replace(
                "{weekday}", "monday"
            )
            if "{" in normalized or "}" in normalized:
                raise ValueError("unknown fact-key template placeholder")
            if re.fullmatch(_FACT_NAME, normalized) is None:
                raise ValueError(f"invalid normalized fact-key {field_name}")
        return self

    @property
    def canonical(self) -> str:
        return f"{len(self.entity)}:{self.entity}{self.attribute}"

    @property
    def display(self) -> str:
        return f"{self.entity}.{self.attribute}"

    def resolve(self, scheduled: datetime, zone: ZoneInfo | None) -> FactKey:
        local = scheduled.astimezone(zone) if zone is not None else scheduled
        substitutions = {
            "{date}": local.date().isoformat(),
            "{weekday}": local.strftime("%A").lower(),
        }

        def substitute(value: str) -> str:
            for placeholder, replacement in substitutions.items():
                value = value.replace(placeholder, replacement)
            return value

        return FactKey(
            entity=substitute(self.entity),
            attribute=substitute(self.attribute),
        )


class FactRevision(MemoryModel):
    """One immutable value revision with an exclusive validity end."""

    revision_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    key: FactKey
    value: ActionValue
    unit: str | None = Field(default=None, min_length=1)
    valid_from: datetime
    valid_to: datetime | None = None
    previous_revision_id: str | None = None
    source_event_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        _require_aware(self.valid_from, "fact.valid_from")
        _validate_finite(self.value, "fact.value")
        if self.valid_to is not None:
            _require_aware(self.valid_to, "fact.valid_to")
            if self.valid_to < self.valid_from:
                raise ValueError("fact.valid_to precedes valid_from")
        if self.revision == 1 and self.previous_revision_id is not None:
            raise ValueError("first fact revision cannot have a predecessor")
        if self.revision > 1 and self.previous_revision_id is None:
            raise ValueError("later fact revision requires a predecessor")
        return self


class Condition(MemoryModel):
    """A closed-vocabulary predicate over the current fact index."""

    key: FactKey | FactKeyTemplate
    operator: Literal["eq", "gte", "lte"]
    value: ActionValue
    unit: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_operand(self) -> Self:
        _validate_finite(self.value, "condition.value")
        if self.operator != "eq" and (
            isinstance(self.value, bool) or not isinstance(self.value, (int, float))
        ):
            raise ValueError("gte/lte conditions require a numeric value")
        return self


class TruthValue(StrEnum):
    """Kleene-style result used by the deterministic condition evaluator."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class AtTrigger(MemoryModel):
    type: Literal["at"] = "at"
    at: datetime

    @model_validator(mode="after")
    def validate_at(self) -> Self:
        _require_aware(self.at, "trigger.at")
        return self


class RecurringTrigger(MemoryModel):
    type: Literal["recurring"] = "recurring"
    local_time: time
    weekdays: tuple[Weekday, ...] = Field(min_length=1)
    start_date: date
    end_date: date
    timezone: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_recurrence(self) -> Self:
        if self.local_time.tzinfo is not None:
            raise ValueError("recurring local_time must not carry a timezone")
        if len(self.weekdays) != len(set(self.weekdays)):
            raise ValueError("recurring weekdays must be unique")
        if self.end_date < self.start_date:
            raise ValueError("recurring end_date precedes start_date")
        if (self.end_date - self.start_date).days > 366:
            raise ValueError("recurring trigger range cannot exceed 366 days")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"unknown trigger timezone: {self.timezone}") from error
        return self


class ConditionTransitionTrigger(MemoryModel):
    type: Literal["condition_transition"] = "condition_transition"
    active_from: datetime
    active_until: datetime

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        _require_aware(self.active_from, "trigger.active_from")
        _require_aware(self.active_until, "trigger.active_until")
        if self.active_until < self.active_from:
            raise ValueError("condition trigger active_until precedes active_from")
        return self


Trigger = AtTrigger | RecurringTrigger | ConditionTransitionTrigger


class ActionTemplate(MemoryModel):
    """Stable reminder shape; recurrence placeholders are resolved locally."""

    kind: Literal["reminder"] = "reminder"
    payload: Mapping[str, ActionValue] = Field(min_length=1)
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        validation_payload: dict[str, ActionValue] = {}
        for key, value in self.payload.items():
            _validate_finite(value, f"action payload {key}")
            if isinstance(value, str):
                normalized = value.replace("{date}", "2000-01-01").replace(
                    "{weekday}", "monday"
                )
                if "{" in normalized or "}" in normalized:
                    raise ValueError("unknown action-template placeholder")
                validation_payload[key] = normalized
            else:
                validation_payload[key] = value
        validate_action_payload(validation_payload)
        object.__setattr__(
            self,
            "payload",
            MappingProxyType(dict(self.payload)),
        )
        return self

    @field_serializer("payload")
    def serialize_payload(
        self,
        value: Mapping[str, ActionValue],
    ) -> dict[str, ActionValue]:
        return dict(value)


def _intent_semantic_leaves(
    *,
    action_key: str,
    trigger: Trigger,
    required_conditions: tuple[Condition, ...],
    blockers: tuple[Condition, ...],
    action_template: ActionTemplate,
    status: IntentStatus,
) -> dict[str, object]:
    values: dict[IntentField, object] = {
        "action_key": action_key,
        "trigger": trigger,
        "required_conditions": required_conditions,
        "blockers": blockers,
        "action_template": action_template,
        "status": status,
    }
    leaves: dict[str, object] = {}
    for field_name, value in values.items():
        leaves.update(_semantic_leaves(field_name, value))
    return leaves


class IntentRevision(MemoryModel):
    """A full immutable intent revision with dotted leaf-level provenance."""

    revision_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    intent_id: str = Field(min_length=1, pattern=_NORMALIZED_NAME)
    action_key: str = Field(min_length=1)
    trigger: Trigger
    required_conditions: tuple[Condition, ...] = ()
    blockers: tuple[Condition, ...] = ()
    action_template: ActionTemplate
    status: IntentStatus = "active"
    valid_from: datetime
    valid_to: datetime | None = None
    previous_revision_id: str | None = None
    source_event_id: str = Field(min_length=1)
    field_provenance: Mapping[str, tuple[str, ...]]

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        _require_aware(self.valid_from, "intent.valid_from")
        if self.valid_to is not None:
            _require_aware(self.valid_to, "intent.valid_to")
            if self.valid_to < self.valid_from:
                raise ValueError("intent.valid_to precedes valid_from")
        if self.revision == 1 and self.previous_revision_id is not None:
            raise ValueError("first intent revision cannot have a predecessor")
        if self.revision > 1 and self.previous_revision_id is None:
            raise ValueError("later intent revision requires a predecessor")
        active_leaves = _intent_semantic_leaves(
            action_key=self.action_key,
            trigger=self.trigger,
            required_conditions=self.required_conditions,
            blockers=self.blockers,
            action_template=self.action_template,
            status=self.status,
        )
        provenance_paths = set(self.field_provenance)
        missing = active_leaves.keys() - provenance_paths
        if missing:
            raise ValueError(
                "intent field_provenance must cover every active semantic leaf: "
                f"{sorted(missing)}"
            )
        for path in provenance_paths - active_leaves.keys():
            if not path.endswith(_REMOVED_LEAF_SUFFIX):
                raise ValueError(f"unknown intent provenance path: {path}")
            removed_path = path.removesuffix(_REMOVED_LEAF_SUFFIX)
            if removed_path in active_leaves:
                raise ValueError(
                    f"removed intent provenance path is active: {removed_path}"
                )
        if any(not sources for sources in self.field_provenance.values()):
            raise ValueError("intent field provenance cannot be empty")
        if (
            isinstance(self.trigger, ConditionTransitionTrigger)
            and not self.required_conditions
        ):
            raise ValueError(
                "condition_transition requires at least one required condition"
            )
        if not isinstance(self.trigger, RecurringTrigger) and any(
            isinstance(condition.key, FactKeyTemplate)
            for condition in (*self.required_conditions, *self.blockers)
        ):
            raise ValueError(
                "fact-key templates are supported only by recurring triggers"
            )
        object.__setattr__(
            self,
            "field_provenance",
            MappingProxyType(dict(self.field_provenance)),
        )
        return self

    @field_serializer("field_provenance")
    def serialize_field_provenance(
        self,
        value: Mapping[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        return dict(value)


class SetFact(MemoryModel):
    op: Literal["set_fact"] = "set_fact"
    key: FactKey
    value: ActionValue
    unit: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        _validate_finite(self.value, "fact mutation value")
        return self


class CreateIntent(MemoryModel):
    op: Literal["create_intent"] = "create_intent"
    intent_id: str = Field(min_length=1, pattern=_NORMALIZED_NAME)
    trigger: Trigger
    required_conditions: tuple[Condition, ...] = ()
    blockers: tuple[Condition, ...] = ()
    action_template: ActionTemplate

    @model_validator(mode="after")
    def validate_conditions(self) -> Self:
        if (
            isinstance(self.trigger, ConditionTransitionTrigger)
            and not self.required_conditions
        ):
            raise ValueError(
                "condition_transition requires at least one required condition"
            )
        if not isinstance(self.trigger, RecurringTrigger) and any(
            isinstance(condition.key, FactKeyTemplate)
            for condition in (*self.required_conditions, *self.blockers)
        ):
            raise ValueError(
                "fact-key templates are supported only by recurring triggers"
            )
        return self


class UpdateIntent(MemoryModel):
    op: Literal["update_intent"] = "update_intent"
    intent_id: str = Field(min_length=1, pattern=_NORMALIZED_NAME)
    trigger: Trigger | None = None
    required_conditions: tuple[Condition, ...] | None = None
    blockers: tuple[Condition, ...] | None = None
    action_template: ActionTemplate | None = None

    @model_validator(mode="after")
    def validate_update(self) -> Self:
        mutable = {
            "trigger",
            "required_conditions",
            "blockers",
            "action_template",
        }
        if not (self.model_fields_set & mutable):
            raise ValueError("intent update must supply at least one changed field")
        if any(
            field in self.model_fields_set and getattr(self, field) is None
            for field in mutable
        ):
            raise ValueError("intent update fields cannot be null")
        return self


class CancelIntent(MemoryModel):
    op: Literal["cancel_intent"] = "cancel_intent"
    intent_id: str = Field(min_length=1, pattern=_NORMALIZED_NAME)


MemoryMutation = SetFact | CreateIntent | UpdateIntent | CancelIntent


class MemoryDelta(MemoryModel):
    """One compiler proposal, applied atomically by the deterministic reducer."""

    mutations: tuple[MemoryMutation, ...] = ()

    @model_validator(mode="after")
    def validate_unique_targets(self) -> Self:
        fact_targets: set[str] = set()
        intent_targets: set[str] = set()
        for mutation in self.mutations:
            if isinstance(mutation, SetFact):
                target = mutation.key.canonical
                if target in fact_targets:
                    raise ValueError(f"duplicate fact mutation target: {target}")
                fact_targets.add(target)
                continue
            if mutation.intent_id in intent_targets:
                raise ValueError(
                    f"duplicate intent mutation target: {mutation.intent_id}"
                )
            intent_targets.add(mutation.intent_id)
        return self


class CompilerFactView(MemoryModel):
    """Only the current semantic fact fields needed by the compiler."""

    entity: str = Field(min_length=1, pattern=_FACT_NAME)
    attribute: str = Field(min_length=1, pattern=_FACT_NAME)
    value: ActionValue
    unit: str | None = Field(default=None, min_length=1)


class CompilerConditionView(MemoryModel):
    """Wire-aligned condition projection safe for direct model reuse."""

    entity: str
    attribute: str
    key_template: bool = False
    operator: Literal["eq", "gte", "lte"]
    value: ActionValue
    unit: str | None = Field(default=None, min_length=1)


class CompilerIntentView(MemoryModel):
    """Active intent semantics without reducer-owned revision metadata."""

    intent_id: str = Field(min_length=1, pattern=_NORMALIZED_NAME)
    trigger: Trigger
    required_conditions: tuple[CompilerConditionView, ...] = ()
    blockers: tuple[CompilerConditionView, ...] = ()
    action_template: ActionTemplate


class CompilerStateView(MemoryModel):
    """Minimal, closed state projection crossing into the LLM compiler."""

    facts: tuple[CompilerFactView, ...] = ()
    intents: tuple[CompilerIntentView, ...] = ()


class Occurrence(MemoryModel):
    """One independently versioned appearance of an intent trigger."""

    occurrence_id: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    previous_revision_id: str | None = None
    intent_id: str = Field(min_length=1)
    intent_revision_id: str = Field(min_length=1)
    action_key: str = Field(min_length=1)
    scheduled_for: datetime
    detected_at: datetime
    checkpoint_event_id: str = Field(min_length=1)
    status: OccurrenceStatus
    status_at: datetime
    action_template: ActionTemplate
    evidence_event_ids: tuple[str, ...] = ()
    fact_revision_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_occurrence(self) -> Self:
        for field_name in ("scheduled_for", "detected_at", "status_at"):
            _require_aware(getattr(self, field_name), f"occurrence.{field_name}")
        if self.revision == 1 and self.previous_revision_id is not None:
            raise ValueError("first occurrence revision cannot have a predecessor")
        if self.revision > 1 and self.previous_revision_id is None:
            raise ValueError("later occurrence revision requires a predecessor")
        return self


class ExecutionRecord(MemoryModel):
    execution_id: str = Field(min_length=1)
    occurrence_id: str = Field(min_length=1)
    action_key: str = Field(min_length=1)
    emitted_at: datetime
    decision_event_id: str = Field(min_length=1)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_event_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_emitted_at(self) -> Self:
        _require_aware(self.emitted_at, "execution.emitted_at")
        return self


class DueCandidate(MemoryModel):
    occurrence_id: str
    intent_id: str
    intent_revision_id: str
    action_key: str
    due_at: datetime
    action_template: ActionTemplate
    evidence_event_ids: tuple[str, ...]
    fact_revision_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_due_at(self) -> Self:
        _require_aware(self.due_at, "due_candidate.due_at")
        return self


class DeltaAuditRecord(MemoryModel):
    event_id: str
    at: datetime
    delta_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    accepted: bool
    error: str | None = None

    @model_validator(mode="after")
    def validate_audit(self) -> Self:
        _require_aware(self.at, "delta_audit.at")
        if self.accepted == (self.error is not None):
            raise ValueError("accepted audit records cannot contain an error")
        return self


class ApplyResult(MemoryModel):
    accepted: bool
    error: str | None = None
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.accepted == (self.error is not None):
            raise ValueError("accepted apply results cannot contain an error")
        return self


class MemorySelection(MemoryModel):
    view: MemoryView
    due_candidates: tuple[DueCandidate, ...] = ()
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def due_candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.occurrence_id for candidate in self.due_candidates)


class CommitResult(MemoryModel):
    executed_occurrence_ids: tuple[str, ...] = ()
    expired_occurrence_ids: tuple[str, ...] = ()
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CompilerRequest(MemoryModel):
    event: ObservableEvent
    active_state: str = Field(min_length=1)


class CompilerCall(MemoryModel):
    delta: MemoryDelta | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = Field(default=0, ge=0)
    parse_error: bool = False
    raw_completion: str | None = None
    usage_complete: bool = True
    cost_complete: bool = False

    @model_validator(mode="after")
    def validate_call(self) -> Self:
        if self.parse_error and self.delta is not None:
            raise ValueError("a parse-error compiler call cannot carry a delta")
        return self


class MemoryCompiler(Protocol):
    """Provider-neutral compiler boundary; adapters live outside this module."""

    name: str

    async def compile(self, request: CompilerRequest) -> CompilerCall: ...


class DeterministicCompiler:
    """Small fake compiler for unit tests and diagnostic ceilings."""

    name = "deterministic"

    def __init__(
        self,
        deltas: Mapping[str, MemoryDelta],
        *,
        default: MemoryDelta | None = None,
    ) -> None:
        self._deltas = dict(deltas)
        self._default = default or MemoryDelta()
        self.requests: list[CompilerRequest] = []

    async def compile(self, request: CompilerRequest) -> CompilerCall:
        self.requests.append(request)
        delta = self._deltas.get(request.event.id, self._default)
        return CompilerCall(
            delta=delta,
            raw_completion=delta.model_dump_json(),
        )


class MemorySemanticError(ValueError):
    """A valid delta that cannot be applied to the current state."""


class InMemoryAnamnesis:
    """Versioned in-process store and deterministic trigger/execution engine."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._events: list[ObservableEvent] = []
        self._event_ids: set[str] = set()
        self._fact_revisions: list[FactRevision] = []
        self._current_facts: dict[str, FactRevision] = {}
        self._intent_revisions: list[IntentRevision] = []
        self._current_intents: dict[str, IntentRevision] = {}
        self._occurrence_revisions: list[Occurrence] = []
        self._current_occurrences: dict[str, Occurrence] = {}
        self._executions: list[ExecutionRecord] = []
        self._delta_audit: list[DeltaAuditRecord] = []
        self._condition_states: dict[str, TruthValue] = {}
        self._selection_event_id: str | None = None
        self._selection_due_ids: tuple[str, ...] = ()
        self._selection_committed = True

    def _fork(self) -> InMemoryAnamnesis:
        """Copy container state while sharing only immutable record values."""

        fork = object.__new__(type(self))
        fork._events = list(self._events)
        fork._event_ids = set(self._event_ids)
        fork._fact_revisions = list(self._fact_revisions)
        fork._current_facts = dict(self._current_facts)
        fork._intent_revisions = list(self._intent_revisions)
        fork._current_intents = dict(self._current_intents)
        fork._occurrence_revisions = list(self._occurrence_revisions)
        fork._current_occurrences = dict(self._current_occurrences)
        fork._executions = list(self._executions)
        fork._delta_audit = list(self._delta_audit)
        fork._condition_states = dict(self._condition_states)
        fork._selection_event_id = self._selection_event_id
        fork._selection_due_ids = self._selection_due_ids
        fork._selection_committed = self._selection_committed
        return fork

    def validate_delta(self, event: ObservableEvent, delta: MemoryDelta) -> None:
        """Dry-run a delta against exact current state without mutating the store."""

        result = self._fork().ingest(event, delta)
        if not result.accepted:
            raise MemorySemanticError(result.error or "memory delta was rejected")

    @property
    def events(self) -> tuple[ObservableEvent, ...]:
        return tuple(self._events)

    @property
    def fact_revisions(self) -> tuple[FactRevision, ...]:
        return tuple(self._fact_revisions)

    @property
    def intent_revisions(self) -> tuple[IntentRevision, ...]:
        return tuple(self._intent_revisions)

    @property
    def occurrence_revisions(self) -> tuple[Occurrence, ...]:
        return tuple(self._occurrence_revisions)

    @property
    def executions(self) -> tuple[ExecutionRecord, ...]:
        return tuple(self._executions)

    @property
    def delta_audit(self) -> tuple[DeltaAuditRecord, ...]:
        return tuple(self._delta_audit)

    @property
    def current_facts(self) -> tuple[FactRevision, ...]:
        return tuple(self._current_facts[key] for key in sorted(self._current_facts))

    @property
    def current_intents(self) -> tuple[IntentRevision, ...]:
        return tuple(
            self._current_intents[key] for key in sorted(self._current_intents)
        )

    def compiler_state(self) -> str:
        """Return minimal active semantics, never revisions or provenance."""

        active_intents = [
            intent for intent in self.current_intents if intent.status == "active"
        ]
        view = CompilerStateView(
            facts=tuple(
                CompilerFactView(
                    entity=fact.key.entity,
                    attribute=fact.key.attribute,
                    value=fact.value,
                    unit=fact.unit,
                )
                for fact in self.current_facts
            ),
            intents=tuple(
                CompilerIntentView(
                    intent_id=intent.intent_id,
                    trigger=intent.trigger,
                    required_conditions=tuple(
                        CompilerConditionView(
                            entity=condition.key.entity,
                            attribute=condition.key.attribute,
                            key_template=isinstance(condition.key, FactKeyTemplate),
                            operator=condition.operator,
                            value=condition.value,
                            unit=condition.unit,
                        )
                        for condition in intent.required_conditions
                    ),
                    blockers=tuple(
                        CompilerConditionView(
                            entity=condition.key.entity,
                            attribute=condition.key.attribute,
                            key_template=isinstance(condition.key, FactKeyTemplate),
                            operator=condition.operator,
                            value=condition.value,
                            unit=condition.unit,
                        )
                        for condition in intent.blockers
                    ),
                    action_template=intent.action_template,
                )
                for intent in active_intents
            ),
        )
        return _canonical_json(view)

    def state_hash(self) -> str:
        """Hash all behaviorally relevant state using canonical JSON."""

        return _sha256(
            {
                "events": [event.model_dump(mode="json") for event in self._events],
                "facts": [
                    revision.model_dump(mode="json")
                    for revision in self._fact_revisions
                ],
                "intents": [
                    revision.model_dump(mode="json")
                    for revision in self._intent_revisions
                ],
                "occurrences": [
                    revision.model_dump(mode="json")
                    for revision in self._occurrence_revisions
                ],
                "executions": [
                    record.model_dump(mode="json") for record in self._executions
                ],
                "delta_audit": [
                    record.model_dump(mode="json") for record in self._delta_audit
                ],
                "condition_states": {
                    key: value.value
                    for key, value in sorted(self._condition_states.items())
                },
                "selection": {
                    "event_id": self._selection_event_id,
                    "due_ids": self._selection_due_ids,
                    "committed": self._selection_committed,
                },
            }
        )

    def ingest(
        self,
        event: ObservableEvent,
        delta: MemoryDelta | None,
    ) -> ApplyResult:
        """Append an observable event and atomically apply its compiler delta."""

        if not self._selection_committed:
            raise RuntimeError("the prior checkpoint must be committed before ingest")
        if event.id in self._event_ids:
            raise ValueError(f"duplicate observable event ID: {event.id}")
        if self._events and event.at < self._events[-1].at:
            raise ValueError("observable events must be ingested chronologically")

        self._events.append(event)
        self._event_ids.add(event.id)
        delta_hash = _sha256(delta) if delta is not None else None

        if event.kind == "clock_tick":
            if delta is not None and delta.mutations:
                return self._record_rejection(
                    event,
                    delta_hash,
                    "clock ticks cannot carry memory mutations",
                )
            self._delta_audit.append(
                DeltaAuditRecord(
                    event_id=event.id,
                    at=event.at,
                    delta_sha256=delta_hash,
                    accepted=True,
                )
            )
            return ApplyResult(accepted=True, state_sha256=self.state_hash())

        if delta is None:
            return self._record_rejection(
                event,
                None,
                "message/observation compiler produced no valid MemoryDelta",
            )

        facts = list(self._fact_revisions)
        current_facts = dict(self._current_facts)
        intents = list(self._intent_revisions)
        current_intents = dict(self._current_intents)
        occurrences = list(self._occurrence_revisions)
        current_occurrences = dict(self._current_occurrences)
        condition_states = dict(self._condition_states)

        try:
            for mutation in delta.mutations:
                if isinstance(mutation, SetFact):
                    self._stage_fact(
                        event,
                        mutation,
                        facts,
                        current_facts,
                    )
                elif isinstance(mutation, CreateIntent):
                    self._stage_create_intent(
                        event,
                        mutation,
                        intents,
                        current_intents,
                    )
                elif isinstance(mutation, UpdateIntent):
                    self._stage_update_intent(
                        event,
                        mutation,
                        intents,
                        current_intents,
                        occurrences,
                        current_occurrences,
                        condition_states,
                    )
                else:
                    self._stage_cancel_intent(
                        event,
                        mutation,
                        intents,
                        current_intents,
                        occurrences,
                        current_occurrences,
                        condition_states,
                    )
        except (MemorySemanticError, ValidationError) as error:
            return self._record_rejection(event, delta_hash, str(error))

        self._fact_revisions = facts
        self._current_facts = current_facts
        self._intent_revisions = intents
        self._current_intents = current_intents
        self._occurrence_revisions = occurrences
        self._current_occurrences = current_occurrences
        self._condition_states = condition_states
        self._delta_audit.append(
            DeltaAuditRecord(
                event_id=event.id,
                at=event.at,
                delta_sha256=delta_hash,
                accepted=True,
            )
        )
        return ApplyResult(accepted=True, state_sha256=self.state_hash())

    def select(self, current: ObservableEvent) -> MemorySelection:
        """Advance triggers at a checkpoint and project a compact memory view."""

        if not self._events or self._events[-1].id != current.id:
            raise ValueError("select requires the most recently ingested event")
        if self._selection_event_id == current.id and not self._selection_committed:
            due = self._due_candidates(self._selection_due_ids)
            return MemorySelection(
                view=self._build_view(due),
                due_candidates=due,
                state_sha256=self.state_hash(),
            )
        if not self._selection_committed:
            raise RuntimeError("the prior checkpoint must be committed before select")

        due_ids: list[str] = []
        for intent in self.current_intents:
            if intent.status != "active" or current.at < intent.valid_from:
                continue
            if isinstance(intent.trigger, AtTrigger):
                self._select_at(intent, current, due_ids)
            elif isinstance(intent.trigger, RecurringTrigger):
                self._select_recurring(intent, current, due_ids)
            else:
                self._select_condition_transition(intent, current, due_ids)

        self._selection_event_id = current.id
        self._selection_due_ids = tuple(due_ids)
        self._selection_committed = False
        due = self._due_candidates(self._selection_due_ids)
        return MemorySelection(
            view=self._build_view(due),
            due_candidates=due,
            state_sha256=self.state_hash(),
        )

    def commit(self, current: ObservableEvent, decision: Decision) -> CommitResult:
        """Record same-key emissions and expire every un-emitted occurrence."""

        if self._selection_event_id != current.id or self._selection_committed:
            raise ValueError("commit requires one uncommitted selection at this event")

        matched_actions: set[int] = set()
        executed: list[str] = []
        expired: list[str] = []
        for occurrence_id in self._selection_due_ids:
            occurrence = self._current_occurrences[occurrence_id]
            action_index = next(
                (
                    index
                    for index, action in enumerate(decision.actions)
                    if index not in matched_actions
                    and action.action_key == occurrence.action_key
                    and action.kind == occurrence.action_template.kind
                    and action.payload == occurrence.action_template.payload
                ),
                None,
            )
            if action_index is None:
                self._transition_occurrence(occurrence, "expired", current)
                expired.append(occurrence_id)
                continue

            action = decision.actions[action_index]
            matched_actions.add(action_index)
            self._transition_occurrence(occurrence, "executed", current)
            evidence = occurrence.evidence_event_ids
            self._executions.append(
                ExecutionRecord(
                    execution_id=f"execution:{occurrence_id}",
                    occurrence_id=occurrence_id,
                    action_key=occurrence.action_key,
                    emitted_at=current.at,
                    decision_event_id=current.id,
                    payload_sha256=_sha256(action.payload),
                    evidence_event_ids=evidence,
                )
            )
            executed.append(occurrence_id)

        self._selection_committed = True
        return CommitResult(
            executed_occurrence_ids=tuple(executed),
            expired_occurrence_ids=tuple(expired),
            state_sha256=self.state_hash(),
        )

    def evaluate_condition(
        self,
        condition: Condition,
        *,
        scheduled: datetime | None = None,
        zone: ZoneInfo | None = None,
    ) -> TruthValue:
        """Evaluate one predicate without coercing missing or mistyped facts."""

        key = self._condition_fact_key(condition, scheduled, zone)
        if key is None:
            return TruthValue.UNKNOWN
        fact = self._current_facts.get(key.canonical)
        if fact is None:
            return TruthValue.UNKNOWN
        if fact.unit != condition.unit:
            return TruthValue.UNKNOWN
        if condition.operator == "eq":
            if isinstance(condition.value, bool) or isinstance(fact.value, bool):
                if not (
                    isinstance(condition.value, bool) and isinstance(fact.value, bool)
                ):
                    return TruthValue.FALSE
            elif isinstance(condition.value, (int, float)) and isinstance(
                fact.value, (int, float)
            ):
                pass
            elif type(condition.value) is not type(fact.value):
                return TruthValue.FALSE
            return (
                TruthValue.TRUE if fact.value == condition.value else TruthValue.FALSE
            )
        if isinstance(fact.value, bool) or not isinstance(fact.value, (int, float)):
            return TruthValue.UNKNOWN
        if condition.operator == "gte":
            matched = fact.value >= condition.value  # type: ignore[operator]
        else:
            matched = fact.value <= condition.value  # type: ignore[operator]
        return TruthValue.TRUE if matched else TruthValue.FALSE

    @staticmethod
    def _condition_fact_key(
        condition: Condition,
        scheduled: datetime | None,
        zone: ZoneInfo | None,
    ) -> FactKey | None:
        if isinstance(condition.key, FactKey):
            return condition.key
        if scheduled is None:
            return None
        return condition.key.resolve(scheduled, zone)

    def _record_rejection(
        self,
        event: ObservableEvent,
        delta_hash: str | None,
        error: str,
    ) -> ApplyResult:
        self._delta_audit.append(
            DeltaAuditRecord(
                event_id=event.id,
                at=event.at,
                delta_sha256=delta_hash,
                accepted=False,
                error=error,
            )
        )
        return ApplyResult(
            accepted=False,
            error=error,
            state_sha256=self.state_hash(),
        )

    @staticmethod
    def _replace_revision(
        revisions: list[FactRevision] | list[IntentRevision],
        revision_id: str,
        replacement: FactRevision | IntentRevision,
    ) -> None:
        for index, revision in enumerate(revisions):
            if revision.revision_id == revision_id:
                revisions[index] = replacement  # type: ignore[assignment]
                return
        raise AssertionError(f"missing revision in history: {revision_id}")

    def _stage_fact(
        self,
        event: ObservableEvent,
        mutation: SetFact,
        revisions: list[FactRevision],
        current: dict[str, FactRevision],
    ) -> None:
        key = mutation.key.canonical
        previous = current.get(key)
        revision_number = 1 if previous is None else previous.revision + 1
        if previous is not None:
            closed = previous.model_copy(update={"valid_to": event.at})
            self._replace_revision(revisions, previous.revision_id, closed)
        revision = FactRevision(
            revision_id=f"fact:{key}:r{revision_number}",
            revision=revision_number,
            key=mutation.key,
            value=mutation.value,
            unit=mutation.unit,
            valid_from=event.at,
            previous_revision_id=(
                previous.revision_id if previous is not None else None
            ),
            source_event_id=event.id,
        )
        revisions.append(revision)
        current[key] = revision

    def _stage_create_intent(
        self,
        event: ObservableEvent,
        mutation: CreateIntent,
        revisions: list[IntentRevision],
        current: dict[str, IntentRevision],
    ) -> None:
        if mutation.intent_id in current:
            raise MemorySemanticError(f"intent already exists: {mutation.intent_id}")
        self._validate_trigger_horizon(mutation.trigger, event.at)
        action_key = event.id
        semantic_leaves = _intent_semantic_leaves(
            action_key=action_key,
            trigger=mutation.trigger,
            required_conditions=mutation.required_conditions,
            blockers=mutation.blockers,
            action_template=mutation.action_template,
            status="active",
        )
        provenance = {path: (event.id,) for path in semantic_leaves}
        revision = IntentRevision(
            revision_id=f"intent:{mutation.intent_id}:r1",
            revision=1,
            intent_id=mutation.intent_id,
            action_key=action_key,
            trigger=mutation.trigger,
            required_conditions=mutation.required_conditions,
            blockers=mutation.blockers,
            action_template=mutation.action_template,
            valid_from=event.at,
            source_event_id=event.id,
            field_provenance=provenance,
        )
        revisions.append(revision)
        current[mutation.intent_id] = revision

    def _stage_update_intent(
        self,
        event: ObservableEvent,
        mutation: UpdateIntent,
        revisions: list[IntentRevision],
        current: dict[str, IntentRevision],
        occurrences: list[Occurrence],
        current_occurrences: dict[str, Occurrence],
        condition_states: dict[str, TruthValue],
    ) -> None:
        previous = current.get(mutation.intent_id)
        if previous is None or previous.status != "active":
            raise MemorySemanticError(
                f"cannot update missing or inactive intent: {mutation.intent_id}"
            )
        values: dict[str, object] = {
            "trigger": previous.trigger,
            "required_conditions": previous.required_conditions,
            "blockers": previous.blockers,
            "action_template": previous.action_template,
        }
        changed: set[str] = set()
        for field_name in values:
            if field_name not in mutation.model_fields_set:
                continue
            value = getattr(mutation, field_name)
            if value != values[field_name]:
                values[field_name] = value
                changed.add(field_name)
        if not changed:
            raise MemorySemanticError("intent update does not change active semantics")
        if (
            isinstance(values["trigger"], ConditionTransitionTrigger)
            and not values["required_conditions"]
        ):
            raise MemorySemanticError(
                "condition_transition requires at least one required condition"
            )
        trigger = values["trigger"]
        assert isinstance(
            trigger,
            AtTrigger | RecurringTrigger | ConditionTransitionTrigger,
        )
        self._validate_trigger_horizon(trigger, event.at)

        closed = previous.model_copy(update={"valid_to": event.at})
        self._replace_revision(revisions, previous.revision_id, closed)
        previous_leaves = _intent_semantic_leaves(
            action_key=previous.action_key,
            trigger=previous.trigger,
            required_conditions=previous.required_conditions,
            blockers=previous.blockers,
            action_template=previous.action_template,
            status=previous.status,
        )
        current_leaves = _intent_semantic_leaves(
            action_key=previous.action_key,
            trigger=values["trigger"],
            required_conditions=values["required_conditions"],
            blockers=values["blockers"],
            action_template=values["action_template"],
            status="active",
        )
        provenance = _revise_leaf_provenance(
            previous.field_provenance,
            previous_leaves,
            current_leaves,
            event.id,
        )
        revision_number = previous.revision + 1
        revision = IntentRevision(
            revision_id=f"intent:{mutation.intent_id}:r{revision_number}",
            revision=revision_number,
            intent_id=previous.intent_id,
            action_key=previous.action_key,
            trigger=values["trigger"],
            required_conditions=values["required_conditions"],
            blockers=values["blockers"],
            action_template=values["action_template"],
            status="active",
            valid_from=event.at,
            previous_revision_id=previous.revision_id,
            source_event_id=event.id,
            field_provenance=provenance,
        )
        revisions.append(revision)
        current[mutation.intent_id] = revision
        self._cancel_staged_pending(
            mutation.intent_id,
            event,
            occurrences,
            current_occurrences,
        )
        if changed & {"trigger", "required_conditions", "blockers"}:
            condition_states.pop(mutation.intent_id, None)

    @staticmethod
    def _validate_trigger_horizon(trigger: Trigger, source_at: datetime) -> None:
        """Reject trigger contracts with no possible future occurrence."""

        if isinstance(trigger, AtTrigger):
            if trigger.at <= source_at:
                raise MemorySemanticError(
                    "at trigger must be strictly after its source event"
                )
            return
        if isinstance(trigger, ConditionTransitionTrigger):
            if trigger.active_until <= source_at:
                raise MemorySemanticError(
                    "condition trigger window must extend past its source event"
                )
            return

        zone = ZoneInfo(trigger.timezone)
        weekdays = {_WEEKDAY_INDEX[name] for name in trigger.weekdays}
        day = trigger.start_date
        while day <= trigger.end_date:
            if day.weekday() in weekdays:
                scheduled = datetime.combine(day, trigger.local_time, tzinfo=zone)
                round_trip = scheduled.astimezone(UTC).astimezone(zone)
                if (
                    round_trip.replace(tzinfo=None) == scheduled.replace(tzinfo=None)
                    and scheduled > source_at
                ):
                    return
            day += timedelta(days=1)
        raise MemorySemanticError(
            "recurring trigger has no valid occurrence after its source event"
        )

    def _stage_cancel_intent(
        self,
        event: ObservableEvent,
        mutation: CancelIntent,
        revisions: list[IntentRevision],
        current: dict[str, IntentRevision],
        occurrences: list[Occurrence],
        current_occurrences: dict[str, Occurrence],
        condition_states: dict[str, TruthValue],
    ) -> None:
        previous = current.get(mutation.intent_id)
        if previous is None or previous.status != "active":
            raise MemorySemanticError(
                f"cannot cancel missing or inactive intent: {mutation.intent_id}"
            )
        closed = previous.model_copy(update={"valid_to": event.at})
        self._replace_revision(revisions, previous.revision_id, closed)
        previous_leaves = _intent_semantic_leaves(
            action_key=previous.action_key,
            trigger=previous.trigger,
            required_conditions=previous.required_conditions,
            blockers=previous.blockers,
            action_template=previous.action_template,
            status=previous.status,
        )
        current_leaves = _intent_semantic_leaves(
            action_key=previous.action_key,
            trigger=previous.trigger,
            required_conditions=previous.required_conditions,
            blockers=previous.blockers,
            action_template=previous.action_template,
            status="cancelled",
        )
        provenance = _revise_leaf_provenance(
            previous.field_provenance,
            previous_leaves,
            current_leaves,
            event.id,
        )
        revision_number = previous.revision + 1
        revision = previous.model_copy(
            update={
                "revision_id": f"intent:{mutation.intent_id}:r{revision_number}",
                "revision": revision_number,
                "status": "cancelled",
                "valid_from": event.at,
                "valid_to": None,
                "previous_revision_id": previous.revision_id,
                "source_event_id": event.id,
                "field_provenance": provenance,
            }
        )
        revisions.append(revision)
        current[mutation.intent_id] = revision
        self._cancel_staged_pending(
            mutation.intent_id,
            event,
            occurrences,
            current_occurrences,
        )
        condition_states.pop(mutation.intent_id, None)

    @staticmethod
    def _cancel_staged_pending(
        intent_id: str,
        event: ObservableEvent,
        revisions: list[Occurrence],
        current: dict[str, Occurrence],
    ) -> None:
        for occurrence_id, occurrence in list(current.items()):
            if occurrence.intent_id != intent_id or occurrence.status != "pending":
                continue
            revision = occurrence.model_copy(
                update={
                    "revision_id": (
                        f"occurrence:{occurrence_id}:r{occurrence.revision + 1}"
                    ),
                    "revision": occurrence.revision + 1,
                    "previous_revision_id": occurrence.revision_id,
                    "status": "cancelled",
                    "status_at": event.at,
                    "checkpoint_event_id": event.id,
                }
            )
            revisions.append(revision)
            current[occurrence_id] = revision

    def _select_at(
        self,
        intent: IntentRevision,
        current: ObservableEvent,
        due_ids: list[str],
    ) -> None:
        trigger = intent.trigger
        assert isinstance(trigger, AtTrigger)
        if current.at < trigger.at:
            return
        self._materialize_time_occurrence(intent, trigger.at, current, due_ids)

    def _select_recurring(
        self,
        intent: IntentRevision,
        current: ObservableEvent,
        due_ids: list[str],
    ) -> None:
        trigger = intent.trigger
        assert isinstance(trigger, RecurringTrigger)
        zone = ZoneInfo(trigger.timezone)
        current_local_date = current.at.astimezone(zone).date()
        first_date = max(
            trigger.start_date,
            intent.valid_from.astimezone(zone).date(),
        )
        final_date = min(trigger.end_date, current_local_date)
        if final_date < first_date:
            return
        day = first_date
        weekdays = {_WEEKDAY_INDEX[name] for name in trigger.weekdays}
        while day <= final_date:
            if day.weekday() in weekdays:
                scheduled = datetime.combine(day, trigger.local_time, tzinfo=zone)
                # Skip nonexistent wall times; choose fold=0 for ambiguous times.
                round_trip = scheduled.astimezone(UTC).astimezone(zone)
                if (
                    round_trip.replace(tzinfo=None) == scheduled.replace(tzinfo=None)
                    and scheduled >= intent.valid_from
                    and scheduled <= current.at
                ):
                    self._materialize_time_occurrence(
                        intent,
                        scheduled,
                        current,
                        due_ids,
                    )
            day += timedelta(days=1)

    def _select_condition_transition(
        self,
        intent: IntentRevision,
        current: ObservableEvent,
        due_ids: list[str],
    ) -> None:
        trigger = intent.trigger
        assert isinstance(trigger, ConditionTransitionTrigger)
        eligibility, facts = self._evaluate_intent(intent, current.at, None)
        previous = self._condition_states.get(intent.intent_id, TruthValue.UNKNOWN)
        self._condition_states[intent.intent_id] = eligibility
        if not trigger.active_from <= current.at <= trigger.active_until:
            return
        # A create or condition-contract update establishes a new baseline;
        # an action-template-only update must not suppress a real fact
        # transition that occurs in the same observable event.
        if current.id == intent.source_event_id and any(
            current.id in sources
            for path, sources in intent.field_provenance.items()
            if path.split(".", maxsplit=1)[0]
            in {"trigger", "required_conditions", "blockers"}
        ):
            return
        if eligibility != TruthValue.TRUE or previous == TruthValue.TRUE:
            return
        if any(
            occurrence.intent_id == intent.intent_id
            and occurrence.status in {"pending", "executed", "expired"}
            for occurrence in self._current_occurrences.values()
        ):
            # condition_transition is a one-shot trigger in the closed v0 DSL.
            return
        occurrence_id = f"{intent.intent_id}@transition:{current.id}"
        if occurrence_id in self._current_occurrences:
            return
        action = self._resolve_action(intent.action_template, current.at, None)
        evidence = self._checkpoint_evidence(intent, facts, current)
        occurrence = Occurrence(
            occurrence_id=occurrence_id,
            revision_id=f"occurrence:{occurrence_id}:r1",
            revision=1,
            intent_id=intent.intent_id,
            intent_revision_id=intent.revision_id,
            action_key=intent.action_key,
            scheduled_for=current.at,
            detected_at=current.at,
            checkpoint_event_id=current.id,
            status="pending",
            status_at=current.at,
            action_template=action,
            evidence_event_ids=evidence,
            fact_revision_ids=tuple(fact.revision_id for fact in facts),
        )
        self._occurrence_revisions.append(occurrence)
        self._current_occurrences[occurrence_id] = occurrence
        due_ids.append(occurrence_id)

    def _materialize_time_occurrence(
        self,
        intent: IntentRevision,
        scheduled: datetime,
        current: ObservableEvent,
        due_ids: list[str],
    ) -> None:
        identity_time = scheduled.astimezone(UTC).isoformat()
        occurrence_id = f"{intent.intent_id}@{identity_time}"
        if occurrence_id in self._current_occurrences:
            return
        zone = (
            ZoneInfo(intent.trigger.timezone)
            if isinstance(intent.trigger, RecurringTrigger)
            else None
        )
        eligibility, facts = self._evaluate_intent(intent, scheduled, zone)
        status: OccurrenceStatus = (
            "pending" if eligibility == TruthValue.TRUE else "suppressed"
        )
        action = self._resolve_action(intent.action_template, scheduled, zone)
        evidence = self._checkpoint_evidence(intent, facts, current)
        occurrence = Occurrence(
            occurrence_id=occurrence_id,
            revision_id=f"occurrence:{occurrence_id}:r1",
            revision=1,
            intent_id=intent.intent_id,
            intent_revision_id=intent.revision_id,
            action_key=intent.action_key,
            scheduled_for=scheduled,
            detected_at=current.at,
            checkpoint_event_id=current.id,
            status=status,
            status_at=current.at,
            action_template=action,
            evidence_event_ids=evidence,
            fact_revision_ids=tuple(fact.revision_id for fact in facts),
        )
        self._occurrence_revisions.append(occurrence)
        self._current_occurrences[occurrence_id] = occurrence
        if status == "pending":
            due_ids.append(occurrence_id)

    def _evaluate_intent(
        self,
        intent: IntentRevision,
        scheduled: datetime | None = None,
        zone: ZoneInfo | None = None,
    ) -> tuple[TruthValue, tuple[FactRevision, ...]]:
        facts: dict[str, FactRevision] = {}
        required_results: list[TruthValue] = []
        for condition in intent.required_conditions:
            result = self.evaluate_condition(
                condition,
                scheduled=scheduled,
                zone=zone,
            )
            required_results.append(result)
            key = self._condition_fact_key(condition, scheduled, zone)
            fact = self._current_facts.get(key.canonical) if key is not None else None
            if fact is not None:
                facts[fact.revision_id] = fact
        if TruthValue.FALSE in required_results:
            required = TruthValue.FALSE
        elif TruthValue.UNKNOWN in required_results:
            required = TruthValue.UNKNOWN
        else:
            required = TruthValue.TRUE

        blocker_true = False
        for blocker in intent.blockers:
            result = self.evaluate_condition(
                blocker,
                scheduled=scheduled,
                zone=zone,
            )
            if result == TruthValue.TRUE:
                blocker_true = True
            key = self._condition_fact_key(blocker, scheduled, zone)
            fact = self._current_facts.get(key.canonical) if key is not None else None
            if fact is not None:
                facts[fact.revision_id] = fact

        if required != TruthValue.TRUE:
            eligibility = required
        elif blocker_true:
            eligibility = TruthValue.FALSE
        else:
            # Unknown blockers intentionally do not suppress a due action.
            eligibility = TruthValue.TRUE
        ordered = tuple(sorted(facts.values(), key=lambda fact: fact.key.canonical))
        return eligibility, ordered

    def _intent_evidence(
        self,
        intent: IntentRevision,
        facts: tuple[FactRevision, ...],
    ) -> tuple[str, ...]:
        evidence: set[str] = set()
        active_leaves = _intent_semantic_leaves(
            action_key=intent.action_key,
            trigger=intent.trigger,
            required_conditions=intent.required_conditions,
            blockers=intent.blockers,
            action_template=intent.action_template,
            status=intent.status,
        )
        active_paths = set(active_leaves)
        active_paths.update(
            path
            for path in intent.field_provenance
            if path.endswith(_REMOVED_LEAF_SUFFIX)
        )
        for path in active_paths:
            evidence.update(intent.field_provenance[path])
        evidence.update(fact.source_event_id for fact in facts)
        order = {event.id: index for index, event in enumerate(self._events)}
        return tuple(sorted(evidence, key=lambda item: (order.get(item, 10**9), item)))

    def _checkpoint_evidence(
        self,
        intent: IntentRevision,
        facts: tuple[FactRevision, ...],
        current: ObservableEvent,
    ) -> tuple[str, ...]:
        """Add the trigger checkpoint to the store-owned causal evidence."""

        evidence = set(self._intent_evidence(intent, facts))
        evidence.add(current.id)
        order = {event.id: index for index, event in enumerate(self._events)}
        return tuple(sorted(evidence, key=lambda item: (order[item], item)))

    @staticmethod
    def _resolve_action(
        template: ActionTemplate,
        scheduled: datetime,
        zone: ZoneInfo | None,
    ) -> ActionTemplate:
        local = scheduled.astimezone(zone) if zone is not None else scheduled
        payload_substitutions = {
            "{date}": local.date().isoformat(),
            "{weekday}": local.strftime("%A").lower(),
        }
        summary_substitutions = {
            "{date}": local.date().isoformat(),
            "{weekday}": local.strftime("%A"),
        }

        def resolve(value: ActionValue) -> ActionValue:
            if not isinstance(value, str):
                return value
            for placeholder, replacement in payload_substitutions.items():
                value = value.replace(placeholder, replacement)
            return value

        summary = template.summary
        for placeholder, replacement in summary_substitutions.items():
            summary = summary.replace(placeholder, replacement)
        return ActionTemplate(
            kind=template.kind,
            payload={key: resolve(value) for key, value in template.payload.items()},
            summary=summary,
        )

    def _due_candidates(
        self,
        occurrence_ids: tuple[str, ...],
    ) -> tuple[DueCandidate, ...]:
        candidates: list[DueCandidate] = []
        for occurrence_id in occurrence_ids:
            occurrence = self._current_occurrences[occurrence_id]
            if occurrence.status != "pending":
                continue
            candidates.append(
                DueCandidate(
                    occurrence_id=occurrence.occurrence_id,
                    intent_id=occurrence.intent_id,
                    intent_revision_id=occurrence.intent_revision_id,
                    action_key=occurrence.action_key,
                    due_at=occurrence.scheduled_for,
                    action_template=occurrence.action_template,
                    evidence_event_ids=occurrence.evidence_event_ids,
                    fact_revision_ids=occurrence.fact_revision_ids,
                )
            )
        return tuple(candidates)

    def _build_view(self, due: tuple[DueCandidate, ...]) -> MemoryView:
        blocks: list[MemoryViewBlock] = []
        fact_ids: set[str] = set()
        action_keys: set[str] = set()
        for candidate in due:
            action_keys.add(candidate.action_key)
            fact_ids.update(candidate.fact_revision_ids)
            blocks.append(
                MemoryViewBlock(
                    kind="due_candidate",
                    title=f"Due reminder: {candidate.action_template.summary}",
                    content=_canonical_json(
                        {
                            "occurrence_id": candidate.occurrence_id,
                            "intent_id": candidate.intent_id,
                            "action_key": candidate.action_key,
                            "due_at": candidate.due_at.isoformat(),
                            "kind": candidate.action_template.kind,
                            "payload": candidate.action_template.payload,
                            "summary": candidate.action_template.summary,
                        }
                    ),
                    evidence_event_ids=list(candidate.evidence_event_ids),
                )
            )

        facts_by_id = {
            fact.revision_id: fact
            for fact in self._fact_revisions
            if fact.valid_to is None
        }
        for revision_id in sorted(fact_ids):
            fact = facts_by_id.get(revision_id)
            if fact is None:
                continue
            blocks.append(
                MemoryViewBlock(
                    kind="fact",
                    title=f"Current fact: {fact.key.display}",
                    content=_canonical_json(
                        {
                            "key": fact.key.display,
                            "value": fact.value,
                            "unit": fact.unit,
                            "valid_from": fact.valid_from.isoformat(),
                        }
                    ),
                    evidence_event_ids=[fact.source_event_id],
                )
            )

        for execution in self._executions:
            if execution.action_key not in action_keys:
                continue
            blocks.append(
                MemoryViewBlock(
                    kind="execution",
                    title=f"Prior execution: {execution.action_key}",
                    content=_canonical_json(
                        {
                            "occurrence_id": execution.occurrence_id,
                            "emitted_at": execution.emitted_at.isoformat(),
                            "payload_sha256": execution.payload_sha256,
                        }
                    ),
                    evidence_event_ids=list(execution.evidence_event_ids),
                )
            )
        return MemoryView(blocks=blocks)

    def _transition_occurrence(
        self,
        occurrence: Occurrence,
        status: Literal["executed", "expired"],
        current: ObservableEvent,
    ) -> None:
        if occurrence.status != "pending":
            raise AssertionError("only pending occurrences can be committed")
        revision = occurrence.model_copy(
            update={
                "revision_id": (
                    f"occurrence:{occurrence.occurrence_id}:r{occurrence.revision + 1}"
                ),
                "revision": occurrence.revision + 1,
                "previous_revision_id": occurrence.revision_id,
                "status": status,
                "status_at": current.at,
                "checkpoint_event_id": current.id,
            }
        )
        self._occurrence_revisions.append(revision)
        self._current_occurrences[occurrence.occurrence_id] = revision


__all__ = [
    "ActionTemplate",
    "ApplyResult",
    "AtTrigger",
    "CancelIntent",
    "CommitResult",
    "CompilerCall",
    "CompilerRequest",
    "Condition",
    "ConditionTransitionTrigger",
    "CreateIntent",
    "DeltaAuditRecord",
    "DeterministicCompiler",
    "DueCandidate",
    "ExecutionRecord",
    "FactKey",
    "FactKeyTemplate",
    "FactRevision",
    "InMemoryAnamnesis",
    "IntentRevision",
    "MemoryCompiler",
    "MemoryDelta",
    "MemorySelection",
    "Occurrence",
    "RecurringTrigger",
    "SetFact",
    "TruthValue",
    "UpdateIntent",
]
