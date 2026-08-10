"""Validated, provider-neutral schemas for scenarios and evaluation runs."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator

EventKind = Literal[
    "user_message",
    "observation",
    "clock_tick",
    "assistant_decision",
]
ObservableEventKind = Literal["user_message", "observation", "clock_tick"]
ActionKind = Literal["reminder"]
ForbiddenReason = Literal[
    "obsolete",
    "condition_satisfied",
    "premature",
    "duplicate",
    "unrequested",
]
ActionValue = str | int | float | bool
OPTIONAL_PAYLOAD_KEYS = frozenset(
    {
        "address",
        "build",
        "date",
        "flight",
        "greenhouse",
        "item",
        "project",
        "quantity",
        "recipient",
        "room",
        "shipment",
        "tank",
        "trip",
    }
)
ALLOWED_PAYLOAD_KEYS = OPTIONAL_PAYLOAD_KEYS | {"subject"}


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a UTC offset")


def validate_action_payload(payload: dict[str, ActionValue]) -> None:
    """Enforce the frozen, scorer-visible canonical reminder payload."""

    unknown = set(payload) - ALLOWED_PAYLOAD_KEYS
    if unknown:
        raise ValueError(f"unknown action payload keys: {sorted(unknown)}")
    subject = payload.get("subject")
    if not isinstance(subject, str):
        raise ValueError("action payload subject must be a string")
    if subject != subject.strip() or subject != subject.casefold():
        raise ValueError("action payload subject must be trimmed and lowercase")
    if len(subject.split()) < 2:
        raise ValueError("action payload subject must be verb plus direct object")
    for key, value in payload.items():
        if key == "quantity":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("action payload quantity must be numeric")
        elif not isinstance(value, str):
            raise ValueError(f"action payload {key} must be a string")
    if "date" in payload:
        try:
            date.fromisoformat(str(payload["date"]))
        except ValueError as error:
            raise ValueError("action payload date must be ISO YYYY-MM-DD") from error


class StrictModel(BaseModel):
    """Base class that rejects silent schema drift."""

    model_config = ConfigDict(extra="forbid")


class ObservableEvent(StrictModel):
    """The sanitized event boundary visible to evaluated systems."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    at: datetime
    kind: ObservableEventKind
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        _require_aware(self.at, "observable_event.at")
        return self


class RuntimeScenario(StrictModel):
    """Author-annotation-free timeline used inside an evaluated runtime."""

    id: str = Field(min_length=1)
    events: list[ObservableEvent] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        event_ids = [event.id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("runtime event IDs must be unique")
        event_times = [event.at for event in self.events]
        if len(event_times) != len(set(event_times)):
            raise ValueError("runtime event timestamps must be unique")
        if self.events != sorted(self.events, key=lambda event: event.at):
            raise ValueError("runtime events must be in chronological order")
        return self


class MemoryViewBlock(StrictModel):
    """One typed, compact Anamnesis memory record shown to the decision model."""

    kind: Literal["due_candidate", "fact", "execution"]
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    evidence_event_ids: list[str] = Field(default_factory=list)


class MemoryView(StrictModel):
    """Compact decision-time projection of structured memory."""

    blocks: list[MemoryViewBlock] = Field(default_factory=list)


class ScenarioEvent(StrictModel):
    """One timestamped observation in a simulated seven-day history."""

    id: str = Field(min_length=1)
    at: datetime
    kind: EventKind
    text: str = Field(min_length=1)
    supersedes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        _require_aware(self.at, "event.at")
        return self

    def to_observable(self) -> ObservableEvent:
        """Drop author-only fields before crossing into an evaluated system."""

        if self.kind == "assistant_decision":
            raise ValueError("assistant_decision is not an authored observable event")
        return ObservableEvent(
            id=self.id,
            at=self.at,
            kind=self.kind,
            text=self.text,
        )


class ExpectedAction(StrictModel):
    """A gold action that must be emitted during an allowed time window."""

    id: str = Field(min_length=1)
    action_key: str = Field(min_length=1)
    kind: ActionKind = "reminder"
    payload: dict[str, ActionValue] = Field(min_length=1)
    window_start: datetime
    window_end: datetime
    acceptable_evidence_sets: list[list[str]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        validate_action_payload(self.payload)
        _require_aware(self.window_start, "expected_action.window_start")
        _require_aware(self.window_end, "expected_action.window_end")
        if self.window_end < self.window_start:
            raise ValueError("expected action window_end precedes window_start")
        return self


class ForbiddenAction(StrictModel):
    """A recognizable action that must not be emitted in a given window."""

    id: str = Field(min_length=1)
    action_key: str = Field(min_length=1)
    kind: ActionKind = "reminder"
    payload: dict[str, ActionValue] = Field(min_length=1)
    window_start: datetime
    window_end: datetime
    reason: ForbiddenReason
    related_event_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        validate_action_payload(self.payload)
        _require_aware(self.window_start, "forbidden_action.window_start")
        _require_aware(self.window_end, "forbidden_action.window_end")
        if self.window_end < self.window_start:
            raise ValueError("forbidden action window_end precedes window_start")
        return self


class Scenario(StrictModel):
    """A fully specified simulated history and its hidden gold actions."""

    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    timezone: str = Field(min_length=1)
    start_at: datetime
    end_at: datetime
    tags: list[str] = Field(default_factory=list)
    events: list[ScenarioEvent] = Field(min_length=1)
    expected_actions: list[ExpectedAction] = Field(default_factory=list)
    forbidden_actions: list[ForbiddenAction] = Field(default_factory=list)

    def to_runtime(self) -> RuntimeScenario:
        """Remove every author-only annotation before Inspect execution."""

        return RuntimeScenario(
            id=self.id,
            events=[event.to_observable() for event in self.events],
        )

    @model_validator(mode="after")
    def validate_scenario(self) -> Self:
        _require_aware(self.start_at, "scenario.start_at")
        _require_aware(self.end_at, "scenario.end_at")
        if self.end_at <= self.start_at:
            raise ValueError("scenario.end_at must follow scenario.start_at")
        try:
            scenario_zone = ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"unknown scenario timezone: {self.timezone}") from error

        timestamps: list[tuple[str, datetime]] = [
            ("scenario.start_at", self.start_at),
            ("scenario.end_at", self.end_at),
        ]
        timestamps.extend((f"event {event.id}", event.at) for event in self.events)
        timestamps.extend(
            (f"expected action {action.id} window_start", action.window_start)
            for action in self.expected_actions
        )
        timestamps.extend(
            (f"expected action {action.id} window_end", action.window_end)
            for action in self.expected_actions
        )
        timestamps.extend(
            (f"forbidden action {action.id} window_start", action.window_start)
            for action in self.forbidden_actions
        )
        timestamps.extend(
            (f"forbidden action {action.id} window_end", action.window_end)
            for action in self.forbidden_actions
        )
        for field_name, timestamp in timestamps:
            if timestamp.utcoffset() != timestamp.astimezone(scenario_zone).utcoffset():
                raise ValueError(
                    f"{field_name} offset disagrees with timezone {self.timezone}"
                )

        event_ids = [event.id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event IDs must be unique within a scenario")
        event_times = [event.at for event in self.events]
        if len(event_times) != len(set(event_times)):
            raise ValueError(
                "event timestamps must be unique so each event is one checkpoint"
            )
        action_ids = [action.id for action in self.expected_actions]
        forbidden_ids = [action.id for action in self.forbidden_actions]
        all_action_ids = action_ids + forbidden_ids
        if len(all_action_ids) != len(set(all_action_ids)):
            raise ValueError("expected and forbidden action IDs must be unique")

        if self.events != sorted(self.events, key=lambda event: event.at):
            raise ValueError("scenario events must be in chronological order")
        known_events: set[str] = set()
        events_by_id = {event.id: event for event in self.events}
        for event in self.events:
            if event.kind == "assistant_decision":
                raise ValueError(
                    "assistant_decision events are generated by the runner, "
                    "not authored in scenarios"
                )
            if not self.start_at <= event.at <= self.end_at:
                raise ValueError(f"event {event.id} falls outside the scenario")
            unknown = set(event.supersedes) - known_events
            if unknown:
                raise ValueError(
                    f"event {event.id} supersedes unknown or future events: "
                    f"{sorted(unknown)}"
                )
            known_events.add(event.id)

        for action in self.expected_actions:
            if action.action_key not in known_events:
                raise ValueError(
                    f"expected action {action.id} has an unknown action_key: "
                    f"{action.action_key}"
                )
            if events_by_id[action.action_key].kind != "user_message":
                raise ValueError(
                    f"expected action {action.id} action_key must identify "
                    "the creating user_message"
                )
            if not self.start_at <= action.window_start <= self.end_at:
                raise ValueError(f"expected action {action.id} starts out of range")
            if not self.start_at <= action.window_end <= self.end_at:
                raise ValueError(f"expected action {action.id} ends out of range")
            for evidence_set in action.acceptable_evidence_sets:
                if not evidence_set:
                    raise ValueError(
                        f"expected action {action.id} has an empty evidence set"
                    )
                if len(evidence_set) != len(set(evidence_set)):
                    raise ValueError(
                        f"expected action {action.id} repeats an evidence ID"
                    )
                unknown = set(evidence_set) - known_events
                if unknown:
                    raise ValueError(
                        f"expected action {action.id} references unknown evidence: "
                        f"{sorted(unknown)}"
                    )
                future = {
                    event.id
                    for event in self.events
                    if event.id in evidence_set and event.at > action.window_end
                }
                if future:
                    raise ValueError(
                        f"expected action {action.id} uses future evidence: "
                        f"{sorted(future)}"
                    )
            if not any(
                action.window_start <= event.at <= action.window_end
                for event in self.events
            ):
                raise ValueError(
                    f"expected action {action.id} has no decision checkpoint "
                    "inside its window"
                )

        for action in self.forbidden_actions:
            if action.action_key not in known_events:
                raise ValueError(
                    f"forbidden action {action.id} has an unknown action_key: "
                    f"{action.action_key}"
                )
            if events_by_id[action.action_key].kind != "user_message":
                raise ValueError(
                    f"forbidden action {action.id} action_key must identify "
                    "the creating user_message"
                )
            if not self.start_at <= action.window_start <= self.end_at:
                raise ValueError(f"forbidden action {action.id} starts out of range")
            if not self.start_at <= action.window_end <= self.end_at:
                raise ValueError(f"forbidden action {action.id} ends out of range")
            unknown = set(action.related_event_ids) - known_events
            if unknown:
                raise ValueError(
                    f"forbidden action {action.id} references unknown events: "
                    f"{sorted(unknown)}"
                )
            if not any(
                action.window_start <= event.at <= action.window_end
                for event in self.events
            ):
                raise ValueError(
                    f"forbidden action {action.id} has no decision checkpoint "
                    "inside its window"
                )
        return self


class ProposedAction(StrictModel):
    """The action payload requested from every evaluated model."""

    kind: ActionKind = "reminder"
    action_key: str = Field(min_length=1)
    payload: dict[str, ActionValue] = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence_event_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        validate_action_payload(self.payload)
        return self


class Decision(StrictModel):
    """A model decision at one simulated point in time."""

    actions: list[ProposedAction] = Field(default_factory=list)


class PredictedAction(ProposedAction):
    """A proposed action annotated by the runner with its emission point."""

    emitted_at: datetime
    decision_event_id: str

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        _require_aware(self.emitted_at, "predicted_action.emitted_at")
        return self


class Usage(StrictModel):
    """Provider and retrieval usage accumulated over a scenario."""

    input_tokens: int = Field(default=0, ge=0)
    uncached_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    cache_write_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    embedding_inputs: int = Field(default=0, ge=0)
    embedding_characters: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_input_breakdown(self) -> Self:
        breakdown = (
            self.uncached_input_tokens
            + self.cache_read_input_tokens
            + self.cache_write_input_tokens
        )
        if breakdown != self.input_tokens:
            raise ValueError("logical input tokens differ from cache breakdown")
        return self

    def plus(self, other: Usage) -> Usage:
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            uncached_input_tokens=(
                self.uncached_input_tokens + other.uncached_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            cache_write_input_tokens=(
                self.cache_write_input_tokens + other.cache_write_input_tokens
            ),
            output_tokens=self.output_tokens + other.output_tokens,
            embedding_inputs=self.embedding_inputs + other.embedding_inputs,
            embedding_characters=(
                self.embedding_characters + other.embedding_characters
            ),
            cost_usd=cost,
        )


class CheckpointAudit(StrictModel):
    """Recomputable trace for one authored decision checkpoint."""

    event_id: str = Field(min_length=1)
    at: datetime
    compiler_called: bool = False
    raw_compiler_output: str | None = None
    memory_delta_json: str | None = None
    memory_delta_accepted: bool | None = None
    memory_delta_error: str | None = None
    state_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    due_candidate_ids: list[str] = Field(default_factory=list)
    rendered_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_decision_output: str
    compiler_usage: Usage = Field(default_factory=Usage)
    decision_usage: Usage = Field(default_factory=Usage)
    compiler_latency_ms: float = Field(default=0, ge=0)
    decision_latency_ms: float = Field(default=0, ge=0)
    local_latency_ms: float = Field(default=0, ge=0)
    compiler_parse_error: bool = False
    decision_parse_error: bool = False

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        _require_aware(self.at, "checkpoint_audit.at")
        if not self.compiler_called:
            if self.raw_compiler_output is not None:
                raise ValueError("a skipped compiler cannot have raw output")
            if self.memory_delta_accepted is not None:
                raise ValueError("a skipped compiler cannot accept a memory delta")
            if self.memory_delta_json is not None:
                raise ValueError("a skipped compiler cannot have a memory delta")
            if self.memory_delta_error is not None:
                raise ValueError("a skipped compiler cannot have a memory error")
            if self.compiler_parse_error:
                raise ValueError("a skipped compiler cannot have a parse error")
        if self.memory_delta_accepted is True and self.memory_delta_error is not None:
            raise ValueError("an accepted memory delta cannot have an error")
        return self


class HostedWarmupAttestation(StrictModel):
    """Auditable setup call excluded from measured scenario headline usage."""

    protocol_version: Literal[1] = 1
    model: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_completion: str = Field(min_length=1)
    usage: Usage
    usage_complete: bool
    cost_complete: bool
    parse_error: bool
    latency_ms: float = Field(ge=0)
    included_in_headline: Literal[False] = False

    @model_validator(mode="after")
    def validate_accounting(self) -> Self:
        if self.usage_complete and self.usage.input_tokens <= 0:
            raise ValueError("usage-complete warmup must report input tokens")
        if self.cost_complete and self.usage.cost_usd is None:
            raise ValueError("cost-complete warmup must report cost")
        return self


class ScenarioRun(StrictModel):
    """Reproducible raw output for one system, scenario, and repetition."""

    schema_version: Literal[2] = 2
    scenario_id: str
    system: str
    repetition: int = Field(ge=1)
    model: str
    prompt_version: str
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    pricing_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    seed: int | None = None
    predictions: list[PredictedAction] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    decision_usage: Usage = Field(default_factory=Usage)
    compiler_usage: Usage = Field(default_factory=Usage)
    usage_complete: bool = True
    cost_complete: bool = False
    decision_latency_ms: float = Field(default=0, ge=0)
    compiler_latency_ms: float = Field(default=0, ge=0)
    local_latency_ms: float = Field(default=0, ge=0)
    setup_latency_ms: float = Field(default=0, ge=0)
    hosted_warmup: HostedWarmupAttestation | None = None
    checkpoint_latency_ms: list[float] = Field(default_factory=list)
    decision_parse_errors: int = Field(default=0, ge=0)
    compiler_parse_errors: int = Field(default=0, ge=0)
    parse_errors: int = Field(default=0, ge=0)
    checkpoints: list[CheckpointAudit] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_accounting(self) -> Self:
        if self.usage.input_tokens != (
            self.decision_usage.input_tokens + self.compiler_usage.input_tokens
        ):
            raise ValueError("total input tokens differ from component usage")
        if self.usage.output_tokens != (
            self.decision_usage.output_tokens + self.compiler_usage.output_tokens
        ):
            raise ValueError("total output tokens differ from component usage")
        for field_name in (
            "uncached_input_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
        ):
            total = getattr(self.usage, field_name)
            components = getattr(self.decision_usage, field_name) + getattr(
                self.compiler_usage, field_name
            )
            if total != components:
                raise ValueError(f"total {field_name} differs from component usage")
        if self.parse_errors != (
            self.decision_parse_errors + self.compiler_parse_errors
        ):
            raise ValueError(
                "parse_errors must equal decision_parse_errors + compiler_parse_errors"
            )
        if self.cost_complete and self.usage.cost_usd is None:
            raise ValueError("a cost-complete run must report total cost")
        if self.cost_complete:
            component_cost = (self.decision_usage.cost_usd or 0.0) + (
                self.compiler_usage.cost_usd or 0.0
            )
            if not math.isclose(
                self.usage.cost_usd or 0.0,
                component_cost,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("total cost differs from component usage")
        if self.checkpoints and len(self.checkpoints) != len(
            self.checkpoint_latency_ms
        ):
            raise ValueError("checkpoint trace and latency counts differ")
        if self.checkpoints:
            traced_decision_usage = Usage()
            traced_compiler_usage = Usage()
            for checkpoint in self.checkpoints:
                traced_decision_usage = traced_decision_usage.plus(
                    checkpoint.decision_usage
                )
                traced_compiler_usage = traced_compiler_usage.plus(
                    checkpoint.compiler_usage
                )
            if traced_decision_usage != self.decision_usage:
                raise ValueError("decision usage differs from checkpoint trace")
            if traced_compiler_usage != self.compiler_usage:
                raise ValueError("compiler usage differs from checkpoint trace")
            latency_checks = (
                (
                    self.decision_latency_ms,
                    sum(item.decision_latency_ms for item in self.checkpoints),
                    "decision",
                ),
                (
                    self.compiler_latency_ms,
                    sum(item.compiler_latency_ms for item in self.checkpoints),
                    "compiler",
                ),
                (
                    self.local_latency_ms,
                    sum(item.local_latency_ms for item in self.checkpoints),
                    "local",
                ),
            )
            for total, traced, component in latency_checks:
                if not math.isclose(total, traced, rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError(
                        f"{component} latency differs from checkpoint trace"
                    )
            if self.decision_parse_errors != sum(
                item.decision_parse_error for item in self.checkpoints
            ):
                raise ValueError("decision parse errors differ from trace")
            if self.compiler_parse_errors != sum(
                item.compiler_parse_error for item in self.checkpoints
            ):
                raise ValueError("compiler parse errors differ from trace")
        return self
