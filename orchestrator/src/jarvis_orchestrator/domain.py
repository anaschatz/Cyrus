"""Framework-independent domain types and stable wire enums."""

from __future__ import annotations

import hashlib
import json
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Mapping


class TaskState(StrEnum):
    ADMITTED = "ADMITTED"
    ROUTING = "ROUTING"
    LOADING_CONTEXT = "LOADING_CONTEXT"
    EXECUTING = "EXECUTING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    DECISION_WAITING_RECONNECT = "DECISION_WAITING_RECONNECT"
    RESPONDING = "RESPONDING"
    COMMITTING = "COMMITTING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    FAILED = "FAILED"


ACTIVE_STATES = frozenset(
    {
        TaskState.ROUTING,
        TaskState.LOADING_CONTEXT,
        TaskState.EXECUTING,
        TaskState.RESPONDING,
        TaskState.COMMITTING,
    }
)
PAUSED_STATES = frozenset(
    {TaskState.AWAITING_APPROVAL, TaskState.DECISION_WAITING_RECONNECT}
)
TERMINAL_STATES = frozenset(
    {
        TaskState.COMPLETED,
        TaskState.CANCELLED,
        TaskState.INTERRUPTED,
        TaskState.BUDGET_EXHAUSTED,
        TaskState.FAILED,
    }
)


class ReasonCode(StrEnum):
    USER_CANCELLED = "user_cancelled"
    SUPERSEDED = "superseded"
    CLIENT_DISCONNECTED = "client_disconnected"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_RECONNECT_TIMEOUT = "approval_reconnect_timeout"
    SERVICE_INTERRUPTED = "service_interrupted"
    ACTIVE_BUDGET_EXHAUSTED = "active_budget_exhausted"
    ABSOLUTE_LIFETIME_EXHAUSTED = "absolute_lifetime_exhausted"
    BACKEND_FAILURE = "backend_failure"
    INTERNAL_FAILURE = "internal_failure"
    CONTINUATION_INCOMPATIBLE = "continuation_incompatible"
    CONVERSATION_DELETED = "conversation_deleted"


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class OrchestratorError(RuntimeError):
    def __init__(self, code: str, *, status: int = 409, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.retryable = retryable


class StaleEpochError(OrchestratorError):
    def __init__(self) -> None:
        super().__init__("stale_epoch", status=409)


class InvalidStateError(OrchestratorError):
    def __init__(self) -> None:
        super().__init__("invalid_state", status=409)


class BudgetExhaustedError(OrchestratorError):
    def __init__(self, code: str = "active_budget_exhausted") -> None:
        super().__init__(code, status=409)


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_text(value: datetime | None = None) -> str:
    selected = value or utc_now()
    return selected.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def new_uuid4() -> str:
    return str(uuid.uuid4())


def require_uuid4(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise OrchestratorError("invalid_request", status=400) from exc
    if parsed.version != 4 or str(parsed) != value.lower():
        raise OrchestratorError("invalid_request", status=400)
    return value.lower()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    task_id: str
    conversation_id: str
    turn_id: str
    response_id: str
    epoch: int
    state: TaskState
    admitted_at: str
    absolute_expires_at: str
    active_elapsed_seconds: float
    active_started_at: str | None
    disconnect_expires_at: str | None
    delivery_expires_at: str | None
    tombstoned: bool
    partial_output: bool

    def remaining_active_seconds(self, *, now: datetime | None = None, limit: float = 120.0) -> float:
        elapsed = self.active_elapsed_seconds
        selected = now or utc_now()
        if self.active_started_at is not None and self.state in ACTIVE_STATES:
            elapsed += max(0.0, (selected - parse_utc(self.active_started_at)).total_seconds())
        return max(0.0, limit - elapsed)

    def remaining_absolute_seconds(self, *, now: datetime | None = None) -> float:
        return max(0.0, (parse_utc(self.absolute_expires_at) - (now or utc_now())).total_seconds())


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    status: str
    ack_event_id: str
    task: TaskSnapshot | None


@dataclass(frozen=True, slots=True)
class LogicalEvent:
    event_id: str
    conversation_id: str
    task_id: str | None
    task_epoch: int | None
    guard_epoch: int
    turn_id: str | None
    response_id: str | None
    ordinal: int
    event_type: str
    payload: Mapping[str, Any]
    original_sent_at: str
    delivery_expires_at: str | None


@dataclass(frozen=True, slots=True)
class ModelRequest:
    task_id: str
    epoch: int
    purpose: str
    system_prompt: str
    messages: tuple[Mapping[str, str], ...]
    output_allowance: int
    timeout_seconds: float | None = None
    stream: bool = True
    tools: tuple[Mapping[str, Any], ...] = ()
    response_format: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ModelUsage:
    output_tokens: int | None = None
    input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelChunk:
    text: str
    usage: ModelUsage | None = None


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    memory_query: str
    research_query: str
    tool_name: str
    response_language: str

    def validate(self, registered_tools: frozenset[str] = frozenset()) -> "RouteDecision":
        if self.route not in {"local", "memory_local", "research_local", "cloud", "tool_local"}:
            raise OrchestratorError("route_validation_failed", status=409)
        if self.response_language not in {"el", "en", "default"}:
            raise OrchestratorError("route_validation_failed", status=409)
        valid_tool = self.tool_name == "none" or self.tool_name in registered_tools
        relationships = (
            (self.route == "memory_local") == bool(self.memory_query),
            (self.route == "research_local") == bool(self.research_query),
            (self.route == "tool_local") == (self.tool_name != "none"),
            valid_tool,
        )
        if not all(relationships):
            raise OrchestratorError("route_validation_failed", status=409)
        return self


@dataclass(frozen=True, slots=True)
class CanonicalEffectV1:
    kind: str
    name: str
    status: str
    arguments_sha256: str
    result_sha256: str = ""


@dataclass(frozen=True, slots=True)
class CanonicalTurnV1:
    turn_id: str
    user_text: str
    user_language: str
    assistant_text: str
    effects: tuple[CanonicalEffectV1, ...] = ()


@dataclass(frozen=True, slots=True)
class CanonicalConversationV1:
    conversation_id: str
    revision: int
    turns: tuple[CanonicalTurnV1, ...] = ()


@dataclass(frozen=True, slots=True)
class ContinuationArtifactV1:
    artifact_id: str
    artifact_version: int
    task_id: str
    conversation_id: str
    epoch: int
    framework_version: str
    canonical_revision: int
    canonical_sha256: str
    model_digest: str
    profile_digest: str
    tool_schema_digest: str
    pydantic_conversation_id: str
    paused_run_id: str
    messages_json: str
    deferred_requests_json: str
    arguments_sha256: str
    principal_sha256: str
    policy_version: str
    usage_json: str
    reservations_json: str
    created_at: str
    expires_at: str


DEFAULT_ACTIVE_BUDGET_SECONDS = 120.0
DEFAULT_ABSOLUTE_LIFETIME = timedelta(minutes=15)
DEFAULT_APPROVAL_TTL = timedelta(minutes=10)
DEFAULT_DISCONNECT_TTL = timedelta(seconds=30)
DEFAULT_DELIVERY_TTL = timedelta(seconds=30)
DEFAULT_MODEL_REQUESTS = 4
DEFAULT_OUTPUT_TOKENS = 2048
