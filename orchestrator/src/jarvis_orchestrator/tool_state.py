"""Frozen Tool/Action v1 state machine and Scheduler dependency identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .tool_contracts import ToolContractError


class ToolRunState(StrEnum):
    PLANNED = "PLANNED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    DISPATCH_COMMITTED = "DISPATCH_COMMITTED"
    DISPATCHED = "DISPATCHED"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    ACCEPTED = "ACCEPTED"
    FAILED = "FAILED"
    AMBIGUOUS = "AMBIGUOUS"
    DENIED = "DENIED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


TERMINAL_TOOL_STATES = frozenset(
    {
        ToolRunState.VERIFIED,
        ToolRunState.ACCEPTED,
        ToolRunState.FAILED,
        ToolRunState.AMBIGUOUS,
        ToolRunState.DENIED,
        ToolRunState.CANCELLED,
        ToolRunState.EXPIRED,
    }
)


class ToolRunEvent(StrEnum):
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    REVALIDATION_FAILED = "REVALIDATION_FAILED"
    PERMIT_COMMITTED = "PERMIT_COMMITTED"
    HOST_RECEIVED = "HOST_RECEIVED"
    PRE_CALL_REJECTED = "PRE_CALL_REJECTED"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    OS_REJECTED = "OS_REJECTED"
    OS_ACCEPTED_WITH_VERIFIER = "OS_ACCEPTED_WITH_VERIFIER"
    OS_ACCEPTED_NO_VERIFIER = "OS_ACCEPTED_NO_VERIFIER"
    VERIFICATION_MATCHED = "VERIFICATION_MATCHED"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"
    VERIFICATION_MISMATCH = "VERIFICATION_MISMATCH"


_TRANSITIONS: dict[tuple[ToolRunState, ToolRunEvent], ToolRunState] = {
    (ToolRunState.PLANNED, ToolRunEvent.APPROVAL_REQUIRED): ToolRunState.AWAITING_APPROVAL,
    (ToolRunState.PLANNED, ToolRunEvent.VALIDATION_FAILED): ToolRunState.FAILED,
    (ToolRunState.PLANNED, ToolRunEvent.CANCELLED): ToolRunState.CANCELLED,
    (ToolRunState.AWAITING_APPROVAL, ToolRunEvent.APPROVED): ToolRunState.APPROVED,
    (ToolRunState.AWAITING_APPROVAL, ToolRunEvent.DENIED): ToolRunState.DENIED,
    (ToolRunState.AWAITING_APPROVAL, ToolRunEvent.EXPIRED): ToolRunState.EXPIRED,
    (ToolRunState.AWAITING_APPROVAL, ToolRunEvent.CANCELLED): ToolRunState.CANCELLED,
    (ToolRunState.APPROVED, ToolRunEvent.EXPIRED): ToolRunState.EXPIRED,
    (ToolRunState.APPROVED, ToolRunEvent.CANCELLED): ToolRunState.CANCELLED,
    (ToolRunState.APPROVED, ToolRunEvent.REVALIDATION_FAILED): ToolRunState.FAILED,
    (ToolRunState.APPROVED, ToolRunEvent.PERMIT_COMMITTED): ToolRunState.DISPATCH_COMMITTED,
    (ToolRunState.DISPATCH_COMMITTED, ToolRunEvent.HOST_RECEIVED): ToolRunState.DISPATCHED,
    (ToolRunState.DISPATCH_COMMITTED, ToolRunEvent.PRE_CALL_REJECTED): ToolRunState.FAILED,
    (ToolRunState.DISPATCH_COMMITTED, ToolRunEvent.OUTCOME_UNKNOWN): ToolRunState.AMBIGUOUS,
    (ToolRunState.DISPATCHED, ToolRunEvent.PRE_CALL_REJECTED): ToolRunState.FAILED,
    (ToolRunState.DISPATCHED, ToolRunEvent.OS_REJECTED): ToolRunState.FAILED,
    (ToolRunState.DISPATCHED, ToolRunEvent.OS_ACCEPTED_WITH_VERIFIER): ToolRunState.VERIFYING,
    (ToolRunState.DISPATCHED, ToolRunEvent.OS_ACCEPTED_NO_VERIFIER): ToolRunState.ACCEPTED,
    (ToolRunState.DISPATCHED, ToolRunEvent.OUTCOME_UNKNOWN): ToolRunState.AMBIGUOUS,
    (ToolRunState.VERIFYING, ToolRunEvent.VERIFICATION_MATCHED): ToolRunState.VERIFIED,
    (ToolRunState.VERIFYING, ToolRunEvent.VERIFICATION_UNAVAILABLE): ToolRunState.ACCEPTED,
    (ToolRunState.VERIFYING, ToolRunEvent.VERIFICATION_MISMATCH): ToolRunState.AMBIGUOUS,
    (ToolRunState.VERIFYING, ToolRunEvent.OUTCOME_UNKNOWN): ToolRunState.AMBIGUOUS,
}


def transition_tool_state(current: ToolRunState | str, event: ToolRunEvent | str) -> ToolRunState:
    selected_state = ToolRunState(current)
    selected_event = ToolRunEvent(event)
    if selected_state in TERMINAL_TOOL_STATES:
        raise ToolContractError("tool.binding_mismatch", status=409)
    try:
        return _TRANSITIONS[(selected_state, selected_event)]
    except KeyError as exc:
        raise ToolContractError("tool.binding_mismatch", status=409) from exc


def cancellation_is_terminal(state: ToolRunState | str) -> bool:
    return ToolRunState(state) in {
        ToolRunState.PLANNED,
        ToolRunState.AWAITING_APPROVAL,
        ToolRunState.APPROVED,
    }


def restart_event(state: ToolRunState | str, *, authenticated_pre_call_rejection: bool) -> ToolRunEvent | None:
    selected = ToolRunState(state)
    if selected is ToolRunState.DISPATCH_COMMITTED:
        return ToolRunEvent.PRE_CALL_REJECTED if authenticated_pre_call_rejection else ToolRunEvent.OUTCOME_UNKNOWN
    if selected in {ToolRunState.DISPATCHED, ToolRunState.VERIFYING} and not authenticated_pre_call_rejection:
        return ToolRunEvent.OUTCOME_UNKNOWN
    return None


def _sha(value: str, *, empty_allowed: bool = False) -> str:
    if empty_allowed and value == "":
        return value
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ToolContractError("tool.readiness_changed", status=503)
    return value


@dataclass(frozen=True, slots=True)
class SchedulerDependencyV1:
    scheduler_schema_version: int
    scheduler_contract_sha256: str
    scheduler_qualification_manifest_sha256: str
    scheduler_ready: bool
    scheduler_observed_at_utc: str
    scheduler_reason_code: str

    def __post_init__(self) -> None:
        if isinstance(self.scheduler_schema_version, bool) or self.scheduler_schema_version < 1:
            raise ToolContractError("tool.readiness_changed", status=503)
        _sha(self.scheduler_contract_sha256)
        _sha(self.scheduler_qualification_manifest_sha256, empty_allowed=True)
        if not isinstance(self.scheduler_ready, bool):
            raise ToolContractError("tool.readiness_changed", status=503)
        if not self.scheduler_observed_at_utc or not self.scheduler_reason_code:
            raise ToolContractError("tool.readiness_changed", status=503)

    def assert_v6_compatible(self, *, expected_contract_sha256: str) -> None:
        if self.scheduler_schema_version != 5 or self.scheduler_contract_sha256 != expected_contract_sha256:
            raise ToolContractError("tool.readiness_changed", status=503)

    def scheduler_capabilities_enabled(self, *, expected_qualification_sha256: str) -> bool:
        return (
            self.scheduler_ready
            and bool(expected_qualification_sha256)
            and self.scheduler_qualification_manifest_sha256 == expected_qualification_sha256
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler_schema_version": self.scheduler_schema_version,
            "scheduler_contract_sha256": self.scheduler_contract_sha256,
            "scheduler_qualification_manifest_sha256": self.scheduler_qualification_manifest_sha256,
            "scheduler_ready": self.scheduler_ready,
            "scheduler_observed_at_utc": self.scheduler_observed_at_utc,
            "scheduler_reason_code": self.scheduler_reason_code,
        }
