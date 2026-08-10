"""Deterministic permission policy; models never grant their own authority."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import OrchestratorError


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    kind: str
    name: str
    approval_required: bool
    policy_version: str = "jarvis-permissions/v1"


class PermissionPolicyV1:
    def __init__(self, *, registered_tools: tuple[str, ...] = ()) -> None:
        if len(set(registered_tools)) != len(registered_tools):
            raise ValueError("tool registrations must be unique")
        self.registered_tools = frozenset(registered_tools)

    def decide(self, kind: str, name: str) -> PermissionDecision:
        if kind == "tool" and name not in self.registered_tools:
            raise OrchestratorError("invalid_request", status=400)
        if kind not in {"research", "cloud_model", "tool"}:
            raise OrchestratorError("invalid_request", status=400)
        return PermissionDecision(kind, name, True)
