"""The only time-authority seam available to current or future timer tools."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .scheduler_contract import CompiledProposal


SCHEDULER_PERSISTENCE_SCHEMA_VERSION = 5
TOOL_ACTION_SCHEMA_VERSION = 6


class SchedulerPort(Protocol):
    def create_proposal(self, compiled: CompiledProposal, **kwargs: Any) -> Mapping[str, Any]: ...
    def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]: ...


class TimerToolAdapterV6:
    """Delegation-only adapter: it has no OS scheduling or persistence path."""

    def __init__(self, scheduler: SchedulerPort) -> None:
        self.scheduler = scheduler

    def propose(self, compiled: CompiledProposal) -> Mapping[str, Any]:
        return self.scheduler.create_proposal(compiled)

    def list(self, *, limit: int = 50, cursor: tuple[str, str] | None = None) -> list[dict[str, Any]]:
        return self.scheduler.list_jobs(limit=limit, cursor=cursor)
