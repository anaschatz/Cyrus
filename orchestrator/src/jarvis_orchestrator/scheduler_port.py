"""The only time-authority seam available to current or future timer tools."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .scheduler_contract import CompiledProposal


class SchedulerPort(Protocol):
    def create_proposal(self, compiled: CompiledProposal, **kwargs: Any) -> Mapping[str, Any]: ...
    def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]: ...
