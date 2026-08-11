"""JARVIS Orchestrator Core v1.0 with dependency-light lazy exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import OrchestratorConfig
    from .domain import TaskState
    from .service import OrchestratorService
    from .store import OrchestratorStore

__all__ = ["OrchestratorConfig", "OrchestratorService", "OrchestratorStore", "TaskState"]

__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    if name == "OrchestratorConfig":
        from .config import OrchestratorConfig

        return OrchestratorConfig
    if name == "TaskState":
        from .domain import TaskState

        return TaskState
    if name == "OrchestratorService":
        from .service import OrchestratorService

        return OrchestratorService
    if name == "OrchestratorStore":
        from .store import OrchestratorStore

        return OrchestratorStore
    raise AttributeError(name)
