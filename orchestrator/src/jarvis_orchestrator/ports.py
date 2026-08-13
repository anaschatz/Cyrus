"""Typed ports. Production v1 wires only voice and local inference."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol

from .cloud_contract import QualifiedEndpointV1
from .cloud_provider import CloudModelResultV1, OpenRouterRevalidationV1
from .cloud_routing import ContextScope, OutputContract, TaskProfileV1
from .domain import ModelChunk, ModelRequest
from .memory import (
    MemoryDeleteResult,
    MemoryObservation,
    MemoryRecall,
    MemoryRecallRequest,
    MemoryWriteResult,
)
from .research import (
    NormalizedDocumentV1,
    RankedPassageV1,
    ResearchCorpusV1,
    ResearchRequestV1,
    SearchPlanV1,
    SpanSelectionV1,
)
from .tool_contracts import ToolDefinitionV1, ToolOutcomeV1


class ModelPort(Protocol):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]: ...


class CloudModelPort(Protocol):
    async def revalidate(
        self, endpoint: QualifiedEndpointV1
    ) -> OpenRouterRevalidationV1: ...

    async def dispatch(self, **kwargs: Any) -> CloudModelResultV1: ...


class TaskProfilerPort(Protocol):
    async def profile_task_with_correction(
        self,
        request: ModelRequest,
        *,
        context_scope: ContextScope,
        output_contract: OutputContract,
        response_language: str,
    ) -> TaskProfileV1: ...


class MemoryPort(Protocol):
    async def search(self, request: MemoryRecallRequest) -> MemoryRecall: ...

    async def remember(self, observation: MemoryObservation) -> MemoryWriteResult: ...

    async def forget(self, observation: MemoryObservation) -> MemoryDeleteResult: ...


class ResearchPort(Protocol):
    async def research(
        self, request: ResearchRequestV1, *, idempotency_key: str
    ) -> ResearchCorpusV1: ...


class ResearchPlannerPort(Protocol):
    async def plan_research(
        self,
        *,
        task_id: str,
        epoch: int,
        query: str,
        language: str,
        timezone: str,
        browser_fallback_allowed: bool,
        output_allowance: int,
    ) -> SearchPlanV1 | None: ...

    async def select_research_spans(
        self,
        *,
        task_id: str,
        epoch: int,
        plan: SearchPlanV1,
        passages: tuple[RankedPassageV1, ...],
        documents: Mapping[str, NormalizedDocumentV1],
        output_allowance: int,
        timeout_seconds: float,
    ) -> tuple[SpanSelectionV1, ...]: ...


class ToolPort(Protocol):
    async def execute(
        self, name: str, arguments: Mapping[str, Any], *, idempotency_key: str
    ) -> Mapping[str, Any]: ...


class ToolPlannerPort(Protocol):
    async def plan_with_correction(self, request: ModelRequest): ...

    async def plan_tool_arguments(
        self,
        request: ModelRequest,
        *,
        definition: ToolDefinitionV1,
    ) -> Mapping[str, Any]: ...


class ToolExecutorPort(Protocol):
    async def execute_typed(
        self,
        *,
        execution_id: str,
        permit_id: str,
        tool_call_id: str,
        definition: ToolDefinitionV1,
        arguments: Mapping[str, Any],
        arguments_sha256: str,
        idempotency_key: str,
        registry_sha256: str,
        policy_sha256: str,
        scheduler_contract_sha256: str,
        user_intent_binding_sha256: str,
        deadline_utc: str,
    ) -> tuple[ToolOutcomeV1, Mapping[str, Any]]: ...

    async def acknowledge(self, execution_id: str, idempotency_key: str) -> None: ...


class DisabledMemoryPort:
    async def search(self, request: MemoryRecallRequest) -> MemoryRecall:
        del request
        return MemoryRecall()

    async def remember(self, observation: MemoryObservation) -> MemoryWriteResult:
        del observation
        return MemoryWriteResult(False, "disabled", "0" * 64)

    async def forget(self, observation: MemoryObservation) -> MemoryDeleteResult:
        del observation
        return MemoryDeleteResult(False, False, 0, "disabled", "0" * 64)


class DisabledResearchPort:
    async def research(
        self, request: ResearchRequestV1, *, idempotency_key: str
    ) -> ResearchCorpusV1:
        del request, idempotency_key
        raise RuntimeError("research_disabled")


class DisabledToolPort:
    async def execute(
        self, name: str, arguments: Mapping[str, Any], *, idempotency_key: str
    ) -> Mapping[str, Any]:
        raise RuntimeError("tool_disabled")
