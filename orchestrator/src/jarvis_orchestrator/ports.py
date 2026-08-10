"""Typed ports. Production v1 wires only voice and local inference."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Mapping, Protocol

from .domain import ModelChunk, ModelRequest
from .memory import MemoryObservation, MemoryRecall, MemoryRecallRequest, MemoryWriteResult


class ModelPort(Protocol):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]: ...


class MemoryPort(Protocol):
    async def search(self, request: MemoryRecallRequest) -> MemoryRecall: ...

    async def remember(self, observation: MemoryObservation) -> MemoryWriteResult: ...


class ResearchPort(Protocol):
    async def research(self, query: str, *, idempotency_key: str) -> Mapping[str, Any]: ...


class ToolPort(Protocol):
    async def execute(self, name: str, arguments: Mapping[str, Any], *, idempotency_key: str) -> Mapping[str, Any]: ...


class DisabledMemoryPort:
    async def search(self, request: MemoryRecallRequest) -> MemoryRecall:
        del request
        return MemoryRecall()

    async def remember(self, observation: MemoryObservation) -> MemoryWriteResult:
        del observation
        return MemoryWriteResult(False, "disabled", "0" * 64)


class DisabledResearchPort:
    async def research(self, query: str, *, idempotency_key: str) -> Mapping[str, Any]:
        raise RuntimeError("research_disabled")


class DisabledToolPort:
    async def execute(self, name: str, arguments: Mapping[str, Any], *, idempotency_key: str) -> Mapping[str, Any]:
        raise RuntimeError("tool_disabled")
