"""Framework-independent memory contracts for active recall and shadow evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class MemoryOperation(StrEnum):
    IGNORE = "ignore"
    UPSERT = "upsert"
    CANCEL = "cancel"


_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_MEMORY_KEY = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class MemoryObservation:
    task_id: str
    conversation_id: str
    turn_id: str
    source_event_id: str
    text: str
    language: str
    epoch: int = 0

    def __post_init__(self) -> None:
        for value in (
            self.task_id,
            self.conversation_id,
            self.turn_id,
            self.source_event_id,
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError("memory observation identifier is invalid")
        if not self.text or len(self.text.encode("utf-8")) > 32_768:
            raise ValueError("memory observation text is empty or too large")
        if not self.language or len(self.language.encode("utf-8")) > 32:
            raise ValueError("memory observation language is invalid")
        if (
            isinstance(self.epoch, bool)
            or not isinstance(self.epoch, int)
            or self.epoch < 0
        ):
            raise ValueError("memory observation epoch is invalid")


@dataclass(frozen=True, slots=True)
class MemoryRecallRequest:
    task_id: str
    conversation_id: str
    epoch: int
    query: str
    limit: int

    def __post_init__(self) -> None:
        for value in (self.task_id, self.conversation_id):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError("memory recall identifier is invalid")
        if (
            isinstance(self.epoch, bool)
            or not isinstance(self.epoch, int)
            or self.epoch < 0
        ):
            raise ValueError("memory recall epoch is invalid")
        if not self.query or len(self.query.encode("utf-8")) > 4096:
            raise ValueError("memory recall query is empty or too large")
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or not 1 <= self.limit <= 20
        ):
            raise ValueError("memory recall limit is invalid")


@dataclass(frozen=True, slots=True)
class MemoryItem:
    text: str
    score: float

    def __post_init__(self) -> None:
        if not self.text or len(self.text.encode("utf-8")) > 16_384:
            raise ValueError("memory item text is empty or too large")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
            or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("memory item score is invalid")


@dataclass(frozen=True, slots=True)
class MemoryRecall:
    items: tuple[MemoryItem, ...] = ()
    provider: str = "disabled"

    def __post_init__(self) -> None:
        if len(self.items) > 20:
            raise ValueError("memory recall returned too many items")
        if not _IDENTIFIER.fullmatch(self.provider):
            raise ValueError("memory recall provider is invalid")


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    stored: bool
    provider: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.provider):
            raise ValueError("memory writer provider is invalid")
        if not _SHA256.fullmatch(self.content_sha256):
            raise ValueError("memory writer content hash is invalid")


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    operation: MemoryOperation
    memory_key: str | None
    supersedes_source_event_ids: tuple[str, ...]
    summary: str
    writer_contract_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation, MemoryOperation):
            raise TypeError("memory proposal operation is invalid")
        if self.operation is MemoryOperation.IGNORE:
            if self.memory_key is not None or self.supersedes_source_event_ids:
                raise ValueError("ignored memory proposal must be empty")
        elif self.memory_key is None or not _MEMORY_KEY.fullmatch(self.memory_key):
            raise ValueError("memory proposal key is invalid")
        if len(self.supersedes_source_event_ids) > 1:
            raise ValueError("memory proposal may supersede at most one source")
        if len(set(self.supersedes_source_event_ids)) != len(
            self.supersedes_source_event_ids
        ):
            raise ValueError("memory proposal superseded sources must be unique")
        if any(
            not _IDENTIFIER.fullmatch(value)
            for value in self.supersedes_source_event_ids
        ):
            raise ValueError("memory proposal superseded source is invalid")
        if (
            self.operation is MemoryOperation.CANCEL
            and len(self.supersedes_source_event_ids) != 1
        ):
            raise ValueError("memory cancellation must identify one active source")
        if not self.summary or len(self.summary.encode("utf-8")) > 512:
            raise ValueError("memory proposal summary is empty or too large")
        if not _SHA256.fullmatch(self.writer_contract_sha256):
            raise ValueError("memory writer contract hash is invalid")


class ShadowMemoryServicePort(Protocol):
    """Proposes memory changes but cannot mutate authoritative state."""

    async def observe(self, observation: MemoryObservation) -> MemoryProposal: ...


class DisabledShadowMemoryService:
    async def observe(self, observation: MemoryObservation) -> MemoryProposal:
        del observation
        raise RuntimeError("memory_shadow_service_disabled")


__all__ = [
    "DisabledShadowMemoryService",
    "MemoryObservation",
    "MemoryItem",
    "MemoryOperation",
    "MemoryProposal",
    "MemoryRecall",
    "MemoryRecallRequest",
    "MemoryWriteResult",
    "ShadowMemoryServicePort",
]
