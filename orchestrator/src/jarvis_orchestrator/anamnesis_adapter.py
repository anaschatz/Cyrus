"""Qualified local Anamnesis retrospective-memory adapter.

Anamnesis remains a research package and its temporal reducer is intentionally
in-memory. JARVIS therefore consumes its closed, non-authoritative recall
boundary while keeping durable records and lifecycle CAS in Orchestrator
SQLite. No Anamnesis or provider type crosses the public MemoryPort.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, ClassVar

from .domain import canonical_json
from .memory import (
    MemoryDeleteResult,
    MemoryItem,
    MemoryObservation,
    MemoryRecall,
    MemoryRecallRequest,
    MemoryWriteResult,
    memory_search_terms,
    parse_explicit_memory_command,
)
from .store import OrchestratorStore

SUPPORTED_ANAMNESIS_CONTRACT_SHA256 = (
    "003fce5939c32a54a53bea28269b384a7a24cd584d76de60a7eed6f6cd877490"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _explicit_fact(value: str) -> str | None:
    command = parse_explicit_memory_command(value)
    return command[1] if command is not None and command[0] == "remember" else None


def _explicit_forget_target(value: str) -> str | None:
    command = parse_explicit_memory_command(value)
    return command[1] if command is not None and command[0] == "forget" else None


class _SQLiteRecallClient:
    """OpenMemory-compatible client backed by the Orchestrator authority."""

    is_local: ClassVar[bool] = True
    mode: ClassVar[str] = "local"
    embedding_provider: ClassVar[str] = "local"

    def __init__(self, store: OrchestratorStore) -> None:
        self._store = store
        self._verification_scopes: dict[str, str] = {}

    def add(
        self,
        content: str,
        *,
        user_id: str,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        required = {
            "conversation_id",
            "task_id",
            "task_epoch",
            "turn_id",
            "source_event_id",
            "language",
            "_anamnesis_namespace",
            "_anamnesis_user_id",
        }
        if set(metadata) != required:
            raise ValueError("memory metadata shape is invalid")
        conversation_id = str(metadata["conversation_id"])
        if user_id != f"anamnesis::jarvis::{conversation_id}":
            raise ValueError("memory scope does not match conversation")
        try:
            epoch = int(str(metadata["task_epoch"]))
        except ValueError as exc:
            raise ValueError("memory task epoch is invalid") from exc
        row = self._store.add_memory_record(
            conversation_id=conversation_id,
            task_id=str(metadata["task_id"]),
            epoch=epoch,
            source_event_id=str(metadata["source_event_id"]),
            scoped_user_id=user_id,
            content=content,
            metadata=metadata,
        )
        record_id = str(row["record_id"])
        self._verification_scopes[record_id] = user_id
        return self._provider_row(row, user_id=user_id)

    async def search(
        self, query: str, *, user_id: str, limit: int
    ) -> list[dict[str, object]]:
        prefix = "anamnesis::jarvis::"
        if not user_id.startswith(prefix) or len(user_id) == len(prefix):
            raise ValueError("memory scope is invalid")
        conversation_id = user_id[len(prefix) :]
        query_terms = memory_search_terms(query)
        if not query_terms:
            return []
        rows = await self._store.search_memory_records(
            conversation_id=conversation_id,
            query=query,
            limit=limit,
        )
        results: list[dict[str, object]] = []
        for row in rows:
            item = self._provider_row(row, user_id=user_id)
            item["score"] = min(1.0, int(row["match_count"]) / len(query_terms))
            results.append(item)
        return results

    def get(self, memory_id: str) -> dict[str, object] | None:
        row = self._store.memory_record(memory_id)
        if row is None:
            return None
        user_id = self._verification_scopes.pop(memory_id, None)
        return self._provider_row(row, user_id=user_id)

    def delete(self, memory_id: str) -> dict[str, bool]:
        return {"deleted": self._store.delete_memory_record(memory_id)}

    @staticmethod
    def _provider_row(
        row: Mapping[str, Any], *, user_id: str | None = None
    ) -> dict[str, object]:
        metadata = row.get("metadata")
        if metadata is None and "metadata_json" in row:
            metadata = json.loads(str(row["metadata_json"]))
        projected_metadata = dict(metadata or {})
        if user_id is None:
            user_id = f"anamnesis::jarvis::{row['conversation_id']}"
        prefix = "anamnesis::jarvis::"
        projected_metadata["_anamnesis_namespace"] = "jarvis"
        projected_metadata["_anamnesis_user_id"] = user_id[len(prefix) :]
        return {
            "id": str(row["record_id"]),
            "content": str(row["content"]),
            "user_id": user_id,
            "metadata": projected_metadata,
        }


class AnamnesisMemoryAdapter:
    """MemoryPort implementation qualified against the pinned Anamnesis contract."""

    provider = "anamnesis-recall"

    def __init__(self, store: OrchestratorStore) -> None:
        try:
            from anamnesis.openmemory_recall import OpenMemoryRecallIndex
            from anamnesis.runtime_contract import anamnesis_runtime_contract
        except ImportError as exc:
            raise RuntimeError(
                "The qualified Anamnesis runtime package is unavailable"
            ) from exc
        contract = anamnesis_runtime_contract()
        digest = _sha256_text(canonical_json(contract))
        if digest != SUPPORTED_ANAMNESIS_CONTRACT_SHA256:
            raise RuntimeError("Anamnesis runtime contract is not qualified")
        self._contract = dict(contract)
        self._contract_sha256 = digest
        self._index_type = OpenMemoryRecallIndex
        self._store = store
        self._client = _SQLiteRecallClient(store)

    @property
    def qualification(self) -> dict[str, object]:
        return {
            "qualified": True,
            "provider": self.provider,
            "contract_sha256": self._contract_sha256,
            "contract": dict(self._contract),
            "persistence_owner": "orchestrator-sqlite",
            "authoritative": False,
            "network": False,
        }

    def _index(self, conversation_id: str):
        return self._index_type(
            namespace="jarvis",
            user_id=conversation_id,
            client=self._client,
            local_only=True,
        )

    async def search(self, request: MemoryRecallRequest) -> MemoryRecall:
        result = await self._index(request.conversation_id).search(
            request.query, limit=request.limit
        )
        items = tuple(
            MemoryItem(match.content, match.score) for match in result.matches
        )
        return MemoryRecall(items=items, provider=self.provider)

    async def remember(self, observation: MemoryObservation) -> MemoryWriteResult:
        fact = _explicit_fact(observation.text)
        if fact is None:
            return MemoryWriteResult(False, self.provider, _sha256_text(""))
        index = self._index(observation.conversation_id)
        await index.add(
            fact,
            metadata={
                "conversation_id": observation.conversation_id,
                "task_id": observation.task_id,
                "task_epoch": str(observation.epoch),
                "turn_id": observation.turn_id,
                "source_event_id": observation.source_event_id,
                "language": observation.language,
            },
        )
        return MemoryWriteResult(True, self.provider, _sha256_text(fact))

    async def forget(self, observation: MemoryObservation) -> MemoryDeleteResult:
        target = _explicit_forget_target(observation.text)
        if target is None:
            return MemoryDeleteResult(False, False, 0, self.provider, _sha256_text(""))
        matches = self._store.memory_records_matching(
            conversation_id=observation.conversation_id,
            target=target,
        )
        selected = matches[0] if len(matches) == 1 else None
        outcome = self._store.record_explicit_memory_delete(
            task_id=observation.task_id,
            conversation_id=observation.conversation_id,
            epoch=observation.epoch,
            target_sha256=_sha256_text(target),
            record_id=None if selected is None else str(selected["record_id"]),
            expected_envelope_sha256=(
                None if selected is None else str(selected["envelope_sha256"])
            ),
            matched_count=len(matches),
        )
        return MemoryDeleteResult(
            True,
            outcome["deleted_count"] == 1,
            int(outcome["matched_count"]),
            self.provider,
            _sha256_text(target),
        )


__all__ = [
    "SUPPORTED_ANAMNESIS_CONTRACT_SHA256",
    "AnamnesisMemoryAdapter",
]
