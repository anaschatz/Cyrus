"""Transparent runtime strategies for the three simple v0 baselines."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from anamnesis.memory import CompilerRequest, InMemoryAnamnesis, MemoryCompiler
from anamnesis.schema import Decision, MemoryView, ObservableEvent, Usage


@dataclass(frozen=True)
class RetrievalUsage:
    """Local retrieval work performed at one decision point."""

    latency_ms: float = 0.0
    embedding_inputs: int = 0
    embedding_characters: int = 0

    def plus(self, other: RetrievalUsage) -> RetrievalUsage:
        return RetrievalUsage(
            latency_ms=self.latency_ms + other.latency_ms,
            embedding_inputs=self.embedding_inputs + other.embedding_inputs,
            embedding_characters=(
                self.embedding_characters + other.embedding_characters
            ),
        )


@dataclass(frozen=True)
class StrategyWork:
    """Accounting and audit data produced by strategy ingest/commit work."""

    local_usage: RetrievalUsage = field(default_factory=RetrievalUsage)
    compiler_usage: Usage = field(default_factory=Usage)
    compiler_latency_ms: float = 0.0
    compiler_called: bool = False
    compiler_parse_error: bool = False
    raw_compiler_output: str | None = None
    memory_delta_json: str | None = None
    memory_delta_accepted: bool | None = None
    memory_delta_error: str | None = None
    state_sha256: str | None = None
    due_candidate_ids: list[str] = field(default_factory=list)
    usage_complete: bool = True
    cost_complete: bool = True

    def plus(self, other: StrategyWork) -> StrategyWork:
        return StrategyWork(
            local_usage=self.local_usage.plus(other.local_usage),
            compiler_usage=self.compiler_usage.plus(other.compiler_usage),
            compiler_latency_ms=(self.compiler_latency_ms + other.compiler_latency_ms),
            compiler_called=self.compiler_called or other.compiler_called,
            compiler_parse_error=(
                self.compiler_parse_error or other.compiler_parse_error
            ),
            raw_compiler_output=(
                other.raw_compiler_output
                if other.raw_compiler_output is not None
                else self.raw_compiler_output
            ),
            memory_delta_json=(
                other.memory_delta_json
                if other.memory_delta_json is not None
                else self.memory_delta_json
            ),
            memory_delta_accepted=(
                other.memory_delta_accepted
                if other.memory_delta_accepted is not None
                else self.memory_delta_accepted
            ),
            memory_delta_error=(
                other.memory_delta_error
                if other.memory_delta_error is not None
                else self.memory_delta_error
            ),
            state_sha256=other.state_sha256 or self.state_sha256,
            due_candidate_ids=other.due_candidate_ids or self.due_candidate_ids,
            usage_complete=self.usage_complete and other.usage_complete,
            cost_complete=self.cost_complete and other.cost_complete,
        )


@dataclass(frozen=True)
class DecisionHistoryRecord:
    """A prior decision kept separately from user-visible observable events."""

    event_id: str
    at: datetime
    decision: Decision


@dataclass(frozen=True)
class ContextSelection:
    events: list[ObservableEvent]
    decisions: list[DecisionHistoryRecord] = field(default_factory=list)
    memory_view: MemoryView | None = None
    retrospective_recall: tuple[str, ...] | None = None
    state_sha256: str | None = None
    due_candidate_ids: list[str] = field(default_factory=list)
    usage: RetrievalUsage = field(default_factory=RetrievalUsage)


class MemoryStrategy(Protocol):
    """Provider-neutral lifecycle shared by all evaluated systems."""

    name: str

    def reset(self) -> None: ...

    async def ingest(self, event: ObservableEvent) -> StrategyWork: ...

    def select(self, current: ObservableEvent) -> ContextSelection: ...

    def commit(self, current: ObservableEvent, decision: Decision) -> StrategyWork: ...


class NoPersistentMemory:
    """Expose only the event at the current decision point."""

    name = "no_memory"

    def reset(self) -> None:
        return None

    async def ingest(self, event: ObservableEvent) -> StrategyWork:
        return StrategyWork()

    def select(self, current: ObservableEvent) -> ContextSelection:
        return ContextSelection(events=[current])

    def commit(self, current: ObservableEvent, decision: Decision) -> StrategyWork:
        return StrategyWork()


class FullContextMemory:
    """Expose every event observed so far in chronological order."""

    name = "full_context"

    def __init__(self) -> None:
        self._events: list[ObservableEvent] = []
        self._decisions: list[DecisionHistoryRecord] = []

    def reset(self) -> None:
        self._events = []
        self._decisions = []

    async def ingest(self, event: ObservableEvent) -> StrategyWork:
        self._events.append(event)
        return StrategyWork()

    def select(self, current: ObservableEvent) -> ContextSelection:
        return ContextSelection(
            events=list(self._events),
            decisions=list(self._decisions),
        )

    def commit(self, current: ObservableEvent, decision: Decision) -> StrategyWork:
        self._decisions.append(
            DecisionHistoryRecord(
                event_id=current.id,
                at=current.at,
                decision=decision,
            )
        )
        return StrategyWork()


class Vectorizer(Protocol):
    """Small interface that keeps the RAG baseline testable and provider-neutral."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedVectorizer:
    """CPU embeddings through ONNX Runtime, loaded only when first used."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        repository: str = "qdrant/bge-small-en-v1.5-onnx-q",
        revision: str | None = None,
        snapshot_path: str | Path | None = None,
    ) -> None:
        if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("embedding revision must be a 40-character commit SHA")
        self.model_name = model_name
        self.repository = repository
        self.revision = revision
        self.snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        self._model: object | None = None
        self._warmed = False
        self._artifact_sha256: str | None = None

    def _get_model(self) -> object:
        if self._model is None:
            from fastembed import TextEmbedding

            if self.revision is None:
                raise ValueError(
                    "FastEmbed requires an exact repository revision before use"
                )
            if self.snapshot_path is None:
                from huggingface_hub import snapshot_download

                self.snapshot_path = Path(
                    snapshot_download(
                        repo_id=self.repository,
                        revision=self.revision,
                    )
                )
            if not self.snapshot_path.is_dir():
                raise ValueError(
                    f"embedding snapshot path is not a directory: {self.snapshot_path}"
                )
            self._artifact_sha256 = _directory_sha256(self.snapshot_path)
            self._model = TextEmbedding(
                model_name=self.model_name,
                specific_model_path=str(self.snapshot_path),
                local_files_only=True,
            )
        return self._model

    @property
    def artifact_sha256(self) -> str:
        if self._artifact_sha256 is None:
            raise RuntimeError("embedding snapshot has not been prepared")
        return self._artifact_sha256

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        vectors = list(model.embed(texts))  # type: ignore[attr-defined]
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        model = self._get_model()
        query_embed = getattr(model, "query_embed", None)
        if query_embed is not None:
            vectors = list(query_embed(text))
        else:
            vectors = list(model.embed([text]))  # type: ignore[attr-defined]
        return np.asarray(vectors[0], dtype=np.float32)

    def warmup(self) -> float:
        """Load and exercise the immutable model outside checkpoint latency."""

        if self._warmed:
            return 0.0
        started = perf_counter()
        self.embed_documents(["Anamnesis embedding model warm-up."])
        latency_ms = (perf_counter() - started) * 1000
        self._warmed = True
        return latency_ms


def _directory_sha256(path: Path) -> str:
    """Hash relative filenames and bytes for one immutable local snapshot."""

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"embedding snapshot contains no files: {path}")
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or query.ndim != 1:
        raise ValueError("cosine search expects a matrix and one query vector")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError("document and query embedding dimensions differ")
    document_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    denominators = document_norms * query_norm
    numerators = matrix @ query
    return np.divide(
        numerators,
        denominators,
        out=np.full_like(numerators, -np.inf, dtype=np.float32),
        where=denominators > 0,
    )


class VectorRAGMemory:
    """Exact top-k retrieval over embedded non-clock conversation records."""

    name = "vector_rag"

    def __init__(self, vectorizer: Vectorizer, top_k: int = 5) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.vectorizer = vectorizer
        self.top_k = top_k
        self._events: list[ObservableEvent] = []
        self._vectors: list[np.ndarray] = []
        self._decisions: list[DecisionHistoryRecord] = []
        self._decision_vectors: list[np.ndarray] = []

    def reset(self) -> None:
        self._events = []
        self._vectors = []
        self._decisions = []
        self._decision_vectors = []

    async def ingest(self, event: ObservableEvent) -> StrategyWork:
        if event.kind == "clock_tick":
            return StrategyWork()
        started = perf_counter()
        vector = self.vectorizer.embed_documents([event.text])[0]
        latency_ms = (perf_counter() - started) * 1000
        self._events.append(event)
        self._vectors.append(vector)
        return StrategyWork(
            local_usage=RetrievalUsage(
                latency_ms=latency_ms,
                embedding_inputs=1,
                embedding_characters=len(event.text),
            )
        )

    def select(self, current: ObservableEvent) -> ContextSelection:
        candidates: list[
            tuple[ObservableEvent | DecisionHistoryRecord, np.ndarray, str]
        ] = []
        candidates.extend(
            (event, vector, event.text)
            for event, vector in zip(self._events, self._vectors, strict=True)
            if event.id != current.id
        )
        candidates.extend(
            (
                decision,
                vector,
                f"Assistant decision output: {decision.decision.model_dump_json()}",
            )
            for decision, vector in zip(
                self._decisions, self._decision_vectors, strict=True
            )
        )
        if not candidates:
            return ContextSelection(events=[current])

        query_text = (
            f"Current time: {current.at.isoformat()}. "
            "Retrieve prior facts and intentions relevant to deciding whether "
            f"an action is due now. Current event: {current.text}"
        )
        started = perf_counter()
        query = self.vectorizer.embed_query(query_text)
        matrix = np.vstack([candidate[1] for candidate in candidates])
        scores = _cosine_scores(matrix, query)
        ranked_positions = sorted(
            range(len(candidates)),
            key=lambda position: (
                -float(scores[position]),
                -position,
            ),
        )[: self.top_k]
        selected_records = [candidates[position][0] for position in ranked_positions]
        selected_events = [
            record for record in selected_records if isinstance(record, ObservableEvent)
        ]
        selected_decisions = [
            record
            for record in selected_records
            if isinstance(record, DecisionHistoryRecord)
        ]
        selected_events.append(current)
        selected_events.sort(key=lambda event: event.at)
        selected_decisions.sort(key=lambda decision: decision.at)
        latency_ms = (perf_counter() - started) * 1000
        return ContextSelection(
            events=selected_events,
            decisions=selected_decisions,
            usage=RetrievalUsage(
                latency_ms=latency_ms,
                embedding_inputs=1,
                embedding_characters=len(query_text),
            ),
        )

    def commit(self, current: ObservableEvent, decision: Decision) -> StrategyWork:
        record = DecisionHistoryRecord(
            event_id=current.id,
            at=current.at,
            decision=decision,
        )
        text = f"Assistant decision output: {decision.model_dump_json()}"
        started = perf_counter()
        vector = self.vectorizer.embed_documents([text])[0]
        latency_ms = (perf_counter() - started) * 1000
        self._decisions.append(record)
        self._decision_vectors.append(vector)
        return StrategyWork(
            local_usage=RetrievalUsage(
                latency_ms=latency_ms,
                embedding_inputs=1,
                embedding_characters=len(text),
            )
        )


class AnamnesisMemoryStrategy:
    """Online compiler plus deterministic versioned temporal memory."""

    name = "anamnesis"

    def __init__(
        self,
        compiler: MemoryCompiler,
        memory: InMemoryAnamnesis | None = None,
    ) -> None:
        self.compiler = compiler
        self.memory = memory or InMemoryAnamnesis()

    def reset(self) -> None:
        self.memory.reset()

    async def ingest(self, event: ObservableEvent) -> StrategyWork:
        if event.kind == "clock_tick":
            started = perf_counter()
            result = self.memory.ingest(event, None)
            local_latency_ms = (perf_counter() - started) * 1000
            return StrategyWork(
                local_usage=RetrievalUsage(latency_ms=local_latency_ms),
                state_sha256=result.state_sha256,
            )

        call = await self.compiler.compile(
            CompilerRequest(
                event=event,
                active_state=self.memory.compiler_state(),
            )
        )
        started = perf_counter()
        result = self.memory.ingest(event, call.delta)
        local_latency_ms = (perf_counter() - started) * 1000
        return StrategyWork(
            local_usage=RetrievalUsage(latency_ms=local_latency_ms),
            compiler_usage=call.usage,
            compiler_latency_ms=call.latency_ms,
            compiler_called=True,
            compiler_parse_error=call.parse_error,
            raw_compiler_output=call.raw_completion,
            memory_delta_json=(
                call.delta.model_dump_json() if call.delta is not None else None
            ),
            memory_delta_accepted=result.accepted,
            memory_delta_error=result.error,
            state_sha256=result.state_sha256,
            usage_complete=call.usage_complete,
            cost_complete=call.cost_complete,
        )

    def select(self, current: ObservableEvent) -> ContextSelection:
        started = perf_counter()
        selection = self.memory.select(current)
        local_latency_ms = (perf_counter() - started) * 1000
        return ContextSelection(
            events=[current],
            memory_view=selection.view,
            state_sha256=selection.state_sha256,
            due_candidate_ids=list(selection.due_candidate_ids),
            usage=RetrievalUsage(latency_ms=local_latency_ms),
        )

    def commit(self, current: ObservableEvent, decision: Decision) -> StrategyWork:
        started = perf_counter()
        result = self.memory.commit(current, decision)
        local_latency_ms = (perf_counter() - started) * 1000
        return StrategyWork(
            local_usage=RetrievalUsage(latency_ms=local_latency_ms),
            state_sha256=result.state_sha256,
        )


def create_strategy(
    name: str,
    *,
    vectorizer: Vectorizer | None = None,
    top_k: int = 5,
) -> MemoryStrategy:
    """Construct one of the frozen simple v0 memory strategies."""

    if name == "no_memory":
        return NoPersistentMemory()
    if name == "full_context":
        return FullContextMemory()
    if name == "vector_rag":
        return VectorRAGMemory(vectorizer or FastEmbedVectorizer(), top_k=top_k)
    raise ValueError(f"unknown baseline: {name}")


# Backward-compatible import name for callers while the v0 runtime migrates.
ContextSelector = MemoryStrategy
create_selector = create_strategy
