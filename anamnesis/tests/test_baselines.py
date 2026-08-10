from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from anamnesis.baselines import (
    FastEmbedVectorizer,
    FullContextMemory,
    NoPersistentMemory,
    VectorRAGMemory,
    _directory_sha256,
)
from anamnesis.schema import ObservableEvent


def event(event_id: str, hour: int, text: str, kind: str = "user_message"):
    return ObservableEvent(
        id=event_id,
        at=datetime.fromisoformat(f"2026-03-02T{hour:02d}:00:00+02:00"),
        kind=kind,
        text=text,
    )


class KeywordVectorizer:
    def _vector(self, text: str) -> np.ndarray:
        lowered = text.casefold()
        return np.asarray(
            [
                float("assignment" in lowered),
                float("dentist" in lowered),
                0.1,
            ],
            dtype=np.float32,
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def test_no_memory_exposes_only_current_event() -> None:
    selector = NoPersistentMemory()
    first = event("e1", 9, "Remember the assignment")
    current = event("e2", 10, "A distractor")
    asyncio.run(selector.ingest(first))
    asyncio.run(selector.ingest(current))

    assert [item.id for item in selector.select(current).events] == ["e2"]


def test_full_context_is_chronological_and_resets() -> None:
    selector = FullContextMemory()
    first = event("e1", 9, "First")
    second = event("e2", 10, "Second")
    asyncio.run(selector.ingest(first))
    asyncio.run(selector.ingest(second))
    assert [item.id for item in selector.select(second).events] == ["e1", "e2"]

    selector.reset()
    asyncio.run(selector.ingest(second))
    assert [item.id for item in selector.select(second).events] == ["e2"]


def test_vector_rag_returns_top_k_prior_events_plus_current() -> None:
    selector = VectorRAGMemory(KeywordVectorizer(), top_k=1)
    assignment = event("e1", 9, "Send the assignment")
    dentist = event("e2", 10, "Dentist tomorrow")
    current = event(
        "e3",
        11,
        "The assignment deadline is now",
        kind="clock_tick",
    )
    asyncio.run(selector.ingest(assignment))
    asyncio.run(selector.ingest(dentist))
    asyncio.run(selector.ingest(current))

    selection = selector.select(current)

    assert [item.id for item in selection.events] == ["e1", "e3"]
    assert selection.usage.embedding_inputs == 1
    assert selection.usage.embedding_characters > 0


def test_vector_rag_does_not_count_current_as_a_retrieved_memory() -> None:
    selector = VectorRAGMemory(KeywordVectorizer(), top_k=1)
    first = event("e1", 9, "Assignment one")
    current = event("e2", 10, "Assignment two")
    asyncio.run(selector.ingest(first))
    asyncio.run(selector.ingest(current))

    assert [item.id for item in selector.select(current).events] == ["e1", "e2"]


def test_fastembed_warmup_runs_once_outside_strategy_accounting() -> None:
    class FakeEmbeddingModel:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return iter(np.ones((len(texts), 3), dtype=np.float32))

    vectorizer = FastEmbedVectorizer()
    fake = FakeEmbeddingModel()
    vectorizer._model = fake

    assert vectorizer.warmup() >= 0
    assert vectorizer.warmup() == 0
    assert fake.calls == 1


def test_fastembed_requires_exact_revision_and_hashes_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="40-character"):
        FastEmbedVectorizer(revision="main")

    (tmp_path / "model.onnx").write_bytes(b"model")
    nested = tmp_path / "tokenizer"
    nested.mkdir()
    (nested / "config.json").write_text("{}", encoding="utf-8")

    first = _directory_sha256(tmp_path)
    second = _directory_sha256(tmp_path)

    assert first == second
    assert len(first) == 64
