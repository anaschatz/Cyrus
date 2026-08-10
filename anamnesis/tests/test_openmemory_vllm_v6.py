import asyncio
import hashlib
import json

import numpy as np

from anamnesis.openmemory_recall import OpenMemoryRecallIndex
from anamnesis.openmemory_vllm_v6 import (
    FIXTURE_PATH,
    PIN_PATH,
    LocalVectorMemoryClient,
    RealMemoryFixture,
    _correct,
    _load_inputs,
)
from anamnesis.schema import Decision, ProposedAction


def test_v6_frozen_inputs_and_matrix() -> None:
    pin, fixture, runtime = _load_inputs()
    assert (
        hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == pin.fixture_raw_sha256
    )
    assert len(fixture.cases) == 8
    assert sum(case.helpful_opportunity for case in fixture.cases) == 4
    assert runtime.served_model == "anamnesis-openmemory-v6"
    assert runtime.max_tokens == 256
    assert json.loads(PIN_PATH.read_text())["top_k"] == 1


class _Vectorizer:
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if "target" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_real_index_add_search_and_scope() -> None:
    async def run() -> None:
        client = LocalVectorMemoryClient(_Vectorizer())  # type: ignore[arg-type]
        index = OpenMemoryRecallIndex(namespace="test", user_id="user", client=client)
        await index.add("irrelevant", metadata={"fixture_id": "memory_b"})
        await index.add("target memory", metadata={"fixture_id": "memory_a"})
        result = await index.search("query", limit=1)
        assert [match.content for match in result.matches] == ["target memory"]
        assert client.last_search_ids == ("memory_a",)
        row = client.rows["memory_a"]
        assert row["user_id"] == "anamnesis::test::user"

    asyncio.run(run())


def test_v6_exact_scoring_rejects_missing_memory_slot() -> None:
    fixture = RealMemoryFixture.model_validate_json(FIXTURE_PATH.read_text())
    case = fixture.cases[0]
    incomplete = Decision(
        actions=[
            ProposedAction(
                kind="reminder",
                action_key=case.event.id,
                payload={"subject": "send observatory calibration packet"},
                summary="Send packet",
                evidence_event_ids=[case.event.id],
            )
        ]
    )
    exact = Decision(
        actions=[
            ProposedAction(
                kind="reminder",
                action_key=case.event.id,
                payload=case.expected.payload,
                summary="Send packet",
                evidence_event_ids=[case.event.id],
            )
        ]
    )
    assert not _correct(case.expected, incomplete)
    assert _correct(case.expected, exact)


def test_v6_fixture_has_no_prior_entities() -> None:
    text = FIXTURE_PATH.read_text().casefold()
    for forbidden in ("meridian archives", "silver gallery", "cobalt inspection"):
        assert forbidden not in text
