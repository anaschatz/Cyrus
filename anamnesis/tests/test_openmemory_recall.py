from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping

import pytest

from anamnesis.memory import InMemoryAnamnesis
from anamnesis.openmemory_recall import (
    OpenMemoryMainClientAdapter,
    OpenMemoryRecallIndex,
    RecallHandle,
    RecallLocalityError,
    RecallProtocolError,
    RecallProviderError,
)

_NAMESPACE = "assistant-a"
_USER_ID = "user-1"
_SCOPED_USER_ID = f"anamnesis::{_NAMESPACE}::{_USER_ID}"
_DEFAULT = object()


def _scope_metadata(
    namespace: str = _NAMESPACE,
    user_id: str = _USER_ID,
) -> dict[str, str]:
    return {
        "_anamnesis_namespace": namespace,
        "_anamnesis_user_id": user_id,
    }


def _valid_row(
    *,
    namespace: str = _NAMESPACE,
    user_id: str = _USER_ID,
    provider_id: str = "om-1",
    content: str = "stored memory",
    score: object = 0.875,
    metadata_key: str = "metadata",
) -> dict[str, object]:
    return {
        "id": provider_id,
        "content": content,
        "score": score,
        "user_id": f"anamnesis::{namespace}::{user_id}",
        metadata_key: _scope_metadata(namespace, user_id),
    }


def _bad_scope_rows() -> list[pytest.ParameterSet]:
    missing_user = _valid_row()
    del missing_user["user_id"]

    wrong_user = _valid_row()
    wrong_user["user_id"] = "anamnesis::assistant-a::someone-else"

    missing_namespace = _valid_row()
    missing_namespace["metadata"] = {"_anamnesis_user_id": _USER_ID}

    missing_user_metadata = _valid_row()
    missing_user_metadata["metadata"] = {"_anamnesis_namespace": _NAMESPACE}

    missing_metadata = _valid_row()
    del missing_metadata["metadata"]

    return [
        pytest.param(missing_user, "user partition", id="missing-user-id"),
        pytest.param(wrong_user, "user partition", id="wrong-user-id"),
        pytest.param(
            missing_namespace,
            "crossed the namespace",
            id="missing-reserved-namespace",
        ),
        pytest.param(
            missing_user_metadata,
            "crossed the namespace",
            id="missing-reserved-user",
        ),
        pytest.param(missing_metadata, "scope metadata", id="missing-metadata"),
    ]


class FakeOpenMemoryClient:
    """Strict direct-client fake whose get/delete reject a user_id keyword."""

    is_local = True
    mode = "local"
    embedding_provider = "synthetic"

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.add_response: object = {"id": "om-1"}
        self.search_response: object = []
        self.get_response: object = _DEFAULT
        self.delete_response: object = None
        self.error_operation: str | None = None
        self.stored_row: object = _valid_row()

    async def add(
        self,
        content: str,
        *,
        user_id: str,
        metadata: Mapping[str, object],
    ) -> object:
        self._maybe_fail("add")
        normalized_metadata = dict(metadata)
        self.calls.append(("add", content, user_id, normalized_metadata))
        self.stored_row = {
            "id": "om-1",
            "content": content,
            "user_id": user_id,
            "metadata": normalized_metadata,
        }
        return self.add_response

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int,
    ) -> object:
        self._maybe_fail("search")
        self.calls.append(("search", query, user_id, limit))
        return self.search_response

    async def get(self, memory_id: str) -> object:
        self._maybe_fail("get")
        self.calls.append(("get", memory_id))
        if self.get_response is _DEFAULT:
            return self.stored_row
        return self.get_response

    async def delete(self, memory_id: str) -> object:
        self._maybe_fail("delete")
        self.calls.append(("delete", memory_id))
        if (
            self.delete_response is None
            or self.delete_response is True
            or (
                isinstance(self.delete_response, Mapping)
                and self.delete_response.get("deleted") is True
            )
        ):
            self.stored_row = None
        return self.delete_response

    def _maybe_fail(self, operation: str) -> None:
        if self.error_operation == operation:
            raise RuntimeError("provider detail must stay behind the recall boundary")


class ModelDumpRow:
    """Object-shaped response normalized through ``model_dump``."""

    def __init__(self, values: Mapping[str, object]) -> None:
        self.values = dict(values)

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        assert mode == "python"
        return dict(self.values)


class SqliteRowLike:
    """Non-Mapping row exposing only sqlite.Row's keys/item access surface."""

    def __init__(self, values: Mapping[str, object]) -> None:
        self._values = dict(values)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._values)

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def as_dict(self) -> dict[str, object]:
        return dict(self._values)


class FakeUpstreamMainMemory:
    """Synchronous main-branch surface using ``meta`` instead of ``metadata``."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.row: SqliteRowLike | None = SqliteRowLike(_valid_row(metadata_key="meta"))

    def add(
        self,
        content: str,
        *,
        user_id: str,
        meta: Mapping[str, object],
    ) -> SqliteRowLike:
        normalized_meta = dict(meta)
        self.calls.append(("add", content, user_id, normalized_meta))
        self.row = SqliteRowLike(
            {
                "id": "om-row-1",
                "content": content,
                "score": 0.75,
                "user_id": user_id,
                "meta": normalized_meta,
            }
        )
        return SqliteRowLike({"id": "om-row-1"})

    def search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int,
    ) -> list[SqliteRowLike]:
        self.calls.append(("search", query, user_id, limit))
        assert self.row is not None
        values = self.row.as_dict()
        return [
            SqliteRowLike(
                {
                    "id": values["id"],
                    "content": values["content"],
                    "score": values["score"],
                }
            )
        ]

    def get(self, memory_id: str) -> SqliteRowLike | None:
        self.calls.append(("get", memory_id))
        return self.row

    def delete(self, memory_id: str) -> None:
        self.calls.append(("delete", memory_id))
        self.row = None


def test_client_is_required_explicitly() -> None:
    with pytest.raises(TypeError, match="client"):
        OpenMemoryRecallIndex(namespace="ns", user_id="user")  # type: ignore[call-arg]


def test_crud_verifies_add_scope_and_uses_opaque_process_handle() -> None:
    client = FakeOpenMemoryClient()
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    handle = asyncio.run(index.add("  remember blue  ", metadata={"kind": "fact"}))
    document = asyncio.run(index.get(handle))
    deleted = asyncio.run(index.delete(handle))

    assert isinstance(handle, RecallHandle)
    assert handle.token != "om-1"
    assert document.content == "remember blue"
    assert document.authoritative is False
    assert document.evidence_event_ids == ()
    assert deleted.deleted is True
    assert client.calls == [
        (
            "add",
            "remember blue",
            _SCOPED_USER_ID,
            {
                "kind": "fact",
                "_anamnesis_namespace": _NAMESPACE,
                "_anamnesis_user_id": _USER_ID,
            },
        ),
        ("get", "om-1"),
        ("get", "om-1"),
        ("delete", "om-1"),
        ("get", "om-1"),
    ]
    with pytest.raises(ValueError, match="unknown or expired"):
        asyncio.run(index.get(handle))


def test_delete_false_keeps_verified_handle_live() -> None:
    client = FakeOpenMemoryClient()
    client.delete_response = False
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )
    handle = asyncio.run(index.add("memory"))

    result = asyncio.run(index.delete(handle))

    assert result.deleted is False
    assert asyncio.run(index.get(handle)).content == "memory"


def test_delete_acknowledgement_fails_if_record_is_still_present() -> None:
    class LyingDeleteClient(FakeOpenMemoryClient):
        async def delete(self, memory_id: str) -> object:
            self.calls.append(("delete", memory_id))
            return {"deleted": True}

    client = LyingDeleteClient()
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )
    handle = asyncio.run(index.add("memory"))

    with pytest.raises(RecallProtocolError, match="still present"):
        asyncio.run(index.delete(handle))

    assert asyncio.run(index.get(handle)).content == "memory"


@pytest.mark.parametrize(("response", "message"), _bad_scope_rows())
def test_add_fails_closed_when_get_verification_cannot_prove_scope(
    response: dict[str, object],
    message: str,
) -> None:
    client = FakeOpenMemoryClient()
    client.get_response = response
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    with pytest.raises(RecallProtocolError, match=message):
        asyncio.run(index.add("memory"))

    assert [call[0] for call in client.calls] == ["add", "get"]


def test_add_verification_rejects_a_different_provider_record() -> None:
    client = FakeOpenMemoryClient()
    client.get_response = _valid_row(provider_id="om-other")
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    with pytest.raises(RecallProtocolError, match="different record"):
        asyncio.run(index.add("memory"))

    assert [call[0] for call in client.calls] == ["add", "get"]


def test_add_verification_requires_the_exact_returned_provider_id() -> None:
    client = FakeOpenMemoryClient()
    client.get_response = _valid_row()
    del client.get_response["id"]  # type: ignore[index]
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    with pytest.raises(RecallProtocolError, match="different record"):
        asyncio.run(index.add("memory"))

    assert [call[0] for call in client.calls] == ["add", "get"]


@pytest.mark.parametrize(("response", "message"), _bad_scope_rows())
def test_get_requires_scoped_user_and_both_reserved_metadata_fields(
    response: dict[str, object],
    message: str,
) -> None:
    client = FakeOpenMemoryClient()
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )
    handle = asyncio.run(index.add("memory"))
    client.get_response = response

    with pytest.raises(RecallProtocolError, match=message):
        asyncio.run(index.get(handle))


@pytest.mark.parametrize(("response", "message"), _bad_scope_rows())
def test_each_search_row_requires_user_and_reserved_metadata(
    response: dict[str, object],
    message: str,
) -> None:
    client = FakeOpenMemoryClient()
    client.search_response = [response]
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    with pytest.raises(RecallProtocolError, match=message):
        asyncio.run(index.search("query"))


def test_search_normalizes_without_provider_ids_metadata_or_evidence() -> None:
    client = FakeOpenMemoryClient()
    row = _valid_row(content="  user prefers dark mode  ")
    metadata = dict(_scope_metadata())
    metadata["provider_internal"] = "must not escape"
    row["metadata"] = metadata
    row["id"] = "openmemory-secret-id"
    client.search_response = {"results": [row], "total": 1}
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    result = asyncio.run(index.search("theme", limit=3))
    dumped = result.model_dump(mode="json")

    assert result.authoritative is False
    assert result.evidence_event_ids == ()
    assert len(result.matches) == 1
    assert result.matches[0].content == "user prefers dark mode"
    assert result.matches[0].score == pytest.approx(0.875)
    assert result.matches[0].authoritative is False
    assert result.matches[0].evidence_event_ids == ()
    assert "openmemory-secret-id" not in str(dumped)
    assert "provider_internal" not in str(dumped)


def test_model_dump_response_objects_are_normalized() -> None:
    client = FakeOpenMemoryClient()
    client.search_response = [ModelDumpRow(_valid_row(content="model row"))]
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    result = asyncio.run(index.search("row"))

    assert result.matches[0].content == "model row"


def test_namespace_and_user_form_the_provider_partition() -> None:
    client = FakeOpenMemoryClient()
    personal = OpenMemoryRecallIndex(
        namespace="personal",
        user_id="same-user",
        client=client,
    )
    work = OpenMemoryRecallIndex(
        namespace="work",
        user_id="same-user",
        client=client,
    )

    client.search_response = [_valid_row(namespace="personal", user_id="same-user")]
    asyncio.run(personal.search("preferences"))
    client.search_response = [_valid_row(namespace="work", user_id="same-user")]
    asyncio.run(work.search("preferences"))

    assert client.calls == [
        ("search", "preferences", "anamnesis::personal::same-user", 5),
        ("search", "preferences", "anamnesis::work::same-user", 5),
    ]


@pytest.mark.parametrize("score", [-0.1, 1.1, math.nan, math.inf, True, "0.8"])
def test_search_rejects_invalid_scores(score: object) -> None:
    client = FakeOpenMemoryClient()
    client.search_response = [_valid_row(score=score)]
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    with pytest.raises(RecallProtocolError, match="score"):
        asyncio.run(index.search("query"))


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"results": "not-a-list"},
        {"results": [], "matches": []},
        [{"content": "memory", "score": 0.5}],
        [_valid_row()] * 2,
    ],
)
def test_malformed_or_excess_search_output_fails_closed(response: object) -> None:
    client = FakeOpenMemoryClient()
    client.search_response = response
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )
    limit = 1 if isinstance(response, list) and len(response) == 2 else 5

    with pytest.raises(RecallProtocolError):
        asyncio.run(index.search("query", limit=limit))


def test_provider_errors_are_wrapped_without_partial_results() -> None:
    client = FakeOpenMemoryClient()
    client.error_operation = "search"
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    with pytest.raises(RecallProviderError, match="search failed") as captured:
        asyncio.run(index.search("query"))

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_local_only_rejects_file_url_with_authority_before_provider_call() -> None:
    client = FakeOpenMemoryClient()
    client.url = "file://remote-host/private/memory.sqlite"
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    with pytest.raises(RecallLocalityError, match="not a loopback"):
        asyncio.run(index.search("query"))

    assert client.calls == []


def test_local_only_requires_positive_evidence_but_has_explicit_opt_out() -> None:
    class UnverifiableClient(FakeOpenMemoryClient):
        is_local = None
        mode = None
        embedding_provider = None

    client = UnverifiableClient()
    guarded = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    with pytest.raises(RecallLocalityError, match="no positive evidence"):
        asyncio.run(guarded.search("query"))
    assert client.calls == []

    opted_out = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
        local_only=False,
    )
    assert asyncio.run(opted_out.search("query")).matches == ()


def test_main_adapter_maps_metadata_to_meta_and_normalizes_row_objects() -> None:
    upstream = FakeUpstreamMainMemory()
    adapter = OpenMemoryMainClientAdapter(
        upstream,
        upstream_revision="a" * 40,
        database_path="/private/tmp/openmemory.sqlite",
        embedding_provider="synthetic",
    )
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=adapter,
    )

    handle = asyncio.run(index.add("remember blue", metadata={"kind": "fact"}))
    search = asyncio.run(index.search("blue", limit=3))
    document = asyncio.run(index.get(handle))
    deleted = asyncio.run(index.delete(handle))

    scoped_meta = {
        "kind": "fact",
        "_anamnesis_namespace": _NAMESPACE,
        "_anamnesis_user_id": _USER_ID,
    }
    assert upstream.calls == [
        ("add", "remember blue", _SCOPED_USER_ID, scoped_meta),
        ("get", "om-row-1"),
        ("search", "blue", _SCOPED_USER_ID, 3),
        ("get", "om-row-1"),
        ("get", "om-row-1"),
        ("delete", "om-row-1"),
        ("get", "om-row-1"),
    ]
    assert search.matches[0].content == "remember blue"
    assert search.matches[0].score == pytest.approx(0.75)
    assert document.content == "remember blue"
    assert deleted.deleted is True
    assert adapter.upstream_revision == "a" * 40
    assert adapter.database_path == "/private/tmp/openmemory.sqlite"


def test_main_adapter_rejects_hydrated_search_id_mismatch() -> None:
    class MismatchedGetMemory(FakeUpstreamMainMemory):
        def search(
            self,
            query: str,
            *,
            user_id: str,
            limit: int,
        ) -> list[SqliteRowLike]:
            self.calls.append(("search", query, user_id, limit))
            return [SqliteRowLike({"id": "om-hit", "content": "memory", "score": 0.5})]

    upstream = MismatchedGetMemory()
    adapter = OpenMemoryMainClientAdapter(
        upstream,
        upstream_revision="a" * 40,
        database_path="/private/tmp/openmemory.sqlite",
        embedding_provider="synthetic",
    )

    with pytest.raises(RecallProtocolError, match="different record"):
        asyncio.run(adapter.search("memory", user_id=_SCOPED_USER_ID, limit=3))

    assert upstream.calls == [
        ("search", "memory", _SCOPED_USER_ID, 3),
        ("get", "om-hit"),
    ]


def test_main_adapter_binds_search_content_to_the_hydrated_row() -> None:
    class ForgedSearchContentMemory(FakeUpstreamMainMemory):
        def search(
            self,
            query: str,
            *,
            user_id: str,
            limit: int,
        ) -> list[SqliteRowLike]:
            self.calls.append(("search", query, user_id, limit))
            return [
                SqliteRowLike(
                    {
                        "id": "om-1",
                        "content": "forged cross-record content",
                        "score": 0.5,
                    }
                )
            ]

    upstream = ForgedSearchContentMemory()
    adapter = OpenMemoryMainClientAdapter(
        upstream,
        upstream_revision="a" * 40,
        database_path="/private/tmp/openmemory.sqlite",
        embedding_provider="synthetic",
    )
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=adapter,
    )

    result = asyncio.run(index.search("memory", limit=3))

    assert result.matches[0].content == "stored memory"


def test_main_adapter_rejects_over_limit_before_hydrating_hits() -> None:
    class OverLimitMemory(FakeUpstreamMainMemory):
        def search(
            self,
            query: str,
            *,
            user_id: str,
            limit: int,
        ) -> list[SqliteRowLike]:
            self.calls.append(("search", query, user_id, limit))
            return [
                SqliteRowLike(
                    {
                        "id": f"om-hit-{number}",
                        "content": "memory",
                        "score": 0.5,
                    }
                )
                for number in range(limit + 1)
            ]

    upstream = OverLimitMemory()
    adapter = OpenMemoryMainClientAdapter(
        upstream,
        upstream_revision="a" * 40,
        database_path="/private/tmp/openmemory.sqlite",
        embedding_provider="synthetic",
    )

    with pytest.raises(RecallProtocolError, match="more than limit"):
        asyncio.run(adapter.search("memory", user_id=_SCOPED_USER_ID, limit=2))

    assert upstream.calls == [("search", "memory", _SCOPED_USER_ID, 2)]


def test_main_adapter_rejects_unpinned_or_nonlocal_configuration() -> None:
    upstream = FakeUpstreamMainMemory()

    with pytest.raises(ValueError, match="40-char SHA"):
        OpenMemoryMainClientAdapter(
            upstream,
            upstream_revision="main",
            database_path="/private/tmp/openmemory.sqlite",
            embedding_provider="synthetic",
        )
    with pytest.raises(ValueError, match="absolute local path"):
        OpenMemoryMainClientAdapter(
            upstream,
            upstream_revision="a" * 40,
            database_path="file://remote-host/private/memory.sqlite",
            embedding_provider="synthetic",
        )
    with pytest.raises(RecallLocalityError, match="not local-only"):
        OpenMemoryMainClientAdapter(
            upstream,
            upstream_revision="a" * 40,
            database_path="/private/tmp/openmemory.sqlite",
            embedding_provider="openai",
        )


def test_utf8_content_and_query_caps_count_bytes_not_characters() -> None:
    client = FakeOpenMemoryClient()
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )
    exact_content = "é" * (16 * 1024 // 2)
    exact_query = "é" * (4 * 1024 // 2)

    asyncio.run(index.add(exact_content))
    calls_after_exact_content = len(client.calls)
    with pytest.raises(ValueError, match="16384 UTF-8 bytes"):
        asyncio.run(index.add(exact_content + "é"))
    assert len(client.calls) == calls_after_exact_content

    asyncio.run(index.search(exact_query))
    calls_after_exact_query = len(client.calls)
    with pytest.raises(ValueError, match="4096 UTF-8 bytes"):
        asyncio.run(index.search(exact_query + "é"))
    assert len(client.calls) == calls_after_exact_query


def test_utf8_metadata_value_and_key_caps_are_enforced_before_add() -> None:
    client = FakeOpenMemoryClient()
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )

    asyncio.run(index.add("memory", metadata={"note": "é" * (4096 // 2)}))
    calls_after_exact_value = len(client.calls)
    with pytest.raises(ValueError, match="4096 UTF-8 bytes"):
        asyncio.run(index.add("memory", metadata={"note": "é" * (4096 // 2 + 1)}))
    assert len(client.calls) == calls_after_exact_value

    asyncio.run(index.add("memory", metadata={"é" * (128 // 2): "value"}))
    calls_after_exact_key = len(client.calls)
    with pytest.raises(ValueError, match="128 UTF-8 bytes"):
        asyncio.run(index.add("memory", metadata={"é" * (128 // 2 + 1): "value"}))
    assert len(client.calls) == calls_after_exact_key


def test_search_enforces_total_utf8_result_cap() -> None:
    client = FakeOpenMemoryClient()
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )
    max_row_content = "é" * (16 * 1024 // 2)
    four_rows = [
        _valid_row(provider_id=f"om-{number}", content=max_row_content)
        for number in range(4)
    ]
    client.search_response = four_rows

    result = asyncio.run(index.search("query", limit=5))
    assert len(result.matches) == 4

    client.search_response = [
        _valid_row(provider_id=f"om-{number}", content=max_row_content)
        for number in range(5)
    ]
    with pytest.raises(RecallProtocolError, match="byte cap"):
        asyncio.run(index.search("query", limit=5))


def test_recall_remains_explicitly_non_authoritative_and_store_independent() -> None:
    client = FakeOpenMemoryClient()
    client.search_response = [_valid_row(content="retrospective context", score=1)]
    index = OpenMemoryRecallIndex(
        namespace=_NAMESPACE,
        user_id=_USER_ID,
        client=client,
    )
    store = InMemoryAnamnesis()
    before = store.state_hash()

    result = asyncio.run(index.search("context"))

    assert index.authoritative is False
    assert index.supports_action_evidence is False
    assert index.mutates_anamnesis is False
    assert result.matches[0].evidence_event_ids == ()
    assert store.state_hash() == before
    assert store.events == ()
    assert store.current_facts == ()
    assert store.current_intents == ()
