"""Optional, non-authoritative recall through an OpenMemory-like client.

This module deliberately does not import or receive :class:`InMemoryAnamnesis`.
OpenMemory can provide retrospective text recall, but its identifiers, scores,
and lifecycle never become temporal-store state or Anamnesis action evidence.
Because upstream is in a breaking rewrite, callers must inject a client and
declare an explicit revision; this module never imports or constructs it
implicitly. A measured cell must independently attest that declaration.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import uuid
from collections.abc import Awaitable, Mapping
from typing import Any, ClassVar, Literal, Protocol, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

RecallMetadataValue = str | int | float | bool | None

_SCOPE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HANDLE_TOKEN = re.compile(r"^[0-9a-f]{32}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RESERVED_METADATA_PREFIX = "_anamnesis_"
_MAX_CONTENT_BYTES = 16 * 1024
_MAX_QUERY_BYTES = 4 * 1024
_MAX_TOTAL_RESULT_BYTES = 64 * 1024
_MAX_METADATA_BYTES = 16 * 1024
_MAX_METADATA_KEYS = 32
_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_LOCAL_EMBEDDING_PROVIDERS = frozenset(
    {"fastembed", "local", "ollama", "sentence-transformers", "synthetic"}
)


class RecallError(RuntimeError):
    """Base error for the optional recall boundary."""


class RecallDependencyError(RecallError):
    """The optional OpenMemory package is unavailable or incompatible."""


class RecallLocalityError(RecallError):
    """A client contradicts the requested local-only policy."""


class RecallProviderError(RecallError):
    """An OpenMemory client call failed before a result could be validated."""


class RecallProtocolError(RecallError):
    """An OpenMemory response failed the closed normalization contract."""


class _RecallResultModel(BaseModel):
    """Immutable output model that rejects drift and non-finite numbers."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
    )


class RecallHandle(_RecallResultModel):
    """Process-local opaque handle; it is never an OpenMemory identifier."""

    token: str = Field(pattern=_HANDLE_TOKEN.pattern)


class RecallDocument(_RecallResultModel):
    """Normalized text returned by ``get`` without provider identifiers."""

    content: str = Field(min_length=1)
    authoritative: Literal[False] = False
    evidence_event_ids: tuple[str, ...] = ()

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        return _bounded_nonblank(value, "recall content", _MAX_CONTENT_BYTES)

    @field_validator("evidence_event_ids")
    @classmethod
    def reject_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value:
            raise ValueError("retrospective recall cannot supply action evidence")
        return value


class RecallMatch(RecallDocument):
    """One bounded recall score and its non-authoritative text."""

    score: float = Field(ge=0.0, le=1.0)


class RecallSearchResult(_RecallResultModel):
    """Closed search result that cannot carry provider IDs or evidence."""

    matches: tuple[RecallMatch, ...] = ()
    authoritative: Literal[False] = False
    evidence_event_ids: tuple[str, ...] = ()

    @field_validator("evidence_event_ids")
    @classmethod
    def reject_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value:
            raise ValueError("retrospective recall cannot supply action evidence")
        return value


class RecallDeleteResult(_RecallResultModel):
    """Normalized deletion acknowledgement for one opaque handle."""

    deleted: bool


class OpenMemoryClientProtocol(Protocol):
    """Minimal main-branch surface; operations may be sync or async."""

    def add(
        self,
        content: str,
        *,
        user_id: str,
        metadata: Mapping[str, RecallMetadataValue],
    ) -> object | Awaitable[object]: ...

    def search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int,
    ) -> object | Awaitable[object]: ...

    def get(self, memory_id: str) -> object | Awaitable[object]: ...

    def delete(self, memory_id: str) -> object | Awaitable[object]: ...


class OpenMemoryMainClientAdapter:
    """Translation shim for one caller-attested upstream ``Memory`` object.

    The wrapper never constructs upstream ``Memory`` because its constructor
    opens the database immediately.  The caller owns that side effect and must
    supply the exact source revision plus local database/embedding policy.
    """

    is_local: Literal[True] = True
    mode: Literal["local"] = "local"

    def __init__(
        self,
        memory: object,
        *,
        upstream_revision: str,
        database_path: str,
        embedding_provider: str,
    ) -> None:
        if _COMMIT_SHA.fullmatch(upstream_revision) is None:
            raise ValueError("OpenMemory upstream_revision must be a 40-char SHA")
        path = urlsplit(database_path)
        if path.scheme or path.netloc or not database_path.startswith("/"):
            raise ValueError("OpenMemory database_path must be an absolute local path")
        normalized_provider = embedding_provider.casefold()
        if normalized_provider not in _LOCAL_EMBEDDING_PROVIDERS:
            raise RecallLocalityError("OpenMemory embedding provider is not local-only")
        _require_client_surface(memory)
        self._memory = memory
        self.upstream_revision = upstream_revision
        self.database_path = database_path
        self.embedding_provider = normalized_provider

    def add(
        self,
        content: str,
        *,
        user_id: str,
        metadata: Mapping[str, RecallMetadataValue],
    ) -> object | Awaitable[object]:
        return self._memory.add(content, user_id=user_id, meta=dict(metadata))

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int,
    ) -> object:
        raw = await _await_result(
            self._memory.search(query, user_id=user_id, limit=limit)
        )
        if not isinstance(raw, (list, tuple)):
            return raw
        if len(raw) > limit:
            raise RecallProtocolError("OpenMemory search returned more than limit")
        hydrated: list[object] = []
        for value in raw:
            hit = _as_mapping(value, "upstream search")
            provider_id = _provider_id(
                hit,
                operation="upstream search",
                required=True,
            )
            assert provider_id is not None
            row = _as_mapping(
                await _await_result(self._memory.get(provider_id)),
                "upstream get",
            )
            row_id = _provider_id(row, operation="upstream get", required=True)
            if row_id != provider_id:
                raise RecallProtocolError(
                    "OpenMemory upstream get returned a different record"
                )
            item = dict(hit)
            for content_key in ("content", "text", "memory"):
                item.pop(content_key, None)
            item["content"] = _content(row, operation="upstream get")
            item["user_id"] = row.get("user_id")
            if "metadata" in row:
                item["metadata"] = row["metadata"]
            elif "meta" in row:
                item["metadata"] = row["meta"]
            hydrated.append(item)
        return hydrated

    async def get(self, memory_id: str) -> object:
        raw = await _await_result(self._memory.get(memory_id))
        if raw is None:
            return None
        return dict(_as_mapping(raw, "upstream get"))

    def delete(self, memory_id: str) -> object | Awaitable[object]:
        return self._memory.delete(memory_id)


class RecallIndex(Protocol):
    """Non-authoritative retrospective-recall facade."""

    name: str
    authoritative: Literal[False]
    supports_action_evidence: Literal[False]
    mutates_anamnesis: Literal[False]

    async def add(
        self,
        content: str,
        *,
        metadata: Mapping[str, RecallMetadataValue] | None = None,
    ) -> RecallHandle: ...

    async def search(self, query: str, *, limit: int = 5) -> RecallSearchResult: ...

    async def get(self, handle: RecallHandle) -> RecallDocument: ...

    async def delete(self, handle: RecallHandle) -> RecallDeleteResult: ...


class OpenMemoryRecallIndex:
    """Isolated OpenMemory recall that never owns Anamnesis truth or evidence."""

    name: ClassVar[str] = "openmemory_recall"
    authoritative: Literal[False] = False
    supports_action_evidence: Literal[False] = False
    mutates_anamnesis: Literal[False] = False

    def __init__(
        self,
        *,
        namespace: str,
        user_id: str,
        client: OpenMemoryClientProtocol,
        local_only: bool = True,
    ) -> None:
        self.namespace = _scope_component(namespace, "namespace")
        self.user_id = _scope_component(user_id, "user_id")
        self.local_only = local_only
        self._client = client
        self._client_checked = False
        self._provider_ids: dict[str, str] = {}

    @property
    def scoped_user_id(self) -> str:
        """Provider partition key containing both required isolation dimensions."""

        return f"anamnesis::{self.namespace}::{self.user_id}"

    async def add(
        self,
        content: str,
        *,
        metadata: Mapping[str, RecallMetadataValue] | None = None,
    ) -> RecallHandle:
        normalized_content = _bounded_nonblank(
            content,
            "content",
            _MAX_CONTENT_BYTES,
        )
        normalized_metadata = _normalize_metadata(metadata)
        normalized_metadata.update(
            {
                "_anamnesis_namespace": self.namespace,
                "_anamnesis_user_id": self.user_id,
            }
        )
        client = self._get_client()
        try:
            raw = await _await_result(
                client.add(
                    normalized_content,
                    user_id=self.scoped_user_id,
                    metadata=normalized_metadata,
                )
            )
        except RecallError:
            raise
        except Exception as error:
            raise RecallProviderError("OpenMemory add failed") from error
        provider_id = _provider_id(raw, operation="add", required=True)
        assert provider_id is not None
        try:
            verification = await _await_result(client.get(provider_id))
        except RecallError:
            raise
        except Exception as error:
            raise RecallProviderError("OpenMemory add verification failed") from error
        _validate_provider_scope(
            verification,
            operation="add verification",
            provider_id=provider_id,
            scoped_user_id=self.scoped_user_id,
            namespace=self.namespace,
            user_id=self.user_id,
        )
        handle = RecallHandle(token=uuid.uuid4().hex)
        self._provider_ids[handle.token] = provider_id
        return handle

    async def search(self, query: str, *, limit: int = 5) -> RecallSearchResult:
        normalized_query = _bounded_nonblank(query, "query", _MAX_QUERY_BYTES)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("limit must be an integer between 1 and 100")
        client = self._get_client()
        try:
            raw = await _await_result(
                client.search(
                    normalized_query,
                    user_id=self.scoped_user_id,
                    limit=limit,
                )
            )
        except RecallError:
            raise
        except Exception as error:
            raise RecallProviderError("OpenMemory search failed") from error
        items = _search_items(raw)
        if len(items) > limit:
            raise RecallProtocolError("OpenMemory search returned more than limit")
        matches: list[RecallMatch] = []
        total_bytes = 0
        for item in items:
            _validate_provider_scope(
                item,
                operation="search",
                provider_id=None,
                scoped_user_id=self.scoped_user_id,
                namespace=self.namespace,
                user_id=self.user_id,
            )
            match = _recall_match(item)
            total_bytes += len(match.content.encode("utf-8"))
            if total_bytes > _MAX_TOTAL_RESULT_BYTES:
                raise RecallProtocolError("OpenMemory search content exceeds byte cap")
            matches.append(match)
        return RecallSearchResult(matches=tuple(matches))

    async def get(self, handle: RecallHandle) -> RecallDocument:
        provider_id = self._resolve_handle(handle)
        client = self._get_client()
        try:
            raw = await _await_result(client.get(provider_id))
        except RecallError:
            raise
        except Exception as error:
            raise RecallProviderError("OpenMemory get failed") from error
        item = _validate_provider_scope(
            raw,
            operation="get",
            provider_id=provider_id,
            scoped_user_id=self.scoped_user_id,
            namespace=self.namespace,
            user_id=self.user_id,
        )
        return RecallDocument(content=_content(item, operation="get"))

    async def delete(self, handle: RecallHandle) -> RecallDeleteResult:
        provider_id = self._resolve_handle(handle)
        client = self._get_client()
        try:
            raw = await _await_result(client.delete(provider_id))
        except RecallError:
            raise
        except Exception as error:
            raise RecallProviderError("OpenMemory delete failed") from error
        deleted = _deleted(raw)
        if deleted:
            try:
                remaining = await _await_result(client.get(provider_id))
            except RecallError:
                raise
            except Exception as error:
                raise RecallProviderError(
                    "OpenMemory delete verification failed"
                ) from error
            if remaining is not None:
                raise RecallProtocolError(
                    "OpenMemory delete acknowledged but record is still present"
                )
            del self._provider_ids[handle.token]
        return RecallDeleteResult(deleted=deleted)

    def _get_client(self) -> OpenMemoryClientProtocol:
        if not self._client_checked:
            _require_client_surface(self._client)
            if self.local_only:
                _guard_local_only(self._client)
            self._client_checked = True
        return self._client

    def _resolve_handle(self, handle: RecallHandle) -> str:
        if not isinstance(handle, RecallHandle):
            raise TypeError("get/delete require a RecallHandle")
        try:
            return self._provider_ids[handle.token]
        except KeyError as error:
            raise ValueError("unknown or expired recall handle") from error


def _scope_component(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _SCOPE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{field_name} must match {_SCOPE_COMPONENT.pattern}")
    return value


async def _await_result(value: object | Awaitable[object]) -> object:
    """Normalize the conflicting sync/async SDK surfaces without guessing APIs."""

    if inspect.isawaitable(value):
        return await value
    return value


def _nonblank(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank")
    return normalized


def _bounded_nonblank(value: str, field_name: str, max_bytes: int) -> str:
    normalized = _nonblank(value, field_name)
    if len(normalized.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_name} exceeds {max_bytes} UTF-8 bytes")
    return normalized


def _normalize_metadata(
    metadata: Mapping[str, RecallMetadataValue] | None,
) -> dict[str, RecallMetadataValue]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if len(metadata) > _MAX_METADATA_KEYS:
        raise ValueError(f"metadata cannot exceed {_MAX_METADATA_KEYS} keys")
    normalized: dict[str, RecallMetadataValue] = {}
    total_bytes = 0
    for key, value in metadata.items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError("metadata keys must be non-empty trimmed strings")
        if key.startswith(_RESERVED_METADATA_PREFIX):
            raise ValueError("metadata cannot override reserved Anamnesis keys")
        key_bytes = len(key.encode("utf-8"))
        if key_bytes > 128:
            raise ValueError("metadata key exceeds 128 UTF-8 bytes")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError("metadata values must be JSON scalar values")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("metadata numeric values must be finite")
        value_bytes = len(str(value).encode("utf-8"))
        if isinstance(value, str) and value_bytes > 4096:
            raise ValueError("metadata string value exceeds 4096 UTF-8 bytes")
        total_bytes += key_bytes + value_bytes
        if total_bytes > _MAX_METADATA_BYTES:
            raise ValueError("metadata exceeds the UTF-8 byte cap")
        normalized[key] = value
    return normalized


def _scope_metadata(item: Mapping[str, Any], operation: str) -> Mapping[str, Any]:
    keys = [key for key in ("metadata", "meta") if key in item]
    if len(keys) != 1:
        raise RecallProtocolError(
            f"OpenMemory {operation} must expose exactly one scope metadata field"
        )
    value = item[keys[0]]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise RecallProtocolError(
                f"OpenMemory {operation} scope metadata is invalid JSON"
            ) from error
    if not isinstance(value, Mapping):
        raise RecallProtocolError(
            f"OpenMemory {operation} scope metadata must be an object"
        )
    return cast(Mapping[str, Any], value)


def _validate_provider_scope(
    value: object,
    *,
    operation: str,
    provider_id: str | None,
    scoped_user_id: str,
    namespace: str,
    user_id: str,
) -> Mapping[str, Any]:
    item = _as_mapping(value, operation)
    returned_id = _provider_id(item, operation=operation, required=False)
    if provider_id is not None and returned_id != provider_id:
        raise RecallProtocolError(f"OpenMemory {operation} returned a different record")
    if item.get("user_id") != scoped_user_id:
        raise RecallProtocolError(f"OpenMemory {operation} crossed the user partition")
    metadata = _scope_metadata(item, operation)
    if (
        metadata.get("_anamnesis_namespace") != namespace
        or metadata.get("_anamnesis_user_id") != user_id
    ):
        raise RecallProtocolError(f"OpenMemory {operation} crossed the namespace")
    return item


def _require_client_surface(client: object) -> None:
    missing = [
        name
        for name in ("add", "search", "get", "delete")
        if not callable(getattr(client, name, None))
    ]
    if missing:
        raise RecallDependencyError(
            f"OpenMemory client lacks async operations: {', '.join(missing)}"
        )


def _guard_local_only(client: object) -> None:
    locality_verified = False
    config = _safe_attribute(client, "config")
    config_values = config if isinstance(config, Mapping) else {}

    for attribute in ("is_local", "local_only"):
        value = _safe_attribute(client, attribute)
        if value is None:
            value = config_values.get(attribute)
        if value is False:
            raise RecallLocalityError(
                f"OpenMemory client reports {attribute}=False in local-only mode"
            )
        if value is True:
            locality_verified = True

    mode = _safe_attribute(client, "mode")
    if mode is None:
        mode = config_values.get("mode")
    if isinstance(mode, str):
        normalized_mode = mode.strip().casefold()
        if normalized_mode == "remote":
            raise RecallLocalityError(
                "OpenMemory client reports remote mode in local-only mode"
            )
        if normalized_mode in {"embedded", "local", "offline"}:
            locality_verified = True

    for attribute in ("base_url", "endpoint", "url"):
        value = _safe_attribute(client, attribute)
        if value is None:
            value = config_values.get(attribute)
        if isinstance(value, str):
            if not _is_local_endpoint(value):
                raise RecallLocalityError(
                    f"OpenMemory {attribute} is not a loopback or "
                    "local database endpoint"
                )
            locality_verified = True

    provider = _safe_attribute(client, "embedding_provider")
    if provider is None:
        provider = config_values.get("embedding_provider")
    if provider is None:
        provider = config_values.get("embeddings")
    if isinstance(provider, str) and provider.casefold() not in (
        _LOCAL_EMBEDDING_PROVIDERS
    ):
        raise RecallLocalityError("OpenMemory embedding provider is not local-only")
    if isinstance(provider, str) and provider.casefold() in _LOCAL_EMBEDDING_PROVIDERS:
        locality_verified = True

    if not locality_verified:
        raise RecallLocalityError(
            "OpenMemory client exposes no positive evidence of local-only operation"
        )


def _safe_attribute(value: object, name: str) -> object | None:
    try:
        return getattr(value, name, None)
    except Exception as error:
        raise RecallLocalityError(
            f"could not inspect OpenMemory locality attribute {name}"
        ) from error


def _is_local_endpoint(value: str) -> bool:
    parsed = urlsplit(value)
    if parsed.scheme in {"file", "sqlite"}:
        return not parsed.netloc and not parsed.query and not parsed.fragment
    if not parsed.scheme:
        if value in {":memory:", "localhost", "127.0.0.1", "::1"}:
            return True
        return value.startswith(("/", "./", "../"))
    return parsed.scheme in {"http", "https"} and parsed.hostname in _LOCAL_HOSTS


def _model_dump(value: object) -> object:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="python")
        except TypeError:
            return dump()
    keys = getattr(value, "keys", None)
    if callable(keys) and hasattr(value, "__getitem__"):
        try:
            return {key: value[key] for key in keys()}
        except (KeyError, TypeError, ValueError):
            return value
    return value


def _as_mapping(value: object, operation: str) -> Mapping[str, Any]:
    normalized = _model_dump(value)
    if not isinstance(normalized, Mapping):
        raise RecallProtocolError(f"OpenMemory {operation} returned a non-object")
    return cast(Mapping[str, Any], normalized)


def _provider_id(
    value: object,
    *,
    operation: str,
    required: bool,
) -> str | None:
    if isinstance(value, str):
        candidate: object | None = value
    else:
        item = _as_mapping(value, operation)
        candidates = [item[key] for key in ("id", "memory_id") if key in item]
        if len(candidates) > 1 and len(set(map(str, candidates))) != 1:
            raise RecallProtocolError(
                f"OpenMemory {operation} returned conflicting identifiers"
            )
        candidate = candidates[0] if candidates else None
    if candidate is None:
        if required:
            raise RecallProtocolError(
                f"OpenMemory {operation} omitted its record identifier"
            )
        return None
    if not isinstance(candidate, str) or not candidate.strip():
        raise RecallProtocolError(
            f"OpenMemory {operation} returned an invalid record identifier"
        )
    return candidate.strip()


def _search_items(value: object) -> list[object]:
    normalized = _model_dump(value)
    if isinstance(normalized, (list, tuple)):
        return list(normalized)
    if not isinstance(normalized, Mapping):
        raise RecallProtocolError("OpenMemory search returned an invalid envelope")
    container_keys = [
        key for key in ("results", "matches", "memories") if key in normalized
    ]
    if len(container_keys) != 1:
        raise RecallProtocolError(
            "OpenMemory search must return exactly one recognized result collection"
        )
    items = normalized[container_keys[0]]
    if not isinstance(items, (list, tuple)):
        raise RecallProtocolError("OpenMemory search result collection is not a list")
    return list(items)


def _content(item: Mapping[str, Any], *, operation: str) -> str:
    keys = [key for key in ("content", "text", "memory") if key in item]
    if len(keys) != 1:
        raise RecallProtocolError(
            f"OpenMemory {operation} must return exactly one content field"
        )
    value = item[keys[0]]
    if not isinstance(value, str):
        raise RecallProtocolError(f"OpenMemory {operation} returned non-string content")
    try:
        return _nonblank(value, "content")
    except (TypeError, ValueError) as error:
        raise RecallProtocolError(
            f"OpenMemory {operation} returned blank content"
        ) from error


def _recall_match(value: object) -> RecallMatch:
    item = _as_mapping(value, "search")
    _provider_id(item, operation="search", required=False)
    score_keys = [key for key in ("score", "similarity") if key in item]
    if len(score_keys) != 1:
        raise RecallProtocolError(
            "OpenMemory search result must return exactly one score field"
        )
    score = item[score_keys[0]]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RecallProtocolError("OpenMemory search score must be numeric")
    if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
        raise RecallProtocolError("OpenMemory search score must be finite in [0, 1]")
    try:
        return RecallMatch(
            content=_content(item, operation="search"),
            score=float(score),
        )
    except ValidationError as error:
        raise RecallProtocolError("OpenMemory search result is invalid") from error


def _deleted(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    item = _as_mapping(value, "delete")
    deleted = item.get("deleted")
    if not isinstance(deleted, bool):
        raise RecallProtocolError(
            "OpenMemory delete must return a boolean deleted field"
        )
    return deleted


__all__ = [
    "OpenMemoryClientProtocol",
    "OpenMemoryMainClientAdapter",
    "OpenMemoryRecallIndex",
    "RecallDeleteResult",
    "RecallDependencyError",
    "RecallDocument",
    "RecallError",
    "RecallHandle",
    "RecallIndex",
    "RecallLocalityError",
    "RecallMatch",
    "RecallMetadataValue",
    "RecallProtocolError",
    "RecallProviderError",
    "RecallSearchResult",
]
