"""Budgeted local-model dispatch with a fail-closed one-shot request guard."""

from __future__ import annotations

import asyncio
import contextvars
import json
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from .domain import ModelChunk, ModelRequest, ModelUsage, OrchestratorError, canonical_json
from .ports import ModelPort
from .store import OrchestratorStore


@dataclass(slots=True)
class DispatchPermit:
    reservation_id: str
    consumed: bool = False


_dispatch_reservation: contextvars.ContextVar[DispatchPermit | None] = contextvars.ContextVar(
    "jarvis_model_dispatch_reservation", default=None
)


def active_dispatch_reservation() -> str | None:
    permit = _dispatch_reservation.get()
    return permit.reservation_id if permit is not None else None


def consume_dispatch_reservation() -> str:
    permit = _dispatch_reservation.get()
    if permit is None or permit.consumed:
        raise OrchestratorError("model_dispatch_without_reservation", status=500)
    permit.consumed = True
    return permit.reservation_id


try:  # Imported only in the pinned production environment.
    import httpx
except ImportError:  # pragma: no cover - dependency-free core
    httpx = None  # type: ignore[assignment]


if httpx is not None:
    class GuardedAsyncTransport(httpx.AsyncBaseTransport):
        """One physical Chat Completions dispatch per persisted permit."""

        def __init__(self, wrapped: httpx.AsyncBaseTransport | None = None) -> None:
            self._wrapped = wrapped or httpx.AsyncHTTPTransport(retries=0)

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.method != "POST" or request.url.path != "/v1/chat/completions":
                raise OrchestratorError("model_transport_endpoint_forbidden", status=500)
            consume_dispatch_reservation()
            return await self._wrapped.handle_async_request(request)

        async def aclose(self) -> None:
            await self._wrapped.aclose()
else:  # pragma: no cover
    class GuardedAsyncTransport:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("httpx from pydantic-ai-slim[openai]==2.27.0 is required")


class BudgetedModelPort:
    """Authoritative reserve -> dispatch -> reconcile wrapper.

    The delegate is private by convention and receives a one-shot reservation
    through a task-local context variable. Local HTTP adapters fail closed when
    invoked without it.
    """

    def __init__(self, store: OrchestratorStore, delegate: ModelPort) -> None:
        self._store = store
        self.__delegate = delegate

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        reservation_id = self._store.reserve_model(
            request.task_id,
            request.epoch,
            purpose=request.purpose,
            output_allowance=request.output_allowance,
        )
        self._store.mark_model_dispatched(reservation_id, request.task_id, request.epoch)
        guarded_request = replace(request, output_allowance=request.output_allowance)
        token = _dispatch_reservation.set(DispatchPermit(reservation_id))
        usage: ModelUsage | None = None
        try:
            async for chunk in self.__delegate.stream(guarded_request):
                if chunk.usage is not None:
                    usage = chunk.usage
                yield chunk
        except BaseException:
            # A concurrent cancellation may already have atomically cancelled
            # this reservation. Preserve the original exception in that race.
            with suppress(Exception):
                self._store.reconcile_model(reservation_id, request.task_id, request.epoch, None)
            raise
        else:
            self._store.reconcile_model(
                reservation_id,
                request.task_id,
                request.epoch,
                usage.output_tokens if usage is not None else None,
            )
        finally:
            _dispatch_reservation.reset(token)


class LocalInferenceModelPort:
    """Strict OpenAI Chat Completions adapter for the loopback appliance."""

    FORBIDDEN_FIELDS = frozenset({"seed", "user", "metadata", "store", "service_tier", "max_tokens"})

    def __init__(self, *, base_url: str, bearer_token: str, model_alias: str = "jarvis-local", timeout_seconds: float = 120.0) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._token = bearer_token
        self._model_alias = model_alias
        self._timeout = timeout_seconds

    @staticmethod
    def validate_payload(payload: Mapping[str, Any], *, reservation_allowance: int) -> None:
        if set(payload) & LocalInferenceModelPort.FORBIDDEN_FIELDS:
            raise OrchestratorError("model_payload_forbidden_field", status=500)
        if payload.get("model") != "jarvis-local":
            raise OrchestratorError("model_payload_alias", status=500)
        if payload.get("max_completion_tokens") != reservation_allowance:
            raise OrchestratorError("model_payload_budget_mismatch", status=500)
        messages = payload.get("messages")
        if not isinstance(messages, list) or any(item.get("role") == "developer" for item in messages if isinstance(item, dict)):
            raise OrchestratorError("model_payload_role", status=500)
        tools = payload.get("tools")
        response_format = payload.get("response_format")
        structured = isinstance(response_format, dict) and response_format.get("type") in {"json_schema", "json_object"}
        if tools and structured:
            raise OrchestratorError("model_payload_tools_xor_schema", status=500)
        if tools:
            if payload.get("parallel_tool_calls") is not False:
                raise OrchestratorError("model_payload_parallel_tools", status=500)
            for tool in tools:
                if tool.get("function", {}).get("strict") is not True:
                    raise OrchestratorError("model_payload_non_strict_tool", status=500)
        if structured and response_format.get("json_schema", {}).get("strict") is not True:
            raise OrchestratorError("model_payload_non_strict_schema", status=500)
        options = payload.get("stream_options")
        if options is not None and (payload.get("stream") is not True or options != {"include_usage": True}):
            raise OrchestratorError("model_payload_stream_options", status=500)

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = [{"role": "system", "content": request.system_prompt}]
        messages.extend(dict(item) for item in request.messages)
        payload: dict[str, Any] = {
            "model": self._model_alias,
            "messages": messages,
            "max_completion_tokens": request.output_allowance,
            "stream": request.stream,
            "temperature": 0,
            "top_p": 1,
        }
        if request.stream:
            payload["stream_options"] = {"include_usage": True}
        if request.tools:
            payload["tools"] = [dict(tool) for tool in request.tools]
            payload["parallel_tool_calls"] = False
            payload["tool_choice"] = "auto"
        elif request.response_format is not None:
            payload["response_format"] = dict(request.response_format)
        self.validate_payload(payload, reservation_allowance=request.output_allowance)
        return payload

    def _request(self, request: ModelRequest) -> list[ModelChunk]:
        consume_dispatch_reservation()
        payload = self._payload(request)
        raw = canonical_json(payload).encode("utf-8")
        http_request = urllib.request.Request(
            self._url,
            data=raw,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if request.stream else "application/json",
            },
        )
        try:
            timeout = self._timeout if request.timeout_seconds is None else min(self._timeout, request.timeout_seconds)
            with urllib.request.urlopen(http_request, timeout=timeout) as response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OrchestratorError("backend_failure", status=503, retryable=True) from exc
        if not request.stream:
            parsed = json.loads(body)
            text = parsed["choices"][0]["message"].get("content") or ""
            usage = parsed.get("usage") or {}
            return [ModelChunk(text, ModelUsage(usage.get("completion_tokens"), usage.get("prompt_tokens")))]
        result: list[ModelChunk] = []
        for line in body.decode("utf-8", errors="strict").splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            parsed = json.loads(data)
            usage_raw = parsed.get("usage")
            usage = None
            if isinstance(usage_raw, dict):
                usage = ModelUsage(usage_raw.get("completion_tokens"), usage_raw.get("prompt_tokens"))
            choices = parsed.get("choices") or []
            text = ""
            if choices:
                text = choices[0].get("delta", {}).get("content") or ""
            result.append(ModelChunk(text, usage))
        return result

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        for chunk in await asyncio.to_thread(self._request, request):
            yield chunk
