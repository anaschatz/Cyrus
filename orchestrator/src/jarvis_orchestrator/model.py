"""Budgeted local-model dispatch with a fail-closed one-shot request guard."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any

from .domain import ModelChunk, ModelRequest, ModelUsage, OrchestratorError
from .ports import ModelPort
from .store import OrchestratorStore


@dataclass(slots=True)
class DispatchPermit:
    reservation_id: str
    consumed: bool = False


_dispatch_reservation: contextvars.ContextVar[DispatchPermit | None] = contextvars.ContextVar(
    "jarvis_model_dispatch_reservation", default=None
)


@contextmanager
def _dispatch_scope(permit: DispatchPermit):
    token = _dispatch_reservation.set(permit)
    try:
        yield
    finally:
        _dispatch_reservation.reset(token)


def active_dispatch_reservation() -> str | None:
    permit = _dispatch_reservation.get()
    return permit.reservation_id if permit is not None else None


def consume_dispatch_reservation() -> str:
    permit = _dispatch_reservation.get()
    if permit is None or permit.consumed:
        raise OrchestratorError("model_dispatch_without_reservation", status=500)
    permit.consumed = True
    return permit.reservation_id


class GuardedAsyncTransport:
    """Lazily create the guarded HTTPX transport at model activation time."""

    def __new__(cls, wrapped: Any | None = None) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError(
                "httpx from pydantic-ai-slim[openai]==2.27.0 is required"
            ) from exc

        class _RuntimeGuardedAsyncTransport(httpx.AsyncBaseTransport):
            def __init__(self, selected: Any | None) -> None:
                self._wrapped = selected or httpx.AsyncHTTPTransport(retries=0)

            async def handle_async_request(
                self, request: httpx.Request
            ) -> httpx.Response:
                if (
                    request.method != "POST"
                    or request.url.path != "/v1/chat/completions"
                ):
                    raise OrchestratorError(
                        "model_transport_endpoint_forbidden", status=500
                    )
                consume_dispatch_reservation()
                return await self._wrapped.handle_async_request(request)

            async def aclose(self) -> None:
                await self._wrapped.aclose()

        return _RuntimeGuardedAsyncTransport(wrapped)


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
        permit = DispatchPermit(reservation_id)
        usage: ModelUsage | None = None
        completed = False
        try:
            self._store.mark_model_dispatched(reservation_id, request.task_id, request.epoch)
            with _dispatch_scope(permit):
                stream = aiter(self.__delegate.stream(request))
            try:
                while True:
                    with _dispatch_scope(permit):
                        try:
                            chunk = await anext(stream)
                        except StopAsyncIteration:
                            break
                    if chunk.usage is not None:
                        usage = chunk.usage
                    yield chunk
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    with _dispatch_scope(permit):
                        await close()
            completed = True
        except (asyncio.CancelledError, GeneratorExit, OrchestratorError):
            raise
        except Exception as exc:
            # SDK-specific failures share one stable lifecycle contract.
            raise OrchestratorError(
                "backend_failure", status=503, retryable=True
            ) from exc
        finally:
            if completed:
                self._store.reconcile_model(
                    reservation_id, request.task_id, request.epoch,
                    usage.output_tokens if usage is not None else None,
                )
            else:
                # Preserve cancellation/epoch failures while charging unknown usage.
                with suppress(Exception):
                    self._store.reconcile_model(
                        reservation_id, request.task_id, request.epoch, None
                    )
