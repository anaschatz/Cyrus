"""Explicit OpenRouter adapter for the qualified M1C cloud path."""

from __future__ import annotations

import asyncio
import contextvars
import os
import stat
import zlib
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from typing import Any, Mapping
from pathlib import Path

from .cloud_contract import (
    CloudModelResultV1,
    EndpointHealthV1,
    OpenRouterRevalidationV1,
    QualifiedEndpointV1,
    normalize_decimal,
)
from .domain import OrchestratorError, utc_text
from .config import _read_private_configuration
from .inference import parse_inference_json
from .jcs import domain_sha256, jcs_bytes
from .store import OrchestratorStore

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_REGISTRY_BYTES = 1024 * 1024


@dataclass(slots=True)
class PhysicalCloudPermit:
    permit_id: str
    consumed: bool = False


_cloud_permit: contextvars.ContextVar[PhysicalCloudPermit | None] = contextvars.ContextVar(
    "jarvis_cloud_dispatch_permit", default=None
)


def consume_physical_cloud_permit() -> str:
    permit = _cloud_permit.get()
    if permit is None or permit.consumed:
        raise OrchestratorError("cloud_dispatch_without_permit", status=500)
    permit.consumed = True
    return permit.permit_id


if httpx is not None:
    class GuardedOpenRouterTransport(httpx.AsyncBaseTransport):
        """Reject every physical cloud POST without a one-shot persisted permit."""

        def __init__(self, wrapped: httpx.AsyncBaseTransport | None = None) -> None:
            self._wrapped = wrapped or httpx.AsyncHTTPTransport(retries=0)

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.method != "POST" or request.url != httpx.URL(f"{OPENROUTER_BASE_URL}/chat/completions"):
                raise OrchestratorError("cloud_transport_endpoint_forbidden", status=500)
            consume_physical_cloud_permit()
            return await self._wrapped.handle_async_request(request)

        async def aclose(self) -> None:
            await self._wrapped.aclose()
else:  # pragma: no cover
    class GuardedOpenRouterTransport:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("httpx is required")


def _token_count(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OrchestratorError("provider_receipt_invalid", status=502)
    return value


def _fraction(value: Any) -> Fraction:
    try:
        return Fraction(normalize_decimal(str(value)))
    except (ValueError, ArithmeticError) as exc:
        raise OrchestratorError("provider_receipt_invalid", status=502) from exc


def _decimal_microusd(value: Any, *, round_up: bool = True) -> int:
    selected = _fraction(value) * 1_000_000
    return ceil(selected) if round_up else floor(selected)


async def revalidate_cloud_endpoints(adapter: Any, endpoints: tuple[QualifiedEndpointV1, ...]) -> tuple[OpenRouterRevalidationV1, ...]:
    """Drain every endpoint check before callers close the shared client/store."""
    results = await asyncio.gather(*(adapter.revalidate(endpoint) for endpoint in endpoints), return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return tuple(results)


def _cloud_token(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 512 or any(not 0x21 <= ord(char) <= 0x7e for char in value):
        raise ValueError("bounded printable OpenRouter bearer token is required")
    return value


async def _response_bytes(response: Any) -> bytes:
    """Bound wire bytes and decompression before allocating the complete JSON."""
    if response.is_stream_consumed:
        # In-memory transports can supply an already decoded response.
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise ValueError("provider response exceeds maximum size")
        return response.content
    encoding = response.headers.get("content-encoding", "identity").lower().strip()
    if encoding not in ("identity", "gzip", "deflate"):
        raise ValueError("unsupported provider content encoding")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS) if encoding != "identity" else None
    body = bytearray()
    wire_size = 0
    async for chunk in response.aiter_raw():
        wire_size += len(chunk)
        if wire_size > _MAX_RESPONSE_BYTES:
            raise ValueError("provider response exceeds maximum size")
        decoded = decoder.decompress(chunk, _MAX_RESPONSE_BYTES - len(body) + 1) if decoder else chunk
        if len(body) + len(decoded) > _MAX_RESPONSE_BYTES:
            raise ValueError("provider response exceeds maximum size")
        body.extend(decoded)
    if decoder and (not decoder.eof or decoder.unused_data):
        raise ValueError("incomplete or trailing compressed provider response")
    return bytes(body)


class OpenRouterCloudAdapter:
    """No SDK retries, redirects, environment proxying or provider fallback."""

    def __init__(
        self,
        store: OrchestratorStore,
        *,
        bearer_token: str,
        timeout_seconds: float = 60.0,
        transport: Any | None = None,
        control_transport: Any | None = None,
    ) -> None:
        if httpx is None:
            raise RuntimeError("httpx is required")
        bearer_token = _cloud_token(bearer_token)
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("positive finite cloud timeout is required")
        try:
            timeout_seconds = float(timeout_seconds)
        except OverflowError:
            raise ValueError("positive finite cloud timeout is required") from None
        if not 0 < timeout_seconds < float("inf"):
            raise ValueError("positive finite cloud timeout is required")
        self._timeout_seconds = timeout_seconds
        self._store = store
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
        self._dispatch_client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
            transport=GuardedOpenRouterTransport(transport),
        )
        self._control_client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(2.0),
            follow_redirects=False,
            trust_env=False,
            transport=control_transport,
        )

    async def aclose(self) -> None:
        try:
            await self._dispatch_client.aclose()
        finally:
            await self._control_client.aclose()

    async def _request_json(self, client: Any, method: str, path: str, **kwargs: Any) -> tuple[dict[str, object], Any]:
        try:
            async with asyncio.timeout(self._timeout_seconds if method == "POST" else 2.0):
                async with client.stream(method, path, **kwargs) as response:
                    response.raise_for_status()
                    if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
                        raise ValueError("provider response must be JSON")
                    raw = await _response_bytes(response)
                    return parse_inference_json(raw, maximum_bytes=_MAX_RESPONSE_BYTES), response.headers
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise OrchestratorError("provider_timeout", status=504) from exc
        except (httpx.HTTPError, ValueError, RecursionError, zlib.error) as exc:
            raise OrchestratorError("provider_response_invalid", status=502) from exc

    async def revalidate(self, endpoint: QualifiedEndpointV1) -> OpenRouterRevalidationV1:
        """Fetch the live ZDR catalog and key brake once, concurrently, no retry."""

        endpoint.validate()
        try:
            async with asyncio.timeout(2.0):
                catalog_response, key_response = await asyncio.gather(
                    self._request_json(self._control_client, "GET", "/endpoints/zdr"),
                    self._request_json(self._control_client, "GET", "/key"),
                    return_exceptions=True,
                )
                for response in (catalog_response, key_response):
                    if isinstance(response, BaseException):
                        raise response
            catalog, _ = catalog_response
            key, _ = key_response
        except BaseException as exc:
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            raise OrchestratorError("revalidation_unavailable", status=503, retryable=True) from exc
        rows = catalog.get("data") if isinstance(catalog, Mapping) else None
        if not isinstance(rows, list):
            raise OrchestratorError("revalidation_unavailable", status=503, retryable=True)
        matches = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("model_id") == endpoint.model_canonical_id
            and row.get("tag") == endpoint.provider_endpoint_tag
        ]
        if len(matches) != 1:
            raise OrchestratorError("endpoint_unhealthy", status=409)
        live = matches[0]
        runtime_identity_matches = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("model_id") == endpoint.model_canonical_id
            and row.get("provider_name") == endpoint.runtime_provider_name
            and row.get("quantization") == endpoint.quantization
        ]
        if len(runtime_identity_matches) != 1 or runtime_identity_matches[0].get("tag") != endpoint.provider_endpoint_tag:
            # The generation receipt exposes provider_name but not full tag or
            # quantization. Qualification is therefore safe only when the live
            # catalog makes that receipt identity unambiguous.
            raise OrchestratorError("endpoint_identity_ambiguous", status=409)
        pricing = live.get("pricing")
        latency = live.get("latency_last_30m")
        if not isinstance(pricing, Mapping) or not isinstance(latency, Mapping):
            raise OrchestratorError("endpoint_unhealthy", status=409)
        try:
            # ZDR catalog prices are per token; frozen request/max-price and
            # ledger rates are represented per million tokens.
            prompt_per_m = _fraction(pricing["prompt"]) * 1_000_000
            completion_per_m = _fraction(pricing["completion"]) * 1_000_000
            request_rate = _fraction(pricing.get("request", "0"))
            if (not isinstance(live["supported_parameters"], list)
                    or any(not isinstance(item, str) for item in live["supported_parameters"])
                    or any(type(live[name]) is not int for name in ("context_length", "max_completion_tokens", "status"))):
                raise ValueError("invalid catalog types")
            uptime = _fraction(live["uptime_last_30m"])
            if uptime > 100:
                raise ValueError("invalid uptime")
            observed_parameters = tuple(sorted(live["supported_parameters"]))
            stable_equal = all(
                (
                    live.get("quantization") == endpoint.quantization,
                    live.get("provider_name") == endpoint.runtime_provider_name,
                    observed_parameters == tuple(sorted(endpoint.supported_parameters)),
                    live["context_length"] == endpoint.context_limit,
                    live["max_completion_tokens"] >= endpoint.output_limit,
                    prompt_per_m == _fraction(endpoint.prompt_rate_per_m),
                    completion_per_m == _fraction(endpoint.completion_rate_per_m),
                    request_rate == _fraction(endpoint.request_rate),
                )
            )
            health = EndpointHealthV1(
                model_canonical_id=endpoint.model_canonical_id,
                provider_endpoint_tag=endpoint.provider_endpoint_tag,
                status=live["status"],
                uptime_last_30m_basis_points=floor(uptime * 100),
                p90_latency_ms=ceil(_fraction(latency["p90"]) * 1000),
                observed_at=utc_text(),
                age_ms=0,
            )
        except (KeyError, TypeError, ValueError, ArithmeticError, OrchestratorError) as exc:
            raise OrchestratorError("endpoint_unhealthy", status=409) from exc
        if not stable_equal:
            raise OrchestratorError("binding_changed", status=409)
        if not health.eligible():
            raise OrchestratorError("endpoint_unhealthy", status=409)
        key_data = key.get("data") if isinstance(key, Mapping) else None
        if not isinstance(key_data, Mapping) or key_data.get("limit_reset") != "monthly":
            raise OrchestratorError("cloud_budget_unavailable", status=409)
        key_limit = _decimal_microusd(key_data.get("limit"))
        if (
            _fraction(key_data.get("limit")) != 5
            or key_data.get("is_management_key") is not False
            or key_data.get("is_provisioning_key") is not False
            or key_data.get("include_byok_in_limit") is not False
        ):
            raise OrchestratorError("cloud_key_policy_mismatch", status=409)
        remaining = _decimal_microusd(key_data.get("limit_remaining"), round_up=False)
        usage = _decimal_microusd(key_data.get("usage_monthly"))
        self._store.store_endpoint_health(
            endpoint.qualification_sha256,
            health.health_sha256,
            {
                "age_ms": health.age_ms,
                "model_canonical_id": health.model_canonical_id,
                "observed_at": health.observed_at,
                "p90_latency_ms": health.p90_latency_ms,
                "provider_endpoint_tag": health.provider_endpoint_tag,
                "status": health.status,
                "uptime_last_30m_basis_points": health.uptime_last_30m_basis_points,
            },
            observed_at=health.observed_at,
        )
        return OpenRouterRevalidationV1(
            endpoint,
            health,
            remaining,
            usage,
            "monthly",
            key_limit,
            True,
        )

    async def dispatch(
        self,
        *,
        task_id: str,
        epoch: int,
        endpoint: QualifiedEndpointV1,
        permit_id: str,
        reservation_id: str,
        dispatch_body: Mapping[str, Any],
        dispatch_body_sha256: str,
        input_token_limit: int,
        output_token_limit: int,
        max_reserved_microusd: int,
    ) -> CloudModelResultV1:
        """Buffer output, verify provider identity, then return publishable text."""

        if domain_sha256("cloud-dispatch-body/1", dispatch_body) != dispatch_body_sha256:
            raise OrchestratorError("binding_mismatch", status=409)
        self._store.consume_cloud_dispatch_permit(
            permit_id,
            task_id,
            epoch,
            dispatch_body_sha256,
        )
        token = _cloud_permit.set(PhysicalCloudPermit(permit_id))
        generation_id: str | None = None
        try:
            document, response_headers = await self._request_json(
                self._dispatch_client, "POST",
                "/chat/completions",
                content=jcs_bytes(dispatch_body),
                headers={
                    "Content-Type": "application/json",
                    "X-OpenRouter-Metadata": "enabled",
                },
            )
            generation_id = response_headers.get("X-Generation-Id")
            if not isinstance(document, Mapping):
                raise OrchestratorError("provider_response_invalid", status=502)
            header_generation_id = generation_id
            generation_id = document.get("id") or header_generation_id
            if not isinstance(generation_id, str) or not generation_id or len(generation_id) > 512 or (
                header_generation_id is not None and header_generation_id != generation_id
            ):
                raise OrchestratorError("provider_response_invalid", status=502)
            choices = document.get("choices")
            usage = document.get("usage")
            if not generation_id or not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, Mapping):
                raise OrchestratorError("provider_response_invalid", status=502)
            _token_count(usage.get("prompt_tokens"))
            _token_count(usage.get("completion_tokens"))
            choice = choices[0]
            if not isinstance(choice, Mapping) or choice.get("finish_reason") != "stop":
                raise OrchestratorError("provider_response_invalid", status=502)
            message = choice.get("message")
            text = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(text, str) or not text.strip():
                raise OrchestratorError("provider_response_invalid", status=502)
            metadata = document.get("openrouter_metadata")
            if metadata is not None:
                if (not isinstance(metadata, Mapping) or type(metadata.get("attempt")) is not int
                        or metadata["attempt"] != 1 or metadata.get("strategy") not in ("direct", "bodybuilder")):
                    raise OrchestratorError("provider_identity_mismatch", status=502)

            receipt_raw, _ = await self._request_json(self._control_client, "GET", "/generation", params={"id": generation_id})
            receipt_data = receipt_raw.get("data") if isinstance(receipt_raw, Mapping) else None
            if not isinstance(receipt_data, Mapping):
                raise OrchestratorError("provider_identity_mismatch", status=502)
            receipt = {
                "generation_id": generation_id,
                "model": str(receipt_data.get("model", "")),
                "native_tokens_completion": _token_count(receipt_data.get("native_tokens_completion")),
                "native_tokens_prompt": _token_count(receipt_data.get("native_tokens_prompt")),
                "native_tokens_reasoning": _token_count(receipt_data.get("native_tokens_reasoning", 0)),
                "provider_name": str(receipt_data.get("provider_name", "")),
                "total_cost_microusd": _decimal_microusd(receipt_data.get("total_cost")),
            }
            verified = (
                receipt_data.get("id") == generation_id
                and receipt["model"] == endpoint.model_canonical_id
                and receipt["provider_name"] == endpoint.runtime_provider_name
                and int(receipt["native_tokens_prompt"]) <= input_token_limit
                and int(receipt["native_tokens_completion"]) + int(receipt["native_tokens_reasoning"]) <= output_token_limit
                and int(receipt["total_cost_microusd"]) <= max_reserved_microusd
            )
            self._store.store_cloud_provider_receipt(
                task_id,
                epoch,
                generation_id=generation_id,
                receipt=receipt,
                verified=verified,
            )
            if not verified:
                self._store.quarantine_endpoint(endpoint.qualification_sha256)
                # A mismatching receipt is not publishable, but the provider
                # may still have billed the physical request.  Charge at
                # least the full reservation, or the reported cost when it is
                # higher, before surfacing the violation.
                self._store.reconcile_cloud_dispatch(
                    reservation_id,
                    actual_microusd=max(
                        int(receipt["total_cost_microusd"]),
                        max_reserved_microusd,
                    ),
                )
                raise OrchestratorError("provider_identity_mismatch", status=502)
            actual = self._store.reconcile_cloud_dispatch(
                reservation_id,
                actual_microusd=int(receipt["total_cost_microusd"]),
            )
            return CloudModelResultV1(
                text=text,
                generation_id=generation_id,
                input_tokens=int(receipt["native_tokens_prompt"]),
                output_tokens=int(receipt["native_tokens_completion"]),
                reasoning_tokens=int(receipt["native_tokens_reasoning"]),
                actual_microusd=actual,
                receipt=receipt,
            )
        except BaseException:
            with suppress(Exception):
                self._store.reconcile_cloud_dispatch(
                    reservation_id,
                    actual_microusd=None,
                    ambiguous=True,
                )
            raise
        finally:
            _cloud_permit.reset(token)


def load_openrouter_secret(path: str | Path) -> str:
    selected = Path(path)
    if not selected.is_absolute():
        raise RuntimeError("cloud secret reference is unavailable")
    try:
        return _cloud_token(_read_private_configuration(selected).decode("utf-8").strip())
    except (ValueError, OSError) as exc:
        raise RuntimeError("cloud secret file is invalid") from exc


def load_endpoint_registry(path: str | Path) -> tuple[QualifiedEndpointV1, ...]:
    selected = Path(path)
    try:
        with open(selected, "rb", opener=lambda name, flags: os.open(
            name, flags | getattr(os, "O_NONBLOCK", 0)
        )) as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("registry must be a regular file")
            document = parse_inference_json(handle.read(_MAX_REGISTRY_BYTES + 1), maximum_bytes=_MAX_REGISTRY_BYTES)
    except (OSError, ValueError, RecursionError) as exc:
        raise RuntimeError("cloud qualification registry is unavailable") from exc
    if not isinstance(document, Mapping) or set(document) != {"version", "endpoints"} or document["version"] != "endpoint-registry/1" or not isinstance(document["endpoints"], list):
        raise RuntimeError("cloud qualification registry contract is invalid")
    expected = {
        "model_canonical_id",
        "provider_endpoint_tag",
        "quantization",
        "supported_parameters",
        "qualification_version",
        "qualification_sha256",
        "context_limit",
        "output_limit",
        "prompt_rate_per_m",
        "completion_rate_per_m",
        "request_rate",
        "runtime_provider_name",
    }
    result: list[QualifiedEndpointV1] = []
    for raw in document["endpoints"]:
        if not isinstance(raw, Mapping) or set(raw) != expected or not isinstance(raw["supported_parameters"], list):
            raise RuntimeError("cloud qualification endpoint is invalid")
        endpoint = QualifiedEndpointV1(**{
            **raw, "supported_parameters": tuple(raw["supported_parameters"])
        }).validate()
        result.append(endpoint)
    if len(result) != len({(item.model_canonical_id, item.provider_endpoint_tag) for item in result}):
        raise RuntimeError("cloud qualification endpoint identity is duplicated")
    return tuple(result)
