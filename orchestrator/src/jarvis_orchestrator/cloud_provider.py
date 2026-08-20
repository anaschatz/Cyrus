"""Explicit OpenRouter adapter for the qualified M1C cloud path."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from typing import Any, Mapping
from pathlib import Path

from .cloud_contract import (
    CloudModelResultV1,
    EndpointHealthV1,
    OpenRouterRevalidationV1,
    QualifiedEndpointV1,
)
from .domain import OrchestratorError, utc_text
from .jcs import domain_sha256, jcs_bytes
from .store import OrchestratorStore

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
            if request.method != "POST" or request.url.path != "/api/v1/chat/completions":
                raise OrchestratorError("cloud_transport_endpoint_forbidden", status=500)
            consume_physical_cloud_permit()
            return await self._wrapped.handle_async_request(request)

        async def aclose(self) -> None:
            await self._wrapped.aclose()
else:  # pragma: no cover
    class GuardedOpenRouterTransport:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("httpx is required")


def _decimal_microusd(value: Any) -> int:
    try:
        selected = Decimal(str(value))
    except Exception as exc:
        raise OrchestratorError("provider_receipt_invalid", status=502) from exc
    if not selected.is_finite() or selected < 0:
        raise OrchestratorError("provider_receipt_invalid", status=502)
    return int((selected * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


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
        if not bearer_token or len(bearer_token.encode("utf-8")) > 512:
            raise ValueError("bounded OpenRouter bearer token is required")
        self._store = store
        self._token = bearer_token
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Accept": "application/json",
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
        await self._dispatch_client.aclose()
        await self._control_client.aclose()

    async def revalidate(self, endpoint: QualifiedEndpointV1) -> OpenRouterRevalidationV1:
        """Fetch the live ZDR catalog and key brake once, concurrently, no retry."""

        endpoint.validate()
        try:
            async with asyncio.timeout(2.0):
                catalog_response, key_response = await asyncio.gather(
                    self._control_client.get("/endpoints/zdr"),
                    self._control_client.get("/key"),
                )
            catalog_response.raise_for_status()
            key_response.raise_for_status()
            catalog = catalog_response.json()
            key = key_response.json()
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
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
            and str(row.get("quantization")) == endpoint.quantization
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
            prompt_per_m = Decimal(str(pricing["prompt"])) * Decimal(1_000_000)
            completion_per_m = Decimal(str(pricing["completion"])) * Decimal(1_000_000)
            request_rate = Decimal(str(pricing.get("request", "0")))
            observed_parameters = tuple(sorted(str(item) for item in live["supported_parameters"]))
            stable_equal = all(
                (
                    str(live.get("quantization")) == endpoint.quantization,
                    str(live.get("provider_name")) == endpoint.runtime_provider_name,
                    observed_parameters == tuple(sorted(endpoint.supported_parameters)),
                    int(live.get("context_length", 0)) == endpoint.context_limit,
                    int(live.get("max_completion_tokens", 0)) >= endpoint.output_limit,
                    prompt_per_m == Decimal(endpoint.prompt_rate_per_m),
                    completion_per_m == Decimal(endpoint.completion_rate_per_m),
                    request_rate == Decimal(endpoint.request_rate),
                )
            )
            health = EndpointHealthV1(
                model_canonical_id=endpoint.model_canonical_id,
                provider_endpoint_tag=endpoint.provider_endpoint_tag,
                status=int(live["status"]),
                uptime_last_30m_basis_points=int((Decimal(str(live["uptime_last_30m"])) * 100).to_integral_value()),
                p90_latency_ms=int((Decimal(str(latency["p90"])) * 1000).to_integral_value(rounding=ROUND_CEILING)),
                observed_at=utc_text(),
                age_ms=0,
            )
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
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
            key_limit != 5_000_000
            or key_data.get("is_management_key") is not False
            or key_data.get("is_provisioning_key") is not False
            or key_data.get("include_byok_in_limit") is not False
        ):
            raise OrchestratorError("cloud_key_policy_mismatch", status=409)
        remaining = _decimal_microusd(key_data.get("limit_remaining"))
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
            response = await self._dispatch_client.post(
                "/chat/completions",
                content=jcs_bytes(dispatch_body),
                headers={
                    "Content-Type": "application/json",
                    "X-OpenRouter-Metadata": "enabled",
                },
            )
            generation_id = response.headers.get("X-Generation-Id")
            response.raise_for_status()
            document = response.json()
            if not isinstance(document, Mapping):
                raise OrchestratorError("provider_response_invalid", status=502)
            generation_id = str(document.get("id") or generation_id or "")
            choices = document.get("choices")
            usage = document.get("usage")
            if not generation_id or not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, Mapping):
                raise OrchestratorError("provider_response_invalid", status=502)
            choice = choices[0]
            if not isinstance(choice, Mapping) or choice.get("finish_reason") != "stop":
                raise OrchestratorError("provider_response_invalid", status=502)
            message = choice.get("message")
            text = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(text, str) or not text.strip():
                raise OrchestratorError("provider_response_invalid", status=502)
            metadata = document.get("openrouter_metadata")
            if isinstance(metadata, Mapping):
                if metadata.get("attempt") != 1 or metadata.get("strategy") not in {"direct", "bodybuilder"}:
                    raise OrchestratorError("provider_identity_mismatch", status=502)

            receipt_response = await self._control_client.get("/generation", params={"id": generation_id})
            receipt_response.raise_for_status()
            receipt_raw = receipt_response.json()
            receipt_data = receipt_raw.get("data") if isinstance(receipt_raw, Mapping) else None
            if not isinstance(receipt_data, Mapping):
                raise OrchestratorError("provider_identity_mismatch", status=502)
            receipt = {
                "generation_id": generation_id,
                "model": str(receipt_data.get("model", "")),
                "native_tokens_completion": int(receipt_data.get("native_tokens_completion", 0)),
                "native_tokens_prompt": int(receipt_data.get("native_tokens_prompt", 0)),
                "native_tokens_reasoning": int(receipt_data.get("native_tokens_reasoning", 0)),
                "provider_name": str(receipt_data.get("provider_name", "")),
                "total_cost_microusd": _decimal_microusd(receipt_data.get("total_cost")),
            }
            verified = (
                receipt["model"] == endpoint.model_canonical_id
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
    if not selected.is_absolute() or not selected.is_file():
        raise RuntimeError("cloud secret reference is unavailable")
    if os.name != "nt" and selected.stat().st_mode & 0o077:
        raise RuntimeError("cloud secret file permissions must be owner-only")
    token = selected.read_text(encoding="utf-8").strip()
    if not token or "\n" in token or len(token.encode("utf-8")) > 512:
        raise RuntimeError("cloud secret file is invalid")
    return token


def load_endpoint_registry(path: str | Path) -> tuple[QualifiedEndpointV1, ...]:
    selected = Path(path)
    try:
        document = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
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
        endpoint = QualifiedEndpointV1(
            model_canonical_id=str(raw["model_canonical_id"]),
            provider_endpoint_tag=str(raw["provider_endpoint_tag"]),
            quantization=str(raw["quantization"]),
            supported_parameters=tuple(str(item) for item in raw["supported_parameters"]),
            qualification_version=str(raw["qualification_version"]),
            qualification_sha256=str(raw["qualification_sha256"]),
            context_limit=raw["context_limit"],
            output_limit=raw["output_limit"],
            prompt_rate_per_m=str(raw["prompt_rate_per_m"]),
            completion_rate_per_m=str(raw["completion_rate_per_m"]),
            request_rate=str(raw["request_rate"]),
            runtime_provider_name=str(raw["runtime_provider_name"]),
        ).validate()
        result.append(endpoint)
    if len(result) != len({(item.model_canonical_id, item.provider_endpoint_tag) for item in result}):
        raise RuntimeError("cloud qualification endpoint identity is duplicated")
    return tuple(result)
