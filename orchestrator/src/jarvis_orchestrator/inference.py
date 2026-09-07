"""Bounded capability and readiness client, separate from model dispatch."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import _parse_strict_json
from .domain import OrchestratorError


def parse_inference_json(raw: bytes, *, maximum_bytes: int) -> dict[str, object]:
    """Reuse the package's strict decoder for bounded control/model JSON objects."""
    if not raw or len(raw) > maximum_bytes:
        raise ValueError("inference response size invalid")
    value = _parse_strict_json(raw)
    if not isinstance(value, dict):
        raise ValueError("inference response must be an object")
    json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return value


@dataclass(frozen=True, slots=True)
class InferenceCapabilities:
    text: bool
    stream: bool
    tools: bool
    json_object: bool
    json_schema: bool


@dataclass(frozen=True, slots=True)
class InferenceRuntimeIdentity:
    manifest_digest: str
    runtime_identity_sha256: str


class InferenceIdentityEndpointUnavailable(RuntimeError):
    """The authenticated legacy appliance explicitly has no identity route."""


class InferenceControlClient:
    def __init__(self, base_url: str, bearer_token: str, *, timeout_seconds: float = 2.0) -> None:
        try:
            parsed = urlsplit(base_url)
            if (parsed.scheme != "http" or not ipaddress.ip_address(parsed.hostname).is_loopback
                    or parsed.username is not None or parsed.password is not None
                    or parsed.port == 0 or parsed.path.rstrip("/") != "/v1"
                    or parsed.query or parsed.fragment):
                raise ValueError
        except (ValueError, TypeError, AttributeError) as exc:
            raise OrchestratorError("service_unavailable", status=503) from exc
        if (not isinstance(bearer_token, str) or not bearer_token.isascii()
                or not 1 <= len(bearer_token) <= 512
                or any(ord(character) < 32 or ord(character) == 127 for character in bearer_token)
                or isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not 0 < timeout_seconds < float("inf")):
            raise OrchestratorError("service_unavailable", status=503)
        self._origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        self._bearer = bearer_token
        self._timeout = min(2.0, timeout_seconds)

    def _get(self, path: str, *, authenticated: bool, maximum_bytes: int = 16 * 1024) -> dict[str, object]:
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._bearer}"
        async def request_bytes() -> bytes:
            async with asyncio.timeout(self._timeout):
                async with httpx.AsyncClient(timeout=self._timeout, trust_env=False, follow_redirects=False) as client:
                    async with client.stream("GET", self._origin + path, headers=headers) as response:
                        response.raise_for_status()
                        if response.status_code != 200 or response.headers.get("content-encoding", "identity").lower() != "identity":
                            raise OrchestratorError("service_unavailable", status=503, retryable=True)
                        body = bytearray()
                        async for chunk in response.aiter_bytes(chunk_size=4096):
                            body.extend(chunk)
                            if len(body) > maximum_bytes:
                                raise OrchestratorError("service_unavailable", status=503, retryable=True)
                        return bytes(body)

        try:
            return parse_inference_json(asyncio.run(request_bytes()), maximum_bytes=maximum_bytes)
        except (TimeoutError, httpx.HTTPError, ValueError) as exc:
            raise OrchestratorError("service_unavailable", status=503, retryable=True) from exc

    def ready(self) -> bool:
        try:
            value = self._get("/readyz", authenticated=False)
            return set(value) == {"ready"} and value["ready"] is True
        except OrchestratorError:
            return False

    def capabilities(self) -> InferenceCapabilities:
        value = self._get("/jarvis/v1/capabilities", authenticated=True)
        if set(value) != {"model", "capabilities"} or value.get("model") != "jarvis-local":
            raise OrchestratorError("service_unavailable", status=503, retryable=True)
        capabilities = value.get("capabilities")
        expected = {"text", "stream", "tools", "json_object", "json_schema"}
        if not isinstance(capabilities, dict) or set(capabilities) != expected:
            raise OrchestratorError("service_unavailable", status=503, retryable=True)
        if any(type(capabilities[name]) is not bool for name in expected):
            raise OrchestratorError("service_unavailable", status=503, retryable=True)
        result = InferenceCapabilities(**capabilities)
        if not result.text or not result.stream:
            raise OrchestratorError("service_unavailable", status=503, retryable=True)
        return result

    def runtime_identity(self) -> InferenceRuntimeIdentity:
        try:
            value = self._get("/jarvis/v1/identity", authenticated=True)
        except OrchestratorError as exc:
            cause = exc.__cause__
            if isinstance(cause, httpx.HTTPStatusError) and cause.response.status_code in {404, 405}:
                raise InferenceIdentityEndpointUnavailable from None
            raise
        expected = {
            "manifest_digest",
            "runtime_identity_sha256",
            "schema_version",
        }
        if (
            set(value) != expected
            or value.get("schema_version")
            != "jarvis-inference-live-identity/1"
        ):
            raise OrchestratorError(
                "service_unavailable", status=503, retryable=True
            )
        manifest_digest = value.get("manifest_digest")
        runtime_digest = value.get("runtime_identity_sha256")
        if (
            not isinstance(manifest_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest)
            or not isinstance(runtime_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", runtime_digest)
        ):
            raise OrchestratorError(
                "service_unavailable", status=503, retryable=True
            )
        return InferenceRuntimeIdentity(manifest_digest, runtime_digest)
