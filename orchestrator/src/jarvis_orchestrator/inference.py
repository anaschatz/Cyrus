"""Bounded capability and readiness client, separate from model dispatch."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from .domain import OrchestratorError


@dataclass(frozen=True, slots=True)
class InferenceCapabilities:
    text: bool
    stream: bool
    tools: bool
    json_object: bool
    json_schema: bool


class InferenceControlClient:
    def __init__(self, base_url: str, bearer_token: str, *, timeout_seconds: float = 2.0) -> None:
        parsed = urlsplit(base_url)
        self._origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        self._bearer = bearer_token
        self._timeout = min(2.0, timeout_seconds)

    def _get(self, path: str, *, authenticated: bool) -> dict[str, object]:
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._bearer}"
        request = urllib.request.Request(self._origin + path, method="GET", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                if response.status != 200:
                    raise OrchestratorError("service_unavailable", status=503, retryable=True)
                body = response.read(16 * 1024 + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OrchestratorError("service_unavailable", status=503, retryable=True) from exc
        if len(body) > 16 * 1024:
            raise OrchestratorError("service_unavailable", status=503, retryable=True)
        try:
            value = json.loads(body.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OrchestratorError("service_unavailable", status=503, retryable=True) from exc
        if not isinstance(value, dict):
            raise OrchestratorError("service_unavailable", status=503, retryable=True)
        return value

    def ready(self) -> bool:
        try:
            return self._get("/readyz", authenticated=False) == {"ready": True}
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
