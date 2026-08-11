"""Authenticated loopback and deterministic fake ResearchPort adapters."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Mapping

from .domain import canonical_json
from .research import (
    ResearchCorpusV1,
    ResearchRequestV1,
    corpus_from_json,
)

MAX_CORPUS_RESPONSE_BYTES = 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_: object, **__: object) -> None:
        return None


class LoopbackResearchAdapter:
    """Thin sidecar adapter; public-network policy remains in the sidecar/proxy."""

    def __init__(self, *, base_url: str, bearer_token: str, timeout_seconds: float = 45.0) -> None:
        if not base_url.startswith(("http://127.0.0.1:", "http://[::1]:")) or base_url.rstrip("/").split("/")[-1] != "v1":
            raise ValueError("research endpoint must be fixed loopback /v1")
        if not bearer_token or len(bearer_token.encode("utf-8")) > 512:
            raise ValueError("bounded research bearer token is required")
        if timeout_seconds != 45.0:
            raise ValueError("research timeout must be 45 seconds")
        self._url = base_url.rstrip("/") + "/research"
        self._qualification_url = base_url.rstrip("/") + "/qualification"
        self._token = bearer_token
        self._timeout = timeout_seconds
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    def qualify(self, expected: Mapping[str, str]) -> None:
        required = {
            "contract_version",
            "model_sha256",
            "profile_sha256",
            "qualification_sha256",
            "language_detector_version",
            "language_detector_sha256",
            "authority_registry_version",
            "browser_qualification_sha256",
        }
        if set(expected) != required:
            raise ValueError("research qualification expectation is incomplete")
        request = urllib.request.Request(
            self._qualification_url,
            method="GET",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
            },
        )
        try:
            with self._opener.open(request, timeout=min(2.0, self._timeout)) as response:
                if response.status != 200 or response.headers.get_content_type() != "application/json":
                    raise RuntimeError("research_qualification_unavailable")
                raw = response.read(16 * 1024 + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("research_qualification_unavailable") from exc
        if len(raw) > 16 * 1024:
            raise RuntimeError("research_qualification_invalid")
        try:
            actual = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("research_qualification_invalid") from exc
        if not isinstance(actual, dict) or set(actual) != required or actual != dict(expected):
            raise RuntimeError("research_qualification_mismatch")

    async def research(
        self, request: ResearchRequestV1, *, idempotency_key: str
    ) -> ResearchCorpusV1:
        return await asyncio.to_thread(self._request, request, idempotency_key)

    def _request(self, request: ResearchRequestV1, idempotency_key: str) -> ResearchCorpusV1:
        body = canonical_json(asdict(request)).encode("utf-8")
        outgoing = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Idempotency-Key": idempotency_key,
            },
        )
        try:
            with self._opener.open(outgoing, timeout=self._timeout) as response:
                if response.status != 200 or response.headers.get_content_type() != "application/json":
                    raise RuntimeError("research_sidecar_invalid_response")
                raw = response.read(MAX_CORPUS_RESPONSE_BYTES + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("research_sidecar_unavailable") from exc
        if len(raw) > MAX_CORPUS_RESPONSE_BYTES:
            raise RuntimeError("research_sidecar_response_too_large")
        try:
            parsed = json.loads(raw)
            if set(parsed) != {"corpus"}:
                raise ValueError("unexpected fields")
            return corpus_from_json(canonical_json(parsed["corpus"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("research_sidecar_contract_failed") from exc


class FakeResearchAdapter:
    """Qualification adapter with stable call and idempotency observations."""

    def __init__(self, results: Mapping[str, ResearchCorpusV1]) -> None:
        self._results = dict(results)
        self.calls: list[tuple[ResearchRequestV1, str]] = []

    async def research(
        self, request: ResearchRequestV1, *, idempotency_key: str
    ) -> ResearchCorpusV1:
        self.calls.append((request, idempotency_key))
        try:
            result = self._results[request.plan.plan_hash]
        except KeyError as exc:
            raise RuntimeError("fake_research_result_missing") from exc
        return result
