"""Bounded SearXNG and direct HTTP adapters for the Research sidecar."""

from __future__ import annotations

import asyncio
import json
import zlib
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from jarvis_orchestrator.research import validate_public_url
from jarvis_orchestrator.research_security import (
    ALLOWED_MIME,
    DIRECT_MAX_BODY_BYTES,
    DIRECT_MAX_REDIRECTS,
    FIXED_USER_AGENT,
    ROBOTS_DEADLINE_SECONDS,
    ROBOTS_MAX_BYTES,
    ROBOTS_MAX_REDIRECTS,
    RetryBudgetV1,
    RobotsDecision,
    robots_decision,
)

SEARCH_RESPONSE_MAX_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class SearchCandidateV1:
    requested_url: str
    title: str
    snippet: str
    engine: str

    def __post_init__(self) -> None:
        validate_public_url(self.requested_url)
        if not isinstance(self.title, str) or not 1 <= len(self.title) <= 512:
            raise ValueError("search title is invalid")
        if not isinstance(self.snippet, str) or len(self.snippet) > 2_000:
            raise ValueError("search snippet is invalid")
        if not isinstance(self.engine, str) or not 1 <= len(self.engine) <= 64:
            raise ValueError("search engine is invalid")


@dataclass(frozen=True, slots=True)
class FetchResultV1:
    requested_url: str
    final_url: str
    mime_type: str
    body: bytes
    redirect_count: int

    def __post_init__(self) -> None:
        validate_public_url(self.requested_url)
        validate_public_url(self.final_url)
        if self.mime_type not in ALLOWED_MIME or len(self.body) > DIRECT_MAX_BODY_BYTES:
            raise ValueError("fetch result is outside contract")
        if not 0 <= self.redirect_count <= DIRECT_MAX_REDIRECTS:
            raise ValueError("redirect count is invalid")


@dataclass(frozen=True, slots=True)
class RetrievalCorpusV1:
    candidates: tuple[SearchCandidateV1, ...]
    fetched: tuple[FetchResultV1, ...]
    technical_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.fetched) > 8 or len(self.candidates) > 8:
            raise ValueError("retrieval corpus exceeds page budget")


class SearXNGSearchAdapter:
    def __init__(self, client: httpx.AsyncClient, *, endpoint: str = "http://searxng:8080/search") -> None:
        parsed = urlsplit(endpoint)
        target = (parsed.scheme, parsed.hostname, parsed.port, parsed.path)
        if target not in {
            ("http", "searxng", 8080, "/search"),
            ("http", "qualification-search", 8080, "/search"),
        }:
            raise ValueError("SearXNG endpoint must be the fixed internal endpoint")
        self._client = client
        self._endpoint = endpoint

    async def search(self, query: str, *, language: str, limit: int) -> tuple[SearchCandidateV1, ...]:
        if not isinstance(query, str) or not 1 <= len(query) <= 512:
            raise ValueError("search query is invalid")
        if language not in {"el", "en"} or isinstance(limit, bool) or not 1 <= limit <= 8:
            raise ValueError("search bounds are invalid")
        async with self._client.stream(
            "POST",
            self._endpoint,
            data={"q": query, "format": "json", "language": language, "safesearch": "1"},
            headers={"Accept": "application/json", "User-Agent": FIXED_USER_AGENT},
        ) as response:
            if response.status_code != 200 or response.headers.get("content-type", "").split(";", 1)[0].strip().casefold() != "application/json":
                raise RuntimeError("searxng_invalid_response")
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > SEARCH_RESPONSE_MAX_BYTES:
                    raise RuntimeError("searxng_response_too_large")
                chunks.append(chunk)
        try:
            parsed = json.loads(b"".join(chunks))
            raw_results = parsed["results"]
            if not isinstance(raw_results, list):
                raise TypeError
            selected: list[SearchCandidateV1] = []
            seen: set[str] = set()
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                url = validate_public_url(item.get("url", ""))
                if url in seen:
                    continue
                selected.append(
                    SearchCandidateV1(
                        url,
                        item.get("title", ""),
                        item.get("content", ""),
                        item.get("engine", "unknown"),
                    )
                )
                seen.add(url)
                if len(selected) == limit:
                    break
            return tuple(selected)
        except (KeyError, TypeError, ValueError, RecursionError) as exc:
            raise RuntimeError("searxng_contract_failed") from exc


class DirectHTTPFetcher:
    def __init__(self, client: httpx.AsyncClient, *, retry_budget: RetryBudgetV1 | None = None) -> None:
        self._client = client
        self._retry = retry_budget or RetryBudgetV1()
        self._robots: dict[str, tuple[int, bytes]] = {}

    @classmethod
    def with_egress_proxy(
        cls,
        proxy_url: str,
        *,
        proxy_headers: dict[str, str] | None = None,
        retry_budget: RetryBudgetV1 | None = None,
    ) -> "DirectHTTPFetcher":
        parsed = urlsplit(proxy_url)
        if (parsed.scheme, parsed.hostname, parsed.port) != ("http", "egress-guard", 8080):
            raise ValueError("egress proxy must be the fixed internal endpoint")
        client = httpx.AsyncClient(
            proxy=httpx.Proxy(proxy_url, headers=proxy_headers),
            follow_redirects=False,
            timeout=httpx.Timeout(10.0, connect=3.0),
            trust_env=False,
            headers={"User-Agent": FIXED_USER_AGENT, "Accept-Encoding": "gzip, deflate"},
        )
        return cls(client, retry_budget=retry_budget)

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def network_retry_count(self) -> int:
        return int(self._retry.used)

    async def fetch(self, url: str, *, deadline_seconds: float = 45.0) -> FetchResultV1:
        requested = validate_public_url(url)
        async with asyncio.timeout(deadline_seconds):
            decision = await self._check_robots(self._origin(requested), self._robots_target(requested))
            if decision != RobotsDecision.ALLOW:
                raise RuntimeError("robots_disallowed" if decision == RobotsDecision.DISALLOW else "robots_failed_closed")
            current = requested
            for redirects in range(DIRECT_MAX_REDIRECTS + 1):
                response = await self._request_with_retry(current)
                try:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        if redirects == DIRECT_MAX_REDIRECTS:
                            raise RuntimeError("redirect_limit")
                        location = response.headers.get("location")
                        if not location:
                            raise RuntimeError("redirect_missing_location")
                    else:
                        if response.status_code != 200:
                            raise RuntimeError("fetch_http_status")
                        mime = response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
                        if mime not in ALLOWED_MIME:
                            raise RuntimeError("fetch_mime_forbidden")
                        body = await self._bounded_body(response)
                        return FetchResultV1(requested, current, mime, body, redirects)
                finally:
                    await response.aclose()
                current = validate_public_url(urljoin(current, location))
                redirected = await self._check_robots(self._origin(current), self._robots_target(current))
                if redirected != RobotsDecision.ALLOW:
                    raise RuntimeError("redirect_robots_failed_closed")
        raise AssertionError("unreachable")

    async def _request_with_retry(self, url: str) -> httpx.Response:
        try:
            response = await self._client.send(
                self._client.build_request("GET", url, headers={"Accept": "text/html,application/xhtml+xml,text/plain"}),
                stream=True,
            )
        except httpx.ConnectTimeout:
            if self._retry.consume(error_code="connect_timeout", remaining_seconds=45):
                return await self._client.send(self._client.build_request("GET", url), stream=True)
            raise
        except httpx.ReadError:
            if self._retry.consume(error_code="connect_reset", remaining_seconds=45):
                return await self._client.send(self._client.build_request("GET", url), stream=True)
            raise
        retry_after = response.headers.get("retry-after", "0")
        try:
            delay = min(0.5, max(0.0, float(retry_after)))
        except ValueError:
            delay = 0.5
        if response.status_code in {429, 502, 503, 504} and self._retry.consume(
            status=response.status_code, delay_seconds=delay, remaining_seconds=45
        ):
            await response.aclose()
            if delay:
                await asyncio.sleep(delay)
            return await self._client.send(self._client.build_request("GET", url), stream=True)
        return response

    async def _check_robots(self, origin: str, path: str) -> str:
        cached = self._robots.get(origin)
        if cached is not None:
            return robots_decision(status=cached[0], body=cached[1], path=path)
        current = origin + "/robots.txt"
        try:
            async with asyncio.timeout(ROBOTS_DEADLINE_SECONDS):
                for redirects in range(ROBOTS_MAX_REDIRECTS + 1):
                    response = await self._request_with_retry(current)
                    try:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            if redirects == ROBOTS_MAX_REDIRECTS or not response.headers.get("location"):
                                return RobotsDecision.FAIL_CLOSED
                            current = validate_public_url(urljoin(current, response.headers["location"]))
                            continue
                        status = response.status_code
                        body = await self._bounded_body(response, limit=ROBOTS_MAX_BYTES) if 200 <= status < 300 else b""
                        self._robots[origin] = (status, body)
                        return robots_decision(status=status, body=body, path=path)
                    finally:
                        await response.aclose()
        except (TimeoutError, httpx.HTTPError, RuntimeError, ValueError, zlib.error):
            return RobotsDecision.FAIL_CLOSED
        return RobotsDecision.FAIL_CLOSED

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urlsplit(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _robots_target(url: str) -> str:
        parsed = urlsplit(url)
        return (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")

    @staticmethod
    async def _read_raw(response: httpx.Response, limit: int) -> bytes:
        if response.is_stream_consumed:
            if len(response.content) > limit:
                raise RuntimeError("response_too_large")
            return response.content
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_raw():
            size += len(chunk)
            if size > limit:
                raise RuntimeError("response_too_large")
            chunks.append(chunk)
        return b"".join(chunks)

    async def _bounded_body(self, response: httpx.Response, *, limit: int = DIRECT_MAX_BODY_BYTES) -> bytes:
        if response.is_stream_consumed:
            if len(response.content) > limit:
                raise RuntimeError("decompressed_body_too_large")
            return response.content
        encoding = response.headers.get("content-encoding", "identity").casefold().strip()
        if encoding in {"", "identity"}:
            return await self._read_raw(response, limit)
        if encoding == "gzip":
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif encoding == "deflate":
            decoder = zlib.decompressobj()
        else:
            raise RuntimeError("content_encoding_forbidden")
        compressed = 0
        output: list[bytes] = []
        decompressed = 0
        async for chunk in response.aiter_raw():
            compressed += len(chunk)
            if compressed > limit:
                raise RuntimeError("compressed_body_too_large")
            value = decoder.decompress(chunk, limit - decompressed + 1)
            decompressed += len(value)
            if decompressed > limit:
                raise RuntimeError("decompressed_body_too_large")
            output.append(value)
        if not decoder.eof:
            raise RuntimeError("compressed_body_incomplete")
        if decoder.unused_data:
            raise RuntimeError("compressed_body_trailing_data")
        value = decoder.flush(limit - decompressed + 1)
        if decompressed + len(value) > limit:
            raise RuntimeError("decompressed_body_too_large")
        output.append(value)
        return b"".join(output)


class ResearchRetrievalPipeline:
    """Bounded search fan-out and direct-fetch stage; snippets never become evidence."""

    def __init__(self, search: SearXNGSearchAdapter, fetch: DirectHTTPFetcher) -> None:
        self._search = search
        self._fetch = fetch

    async def retrieve(self, subqueries: tuple[str, ...], *, language: str, page_limit: int = 8) -> RetrievalCorpusV1:
        if not 1 <= len(subqueries) <= 4 or not 1 <= page_limit <= 8:
            raise ValueError("retrieval plan bounds are invalid")
        search_results = await asyncio.gather(
            *(self._search.search(query, language=language, limit=page_limit) for query in subqueries),
            return_exceptions=True,
        )
        candidates: list[SearchCandidateV1] = []
        failures: list[str] = []
        seen: set[str] = set()
        for result in search_results:
            if isinstance(result, BaseException):
                failures.append("search_failed")
                continue
            for candidate in result:
                if candidate.requested_url not in seen:
                    candidates.append(candidate)
                    seen.add(candidate.requested_url)
                if len(candidates) == page_limit:
                    break
            if len(candidates) == page_limit:
                break
        semaphore = asyncio.Semaphore(4)

        async def one(candidate: SearchCandidateV1):
            async with semaphore:
                try:
                    return await self._fetch.fetch(candidate.requested_url)
                except Exception:
                    return "fetch_failed"

        results = await asyncio.gather(*(one(candidate) for candidate in candidates))
        fetched: list[FetchResultV1] = []
        for result in results:
            if isinstance(result, FetchResultV1):
                fetched.append(result)
            else:
                failures.append(result)
        return RetrievalCorpusV1(tuple(candidates), tuple(fetched), tuple(failures))
