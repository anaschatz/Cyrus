"""Authenticated, bounded HTTP API for the model-free Research sidecar."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import httpx

from jarvis_orchestrator.domain import canonical_json
from jarvis_orchestrator.research import ResearchCorpusV1, ResearchRequestV1, SearchFacetV1, SearchPlanV1
from jarvis_orchestrator.research_security import RetryBudgetV1, reject_sensitive_text

from .adapters import DirectHTTPFetcher, ResearchRetrievalPipeline, SearXNGSearchAdapter
from .config import RuntimeConfigV1
from .corpus import CorpusBuilderV1

MAX_REQUEST_BYTES = 32 * 1024
MAX_IDEMPOTENCY_KEY_BYTES = 128
MAX_CACHE_ENTRIES = 128
JSON_CONTENT_TYPE = "application/json"


class APIError(Exception):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


def _strict_json(raw: bytes) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate_field")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite_number")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise APIError(HTTPStatus.BAD_REQUEST, "request_json_invalid") from exc
    if not isinstance(value, dict):
        raise APIError(HTTPStatus.BAD_REQUEST, "request_schema_invalid")
    return value


def parse_research_request(raw: bytes) -> ResearchRequestV1:
    value = _strict_json(raw)
    expected = {
        "task_id", "conversation_id", "epoch", "plan", "deadline_seconds",
        "max_documents", "max_passages", "max_claims", "span_selection_output_tokens",
    }
    if set(value) != expected or not isinstance(value.get("plan"), dict):
        raise APIError(HTTPStatus.BAD_REQUEST, "request_schema_invalid")
    plan_value = value["plan"]
    assert isinstance(plan_value, dict)
    plan_expected = {
        "canonical_query", "facets", "subqueries", "language", "freshness", "timezone",
        "page_limit", "deadline_seconds", "browser_fallback_allowed", "version",
    }
    if set(plan_value) != plan_expected or not isinstance(plan_value.get("facets"), list) or not isinstance(plan_value.get("subqueries"), list):
        raise APIError(HTTPStatus.BAD_REQUEST, "request_schema_invalid")
    facets: list[SearchFacetV1] = []
    for item in plan_value["facets"]:
        if not isinstance(item, dict) or set(item) != {"facet_id", "label"}:
            raise APIError(HTTPStatus.BAD_REQUEST, "request_schema_invalid")
        try:
            facets.append(SearchFacetV1(**item))
        except (TypeError, ValueError) as exc:
            raise APIError(HTTPStatus.BAD_REQUEST, "request_schema_invalid") from exc
    try:
        plan = SearchPlanV1(
            canonical_query=plan_value["canonical_query"],
            facets=tuple(facets),
            subqueries=tuple(plan_value["subqueries"]),
            language=plan_value["language"],
            freshness=plan_value["freshness"],
            timezone=plan_value["timezone"],
            page_limit=plan_value["page_limit"],
            deadline_seconds=plan_value["deadline_seconds"],
            browser_fallback_allowed=plan_value["browser_fallback_allowed"],
            version=plan_value["version"],
        )
        if plan.browser_fallback_allowed:
            raise ValueError("browser is not qualified")
        reject_sensitive_text(plan.canonical_query)
        for query in plan.subqueries:
            reject_sensitive_text(query)
        return ResearchRequestV1(
            task_id=value["task_id"],
            conversation_id=value["conversation_id"],
            epoch=value["epoch"],
            plan=plan,
            deadline_seconds=value["deadline_seconds"],
            max_documents=value["max_documents"],
            max_passages=value["max_passages"],
            max_claims=value["max_claims"],
            span_selection_output_tokens=value["span_selection_output_tokens"],
        )
    except (TypeError, ValueError) as exc:
        raise APIError(HTTPStatus.BAD_REQUEST, "request_schema_invalid") from exc


def validate_post_envelope(
    *, content_type: str | None, transfer_encoding: str | None, content_length: str | None,
    idempotency_key: str | None,
) -> tuple[int, str]:
    if transfer_encoding is not None:
        raise APIError(HTTPStatus.BAD_REQUEST, "request_framing_invalid")
    if (content_type or "").casefold() != JSON_CONTENT_TYPE:
        raise APIError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "content_type_invalid")
    key = idempotency_key
    if key is None or not key.isascii() or not 1 <= len(key.encode("ascii")) <= MAX_IDEMPOTENCY_KEY_BYTES or any(ord(ch) < 33 or ord(ch) > 126 for ch in key):
        raise APIError(HTTPStatus.BAD_REQUEST, "idempotency_key_invalid")
    try:
        length = int(content_length or "")
    except ValueError as exc:
        raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_size_invalid") from exc
    if not 1 <= length <= MAX_REQUEST_BYTES:
        raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_size_invalid")
    return length, key


class ResearchEngineV1:
    """One bounded network pipeline; deliberately contains no model port."""

    def __init__(self, config: RuntimeConfigV1, *, builder: CorpusBuilderV1 | None = None) -> None:
        self._config = config
        self._builder = builder or CorpusBuilderV1()

    async def research(self, request: ResearchRequestV1) -> ResearchCorpusV1:
        retry = RetryBudgetV1()
        proxy_headers = {"Proxy-Authorization": "Bearer " + self._config.proxy_credentials}
        search_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(5.0, connect=3.0),
            trust_env=False,
        )
        fetcher = DirectHTTPFetcher.with_egress_proxy(
            self._config.egress_proxy_url, proxy_headers=proxy_headers, retry_budget=retry
        )
        try:
            search = SearXNGSearchAdapter(search_client, endpoint=self._config.searxng_url)
            pipeline = ResearchRetrievalPipeline(search, fetcher)
            async with asyncio.timeout(request.deadline_seconds):
                retrieval = await pipeline.retrieve(
                    request.plan.subqueries,
                    language=request.plan.language,
                    page_limit=request.plan.page_limit,
                )
            return self._builder.build(request.plan, retrieval, network_retry_count=int(retry.used))
        finally:
            await search_client.aclose()
            await fetcher.aclose()


class IdempotencyCacheV1:
    def __init__(self, maximum: int = MAX_CACHE_ENTRIES) -> None:
        self._maximum = maximum
        self._values: OrderedDict[str, tuple[str, bytes]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, request_hash: str) -> bytes | None:
        with self._lock:
            value = self._values.get(key)
            if value is None:
                return None
            if value[0] != request_hash:
                raise APIError(HTTPStatus.CONFLICT, "idempotency_conflict")
            self._values.move_to_end(key)
            return value[1]

    def put(self, key: str, request_hash: str, response: bytes) -> None:
        with self._lock:
            existing = self._values.get(key)
            if existing is not None and existing[0] != request_hash:
                raise APIError(HTTPStatus.CONFLICT, "idempotency_conflict")
            self._values[key] = (request_hash, response)
            self._values.move_to_end(key)
            while len(self._values) > self._maximum:
                self._values.popitem(last=False)


class ResearchApplicationV1:
    def __init__(
        self,
        config: RuntimeConfigV1,
        *,
        engine: ResearchEngineV1 | None = None,
        audit: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.config = config
        self.engine = engine or ResearchEngineV1(config)
        self.cache = IdempotencyCacheV1()
        self._active = threading.BoundedSemaphore(4)
        self._audit = audit or (lambda value: logging.getLogger("jarvis_research.audit").info(canonical_json(value)))

    def authenticated(self, header: str | None) -> bool:
        expected = "Bearer " + self.config.bearer_token
        return header is not None and hmac.compare_digest(header, expected)

    def qualification(self) -> bytes:
        return canonical_json(self.config.qualification.as_dict()).encode("utf-8")

    def execute(self, raw: bytes, idempotency_key: str) -> bytes:
        started = time.monotonic()
        request_hash = hashlib.sha256(raw).hexdigest()
        cached = self.cache.get(idempotency_key, request_hash)
        if cached is not None:
            self._emit("research_cache_hit", started, request_hash, 0)
            return cached
        if not self._active.acquire(blocking=False):
            raise APIError(HTTPStatus.TOO_MANY_REQUESTS, "research_capacity_exceeded")
        try:
            request = parse_research_request(raw)
            corpus = asyncio.run(self.engine.research(request))
            response = canonical_json({"corpus": asdict(corpus)}).encode("utf-8")
            self.cache.put(idempotency_key, request_hash, response)
            self._emit("research_complete", started, request_hash, len(corpus.sources))
            return response
        except TimeoutError as exc:
            self._emit("research_deadline", started, request_hash, 0)
            raise APIError(HTTPStatus.GATEWAY_TIMEOUT, "research_deadline") from exc
        finally:
            self._active.release()

    def _emit(self, code: str, started: float, request_hash: str, count: int) -> None:
        self._audit(
            {
                "code": code,
                "count": count,
                "duration_ms": max(0, int((time.monotonic() - started) * 1000)),
                "request_sha256": request_hash,
            }
        )


class ResearchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: ResearchApplicationV1) -> None:
        self.app = app
        super().__init__(address, ResearchRequestHandler)


class ResearchRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "JARVISResearch/1"
    sys_version = ""

    @property
    def app(self) -> ResearchApplicationV1:
        return self.server.app  # type: ignore[attr-defined,no-any-return]

    def log_message(self, *_: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._reply(HTTPStatus.OK, b'{"status":"ok"}')
            return
        if self.path != "/v1/qualification":
            self._error(HTTPStatus.NOT_FOUND, "not_found")
            return
        if not self.app.authenticated(self.headers.get("Authorization")):
            self._error(HTTPStatus.UNAUTHORIZED, "authentication_required")
            return
        self._reply(HTTPStatus.OK, self.app.qualification())

    def do_POST(self) -> None:
        if self.path != "/v1/research":
            self._error(HTTPStatus.NOT_FOUND, "not_found")
            return
        if not self.app.authenticated(self.headers.get("Authorization")):
            self._error(HTTPStatus.UNAUTHORIZED, "authentication_required")
            return
        try:
            length, key = validate_post_envelope(
                content_type=self.headers.get("Content-Type"),
                transfer_encoding=self.headers.get("Transfer-Encoding"),
                content_length=self.headers.get("Content-Length"),
                idempotency_key=self.headers.get("Idempotency-Key"),
            )
        except APIError as exc:
            self._error(exc.status, exc.code)
            return
        raw = self.rfile.read(length)
        if len(raw) != length:
            self._error(HTTPStatus.BAD_REQUEST, "request_body_incomplete")
            return
        try:
            response = self.app.execute(raw, key)
        except APIError as exc:
            self._error(exc.status, exc.code)
            return
        except Exception:
            self._error(HTTPStatus.BAD_GATEWAY, "research_pipeline_failed")
            return
        self._reply(HTTPStatus.OK, response)

    def _error(self, status: int, code: str) -> None:
        self._reply(status, canonical_json({"error": {"code": code}}).encode("utf-8"))

    def _reply(self, status: int, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", JSON_CONTENT_TYPE)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A cancelled caller must not turn into an unsanitized server traceback.
            return


def serve(config: RuntimeConfigV1 | None = None) -> None:
    runtime = config or RuntimeConfigV1.from_environment()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # httpx/httpcore INFO records include request URLs and are forbidden here.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    server = ResearchHTTPServer((runtime.bind_host, runtime.bind_port), ResearchApplicationV1(runtime))
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
