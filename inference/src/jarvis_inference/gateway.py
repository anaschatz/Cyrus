from __future__ import annotations

import hashlib
import hmac
import queue
import re
import secrets
import select
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    NormalizedRequest,
    normalize_request,
    validate_generated_tool,
    validate_structured_content,
)
from .durable import read_report
from .engine import (
    Engine,
    EngineFailure,
    EngineProtocolFailure,
    GenerationEvent,
    PreparedPrompt,
)
from .errors import ApplianceError, backend_error
from .limits import (
    BLOCKED_WRITE_SECONDS,
    BODY_BYTES,
    BODY_SECONDS,
    CANCEL_SECONDS,
    OUTBOUND_BYTES,
    OUTBOUND_EVENTS,
    PREFILL_SECONDS,
    SSE_PAYLOAD_BYTES,
    STREAM_IDLE_SECONDS,
    TOTAL_SECONDS,
)
from .manifest import (
    Manifest,
    hash_file,
    resolve_local,
    runtime_identity,
    verify_artifacts,
)
from .profile_report import profile_report_matches
from .strict_json import canonical_bytes, parse_body

HOST = "127.0.0.1"
PORT = 8766
ALLOWED_HOSTS = {"127.0.0.1:8766", "localhost:8766"}
FORWARDED_HEADERS = {"origin", "forwarded"}


def package_digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def effective_configuration() -> dict[str, Any]:
    return {
        "host": HOST,
        "port": PORT,
        "single_active_request": True,
        "queue": False,
        # RFC 8785 serializes integral floats as JSON integers. Keep this
        # persisted qualification identity type-stable across a restart.
        "total_request_seconds": int(TOTAL_SECONDS),
        "prefill_seconds": int(PREFILL_SECONDS),
        "stream_idle_seconds": int(STREAM_IDLE_SECONDS),
        "blocked_write_seconds": int(BLOCKED_WRITE_SECONDS),
        "pre_output_retries": 1,
    }


class EventQueue:
    def __init__(self) -> None:
        self.events: deque[tuple[GenerationEvent | BaseException | object, int]] = deque()
        self.bytes = 0
        self.condition = threading.Condition()

    @staticmethod
    def _size(value: GenerationEvent | BaseException | object) -> int:
        if isinstance(value, GenerationEvent):
            return len(canonical_bytes({
                "kind": value.kind, "text": value.text, "tool_name": value.tool_name,
                "tool_arguments": value.tool_arguments, "finish_reason": value.finish_reason,
                "completion_tokens": value.completion_tokens,
            }))
        return 256

    def put(self, value: GenerationEvent | BaseException | object, cancel: threading.Event, deadline: float) -> None:
        size = self._size(value)
        if size > OUTBOUND_BYTES:
            value = EngineProtocolFailure("outbound event exceeds its byte limit")
            size = 256
        with self.condition:
            while not cancel.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                if len(self.events) < OUTBOUND_EVENTS and self.bytes + size <= OUTBOUND_BYTES:
                    self.events.append((value, size))
                    self.bytes += size
                    self.condition.notify_all()
                    return
                self.condition.wait(min(0.05, remaining))

    def get(self, timeout: float) -> GenerationEvent | BaseException | object:
        stop = time.monotonic() + timeout
        with self.condition:
            while not self.events:
                remaining = stop - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self.condition.wait(remaining)
            value, size = self.events.popleft()
            self.bytes -= size
            self.condition.notify_all()
            return value


END = object()


@dataclass(slots=True)
class Production:
    manifest: Manifest
    secret: str
    engine_factory: Any
    profile_validator: Any = profile_report_matches
    engine: Engine | None = None
    identity: dict[str, Any] | None = None
    admission: threading.Lock = field(init=False)
    engine_transition: threading.Lock = field(init=False)
    state_lock: threading.Lock = field(init=False)
    stop: threading.Event = field(init=False)
    loader: threading.Thread | None = field(init=False, default=None)
    ready_state: bool = field(init=False, default=False)
    loading_engine: Engine | None = field(init=False, default=None)
    teardown_proven: bool = field(init=False, default=False)
    loader_exited: threading.Event = field(init=False)
    loader_failure: BaseException | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.admission = threading.Lock()
        self.engine_transition = threading.Lock()
        self.state_lock = threading.Lock()
        self.stop = threading.Event()
        self.loader_exited = threading.Event()

    def refresh(self) -> bool:
        """Return the cached state; readiness handlers never load an engine."""
        return self.is_ready()

    def is_ready(self) -> bool:
        with self.state_lock:
            return bool(self.ready_state and self.engine is not None and self.engine.ready())

    def start_background(self) -> None:
        with self.state_lock:
            if self.loader is not None:
                return
            if self.stop.is_set():
                raise EngineFailure("service is shutting down")
            self.teardown_proven = False
            self.loader_failure = None
            self.loader_exited.clear()
            self.loader = threading.Thread(target=self._supervise, name="jarvis-inference-loader", daemon=True)
            self.loader.start()

    def _supervise(self) -> None:
        try:
            self._supervise_loop()
        except BaseException as exc:  # noqa: BLE001 - surfaced by close proof
            self.loader_failure = exc
        finally:
            self.loader_exited.set()

    def _supervise_loop(self) -> None:
        while not self.stop.is_set():
            current = runtime_identity(self.manifest, package_digest(), effective_configuration())
            report = read_report(self.manifest.qualification_report)
            valid = bool(
                report
                and report.get("report_schema_version") == 1
                and report.get("status") == "pass"
                and report.get("identity") == current
                and self._report_artifacts_match(report)
                and self.profile_validator(self.manifest, current, report)
            )
            with self.state_lock:
                active = self.engine
                loading = self.loading_engine
                identity_matches = self.identity == current
            if not valid:
                with self.state_lock:
                    self.ready_state = False
                    self.identity = None
                    active = self.engine
                    loading = self.loading_engine
                    self.engine = None
                    self.loading_engine = None
                if active is not None:
                    active.close()
                if loading is not None and loading is not active:
                    loading.close()
            elif loading is not None:
                pass
            elif active is None or not identity_matches or not active.ready():
                with self.engine_transition:
                    self._load_engine(current)
            else:
                with self.state_lock:
                    self.ready_state = True
            self.stop.wait(1.0)

    def _load_engine(self, current: dict[str, Any]) -> None:
        """Load at most one engine after rechecking transition-owned state."""

        with self.state_lock:
            active = self.engine
            loading = self.loading_engine
            identity_matches = self.identity == current
        if self.stop.is_set() or loading is not None:
            return
        if active is not None and identity_matches and active.ready():
            with self.state_lock:
                if self.engine is active:
                    self.ready_state = True
            return
        candidate = None
        try:
            if active is not None:
                breaker = getattr(active, "breaker", None)
                already_recorded = getattr(active, "restart_recorded", False)
                active.close()
                if breaker is not None and not already_recorded:
                    breaker.record_restart()
                with self.state_lock:
                    if self.engine is active:
                        self.engine = None
                        self.ready_state = False
            if self.stop.is_set():
                return
            candidate = self.engine_factory(self.manifest)
            with self.state_lock:
                shutting_down = self.stop.is_set()
                if not shutting_down:
                    self.loading_engine = candidate
            if shutting_down:
                candidate.close()
                return
            start = getattr(candidate, "start", None)
            if callable(start):
                start(time.monotonic() + 600.0)
            if self.stop.is_set():
                candidate.close()
                with self.state_lock:
                    if self.loading_engine is candidate:
                        self.loading_engine = None
                return
            if not candidate.ready():
                raise EngineFailure("engine did not become ready")
            with self.state_lock:
                if self.stop.is_set():
                    publish = False
                else:
                    self.engine = candidate
                    self.identity = current
                    self.ready_state = True
                    publish = True
                if self.loading_engine is candidate:
                    self.loading_engine = None
            if not publish:
                candidate.close()
        except (ApplianceError, EngineFailure, OSError):
            with self.state_lock:
                if candidate is not None and self.loading_engine is candidate:
                    self.loading_engine = None
                self.ready_state = False
            if candidate is not None:
                candidate.close()

    def _report_artifacts_match(self, report: dict[str, Any]) -> bool:
        runner = report.get("fixture_runner")
        if runner is None:
            return False
        if not isinstance(runner, dict) or set(runner) != {"path", "size_bytes", "sha256"}:
            return False
        try:
            path = resolve_local(self.manifest.root, runner["path"], "fixture_runner")
            size, digest = hash_file(path)
            return type(runner["size_bytes"]) is int and size == runner["size_bytes"] and digest == runner["sha256"]
        except (ApplianceError, OSError, TypeError):
            return False

    def close(self) -> None:
        self.stop.set()
        self.teardown_proven = False
        failures: list[BaseException] = []
        active, loading = self._detach_engines()
        self._close_detached(active, loading, failures)
        loader = self.loader
        if loader is threading.current_thread():
            failures.append(EngineFailure("loader cannot prove its own shutdown"))
        elif loader is not None:
            loader.join(timeout=10.0)
            if loader.is_alive() or not self.loader_exited.is_set():
                failures.append(EngineFailure("loader shutdown was not proven"))
        if self.loader_failure is not None:
            failures.append(self.loader_failure)
        # A candidate can be created or published after the first detach but
        # before the loader observes stop.  Re-drain only after the join proof.
        late_active, late_loading = self._detach_engines()
        self._close_detached(late_active, late_loading, failures)
        loader_stopped = loader is None or (
            loader is not threading.current_thread()
            and not loader.is_alive()
            and self.loader_exited.is_set()
        )
        with self.state_lock:
            empty = self.engine is None and self.loading_engine is None
            self.teardown_proven = bool(loader_stopped and empty and not failures)
        if failures:
            raise EngineFailure("inference teardown could not be proven") from failures[0]

    def _detach_engines(self) -> tuple[Engine | None, Engine | None]:
        with self.state_lock:
            self.ready_state = False
            self.identity = None
            active = self.engine
            loading = self.loading_engine
            self.engine = None
            self.loading_engine = None
        return active, loading

    @staticmethod
    def _close_detached(
        active: Engine | None,
        loading: Engine | None,
        failures: list[BaseException],
    ) -> None:
        for engine in (active, loading if loading is not active else None):
            if engine is None:
                continue
            try:
                engine.close()
            except BaseException as exc:  # noqa: BLE001 - drain all owned engines
                failures.append(exc)

    def require_teardown_proof(self) -> None:
        loader = self.loader
        with self.state_lock:
            proven = bool(
                self.teardown_proven
                and self.engine is None
                and self.loading_engine is None
                and (
                    loader is None
                    or (not loader.is_alive() and self.loader_exited.is_set())
                )
            )
        if not proven:
            raise EngineFailure("inference teardown was not proven")

    def restart_engine(self, deadline: float) -> Engine:
        with self.engine_transition:
            return self._restart_engine(deadline)

    def _restart_engine(self, deadline: float) -> Engine:
        with self.state_lock:
            active = self.engine
            self.engine = None
            self.ready_state = False
        if active is not None:
            breaker = getattr(active, "breaker", None)
            already_recorded = getattr(active, "restart_recorded", False)
            active.close()
            if breaker is not None and not already_recorded:
                breaker.record_restart()
        engine = self.engine_factory(self.manifest)
        with self.state_lock:
            if self.stop.is_set():
                engine.close()
                raise EngineFailure("service is shutting down")
            self.loading_engine = engine
        try:
            start = getattr(engine, "start", None)
            if callable(start):
                start(deadline)
            with self.state_lock:
                if self.stop.is_set():
                    engine.close()
                    raise EngineFailure("service is shutting down")
                self.engine = engine
                self.loading_engine = None
                self.ready_state = engine.ready()
            return engine
        except Exception:
            with self.state_lock:
                if self.loading_engine is engine:
                    self.loading_engine = None
            engine.close()
            raise


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "JarvisInference/1"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    @property
    def production(self) -> Production:
        return self.server.production  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        # Deliberately content-free. Request paths can contain user data and are
        # therefore not logged by this appliance.
        return

    def do_GET(self) -> None:
        try:
            self._admit_peer_host()
            if self.path == "/readyz":
                ready = self.production.refresh()
                self._json(200 if ready else 503, {"ready": ready}, extra={"Cache-Control": "no-store"})
                return
            self._authenticate()
            runtime = getattr(self.production, "qualification_runtime", None)
            if self.path == "/jarvis/qualification/runtime" and callable(runtime):
                self._json(200, runtime())
                return
            if self.path == "/v1/models":
                self._require_ready()
                self._json(200, {"object": "list", "data": [{"id": self.production.manifest.alias, "object": "model", "owned_by": "jarvis-local"}]})
                return
            if self.path == "/jarvis/v1/capabilities":
                self._require_ready()
                caps = self.production.manifest.capabilities
                self._json(200, {"model": self.production.manifest.alias, "capabilities": {
                    "text": caps.text, "stream": caps.stream, "tools": caps.tools,
                    "json_object": caps.json_object, "json_schema": caps.json_schema,
                }})
                return
            if self.path == "/jarvis/v1/identity":
                self._require_ready()
                identity = self.production.identity
                if not isinstance(identity, dict):
                    raise ApplianceError(
                        503,
                        "service_unavailable",
                        "The inference appliance is not ready.",
                    )
                self._json(
                    200,
                    {
                        "manifest_digest": self.production.manifest.digest,
                        "runtime_identity_sha256": hashlib.sha256(
                            canonical_bytes(identity)
                        ).hexdigest(),
                        "schema_version": "jarvis-inference-live-identity/1",
                    },
                )
                return
            raise ApplianceError(404, "not_found", "The endpoint does not exist.")
        except ApplianceError as exc:
            self._error(exc)

    def do_POST(self) -> None:
        started = time.monotonic()
        try:
            self._admit_peer_host()
            self._authenticate()
            inject_cancel = getattr(self.production, "inject_qualification_cancel", None)
            if self.path == "/jarvis/qualification/cancel" and callable(inject_cancel):
                self._media_headers()
                document = parse_body(self._read_body(started))
                if not isinstance(document, dict) or set(document) != {"phase"} or document["phase"] not in {"prefill", "stream"}:
                    raise ApplianceError(400, "invalid_qualification_control", "The qualification control payload is invalid.")
                self._json(200, inject_cancel(document["phase"]))
                return
            inject_crash = getattr(self.production, "inject_qualification_crash", None)
            if self.path == "/jarvis/qualification/crash" and callable(inject_crash):
                self._media_headers()
                document = parse_body(self._read_body(started))
                if document != {}:
                    raise ApplianceError(400, "invalid_qualification_control", "The qualification control payload is invalid.")
                self._json(200, inject_crash())
                return
            arm_generation = getattr(
                self.production, "qualification_arm_generation", None
            )
            if (
                self.path == "/jarvis/qualification/generation/arm"
                and callable(arm_generation)
            ):
                self._media_headers()
                document = parse_body(self._read_body(started))
                if document != {}:
                    raise ApplianceError(
                        400,
                        "invalid_qualification_control",
                        "The qualification control payload is invalid.",
                    )
                self._json(200, arm_generation())
                return
            generation_metric = getattr(
                self.production, "qualification_generation_metric", None
            )
            if (
                self.path == "/jarvis/qualification/generation/metric"
                and callable(generation_metric)
            ):
                self._media_headers()
                document = parse_body(self._read_body(started))
                if (
                    not isinstance(document, dict)
                    or set(document) != {"expected_serial"}
                    or type(document["expected_serial"]) is not int
                    or document["expected_serial"] <= 0
                ):
                    raise ApplianceError(
                        400,
                        "invalid_qualification_control",
                        "The qualification control payload is invalid.",
                    )
                self._json(
                    200, generation_metric(document["expected_serial"])
                )
                return
            tokenize = getattr(self.production, "qualification_tokenize", None)
            if self.path == "/jarvis/qualification/tokenize" and callable(tokenize):
                self._media_headers()
                document = parse_body(self._read_body(started))
                request = normalize_request(
                    document,
                    alias=self.production.manifest.alias,
                    capabilities=self.production.manifest.capabilities,
                    context_window=self.production.manifest.context_window,
                )
                self._json(200, tokenize(request))
                return
            if self.path != "/v1/chat/completions":
                raise ApplianceError(404, "not_found", "The endpoint does not exist.")
            self._media_headers()
            raw = self._read_body(started)
            document = parse_body(raw)
            request = normalize_request(
                document,
                alias=self.production.manifest.alias,
                capabilities=self.production.manifest.capabilities,
                context_window=self.production.manifest.context_window,
            )
            self._require_ready()
            if not self.production.admission.acquire(blocking=False):
                raise ApplianceError(429, "server_busy", "The inference appliance is busy.")
            try:
                bind_request_start = getattr(
                    self.production, "qualification_bind_request_start", None
                )
                if callable(bind_request_start):
                    bind_request_start(started)
                self._complete(request, started)
            finally:
                self.production.admission.release()
        except ApplianceError as exc:
            self._error(exc, retry_after=exc.code == "server_busy")
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return

    def _unsupported_method(self) -> None:
        try:
            self._admit_peer_host()
            self._authenticate()
            raise ApplianceError(404, "not_found", "The endpoint does not exist.")
        except ApplianceError as exc:
            self._error(exc)

    do_PUT = _unsupported_method
    do_PATCH = _unsupported_method
    do_DELETE = _unsupported_method
    do_OPTIONS = _unsupported_method
    do_HEAD = _unsupported_method

    def _admit_peer_host(self) -> None:
        try:
            if not socket.inet_pton(socket.AF_INET, self.client_address[0]) or self.client_address[0] != HOST:
                raise ValueError
        except (OSError, ValueError):
            raise ApplianceError(403, "forbidden_peer", "Only direct IPv4 loopback clients are permitted.")
        hosts = self.headers.get_all("Host", failobj=[])
        allowed_hosts = getattr(self.server, "allowed_hosts", ALLOWED_HOSTS)
        if len(hosts) != 1 or hosts[0].lower() not in allowed_hosts:
            raise ApplianceError(400, "invalid_host", "The Host header is invalid.")
        for name in self.headers.keys():
            lowered = name.lower()
            if lowered in FORWARDED_HEADERS or lowered.startswith("x-forwarded-"):
                raise ApplianceError(400, "forwarded_request_forbidden", "Forwarded or browser-origin requests are not permitted.")

    def _authenticate(self) -> None:
        values = self.headers.get_all("Authorization", failobj=[])
        expected = "Bearer " + self.production.secret
        if len(values) != 1 or not hmac.compare_digest(values[0], expected):
            raise ApplianceError(401, "invalid_api_key", "Authentication failed.")

    def _media_headers(self) -> None:
        types = self.headers.get_all("Content-Type", failobj=[])
        if len(types) != 1:
            raise ApplianceError(415, "unsupported_media_type", "Exactly one supported Content-Type is required.")
        segments = [segment.strip().lower() for segment in types[0].split(";")]
        if not segments or segments[0] != "application/json" or len(segments) > 2 or (len(segments) == 2 and segments[1] != "charset=utf-8"):
            raise ApplianceError(415, "unsupported_media_type", "Exactly one supported Content-Type is required.")
        encodings = self.headers.get_all("Content-Encoding", failobj=[])
        if len(encodings) > 1 or (encodings and encodings[0].strip().lower() != "identity"):
            raise ApplianceError(415, "unsupported_content_encoding", "Compressed request bodies are not supported.")
        if self.headers.get_all("Transfer-Encoding", failobj=[]):
            raise ApplianceError(415, "unsupported_content_encoding", "Transfer encodings are not supported.")

    def _read_body(self, started: float) -> bytes:
        values = self.headers.get_all("Content-Length", failobj=[])
        if len(values) != 1:
            raise ApplianceError(411, "length_required", "Exactly one Content-Length header is required.")
        try:
            if not re.fullmatch(r"[0-9]+", values[0]):
                raise ValueError
            length = int(values[0], 10)
        except ValueError as exc:
            raise ApplianceError(400, "invalid_content_length", "Content-Length is invalid.") from exc
        if length < 0 or length > BODY_BYTES:
            raise ApplianceError(413, "request_too_large", "The request body exceeds its size limit.")
        self.connection.settimeout(min(BODY_SECONDS, max(0.001, started + TOTAL_SECONDS - time.monotonic())))
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ApplianceError(400, "incomplete_body", "The request body is incomplete.")
        return raw

    def _require_ready(self) -> None:
        if not self.production.is_ready():
            raise ApplianceError(503, "service_unavailable", "The inference appliance is not ready.")

    def _complete(self, request: NormalizedRequest, started: float) -> None:
        deadline = started + TOTAL_SECONDS
        self._response_deadline = deadline
        engine = self.production.engine
        if engine is None:
            raise ApplianceError(503, "service_unavailable", "The inference appliance is not ready.")
        cancel = threading.Event()
        last_error: BaseException | None = None
        for attempt in range(2):
            disconnected = threading.Event()
            watcher = threading.Thread(target=self._watch_disconnect, args=(cancel, disconnected), daemon=True)
            watcher.start()
            if cancel.is_set() or time.monotonic() >= deadline:
                raise ApplianceError(504, "request_timeout", "The request deadline expired.")
            try:
                prepared = engine.prepare(request, min(deadline, time.monotonic() + PREFILL_SECONDS))
                if len(prepared.token_ids) + request.max_completion_tokens > self.production.manifest.context_window:
                    raise ApplianceError(400, "context_length_exceeded", "The rendered prompt exceeds the qualified context window.")
                events = EventQueue()
                producer = threading.Thread(target=self._produce, args=(engine, prepared, cancel, deadline, events), daemon=True)
                producer.start()
                first = events.get(timeout=max(0.001, min(PREFILL_SECONDS, deadline - time.monotonic())))
                if isinstance(first, BaseException):
                    raise first
                if first is END or not isinstance(first, GenerationEvent):
                    raise EngineProtocolFailure("backend produced no valid generation event")
                self._validate_event(first)
                self._validate_first_for_request(first, request)
                if request.stream:
                    self._stream(request, prepared, first, events, engine, cancel, deadline)
                else:
                    self._nonstream(request, prepared, first, events, engine, cancel, deadline)
                cancel.set()
                return
            except ApplianceError as exc:
                cancel.set()
                if exc.status >= 500:
                    try:
                        engine.cancel()
                        engine.cleanup(min(deadline, time.monotonic() + CANCEL_SECONDS))
                    except Exception:
                        pass
                raise
            except (EngineFailure, queue.Empty) as exc:
                last_error = exc
                cancel.set()
                try:
                    engine.cancel()
                    engine.cleanup(min(deadline, time.monotonic() + CANCEL_SECONDS))
                except Exception:
                    pass
                if attempt == 0 and not isinstance(exc, EngineProtocolFailure) and not disconnected.is_set():
                    cancel = threading.Event()
                    engine = self.production.restart_engine(min(deadline, time.monotonic() + PREFILL_SECONDS))
                    continue
                raise backend_error() from exc
        raise backend_error() from last_error

    @staticmethod
    def _produce(engine: Engine, prepared: PreparedPrompt, cancel: threading.Event, deadline: float, events: EventQueue) -> None:
        try:
            for event in engine.generate(prepared, cancel, deadline):
                events.put(event, cancel, deadline)
            events.put(END, cancel, deadline)
        except Exception as exc:
            events.put(exc, cancel, deadline)

    @staticmethod
    def _validate_event(event: GenerationEvent) -> None:
        if event.kind not in {"text", "tool", "final"}:
            raise EngineProtocolFailure("unknown backend event")
        if event.kind == "text" and (not isinstance(event.text, str) or not event.text):
            raise EngineProtocolFailure("empty or invalid backend text event")
        if event.kind == "tool" and (not event.tool_name or event.tool_arguments is None):
            raise EngineProtocolFailure("invalid backend tool event")
        if event.kind == "final":
            if event.finish_reason not in {"stop", "length", "tool_calls"} or type(event.completion_tokens) is not int or event.completion_tokens < 0:
                raise EngineProtocolFailure("invalid backend terminal event")

    @staticmethod
    def _validate_first_for_request(event: GenerationEvent, request: NormalizedRequest) -> None:
        required = request.tool_choice == "required" or isinstance(request.tool_choice, tuple)
        if required and event.kind != "tool":
            raise EngineProtocolFailure("a required tool request did not begin with a complete tool call")
        if event.kind == "tool":
            if request.tool_choice == "none":
                raise EngineProtocolFailure("the backend emitted a forbidden tool call")
            validate_generated_tool(event.tool_name or "", event.tool_arguments or "", request)

    def _drain(self, first: GenerationEvent, events: EventQueue, deadline: float) -> Iterable[GenerationEvent]:
        terminal = first.kind == "final"
        yield first
        while True:
            remaining = min(STREAM_IDLE_SECONDS, deadline - time.monotonic())
            if remaining <= 0:
                raise EngineProtocolFailure("request or stream-idle deadline expired")
            try:
                event = events.get(timeout=remaining)
            except queue.Empty as exc:
                raise EngineProtocolFailure("stream-idle deadline expired") from exc
            if isinstance(event, BaseException):
                raise event
            if event is END:
                if not terminal:
                    raise EngineProtocolFailure("backend ended before one terminal event")
                return
            if not isinstance(event, GenerationEvent):
                raise EngineProtocolFailure("invalid backend event")
            self._validate_event(event)
            if terminal:
                raise EngineProtocolFailure("backend emitted an event after its terminal event")
            terminal = event.kind == "final"
            yield event

    def _nonstream(self, request: NormalizedRequest, prepared: PreparedPrompt, first: GenerationEvent, events: EventQueue, engine: Engine, cancel: threading.Event, deadline: float) -> None:
        text: list[str] = []
        tool: GenerationEvent | None = None
        final: GenerationEvent | None = None
        for event in self._drain(first, events, deadline):
            if event.kind == "text":
                text.append(event.text)
            elif event.kind == "tool":
                if tool is not None:
                    raise EngineProtocolFailure("multiple tool calls are unsupported")
                tool = event
            else:
                final = event
        if final is None:
            raise EngineProtocolFailure("missing terminal event")
        required = request.tool_choice == "required" or isinstance(request.tool_choice, tuple)
        if required and tool is None:
            raise EngineProtocolFailure("the required tool call is missing")
        if (tool is not None) != (final.finish_reason == "tool_calls") or (tool is not None and text):
            raise EngineProtocolFailure("tool output has an inconsistent finish reason or mixed content")
        if tool is not None:
            canonical_arguments = validate_generated_tool(tool.tool_name or "", tool.tool_arguments or "", request)
            tool = GenerationEvent("tool", tool_name=tool.tool_name, tool_arguments=canonical_arguments)
        content = validate_structured_content("".join(text), request) if tool is None else None
        # The validated terminal event means generation has ended. From here
        # cleanup must run to completion even if the client closes immediately.
        cancel.set()
        engine.cleanup(deadline)
        response_id = "chatcmpl-" + secrets.token_hex(16)
        created = int(time.time())
        if tool is None:
            message: dict[str, Any] = {"role": "assistant", "content": content}
        else:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_" + secrets.token_hex(16), "type": "function",
                "function": {"name": tool.tool_name, "arguments": tool.tool_arguments},
            }]}
        body = {
            "id": response_id, "object": "chat.completion", "created": created,
            "model": request.model,
            "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": final.finish_reason}],
            "usage": self._usage(prepared, final),
        }
        if time.monotonic() >= deadline:
            raise EngineFailure("request deadline expired before final write")
        self._json(200, body)

    def _stream(self, request: NormalizedRequest, prepared: PreparedPrompt, first: GenerationEvent, events: EventQueue, engine: Engine, cancel: threading.Event, deadline: float) -> None:
        response_id = "chatcmpl-" + secrets.token_hex(16)
        call_id = "call_" + secrets.token_hex(16)
        created = int(time.time())
        self.connection.settimeout(BLOCKED_WRITE_SECONDS)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self._chunk(response_id, created, request, {"role": "assistant", "content": ""}, None)
            final: GenerationEvent | None = None
            tool_seen = False
            text_seen = False
            for event in self._drain(first, events, deadline):
                if event.kind == "text":
                    if tool_seen:
                        raise EngineProtocolFailure("mixed text and tool output")
                    text_seen = True
                    self._chunk(response_id, created, request, {"content": event.text}, None)
                elif event.kind == "tool":
                    if tool_seen or text_seen or request.tool_choice == "none":
                        raise EngineProtocolFailure("invalid tool output cardinality")
                    tool_seen = True
                    arguments = validate_generated_tool(event.tool_name or "", event.tool_arguments or "", request)
                    self._chunk(response_id, created, request, {"tool_calls": [{
                        "index": 0, "id": call_id, "type": "function",
                        "function": {"name": event.tool_name, "arguments": arguments},
                    }]}, None)
                else:
                    final = event
            if final is None:
                raise EngineProtocolFailure("missing terminal backend event")
            required = request.tool_choice == "required" or isinstance(request.tool_choice, tuple)
            if required and not tool_seen:
                raise EngineProtocolFailure("the required tool call is missing")
            if tool_seen != (final.finish_reason == "tool_calls"):
                raise EngineProtocolFailure("tool output has an inconsistent finish reason")
            cancel.set()
            engine.cleanup(deadline)
            self._chunk(response_id, created, request, {}, final.finish_reason)
            if request.include_usage:
                self._sse({
                    "id": response_id, "object": "chat.completion.chunk", "created": created,
                    "model": request.model, "choices": [], "usage": self._usage(prepared, final),
                })
            if time.monotonic() >= deadline:
                raise EngineFailure("request deadline expired before stream completion")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            cancel.set()
            engine.cancel()
            self._abort()
            return
        finally:
            self.close_connection = True

    def _chunk(self, response_id: str, created: int, request: NormalizedRequest, delta: dict[str, Any], finish: str | None) -> None:
        value: dict[str, Any] = {
            "id": response_id, "object": "chat.completion.chunk", "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
        }
        if request.include_usage:
            value["usage"] = None
        self._sse(value)

    def _sse(self, value: dict[str, Any]) -> None:
        if time.monotonic() >= getattr(self, "_response_deadline", float("inf")):
            raise EngineFailure("request deadline expired during stream write")
        payload = canonical_bytes(value)
        if len(payload) > SSE_PAYLOAD_BYTES:
            raise EngineProtocolFailure("normalized SSE payload exceeds its limit")
        self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.flush()

    @staticmethod
    def _usage(prepared: PreparedPrompt, final: GenerationEvent) -> dict[str, int]:
        completion = final.completion_tokens
        if type(completion) is not int or completion < 0:
            raise EngineProtocolFailure("invalid completion token accounting")
        if completion > prepared.request.max_completion_tokens:
            raise EngineProtocolFailure("completion token accounting exceeds the admitted limit")
        prompt = len(prepared.token_ids)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}

    def _watch_disconnect(self, cancel: threading.Event, disconnected: threading.Event) -> None:
        while not cancel.is_set():
            try:
                readable, _, _ = select.select([self.connection], [], [], 0.05)
                if cancel.is_set():
                    return
                if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                    disconnected.set()
                    cancel.set()
                    engine = self.production.engine
                    if engine is not None:
                        engine.cancel()
                    return
            except (OSError, ValueError):
                if cancel.is_set():
                    return
                disconnected.set()
                cancel.set()
                engine = self.production.engine
                if engine is not None:
                    engine.cancel()
                return

    def _json(self, status: int, value: dict[str, Any], *, extra: dict[str, str] | None = None) -> None:
        body = canonical_bytes(value)
        self.connection.settimeout(BLOCKED_WRITE_SECONDS)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, item in (extra or {}).items():
            if name.lower() != "cache-control":
                self.send_header(name, item)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _error(self, error: ApplianceError, *, retry_after: bool = False) -> None:
        try:
            extra = {"Retry-After": "1"} if retry_after else None
            self._json(error.status, error.public(), extra=extra)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _abort(self) -> None:
        self.close_connection = True
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.connection.close()
        except OSError:
            pass


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, production: Production, address: tuple[str, int] = (HOST, PORT), allowed_hosts: set[str] | None = None):
        super().__init__(address, GatewayHandler, bind_and_activate=True)
        self.production = production
        actual_port = int(self.server_address[1])
        self.allowed_hosts = allowed_hosts or ({"127.0.0.1:" + str(actual_port), "localhost:" + str(actual_port)} if actual_port != PORT else ALLOWED_HOSTS)
        production.start_background()

    def handle_error(self, request: Any, client_address: Any) -> None:
        # socketserver's default prints tracebacks for expected disconnects.
        # The appliance keeps process/request diagnostics content-free.
        return


def serve(production: Production) -> None:
    server = GatewayServer(production)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        production.close()
