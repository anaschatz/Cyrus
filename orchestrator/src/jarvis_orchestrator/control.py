"""Strict loopback control API with content-free errors."""

from __future__ import annotations

import hmac
import base64
import hashlib
import json
import re
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .config import OrchestratorConfig
from .domain import OrchestratorError, require_uuid4, utc_text
from .scheduler_contract import (MINIMUM_INTERVAL_MS, MINIMUM_LEAD_MS, MAXIMUM_SNOOZE_GENERATIONS,
                                 MANUAL_CANONICAL_API_SHA256)
from .jcs import domain_sha256, jcs_bytes
from .service import OrchestratorService
from .scheduler_contract import ScheduleContractError, compile_proposal


APPROVAL_DECISION = re.compile(r"^/v1/approvals/([0-9a-f-]{36})/decision$")
SCHEDULER_APPROVAL_DECISION = re.compile(r"^/v1/scheduler/approvals/([0-9a-f-]{36})/decision$")
TASK_CANCEL = re.compile(r"^/v1/tasks/([0-9a-f-]{36})/cancel$")
CONVERSATION_DELETE = re.compile(r"^/v1/conversations/([0-9a-f-]{36})$")
ACTION_APPROVAL_PRESENTATION = re.compile(
    r"^/v1/action-approvals/([0-9a-f-]{36})/presentation$"
)
MAX_BODY = 16 * 1024
MAX_HEADERS = 32
MAX_HEADER_BYTES = 8 * 1024
MAX_FIELD_BYTES = 1024


def _cursor_encode(kind: str, binding: dict[str, Any], key: list[Any]) -> str:
    core = {"binding": binding, "key": key, "kind": kind, "version": "scheduler-cursor/1"}
    value = {**core, "sha256": hashlib.sha256(jcs_bytes(core)).hexdigest()}
    return base64.urlsafe_b64encode(jcs_bytes(value)).rstrip(b"=").decode("ascii")


def _cursor_decode(token: str, *, kind: str, binding: dict[str, Any]) -> list[Any]:
    if not isinstance(token, str) or len(token) > 2048 or not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        raise OrchestratorError("invalid_cursor", status=400)
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        value = json.loads(raw)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OrchestratorError("invalid_cursor", status=400) from exc
    if not isinstance(value, dict) or set(value) != {"binding", "key", "kind", "version", "sha256"}:
        raise OrchestratorError("invalid_cursor", status=400)
    core = {name: value[name] for name in ("binding", "key", "kind", "version")}
    if (value["kind"] != kind or value["binding"] != binding or value["version"] != "scheduler-cursor/1"
            or not hmac.compare_digest(value["sha256"], hashlib.sha256(jcs_bytes(core)).hexdigest())
            or not isinstance(value["key"], list)):
        raise OrchestratorError("cursor_binding_mismatch", status=400)
    return value["key"]


class ControlApplication:
    def __init__(self, config: OrchestratorConfig, service: OrchestratorService) -> None:
        self.config = config
        self.service = service
        self._mutation_lock = threading.RLock()

    @staticmethod
    def _exact_object(value: Any, fields: set[str]) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != fields:
            raise OrchestratorError("invalid_request", status=400)
        return value

    @staticmethod
    def known_path(path: str) -> bool:
        path = urlsplit(path).path
        return bool(
            path in {
                "/readyz", "/v1/approvals", "/v1/conversations/active", "/v1/conversations",
                "/v1/readiness", "/v1/capabilities", "/v1/scheduler/proposals",
                "/v1/scheduler/compile",
                "/v1/scheduler/jobs", "/v1/scheduler/occurrences", "/v1/scheduler/ledger",
            }
            or APPROVAL_DECISION.fullmatch(path)
            or SCHEDULER_APPROVAL_DECISION.fullmatch(path)
            or TASK_CANCEL.fullmatch(path)
            or CONVERSATION_DELETE.fullmatch(path)
            or ACTION_APPROVAL_PRESENTATION.fullmatch(path)
        )

    def action_presentation(
        self,
        path: str,
        *,
        presentation_id: str,
        preview_nonce: str,
        epoch: int,
    ) -> tuple[int, Any]:
        match = ACTION_APPROVAL_PRESENTATION.fullmatch(urlsplit(path).path)
        if match is None or self.service.tool_store is None:
            raise OrchestratorError("not_found", status=404)
        return 200, self.service.tool_store.consume_preview(
            approval_id=require_uuid4(match.group(1)),
            presentation_id=require_uuid4(presentation_id),
            preview_nonce=require_uuid4(preview_nonce),
            epoch=epoch,
        )

    def get(self, path: str) -> tuple[int, Any]:
        split = urlsplit(path)
        route = split.path
        query = parse_qs(split.query, strict_parsing=True, keep_blank_values=False)
        if route == "/readyz":
            ready = self.service.ready()
            return (200 if ready else 503), {"ready": ready}
        if route == "/v1/readiness":
            inference_ready = self.service.ready()
            scheduler_ready = self.service.scheduler is not None and self.service.scheduler.readiness()["state"] == "READY"
            capabilities = ["control"] + (["scheduler"] if scheduler_ready else [])
            return 200, {
                "core_ready": True,
                "scheduler_ready": scheduler_ready,
                "inference_ready": inference_ready,
                "mode": "ready" if inference_ready else "degraded",
                "capabilities": capabilities,
            }
        if route == "/v1/capabilities":
            scheduler = self.service.scheduler
            scheduler_state = scheduler.readiness() if scheduler is not None else {
                "state": "BLOCKED", "reason_code": "scheduler_key_or_timezone_unavailable"
            }
            with self.service.store._lock:
                schema = self.service.store._connection.execute(
                    "SELECT value FROM schema_meta WHERE key='scheduler_schema_version'"
                ).fetchone()
            arbiter_ready = schema is not None and schema["value"] == "5"
            native_delivery_ready = (
                bool(self.service.config.voice_scheduler_compatibility_sha256)
                and self.service.has_scheduler_client()
            )
            compiler_ready = self.service.schedule_compiler is not None
            notification_actions_ready = False
            reminders_ready = (
                arbiter_ready
                and scheduler_state["state"] == "READY"
                and compiler_ready
                and native_delivery_ready
                and notification_actions_ready
            )
            return 200, {"capabilities": [
                {"name": "approval_arbiter", "state": "READY" if arbiter_ready else "BLOCKED",
                 "reason_code": "ready" if arbiter_ready else "scheduler_schema_mismatch",
                 "qualification_sha256": None},
                {"name": "scheduler_core", "state": scheduler_state["state"],
                 "reason_code": scheduler_state["reason_code"], "qualification_sha256": None},
                {"name": "schedule_compiler_local",
                 "state": "READY" if compiler_ready else "BLOCKED",
                 "reason_code": "ready" if compiler_ready else "compiler_qualification_missing",
                 "qualification_sha256": (
                     self.service.schedule_compiler.qualification.sha256
                     if self.service.schedule_compiler is not None else None
                 )},
                {"name": "notification_delivery",
                 "state": "READY" if native_delivery_ready else "BLOCKED",
                 "reason_code": "ready" if native_delivery_ready else "qualified_scheduler_client_unavailable",
                 "qualification_sha256": (
                     self.service.config.voice_scheduler_compatibility_sha256 or None
                 )},
                {"name": "notification_actions", "state": "BLOCKED",
                 "reason_code": "native_action_callback_not_qualified",
                 "qualification_sha256": None},
                {"name": "reminders_overall",
                 "state": "READY" if reminders_ready else "BLOCKED",
                 "reason_code": "ready" if reminders_ready else "required_capability_unavailable",
                 "qualification_sha256": (
                     self.service.config.voice_scheduler_compatibility_sha256
                     if reminders_ready else None
                 )},
                {"name": "reminder_tts", "state": "DISABLED",
                 "reason_code": "disabled_by_policy", "qualification_sha256": None},
            ]}
        if route == "/v1/approvals":
            return 200, {"approvals": self.service.store.list_pending_approvals()}
        if route == "/v1/conversations/active":
            return 200, {"conversation": self.service.store.active_conversation()}
        if route.startswith("/v1/scheduler/"):
            scheduler = self.service.scheduler
            if scheduler is None:
                raise OrchestratorError("scheduler_unavailable", status=503)
            if any(len(values) != 1 for values in query.values()):
                raise OrchestratorError("invalid_request", status=400)
            try:
                limit = int(query.get("limit", ["50"])[0])
            except ValueError as exc:
                raise OrchestratorError("invalid_request", status=400) from exc
            if route == "/v1/scheduler/jobs":
                if set(query) - {"limit", "cursor"}:
                    raise OrchestratorError("invalid_request", status=400)
                binding = {"filter": "all"}
                cursor = None
                if "cursor" in query:
                    key = _cursor_decode(query["cursor"][0], kind="jobs", binding=binding)
                    if len(key) != 2:
                        raise OrchestratorError("invalid_cursor", status=400)
                    cursor = (key[0], key[1])
                jobs = scheduler.list_jobs(limit=limit, cursor=cursor)
                next_cursor = (_cursor_encode("jobs", binding, [jobs[-1]["created_at_utc"], jobs[-1]["job_id"]])
                               if len(jobs) == limit else None)
                return 200, {"jobs": jobs, "next_cursor": next_cursor}
            if route == "/v1/scheduler/occurrences":
                required = {"job_id", "start_at_utc", "end_at_utc"}
                if not required.issubset(query) or set(query) - required - {"limit", "cursor"}:
                    raise OrchestratorError("invalid_request", status=400)
                binding = {name: query[name][0] for name in sorted(required)}
                cursor = None
                if "cursor" in query:
                    key = _cursor_decode(query["cursor"][0], kind="occurrences", binding=binding)
                    if len(key) != 2:
                        raise OrchestratorError("invalid_cursor", status=400)
                    cursor = (key[0], key[1])
                occurrences = scheduler.list_occurrences(
                    job_id=query["job_id"][0], start_at_utc=query["start_at_utc"][0],
                    end_at_utc=query["end_at_utc"][0], limit=limit, cursor=cursor,
                )
                next_cursor = (_cursor_encode("occurrences", binding,
                    [occurrences[-1]["scheduled_at_utc"], occurrences[-1]["occurrence_id"]])
                    if len(occurrences) == limit else None)
                return 200, {"occurrences": occurrences, "next_cursor": next_cursor}
            if route == "/v1/scheduler/ledger":
                if set(query) - {"limit", "after_seq"}:
                    raise OrchestratorError("invalid_request", status=400)
                try:
                    after_seq = int(query.get("after_seq", ["0"])[0])
                except ValueError as exc:
                    raise OrchestratorError("invalid_request", status=400) from exc
                return 200, {"ledger": scheduler.ledger(after_seq=after_seq, limit=limit)}
        if self.known_path(path):
            raise OrchestratorError("method_not_allowed", status=405)
        raise OrchestratorError("not_found", status=404)

    def mutate(self, method: str, path: str, body: Any, key: str, epoch: int) -> tuple[int, Any | None]:
        with self._mutation_lock:
            replay = self.service.store.lookup_command(key, method, path, body, epoch)
            if replay is not None:
                status, response_json = replay
                return status, json.loads(response_json) if response_json is not None else None
            approval = APPROVAL_DECISION.fullmatch(path)
            cancel = TASK_CANCEL.fullmatch(path)
            deletion = CONVERSATION_DELETE.fullmatch(path)
            try:
                if method == "POST" and approval:
                    payload = self._exact_object(body, {"decision"})
                    result = self.service.decide_approval(require_uuid4(approval.group(1)), epoch, payload["decision"])
                    status = 200
                elif method == "POST" and cancel:
                    payload = self._exact_object(body, {"reason_code"})
                    if payload["reason_code"] != "user_cancelled":
                        raise OrchestratorError("invalid_request", status=400)
                    result = self.service.cancel_task(require_uuid4(cancel.group(1)), epoch)
                    status = 200
                elif method == "POST" and path == "/v1/conversations":
                    self._exact_object(body, set())
                    result = self.service.store.create_conversation(epoch)
                    status = 201
                elif method == "DELETE" and deletion:
                    if body is not None:
                        raise OrchestratorError("invalid_request", status=400)
                    self.service.store.delete_conversation(require_uuid4(deletion.group(1)), epoch)
                    result = None
                    status = 204
                else:
                    known = self.known_path(path)
                    raise OrchestratorError(
                        "method_not_allowed" if known else "not_found",
                        status=405 if known else 404,
                    )
            except OrchestratorError as exc:
                status = exc.status
                result = {"error": {"code": exc.code, "retryable": exc.retryable}}
            response_json = None if result is None else json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            self.service.store.store_command(key, method, path, body, epoch, status, response_json)
            return status, result

    def scheduler_mutate(self, method: str, path: str, body: Any, key: str) -> tuple[int, Any | None]:
        scheduler = self.service.scheduler
        if scheduler is None:
            raise OrchestratorError("scheduler_unavailable", status=503)
        with self._mutation_lock:
            command_semantic = domain_sha256(
                "scheduler-control-command/1", {"body": body, "method": method, "path": path}
            )
            if path in {"/v1/scheduler/compile", "/v1/scheduler/proposals"}:
                replay = scheduler.lookup_api_result(key, command_semantic)
                if replay is not None:
                    return int(replay["status"]), replay["result"]
            if method == "POST" and path == "/v1/scheduler/compile":
                payload = self._exact_object(
                    body, {"request_text", "selected_target", "default_timezone", "default_locale"}
                )
                if self.service.schedule_compiler is None:
                    raise OrchestratorError("schedule_compiler_not_qualified", status=503)
                model_input = {
                    "schema_version": "schedule-proposal-input/1",
                    "request_id": str(uuid.uuid4()),
                    "request_text": payload["request_text"],
                    "request_received_at_utc": utc_text(),
                    "default_timezone": payload["default_timezone"],
                    "default_locale": payload["default_locale"],
                    "selected_target": payload["selected_target"],
                    "policy": {
                        "minimum_lead_ms": MINIMUM_LEAD_MS,
                        "minimum_interval_ms": MINIMUM_INTERVAL_MS,
                        "maximum_snooze_generations": MAXIMUM_SNOOZE_GENERATIONS,
                    },
                }
                try:
                    compiled = self.service.schedule_compiler.compile(model_input)
                except ScheduleContractError as exc:
                    raise OrchestratorError(exc.code, status=exc.status) from exc
                if compiled.proposal.get("outcome") in {"NEEDS_CLARIFICATION", "UNSUPPORTED"}:
                    result = {"proposal": dict(compiled.proposal), "persisted": False}
                    scheduler.store_api_result(
                        event_id=key, semantic_sha256=command_semantic,
                        event_type="scheduler.control.compile_result", status=200, result=result,
                    )
                    return 200, result
                return 201, scheduler.create_proposal(
                    compiled, api_event_id=key, api_semantic_sha256=command_semantic
                )
            if method == "POST" and path == "/v1/scheduler/proposals":
                payload = self._exact_object(
                    body, {"input", "output"}
                )
                try:
                    compiled = compile_proposal(
                        payload["input"], payload["output"],
                        compiler_qualification_sha256=MANUAL_CANONICAL_API_SHA256,
                        zones=scheduler.zones,
                    )
                except ScheduleContractError as exc:
                    raise OrchestratorError(exc.code, status=exc.status) from exc
                if compiled.proposal.get("outcome") in {"NEEDS_CLARIFICATION", "UNSUPPORTED"}:
                    result = {"proposal": dict(compiled.proposal), "persisted": False}
                    scheduler.store_api_result(
                        event_id=key, semantic_sha256=command_semantic,
                        event_type="scheduler.control.manual_result", status=200, result=result,
                    )
                    return 200, result
                return 201, scheduler.create_proposal(
                    compiled, api_event_id=key, api_semantic_sha256=command_semantic
                )
            decision = SCHEDULER_APPROVAL_DECISION.fullmatch(path)
            if method == "POST" and decision:
                payload = self._exact_object(body, {"proposal_id", "proposal_sha256", "decision"})
                result = scheduler.decide_approval(
                    approval_id=require_uuid4(decision.group(1)),
                    proposal_id=require_uuid4(payload["proposal_id"]),
                    proposal_sha256=payload["proposal_sha256"],
                    decision=payload["decision"],
                    cause_event_id=key,
                )
                return 200, result
        raise OrchestratorError("method_not_allowed" if self.known_path(path) else "not_found",
                                status=405 if self.known_path(path) else 404)


def handler_factory(application: ControlApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "JARVIS-Control/1"
        sys_version = ""

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, status: int, value: Any | None) -> None:
            if value is None:
                encoded = b""
            else:
                encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Cache-Control", "no-store")
            if value is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            if encoded:
                self.wfile.write(encoded)

        def _error(self, error: OrchestratorError) -> None:
            self._json(error.status, {"error": {"code": error.code, "retryable": error.retryable}})

        def _admit(self, *, authenticate: bool) -> None:
            if self.client_address[0] != "127.0.0.1":
                raise OrchestratorError("forbidden", status=403)
            items = list(self.headers.items())
            if len(items) > MAX_HEADERS:
                raise OrchestratorError("invalid_request", status=400)
            total = sum(len(name.encode()) + len(value.encode()) for name, value in items)
            if total > MAX_HEADER_BYTES or any(len(name.encode()) > MAX_FIELD_BYTES or len(value.encode()) > MAX_FIELD_BYTES for name, value in items):
                raise OrchestratorError("invalid_request", status=400)
            hosts = self.headers.get_all("Host", failobj=[])
            expected_host = f"127.0.0.1:{application.config.control_port}"
            if hosts != [expected_host]:
                raise OrchestratorError("forbidden", status=403)
            forbidden = {name.lower() for name, _ in items if name.lower() in {"origin", "forwarded"} or name.lower().startswith("x-forwarded-")}
            if forbidden:
                raise OrchestratorError("forbidden", status=403)
            if authenticate:
                authorizations = self.headers.get_all("Authorization", failobj=[])
                if len(authorizations) != 1:
                    raise OrchestratorError("unauthenticated", status=401)
                authorization = authorizations[0]
                expected = f"Bearer {application.config.control_bearer_token}"
                if len(authorization.encode()) > 512 or not hmac.compare_digest(authorization, expected):
                    raise OrchestratorError("unauthenticated", status=401)

        def _mutation_headers(self) -> tuple[str, int]:
            keys = self.headers.get_all("Idempotency-Key", failobj=[])
            epochs = self.headers.get_all("If-Match-Epoch", failobj=[])
            if not keys or not epochs:
                raise OrchestratorError("precondition_required", status=428)
            if len(keys) != 1 or len(epochs) != 1:
                raise OrchestratorError("invalid_request", status=400)
            key, epoch_raw = keys[0], epochs[0]
            try:
                parsed = uuid.UUID(key)
                if parsed.version != 4 or str(parsed) != key.lower():
                    raise ValueError
                epoch = int(epoch_raw)
            except (ValueError, TypeError) as exc:
                raise OrchestratorError("invalid_request", status=400) from exc
            if epoch < 0:
                raise OrchestratorError("invalid_request", status=400)
            return key.lower(), epoch

        def _idempotency_header(self) -> str:
            keys = self.headers.get_all("Idempotency-Key", failobj=[])
            if len(keys) != 1:
                raise OrchestratorError("precondition_required", status=428)
            try:
                parsed = uuid.UUID(keys[0])
                if parsed.version != 4 or str(parsed) != keys[0].lower():
                    raise ValueError
            except (ValueError, TypeError) as exc:
                raise OrchestratorError("invalid_request", status=400) from exc
            return keys[0].lower()

        def _post_body(self) -> Any:
            lengths = self.headers.get_all("Content-Length", failobj=[])
            if not lengths:
                raise OrchestratorError("length_required", status=411)
            media_types = self.headers.get_all("Content-Type", failobj=[])
            if len(lengths) != 1 or len(media_types) != 1:
                raise OrchestratorError("invalid_request", status=400)
            length_raw = lengths[0]
            if media_types[0].split(";", 1)[0].strip().lower() != "application/json":
                raise OrchestratorError("unsupported_media_type", status=415)
            try:
                length = int(length_raw)
            except ValueError as exc:
                raise OrchestratorError("invalid_request", status=400) from exc
            if length < 0 or length > MAX_BODY:
                raise OrchestratorError("request_too_large", status=413)
            try:
                return json.loads(self.rfile.read(length).decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OrchestratorError("invalid_request", status=400) from exc

        def do_GET(self) -> None:
            try:
                self._admit(authenticate=self.path != "/readyz")
                if ACTION_APPROVAL_PRESENTATION.fullmatch(urlsplit(self.path).path):
                    presentation = self.headers.get_all("X-Presentation-Id", failobj=[])
                    nonce = self.headers.get_all("X-Preview-Nonce", failobj=[])
                    epochs = self.headers.get_all("If-Match-Epoch", failobj=[])
                    if len(presentation) != 1 or len(nonce) != 1 or len(epochs) != 1:
                        raise OrchestratorError("invalid_request", status=400)
                    try:
                        epoch = int(epochs[0])
                    except ValueError as exc:
                        raise OrchestratorError("invalid_request", status=400) from exc
                    status, result = application.action_presentation(
                        self.path,
                        presentation_id=presentation[0],
                        preview_nonce=nonce[0],
                        epoch=epoch,
                    )
                else:
                    status, result = application.get(self.path)
                self._json(status, result)
            except OrchestratorError as exc:
                self._error(exc)
            except Exception:
                self._error(OrchestratorError("internal_error", status=500))

        def do_POST(self) -> None:
            try:
                self._admit(authenticate=True)
                body = self._post_body()
                route = urlsplit(self.path).path
                if route.startswith("/v1/scheduler/"):
                    key = self._idempotency_header()
                    status, result = application.scheduler_mutate("POST", route, body, key)
                else:
                    key, epoch = self._mutation_headers()
                    status, result = application.mutate("POST", route, body, key, epoch)
                self._json(status, result)
            except OrchestratorError as exc:
                self._error(exc)
            except Exception:
                self._error(OrchestratorError("internal_error", status=500))

        def do_DELETE(self) -> None:
            try:
                self._admit(authenticate=True)
                key, epoch = self._mutation_headers()
                lengths = self.headers.get_all("Content-Length", failobj=[])
                if len(lengths) > 1 or (lengths and lengths[0] != "0"):
                    raise OrchestratorError("invalid_request", status=400)
                status, result = application.mutate("DELETE", self.path, None, key, epoch)
                self._json(status, result)
            except OrchestratorError as exc:
                self._error(exc)
            except Exception:
                self._error(OrchestratorError("internal_error", status=500))

        def do_PUT(self) -> None:
            try:
                self._admit(authenticate=True)
                if application.known_path(self.path):
                    self._error(OrchestratorError("method_not_allowed", status=405))
                else:
                    self._error(OrchestratorError("not_found", status=404))
            except OrchestratorError as exc:
                self._error(exc)

        do_PATCH = do_PUT
        do_HEAD = do_PUT
        do_OPTIONS = do_PUT
        do_TRACE = do_PUT
        do_CONNECT = do_PUT

    return Handler


class ControlServer:
    def __init__(self, application: ControlApplication) -> None:
        self.application = application
        self.httpd = ThreadingHTTPServer(
            (application.config.control_host, application.config.control_port),
            handler_factory(application),
        )

    def serve_forever(self) -> None:
        self.httpd.serve_forever(poll_interval=0.2)

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
