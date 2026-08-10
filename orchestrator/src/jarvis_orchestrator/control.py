"""Strict loopback control API with content-free errors."""

from __future__ import annotations

import hmac
import json
import re
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import OrchestratorConfig
from .domain import OrchestratorError, require_uuid4
from .service import OrchestratorService


APPROVAL_DECISION = re.compile(r"^/v1/approvals/([0-9a-f-]{36})/decision$")
TASK_CANCEL = re.compile(r"^/v1/tasks/([0-9a-f-]{36})/cancel$")
CONVERSATION_DELETE = re.compile(r"^/v1/conversations/([0-9a-f-]{36})$")
MAX_BODY = 16 * 1024
MAX_HEADERS = 32
MAX_HEADER_BYTES = 8 * 1024
MAX_FIELD_BYTES = 1024


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
        return bool(
            path in {"/readyz", "/v1/approvals", "/v1/conversations/active", "/v1/conversations"}
            or APPROVAL_DECISION.fullmatch(path)
            or TASK_CANCEL.fullmatch(path)
            or CONVERSATION_DELETE.fullmatch(path)
        )

    def get(self, path: str) -> tuple[int, Any]:
        if path == "/readyz":
            ready = self.service.ready()
            return (200 if ready else 503), {"ready": ready}
        if path == "/v1/approvals":
            return 200, {"approvals": self.service.store.list_pending_approvals()}
        if path == "/v1/conversations/active":
            return 200, {"conversation": self.service.store.active_conversation()}
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
                status, result = application.get(self.path)
                self._json(status, result)
            except OrchestratorError as exc:
                self._error(exc)
            except Exception:
                self._error(OrchestratorError("internal_error", status=500))

        def do_POST(self) -> None:
            try:
                self._admit(authenticate=True)
                key, epoch = self._mutation_headers()
                body = self._post_body()
                status, result = application.mutate("POST", self.path, body, key, epoch)
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
