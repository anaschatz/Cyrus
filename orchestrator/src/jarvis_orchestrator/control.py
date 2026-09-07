"""Strict loopback control API with content-free errors."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .config import OrchestratorConfig
from .domain import OrchestratorError, parse_utc, require_uuid4, utc_text
from .jcs import domain_sha256, jcs_bytes
from .memory_crypto import MemoryCryptoError
from .scheduler_contract import (
    MANUAL_CANONICAL_API_SHA256,
    MAXIMUM_SNOOZE_GENERATIONS,
    MINIMUM_INTERVAL_MS,
    MINIMUM_LEAD_MS,
    ScheduleContractError,
    compile_proposal,
)
from .service import (
    BETA_SESSION_SCHEMA_VERSION,
    OrchestratorService,
)
from .store import (
    SQLITE_MAX_INTEGER,
    ControlMutationKind,
    ControlMutationRequest,
    MemoryStorageUnavailable,
)

APPROVAL_DECISION = re.compile(r"^/v1/approvals/([0-9a-f-]{36})/decision$")
SCHEDULER_APPROVAL_DECISION = re.compile(r"^/v1/scheduler/approvals/([0-9a-f-]{36})/decision$")
TASK_CANCEL = re.compile(r"^/v1/tasks/([0-9a-f-]{36})/cancel$")
CONVERSATION_DELETE = re.compile(r"^/v1/conversations/([0-9a-f-]{36})$")
MEMORY_RECORD = re.compile(r"^/v1/memory/records/([0-9a-f-]{36})$")
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


def _cursor_decode(token: str, *, kind: str, binding: dict[str, Any]) -> list[str]:
    if not isinstance(token, str) or len(token) > 2048 or not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        raise OrchestratorError("invalid_cursor", status=400)
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {"binding", "key", "kind", "version", "sha256"}:
            raise OrchestratorError("invalid_cursor", status=400)
        core = {name: value[name] for name in ("binding", "key", "kind", "version")}
        digest_matches = hmac.compare_digest(value["sha256"], hashlib.sha256(jcs_bytes(core)).hexdigest())
    except (TypeError, ValueError, RecursionError) as exc:
        raise OrchestratorError("invalid_cursor", status=400) from exc
    if (value["kind"] != kind or value["binding"] != binding or value["version"] != "scheduler-cursor/1"
            or not digest_matches
            or not isinstance(value["key"], list) or len(value["key"]) != 2
            or not all(isinstance(item, str) and item for item in value["key"])):
        raise OrchestratorError("cursor_binding_mismatch", status=400)
    return value["key"]


def _memory_cursor_encode(
    secret: str,
    *,
    revision: int,
    after: tuple[str, str],
) -> str:
    core = {
        "after": list(after),
        "memory_revision": revision,
        "scope": "local_owner",
        "version": "memory-records-cursor/1",
    }
    value = {
        **core,
        "hmac_sha256": hmac.new(
            secret.encode("utf-8"),
            jcs_bytes(core),
            hashlib.sha256,
        ).hexdigest(),
    }
    return base64.urlsafe_b64encode(jcs_bytes(value)).rstrip(b"=").decode("ascii")


def _memory_cursor_decode(
    secret: str,
    token: str,
) -> tuple[int, tuple[str, str]]:
    if (
        not isinstance(token, str)
        or len(token) > 2048
        or re.fullmatch(r"[A-Za-z0-9_-]+", token) is None
    ):
        raise OrchestratorError("invalid_cursor", status=400)
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        value = json.loads(raw)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OrchestratorError("invalid_cursor", status=400) from exc
    expected_fields = {
        "after",
        "hmac_sha256",
        "memory_revision",
        "scope",
        "version",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise OrchestratorError("invalid_cursor", status=400)
    digest = value["hmac_sha256"]
    revision = value["memory_revision"]
    after = value["after"]
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or value["scope"] != "local_owner"
        or value["version"] != "memory-records-cursor/1"
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or not isinstance(after, list)
        or len(after) != 2
        or not all(isinstance(item, str) and item for item in after)
    ):
        raise OrchestratorError("invalid_cursor", status=400)
    core = {
        name: value[name]
        for name in ("after", "memory_revision", "scope", "version")
    }
    try:
        canonical_value = jcs_bytes(value)
        expected_hmac = hmac.new(
            secret.encode("utf-8"),
            jcs_bytes(core),
            hashlib.sha256,
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise OrchestratorError("invalid_cursor", status=400) from exc
    if canonical_value != raw or not hmac.compare_digest(digest, expected_hmac):
        raise OrchestratorError("invalid_cursor", status=400)
    try:
        if utc_text(parse_utc(after[0])) != after[0]:
            raise ValueError
        record_id = require_uuid4(after[1])
    except (OrchestratorError, ValueError) as exc:
        raise OrchestratorError("invalid_cursor", status=400) from exc
    return revision, (after[0], record_id)


class ControlApplication:
    def __init__(
        self,
        config: OrchestratorConfig,
        service: OrchestratorService,
        *,
        runtime_health: Callable[[], Mapping[str, Any]] | None = None,
        business_admission: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.service = service
        self._runtime_health_provider = runtime_health
        # Transport admission is fail-closed unless a lifecycle owner
        # explicitly injects the shared readiness predicate.
        self._business_admission_provider = business_admission or (lambda: False)
        self._mutation_lock = threading.RLock()

    def require_business_admission(self, path: str) -> None:
        try:
            route = urlsplit(path).path
        except ValueError as exc:
            raise OrchestratorError("invalid_request", status=400) from exc
        if route in {"/readyz", "/v1/readiness"}:
            return
        try:
            admitted = self._business_admission_provider() is True
        except Exception:  # noqa: BLE001 - admission must fail closed
            admitted = False
        if not admitted:
            raise OrchestratorError(
                "runtime_not_ready",
                status=503,
                retryable=True,
            )

    def _runtime_health(self) -> tuple[bool, str, str | None]:
        if self._runtime_health_provider is None:
            return False, "UNSUPERVISED", "runtime_health_unavailable"
        try:
            health = self._runtime_health_provider()
            state = health.get("state")
            failure_code = health.get("failure_code")
            ready = health.get("ready") is True and state == "RUNNING"
        except Exception:  # noqa: BLE001 - runtime health must fail closed
            return False, "FAILED", "runtime_health_unavailable"
        if not isinstance(state, str) or state not in (
            "FAILED",
            "RUNNING",
            "STOPPED",
            "STOPPING",
        ):
            return False, "FAILED", "runtime_health_invalid"
        if failure_code is not None and (
            not isinstance(failure_code, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", failure_code) is None
        ):
            return False, "FAILED", "runtime_health_invalid"
        return ready, state, failure_code

    @staticmethod
    def _exact_object(value: Any, fields: set[str]) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != fields:
            raise OrchestratorError("invalid_request", status=400)
        return value

    @staticmethod
    def _public_memory_record(record: Mapping[str, Any]) -> dict[str, Any]:
        """Return the closed management projection, excluding crypto internals."""

        return {
            "record_id": record["record_id"],
            "content": record["content"],
            "created_at": record["created_at"],
            "expires_at": record["expires_at"],
        }

    @staticmethod
    def _memory_unavailable() -> OrchestratorError:
        return OrchestratorError(
            "memory_storage_unavailable",
            status=503,
            retryable=True,
        )

    @staticmethod
    def known_path(path: str) -> bool:
        path = urlsplit(path).path
        return bool(
            path in {
                "/readyz", "/v1/approvals", "/v1/conversations/active", "/v1/conversations",
                "/v1/readiness", "/v1/beta-session", "/v1/capabilities",
                "/v1/memory/records",
                "/v1/scheduler/proposals",
                "/v1/scheduler/compile",
                "/v1/scheduler/jobs", "/v1/scheduler/occurrences", "/v1/scheduler/ledger",
            }
            or APPROVAL_DECISION.fullmatch(path)
            or SCHEDULER_APPROVAL_DECISION.fullmatch(path)
            or TASK_CANCEL.fullmatch(path)
            or CONVERSATION_DELETE.fullmatch(path)
            or MEMORY_RECORD.fullmatch(path)
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
        try:
            query = parse_qs(
                split.query,
                strict_parsing=True,
                keep_blank_values=True,
                max_num_fields=16,
            )
        except ValueError as exc:
            raise OrchestratorError("invalid_request", status=400) from exc
        if route == "/readyz":
            runtime_ready, _, _ = self._runtime_health()
            readiness = self.service.readiness_snapshot()
            ready = readiness["core_ready"] and runtime_ready
            return (200 if ready else 503), {"ready": ready}
        if route == "/v1/readiness":
            readiness = self.service.readiness_snapshot()
            runtime_ready, runtime_state, runtime_failure_code = self._runtime_health()
            core_ready = readiness["core_ready"] and runtime_ready
            scheduler_ready = self.service.scheduler is not None and self.service.scheduler.readiness()["state"] == "READY"
            capabilities = ["control"] + (["scheduler"] if scheduler_ready else [])
            return 200, {
                "core_ready": core_ready,
                "scheduler_ready": scheduler_ready,
                "inference_ready": readiness["inference_ready"],
                "memory_ready": readiness["memory_ready"],
                "memory_state": readiness["memory_state"],
                "storage_ready": readiness["storage_ready"],
                "runtime_ready": runtime_ready,
                "runtime_state": runtime_state,
                "runtime_failure_code": runtime_failure_code,
                "mode": "ready" if core_ready else "degraded",
                "capabilities": capabilities,
            }
        if route == "/v1/beta-session":
            readiness = self.service.readiness_snapshot()
            runtime_ready, _, _ = self._runtime_health()
            core_ready = readiness["core_ready"] and runtime_ready
            connected, tracker_id, generation = (
                self.service.voice_connection_state()
            )
            return 200, {
                "core_ready": core_ready,
                "effective_capabilities": {
                    "cloud": self.service.cloud_approval_available(),
                    "memory_retrospective": readiness[
                        "memory_retrospective_ready"
                    ],
                    "memory_shadow": self.service.config.memory_shadow_enabled,
                    "research": self.service.config.research_enabled,
                    "tools": self.service.config.tools_enabled,
                },
                "schema_version": BETA_SESSION_SCHEMA_VERSION,
                "voice_client_connected": connected,
                "voice_connection_generation": generation,
                "voice_connection_tracker_id": tracker_id,
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
            notification_actions_ready = (
                bool(self.service.config.voice_scheduler_callback_qualification_sha256)
                and self.service.has_scheduler_action_client()
            )
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
                {"name": "notification_actions",
                 "state": "READY" if notification_actions_ready else "BLOCKED",
                 "reason_code": "ready" if notification_actions_ready else "native_action_callback_not_qualified",
                 "qualification_sha256": (
                     self.service.config.voice_scheduler_callback_qualification_sha256
                     if notification_actions_ready else None
                 )},
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
        if route == "/v1/memory/records":
            if (
                split.fragment
                or any(len(values) != 1 for values in query.values())
                or set(query) - {"limit", "after"}
            ):
                raise OrchestratorError("invalid_request", status=400)
            try:
                limit = int(query.get("limit", ["50"])[0])
            except ValueError as exc:
                raise OrchestratorError("invalid_request", status=400) from exc
            if not 1 <= limit <= 50:
                raise OrchestratorError("invalid_request", status=400)
            expected_revision = None
            after = None
            if "after" in query:
                expected_revision, after = _memory_cursor_decode(
                    self.config.control_bearer_token,
                    query["after"][0],
                )
            try:
                page = self.service.store.control_memory_record_page(
                    limit=limit,
                    after=after,
                    expected_revision=expected_revision,
                )
            except OrchestratorError:
                raise
            except (
                MemoryCryptoError,
                MemoryStorageUnavailable,
                RuntimeError,
            ) as exc:
                raise self._memory_unavailable() from exc
            next_cursor = (
                None
                if page["next_cursor"] is None
                else _memory_cursor_encode(
                    self.config.control_bearer_token,
                    revision=page["memory_revision"],
                    after=page["next_cursor"],
                )
            )
            return 200, {
                "records": [
                    self._public_memory_record(record)
                    for record in page["items"]
                ],
                "memory_revision": page["memory_revision"],
                "next_cursor": next_cursor,
            }
        memory_record = MEMORY_RECORD.fullmatch(route)
        if memory_record is not None:
            if query or split.fragment:
                raise OrchestratorError("invalid_request", status=400)
            record_id = require_uuid4(memory_record.group(1))
            try:
                snapshot = self.service.store.control_memory_record_snapshot(
                    record_id
                )
            except OrchestratorError:
                raise
            except (
                MemoryCryptoError,
                MemoryStorageUnavailable,
                RuntimeError,
            ) as exc:
                raise self._memory_unavailable() from exc
            if snapshot["record"] is None:
                raise OrchestratorError("not_found", status=404)
            return 200, {
                "record": self._public_memory_record(snapshot["record"]),
                "memory_revision": snapshot["memory_revision"],
            }
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
            approval = APPROVAL_DECISION.fullmatch(path)
            cancel = TASK_CANCEL.fullmatch(path)
            deletion = CONVERSATION_DELETE.fullmatch(path)
            if method == "POST" and approval:
                request = ControlMutationRequest(
                    ControlMutationKind.DECIDE_APPROVAL,
                    approval.group(1),
                )
            elif method == "POST" and cancel:
                request = ControlMutationRequest(
                    ControlMutationKind.CANCEL_TASK,
                    cancel.group(1),
                )
            elif method == "POST" and path == "/v1/conversations":
                request = ControlMutationRequest(
                    ControlMutationKind.CREATE_CONVERSATION
                )
            elif method == "DELETE" and deletion:
                request = ControlMutationRequest(
                    ControlMutationKind.DELETE_CONVERSATION,
                    deletion.group(1),
                )
            elif self.known_path(path):
                request = ControlMutationRequest(
                    ControlMutationKind.METHOD_NOT_ALLOWED
                )
            else:
                request = ControlMutationRequest(ControlMutationKind.NOT_FOUND)
            status, response_json = self.service.store.execute_control_mutation(
                key,
                method,
                path,
                body,
                epoch,
                request,
                client_connected=self.service.has_client(),
            )
            result = (
                None if response_json is None else json.loads(response_json)
            )
            if (
                request.kind is ControlMutationKind.DECIDE_APPROVAL
                and status == 200
                and isinstance(result, dict)
                and result.get("task_state") == "EXECUTING"
                and isinstance(result.get("task_id"), str)
            ):
                # Scheduling is intentionally outside the SQLite transaction;
                # task execution is guarded by its durable state/epoch CAS.
                self.service._schedule(result["task_id"])
            return status, result

    def memory_mutate(
        self,
        method: str,
        path: str,
        body: Any,
        key: str,
        memory_revision: int,
    ) -> tuple[int, Any | None]:
        """Apply one revision-bound saved-memory mutation."""

        with self._mutation_lock:
            match = MEMORY_RECORD.fullmatch(path)
            if method == "DELETE" and match is not None:
                request = ControlMutationRequest(
                    ControlMutationKind.DELETE_MEMORY_RECORD,
                    require_uuid4(match.group(1)),
                )
            elif self.known_path(path):
                request = ControlMutationRequest(
                    ControlMutationKind.METHOD_NOT_ALLOWED
                )
            else:
                request = ControlMutationRequest(ControlMutationKind.NOT_FOUND)
            status, response_json = self.service.store.execute_control_mutation(
                key,
                method,
                path,
                body,
                memory_revision,
                request,
                client_connected=self.service.has_client(),
            )
            return (
                status,
                None if response_json is None else json.loads(response_json),
            )

    def scheduler_mutate(self, method: str, path: str, body: Any, key: str) -> tuple[int, Any | None]:
        scheduler = self.service.scheduler
        if scheduler is None:
            raise OrchestratorError("scheduler_unavailable", status=503)
        with self._mutation_lock:
            try:
                command_semantic = domain_sha256(
                    "scheduler-control-command/1", {"body": body, "method": method, "path": path}
                )
            except (TypeError, ValueError) as exc:
                raise OrchestratorError("invalid_request", status=400) from exc
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
            if self.headers.get_all("Transfer-Encoding", failobj=[]):
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
                authorization = authorizations[0].encode("utf-8")
                expected = f"Bearer {application.config.control_bearer_token}".encode("utf-8")
                if len(authorization) > 512 or not hmac.compare_digest(authorization, expected):
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

        def _memory_mutation_headers(self) -> tuple[str, int]:
            keys = self.headers.get_all("Idempotency-Key", failobj=[])
            revisions = self.headers.get_all(
                "If-Match-Memory-Revision",
                failobj=[],
            )
            epochs = self.headers.get_all("If-Match-Epoch", failobj=[])
            if epochs:
                raise OrchestratorError("invalid_request", status=400)
            if not keys or not revisions:
                raise OrchestratorError("precondition_required", status=428)
            if len(keys) != 1 or len(revisions) != 1:
                raise OrchestratorError("invalid_request", status=400)
            key, revision_raw = keys[0], revisions[0]
            try:
                parsed = uuid.UUID(key)
                if parsed.version != 4 or str(parsed) != key.lower():
                    raise ValueError
                revision = int(revision_raw)
            except (ValueError, TypeError) as exc:
                raise OrchestratorError("invalid_request", status=400) from exc
            if revision < 0 or revision > SQLITE_MAX_INTEGER:
                raise OrchestratorError("invalid_request", status=400)
            return key.lower(), revision

        def _memory_delete_body(self) -> Any | None:
            lengths = self.headers.get_all("Content-Length", failobj=[])
            if len(lengths) > 1:
                raise OrchestratorError("invalid_request", status=400)
            if not lengths:
                return None
            try:
                length = int(lengths[0])
            except ValueError as exc:
                raise OrchestratorError("invalid_request", status=400) from exc
            if length < 0 or length > MAX_BODY:
                raise OrchestratorError("request_too_large", status=413)
            if length == 0:
                return None
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise OrchestratorError("invalid_request", status=400)
            return {
                "unexpected_body_bytes": length,
                "unexpected_body_sha256": hashlib.sha256(raw).hexdigest(),
            }

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
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise OrchestratorError("invalid_request", status=400)
            try:
                return json.loads(raw.decode("utf-8", errors="strict"))
            except (ValueError, RecursionError) as exc:
                raise OrchestratorError("invalid_request", status=400) from exc

        def do_GET(self) -> None:
            try:
                self._admit(authenticate=self.path != "/readyz")
                application.require_business_admission(self.path)
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
                application.require_business_admission(self.path)
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
                application.require_business_admission(self.path)
                route = urlsplit(self.path).path
                if MEMORY_RECORD.fullmatch(route):
                    if self.path != route:
                        raise OrchestratorError("invalid_request", status=400)
                    key, revision = self._memory_mutation_headers()
                    body = self._memory_delete_body()
                    status, result = application.memory_mutate(
                        "DELETE",
                        route,
                        body,
                        key,
                        revision,
                    )
                else:
                    key, epoch = self._mutation_headers()
                    lengths = self.headers.get_all(
                        "Content-Length",
                        failobj=[],
                    )
                    if len(lengths) > 1 or (lengths and lengths[0] != "0"):
                        raise OrchestratorError("invalid_request", status=400)
                    status, result = application.mutate(
                        "DELETE",
                        self.path,
                        None,
                        key,
                        epoch,
                    )
                self._json(status, result)
            except OrchestratorError as exc:
                self._error(exc)
            except Exception:
                self._error(OrchestratorError("internal_error", status=500))

        def do_PUT(self) -> None:
            try:
                self._admit(authenticate=True)
                application.require_business_admission(self.path)
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
