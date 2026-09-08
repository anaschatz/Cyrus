"""Frozen Tool/Action v1 schemas, normalization, and canonical outcomes.

This module contains no adapters and performs no operating-system calls.  It is
the code-owned authority for the schemas exposed to inference and for the
validation repeated immediately before dispatch.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .common_schema import transform_common_schema
from .domain import OrchestratorError
from .jcs import domain_sha256, jcs_bytes


TOOL_RESULT_CONTRACT = "jarvis.tool-result/1"
TOOL_REGISTRY_CONTRACT = "jarvis.tool-registry/1"
TOOL_ACTION_SCHEMA_VERSION = 6
SCHEDULER_PERSISTENCE_SCHEMA_VERSION = 5
MAX_ADAPTER_RESULT_BYTES = 256 * 1024
MAX_MODEL_PROJECTION_BYTES = 32 * 1024
MAX_PAGE_SIZE = 50
MAX_CURSOR_BYTES = 512
CURSOR_TTL_SECONDS = 600
MAX_CALENDAR_WINDOW = timedelta(days=31)


class ToolContractError(OrchestratorError):
    def __init__(self, code: str = "tool.schema_invalid", *, status: int = 400) -> None:
        super().__init__(code, status=status)


class ToolStatus(StrEnum):
    VERIFIED = "VERIFIED"
    ACCEPTED = "ACCEPTED"
    FAILED = "FAILED"
    AMBIGUOUS = "AMBIGUOUS"


class VerificationStatus(StrEnum):
    MATCHED = "MATCHED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MISMATCH = "MISMATCH"
    INCONCLUSIVE = "INCONCLUSIVE"


_OUTCOME_COMBINATIONS = frozenset(
    {
        (ToolStatus.VERIFIED, VerificationStatus.MATCHED),
        (ToolStatus.ACCEPTED, VerificationStatus.UNAVAILABLE),
        (ToolStatus.FAILED, VerificationStatus.NOT_APPLICABLE),
        (ToolStatus.AMBIGUOUS, VerificationStatus.MISMATCH),
        (ToolStatus.AMBIGUOUS, VerificationStatus.INCONCLUSIVE),
    }
)


COMMON_REASON_CODES = frozenset(
    {
        "ok",
        "tool.unknown",
        "tool.disabled",
        "tool.unsupported_platform",
        "tool.schema_invalid",
        "tool.payload_too_large",
        "tool.output_invalid",
        "tool.approval_busy",
        "tool.approval_denied",
        "tool.approval_expired",
        "tool.binding_mismatch",
        "tool.policy_changed",
        "tool.readiness_changed",
        "tool.idempotency_conflict",
        "tool.cancelled_pre_dispatch",
        "tool.timeout_pre_dispatch",
        "tool.ambiguous_after_dispatch",
        "tool.verification_unavailable",
        "tool.verification_mismatch",
        "tool.internal_safe_failure",
    }
)

ADAPTER_REASON_CODES = frozenset(
    {
        "file.root_unavailable",
        "file.ref_invalid",
        "file.ref_expired",
        "file.identity_changed",
        "file.outside_root",
        "file.type_denied",
        "file.handler_denied",
        "calendar.full_access_required",
        "calendar.write_only_no_readback",
        "calendar.range_too_large",
        "calendar.time_nonexistent",
        "calendar.time_ambiguous",
        "calendar.permission_revoked",
        "calendar.event_missing_after_write",
        "reminder.full_access_required",
        "reminder.ref_invalid",
        "reminder.not_found",
        "reminder.permission_revoked",
        "note.storage_unavailable",
        "note.verify_failed",
        "email.address_invalid",
        "email.permission_revoked",
        "email.mail_unavailable",
        "email.draft_receipt_missing",
        "host.protocol_mismatch",
        "host.auth_failed",
        "host.replay",
        "host.sequence_gap",
        "host.frame_too_large",
        "host.deadline_expired",
        "host.idempotency_conflict",
        "host.adapter_rejected",
        "host.receipt_unknown",
        "host.internal_safe_failure",
    }
)

REASON_CODES = COMMON_REASON_CODES | ADAPTER_REASON_CODES


@dataclass(frozen=True, slots=True)
class ToolOutcomeV1:
    status: ToolStatus
    verification: VerificationStatus
    reason_code: str

    def __post_init__(self) -> None:
        if (self.status, self.verification) not in _OUTCOME_COMBINATIONS:
            raise ToolContractError("tool.output_invalid")
        if self.reason_code not in REASON_CODES:
            raise ToolContractError("tool.output_invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "status": self.status.value,
            "verification": self.verification.value,
            "reason_code": self.reason_code,
        }


def utf16_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _reject_forbidden_unicode(value: str, *, multiline: bool) -> None:
    for character in value:
        if multiline and character == "\n":
            continue
        if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            raise ToolContractError()


def normalize_single_line(
    value: Any,
    *,
    utf8_limit: int,
    utf16_limit: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ToolContractError()
    normalized = unicodedata.normalize("NFC", value)
    _reject_forbidden_unicode(normalized, multiline=False)
    normalized = re.sub(r"\s+", " ", normalized, flags=re.UNICODE).strip()
    normalized = unicodedata.normalize("NFC", normalized)
    if not normalized and not allow_empty:
        raise ToolContractError()
    if len(normalized.encode("utf-8")) > utf8_limit or utf16_units(normalized) > utf16_limit:
        raise ToolContractError("tool.payload_too_large")
    return normalized


def normalize_multiline(
    value: Any,
    *,
    utf8_limit: int,
    utf16_limit: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ToolContractError()
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    normalized = unicodedata.normalize("NFC", normalized)
    _reject_forbidden_unicode(normalized, multiline=True)
    lines = [re.sub(r"[ \u00a0]+$", "", line) for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    normalized = unicodedata.normalize("NFC", "\n".join(lines))
    if not normalized and not allow_empty:
        raise ToolContractError()
    if len(normalized.encode("utf-8")) > utf8_limit or utf16_units(normalized) > utf16_limit:
        raise ToolContractError("tool.payload_too_large")
    return normalized


_UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")
_LOCAL_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}")


def parse_utc_strict(value: Any) -> datetime:
    if not isinstance(value, str) or _UTC_PATTERN.fullmatch(value) is None:
        raise ToolContractError()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ToolContractError() from exc
    if parsed.utcoffset() != timedelta(0) or parsed.microsecond % 1000:
        raise ToolContractError()
    return parsed.astimezone(UTC)


def canonical_timezone(value: Any) -> str:
    selected = normalize_single_line(value, utf8_limit=64, utf16_limit=64)
    if selected.startswith("/") or "\\" in selected or any(part in {"", ".", ".."} for part in selected.split("/")):
        raise ToolContractError()
    try:
        ZoneInfo(selected)
    except ZoneInfoNotFoundError as exc:
        raise ToolContractError() from exc
    return selected


def parse_local_strict(value: Any, timezone_name: str) -> tuple[datetime, datetime]:
    if not isinstance(value, str) or _LOCAL_PATTERN.fullmatch(value) is None:
        raise ToolContractError()
    try:
        local = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ToolContractError() from exc
    if local.tzinfo is not None or local.microsecond % 1000:
        raise ToolContractError()
    zone = ZoneInfo(canonical_timezone(timezone_name))
    valid: list[datetime] = []
    for fold in (0, 1):
        candidate = local.replace(tzinfo=zone, fold=fold)
        try:
            round_trip = candidate.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
        except OverflowError as exc:
            raise ToolContractError() from exc
        if round_trip == local:
            valid.append(candidate)
    unique_offsets = {item.utcoffset() for item in valid}
    if not valid:
        raise ToolContractError("calendar.time_nonexistent")
    if len(unique_offsets) > 1:
        raise ToolContractError("calendar.time_ambiguous")
    aware = valid[0]
    return local, aware.astimezone(UTC)


def canonical_uuid4(value: Any) -> str:
    if not isinstance(value, str):
        raise ToolContractError()
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ToolContractError() from exc
    if parsed.version != 4 or str(parsed) != value.lower():
        raise ToolContractError()
    return value.lower()


_LOCAL_PART = re.compile(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+")


def normalize_email_address(value: Any) -> str:
    if not isinstance(value, str) or any(character in value for character in "\r\n<>()[],:;\""):
        raise ToolContractError("email.address_invalid")
    if value.count("@") != 1:
        raise ToolContractError("email.address_invalid")
    local, domain = value.split("@", 1)
    if (
        not 1 <= len(local.encode("ascii", errors="ignore")) <= 64
        or not local.isascii()
        or _LOCAL_PART.fullmatch(local) is None
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
    ):
        raise ToolContractError("email.address_invalid")
    try:
        canonical_domain = domain.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ToolContractError("email.address_invalid") from exc
    if not canonical_domain or len(canonical_domain) > 253:
        raise ToolContractError("email.address_invalid")
    labels = canonical_domain.split(".")
    if any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or re.fullmatch(r"[a-z0-9-]+", label) is None
        for label in labels
    ):
        raise ToolContractError("email.address_invalid")
    result = f"{local}@{canonical_domain}"
    if len(result.encode("ascii")) > 254:
        raise ToolContractError("email.address_invalid")
    return result


def _exact(value: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ToolContractError()
    return dict(value)


def _integer(value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ToolContractError()
    return value


def _ascii(value: Any, *, minimum: int = 0, maximum: int) -> str:
    if not isinstance(value, str) or not value.isascii() or not minimum <= len(value.encode("ascii")) <= maximum:
        raise ToolContractError()
    return value


def _enum(value: Any, values: set[str]) -> str:
    if not isinstance(value, str) or value not in values:
        raise ToolContractError()
    return value


def _unique_sorted_strings(value: Any, *, values: set[str], minimum: int, maximum: int) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ToolContractError()
    if any(not isinstance(item, str) or item not in values for item in value):
        raise ToolContractError()
    if value != sorted(set(value)):
        raise ToolContractError()
    return list(value)


def _page(row: dict[str, Any]) -> None:
    _integer(row["limit"], 1, MAX_PAGE_SIZE)
    _ascii(row["cursor"], maximum=MAX_CURSOR_BYTES)


def _validate_file_search(value: Mapping[str, Any], root_ids: set[str]) -> dict[str, Any]:
    row = _exact(value, {"query", "root_ids", "kinds", "limit", "cursor"})
    row["query"] = normalize_single_line(row["query"], utf8_limit=256, utf16_limit=128)
    row["root_ids"] = _unique_sorted_strings(row["root_ids"], values=root_ids, minimum=1, maximum=4)
    row["kinds"] = _unique_sorted_strings(row["kinds"], values={"FILE", "DIRECTORY"}, minimum=1, maximum=2)
    _page(row)
    return row


def _validate_ref(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    row = _exact(value, {field})
    row[field] = canonical_uuid4(row[field])
    return row


def _validate_app_open(value: Mapping[str, Any], app_ids: set[str]) -> dict[str, Any]:
    row = _exact(value, {"app_id"})
    row["app_id"] = _enum(row["app_id"], app_ids)
    return row


def _calendar_window(row: dict[str, Any], start_name: str, end_name: str, timezone_name: str) -> None:
    zone = canonical_timezone(row[timezone_name])
    start_local, start_utc = parse_local_strict(row[start_name], zone)
    end_local, end_utc = parse_local_strict(row[end_name], zone)
    if end_utc <= start_utc:
        raise ToolContractError()
    if end_utc - start_utc > MAX_CALENDAR_WINDOW:
        raise ToolContractError("calendar.range_too_large")
    row[start_name] = start_local.isoformat(timespec="milliseconds")
    row[end_name] = end_local.isoformat(timespec="milliseconds")
    row[timezone_name] = zone


def _validate_calendar_read(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(value, {"start_local", "end_local", "timezone", "limit", "cursor"})
    _calendar_window(row, "start_local", "end_local", "timezone")
    _page(row)
    return row


def _validate_calendar_create(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(value, {"title", "start_local", "end_local", "timezone", "location", "notes", "availability"})
    row["title"] = normalize_single_line(row["title"], utf8_limit=192, utf16_limit=63)
    _calendar_window(row, "start_local", "end_local", "timezone")
    row["location"] = normalize_single_line(row["location"], utf8_limit=384, utf16_limit=128, allow_empty=True)
    row["notes"] = normalize_multiline(row["notes"], utf8_limit=4096, utf16_limit=2048, allow_empty=True)
    row["availability"] = _enum(row["availability"], {"BUSY", "FREE"})
    return row


def _validate_reminder_list(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(value, {"status", "due_filter", "due_from_local", "due_to_local", "timezone", "limit", "cursor"})
    row["status"] = _enum(row["status"], {"INCOMPLETE", "COMPLETED"})
    row["due_filter"] = _enum(row["due_filter"], {"ANY", "WINDOW"})
    if row["due_filter"] == "ANY":
        if any(row[field] != "" for field in ("due_from_local", "due_to_local", "timezone")):
            raise ToolContractError()
    else:
        if not row["due_from_local"] or not row["due_to_local"] or not row["timezone"]:
            raise ToolContractError()
        _calendar_window(row, "due_from_local", "due_to_local", "timezone")
    _page(row)
    return row


def _validate_reminder_create(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(value, {"title", "notes", "has_due", "due_local", "timezone", "priority"})
    row["title"] = normalize_single_line(row["title"], utf8_limit=192, utf16_limit=63)
    row["notes"] = normalize_multiline(row["notes"], utf8_limit=4096, utf16_limit=2048, allow_empty=True)
    if not isinstance(row["has_due"], bool):
        raise ToolContractError()
    if row["has_due"]:
        if not row["due_local"] or not row["timezone"]:
            raise ToolContractError()
        zone = canonical_timezone(row["timezone"])
        local, _ = parse_local_strict(row["due_local"], zone)
        row["due_local"] = local.isoformat(timespec="milliseconds")
        row["timezone"] = zone
    elif row["due_local"] != "" or row["timezone"] != "":
        raise ToolContractError()
    row["priority"] = _enum(row["priority"], {"NONE", "LOW", "MEDIUM", "HIGH"})
    return row


def _validate_note(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(value, {"title", "body"})
    row["title"] = normalize_single_line(row["title"], utf8_limit=360, utf16_limit=120)
    row["body"] = normalize_multiline(row["body"], utf8_limit=16384, utf16_limit=8192)
    return row


def _validate_email(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(value, {"to", "subject", "body"})
    if not isinstance(row["to"], list) or not 1 <= len(row["to"]) <= 10:
        raise ToolContractError("email.address_invalid")
    recipients = [normalize_email_address(item) for item in row["to"]]
    if len({item.casefold() for item in recipients}) != len(recipients):
        raise ToolContractError("email.address_invalid")
    row["to"] = recipients
    row["subject"] = normalize_single_line(row["subject"], utf8_limit=384, utf16_limit=128, allow_empty=True)
    row["body"] = normalize_multiline(row["body"], utf8_limit=32768, utf16_limit=16384, allow_empty=True)
    if not row["subject"] and not row["body"]:
        raise ToolContractError()
    return row


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return transform_common_schema({"type": "object", "properties": dict(properties), "required": list(properties), "additionalProperties": False})


def _string(maximum: int, *, minimum: int = 0, values: Sequence[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "string", "minLength": minimum, "maxLength": maximum}
    if values is not None:
        result["enum"] = list(values)
    return result


def _array(items: Mapping[str, Any], minimum: int, maximum: int) -> dict[str, Any]:
    return {"type": "array", "items": dict(items), "minItems": minimum, "maxItems": maximum}


@dataclass(frozen=True, slots=True)
class ToolDefinitionV1:
    tool_id: str
    side_effect: bool
    approval_required: bool
    timeout_seconds: int
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    preview_fields: tuple[str, ...]
    validate_input: Callable[[Mapping[str, Any]], dict[str, Any]]

    @property
    def schema_sha256(self) -> str:
        return domain_sha256(
            "jarvis.tool-schema/1",
            {"input": self.input_schema, "output": self.output_schema, "preview_fields": list(self.preview_fields), "tool_id": self.tool_id},
        )

    @property
    def output_schema_sha256(self) -> str:
        """Digest bound by Action Host v2 terminal results.

        This additive digest does not alter the frozen combined v1 tool-schema
        or registry digests.
        """

        return domain_sha256(
            "jarvis.tool-output-schema/1",
            {"output": self.output_schema, "tool_id": self.tool_id},
        )


def build_tool_definitions(
    *,
    root_ids: Sequence[str],
    app_ids: Sequence[str],
) -> tuple[ToolDefinitionV1, ...]:
    roots = tuple(sorted(set(root_ids)))
    apps = tuple(sorted(set(app_ids)))
    if not roots or not apps or any(not item or not item.isascii() for item in (*roots, *apps)):
        raise ToolContractError()
    uuid_schema = _string(36, minimum=36)
    timestamp = _string(24, minimum=24)
    local_timestamp = _string(23, minimum=23)
    cursor = _string(MAX_CURSOR_BYTES)
    page = {"limit": {"type": "integer", "minimum": 1, "maximum": 50}, "cursor": cursor}
    definitions: list[ToolDefinitionV1] = []

    definitions.append(ToolDefinitionV1(
        "file.search@1", False, False, 5,
        _object({"query": _string(128, minimum=1), "root_ids": _array(_string(max(map(len, roots)), values=roots), 1, 4), "kinds": _array(_string(9, values=("DIRECTORY", "FILE")), 1, 2), **page}),
        _object({"observed_at_utc": timestamp, "items": _array(_object({"file_ref": uuid_schema, "root_id": _string(max(map(len, roots)), values=roots), "display_name": _string(255), "relative_parent": _string(1024), "kind": _string(9, values=("FILE", "DIRECTORY")), "uti": _string(255, minimum=1), "size_bytes": {"type": "integer", "minimum": 0, "maximum": 2**63 - 1}, "modified_at_utc": timestamp, "ref_expires_at_utc": timestamp}), 0, 50), "next_cursor": cursor, "truncated": {"type": "boolean"}}),
        (), lambda value: _validate_file_search(value, set(roots)),
    ))
    definitions.append(ToolDefinitionV1(
        "file.open@1", True, True, 10,
        _object({"file_ref": uuid_schema}),
        _object({"file_ref": uuid_schema, "handler_id": _string(max(map(len, apps)), values=apps), "request_observed_at_utc": timestamp}),
        ("display_name", "root_label", "relative_parent", "uti", "size_bytes", "modified_at_utc", "handler_id", "identity_sha256", "ref_expires_at_utc"),
        lambda value: _validate_ref(value, "file_ref"),
    ))
    definitions.append(ToolDefinitionV1(
        "app.open@1", True, True, 10,
        _object({"app_id": _string(max(map(len, apps)), values=apps)}),
        _object({"app_id": _string(max(map(len, apps)), values=apps), "bundle_id": _string(255, minimum=1), "request_observed_at_utc": timestamp}),
        ("display_name", "bundle_id", "code_identity_sha256"),
        lambda value: _validate_app_open(value, set(apps)),
    ))
    definitions.append(ToolDefinitionV1(
        "calendar.read@1", False, True, 20,
        _object({"start_local": local_timestamp, "end_local": local_timestamp, "timezone": _string(64, minimum=1), **page}),
        _object({"observed_at_utc": timestamp, "events": _array(_object({"event_ref": uuid_schema, "calendar_label": _string(63), "title": _string(63), "start_local": local_timestamp, "end_local": local_timestamp, "start_utc": timestamp, "end_utc": timestamp, "timezone": _string(64, minimum=1), "all_day": {"type": "boolean"}, "location": _string(128), "notes": _string(2048)}), 0, 50), "next_cursor": cursor, "truncated": {"type": "boolean"}}),
        ("calendar_label", "start_local", "end_local", "timezone", "limit"),
        _validate_calendar_read,
    ))
    definitions.append(ToolDefinitionV1(
        "calendar.create@1", True, True, 20,
        _object({"title": _string(63, minimum=1), "start_local": local_timestamp, "end_local": local_timestamp, "timezone": _string(64, minimum=1), "location": _string(128), "notes": _string(2048), "availability": _string(4, values=("BUSY", "FREE"))}),
        _object({"event_ref": uuid_schema, "authorization_mode": _string(11, values=("WRITE_ONLY", "FULL_ACCESS")), "title": _string(63, minimum=1), "start_local": local_timestamp, "end_local": local_timestamp, "timezone": _string(64, minimum=1), "location": _string(128), "notes_sha256": _string(64, minimum=64), "availability": _string(4, values=("BUSY", "FREE"))}),
        ("calendar_label", "title", "start_local", "end_local", "timezone", "location", "notes", "availability", "authorization_mode", "expected_verification", "arguments_sha256"),
        _validate_calendar_create,
    ))
    definitions.append(ToolDefinitionV1(
        "reminder.list@1", False, True, 20,
        _object({"status": _string(10, values=("INCOMPLETE", "COMPLETED")), "due_filter": _string(6, values=("ANY", "WINDOW")), "due_from_local": _string(23), "due_to_local": _string(23), "timezone": _string(64), **page}),
        _object({"observed_at_utc": timestamp, "reminders": _array(_object({"reminder_ref": uuid_schema, "title": _string(63), "notes": _string(2048), "has_due": {"type": "boolean"}, "due_local": _string(23), "due_utc": _string(24), "timezone": _string(64), "is_completed": {"type": "boolean"}, "priority": _string(6, values=("NONE", "LOW", "MEDIUM", "HIGH"))}), 0, 50), "next_cursor": cursor, "truncated": {"type": "boolean"}}),
        ("status", "due_filter", "due_from_local", "due_to_local", "timezone", "limit"),
        _validate_reminder_list,
    ))
    definitions.append(ToolDefinitionV1(
        "reminder.create@1", True, True, 20,
        _object({"title": _string(63, minimum=1), "notes": _string(2048), "has_due": {"type": "boolean"}, "due_local": _string(23), "timezone": _string(64), "priority": _string(6, values=("NONE", "LOW", "MEDIUM", "HIGH"))}),
        _object({"reminder_ref": uuid_schema, "title": _string(63, minimum=1), "notes_sha256": _string(64, minimum=64), "has_due": {"type": "boolean"}, "due_local": _string(23), "due_utc": _string(24), "timezone": _string(64), "priority": _string(6, values=("NONE", "LOW", "MEDIUM", "HIGH"))}),
        ("title", "notes", "has_due", "due_local", "timezone", "priority", "arguments_sha256"),
        _validate_reminder_create,
    ))
    definitions.append(ToolDefinitionV1(
        "reminder.complete@1", True, True, 20,
        _object({"reminder_ref": uuid_schema}),
        _object({"reminder_ref": uuid_schema, "completed_at_utc": timestamp}),
        ("title", "has_due", "due_local", "timezone", "previous_completion_state", "reminder_ref", "identity_sha256"),
        lambda value: _validate_ref(value, "reminder_ref"),
    ))
    definitions.append(ToolDefinitionV1(
        "note.capture@1", True, True, 5,
        _object({"title": _string(120, minimum=1), "body": _string(8192, minimum=1)}),
        _object({"note_ref": uuid_schema, "title": _string(120, minimum=1), "body_sha256": _string(64, minimum=64), "created_at_utc": timestamp, "storage_domain": _string(20, values=("JARVIS_PRIVATE_NOTES",))}),
        ("title", "body", "utf8_bytes", "utf16_units", "storage_domain"),
        _validate_note,
    ))
    definitions.append(ToolDefinitionV1(
        "email.draft@1", True, True, 30,
        _object({"to": _array(_string(254, minimum=3), 1, 10), "subject": _string(128), "body": _string(16384)}),
        _object({"draft_ref": uuid_schema, "to": _array(_string(254, minimum=3), 1, 10), "subject": _string(128), "body_sha256": _string(64, minimum=64), "created_at_utc": timestamp}),
        ("to", "subject", "body"),
        _validate_email,
    ))
    return tuple(definitions)


@dataclass(frozen=True, slots=True)
class ToolRegistryV1:
    definitions: tuple[ToolDefinitionV1, ...]

    def __post_init__(self) -> None:
        names = [item.tool_id for item in self.definitions]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ToolContractError()
        forbidden = {"timer.create@1", "timer.list@1", "timer.cancel@1", "email.send@1", "shell@1"}
        if forbidden.intersection(names):
            raise ToolContractError()

    @classmethod
    def build(cls, *, root_ids: Sequence[str], app_ids: Sequence[str]) -> "ToolRegistryV1":
        return cls(tuple(sorted(build_tool_definitions(root_ids=root_ids, app_ids=app_ids), key=lambda item: item.tool_id)))

    @property
    def digest(self) -> str:
        return domain_sha256(
            TOOL_REGISTRY_CONTRACT,
            [{"approval_required": item.approval_required, "input": item.input_schema, "output": item.output_schema, "preview_fields": list(item.preview_fields), "schema_sha256": item.schema_sha256, "side_effect": item.side_effect, "timeout_seconds": item.timeout_seconds, "tool_id": item.tool_id} for item in self.definitions],
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.tool_id for item in self.definitions)

    def resolve(self, tool_id: str) -> ToolDefinitionV1:
        for definition in self.definitions:
            if definition.tool_id == tool_id:
                return definition
        raise ToolContractError("tool.unknown", status=404)

    def validate(self, tool_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        canonical = self.resolve(tool_id).validate_input(arguments)
        if len(jcs_bytes(canonical)) > MAX_ADAPTER_RESULT_BYTES:
            raise ToolContractError("tool.payload_too_large")
        return canonical

    def validate_output(self, tool_id: str, output: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(output, Mapping):
            raise ToolContractError("tool.output_invalid")
        selected = dict(output)
        _validate_schema_value(self.resolve(tool_id).output_schema, selected)
        if len(jcs_bytes(selected)) > MAX_ADAPTER_RESULT_BYTES:
            raise ToolContractError("tool.payload_too_large")
        return selected

    def model_projection(self, tool_id: str, output: Mapping[str, Any]) -> dict[str, Any]:
        """Return an explicit bounded local-model projection, never silent truncation."""

        selected = self.validate_output(tool_id, output)
        encoded = jcs_bytes(selected)
        if len(encoded) <= MAX_MODEL_PROJECTION_BYTES:
            return {"model_projection_truncated": False, "result": selected}
        return {
            "model_projection_truncated": True,
            "result_sha256": hashlib.sha256(encoded).hexdigest(),
            "result_size_bytes": len(encoded),
        }


def _validate_schema_value(schema: Mapping[str, Any], value: Any) -> None:
    selected = schema.get("type")
    if selected == "object":
        if not isinstance(value, Mapping) or set(value) != set(schema["properties"]):
            raise ToolContractError("tool.output_invalid")
        for name, child in schema["properties"].items():
            _validate_schema_value(child, value[name])
        return
    if selected == "array":
        if not isinstance(value, list) or not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 2**31):
            raise ToolContractError("tool.output_invalid")
        for item in value:
            _validate_schema_value(schema["items"], item)
        return
    if selected == "string":
        if not isinstance(value, str) or not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 2**31):
            raise ToolContractError("tool.output_invalid")
        if "enum" in schema and value not in schema["enum"]:
            raise ToolContractError("tool.output_invalid")
        _reject_forbidden_unicode(value, multiline="\n" in value)
        return
    if selected == "integer":
        if isinstance(value, bool) or not isinstance(value, int) or not schema.get("minimum", -(2**63)) <= value <= schema.get("maximum", 2**63 - 1):
            raise ToolContractError("tool.output_invalid")
        return
    if selected == "boolean":
        if not isinstance(value, bool):
            raise ToolContractError("tool.output_invalid")
        return
    raise ToolContractError("tool.output_invalid")


def arguments_sha256(arguments: Mapping[str, Any]) -> str:
    return hashlib.sha256(jcs_bytes(arguments)).hexdigest()
