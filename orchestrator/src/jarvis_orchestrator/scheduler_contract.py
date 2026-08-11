"""Frozen deterministic scheduler proposal, time, and identity contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import io
import re
import sys
import unicodedata
from calendar import monthrange
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from functools import lru_cache
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from .domain import OrchestratorError, parse_utc, require_uuid4, utc_text
from .jcs import domain_sha256, jcs_bytes, jcs_text


PROPOSAL_INPUT_SCHEMA = "schedule-proposal-input/1"
PROPOSAL_SCHEMA = "schedule-proposal/1"
SCHEDULE_CONTRACT = "schedule-contract/1"
RECURRENCE_CONTRACT = "scheduler-recurrence/1"
CPYTHON_VERSION = (3, 11, 14)
UNICODE_VERSION = "14.0.0"
TZDATA_VERSION = "2026.3"
TZDATA_WHEEL_SHA256 = "dc096730c87af6cab1b171c9d532be840741ff5d459015e7f6947bd7d7e54931"
DEFAULT_TIMEZONE = "Europe/Athens"
DEFAULT_LOCALE = "el-GR"
LOCALES = frozenset({"el-GR", "en-US"})
MINIMUM_LEAD_MS = 10_000
MINIMUM_INTERVAL_MS = 60_000
MAXIMUM_SNOOZE_GENERATIONS = 8
MAXIMUM_ACTIVE_JOBS = 256
MAXIMUM_QUERY_HORIZON_DAYS = 366
DEFAULT_PAGE_SIZE = 50
MAXIMUM_PAGE_SIZE = 100
APPROVAL_TTL_SECONDS = 600
MISSED_GRACE_MS = 24 * 60 * 60 * 1000
CONTENT_RETENTION_MS = MISSED_GRACE_MS
MANUAL_CANONICAL_API_SHA256 = hashlib.sha256(b"jarvis.scheduler.manual-canonical-api/1").hexdigest()


class ProposalOutcome(StrEnum):
    PROPOSAL = "PROPOSAL"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"


class ProposalOperation(StrEnum):
    CREATE_JOB = "CREATE_JOB"
    CANCEL_JOB = "CANCEL_JOB"
    SNOOZE_OCCURRENCE = "SNOOZE_OCCURRENCE"
    NONE = "NONE"


class TemporalKind(StrEnum):
    NONE = "NONE"
    ABSOLUTE_LOCAL = "ABSOLUTE_LOCAL"
    CALENDAR_DATE = "CALENDAR_DATE"
    AFTER_DURATION = "AFTER_DURATION"
    RELATIVE_DAY = "RELATIVE_DAY"
    WEEKDAY = "WEEKDAY"


class RecurrenceKind(StrEnum):
    NONE = "NONE"
    FIXED_INTERVAL = "FIXED_INTERVAL"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class ScheduleContractError(OrchestratorError):
    def __init__(self, code: str = "proposal_invalid", *, status: int = 400) -> None:
        super().__init__(code, status=status)


TEMPORAL_FIELDS = {
    "kind",
    "local_datetime",
    "year",
    "month",
    "day",
    "local_time",
    "timezone",
    "amount",
    "unit",
    "relation",
    "weekday",
}
RECURRENCE_FIELDS = {
    "kind",
    "interval_amount",
    "interval_unit",
    "weekdays",
    "day_of_month",
    "count",
    "until",
}
NOTIFICATION_FIELDS = {"title", "body", "speakable_text", "locale", "visibility"}
PROPOSAL_FIELDS = {
    "schema_version",
    "request_id",
    "outcome",
    "operation",
    "target_job_id",
    "target_job_revision",
    "target_occurrence_id",
    "time_expression",
    "recurrence",
    "notification",
    "clarification_reason",
    "clarification_question",
}
INPUT_FIELDS = {
    "schema_version",
    "request_id",
    "request_text",
    "request_received_at_utc",
    "default_timezone",
    "default_locale",
    "selected_target",
    "policy",
}
TARGET_FIELDS = {"job_id", "job_revision", "occurrence_id", "snooze_generation"}
POLICY_FIELDS = {"minimum_lead_ms", "minimum_interval_ms", "maximum_snooze_generations"}
CLARIFICATION_REASONS = frozenset({
    "NONE", "MISSING_TIME", "AMBIGUOUS_DATE", "AMBIGUOUS_TIME", "AMBIGUOUS_TIMEZONE",
    "AMBIGUOUS_RECURRENCE", "AMBIGUOUS_TARGET", "CONFLICTING_FIELDS",
    "MISSING_NOTIFICATION_TEXT", "UNSUPPORTED_TEMPORAL_EXPRESSION", "UNSUPPORTED_RECURRENCE",
    "TARGET_BINDING_MISMATCH", "PROPOSAL_INVALID",
})


def _exact(value: Any, fields: set[str], code: str = "proposal_invalid") -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ScheduleContractError(code)
    return dict(value)


def _integer(value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScheduleContractError()
    if minimum is not None and value < minimum:
        raise ScheduleContractError()
    if maximum is not None and value > maximum:
        raise ScheduleContractError()
    return value


def _nullable_integer(value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int | None:
    return None if value is None else _integer(value, minimum=minimum, maximum=maximum)


def _utf16_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def canonical_content(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ScheduleContractError("invalid_notification_content")
    normalized = unicodedata.normalize("NFC", value)
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs", "Co", "Cn"} for character in normalized):
        raise ScheduleContractError("invalid_notification_content")
    folded = re.sub(r"\s+", " ", normalized, flags=re.UNICODE).strip()
    folded = unicodedata.normalize("NFC", folded)
    byte_limit, unit_limit = {
        "title": (192, 63),
        "body": (768, 255),
        "speakable_text": (960, 240),
        "clarification_question": (960, 240),
    }[field]
    if not folded or len(folded.encode("utf-8")) > byte_limit or _utf16_units(folded) > unit_limit:
        raise ScheduleContractError("invalid_notification_content")
    return folded


def validate_runtime() -> None:
    if sys.version_info[:3] != CPYTHON_VERSION or unicodedata.unidata_version != UNICODE_VERSION:
        raise ScheduleContractError("unicode_runtime_mismatch", status=503)


def epoch_ms(value: datetime) -> int:
    if value.tzinfo is None:
        raise ScheduleContractError("timestamp_timezone_required")
    utc = value.astimezone(UTC)
    if utc.microsecond % 1000:
        raise ScheduleContractError("submillisecond_precision_forbidden")
    return int(utc.timestamp() * 1000)


def from_epoch_ms(value: int) -> datetime:
    return datetime.fromtimestamp(_integer(value) / 1000, tz=UTC)


def strict_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value):
        raise ScheduleContractError("invalid_timestamp")
    parsed = parse_utc(value)
    epoch_ms(parsed)
    return parsed


def _strict_local_datetime(value: Any) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}", value):
        raise ScheduleContractError("invalid_local_datetime")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ScheduleContractError("invalid_local_datetime") from exc
    if parsed.tzinfo is not None or parsed.microsecond % 1000:
        raise ScheduleContractError("invalid_local_datetime")
    return parsed


def _strict_local_time(value: Any) -> time:
    if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}:\d{2}\.\d{3}", value):
        raise ScheduleContractError("invalid_local_time")
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise ScheduleContractError("invalid_local_time") from exc


class PackagedTimezoneProvider:
    """Loads exact TZif resources from the pinned first-party tzdata wheel."""

    def __init__(self, expected_distribution: str = TZDATA_VERSION) -> None:
        try:
            actual = importlib.metadata.version("tzdata")
            import tzdata
        except (importlib.metadata.PackageNotFoundError, ImportError) as exc:
            raise ScheduleContractError("tzdata_unavailable", status=503) from exc
        if actual != expected_distribution:
            raise ScheduleContractError("tzdata_version_mismatch", status=503)
        self.distribution_version = actual
        self.iana_version = str(getattr(tzdata, "IANA_VERSION", ""))
        if not self.iana_version:
            raise ScheduleContractError("tzdata_version_mismatch", status=503)

    @staticmethod
    def _segments(key: str) -> tuple[str, ...]:
        if not isinstance(key, str) or not key or key.startswith("/") or "\\" in key:
            raise ScheduleContractError("invalid_timezone")
        segments = tuple(key.split("/"))
        if any(not item or item in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_+\-]+", item) for item in segments):
            raise ScheduleContractError("invalid_timezone")
        return segments

    @lru_cache(maxsize=128)
    def load(self, key: str) -> tuple[ZoneInfo, str]:
        resource = importlib.resources.files("tzdata.zoneinfo")
        for segment in self._segments(key):
            resource = resource.joinpath(segment)
        try:
            raw = resource.read_bytes()
        except (FileNotFoundError, IsADirectoryError, OSError) as exc:
            raise ScheduleContractError("invalid_timezone") from exc
        digest = hashlib.sha256(raw).hexdigest()
        try:
            zone = ZoneInfo.from_file(io.BytesIO(raw), key=key)
        except Exception as exc:
            raise ScheduleContractError("invalid_timezone") from exc
        return zone, digest


@dataclass(frozen=True, slots=True)
class ProposalInput:
    request_id: str
    request_text: str
    request_received_at_utc: str
    default_timezone: str
    default_locale: str
    selected_target: Mapping[str, Any]

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ProposalInput":
        value = _exact(raw, INPUT_FIELDS)
        if value["schema_version"] != PROPOSAL_INPUT_SCHEMA:
            raise ScheduleContractError()
        request_id = require_uuid4(value["request_id"])
        request = value["request_text"]
        if not isinstance(request, str):
            raise ScheduleContractError()
        request = unicodedata.normalize("NFC", request).strip()
        if not request or len(request.encode("utf-8")) > 4096:
            raise ScheduleContractError()
        strict_utc(value["request_received_at_utc"])
        timezone_key = value["default_timezone"]
        locale = value["default_locale"]
        if not isinstance(timezone_key, str) or locale not in LOCALES:
            raise ScheduleContractError()
        target = _exact(value["selected_target"], TARGET_FIELDS)
        if target["job_id"] is not None:
            require_uuid4(target["job_id"])
        if target["occurrence_id"] is not None and (
            not isinstance(target["occurrence_id"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", target["occurrence_id"])
        ):
            raise ScheduleContractError()
        _nullable_integer(target["job_revision"], minimum=1)
        _nullable_integer(target["snooze_generation"], minimum=0, maximum=MAXIMUM_SNOOZE_GENERATIONS)
        policy = _exact(value["policy"], POLICY_FIELDS)
        if policy != {
            "maximum_snooze_generations": MAXIMUM_SNOOZE_GENERATIONS,
            "minimum_interval_ms": MINIMUM_INTERVAL_MS,
            "minimum_lead_ms": MINIMUM_LEAD_MS,
        }:
            raise ScheduleContractError()
        return cls(request_id, request, value["request_received_at_utc"], timezone_key, locale, target)


@dataclass(frozen=True, slots=True)
class TemporalExpression:
    raw: Mapping[str, Any]

    @classmethod
    def parse(cls, raw: Any) -> "TemporalExpression":
        value = _exact(raw, TEMPORAL_FIELDS)
        try:
            kind = TemporalKind(value["kind"])
        except (ValueError, TypeError) as exc:
            raise ScheduleContractError() from exc
        if value["timezone"] is not None and not isinstance(value["timezone"], str):
            raise ScheduleContractError()
        for name in ("local_datetime", "local_time", "unit", "relation"):
            if value[name] is not None and not isinstance(value[name], str):
                raise ScheduleContractError()
        _nullable_integer(value["year"], minimum=1, maximum=9999)
        _nullable_integer(value["month"], minimum=1, maximum=12)
        _nullable_integer(value["day"], minimum=1, maximum=31)
        _nullable_integer(value["amount"], minimum=1, maximum=10_000_000)
        _nullable_integer(value["weekday"], minimum=1, maximum=7)
        if kind is TemporalKind.ABSOLUTE_LOCAL:
            _strict_local_datetime(value["local_datetime"])
        elif kind is TemporalKind.CALENDAR_DATE:
            if value["month"] is None or value["day"] is None or value["local_time"] is None:
                raise ScheduleContractError("missing_time")
            _strict_local_time(value["local_time"])
        elif kind is TemporalKind.AFTER_DURATION:
            if value["amount"] is None or value["unit"] not in {"SECOND", "MINUTE", "HOUR", "DAY", "WEEK"}:
                raise ScheduleContractError("missing_time")
        elif kind is TemporalKind.RELATIVE_DAY:
            if value["relation"] not in {"TODAY", "TOMORROW"} or value["local_time"] is None:
                raise ScheduleContractError("missing_time")
            _strict_local_time(value["local_time"])
        elif kind is TemporalKind.WEEKDAY:
            if value["relation"] not in {"THIS", "NEXT", "UPCOMING"} or value["weekday"] is None or value["local_time"] is None:
                raise ScheduleContractError("missing_time")
            _strict_local_time(value["local_time"])
        permitted = {
            TemporalKind.NONE: {"kind"},
            TemporalKind.ABSOLUTE_LOCAL: {"kind", "local_datetime", "timezone"},
            TemporalKind.CALENDAR_DATE: {"kind", "year", "month", "day", "local_time", "timezone"},
            TemporalKind.AFTER_DURATION: {"kind", "amount", "unit"},
            TemporalKind.RELATIVE_DAY: {"kind", "local_time", "timezone", "relation"},
            TemporalKind.WEEKDAY: {"kind", "local_time", "timezone", "relation", "weekday"},
        }[kind]
        if any(value[name] is not None for name in TEMPORAL_FIELDS - permitted):
            raise ScheduleContractError("conflicting_fields")
        return cls(value)

    @property
    def kind(self) -> TemporalKind:
        return TemporalKind(self.raw["kind"])


@dataclass(frozen=True, slots=True)
class CompiledProposal:
    proposal: Mapping[str, Any]
    proposal_sha256: str
    first_due_at_utc: str | None
    timezone_tzif_sha256: str | None


def _duration_ms(amount: int, unit: str) -> int:
    multiplier = {
        "SECOND": 1000,
        "MINUTE": 60_000,
        "HOUR": 3_600_000,
        "DAY": 86_400_000,
        "WEEK": 604_800_000,
    }[unit]
    return amount * multiplier


def _valid_local_instants(local: datetime, zone: ZoneInfo) -> list[datetime]:
    results: list[datetime] = []
    for fold in (0, 1):
        candidate = local.replace(tzinfo=zone, fold=fold)
        roundtrip = candidate.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
        if roundtrip == local:
            utc = candidate.astimezone(UTC)
            if utc not in results:
                results.append(utc)
    return sorted(results)


def resolve_temporal(
    expression: TemporalExpression,
    *,
    reference_at_utc: datetime,
    default_timezone: str,
    zones: PackagedTimezoneProvider,
    one_shot: bool,
    recurrence: Mapping[str, Any] | None = None,
) -> tuple[datetime | None, str, str]:
    raw = expression.raw
    timezone_key = raw["timezone"] or default_timezone
    zone, tzif_sha256 = zones.load(timezone_key)
    reference_local = reference_at_utc.astimezone(zone)
    if expression.kind is TemporalKind.NONE:
        return None, timezone_key, tzif_sha256
    if expression.kind is TemporalKind.AFTER_DURATION:
        resolved = reference_at_utc + timedelta(milliseconds=_duration_ms(raw["amount"], raw["unit"]))
        return resolved, timezone_key, tzif_sha256
    if expression.kind is TemporalKind.ABSOLUTE_LOCAL:
        local = _strict_local_datetime(raw["local_datetime"])
    elif expression.kind is TemporalKind.CALENDAR_DATE:
        year = raw["year"]
        if year is None:
            year = reference_local.year
            while True:
                try:
                    candidate_date = date(year, raw["month"], raw["day"])
                except ValueError:
                    year += 1
                    continue
                local = datetime.combine(candidate_date, _strict_local_time(raw["local_time"]))
                instants = _valid_local_instants(local, zone)
                if instants and instants[0] > reference_at_utc:
                    break
                year += 1
        else:
            try:
                local = datetime.combine(date(year, raw["month"], raw["day"]), _strict_local_time(raw["local_time"]))
            except ValueError as exc:
                raise ScheduleContractError("ambiguous_date") from exc
    elif expression.kind is TemporalKind.RELATIVE_DAY:
        target_date = reference_local.date() + timedelta(days=1 if raw["relation"] == "TOMORROW" else 0)
        local = datetime.combine(target_date, _strict_local_time(raw["local_time"]))
    elif expression.kind is TemporalKind.WEEKDAY:
        desired = raw["weekday"]
        if raw["relation"] == "THIS":
            target_date = reference_local.date() + timedelta(days=desired - reference_local.isoweekday())
        elif raw["relation"] == "NEXT":
            target_date = reference_local.date() + timedelta(days=(8 - reference_local.isoweekday()) + desired - 1)
        else:
            delta = (desired - reference_local.isoweekday()) % 7
            target_date = reference_local.date() + timedelta(days=delta)
            tentative = datetime.combine(target_date, _strict_local_time(raw["local_time"]))
            instants = _valid_local_instants(tentative, zone)
            if not instants or instants[0] < reference_at_utc + timedelta(milliseconds=MINIMUM_LEAD_MS):
                target_date += timedelta(days=7)
        local = datetime.combine(target_date, _strict_local_time(raw["local_time"]))
    else:  # pragma: no cover - exhaustive enum guard
        raise ScheduleContractError("unsupported_temporal_expression")
    instants = _valid_local_instants(local, zone)
    if not instants:
        if one_shot or recurrence is None:
            raise ScheduleContractError("one_shot_dst_gap")
        recurrence_kind = RecurrenceKind(recurrence["kind"])
        for _ in range(370):
            if recurrence_kind is RecurrenceKind.DAILY:
                local += timedelta(days=1)
            elif recurrence_kind is RecurrenceKind.WEEKLY:
                weekdays = sorted(set(recurrence["weekdays"]))
                if not weekdays:
                    raise ScheduleContractError("ambiguous_recurrence")
                while True:
                    local += timedelta(days=1)
                    if local.isoweekday() in weekdays:
                        break
            elif recurrence_kind is RecurrenceKind.MONTHLY:
                day = recurrence["day_of_month"] or local.day
                year, month = local.year, local.month
                while True:
                    month += 1
                    if month == 13:
                        year += 1
                        month = 1
                    if day <= monthrange(year, month)[1]:
                        local = local.replace(year=year, month=month, day=day)
                        break
            elif recurrence_kind is RecurrenceKind.FIXED_INTERVAL:
                local += timedelta(milliseconds=_duration_ms(
                    recurrence["interval_amount"], recurrence["interval_unit"]
                ))
            else:
                raise ScheduleContractError("recurring_dst_gap")
            instants = _valid_local_instants(local, zone)
            if instants:
                break
        if not instants:
            raise ScheduleContractError("recurring_dst_gap")
    return instants[0], timezone_key, tzif_sha256


def _parse_notification(raw: Any, *, operation: ProposalOperation) -> dict[str, Any]:
    value = _exact(raw, NOTIFICATION_FIELDS)
    if operation is not ProposalOperation.CREATE_JOB:
        if any(value[name] is not None for name in ("title", "body", "speakable_text")):
            raise ScheduleContractError("conflicting_fields")
        if value["locale"] not in LOCALES or value["visibility"] not in {"private", "title_only", "full"}:
            raise ScheduleContractError()
        return value
    title = canonical_content(value["title"], field="title")
    body = canonical_content(value["body"], field="body")
    speakable = None if value["speakable_text"] is None else canonical_content(value["speakable_text"], field="speakable_text")
    if value["locale"] not in LOCALES or value["visibility"] not in {"private", "title_only", "full"}:
        raise ScheduleContractError()
    return {**value, "title": title, "body": body, "speakable_text": speakable}


def compile_proposal(
    input_raw: Mapping[str, Any],
    output_raw: Mapping[str, Any],
    *,
    compiler_qualification_sha256: str,
    zones: PackagedTimezoneProvider,
    enforce_runtime: bool = True,
) -> CompiledProposal:
    if not isinstance(compiler_qualification_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", compiler_qualification_sha256
    ):
        raise ScheduleContractError("compiler_qualification_invalid", status=503)
    if enforce_runtime:
        validate_runtime()
    proposal_input = ProposalInput.parse(input_raw)
    output = _exact(output_raw, PROPOSAL_FIELDS)
    if output["schema_version"] != PROPOSAL_SCHEMA or output["request_id"] != proposal_input.request_id:
        raise ScheduleContractError()
    try:
        outcome = ProposalOutcome(output["outcome"])
        operation = ProposalOperation(output["operation"])
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError() from exc
    expression = TemporalExpression.parse(output["time_expression"])
    recurrence = _exact(output["recurrence"], RECURRENCE_FIELDS)
    until = TemporalExpression.parse(recurrence["until"])
    try:
        recurrence_kind = RecurrenceKind(recurrence["kind"])
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError() from exc
    weekdays = recurrence["weekdays"]
    if not isinstance(weekdays, list) or any(isinstance(day, bool) or not isinstance(day, int) or not 1 <= day <= 7 for day in weekdays):
        raise ScheduleContractError()
    weekdays = sorted(set(weekdays))
    count = _nullable_integer(recurrence["count"], minimum=1, maximum=100_000)
    _nullable_integer(recurrence["day_of_month"], minimum=1, maximum=31)
    _nullable_integer(recurrence["interval_amount"], minimum=1, maximum=10_000_000)
    if recurrence["interval_unit"] is not None and recurrence["interval_unit"] not in {"MINUTE", "HOUR", "DAY", "WEEK"}:
        raise ScheduleContractError()
    if recurrence_kind is RecurrenceKind.NONE:
        if any(recurrence[name] is not None for name in ("interval_amount", "interval_unit", "day_of_month", "count")) or weekdays or until.kind is not TemporalKind.NONE:
            raise ScheduleContractError("conflicting_fields")
    elif recurrence_kind is RecurrenceKind.FIXED_INTERVAL:
        if weekdays or recurrence["day_of_month"] is not None:
            raise ScheduleContractError("conflicting_fields")
    elif recurrence_kind is RecurrenceKind.DAILY:
        if recurrence["interval_amount"] is not None or recurrence["interval_unit"] is not None or weekdays or recurrence["day_of_month"] is not None:
            raise ScheduleContractError("conflicting_fields")
    elif recurrence_kind is RecurrenceKind.WEEKLY:
        if recurrence["interval_amount"] is not None or recurrence["interval_unit"] is not None or recurrence["day_of_month"] is not None:
            raise ScheduleContractError("conflicting_fields")
    elif recurrence_kind is RecurrenceKind.MONTHLY:
        if recurrence["interval_amount"] is not None or recurrence["interval_unit"] is not None or weekdays:
            raise ScheduleContractError("conflicting_fields")
    notification = _parse_notification(output["notification"], operation=operation)
    if output["clarification_reason"] not in CLARIFICATION_REASONS:
        raise ScheduleContractError()
    if outcome is not ProposalOutcome.PROPOSAL:
        if operation is not ProposalOperation.NONE or output["clarification_reason"] == "NONE":
            raise ScheduleContractError()
        question = output["clarification_question"]
        if question is not None:
            question = canonical_content(question, field="clarification_question")
        canonical = {**output, "clarification_question": question, "notification": notification}
        digest = domain_sha256(PROPOSAL_SCHEMA, canonical)
        return CompiledProposal(canonical, digest, None, None)
    if output["clarification_reason"] != "NONE" or output["clarification_question"] is not None:
        raise ScheduleContractError("conflicting_fields")
    if operation is ProposalOperation.CREATE_JOB and any(
        output[name] is not None for name in ("target_job_id", "target_job_revision", "target_occurrence_id")
    ):
        raise ScheduleContractError("conflicting_fields")
    target = proposal_input.selected_target
    if operation in {ProposalOperation.CANCEL_JOB, ProposalOperation.SNOOZE_OCCURRENCE}:
        if target["job_id"] is None or target["job_revision"] is None:
            raise ScheduleContractError("ambiguous_target")
        if output["target_job_id"] != target["job_id"] or output["target_job_revision"] != target["job_revision"]:
            raise ScheduleContractError("target_binding_mismatch")
    if operation is ProposalOperation.SNOOZE_OCCURRENCE:
        if target["occurrence_id"] is None or target["snooze_generation"] is None:
            raise ScheduleContractError("ambiguous_target")
        if output["target_occurrence_id"] != target["occurrence_id"]:
            raise ScheduleContractError("target_binding_mismatch")
    reference = strict_utc(proposal_input.request_received_at_utc)
    first, timezone_key, tzif_sha256 = resolve_temporal(
        expression,
        reference_at_utc=reference,
        default_timezone=proposal_input.default_timezone,
        zones=zones,
        one_shot=recurrence_kind is RecurrenceKind.NONE,
        recurrence=recurrence,
    )
    if operation is ProposalOperation.CREATE_JOB and first is None:
        raise ScheduleContractError("missing_time")
    until_at: datetime | None = None
    if until.kind is not TemporalKind.NONE:
        until_at, until_zone, until_hash = resolve_temporal(
            until,
            reference_at_utc=reference,
            default_timezone=proposal_input.default_timezone,
            zones=zones,
            one_shot=True,
        )
        if until_zone != timezone_key or until_hash != tzif_sha256:
            raise ScheduleContractError("conflicting_fields")
    if count is not None and until_at is not None:
        raise ScheduleContractError("conflicting_fields")
    schedule: dict[str, Any]
    if operation in {ProposalOperation.CANCEL_JOB}:
        if expression.kind is not TemporalKind.NONE or recurrence_kind is not RecurrenceKind.NONE:
            raise ScheduleContractError("conflicting_fields")
        schedule = {"kind": "NONE"}
    elif operation is ProposalOperation.SNOOZE_OCCURRENCE:
        if recurrence_kind is not RecurrenceKind.NONE or first is None:
            raise ScheduleContractError("conflicting_fields")
        schedule = {"kind": "SNOOZE", "snooze_until_at_utc": utc_text(first)}
    elif recurrence_kind is RecurrenceKind.NONE:
        schedule = {
            "count": None,
            "dtstart_at_utc": utc_text(first),
            "kind": "ONE_SHOT",
            "recurrence_contract_version": RECURRENCE_CONTRACT,
            "timezone": timezone_key,
            "until_at_utc": None,
        }
    elif recurrence_kind is RecurrenceKind.FIXED_INTERVAL:
        if recurrence["interval_amount"] is None or recurrence["interval_unit"] is None:
            raise ScheduleContractError("ambiguous_recurrence")
        interval_ms = _duration_ms(recurrence["interval_amount"], recurrence["interval_unit"])
        if interval_ms < MINIMUM_INTERVAL_MS:
            raise ScheduleContractError("interval_too_short")
        schedule = {
            "count": count,
            "dtstart_at_utc": utc_text(first),
            "interval_ms": interval_ms,
            "kind": "FIXED_INTERVAL",
            "recurrence_contract_version": RECURRENCE_CONTRACT,
            "timezone": timezone_key,
            "until_at_utc": utc_text(until_at) if until_at else None,
        }
    else:
        assert first is not None
        local_start = first.astimezone(zones.load(timezone_key)[0]).replace(tzinfo=None)
        if recurrence_kind is RecurrenceKind.WEEKLY and not weekdays:
            raise ScheduleContractError("ambiguous_recurrence")
        if recurrence_kind is RecurrenceKind.WEEKLY and local_start.isoweekday() not in weekdays:
            raise ScheduleContractError("conflicting_fields")
        day_of_month = recurrence["day_of_month"]
        if recurrence_kind is RecurrenceKind.MONTHLY:
            day_of_month = day_of_month or local_start.day
            if local_start.day != day_of_month:
                raise ScheduleContractError("conflicting_fields")
        schedule = {
            "count": count,
            "day_of_month": day_of_month,
            "dtstart_local": local_start.isoformat(timespec="milliseconds"),
            "kind": f"CALENDAR_{recurrence_kind.value}",
            "missing_day_policy": "skip" if recurrence_kind is RecurrenceKind.MONTHLY else None,
            "recurrence_contract_version": RECURRENCE_CONTRACT,
            "timezone": timezone_key,
            "until_at_utc": utc_text(until_at) if until_at else None,
            "weekdays": weekdays,
        }
    timezone_contract = {
        "distribution": TZDATA_VERSION,
        "iana_version": zones.iana_version,
        "tzif_sha256": tzif_sha256,
        "wheel_sha256": TZDATA_WHEEL_SHA256,
    }
    if operation is ProposalOperation.CREATE_JOB:
        assert first is not None
        resolved_local = first.astimezone(zones.load(timezone_key)[0])
        schedule["dtstart_fold"] = resolved_local.fold
        schedule.setdefault(
            "dtstart_local", resolved_local.replace(tzinfo=None).isoformat(timespec="milliseconds")
        )
        schedule["timezone_contract"] = timezone_contract
    first_due = first if operation is not ProposalOperation.CANCEL_JOB else None
    canonical = {
        "compiler_qualification_sha256": compiler_qualification_sha256,
        "content": notification,
        "operation": operation.value,
        "policy": {
            "approval_ttl_seconds": APPROVAL_TTL_SECONDS,
            "maximum_snooze_generations": MAXIMUM_SNOOZE_GENERATIONS,
            "minimum_lead_ms": MINIMUM_LEAD_MS,
            "missed_grace_ms": MISSED_GRACE_MS,
        },
        "request_id": proposal_input.request_id,
        "request_received_at_utc": proposal_input.request_received_at_utc,
        "schedule": schedule,
        "schema_version": SCHEDULE_CONTRACT,
        "target_job_id": output["target_job_id"],
        "target_job_revision": output["target_job_revision"],
        "target_occurrence_id": output["target_occurrence_id"],
        "target_snooze_generation": target["snooze_generation"],
        "timezone_contract": timezone_contract,
        "unicode_contract": {"cpython": "3.11.14", "unidata_version": UNICODE_VERSION},
    }
    return CompiledProposal(
        proposal=canonical,
        proposal_sha256=domain_sha256(SCHEDULE_CONTRACT, canonical),
        first_due_at_utc=utc_text(first_due) if first_due else None,
        timezone_tzif_sha256=tzif_sha256,
    )


def ensure_minimum_lead(first_due_at_utc: str, authoritative_now: datetime) -> None:
    due = strict_utc(first_due_at_utc)
    if epoch_ms(due) - epoch_ms(authoritative_now) < MINIMUM_LEAD_MS:
        raise ScheduleContractError("minimum_lead_violated", status=410)


def occurrence_id(job_id: str, job_revision: int, scheduled_at_utc: str) -> str:
    require_uuid4(job_id)
    strict_utc(scheduled_at_utc)
    return hashlib.sha256(
        jcs_bytes(["jarvis.scheduler-occurrence/1", job_id, _integer(job_revision, minimum=1), scheduled_at_utc])
    ).hexdigest()


def notification_id(occurrence: str, delivery_generation: int) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", occurrence):
        raise ScheduleContractError("invalid_occurrence_id")
    return hashlib.sha256(
        jcs_bytes(["jarvis.scheduler-notification/1", occurrence, _integer(delivery_generation, minimum=0)])
    ).hexdigest()


def next_calendar_local(schedule: Mapping[str, Any], current_local: datetime) -> datetime:
    kind = schedule["kind"]
    if kind == "CALENDAR_DAILY":
        return current_local + timedelta(days=1)
    if kind == "CALENDAR_WEEKLY":
        weekdays = list(schedule["weekdays"])
        for offset in range(1, 8):
            candidate = current_local + timedelta(days=offset)
            if candidate.isoweekday() in weekdays:
                return candidate
        raise ScheduleContractError("invalid_recurrence")
    if kind == "CALENDAR_MONTHLY":
        year, month = current_local.year, current_local.month
        day = int(schedule["day_of_month"])
        while True:
            month += 1
            if month == 13:
                year += 1
                month = 1
            if day <= monthrange(year, month)[1]:
                return current_local.replace(year=year, month=month, day=day)
    raise ScheduleContractError("invalid_recurrence")


def compiler_qualification_digest(value: Mapping[str, Any]) -> str:
    required = {
        "compiler_artifact_sha256",
        "input_schema_sha256",
        "model_artifact_sha256",
        "model_profile_sha256",
        "output_schema_sha256",
        "prompt_template_sha256",
        "qualification_report_sha256",
        "runtime_sha256",
        "tokenizer_sha256",
    }
    if set(value) != required or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in value.values()):
        raise ScheduleContractError("compiler_qualification_invalid")
    artifact = {
        **dict(value),
        "contract": "schedule-compiler-qualification/1",
        "cpython": "3.11.14",
        "inference_parameters": {
            "max_output_tokens": 512,
            "retries": 0,
            "temperature": 0,
            "tools": [],
            "top_p": 1,
        },
        "tzdata": TZDATA_VERSION,
        "tzdata_wheel_sha256": TZDATA_WHEEL_SHA256,
        "unicode": UNICODE_VERSION,
    }
    return domain_sha256("schedule-compiler-qualification/1", artifact)
