"""Single-shot, local-only model adapter for ``schedule-proposal/1``."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .jcs import jcs_bytes
from .scheduler_contract import (
    PackagedTimezoneProvider,
    ProposalInput,
    ScheduleContractError,
    CompiledProposal,
    compile_proposal,
    compiler_qualification_digest,
)


PROMPT_TEMPLATE = (
    "Convert exactly the supplied current explicit reminder request into schedule-proposal/1. "
    "Do not invent targets or infer database identifiers. Use NEEDS_CLARIFICATION for ambiguity. "
    "Return only the strict JSON object."
)
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()

INPUT_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "request_id", "request_text", "request_received_at_utc",
                 "default_timezone", "default_locale", "selected_target", "policy"],
    "properties": {
        "schema_version": {"const": "schedule-proposal-input/1"},
        "request_id": {"type": "string", "format": "uuid"},
        "request_text": {"type": "string", "minLength": 1},
        "request_received_at_utc": {"type": "string"},
        "default_timezone": {"type": "string"},
        "default_locale": {"enum": ["el-GR", "en-US"]},
        "selected_target": {
            "type": "object", "additionalProperties": False,
            "required": ["job_id", "job_revision", "occurrence_id", "snooze_generation"],
            "properties": {
                "job_id": {"anyOf": [{"type": "string", "format": "uuid"}, {"type": "null"}]},
                "job_revision": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
                "occurrence_id": {"anyOf": [{"type": "string", "pattern": "^[0-9a-f]{64}$"}, {"type": "null"}]},
                "snooze_generation": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 8}, {"type": "null"}]},
            },
        },
        "policy": {
            "type": "object", "additionalProperties": False,
            "required": ["minimum_lead_ms", "minimum_interval_ms", "maximum_snooze_generations"],
            "properties": {
                "minimum_lead_ms": {"const": 10000},
                "minimum_interval_ms": {"const": 60000},
                "maximum_snooze_generations": {"const": 8},
            },
        },
    },
}


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {"anyOf": [dict(schema), {"type": "null"}]}


def _temporal_schema() -> dict[str, Any]:
    properties = {
        "kind": {"enum": ["NONE", "ABSOLUTE_LOCAL", "CALENDAR_DATE", "AFTER_DURATION", "RELATIVE_DAY", "WEEKDAY"]},
        "local_datetime": _nullable({"type": "string"}),
        "year": _nullable({"type": "integer", "minimum": 1, "maximum": 9999}),
        "month": _nullable({"type": "integer", "minimum": 1, "maximum": 12}),
        "day": _nullable({"type": "integer", "minimum": 1, "maximum": 31}),
        "local_time": _nullable({"type": "string"}),
        "timezone": _nullable({"type": "string"}),
        "amount": _nullable({"type": "integer", "minimum": 1}),
        "unit": _nullable({"enum": ["SECOND", "MINUTE", "HOUR", "DAY", "WEEK"]}),
        "relation": _nullable({"enum": ["TODAY", "TOMORROW", "THIS", "NEXT", "UPCOMING"]}),
        "weekday": _nullable({"type": "integer", "minimum": 1, "maximum": 7}),
    }
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


TEMPORAL_JSON_SCHEMA = _temporal_schema()
RECURRENCE_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["kind", "interval_amount", "interval_unit", "weekdays", "day_of_month", "count", "until"],
    "properties": {
        "kind": {"enum": ["NONE", "FIXED_INTERVAL", "DAILY", "WEEKLY", "MONTHLY"]},
        "interval_amount": _nullable({"type": "integer", "minimum": 1}),
        "interval_unit": _nullable({"enum": ["MINUTE", "HOUR", "DAY", "WEEK"]}),
        "weekdays": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 7}, "maxItems": 7},
        "day_of_month": _nullable({"type": "integer", "minimum": 1, "maximum": 31}),
        "count": _nullable({"type": "integer", "minimum": 1, "maximum": 100000}),
        "until": TEMPORAL_JSON_SCHEMA,
    },
}
OUTPUT_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "request_id", "outcome", "operation", "target_job_id",
                 "target_job_revision", "target_occurrence_id", "time_expression", "recurrence",
                 "notification", "clarification_reason", "clarification_question"],
    "properties": {
        "schema_version": {"const": "schedule-proposal/1"},
        "request_id": {"type": "string", "format": "uuid"},
        "outcome": {"enum": ["PROPOSAL", "NEEDS_CLARIFICATION", "UNSUPPORTED"]},
        "operation": {"enum": ["CREATE_JOB", "CANCEL_JOB", "SNOOZE_OCCURRENCE", "NONE"]},
        "target_job_id": _nullable({"type": "string"}),
        "target_job_revision": _nullable({"type": "integer", "minimum": 1}),
        "target_occurrence_id": _nullable({"type": "string"}),
        "time_expression": TEMPORAL_JSON_SCHEMA,
        "recurrence": RECURRENCE_JSON_SCHEMA,
        "notification": {
            "type": "object", "additionalProperties": False,
            "required": ["title", "body", "speakable_text", "locale", "visibility"],
            "properties": {
                "title": _nullable({"type": "string"}), "body": _nullable({"type": "string"}),
                "speakable_text": _nullable({"type": "string"}),
                "locale": {"enum": ["el-GR", "en-US"]},
                "visibility": {"enum": ["private", "title_only", "full"]},
            },
        },
        "clarification_reason": {"enum": ["NONE", "MISSING_TIME", "AMBIGUOUS_DATE", "AMBIGUOUS_TIME",
            "AMBIGUOUS_TIMEZONE", "AMBIGUOUS_RECURRENCE", "AMBIGUOUS_TARGET", "CONFLICTING_FIELDS",
            "MISSING_NOTIFICATION_TEXT", "UNSUPPORTED_TEMPORAL_EXPRESSION", "UNSUPPORTED_RECURRENCE",
            "TARGET_BINDING_MISMATCH", "PROPOSAL_INVALID"]},
        "clarification_question": _nullable({"type": "string"}),
    },
}


@dataclass(frozen=True, slots=True)
class CompilerQualification:
    sha256: str
    artifact: Mapping[str, Any]


def load_compiler_qualification(path: Path, expected_sha256: str) -> CompilerQualification:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ScheduleContractError("compiler_qualification_missing", status=503)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ScheduleContractError("compiler_qualification_mismatch", status=503)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or set(value) != {"qualification_inputs", "qualification_sha256"}:
        raise ScheduleContractError("compiler_qualification_invalid", status=503)
    inputs = dict(value["qualification_inputs"])
    inputs["prompt_template_sha256"] = PROMPT_TEMPLATE_SHA256
    inputs["input_schema_sha256"] = hashlib.sha256(jcs_bytes(INPUT_JSON_SCHEMA)).hexdigest()
    inputs["output_schema_sha256"] = hashlib.sha256(jcs_bytes(OUTPUT_JSON_SCHEMA)).hexdigest()
    calculated = compiler_qualification_digest(inputs)
    if value["qualification_sha256"] != calculated:
        raise ScheduleContractError("compiler_qualification_mismatch", status=503)
    return CompilerQualification(calculated, value)


class LocalScheduleCompiler:
    def __init__(self, *, base_url: str, bearer_token: str, model_alias: str,
                 qualification: CompilerQualification, timeout_seconds: float,
                 zones: PackagedTimezoneProvider) -> None:
        parsed = urlsplit(base_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ScheduleContractError("compiler_not_loopback", status=503) from exc
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path.rstrip("/") != "/v1"
            or parsed.query
            or parsed.fragment
        ):
            raise ScheduleContractError("compiler_not_loopback", status=503)
        if model_alias != "jarvis-local":
            raise ScheduleContractError("compiler_model_alias_mismatch", status=503)
        if not isinstance(bearer_token, str) or not 1 <= len(bearer_token.encode("utf-8")) <= 512:
            raise ScheduleContractError("compiler_credentials_invalid", status=503)
        if not 0 < timeout_seconds <= 60:
            raise ScheduleContractError("compiler_timeout_invalid", status=503)
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token
        self.model_alias = model_alias
        self.qualification = qualification
        self.timeout_seconds = timeout_seconds
        self.zones = zones

    def compile(self, input_raw: Mapping[str, Any]) -> CompiledProposal:
        import httpx

        parsed = ProposalInput.parse(input_raw)
        request = {**dict(input_raw), "request_text": parsed.request_text}
        body = {
            "model": self.model_alias,
            "messages": [
                {"role": "system", "content": PROMPT_TEMPLATE},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False, separators=(",", ":"))},
            ],
            "temperature": 0,
            "top_p": 1,
            "stream": False,
            "tools": [],
            "max_completion_tokens": 512,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "schedule_proposal_v1", "strict": True, "schema": OUTPUT_JSON_SCHEMA,
            }},
        }
        with httpx.Client(timeout=self.timeout_seconds, trust_env=False, follow_redirects=False) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.bearer_token}", "Content-Type": "application/json"},
                json=body,
            )
        response.raise_for_status()
        try:
            output = json.loads(response.json()["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ScheduleContractError("PROPOSAL_INVALID") from exc
        return compile_proposal(
            request, output,
            compiler_qualification_sha256=self.qualification.sha256,
            zones=self.zones,
        )
