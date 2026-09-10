"""Single-shot, local-only model adapter for ``schedule-proposal/1``."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .domain import OrchestratorError
from .inference import InferenceControlClient, parse_inference_json
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


def compiler_artifact_sha256() -> str:
    # Pin the installed package, including deterministic compiler dependencies.
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py"), key=lambda item: item.name):
        if not path.stem.isidentifier():
            continue
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


COMPILER_ARTIFACT_SHA256 = compiler_artifact_sha256()

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
    return {"anyOf": [{"type": "string", **dict(schema)}, {"type": "null"}]}


def _temporal_schema() -> dict[str, Any]:
    properties = {
        "kind": {"type": "string", "enum": ["NONE", "ABSOLUTE_LOCAL", "CALENDAR_DATE", "AFTER_DURATION", "RELATIVE_DAY", "WEEKDAY"]},
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
        "kind": {"type": "string", "enum": ["NONE", "FIXED_INTERVAL", "DAILY", "WEEKLY", "MONTHLY"]},
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
        "schema_version": {"type": "string", "const": "schedule-proposal/1"},
        "request_id": {"type": "string"},
        "outcome": {"type": "string", "enum": ["PROPOSAL", "NEEDS_CLARIFICATION", "UNSUPPORTED"]},
        "operation": {"type": "string", "enum": ["CREATE_JOB", "CANCEL_JOB", "SNOOZE_OCCURRENCE", "NONE"]},
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
                "locale": {"type": "string", "enum": ["el-GR", "en-US"]},
                "visibility": {"type": "string", "enum": ["private", "title_only", "full"]},
            },
        },
        "clarification_reason": {"type": "string", "enum": ["NONE", "MISSING_TIME", "AMBIGUOUS_DATE", "AMBIGUOUS_TIME",
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
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ScheduleContractError("compiler_qualification_missing", status=503)
    try:
        with open(path, "rb", opener=lambda name, flags: os.open(
            name, flags | getattr(os, "O_NONBLOCK", 0)
        )) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("compiler qualification must be a regular file")
            raw = stream.read(1024 * 1024 + 1)
        if not raw or len(raw) > 1024 * 1024:
            raise ValueError("compiler qualification size invalid")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ScheduleContractError("compiler_qualification_mismatch", status=503)
        value = parse_inference_json(raw, maximum_bytes=1024 * 1024)
    except (OSError, ValueError) as exc:
        raise ScheduleContractError("compiler_qualification_invalid", status=503) from exc
    return _validate_qualification(value)


def _validate_qualification(value: Mapping[str, Any]) -> CompilerQualification:
    if set(value) != {"qualification_inputs", "qualification_sha256"}:
        raise ScheduleContractError("compiler_qualification_invalid", status=503)
    inputs = value["qualification_inputs"]
    expected_inputs = {
        "compiler_artifact_sha256": COMPILER_ARTIFACT_SHA256,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "input_schema_sha256": hashlib.sha256(jcs_bytes(INPUT_JSON_SCHEMA)).hexdigest(),
        "output_schema_sha256": hashlib.sha256(jcs_bytes(OUTPUT_JSON_SCHEMA)).hexdigest(),
    }
    if not isinstance(inputs, dict) or any(inputs.get(name) != digest for name, digest in expected_inputs.items()):
        raise ScheduleContractError("compiler_qualification_mismatch", status=503)
    calculated = compiler_qualification_digest(inputs)
    if value["qualification_sha256"] != calculated:
        raise ScheduleContractError("compiler_qualification_mismatch", status=503)
    return CompilerQualification(calculated, value)


class LocalScheduleCompiler:
    def __init__(self, *, base_url: str, bearer_token: str, model_alias: str,
                 qualification: CompilerQualification, timeout_seconds: float,
                 zones: PackagedTimezoneProvider) -> None:
        if not isinstance(base_url, str):
            raise ScheduleContractError("compiler_not_loopback", status=503)
        try:
            parsed = urlsplit(base_url)
            port = parsed.port
        except ValueError as exc:
            raise ScheduleContractError("compiler_not_loopback", status=503) from exc
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or port is None or port == 0
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path.rstrip("/") != "/v1"
            or parsed.query
            or parsed.fragment
        ):
            raise ScheduleContractError("compiler_not_loopback", status=503)
        if model_alias != "jarvis-local":
            raise ScheduleContractError("compiler_model_alias_mismatch", status=503)
        if (not isinstance(bearer_token, str) or not bearer_token.isascii()
                or not 1 <= len(bearer_token) <= 512
                or any(ord(character) < 32 or ord(character) == 127 for character in bearer_token)):
            raise ScheduleContractError("compiler_credentials_invalid", status=503)
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not 0 < timeout_seconds <= 60):
            raise ScheduleContractError("compiler_timeout_invalid", status=503)
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token
        self.model_alias = model_alias
        self.qualification = qualification
        self.timeout_seconds = timeout_seconds
        self.zones = zones
        validated = _validate_qualification(qualification.artifact)
        if validated.sha256 != qualification.sha256:
            raise ScheduleContractError("compiler_qualification_mismatch", status=503)
        inputs = validated.artifact["qualification_inputs"]
        self.runtime_identity = {
            "schema_version": "jarvis-inference-compiler-identity/1",
            **{name: inputs[name] for name in (
                "model_artifact_sha256", "model_profile_sha256", "runtime_sha256", "tokenizer_sha256"
            )},
        }
        self.runtime_identity_sha256 = hashlib.sha256(jcs_bytes(self.runtime_identity)).hexdigest()

    def verify_runtime(self) -> None:
        try:
            actual = InferenceControlClient(
                self.base_url, self.bearer_token, timeout_seconds=self.timeout_seconds,
            ).compiler_identity()
        except (OrchestratorError, RuntimeError) as exc:
            raise ScheduleContractError("compiler_identity_unavailable", status=503) from exc
        if actual != self.runtime_identity:
            raise ScheduleContractError("compiler_identity_mismatch", status=503)

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
            "max_completion_tokens": 512,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "schedule_proposal_v1", "strict": True, "schema": OUTPUT_JSON_SCHEMA,
            }},
        }
        async def request_bytes() -> bytes:
            # One deadline covers headers and the complete body, including a slowly trickling peer.
            async with asyncio.timeout(self.timeout_seconds):
                async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False, follow_redirects=False) as client:
                    async with client.stream(
                        "POST", f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.bearer_token}",
                                 "X-Jarvis-Compiler-Identity": self.runtime_identity_sha256,
                                 "Content-Type": "application/json", "Accept-Encoding": "identity"},
                        json=body,
                    ) as response:
                        response.raise_for_status()
                        if response.headers.get("X-Jarvis-Compiler-Identity") != self.runtime_identity_sha256:
                            raise ScheduleContractError("compiler_identity_mismatch", status=503)
                        if (response.headers.get("content-encoding", "identity").lower() != "identity"
                                or response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json"):
                            raise ScheduleContractError("PROPOSAL_INVALID")
                        raw = bytearray()
                        async for chunk in response.aiter_bytes(chunk_size=16384):
                            raw.extend(chunk)
                            if len(raw) > 65536:
                                raise ScheduleContractError("PROPOSAL_INVALID")
                        return bytes(raw)

        try:
            raw = asyncio.run(request_bytes())
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ScheduleContractError("compiler_timeout", status=503) from exc
        except httpx.HTTPError as exc:
            raise ScheduleContractError("compiler_unavailable", status=503) from exc
        try:
            envelope = parse_inference_json(raw, maximum_bytes=65536)
            choices = envelope["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError
            choice = choices[0]
            if (type(choice["index"]) is not int or choice["index"] != 0
                    or choice["finish_reason"] != "stop"):
                raise ValueError
            message = choice["message"]
            if (message["role"] != "assistant" or not isinstance(message["content"], str)
                    or message.get("tool_calls") not in (None, []) or message.get("refusal") is not None):
                raise ValueError
            output = parse_inference_json(message["content"].encode("utf-8"), maximum_bytes=16384)
        except (KeyError, IndexError, TypeError, ValueError, RecursionError) as exc:
            raise ScheduleContractError("PROPOSAL_INVALID") from exc
        return compile_proposal(
            request, output,
            compiler_qualification_sha256=self.qualification.sha256,
            zones=self.zones,
        )
