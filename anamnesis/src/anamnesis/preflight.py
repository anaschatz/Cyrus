"""Pinned hosted-model compatibility attestation for measured experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ChatMessageUser, ModelCost, ModelUsage, ResponseSchema

from anamnesis.experiment import ArtifactPin, ModelProviderArgs
from anamnesis.inspect_adapter import (
    ModelPreflightResult,
    _decision_schema,
    _memory_delta_schema,
)
from anamnesis.prompts import build_decision_prompt, build_memory_compiler_prompt
from anamnesis.schema import ObservableEvent
from anamnesis.wire import DecisionWire, MemoryDeltaWire

MODEL_PREFLIGHT_TASK_VERSION = "0.3"
MODEL_PREFLIGHT_PURPOSE = "model-compatibility-preflight"
MODEL_PREFLIGHT_SAMPLE_ID = "model-preflight-v0"
MODEL_PREFLIGHT_EVENT_ID = "preflight-event"
MODEL_PREFLIGHT_ACTIVE_STATE = '{"facts":[],"intents":[]}'
OPENAI_CHAT_COMPLETIONS_MODEL_ARGS = ModelProviderArgs(responses_api=False).model_dump(
    mode="python"
)
OPENAI_CHAT_REQUEST_KEYS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "extra_headers",
        "seed",
        "temperature",
        "response_format",
    }
)
INSPECT_REQUEST_ID_HEADER = "x-irid"


def _verified_artifact_bytes(name: str, artifact: ArtifactPin) -> bytes:
    path = Path(artifact.path)
    if path.suffix != ".eval" and name == "model.preflight":
        raise ValueError("model preflight artifact must be an Inspect .eval log")
    if not path.is_file():
        raise ValueError(f"manifest artifact {name} does not exist: {path}")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if artifact.sha256 != digest:
        raise ValueError(f"manifest artifact hash mismatch: {name}")
    return content


def _configured_model_cost(artifact: ArtifactPin, model_name: str) -> ModelCost:
    path = Path(artifact.path)
    if not path.is_file():
        raise ValueError(f"manifest artifact model.pricing does not exist: {path}")
    content = path.read_bytes()
    if artifact.sha256 != hashlib.sha256(content).hexdigest():
        raise ValueError("manifest artifact hash mismatch: model.pricing")
    raw_text = content.decode("utf-8")
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        import yaml

        raw = yaml.safe_load(raw_text)
    if not isinstance(raw, dict) or model_name not in raw:
        raise ValueError(f"pricing config has no entry for {model_name}")
    return ModelCost.model_validate(raw[model_name])


def _expected_preflight_event() -> ObservableEvent:
    return ObservableEvent.model_validate(
        {
            "id": MODEL_PREFLIGHT_EVENT_ID,
            "at": "2026-01-05T09:00:00+00:00",
            "kind": "user_message",
            "text": "At 17:00 today remind me to run the compatibility check.",
        }
    )


def _expected_prompts() -> tuple[str, str]:
    event = _expected_preflight_event()
    return (
        build_memory_compiler_prompt(
            event=event,
            active_state=MODEL_PREFLIGHT_ACTIVE_STATE,
        ),
        build_decision_prompt(
            now=event.at.isoformat(),
            current_event_id=event.id,
            context_events=[event],
            decision_history=[],
            memory_view=None,
        ),
    )


def _canonical_response_schema(schema: ResponseSchema) -> str:
    return json.dumps(
        schema.model_dump(mode="json", exclude_none=False),
        sort_keys=True,
        separators=(",", ":"),
    )


def _openai_service_snapshot(model_name: str) -> str:
    prefix = "openai/"
    if not model_name.startswith(prefix):
        raise ValueError("v0 requires the direct OpenAI provider")
    service_model = model_name.removeprefix(prefix)
    if re.search(r"-\d{4}-\d{2}-\d{2}$", service_model) is None:
        raise ValueError("OpenAI model must be an immutable dated snapshot")
    return service_model


def _validate_openai_chat_completion_event(
    model_event: ModelEvent,
    *,
    model_name: str,
    temperature: float,
    seed: int,
) -> None:
    """Verify the post-adapter request, not only its declared GenerateConfig."""

    service_model = _openai_service_snapshot(model_name)
    if model_event.model != model_name:
        raise ValueError("model event differs from the pinned OpenAI snapshot")
    if (
        model_event.cache is not None
        or model_event.retries not in (None, 0)
        or model_event.error is not None
        or model_event.output.error is not None
    ):
        raise ValueError("model event was cached, retried, or failed")
    call = model_event.call
    if call is None:
        raise ValueError("model event is missing its raw API call")
    if call.error:
        raise ValueError("raw OpenAI model call contains an error")

    request = call.request
    if set(request) != OPENAI_CHAT_REQUEST_KEYS:
        raise ValueError("raw request contains unpinned generation or transport fields")
    if len(model_event.input) != 1 or not isinstance(
        model_event.input[0], ChatMessageUser
    ):
        raise ValueError("model event must contain one exact user prompt")
    prompt = model_event.input[0].content
    if not isinstance(prompt, str):
        raise ValueError("model event user prompt must be text")
    messages = request.get("messages")
    if messages != [{"role": "user", "content": prompt}]:
        raise ValueError("raw request messages differ from the model event input")
    if request.get("tools") is not None or request.get("tool_choice") is not None:
        raise ValueError("raw request contains tools")
    if request.get("model") != service_model:
        raise ValueError("raw request model differs from the dated snapshot")
    if request.get("temperature") != temperature:
        raise ValueError("raw request dropped or changed temperature")
    if request.get("seed") != seed:
        raise ValueError("raw request dropped or changed seed")
    extra_headers = request.get("extra_headers")
    if (
        not isinstance(extra_headers, dict)
        or set(extra_headers) != {INSPECT_REQUEST_ID_HEADER}
        or not isinstance(extra_headers[INSPECT_REQUEST_ID_HEADER], str)
        or not extra_headers[INSPECT_REQUEST_ID_HEADER]
    ):
        raise ValueError("raw request contains unpinned headers")

    response_schema = model_event.config.response_schema
    if response_schema is None or response_schema.strict is not True:
        raise ValueError("model event is missing its strict response schema")
    allowed_schemas = {
        _canonical_response_schema(_decision_schema(model_name)),
        _canonical_response_schema(_memory_delta_schema(model_name)),
    }
    if _canonical_response_schema(response_schema) not in allowed_schemas:
        raise ValueError("model event uses an unpinned response schema")
    effective_event_config = model_event.config.model_dump(
        mode="json", exclude_none=True
    )
    effective_event_config.pop("response_schema", None)
    expected_event_config = {
        "max_retries": 0,
        "max_connections": 1,
        "adaptive_connections": False,
        "temperature": temperature,
        "seed": seed,
        "cache": False,
    }
    if effective_event_config != expected_event_config:
        raise ValueError("model event contains unpinned generation configuration")
    response_format = request.get("response_format")
    if not isinstance(response_format, dict) or response_format.get("type") != (
        "json_schema"
    ):
        raise ValueError("raw request is missing JSON Schema response format")
    raw_schema = response_format.get("json_schema")
    if not isinstance(raw_schema, dict):
        raise ValueError("raw request is missing its strict JSON Schema")
    if (
        set(response_format) != {"type", "json_schema"}
        or set(raw_schema) != {"name", "schema", "description", "strict"}
        or raw_schema.get("description") is not None
    ):
        raise ValueError("raw request contains unpinned response-format fields")
    expected_body = response_schema.json_schema.model_dump(
        mode="json", exclude_none=True
    )
    if (
        raw_schema.get("name") != response_schema.name
        or raw_schema.get("strict") is not True
        or raw_schema.get("schema") != expected_body
    ):
        raise ValueError("raw request strict response schema differs from the event")

    response = call.response
    if not isinstance(response, dict) or response.get("model") != service_model:
        raise ValueError("raw response model differs from the dated snapshot")


def _validate_openai_chat_completion_log(
    log: EvalLog,
    *,
    model_name: str,
    temperature: float,
    seed: int,
) -> None:
    """Apply the raw-request contract to every model event in an Inspect log."""

    if log.samples is None:
        raise ValueError("Inspect log does not contain sample records")
    model_events = [
        event
        for sample in log.samples
        for event in sample.events
        if isinstance(event, ModelEvent)
    ]
    if not model_events:
        raise ValueError("Inspect log contains no auditable model events")
    for model_event in model_events:
        _validate_openai_chat_completion_event(
            model_event,
            model_name=model_name,
            temperature=temperature,
            seed=seed,
        )


def _require_exact_user_prompt(model_event: ModelEvent, expected: str) -> None:
    if len(model_event.input) != 1 or not isinstance(
        model_event.input[0], ChatMessageUser
    ):
        raise ValueError("preflight model call must contain one exact user prompt")
    if model_event.input[0].content != expected:
        raise ValueError("preflight model prompt differs from the current contract")


def _computed_cost(cost: ModelCost, usage: ModelUsage) -> float:
    total = usage.input_tokens * cost.input / 1_000_000
    total += usage.output_tokens * cost.output / 1_000_000
    total += (usage.input_tokens_cache_write or 0) * cost.input_cache_write / 1_000_000
    total += (usage.input_tokens_cache_read or 0) * cost.input_cache_read / 1_000_000
    return total


def _validate_model_event_usage(
    model_event: ModelEvent,
    *,
    pricing: ModelCost,
) -> None:
    usage = model_event.output.usage
    if usage is None:
        raise ValueError("preflight model call is missing token usage")
    logical_input = (
        usage.input_tokens
        + (usage.input_tokens_cache_read or 0)
        + (usage.input_tokens_cache_write or 0)
    )
    if logical_input <= 0 or usage.output_tokens <= 0:
        raise ValueError("preflight model call must report positive token usage")
    if usage.total_cost is None:
        raise ValueError("preflight model call is missing provider cost")
    expected_cost = _computed_cost(pricing, usage)
    if not math.isclose(
        usage.total_cost,
        expected_cost,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("preflight model cost differs from pinned pricing")


def _validate_preflight_execution_policy(log: EvalLog, model_name: str) -> None:
    # Import lazily: cli owns the reusable projection/policy helper and calls this
    # module from its report validator, so an eager import would create a cycle.
    from anamnesis.cli import (
        InspectLogPolicy,
        _inspect_log_metadata,
        _validate_inspect_log_policy,
    )

    metadata = _inspect_log_metadata(log)
    revision = metadata.revision_commit
    _validate_inspect_log_policy(
        metadata,
        InspectLogPolicy(
            model=model_name,
            temperature=0.0,
            seed=101,
            response_cache=False,
            max_connections=1,
            max_samples=1,
            max_tasks=1,
            model_base_url=None,
            model_args=OPENAI_CHAT_COMPLETIONS_MODEL_ARGS,
            max_retries=0,
            log_model_api=True,
            # A preflight necessarily predates the final manifest commit. Reuse
            # the helper to require a syntactically valid present/clean revision,
            # while prompts and schemas below bind compatibility to current code.
            git_commit=revision,
        ),
    )


def validate_model_preflight_log(
    log: EvalLog,
    *,
    model_name: str,
    pricing: ModelCost,
) -> None:
    """Semantically validate one already-loaded compatibility log."""

    _validate_preflight_execution_policy(log, model_name)
    _validate_openai_chat_completion_log(
        log,
        model_name=model_name,
        temperature=0.0,
        seed=101,
    )
    task_name = (log.eval.task_registry_name or log.eval.task).rsplit("@", 1)[-1]
    if task_name != "model_preflight":
        raise ValueError("model preflight artifact contains the wrong Inspect task")
    if str(log.eval.task_version) != MODEL_PREFLIGHT_TASK_VERSION:
        raise ValueError("model preflight task version differs from the current gate")
    metadata = log.eval.metadata or {}
    if metadata.get("purpose") != MODEL_PREFLIGHT_PURPOSE:
        raise ValueError("model preflight purpose metadata is missing or invalid")
    if log.samples is None or len(log.samples) != 1:
        raise ValueError("model preflight must contain exactly one sample")

    sample = log.samples[0]
    if str(sample.id) != MODEL_PREFLIGHT_SAMPLE_ID or sample.epoch != 1:
        raise ValueError("model preflight sample identity is invalid")
    if (
        sample.error is not None
        or sample.invalidation is not None
        or sample.error_retries
    ):
        raise ValueError("model preflight sample contains an error, retry, or edit")

    model_events = [event for event in sample.events if isinstance(event, ModelEvent)]
    if len(model_events) != 2:
        raise ValueError("model preflight must contain exactly two model calls")
    expected_prompts = _expected_prompts()
    expected_schemas = (
        _memory_delta_schema(model_name),
        _decision_schema(model_name),
    )
    expected_names = ("anamnesis_memory_delta", "anamnesis_decision")
    validators = (MemoryDeltaWire, DecisionWire)
    calls = zip(
        model_events,
        expected_prompts,
        expected_schemas,
        expected_names,
        strict=True,
    )
    for index, (
        model_event,
        expected_prompt,
        expected_schema,
        expected_name,
    ) in enumerate(calls):
        if model_event.model != model_name:
            raise ValueError("preflight model call differs from the pinned snapshot")
        if (
            model_event.cache is not None
            or model_event.error is not None
            or model_event.output.error is not None
            or model_event.retries not in (None, 0)
        ):
            raise ValueError("preflight model call was cached, retried, or failed")
        schema = model_event.config.response_schema
        if schema is None or schema.name != expected_name or schema.strict is not True:
            raise ValueError(
                "preflight model call did not use the required strict schema"
            )
        if _canonical_response_schema(schema) != _canonical_response_schema(
            expected_schema
        ):
            raise ValueError("preflight response schema differs from the current wire")
        _require_exact_user_prompt(model_event, expected_prompt)
        try:
            validators[index].model_validate_json(model_event.output.completion)
        except ValueError as error:
            raise ValueError(
                "preflight model output does not match its wire schema"
            ) from error
        _validate_model_event_usage(model_event, pricing=pricing)

    if sample.output is None:
        raise ValueError("model preflight sample has no final result")
    result = ModelPreflightResult.model_validate_json(sample.output.completion)
    if result.model != model_name or not all(
        (
            result.strict_schema_supported,
            not result.compiler_parse_error,
            not result.decision_parse_error,
            result.compiler_usage_complete,
            result.decision_usage_complete,
            result.compiler_cost_complete,
            result.decision_cost_complete,
            result.passed,
        )
    ):
        raise ValueError("model preflight final result did not pass every gate")


def validate_model_preflight_artifact(
    preflight: ArtifactPin,
    *,
    model_name: str,
    pricing: ArtifactPin,
) -> None:
    """Verify the pinned bytes and semantics of a compatibility `.eval`."""

    _verified_artifact_bytes("model.preflight", preflight)
    model_cost = _configured_model_cost(pricing, model_name)
    validate_model_preflight_log(
        read_eval_log(Path(preflight.path)),
        model_name=model_name,
        pricing=model_cost,
    )


__all__ = [
    "MODEL_PREFLIGHT_PURPOSE",
    "MODEL_PREFLIGHT_TASK_VERSION",
    "validate_model_preflight_artifact",
    "validate_model_preflight_log",
]
