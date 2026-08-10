"""Validation of the frozen local Ollama semantic-preflight artifact."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ChatMessageUser, ResponseSchema
from inspect_ai.model._openai import openai_completion_params
from pydantic import ValidationError

from anamnesis.experiment import ArtifactPin
from anamnesis.local_experiment import (
    LOCAL_W3_M2_PROTOCOL_SHA256,
    LOCAL_W3_M2_T1_PROTOCOL_SHA256,
    LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256,
    LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
    LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
)
from anamnesis.local_runtime import (
    LOCAL_MODEL_PREFLIGHT_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_TASK_VERSION,
    LOCAL_MODEL_PREFLIGHT_W2_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W2_SAMPLE_ID,
    LOCAL_MODEL_PREFLIGHT_W2_TASK_VERSION,
    LOCAL_MODEL_PREFLIGHT_W3_M2_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W3_M2_SAMPLE_ID,
    LOCAL_MODEL_PREFLIGHT_W3_M2_T1_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W3_M2_T1_SAMPLE_ID,
    LOCAL_MODEL_PREFLIGHT_W3_M2_T1_TASK_VERSION,
    LOCAL_MODEL_PREFLIGHT_W3_M2_TASK_VERSION,
    LOCAL_MODEL_PREFLIGHT_W3_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W3_SAMPLE_ID,
    LOCAL_MODEL_PREFLIGHT_W3_TASK_VERSION,
    LOCAL_OLLAMA_BASE_URL,
    LOCAL_OLLAMA_MODEL,
    LOCAL_OLLAMA_SERVICE_MODEL,
    LOCAL_W3_M2_OLLAMA_MODEL,
    LOCAL_W3_M2_OLLAMA_SERVICE_MODEL,
    LocalDecisionWire,
    LocalModelPreflightResult,
    LocalModelPreflightW2Result,
    LocalModelPreflightW3Result,
    _compiler_preflight_semantics,
    _local_decision_schema,
    _local_memory_delta_schema,
    _local_usage_from_output,
    _w2_compiler_semantic_valid,
    _w2_decision_semantic_valid,
    _w3_compiler_semantic_valid,
    build_local_decision_prompt,
    load_local_w2_preflight_fixture,
    load_local_w3_preflight_fixture,
    load_local_w3_preflight_protocol,
    local_w2_preflight_prompts,
    local_w3_preflight_prompts,
)
from anamnesis.local_wire import (
    LocalMemoryDeltaWire,
    build_local_memory_compiler_prompt,
)
from anamnesis.memory import CompilerCall
from anamnesis.schema import ObservableEvent, Usage

LOCAL_PREFLIGHT_SAMPLE_ID = "local-model-preflight-v0"
LOCAL_PREFLIGHT_EVENT_ID = "local-preflight-event"
LOCAL_PREFLIGHT_ACTIVE_STATE = '{"facts":[],"intents":[]}'
INSPECT_REQUEST_ID_HEADER = "x-irid"


def local_preflight_event() -> ObservableEvent:
    return ObservableEvent(
        id=LOCAL_PREFLIGHT_EVENT_ID,
        at="2026-01-05T09:00:00+00:00",
        kind="user_message",
        text="At 17:00 today remind me to perform compatibility check.",
    )


def local_preflight_prompts() -> tuple[str, str]:
    event = local_preflight_event()
    return (
        build_local_memory_compiler_prompt(
            event=event,
            active_state=LOCAL_PREFLIGHT_ACTIVE_STATE,
        ),
        build_local_decision_prompt(
            now=event.at.isoformat(),
            current_event_id=event.id,
            context_events=[event],
            decision_history=[],
            memory_view=None,
        ),
    )


def _canonical_schema(schema: ResponseSchema) -> str:
    return schema.model_dump_json(exclude_none=False)


def _validate_local_model_event(
    event: ModelEvent,
    *,
    prompt: str,
    schema: ResponseSchema,
    seed: int,
    expected_model: str = LOCAL_OLLAMA_MODEL,
    expected_service_model: str = LOCAL_OLLAMA_SERVICE_MODEL,
    expected_extra_body: Mapping[str, Any] | None = None,
) -> None:
    if event.model != expected_model:
        raise ValueError("preflight model event differs from the pinned local model")
    if (
        event.cache is not None
        or event.retries not in (None, 0)
        or event.error is not None
        or event.output.error is not None
    ):
        raise ValueError("local model event was cached, retried, or failed")
    if event.output.model not in {expected_model, expected_service_model}:
        raise ValueError("local response model differs from the pinned model")
    if event.output.usage is None or event.output.usage.input_tokens <= 0:
        raise ValueError("local model event is missing positive input-token usage")
    if event.output.usage.total_cost not in (None, 0, 0.0):
        raise ValueError("local model event reports a non-zero provider cost")
    if len(event.input) != 1 or not isinstance(event.input[0], ChatMessageUser):
        raise ValueError("local model event must contain one exact user prompt")
    if event.input[0].content != prompt:
        raise ValueError("local model event prompt differs from the frozen prompt")
    if event.config.response_schema is None or _canonical_schema(
        event.config.response_schema
    ) != _canonical_schema(schema):
        raise ValueError("local model event uses an unpinned response schema")
    effective_config = event.config.model_dump(mode="json", exclude_none=True)
    effective_config.pop("response_schema", None)
    expected_config: dict[str, Any] = {
        "max_retries": 0,
        "max_connections": 1,
        "adaptive_connections": False,
        "temperature": 0.0,
        "seed": seed,
        "cache": False,
    }
    if expected_extra_body is not None:
        expected_config["extra_body"] = dict(expected_extra_body)
    if effective_config != expected_config:
        raise ValueError("local model event uses unpinned generation settings")

    call = event.call
    if call is None or call.error:
        raise ValueError("local model event is missing a successful raw API call")
    expected_params = openai_completion_params(
        expected_service_model,
        event.config,
        tools=False,
    )
    request = call.request
    expected_keys = set(expected_params) | {
        "messages",
        "tools",
        "tool_choice",
        "extra_headers",
    }
    if set(request) != expected_keys:
        raise ValueError("raw local request contains unpinned fields")
    if any(request.get(key) != value for key, value in expected_params.items()):
        raise ValueError("raw local request differs from effective model settings")
    if request.get("messages") != [{"role": "user", "content": prompt}]:
        raise ValueError("raw local request messages differ from the prompt")
    if request.get("tools") is not None or request.get("tool_choice") is not None:
        raise ValueError("raw local request contains tools")
    headers = request.get("extra_headers")
    if (
        not isinstance(headers, dict)
        or set(headers) != {INSPECT_REQUEST_ID_HEADER}
        or not isinstance(headers[INSPECT_REQUEST_ID_HEADER], str)
        or not headers[INSPECT_REQUEST_ID_HEADER]
    ):
        raise ValueError("raw local request contains unpinned headers")
    response = call.response
    if not isinstance(response, dict) or response.get("model") not in {
        expected_model,
        expected_service_model,
    }:
        raise ValueError("raw local response model differs from the pin")


def validate_local_preflight_log(
    log: EvalLog,
    *,
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
) -> None:
    """Validate an exact two-call semantic local compatibility attestation."""

    if log.status != "success" or log.invalidated:
        raise ValueError("local preflight log is not a valid successful evaluation")
    spec = log.eval
    if (
        spec.task_registry_name != "local_model_preflight"
        or spec.task_version != LOCAL_MODEL_PREFLIGHT_TASK_VERSION
        or (spec.metadata or {}).get("purpose") != LOCAL_MODEL_PREFLIGHT_PURPOSE
    ):
        raise ValueError("local preflight task identity differs from the protocol")
    if (spec.metadata or {}).get("pricing_config_sha256") != (expected_pricing_sha256):
        raise ValueError("local preflight pricing hash differs from the pin")
    if spec.model != LOCAL_OLLAMA_MODEL:
        raise ValueError("local preflight used a different model")
    if str(spec.model_base_url).rstrip("/") != LOCAL_OLLAMA_BASE_URL:
        raise ValueError("local preflight used a different provider route")
    if spec.model_args != {}:
        raise ValueError("local preflight used unpinned provider arguments")
    effective_model_config = spec.model_generate_config.merge(
        log.plan.config
    ).model_dump(
        mode="json",
        exclude_none=True,
    )
    if effective_model_config != {
        "max_retries": 0,
        "max_connections": 1,
        "adaptive_connections": False,
        "temperature": 0.0,
        "seed": seed,
        "cache": False,
    }:
        raise ValueError("local preflight model configuration differs from the pin")
    revision = spec.revision
    revision_commit = revision.commit if revision is not None else None
    if (
        revision is None
        or revision.dirty is not False
        or not isinstance(revision_commit, str)
        or re.fullmatch(r"[0-9a-f]{7,40}", revision_commit) is None
        or not expected_git_commit.startswith(revision_commit)
    ):
        raise ValueError("local preflight was not run from the frozen clean commit")
    if spec.config.log_model_api is not True:
        raise ValueError("local preflight did not retain raw model API calls")
    if (
        spec.config.max_samples != 1
        or spec.config.max_tasks != 1
        or spec.config.epochs != 1
    ):
        raise ValueError("local preflight has an invalid sample/task policy")
    if log.samples is None or len(log.samples) != 1:
        raise ValueError("local preflight requires exactly one synthetic sample")
    sample = log.samples[0]
    if (
        sample.id != LOCAL_PREFLIGHT_SAMPLE_ID
        or sample.epoch != 1
        or sample.error is not None
        or sample.invalidation is not None
        or bool(sample.error_retries)
        or sample.output is None
    ):
        raise ValueError("local preflight sample failed or has the wrong identity")
    if sample.output.model != LOCAL_OLLAMA_MODEL:
        raise ValueError("local preflight final output has the wrong model identity")
    result = LocalModelPreflightResult.model_validate_json(sample.output.completion)
    if not result.passed or result.loaded_model is None:
        raise ValueError("local semantic preflight result did not pass")
    if not all(
        (
            result.same_model_for_compiler_and_decision,
            not result.compiler_parse_error,
            not result.decision_parse_error,
            result.compiler_semantic_valid,
            result.decision_semantic_valid,
            result.compiler_usage_complete,
            result.decision_usage_complete,
            result.compiler_cost_complete,
            result.decision_cost_complete,
        )
    ):
        raise ValueError("local semantic preflight flags are internally inconsistent")
    if result.compiler_usage.cost_usd != 0.0 or result.decision_usage.cost_usd != 0.0:
        raise ValueError("local semantic preflight did not record zero API cost")

    events = [event for event in sample.events if isinstance(event, ModelEvent)]
    if len(events) != 2:
        raise ValueError("local preflight must contain exactly two model calls")
    compiler_prompt, decision_prompt = local_preflight_prompts()
    _validate_local_model_event(
        events[0],
        prompt=compiler_prompt,
        schema=_local_memory_delta_schema(LOCAL_OLLAMA_MODEL),
        seed=seed,
    )
    if result.compiler_usage != _local_usage_from_output(events[0].output):
        raise ValueError("compiler usage differs from the raw local model event")
    if result.decision_usage != _local_usage_from_output(events[1].output):
        raise ValueError("decision usage differs from the raw local model event")
    _validate_local_model_event(
        events[1],
        prompt=decision_prompt,
        schema=_local_decision_schema(LOCAL_OLLAMA_MODEL),
        seed=seed,
    )

    try:
        delta = LocalMemoryDeltaWire.model_validate_json(
            events[0].output.completion
        ).to_domain()
    except (ValidationError, ValueError) as error:
        raise ValueError("local compiler preflight output is invalid") from error
    compiler_call = CompilerCall(
        delta=delta,
        usage=Usage(cost_usd=0.0),
        raw_completion=events[0].output.completion,
    )
    if not _compiler_preflight_semantics(compiler_call):
        raise ValueError("local compiler preflight semantics are invalid")
    try:
        decision = LocalDecisionWire.model_validate_json(
            events[1].output.completion
        ).to_domain()
    except (ValidationError, ValueError) as error:
        raise ValueError("local decision preflight output is invalid") from error
    if decision.actions:
        raise ValueError("local decision preflight emitted a false reminder")


def validate_local_w2_preflight_model_events(
    events: list[ModelEvent],
    *,
    fixture: Mapping[str, Any],
    result: LocalModelPreflightW2Result,
    seed: int = 101,
) -> Usage:
    """Validate exactly C1,C2,C3,D1 and return their aggregate setup usage."""

    if len(events) != 4:
        raise ValueError("local W2 preflight must contain exactly four model calls")
    if (
        not result.passed
        or result.loaded_model is None
        or not result.same_model_for_compiler_and_decision
        or result.fixture_sha256 != LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256
    ):
        raise ValueError("local W2 semantic preflight result did not pass")
    if any(
        case.parse_error
        or not case.semantic_valid
        or not case.usage_complete
        or not case.cost_complete
        or case.usage.cost_usd != 0.0
        for case in result.cases
    ):
        raise ValueError("local W2 preflight result flags are inconsistent")

    prompts = local_w2_preflight_prompts(fixture)
    compiler_cases = fixture.get("compiler_cases")
    if not isinstance(compiler_cases, list) or len(compiler_cases) != 3:
        raise ValueError("local W2 compiler fixture cases are invalid")
    aggregate = Usage(cost_usd=0.0)
    for index, case in enumerate(compiler_cases):
        if not isinstance(case, dict):
            raise ValueError("local W2 compiler fixture case is invalid")
        event = events[index]
        _validate_local_model_event(
            event,
            prompt=prompts[index],
            schema=_local_memory_delta_schema(LOCAL_OLLAMA_MODEL),
            seed=seed,
        )
        usage = _local_usage_from_output(event.output)
        if usage.input_tokens <= 0 or usage.output_tokens <= 0:
            raise ValueError("W2 compiler event has incomplete token usage")
        if result.cases[index].usage != usage:
            raise ValueError("W2 compiler usage differs from the raw model event")
        parse_error, semantic_valid = _w2_compiler_semantic_valid(
            event.output.completion,
            case,
        )
        if (
            result.cases[index].parse_error != parse_error
            or result.cases[index].semantic_valid != semantic_valid
        ):
            raise ValueError("W2 compiler semantics differ from the raw model event")
        aggregate = aggregate.plus(usage)

    decision_event = events[3]
    _validate_local_model_event(
        decision_event,
        prompt=prompts[3],
        schema=_local_decision_schema(LOCAL_OLLAMA_MODEL),
        seed=seed,
    )
    decision_usage = _local_usage_from_output(decision_event.output)
    if decision_usage.input_tokens <= 0 or decision_usage.output_tokens <= 0:
        raise ValueError("W2 decision event has incomplete token usage")
    if result.cases[3].usage != decision_usage:
        raise ValueError("W2 decision usage differs from the raw model event")
    parse_error, semantic_valid = _w2_decision_semantic_valid(
        decision_event.output.completion
    )
    if (
        result.cases[3].parse_error != parse_error
        or result.cases[3].semantic_valid != semantic_valid
    ):
        raise ValueError("W2 decision semantics differ from the raw model event")
    aggregate = aggregate.plus(decision_usage)
    if aggregate != result.usage:
        raise ValueError("W2 aggregate usage differs from the result")
    return aggregate


def validate_local_w2_preflight_log(
    log: EvalLog,
    *,
    fixture: Mapping[str, Any],
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
) -> LocalModelPreflightW2Result:
    """Validate the exact four-call frozen W2 standalone preflight log."""

    if log.status != "success" or log.invalidated:
        raise ValueError("local W2 preflight log is not a successful evaluation")
    spec = log.eval
    expected_metadata = {
        "purpose": LOCAL_MODEL_PREFLIGHT_W2_PURPOSE,
        "track": "local_zero_api_cost",
        "hypothesis_test_eligible": False,
        "pricing_config_sha256": expected_pricing_sha256,
        "preflight_fixture_sha256": LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256,
    }
    if (
        spec.task != "local_model_preflight_w2"
        or spec.task_registry_name != "local_model_preflight_w2"
        or spec.task_version != LOCAL_MODEL_PREFLIGHT_W2_TASK_VERSION
        or (spec.metadata or {}) != expected_metadata
    ):
        raise ValueError("local W2 preflight task identity differs from the protocol")
    if spec.model != LOCAL_OLLAMA_MODEL:
        raise ValueError("local W2 preflight used a different model")
    if str(spec.model_base_url).rstrip("/") != LOCAL_OLLAMA_BASE_URL:
        raise ValueError("local W2 preflight used a different provider route")
    if spec.model_args != {}:
        raise ValueError("local W2 preflight used unpinned provider arguments")
    effective_model_config = spec.model_generate_config.merge(
        log.plan.config
    ).model_dump(mode="json", exclude_none=True)
    if effective_model_config != {
        "max_retries": 0,
        "max_connections": 1,
        "adaptive_connections": False,
        "temperature": 0.0,
        "seed": seed,
        "cache": False,
    }:
        raise ValueError("local W2 preflight model configuration differs from the pin")
    revision = spec.revision
    revision_commit = revision.commit if revision is not None else None
    if (
        revision is None
        or revision.dirty is not False
        or not isinstance(revision_commit, str)
        or re.fullmatch(r"[0-9a-f]{7,40}", revision_commit) is None
        or not expected_git_commit.startswith(revision_commit)
    ):
        raise ValueError("local W2 preflight was not run from the frozen clean commit")
    if spec.config.log_model_api is not True:
        raise ValueError("local W2 preflight did not retain raw model API calls")
    if (
        spec.config.max_samples != 1
        or spec.config.max_tasks != 1
        or spec.config.epochs != 1
    ):
        raise ValueError("local W2 preflight has an invalid sample/task policy")
    if log.samples is None or len(log.samples) != 1:
        raise ValueError("local W2 preflight requires exactly one synthetic sample")
    sample = log.samples[0]
    if (
        sample.id != LOCAL_MODEL_PREFLIGHT_W2_SAMPLE_ID
        or sample.epoch != 1
        or sample.error is not None
        or sample.invalidation is not None
        or bool(sample.error_retries)
        or sample.output is None
    ):
        raise ValueError("local W2 preflight sample failed or has the wrong identity")
    if sample.output.model != LOCAL_OLLAMA_MODEL:
        raise ValueError("local W2 preflight output has the wrong model identity")
    result = LocalModelPreflightW2Result.model_validate_json(sample.output.completion)
    serialized_result = result.model_dump(mode="json")
    if sample.metadata != {"anamnesis.local_preflight_w2": serialized_result}:
        raise ValueError("local W2 preflight sample metadata differs from its result")
    if sample.store != {"anamnesis.local_preflight_w2": serialized_result}:
        raise ValueError("local W2 preflight sample store differs from its result")
    events = [event for event in sample.events if isinstance(event, ModelEvent)]
    validate_local_w2_preflight_model_events(
        events,
        fixture=fixture,
        result=result,
        seed=seed,
    )
    return result


def validate_local_w3_preflight_model_events(
    events: list[ModelEvent],
    *,
    fixture: Mapping[str, Any],
    result: LocalModelPreflightW3Result,
    seed: int = 101,
    expected_model: str = LOCAL_OLLAMA_MODEL,
    expected_service_model: str = LOCAL_OLLAMA_SERVICE_MODEL,
    expected_extra_body: Mapping[str, Any] | None = None,
) -> Usage:
    """Validate exactly C1-C8,D1 and return aggregate setup usage."""

    if len(events) != 9:
        raise ValueError("local W3 preflight must contain exactly nine model calls")
    if (
        not result.passed
        or result.loaded_model is None
        or not result.same_model_for_compiler_and_decision
        or result.fixture_sha256 != LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256
        or result.protocol_sha256 != LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256
    ):
        raise ValueError("local W3 semantic preflight result did not pass")
    if any(
        case.parse_error
        or not case.semantic_valid
        or not case.usage_complete
        or not case.cost_complete
        or case.usage.cost_usd != 0.0
        for case in result.cases
    ):
        raise ValueError("local W3 preflight result flags are inconsistent")

    prompts = local_w3_preflight_prompts(fixture)
    compiler_cases = fixture.get("compiler_cases")
    if not isinstance(compiler_cases, list) or len(compiler_cases) != 8:
        raise ValueError("local W3 compiler fixture cases are invalid")
    aggregate = Usage(cost_usd=0.0)
    for index, case in enumerate(compiler_cases):
        if not isinstance(case, dict):
            raise ValueError("local W3 compiler fixture case is invalid")
        event = events[index]
        _validate_local_model_event(
            event,
            prompt=prompts[index],
            schema=_local_memory_delta_schema(expected_model),
            seed=seed,
            expected_model=expected_model,
            expected_service_model=expected_service_model,
            expected_extra_body=expected_extra_body,
        )
        usage = _local_usage_from_output(event.output)
        if usage.input_tokens <= 0 or usage.output_tokens <= 0:
            raise ValueError("W3 compiler event has incomplete token usage")
        if result.cases[index].usage != usage:
            raise ValueError("W3 compiler usage differs from the raw model event")
        parse_error, semantic_valid = _w3_compiler_semantic_valid(
            event.output.completion,
            case,
        )
        if (
            result.cases[index].parse_error != parse_error
            or result.cases[index].semantic_valid != semantic_valid
        ):
            raise ValueError("W3 compiler semantics differ from the raw model event")
        aggregate = aggregate.plus(usage)

    decision_event = events[8]
    _validate_local_model_event(
        decision_event,
        prompt=prompts[8],
        schema=_local_decision_schema(expected_model),
        seed=seed,
        expected_model=expected_model,
        expected_service_model=expected_service_model,
        expected_extra_body=expected_extra_body,
    )
    decision_usage = _local_usage_from_output(decision_event.output)
    if decision_usage.input_tokens <= 0 or decision_usage.output_tokens <= 0:
        raise ValueError("W3 decision event has incomplete token usage")
    if result.cases[8].usage != decision_usage:
        raise ValueError("W3 decision usage differs from the raw model event")
    parse_error, semantic_valid = _w2_decision_semantic_valid(
        decision_event.output.completion
    )
    if (
        result.cases[8].parse_error != parse_error
        or result.cases[8].semantic_valid != semantic_valid
    ):
        raise ValueError("W3 decision semantics differ from the raw model event")
    aggregate = aggregate.plus(decision_usage)
    if aggregate != result.usage:
        raise ValueError("W3 aggregate usage differs from the result")
    return aggregate


def validate_local_w3_preflight_log(
    log: EvalLog,
    *,
    fixture: Mapping[str, Any],
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
    expected_model: str = LOCAL_OLLAMA_MODEL,
    expected_service_model: str = LOCAL_OLLAMA_SERVICE_MODEL,
    expected_task: str = "local_model_preflight_w3",
    expected_task_version: str = LOCAL_MODEL_PREFLIGHT_W3_TASK_VERSION,
    expected_purpose: str = LOCAL_MODEL_PREFLIGHT_W3_PURPOSE,
    expected_sample_id: str = LOCAL_MODEL_PREFLIGHT_W3_SAMPLE_ID,
    expected_store_key: str = "anamnesis.local_preflight_w3",
    expected_metadata_extra: Mapping[str, Any] | None = None,
    expected_extra_body: Mapping[str, Any] | None = None,
) -> LocalModelPreflightW3Result:
    """Validate the exact nine-call frozen W3 standalone preflight log."""

    if log.status != "success" or log.invalidated:
        raise ValueError("local W3 preflight log is not a successful evaluation")
    spec = log.eval
    expected_metadata = {
        "purpose": expected_purpose,
        "track": "local_zero_api_cost",
        "hypothesis_test_eligible": False,
        "pricing_config_sha256": expected_pricing_sha256,
        "preflight_fixture_sha256": LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
        "preflight_protocol_sha256": LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
    }
    if expected_metadata_extra is not None:
        expected_metadata.update(expected_metadata_extra)
    if (
        spec.task != expected_task
        or spec.task_registry_name != expected_task
        or spec.task_version != expected_task_version
        or (spec.metadata or {}) != expected_metadata
    ):
        raise ValueError("local W3 preflight task identity differs from the protocol")
    if spec.model != expected_model:
        raise ValueError("local W3 preflight used a different model")
    if str(spec.model_base_url).rstrip("/") != LOCAL_OLLAMA_BASE_URL:
        raise ValueError("local W3 preflight used a different provider route")
    if spec.model_args != {}:
        raise ValueError("local W3 preflight used unpinned provider arguments")
    effective_model_config = spec.model_generate_config.merge(
        log.plan.config
    ).model_dump(mode="json", exclude_none=True)
    expected_model_config: dict[str, Any] = {
        "max_retries": 0,
        "max_connections": 1,
        "adaptive_connections": False,
        "temperature": 0.0,
        "seed": seed,
        "cache": False,
    }
    if expected_extra_body is not None:
        expected_model_config["extra_body"] = dict(expected_extra_body)
    if effective_model_config != expected_model_config:
        raise ValueError("local W3 preflight model configuration differs from the pin")
    revision = spec.revision
    revision_commit = revision.commit if revision is not None else None
    if (
        revision is None
        or revision.dirty is not False
        or not isinstance(revision_commit, str)
        or re.fullmatch(r"[0-9a-f]{7,40}", revision_commit) is None
        or not expected_git_commit.startswith(revision_commit)
    ):
        raise ValueError("local W3 preflight was not run from the frozen clean commit")
    if spec.config.log_model_api is not True:
        raise ValueError("local W3 preflight did not retain raw model API calls")
    if (
        spec.config.max_samples != 1
        or spec.config.max_tasks != 1
        or spec.config.epochs != 1
    ):
        raise ValueError("local W3 preflight has an invalid sample/task policy")
    if log.samples is None or len(log.samples) != 1:
        raise ValueError("local W3 preflight requires exactly one synthetic sample")
    sample = log.samples[0]
    if (
        sample.id != expected_sample_id
        or sample.epoch != 1
        or sample.error is not None
        or sample.invalidation is not None
        or bool(sample.error_retries)
        or sample.output is None
    ):
        raise ValueError("local W3 preflight sample failed or has the wrong identity")
    if sample.output.model != expected_model:
        raise ValueError("local W3 preflight output has the wrong model identity")
    result = LocalModelPreflightW3Result.model_validate_json(sample.output.completion)
    serialized_result = result.model_dump(mode="json")
    if result.model != expected_model:
        raise ValueError("local W3 preflight result has the wrong model identity")
    expected_result_store = {expected_store_key: serialized_result}
    if sample.metadata != expected_result_store:
        raise ValueError("local W3 preflight sample metadata differs from its result")
    if sample.store != expected_result_store:
        raise ValueError("local W3 preflight sample store differs from its result")
    events = [event for event in sample.events if isinstance(event, ModelEvent)]
    validate_local_w3_preflight_model_events(
        events,
        fixture=fixture,
        result=result,
        seed=seed,
        expected_model=expected_model,
        expected_service_model=expected_service_model,
        expected_extra_body=expected_extra_body,
    )
    return result


def validate_local_w3_m2_preflight_log(
    log: EvalLog,
    *,
    fixture: Mapping[str, Any],
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
) -> LocalModelPreflightW3Result:
    """Validate one exact W3 model-only preflight for the pinned M2 model."""

    return validate_local_w3_preflight_log(
        log,
        fixture=fixture,
        expected_git_commit=expected_git_commit,
        expected_pricing_sha256=expected_pricing_sha256,
        seed=seed,
        expected_model=LOCAL_W3_M2_OLLAMA_MODEL,
        expected_service_model=LOCAL_W3_M2_OLLAMA_SERVICE_MODEL,
        expected_task="local_model_preflight_w3_m2",
        expected_task_version=LOCAL_MODEL_PREFLIGHT_W3_M2_TASK_VERSION,
        expected_purpose=LOCAL_MODEL_PREFLIGHT_W3_M2_PURPOSE,
        expected_sample_id=LOCAL_MODEL_PREFLIGHT_W3_M2_SAMPLE_ID,
        expected_store_key="anamnesis.local_preflight_w3_m2",
        expected_metadata_extra={
            "intervention": "model_only",
            "parent_prompt_cell": "W3",
            "model_artifact_sha256": (
                "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
            ),
            "model_only_protocol_sha256": LOCAL_W3_M2_PROTOCOL_SHA256,
        },
    )


def validate_local_w3_m2_t1_preflight_log(
    log: EvalLog,
    *,
    fixture: Mapping[str, Any],
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
) -> LocalModelPreflightW3Result:
    """Validate the exact W3-M2 transport-only no-thinking preflight."""

    return validate_local_w3_preflight_log(
        log,
        fixture=fixture,
        expected_git_commit=expected_git_commit,
        expected_pricing_sha256=expected_pricing_sha256,
        seed=seed,
        expected_model=LOCAL_W3_M2_OLLAMA_MODEL,
        expected_service_model=LOCAL_W3_M2_OLLAMA_SERVICE_MODEL,
        expected_task="local_model_preflight_w3_m2_t1",
        expected_task_version=LOCAL_MODEL_PREFLIGHT_W3_M2_T1_TASK_VERSION,
        expected_purpose=LOCAL_MODEL_PREFLIGHT_W3_M2_T1_PURPOSE,
        expected_sample_id=LOCAL_MODEL_PREFLIGHT_W3_M2_T1_SAMPLE_ID,
        expected_store_key="anamnesis.local_preflight_w3_m2_t1",
        expected_metadata_extra={
            "intervention": "transport_only",
            "parent_cell": "W3-M2",
            "transport_field": "reasoning_effort=none",
            "model_artifact_sha256": (
                "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
            ),
            "model_only_protocol_sha256": LOCAL_W3_M2_PROTOCOL_SHA256,
            "transport_protocol_sha256": LOCAL_W3_M2_T1_PROTOCOL_SHA256,
        },
        expected_extra_body={"reasoning_effort": "none"},
    )


def validate_local_preflight_artifact(
    artifact: ArtifactPin,
    *,
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
) -> None:
    """Bind local preflight semantics to the exact tracked Inspect log bytes."""

    path = Path(artifact.path)
    if path.suffix != ".eval" or not path.is_file():
        raise ValueError("local model preflight artifact must be an Inspect .eval log")
    content = path.read_bytes()
    if artifact.sha256 != hashlib.sha256(content).hexdigest():
        raise ValueError("local model preflight artifact hash mismatch")
    validate_local_preflight_log(
        read_eval_log(path, resolve_attachments=True),
        expected_git_commit=expected_git_commit,
        expected_pricing_sha256=expected_pricing_sha256,
        seed=seed,
    )


def validate_local_w2_preflight_artifact(
    artifact: ArtifactPin,
    *,
    fixture_artifact: ArtifactPin,
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
) -> LocalModelPreflightW2Result:
    """Bind W2 semantics to exact fixture and Inspect `.eval` artifact bytes."""

    fixture_path = Path(fixture_artifact.path)
    if not fixture_path.is_file():
        raise ValueError("local W2 preflight fixture artifact does not exist")
    fixture_content = fixture_path.read_bytes()
    if fixture_artifact.sha256 != hashlib.sha256(fixture_content).hexdigest():
        raise ValueError("local W2 preflight fixture artifact hash mismatch")
    fixture = load_local_w2_preflight_fixture(fixture_path)

    path = Path(artifact.path)
    if path.suffix != ".eval" or not path.is_file():
        raise ValueError("local W2 preflight artifact must be an Inspect .eval log")
    content = path.read_bytes()
    if artifact.sha256 != hashlib.sha256(content).hexdigest():
        raise ValueError("local W2 preflight artifact hash mismatch")
    return validate_local_w2_preflight_log(
        read_eval_log(path, resolve_attachments=True),
        fixture=fixture,
        expected_git_commit=expected_git_commit,
        expected_pricing_sha256=expected_pricing_sha256,
        seed=seed,
    )


def validate_local_w3_preflight_artifact(
    artifact: ArtifactPin,
    *,
    fixture_artifact: ArtifactPin,
    protocol_artifact: ArtifactPin,
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
) -> LocalModelPreflightW3Result:
    """Bind W3 semantics to exact protocol, fixture and `.eval` bytes."""

    protocol_path = Path(protocol_artifact.path)
    if not protocol_path.is_file():
        raise ValueError("local W3 preflight protocol artifact does not exist")
    protocol_content = protocol_path.read_bytes()
    if protocol_artifact.sha256 != hashlib.sha256(protocol_content).hexdigest():
        raise ValueError("local W3 preflight protocol artifact hash mismatch")
    load_local_w3_preflight_protocol(protocol_path)

    fixture_path = Path(fixture_artifact.path)
    if not fixture_path.is_file():
        raise ValueError("local W3 preflight fixture artifact does not exist")
    fixture_content = fixture_path.read_bytes()
    if fixture_artifact.sha256 != hashlib.sha256(fixture_content).hexdigest():
        raise ValueError("local W3 preflight fixture artifact hash mismatch")
    fixture = load_local_w3_preflight_fixture(fixture_path)

    path = Path(artifact.path)
    if path.suffix != ".eval" or not path.is_file():
        raise ValueError("local W3 preflight artifact must be an Inspect .eval log")
    content = path.read_bytes()
    if artifact.sha256 != hashlib.sha256(content).hexdigest():
        raise ValueError("local W3 preflight artifact hash mismatch")
    return validate_local_w3_preflight_log(
        read_eval_log(path, resolve_attachments=True),
        fixture=fixture,
        expected_git_commit=expected_git_commit,
        expected_pricing_sha256=expected_pricing_sha256,
        seed=seed,
    )


def validate_local_w3_m2_preflight_artifact(
    artifact: ArtifactPin,
    *,
    fixture_artifact: ArtifactPin,
    protocol_artifact: ArtifactPin,
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
) -> LocalModelPreflightW3Result:
    """Bind the W3-M2 result to exact W3 inputs and Inspect log bytes."""

    protocol_path = Path(protocol_artifact.path)
    fixture_path = Path(fixture_artifact.path)
    if (
        not protocol_path.is_file()
        or protocol_artifact.sha256
        != hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    ):
        raise ValueError("local W3-M2 protocol artifact mismatch")
    load_local_w3_preflight_protocol(protocol_path)
    if (
        not fixture_path.is_file()
        or fixture_artifact.sha256
        != hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    ):
        raise ValueError("local W3-M2 fixture artifact mismatch")
    fixture = load_local_w3_preflight_fixture(fixture_path)
    path = Path(artifact.path)
    if (
        path.suffix != ".eval"
        or not path.is_file()
        or artifact.sha256 != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("local W3-M2 preflight artifact mismatch")
    return validate_local_w3_m2_preflight_log(
        read_eval_log(path, resolve_attachments=True),
        fixture=fixture,
        expected_git_commit=expected_git_commit,
        expected_pricing_sha256=expected_pricing_sha256,
        seed=seed,
    )


__all__ = [
    "LOCAL_PREFLIGHT_ACTIVE_STATE",
    "LOCAL_PREFLIGHT_EVENT_ID",
    "LOCAL_PREFLIGHT_SAMPLE_ID",
    "local_preflight_event",
    "local_preflight_prompts",
    "validate_local_preflight_artifact",
    "validate_local_preflight_log",
    "validate_local_w2_preflight_artifact",
    "validate_local_w2_preflight_log",
    "validate_local_w2_preflight_model_events",
    "validate_local_w3_preflight_artifact",
    "validate_local_w3_preflight_log",
    "validate_local_w3_preflight_model_events",
    "validate_local_w3_m2_t1_preflight_log",
]
