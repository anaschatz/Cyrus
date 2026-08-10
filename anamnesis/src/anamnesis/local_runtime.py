"""Isolated Ollama runtime for zero-API-cost diagnostic experiments.

The hosted OpenAI adapter remains the preregistered headline contract.  This
module deliberately uses separate prompts, wire schemas, endpoint checks and
accounting so a local compatibility concession cannot silently weaken that
contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.client
import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageUser,
    ModelCost,
    ModelOutput,
    ResponseSchema,
    get_model,
    get_model_info,
)
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import JSONSchema, json_schema
from pydantic import Field, ValidationError, model_validator

from anamnesis.baselines import (
    AnamnesisMemoryStrategy,
    FastEmbedVectorizer,
    Vectorizer,
    create_strategy,
)
from anamnesis.inspect_adapter import (
    SCENARIO_RUN_STORE_KEY,
    _scenario_from_state,
    _scenario_sha256_from_state,
)
from anamnesis.local_experiment import (
    LOCAL_BASE_URL,
    LOCAL_MODEL_ID,
    LOCAL_OLLAMA_VERSION,
    LOCAL_W3_M2_MODEL_ID,
    LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256,
    LOCAL_WRITER_W3_DATASET_SHA256,
    LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
    LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
    LOCAL_WRITER_W3_REFERENCE_SHA256,
    require_local_only_environment,
)
from anamnesis.local_wire import (
    LOCAL_MEMORY_COMPILER_VERSION,
    LOCAL_MEMORY_COMPILER_W2_VERSION,
    LOCAL_MEMORY_COMPILER_W3_ADDENDUM,
    LOCAL_MEMORY_COMPILER_W3_VERSION,
    LocalAtTriggerWire,
    LocalConditionTransitionTriggerWire,
    LocalMemoryDeltaWire,
    LocalPayloadWire,
    LocalRecurringTriggerWire,
    build_local_memory_compiler_prompt,
    build_local_memory_compiler_w2_prompt,
    build_local_memory_compiler_w3_prompt,
    local_memory_compiler_w3_contract,
)
from anamnesis.memory import (
    AtTrigger,
    CompilerCall,
    CompilerRequest,
    CreateIntent,
    SetFact,
    UpdateIntent,
)
from anamnesis.oracle import (
    ORACLE_COMPILER_VERSION,
    ORACLE_SYSTEM_NAME,
    OracleAnamnesisMemoryStrategy,
    OracleCompiler,
    OracleCompilerArtifact,
)
from anamnesis.prompts import ACTION_OUTPUT_GUIDE, build_decision_prompt
from anamnesis.runner import (
    DecisionCall,
    DecisionModel,
    DecisionRequest,
    run_scenario,
)
from anamnesis.runtime_contract import anamnesis_runtime_contract
from anamnesis.schema import (
    Decision,
    MemoryView,
    ObservableEvent,
    ProposedAction,
    RuntimeScenario,
    ScenarioRun,
    StrictModel,
    Usage,
)

LOCAL_RUNTIME_VERSION = "ollama.local.v0.1"
LOCAL_DECISION_VERSION = "ollama.decision.v0.2"
LOCAL_OLLAMA_MODEL = LOCAL_MODEL_ID
LOCAL_OLLAMA_SERVICE_MODEL = "qwen3:4b-instruct"
LOCAL_W3_M2_OLLAMA_MODEL = LOCAL_W3_M2_MODEL_ID
LOCAL_W3_M2_OLLAMA_SERVICE_MODEL = "qwen3.5:9b-q4_K_M"
LOCAL_OLLAMA_BASE_URL = LOCAL_BASE_URL
LOCAL_OLLAMA_PS_URL = "http://127.0.0.1:11434/api/ps"
LOCAL_OLLAMA_MANIFEST_SHA256 = (
    "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
)
LOCAL_OLLAMA_MODEL_BLOB_SHA256 = (
    "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
)
LOCAL_OLLAMA_CONTEXT_LENGTH = 4096
LOCAL_OLLAMA_FAMILY = "qwen3"
LOCAL_OLLAMA_PARAMETER_SIZE = "4.0B"
LOCAL_OLLAMA_QUANTIZATION = "Q4_K_M"
LOCAL_W3_M2_OLLAMA_MANIFEST_SHA256 = (
    "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
)
LOCAL_W3_M2_OLLAMA_MODEL_BLOB_SHA256 = (
    "dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
)
LOCAL_W3_M2_OLLAMA_FAMILY = "qwen35"
LOCAL_W3_M2_OLLAMA_PARAMETER_SIZE = "9.7B"
LOCAL_NO_CLOUD_ENV = "OLLAMA_NO_CLOUD"
LOCAL_CONTEXT_ENV = "OLLAMA_CONTEXT_LENGTH"
LOCAL_HOST_ENV = "OLLAMA_HOST"
LOCAL_NUM_PARALLEL_ENV = "OLLAMA_NUM_PARALLEL"
LOCAL_MAX_LOADED_MODELS_ENV = "OLLAMA_MAX_LOADED_MODELS"
LOCAL_PREFLIGHT_METADATA_KEY = "anamnesis.local_preflight"
LOCAL_PREFLIGHT_STORE_KEY = "anamnesis.local_preflight"
LOCAL_PREFLIGHT_W2_METADATA_KEY = "anamnesis.local_preflight_w2"
LOCAL_PREFLIGHT_W2_STORE_KEY = "anamnesis.local_preflight_w2"
LOCAL_PREFLIGHT_W3_METADATA_KEY = "anamnesis.local_preflight_w3"
LOCAL_PREFLIGHT_W3_STORE_KEY = "anamnesis.local_preflight_w3"
LOCAL_PREFLIGHT_W3_M2_METADATA_KEY = "anamnesis.local_preflight_w3_m2"
LOCAL_PREFLIGHT_W3_M2_STORE_KEY = "anamnesis.local_preflight_w3_m2"
LOCAL_PREFLIGHT_W3_M2_T1_METADATA_KEY = "anamnesis.local_preflight_w3_m2_t1"
LOCAL_PREFLIGHT_W3_M2_T1_STORE_KEY = "anamnesis.local_preflight_w3_m2_t1"
LOCAL_MODEL_PREFLIGHT_TASK_VERSION = "local.0.1"
LOCAL_MODEL_PREFLIGHT_PURPOSE = "local-model-semantic-preflight"
LOCAL_MODEL_PREFLIGHT_W2_TASK_VERSION = "local.w2.0.1"
LOCAL_MODEL_PREFLIGHT_W2_PURPOSE = "local-writer-w2-semantic-preflight"
LOCAL_MODEL_PREFLIGHT_W2_SAMPLE_ID = "local-model-preflight-w2-v1"
LOCAL_MODEL_PREFLIGHT_W3_TASK_VERSION = "local.w3.0.1"
LOCAL_MODEL_PREFLIGHT_W3_PURPOSE = "local-writer-w3-semantic-preflight"
LOCAL_MODEL_PREFLIGHT_W3_SAMPLE_ID = "local-model-preflight-w3-v1"
LOCAL_MODEL_PREFLIGHT_W3_M2_TASK_VERSION = "local.w3-m2.0.1"
LOCAL_MODEL_PREFLIGHT_W3_M2_PURPOSE = "local-writer-w3-model-only-preflight"
LOCAL_MODEL_PREFLIGHT_W3_M2_SAMPLE_ID = "local-model-preflight-w3-m2-v1"
LOCAL_MODEL_PREFLIGHT_W3_M2_T1_TASK_VERSION = "local.w3-m2-t1.0.1"
LOCAL_MODEL_PREFLIGHT_W3_M2_T1_PURPOSE = "local-writer-w3-model-transport-preflight"
LOCAL_MODEL_PREFLIGHT_W3_M2_T1_SAMPLE_ID = "local-model-preflight-w3-m2-t1-v1"
LOCAL_SCENARIO_TASK_VERSION = "local.0.1"
LOCAL_SCENARIO_W3_TASK_VERSION = "local.w3.0.1"
LOCAL_ZERO_MODEL_COST = ModelCost(
    input=0.0,
    output=0.0,
    input_cache_write=0.0,
    input_cache_read=0.0,
)

_LOCAL_MODEL_SPECS = {
    LOCAL_OLLAMA_MODEL: {
        "service_model": LOCAL_OLLAMA_SERVICE_MODEL,
        "manifest_sha256": LOCAL_OLLAMA_MANIFEST_SHA256,
        "family": LOCAL_OLLAMA_FAMILY,
        "parameter_size": LOCAL_OLLAMA_PARAMETER_SIZE,
        "quantization": LOCAL_OLLAMA_QUANTIZATION,
    },
    LOCAL_W3_M2_OLLAMA_MODEL: {
        "service_model": LOCAL_W3_M2_OLLAMA_SERVICE_MODEL,
        "manifest_sha256": LOCAL_W3_M2_OLLAMA_MANIFEST_SHA256,
        "family": LOCAL_W3_M2_OLLAMA_FAMILY,
        "parameter_size": LOCAL_W3_M2_OLLAMA_PARAMETER_SIZE,
        "quantization": LOCAL_OLLAMA_QUANTIZATION,
    },
}
LocalSystemName = Literal[
    "no_memory",
    "full_context",
    "vector_rag",
    "anamnesis",
    "anamnesis_oracle_compiler",
]

_HOSTED_NO_ACTION_SENTENCE = (
    '- Return JSON matching the supplied schema. Return {"actions": []} when no '
    "action is due.\n"
)
_LOCAL_NO_ACTION_SENTENCE = (
    "- Return JSON matching the supplied schema. Set mode=no_action with an "
    "empty actions array when no action is due. Set mode=emit only when at "
    "least one reminder is due now.\n"
)
_HOSTED_UNUSED_PAYLOAD_SENTENCE = (
    "- The only optional payload slots are address, build, date, flight, "
    "greenhouse, item, project, quantity, recipient, room, shipment, tank, "
    "and trip. Use null for every unused wire slot.\n"
)
_LOCAL_UNUSED_PAYLOAD_SENTENCE = (
    "- The only optional payload slots are address, build, date, flight, "
    "greenhouse, item, project, quantity, recipient, room, shipment, tank, "
    "and trip. Omit every unused optional payload slot.\n"
)
LOCAL_ACTION_OUTPUT_GUIDE = (
    "Choose exactly one explicit decision mode:\n"
    '- no action: {"mode":"no_action","actions":[]}\n'
    '- emit: {"mode":"emit","actions":[{"kind":"reminder",'
    '"action_key":"creating-event-id","payload":{'
    '"subject":"send the assignment"},"summary":"what to remind",'
    '"evidence_event_ids":["event-id"]}]}\n'
    "For emit, payload.subject is required. Optional payload slots are address, "
    "build, date, flight, greenhouse, item, project, quantity, recipient, room, "
    "shipment, tank, and trip; omit unused slots.\n"
)
LOCAL_STRUCTURED_MEMORY_PRECEDENCE = (
    "Structured-memory precedence (D1):\n"
    "- When Structured memory view is provided by this system, it is "
    "authoritative.\n"
    "- If it contains zero DUE_CANDIDATE blocks, set mode=no_action with an "
    "empty actions array regardless of wording in Available context.\n"
    "- For each DUE_CANDIDATE block, emit exactly one action and copy kind, "
    "action_key, payload, and summary value-for-value from that block's JSON.\n"
    "- Set evidence_event_ids to exactly the block's evidence IDs in displayed "
    "order, followed by Current decision event if it is not already present; "
    "include no other IDs.\n"
    "- A prior EXECUTION suppresses only a DUE_CANDIDATE with the same "
    "occurrence_id. A different occurrence_id or date is a distinct recurring "
    "occurrence even when action_key is the same.\n"
    "- When Structured memory view is not provided by this system, use the "
    "general Rules above.\n"
)


class LocalProposedActionWire(StrictModel):
    """Smaller local action envelope mapped to the unchanged domain action."""

    kind: Literal["reminder"] = "reminder"
    action_key: str
    payload: LocalPayloadWire
    summary: str
    evidence_event_ids: list[str]

    def to_domain(self) -> ProposedAction:
        return ProposedAction(
            kind=self.kind,
            action_key=self.action_key,
            payload=self.payload.to_payload(),
            summary=self.summary,
            evidence_event_ids=self.evidence_event_ids,
        )


class LocalDecisionWire(StrictModel):
    """Decision envelope with an explicit no-action/emit discriminator."""

    mode: Literal["no_action", "emit"]
    actions: list[LocalProposedActionWire]

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode == "no_action" and self.actions:
            raise ValueError("no_action mode requires an empty actions array")
        if self.mode == "emit" and not self.actions:
            raise ValueError("emit mode requires at least one action")
        return self

    def to_domain(self) -> Decision:
        return Decision(actions=[action.to_domain() for action in self.actions])


class LocalOllamaRuntimeAttestation(StrictModel):
    """Static client-side proof that Inspect targets the pinned local route."""

    model: str
    base_url: Literal[LOCAL_OLLAMA_BASE_URL]
    no_cloud: Literal["1"]
    context_length: Literal[LOCAL_OLLAMA_CONTEXT_LENGTH]
    host: Literal["127.0.0.1:11434"]
    num_parallel: Literal[1]
    max_loaded_models: Literal[1]

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.model not in _LOCAL_MODEL_SPECS:
            raise ValueError("runtime attestation identifies an unpinned model")
        return self


class LocalLoadedModelAttestation(StrictModel):
    """Evidence from Ollama that the pinned artifact is resident locally."""

    model: str
    digest: str
    family: str
    parameter_size: str
    quantization_level: str
    context_length: Literal[LOCAL_OLLAMA_CONTEXT_LENGTH]
    size_vram: int = Field(gt=0)
    ollama_version: Literal[LOCAL_OLLAMA_VERSION]

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        matches = [
            spec
            for spec in _LOCAL_MODEL_SPECS.values()
            if spec["service_model"] == self.model
        ]
        if len(matches) != 1:
            raise ValueError("loaded-model attestation identifies an unpinned model")
        spec = matches[0]
        expected = (
            spec["manifest_sha256"],
            spec["family"],
            spec["parameter_size"],
            spec["quantization"],
        )
        if (
            self.digest,
            self.family,
            self.parameter_size,
            self.quantization_level,
        ) != expected:
            raise ValueError(
                "loaded-model digest/family/parameter-size/quantization differs "
                "from the model pin"
            )
        return self


class LocalModelPreflightResult(StrictModel):
    """Parse, semantic, accounting and local-residency compatibility gate."""

    model: Literal[LOCAL_OLLAMA_MODEL]
    runtime: LocalOllamaRuntimeAttestation
    loaded_model: LocalLoadedModelAttestation | None = None
    same_model_for_compiler_and_decision: bool
    compiler_parse_error: bool
    decision_parse_error: bool
    compiler_semantic_valid: bool
    decision_semantic_valid: bool
    compiler_usage: Usage
    decision_usage: Usage
    compiler_usage_complete: bool
    decision_usage_complete: bool
    compiler_cost_complete: bool
    decision_cost_complete: bool
    compiler_latency_ms: float = Field(ge=0)
    decision_latency_ms: float = Field(ge=0)
    residency_probe_latency_ms: float = Field(ge=0)
    passed: bool

    @property
    def setup_latency_ms(self) -> float:
        return (
            self.compiler_latency_ms
            + self.decision_latency_ms
            + self.residency_probe_latency_ms
        )


class LocalModelPreflightW2CaseResult(StrictModel):
    """One ordered semantic result from the frozen W2 compatibility fixture."""

    case_id: Literal["C1", "C2", "C3", "D1"]
    role: Literal["compiler", "decision"]
    parse_error: bool
    semantic_valid: bool
    usage: Usage
    usage_complete: bool
    cost_complete: bool
    latency_ms: float = Field(ge=0)


class LocalModelPreflightW2Result(StrictModel):
    """Four-call W2 semantic, accounting and local-residency gate result."""

    model: Literal[LOCAL_OLLAMA_MODEL]
    runtime: LocalOllamaRuntimeAttestation
    loaded_model: LocalLoadedModelAttestation | None = None
    same_model_for_compiler_and_decision: bool
    cases: list[LocalModelPreflightW2CaseResult] = Field(min_length=4, max_length=4)
    residency_probe_latency_ms: float = Field(ge=0)
    fixture_sha256: Literal[LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256]
    passed: bool

    @model_validator(mode="after")
    def validate_case_order(self) -> Self:
        identity = [(case.case_id, case.role) for case in self.cases]
        expected = [
            ("C1", "compiler"),
            ("C2", "compiler"),
            ("C3", "compiler"),
            ("D1", "decision"),
        ]
        if identity != expected:
            raise ValueError("W2 preflight cases must be ordered C1,C2,C3,D1")
        return self

    @property
    def setup_latency_ms(self) -> float:
        return sum(case.latency_ms for case in self.cases) + (
            self.residency_probe_latency_ms
        )

    @property
    def usage(self) -> Usage:
        total = Usage(cost_usd=0.0)
        for case in self.cases:
            total = total.plus(case.usage)
        return total


class LocalModelPreflightW3CaseResult(StrictModel):
    """One ordered semantic result from the frozen W3 compatibility fixture."""

    case_id: Literal["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "D1"]
    role: Literal["compiler", "decision"]
    parse_error: bool
    semantic_valid: bool
    usage: Usage
    usage_complete: bool
    cost_complete: bool
    latency_ms: float = Field(ge=0)


class LocalModelPreflightW3Result(StrictModel):
    """Nine-call W3 semantic, accounting and local-residency gate result."""

    model: str
    runtime: LocalOllamaRuntimeAttestation
    loaded_model: LocalLoadedModelAttestation | None = None
    same_model_for_compiler_and_decision: bool
    cases: list[LocalModelPreflightW3CaseResult] = Field(min_length=9, max_length=9)
    residency_probe_latency_ms: float = Field(ge=0)
    fixture_sha256: Literal[LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256]
    protocol_sha256: Literal[LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256]
    passed: bool

    @model_validator(mode="after")
    def validate_case_order(self) -> Self:
        if self.model not in _LOCAL_MODEL_SPECS:
            raise ValueError("W3 preflight result identifies an unpinned model")
        identity = [(case.case_id, case.role) for case in self.cases]
        expected = [
            ("C1", "compiler"),
            ("C2", "compiler"),
            ("C3", "compiler"),
            ("C4", "compiler"),
            ("C5", "compiler"),
            ("C6", "compiler"),
            ("C7", "compiler"),
            ("C8", "compiler"),
            ("D1", "decision"),
        ]
        if identity != expected:
            raise ValueError("W3 preflight cases must be ordered C1-C8,D1")
        return self

    @property
    def setup_latency_ms(self) -> float:
        return sum(case.latency_ms for case in self.cases) + (
            self.residency_probe_latency_ms
        )

    @property
    def usage(self) -> Usage:
        total = Usage(cost_usd=0.0)
        for case in self.cases:
            total = total.plus(case.usage)
        return total


def build_local_decision_prompt(
    *,
    now: str,
    current_event_id: str,
    context_events: list[ObservableEvent],
    decision_history: list[Any] | None = None,
    memory_view: Any | None = None,
    retrospective_recall: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Render the common decision information with a local-only wire guide."""

    hosted = build_decision_prompt(
        now=now,
        current_event_id=current_event_id,
        context_events=context_events,
        decision_history=decision_history,
        memory_view=memory_view,
        retrospective_recall=retrospective_recall,
    )
    if (
        _HOSTED_NO_ACTION_SENTENCE not in hosted
        or _HOSTED_UNUSED_PAYLOAD_SENTENCE not in hosted
        or ACTION_OUTPUT_GUIDE not in hosted
    ):
        raise RuntimeError("hosted decision prompt changed without local prompt review")
    return (
        hosted.replace(
            _HOSTED_NO_ACTION_SENTENCE,
            f"{_LOCAL_NO_ACTION_SENTENCE}\n{LOCAL_STRUCTURED_MEMORY_PRECEDENCE}",
        )
        .replace(
            _HOSTED_UNUSED_PAYLOAD_SENTENCE,
            _LOCAL_UNUSED_PAYLOAD_SENTENCE,
        )
        .replace(ACTION_OUTPUT_GUIDE, LOCAL_ACTION_OUTPUT_GUIDE)
    )


def local_decision_contract() -> str:
    """Return the complete local decision prompt/schema fingerprint input."""

    prompt = local_decision_prompt_contract()
    schema = local_decision_schema_contract()
    return f"{LOCAL_DECISION_VERSION}\n{prompt}\n{schema}"


def local_decision_prompt_contract() -> str:
    """Render the static local decision prompt sentinel used for hashing."""

    sentinel = ObservableEvent(
        id="<event-id>",
        at=datetime.fromisoformat("2000-01-01T00:00:00+00:00"),
        kind="user_message",
        text="<event-text>",
    )
    return build_local_decision_prompt(
        now="<current-time>",
        current_event_id="<current-event-id>",
        context_events=[sentinel],
        decision_history=[],
        memory_view=None,
    )


def local_decision_schema_contract() -> str:
    """Serialize the exact Inspect response schema sent to Ollama."""

    return json.dumps(
        _local_decision_schema(LOCAL_OLLAMA_MODEL).model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def local_memory_compiler_prompt_contract() -> str:
    """Render the static local compiler prompt sentinel used for hashing."""

    sentinel = ObservableEvent(
        id="<event-id>",
        at=datetime.fromisoformat("2000-01-01T00:00:00+00:00"),
        kind="user_message",
        text="<event-text>",
    )
    return build_local_memory_compiler_prompt(
        event=sentinel,
        active_state='{"facts":[],"intents":[]}',
    )


def local_memory_compiler_w2_prompt_contract() -> str:
    """Render the static W2 compiler prompt sentinel used for hashing."""

    sentinel = ObservableEvent(
        id="<event-id>",
        at=datetime.fromisoformat("2000-01-01T00:00:00+00:00"),
        kind="user_message",
        text="<event-text>",
    )
    return build_local_memory_compiler_w2_prompt(
        event=sentinel,
        active_state='{"facts":[],"intents":[]}',
    )


def local_memory_compiler_w3_prompt_contract() -> str:
    """Render the static W3 compiler prompt sentinel used for hashing."""

    sentinel = ObservableEvent(
        id="<event-id>",
        at=datetime.fromisoformat("2000-01-01T00:00:00+00:00"),
        kind="user_message",
        text="<event-text>",
    )
    return build_local_memory_compiler_w3_prompt(
        event=sentinel,
        active_state='{"facts":[],"intents":[]}',
    )


def local_memory_compiler_schema_contract() -> str:
    """Serialize the exact Inspect compiler response schema sent to Ollama."""

    return json.dumps(
        _local_memory_delta_schema(LOCAL_OLLAMA_MODEL).model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def local_memory_compiler_transport_contract() -> str:
    """Bind the local compiler prompt to its actual Inspect wire schema."""

    return (
        f"{LOCAL_MEMORY_COMPILER_VERSION}\n"
        f"{local_memory_compiler_prompt_contract()}\n"
        f"{local_memory_compiler_schema_contract()}"
    )


def local_memory_compiler_w2_transport_contract() -> str:
    """Bind the W2 prompt to the unchanged Inspect compiler wire schema."""

    return (
        f"{LOCAL_MEMORY_COMPILER_W2_VERSION}\n"
        f"{local_memory_compiler_w2_prompt_contract()}\n"
        f"{local_memory_compiler_schema_contract()}"
    )


def local_memory_compiler_w3_transport_contract() -> str:
    """Bind the W3 prompt to the unchanged Inspect compiler wire schema."""

    return (
        f"{LOCAL_MEMORY_COMPILER_W3_VERSION}\n"
        f"{local_memory_compiler_w3_prompt_contract()}\n"
        f"{local_memory_compiler_schema_contract()}"
    )


def _local_decision_schema(model_name: str) -> ResponseSchema:
    if model_name not in _LOCAL_MODEL_SPECS:
        raise ValueError("local decision schema requires the pinned Ollama model")
    return ResponseSchema(
        name="anamnesis_local_decision",
        json_schema=json_schema(LocalDecisionWire),
        strict=None,
    )


def _local_memory_delta_schema(model_name: str) -> ResponseSchema:
    if model_name not in _LOCAL_MODEL_SPECS:
        raise ValueError("local compiler schema requires the pinned Ollama model")
    compiler_schema = json_schema(LocalMemoryDeltaWire)
    if compiler_schema.properties is None:
        raise RuntimeError("local compiler schema has no root properties")

    trigger_variants: list[JSONSchema] = []
    for model, trigger_type in (
        (LocalAtTriggerWire, "at"),
        (LocalRecurringTriggerWire, "recurring"),
        (LocalConditionTransitionTriggerWire, "condition_transition"),
    ):
        variant = json_schema(model)
        if variant.properties is None or "type" not in variant.properties:
            raise RuntimeError("local trigger schema is missing its discriminator")
        variant.properties["type"].enum = [trigger_type]
        trigger_variants.append(variant)
    trigger_schema = JSONSchema(anyOf=trigger_variants)

    creates = compiler_schema.properties["intent_creates"].items
    updates = compiler_schema.properties["intent_updates"].items
    if (
        creates is None
        or creates.properties is None
        or updates is None
        or updates.properties is None
    ):
        raise RuntimeError("local compiler intent schemas are incomplete")
    creates.properties["trigger"] = trigger_schema
    updates.properties["trigger"] = JSONSchema(
        anyOf=[*trigger_variants, JSONSchema(type="null")]
    )
    return ResponseSchema(
        name="anamnesis_local_memory_delta",
        json_schema=compiler_schema,
        strict=None,
    )


def _active_model_base_url(active_model: object) -> str:
    api = getattr(active_model, "api", None)
    base_url = getattr(api, "base_url", None)
    client = getattr(api, "client", None)
    client_base_url = getattr(client, "base_url", None)
    if base_url is None or client_base_url is None:
        raise ValueError("active Ollama model is missing its endpoint")
    normalized_api = str(base_url).rstrip("/")
    normalized_client = str(client_base_url).rstrip("/")
    if normalized_api != normalized_client:
        raise ValueError("Ollama provider and client endpoints differ")
    return normalized_api


def _verify_local_ollama_runtime(
    active_model: object,
    model_name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> LocalOllamaRuntimeAttestation:
    """Fail closed unless Inspect targets the exact local, no-cloud runtime."""

    if model_name not in _LOCAL_MODEL_SPECS:
        raise ValueError("local track requires the pinned Ollama model")
    api = getattr(active_model, "api", None)
    if getattr(api, "service", None) != "Ollama":
        raise ValueError("active model is not the Inspect Ollama provider")
    base_url = _active_model_base_url(active_model)
    parsed = urlsplit(base_url)
    if (
        base_url != LOCAL_OLLAMA_BASE_URL
        or parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 11434
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("active Ollama endpoint is not the pinned localhost route")

    active_environ = os.environ if environ is None else environ
    require_local_only_environment(active_environ)
    return LocalOllamaRuntimeAttestation(
        model=model_name,
        base_url=LOCAL_OLLAMA_BASE_URL,
        no_cloud="1",
        context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
        host="127.0.0.1:11434",
        num_parallel=1,
        max_loaded_models=1,
    )


def _verify_local_output_model(
    output: ModelOutput,
    *,
    expected_model: str | None = None,
) -> None:
    allowed = {
        value
        for model, spec in _LOCAL_MODEL_SPECS.items()
        for value in (model, str(spec["service_model"]))
        if expected_model is None or model == expected_model
    }
    if output.model not in allowed:
        raise ValueError("Ollama response model differs from the pinned local model")


def _verify_effective_zero_model_cost(active_model: object) -> None:
    info = get_model_info(active_model)  # type: ignore[arg-type]
    if info is None or info.cost != LOCAL_ZERO_MODEL_COST:
        raise ValueError("active Inspect model lacks the pinned all-zero local pricing")


def _local_usage_from_output(
    output: ModelOutput,
    *,
    expected_model: str | None = None,
) -> Usage:
    """Account local inference as complete API cost $0, never as unknown cost."""

    _verify_local_output_model(output, expected_model=expected_model)
    model_usage = output.usage
    if model_usage is None:
        return Usage(cost_usd=0.0)
    reported_cost = model_usage.total_cost
    if reported_cost not in (None, 0, 0.0):
        raise ValueError("verified local Ollama output reported a non-zero API cost")
    uncached = int(model_usage.input_tokens)
    cache_read = int(getattr(model_usage, "input_tokens_cache_read", 0) or 0)
    cache_write = int(getattr(model_usage, "input_tokens_cache_write", 0) or 0)
    return Usage(
        input_tokens=uncached + cache_read + cache_write,
        uncached_input_tokens=uncached,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        output_tokens=int(model_usage.output_tokens),
        cost_usd=0.0,
    )


def _local_usage_is_complete(
    output: ModelOutput,
    *,
    expected_model: str | None = None,
) -> bool:
    usage = _local_usage_from_output(output, expected_model=expected_model)
    return (
        output.usage is not None and usage.input_tokens > 0 and usage.output_tokens > 0
    )


class LocalInspectDecisionModel(DecisionModel):
    """Inspect adapter that can only talk to the pinned local Ollama runtime."""

    def __init__(self, state: TaskState, generate: Generate) -> None:
        active_model = get_model()
        self.name = str(state.model)
        self.runtime_attestation = _verify_local_ollama_runtime(
            active_model,
            self.name,
        )
        _verify_effective_zero_model_cost(active_model)
        self.state = state
        self._generate = generate
        self._response_schema = _local_decision_schema(self.name)

    async def complete_structured(
        self,
        *,
        prompt: str,
        response_schema: ResponseSchema,
    ) -> tuple[ModelOutput, float]:
        """Make one schema-constrained call; the task config forbids retries."""

        self.state.messages = [ChatMessageUser(content=prompt)]
        started = perf_counter()
        self.state = await self._generate(
            self.state,
            tool_calls="none",
            response_schema=response_schema,
        )
        output = self.state.output
        _verify_local_output_model(output, expected_model=self.name)
        return output, max(0.0, (perf_counter() - started) * 1000)

    async def decide(self, request: DecisionRequest) -> DecisionCall:
        output, latency_ms = await self.complete_structured(
            prompt=request.prompt,
            response_schema=self._response_schema,
        )
        try:
            decision = LocalDecisionWire.model_validate_json(
                output.completion
            ).to_domain()
            parse_error = False
        except (ValidationError, ValueError):
            decision = Decision()
            parse_error = True
        usage = _local_usage_from_output(output, expected_model=self.name)
        usage_complete = _local_usage_is_complete(output, expected_model=self.name)
        return DecisionCall(
            decision=decision,
            usage=usage,
            latency_ms=latency_ms,
            parse_error=parse_error,
            raw_completion=output.completion,
            usage_complete=usage_complete,
            cost_complete=usage_complete,
        )


class LocalInspectMemoryCompiler:
    """Local compiler sharing the exact decision model and Inspect session."""

    def __init__(
        self,
        model: LocalInspectDecisionModel,
        *,
        prompt_builder: Callable[..., str] = build_local_memory_compiler_prompt,
    ) -> None:
        self.name = model.name
        self._model = model
        self._response_schema = _local_memory_delta_schema(self.name)
        self._prompt_builder = prompt_builder

    async def compile(self, request: CompilerRequest) -> CompilerCall:
        output, latency_ms = await self._model.complete_structured(
            prompt=self._prompt_builder(
                event=request.event,
                active_state=request.active_state,
            ),
            response_schema=self._response_schema,
        )
        try:
            delta = LocalMemoryDeltaWire.model_validate_json(
                output.completion
            ).to_domain()
            parse_error = False
        except (ValidationError, ValueError):
            delta = None
            parse_error = True
        usage = _local_usage_from_output(output)
        usage_complete = _local_usage_is_complete(output)
        return CompilerCall(
            delta=delta,
            usage=usage,
            latency_ms=latency_ms,
            parse_error=parse_error,
            raw_completion=output.completion,
            usage_complete=usage_complete,
            cost_complete=usage_complete,
        )


def _loaded_model_from_ps(
    payload: object,
    *,
    ollama_version: str = LOCAL_OLLAMA_VERSION,
    expected_model: str = LOCAL_OLLAMA_MODEL,
) -> LocalLoadedModelAttestation:
    try:
        spec = _LOCAL_MODEL_SPECS[expected_model]
    except KeyError as error:
        raise ValueError("residency probe requires a pinned model") from error
    service_model = str(spec["service_model"])
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Ollama /api/ps returned an invalid response")
    matches = [
        item
        for item in payload["models"]
        if isinstance(item, dict)
        and item.get("model", item.get("name")) == service_model
    ]
    if len(matches) != 1:
        raise ValueError("pinned Ollama model is not uniquely loaded locally")
    model = matches[0]
    details = model.get("details")
    if not isinstance(details, dict):
        raise ValueError("Ollama /api/ps omitted loaded-model details")
    return LocalLoadedModelAttestation(
        model=service_model,
        digest=model.get("digest"),
        family=details.get("family"),
        parameter_size=details.get("parameter_size"),
        quantization_level=details.get("quantization_level"),
        context_length=model.get("context_length"),
        size_vram=model.get("size_vram"),
        ollama_version=ollama_version,
    )


def probe_loaded_local_model(
    expected_model: str = LOCAL_OLLAMA_MODEL,
) -> LocalLoadedModelAttestation:
    """Query only Ollama's loopback process endpoint for residency evidence."""

    # HTTPConnection is intentionally used instead of a URL opener so proxy
    # environment variables cannot route this attestation away from loopback.
    connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=5)
    try:
        connection.request(
            "GET",
            "/api/version",
            headers={"Accept": "application/json"},
        )
        version_response = connection.getresponse()
        if version_response.status != 200:
            raise ValueError("Ollama version probe did not return HTTP 200")
        version_payload = json.loads(version_response.read())
        if not isinstance(version_payload, dict):
            raise ValueError("Ollama version probe returned invalid JSON")
        ollama_version = version_payload.get("version")
        connection.request("GET", "/api/ps", headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("Ollama residency probe did not return HTTP 200")
        payload = json.loads(response.read())
    finally:
        connection.close()
    return _loaded_model_from_ps(
        payload,
        ollama_version=ollama_version,
        expected_model=expected_model,
    )


def _compiler_preflight_semantics(call: CompilerCall) -> bool:
    delta = call.delta
    if delta is None or len(delta.mutations) != 1:
        return False
    mutation = delta.mutations[0]
    if not isinstance(mutation, CreateIntent):
        return False
    expected_at = datetime.fromisoformat("2026-01-05T17:00:00+00:00")
    return bool(
        isinstance(mutation.trigger, AtTrigger)
        and mutation.trigger.at == expected_at
        and not mutation.required_conditions
        and not mutation.blockers
        and mutation.action_template.kind == "reminder"
        and mutation.action_template.payload
        == {"subject": "perform compatibility check"}
    )


async def run_local_model_preflight(
    model: LocalInspectDecisionModel,
    *,
    residency_probe: Callable[[], LocalLoadedModelAttestation] = (
        probe_loaded_local_model
    ),
) -> LocalModelPreflightResult:
    """Make one compiler and one contemporaneous decision call, without retry."""

    compiler = LocalInspectMemoryCompiler(model)
    event = ObservableEvent(
        id="local-preflight-event",
        at=datetime.fromisoformat("2026-01-05T09:00:00+00:00"),
        kind="user_message",
        text="At 17:00 today remind me to perform compatibility check.",
    )
    compiler_call = await compiler.compile(
        CompilerRequest(event=event, active_state='{"facts":[],"intents":[]}')
    )
    decision_call = await model.decide(
        DecisionRequest(
            event=event,
            prompt=build_local_decision_prompt(
                now=event.at.isoformat(),
                current_event_id=event.id,
                context_events=[event],
                decision_history=[],
                memory_view=None,
            ),
        )
    )
    probe_started = perf_counter()
    try:
        loaded_model = await asyncio.to_thread(residency_probe)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError):
        loaded_model = None
    probe_latency_ms = max(0.0, (perf_counter() - probe_started) * 1000)
    compiler_semantic_valid = _compiler_preflight_semantics(compiler_call)
    decision_semantic_valid = (
        not decision_call.parse_error and not decision_call.decision.actions
    )
    same_model = compiler.name == model.name and compiler._model is model
    passed = all(
        (
            loaded_model is not None,
            same_model,
            not compiler_call.parse_error,
            not decision_call.parse_error,
            compiler_semantic_valid,
            decision_semantic_valid,
            compiler_call.usage_complete,
            decision_call.usage_complete,
            compiler_call.cost_complete,
            decision_call.cost_complete,
            compiler_call.usage.cost_usd == 0.0,
            decision_call.usage.cost_usd == 0.0,
        )
    )
    return LocalModelPreflightResult(
        model=LOCAL_OLLAMA_MODEL,
        runtime=model.runtime_attestation,
        loaded_model=loaded_model,
        same_model_for_compiler_and_decision=same_model,
        compiler_parse_error=compiler_call.parse_error,
        decision_parse_error=decision_call.parse_error,
        compiler_semantic_valid=compiler_semantic_valid,
        decision_semantic_valid=decision_semantic_valid,
        compiler_usage=compiler_call.usage,
        decision_usage=decision_call.usage,
        compiler_usage_complete=compiler_call.usage_complete,
        decision_usage_complete=decision_call.usage_complete,
        compiler_cost_complete=compiler_call.cost_complete,
        decision_cost_complete=decision_call.cost_complete,
        compiler_latency_ms=compiler_call.latency_ms,
        decision_latency_ms=decision_call.latency_ms,
        residency_probe_latency_ms=probe_latency_ms,
        passed=passed,
    )


def _validate_local_w2_preflight_fixture(
    fixture: Mapping[str, Any],
) -> None:
    if set(fixture) != {
        "schema_version",
        "purpose",
        "hypothesis_test_eligible",
        "contracts",
        "compiler_cases",
        "decision_cases",
    }:
        raise ValueError("W2 preflight fixture has an invalid top-level shape")
    if (
        fixture.get("schema_version") != 1
        or fixture.get("purpose") != "diagnostic"
        or fixture.get("hypothesis_test_eligible") is not False
    ):
        raise ValueError("W2 preflight fixture identity differs from the protocol")
    contracts = fixture.get("contracts")
    if not isinstance(contracts, dict):
        raise ValueError("W2 preflight fixture contracts are missing")
    expected_contracts = {
        "compiler": {
            "prompt_version": LOCAL_MEMORY_COMPILER_W2_VERSION,
            "prompt_sha256": hashlib.sha256(
                local_memory_compiler_w2_prompt_contract().encode()
            ).hexdigest(),
            "schema_sha256": hashlib.sha256(
                local_memory_compiler_schema_contract().encode()
            ).hexdigest(),
        },
        "decision": {
            "prompt_version": LOCAL_DECISION_VERSION,
            "prompt_sha256": hashlib.sha256(
                local_decision_prompt_contract().encode()
            ).hexdigest(),
            "schema_sha256": hashlib.sha256(
                local_decision_schema_contract().encode()
            ).hexdigest(),
        },
    }
    if contracts != expected_contracts:
        raise ValueError("W2 preflight fixture contract hashes differ from runtime")
    compiler_cases = fixture.get("compiler_cases")
    decision_cases = fixture.get("decision_cases")
    if not isinstance(compiler_cases, list) or not isinstance(decision_cases, list):
        raise ValueError("W2 preflight fixture cases are missing")
    compiler_identity = [
        (case.get("id"), case.get("category"))
        for case in compiler_cases
        if isinstance(case, dict)
    ]
    if compiler_identity != [
        ("C1", "trivial_explicit_same_day_at_subject_only"),
        ("C2", "trivial_explicit_next_day_at_address_only"),
        ("C3", "irrelevant_observation_empty_memory_delta"),
    ]:
        raise ValueError("W2 preflight compiler cases must be ordered C1,C2,C3")
    decision_identity = [
        (case.get("id"), case.get("category"))
        for case in decision_cases
        if isinstance(case, dict)
    ]
    if decision_identity != [
        ("D1", "structured_memory_empty_irrelevant_raw_event_no_action")
    ]:
        raise ValueError("W2 preflight decision cases must contain only D1")
    for case in [*compiler_cases, *decision_cases]:
        if not isinstance(case, dict) or not isinstance(case.get("input"), dict):
            raise ValueError("W2 preflight case input is invalid")
        if not isinstance(case.get("acceptance"), dict):
            raise ValueError("W2 preflight case acceptance is invalid")


def load_local_w2_preflight_fixture(
    path: str | Path,
) -> dict[str, Any]:
    """Load only the one content-addressed W2 synthetic preflight fixture."""

    fixture_path = Path(path)
    content = fixture_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256:
        raise ValueError("W2 preflight fixture hash differs from the frozen pin")
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("W2 preflight fixture must be a JSON object")
    _validate_local_w2_preflight_fixture(raw)
    return raw


def local_w2_preflight_prompts(
    fixture: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    """Render C1,C2,C3,D1 from fixture inputs, excluding all other fields."""

    _validate_local_w2_preflight_fixture(fixture)
    compiler_cases = fixture["compiler_cases"]
    decision_cases = fixture["decision_cases"]
    assert isinstance(compiler_cases, list)
    assert isinstance(decision_cases, list)
    prompts: list[str] = []
    for case in compiler_cases:
        assert isinstance(case, dict)
        case_input = case["input"]
        assert isinstance(case_input, dict)
        event = ObservableEvent.model_validate(case_input["event"])
        active_state = case_input.get("active_state")
        if not isinstance(active_state, str):
            raise ValueError("W2 compiler preflight active_state must be a string")
        prompts.append(
            build_local_memory_compiler_w2_prompt(
                event=event,
                active_state=active_state,
            )
        )
    decision_case = decision_cases[0]
    assert isinstance(decision_case, dict)
    decision_input = decision_case["input"]
    assert isinstance(decision_input, dict)
    raw_events = decision_input.get("context_events")
    if not isinstance(raw_events, list):
        raise ValueError("W2 decision preflight context_events must be a list")
    context_events = [ObservableEvent.model_validate(event) for event in raw_events]
    raw_history = decision_input.get("decision_history")
    if not isinstance(raw_history, list):
        raise ValueError("W2 decision preflight history must be a list")
    memory_view = MemoryView.model_validate(decision_input.get("memory_view"))
    prompts.append(
        build_local_decision_prompt(
            now=str(decision_input.get("now")),
            current_event_id=str(decision_input.get("current_event_id")),
            context_events=context_events,
            decision_history=raw_history,
            memory_view=memory_view,
        )
    )
    return tuple(prompts)  # type: ignore[return-value]


def _w2_compiler_semantic_valid(
    completion: str,
    case: Mapping[str, Any],
) -> tuple[bool, bool]:
    try:
        delta = LocalMemoryDeltaWire.model_validate_json(completion).to_domain()
    except (ValidationError, ValueError):
        return True, False
    acceptance = case.get("acceptance")
    if not isinstance(acceptance, dict):
        return False, False
    if acceptance.get("mutation_type") == "empty_delta":
        return False, not delta.mutations
    if acceptance.get("mutation_type") != "create_intent":
        return False, False
    if len(delta.mutations) != 1 or not isinstance(delta.mutations[0], CreateIntent):
        return False, False
    mutation = delta.mutations[0].model_dump(mode="json")
    action_template = mutation.get("action_template")
    return False, bool(
        mutation.get("op") == "create_intent"
        and mutation.get("trigger") == acceptance.get("trigger")
        and mutation.get("required_conditions") == acceptance.get("required_conditions")
        and mutation.get("blockers") == acceptance.get("blockers")
        and isinstance(mutation.get("intent_id"), str)
        and mutation.get("intent_id")
        and isinstance(action_template, dict)
        and action_template.get("kind") == acceptance.get("kind")
        and action_template.get("payload") == acceptance.get("payload")
        and isinstance(action_template.get("summary"), str)
        and action_template.get("summary")
    )


def _w2_decision_semantic_valid(completion: str) -> tuple[bool, bool]:
    try:
        wire = LocalDecisionWire.model_validate_json(completion)
        decision = wire.to_domain()
    except (ValidationError, ValueError):
        return True, False
    return False, wire.mode == "no_action" and not decision.actions


async def run_local_model_preflight_w2(
    model: LocalInspectDecisionModel,
    *,
    fixture: Mapping[str, Any],
    residency_probe: Callable[[], LocalLoadedModelAttestation] = (
        probe_loaded_local_model
    ),
) -> LocalModelPreflightW2Result:
    """Run frozen C1,C2,C3,D1 exactly once each, without retry or repair."""

    _validate_local_w2_preflight_fixture(fixture)
    compiler_cases = fixture["compiler_cases"]
    decision_cases = fixture["decision_cases"]
    assert isinstance(compiler_cases, list)
    assert isinstance(decision_cases, list)
    compiler = LocalInspectMemoryCompiler(
        model,
        prompt_builder=build_local_memory_compiler_w2_prompt,
    )
    results: list[LocalModelPreflightW2CaseResult] = []
    for case in compiler_cases:
        assert isinstance(case, dict)
        case_input = case["input"]
        assert isinstance(case_input, dict)
        event = ObservableEvent.model_validate(case_input["event"])
        active_state = case_input.get("active_state")
        if not isinstance(active_state, str):
            raise ValueError("W2 compiler preflight active_state must be a string")
        call = await compiler.compile(
            CompilerRequest(event=event, active_state=active_state)
        )
        parse_error, semantic_valid = _w2_compiler_semantic_valid(
            call.raw_completion,
            case,
        )
        if parse_error != call.parse_error:
            raise ValueError("W2 compiler parse accounting is inconsistent")
        results.append(
            LocalModelPreflightW2CaseResult(
                case_id=case["id"],
                role="compiler",
                parse_error=parse_error,
                semantic_valid=semantic_valid,
                usage=call.usage,
                usage_complete=call.usage_complete,
                cost_complete=call.cost_complete,
                latency_ms=call.latency_ms,
            )
        )

    decision_case = decision_cases[0]
    assert isinstance(decision_case, dict)
    decision_input = decision_case["input"]
    assert isinstance(decision_input, dict)
    raw_events = decision_input.get("context_events")
    raw_history = decision_input.get("decision_history")
    if not isinstance(raw_events, list) or not isinstance(raw_history, list):
        raise ValueError("W2 decision preflight input is invalid")
    context_events = [ObservableEvent.model_validate(event) for event in raw_events]
    memory_view = MemoryView.model_validate(decision_input.get("memory_view"))
    decision_call = await model.decide(
        DecisionRequest(
            event=context_events[-1],
            prompt=build_local_decision_prompt(
                now=str(decision_input.get("now")),
                current_event_id=str(decision_input.get("current_event_id")),
                context_events=context_events,
                decision_history=raw_history,
                memory_view=memory_view,
            ),
        )
    )
    parse_error, semantic_valid = _w2_decision_semantic_valid(
        decision_call.raw_completion
    )
    if parse_error != decision_call.parse_error:
        raise ValueError("W2 decision parse accounting is inconsistent")
    results.append(
        LocalModelPreflightW2CaseResult(
            case_id="D1",
            role="decision",
            parse_error=parse_error,
            semantic_valid=semantic_valid,
            usage=decision_call.usage,
            usage_complete=decision_call.usage_complete,
            cost_complete=decision_call.cost_complete,
            latency_ms=decision_call.latency_ms,
        )
    )

    probe_started = perf_counter()
    try:
        loaded_model = await asyncio.to_thread(residency_probe)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError):
        loaded_model = None
    probe_latency_ms = max(0.0, (perf_counter() - probe_started) * 1000)
    same_model = compiler.name == model.name and compiler._model is model
    passed = bool(
        loaded_model is not None
        and same_model
        and all(
            not result.parse_error
            and result.semantic_valid
            and result.usage_complete
            and result.cost_complete
            and result.usage.cost_usd == 0.0
            for result in results
        )
    )
    return LocalModelPreflightW2Result(
        model=LOCAL_OLLAMA_MODEL,
        runtime=model.runtime_attestation,
        loaded_model=loaded_model,
        same_model_for_compiler_and_decision=same_model,
        cases=results,
        residency_probe_latency_ms=probe_latency_ms,
        fixture_sha256=LOCAL_WRITER_W2_PREFLIGHT_FIXTURE_SHA256,
        passed=passed,
    )


_W3_COMPILER_CASE_IDENTITY = [
    ("C1", "normalization_fact"),
    ("C2", "bare_weekday_at"),
    ("C3", "condition_transition_and"),
    ("C4", "recurrence_iana_range"),
    ("C5", "stable_id_trigger_update"),
    ("C6", "full_action_template_update"),
    ("C7", "complete_sparse_payload_including_zero"),
    ("C8", "ambiguous_empty"),
]


def load_local_w3_preflight_protocol(path: str | Path) -> dict[str, Any]:
    """Load the exact frozen W3 protocol without accepting alternate bytes."""

    protocol_path = Path(path)
    content = protocol_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256:
        raise ValueError("W3 preflight protocol hash differs from the frozen pin")
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("W3 preflight protocol must be a JSON object")
    ordered = raw.get("preflight")
    ordered = ordered.get("ordered_categories") if isinstance(ordered, dict) else None
    if not isinstance(ordered, list):
        raise ValueError("W3 preflight protocol omits ordered categories")
    identity = [
        (case.get("id"), case.get("role"), case.get("category"))
        for case in ordered
        if isinstance(case, dict)
    ]
    expected = [
        (case_id, "compiler", category)
        for case_id, category in _W3_COMPILER_CASE_IDENTITY
    ]
    expected.append(("D1", "decision", "no_action"))
    if (
        raw.get("protocol_id") != "local_writer_w3.protocol.v1"
        or raw.get("phase") != "writer_diagnostic_w3"
        or raw.get("hypothesis_test_eligible") is not False
        or identity != expected
    ):
        raise ValueError("W3 preflight protocol identity differs from the pin")
    return raw


def _validate_local_w3_preflight_fixture(fixture: Mapping[str, Any]) -> None:
    if set(fixture) != {
        "schema_version",
        "fixture_id",
        "purpose",
        "hypothesis_test_eligible",
        "authorship",
        "contracts",
        "custodian_audit",
        "compiler_cases",
        "decision_cases",
    }:
        raise ValueError("W3 preflight fixture has an invalid top-level shape")
    if (
        fixture.get("schema_version") != 1
        or fixture.get("fixture_id") != "local_writer_w3.v1"
        or fixture.get("purpose") != "diagnostic"
        or fixture.get("hypothesis_test_eligible") is not False
    ):
        raise ValueError("W3 preflight fixture identity differs from the protocol")
    authorship = fixture.get("authorship")
    if (
        not isinstance(authorship, dict)
        or authorship.get("protocol_id") != "local_writer_w3.protocol.v1"
        or authorship.get("protocol_sha256")
        != LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256
        or authorship.get("model_boundary") != "case_input_only"
    ):
        raise ValueError("W3 preflight fixture authorship binding is invalid")
    custodian = fixture.get("custodian_audit")
    dataset = custodian.get("dataset") if isinstance(custodian, dict) else None
    oracle = custodian.get("oracle") if isinstance(custodian, dict) else None
    if (
        not isinstance(custodian, dict)
        or custodian.get("status") != "passed"
        or not isinstance(dataset, dict)
        or dataset.get("raw_sha256") != LOCAL_WRITER_W3_DATASET_SHA256
        or not isinstance(oracle, dict)
        or oracle.get("raw_sha256") != LOCAL_WRITER_W3_REFERENCE_SHA256
        or custodian.get("model_boundary_unchanged") is not True
    ):
        raise ValueError("W3 preflight fixture custodian binding is invalid")
    contracts = fixture.get("contracts")
    if not isinstance(contracts, dict):
        raise ValueError("W3 preflight fixture contracts are missing")
    expected_contracts = {
        "compiler": {
            "prompt_version": LOCAL_MEMORY_COMPILER_W3_VERSION,
            "addendum_sha256": hashlib.sha256(
                LOCAL_MEMORY_COMPILER_W3_ADDENDUM.encode()
            ).hexdigest(),
            "prompt_sha256": hashlib.sha256(
                local_memory_compiler_w3_prompt_contract().encode()
            ).hexdigest(),
            "local_wire_contract_sha256": hashlib.sha256(
                local_memory_compiler_w3_contract().encode()
            ).hexdigest(),
            "local_wire_model_schema_sha256": (
                "f0e0ab9c3aef10f9b99ca5055d1ee1f2e6d7f091be666ee95035040e564302ec"
            ),
            "inspect_response_schema_sha256": hashlib.sha256(
                local_memory_compiler_schema_contract().encode()
            ).hexdigest(),
            "inspect_response_schema_unchanged_from_w2": True,
        },
        "decision": {
            "prompt_version": LOCAL_DECISION_VERSION,
            "prompt_sha256": hashlib.sha256(
                local_decision_prompt_contract().encode()
            ).hexdigest(),
            "schema_sha256": hashlib.sha256(
                local_decision_schema_contract().encode()
            ).hexdigest(),
            "unchanged_from_w2": True,
        },
    }
    if contracts != expected_contracts:
        raise ValueError("W3 preflight fixture contract hashes differ from runtime")
    compiler_cases = fixture.get("compiler_cases")
    decision_cases = fixture.get("decision_cases")
    if not isinstance(compiler_cases, list) or not isinstance(decision_cases, list):
        raise ValueError("W3 preflight fixture cases are missing")
    compiler_identity = [
        (case.get("id"), case.get("category"))
        for case in compiler_cases
        if isinstance(case, dict)
    ]
    decision_identity = [
        (case.get("id"), case.get("category"))
        for case in decision_cases
        if isinstance(case, dict)
    ]
    if compiler_identity != _W3_COMPILER_CASE_IDENTITY:
        raise ValueError("W3 preflight compiler cases must be ordered C1-C8")
    if decision_identity != [("D1", "no_action")]:
        raise ValueError("W3 preflight decision cases must contain only D1")
    for case in [*compiler_cases, *decision_cases]:
        if not isinstance(case, dict) or not isinstance(case.get("input"), dict):
            raise ValueError("W3 preflight case input is invalid")
        if not isinstance(case.get("acceptance"), dict):
            raise ValueError("W3 preflight case acceptance is invalid")


def load_local_w3_preflight_fixture(path: str | Path) -> dict[str, Any]:
    """Load only the content-addressed W3 synthetic preflight fixture."""

    fixture_path = Path(path)
    content = fixture_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256:
        raise ValueError("W3 preflight fixture hash differs from the frozen pin")
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("W3 preflight fixture must be a JSON object")
    _validate_local_w3_preflight_fixture(raw)
    return raw


def local_w3_preflight_prompts(fixture: Mapping[str, Any]) -> tuple[str, ...]:
    """Render C1-C8,D1 from input fields only, never fixture acceptance data."""

    _validate_local_w3_preflight_fixture(fixture)
    compiler_cases = fixture["compiler_cases"]
    decision_cases = fixture["decision_cases"]
    assert isinstance(compiler_cases, list)
    assert isinstance(decision_cases, list)
    prompts: list[str] = []
    for case in compiler_cases:
        assert isinstance(case, dict)
        case_input = case["input"]
        assert isinstance(case_input, dict)
        event = ObservableEvent.model_validate(case_input["event"])
        active_state = case_input.get("active_state")
        if not isinstance(active_state, str):
            raise ValueError("W3 compiler preflight active_state must be a string")
        prompts.append(
            build_local_memory_compiler_w3_prompt(
                event=event,
                active_state=active_state,
            )
        )
    decision_case = decision_cases[0]
    assert isinstance(decision_case, dict)
    decision_input = decision_case["input"]
    assert isinstance(decision_input, dict)
    raw_events = decision_input.get("context_events")
    raw_history = decision_input.get("decision_history")
    if not isinstance(raw_events, list) or not isinstance(raw_history, list):
        raise ValueError("W3 decision preflight input is invalid")
    prompts.append(
        build_local_decision_prompt(
            now=str(decision_input.get("now")),
            current_event_id=str(decision_input.get("current_event_id")),
            context_events=[
                ObservableEvent.model_validate(event) for event in raw_events
            ],
            decision_history=raw_history,
            memory_view=MemoryView.model_validate(decision_input.get("memory_view")),
        )
    )
    return tuple(prompts)


def _same_json(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
    )


def _w3_create_semantic_valid(
    mutation: CreateIntent,
    acceptance: Mapping[str, Any],
) -> bool:
    raw = mutation.model_dump(mode="json")
    actual_trigger = raw.get("trigger")
    expected_trigger = acceptance.get("trigger")
    if not isinstance(actual_trigger, dict) or not isinstance(expected_trigger, dict):
        return False
    if acceptance.get("weekdays_match") == "set":
        actual_weekdays = actual_trigger.pop("weekdays", None)
        expected_weekdays = expected_trigger.get("weekdays")
        expected_trigger = dict(expected_trigger)
        expected_trigger.pop("weekdays", None)
        if (
            not isinstance(actual_weekdays, list)
            or not isinstance(expected_weekdays, list)
            or set(actual_weekdays) != set(expected_weekdays)
            or len(actual_weekdays) != len(set(actual_weekdays))
        ):
            return False
    if not _same_json(actual_trigger, expected_trigger):
        return False
    actual_conditions = raw.get("required_conditions")
    expected_conditions = acceptance.get("required_conditions")
    if acceptance.get("required_conditions_match") == "canonical_multiset":
        if not isinstance(actual_conditions, list) or not isinstance(
            expected_conditions, list
        ):
            return False
        actual_conditions = sorted(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in actual_conditions
        )
        expected_conditions = sorted(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in expected_conditions
        )
    action_template = raw.get("action_template")
    expected_action = acceptance.get("action_template")
    return bool(
        raw.get("op") == "create_intent"
        and isinstance(raw.get("intent_id"), str)
        and raw.get("intent_id")
        and _same_json(actual_conditions, expected_conditions)
        and _same_json(raw.get("blockers"), acceptance.get("blockers"))
        and isinstance(action_template, dict)
        and isinstance(expected_action, dict)
        and action_template.get("kind") == expected_action.get("kind")
        and _same_json(
            action_template.get("payload"),
            expected_action.get("payload"),
        )
        and isinstance(action_template.get("summary"), str)
        and action_template.get("summary")
    )


def _w3_compiler_semantic_valid(
    completion: str,
    case: Mapping[str, Any],
) -> tuple[bool, bool]:
    try:
        delta = LocalMemoryDeltaWire.model_validate_json(completion).to_domain()
    except (ValidationError, ValueError):
        return True, False
    acceptance = case.get("acceptance")
    if not isinstance(acceptance, dict):
        return False, False
    mutation_type = acceptance.get("mutation_type")
    if mutation_type == "empty_delta":
        return False, not delta.mutations
    if len(delta.mutations) != acceptance.get("mutation_count"):
        return False, False
    mutation = delta.mutations[0]
    if mutation_type == "set_fact":
        if not isinstance(mutation, SetFact):
            return False, False
        raw = mutation.model_dump(mode="json")
        return False, bool(
            raw.get("op") == "set_fact"
            and _same_json(raw.get("key"), acceptance.get("key"))
            and _same_json(raw.get("value"), acceptance.get("value"))
            and _same_json(raw.get("unit"), acceptance.get("unit"))
        )
    if mutation_type == "create_intent":
        return False, isinstance(mutation, CreateIntent) and (
            _w3_create_semantic_valid(mutation, acceptance)
        )
    if mutation_type == "update_intent":
        if not isinstance(mutation, UpdateIntent):
            return False, False
        raw = mutation.model_dump(mode="json", exclude_none=True)
        changed_fields = acceptance.get("changed_fields")
        if not isinstance(changed_fields, list):
            return False, False
        expected_keys = {"op", "intent_id", *changed_fields}
        return False, bool(
            set(raw) == expected_keys
            and raw.get("op") == "update_intent"
            and raw.get("intent_id") == acceptance.get("intent_id")
            and all(
                _same_json(raw.get(field), acceptance.get(field))
                for field in changed_fields
            )
        )
    return False, False


async def run_local_model_preflight_w3(
    model: LocalInspectDecisionModel,
    *,
    fixture: Mapping[str, Any],
    residency_probe: Callable[[], LocalLoadedModelAttestation] | None = None,
) -> LocalModelPreflightW3Result:
    """Run frozen C1-C8,D1 exactly once each, without retry or repair."""

    _validate_local_w3_preflight_fixture(fixture)
    compiler_cases = fixture["compiler_cases"]
    decision_cases = fixture["decision_cases"]
    assert isinstance(compiler_cases, list)
    assert isinstance(decision_cases, list)
    compiler = LocalInspectMemoryCompiler(
        model,
        prompt_builder=build_local_memory_compiler_w3_prompt,
    )
    results: list[LocalModelPreflightW3CaseResult] = []
    for case in compiler_cases:
        assert isinstance(case, dict)
        case_input = case["input"]
        assert isinstance(case_input, dict)
        event = ObservableEvent.model_validate(case_input["event"])
        active_state = case_input.get("active_state")
        if not isinstance(active_state, str):
            raise ValueError("W3 compiler preflight active_state must be a string")
        call = await compiler.compile(
            CompilerRequest(event=event, active_state=active_state)
        )
        parse_error, semantic_valid = _w3_compiler_semantic_valid(
            call.raw_completion,
            case,
        )
        if parse_error != call.parse_error:
            raise ValueError("W3 compiler parse accounting is inconsistent")
        results.append(
            LocalModelPreflightW3CaseResult(
                case_id=case["id"],
                role="compiler",
                parse_error=parse_error,
                semantic_valid=semantic_valid,
                usage=call.usage,
                usage_complete=call.usage_complete,
                cost_complete=call.cost_complete,
                latency_ms=call.latency_ms,
            )
        )

    decision_case = decision_cases[0]
    assert isinstance(decision_case, dict)
    decision_input = decision_case["input"]
    assert isinstance(decision_input, dict)
    raw_events = decision_input.get("context_events")
    raw_history = decision_input.get("decision_history")
    if not isinstance(raw_events, list) or not isinstance(raw_history, list):
        raise ValueError("W3 decision preflight input is invalid")
    context_events = [ObservableEvent.model_validate(event) for event in raw_events]
    decision_call = await model.decide(
        DecisionRequest(
            event=context_events[-1],
            prompt=build_local_decision_prompt(
                now=str(decision_input.get("now")),
                current_event_id=str(decision_input.get("current_event_id")),
                context_events=context_events,
                decision_history=raw_history,
                memory_view=MemoryView.model_validate(
                    decision_input.get("memory_view")
                ),
            ),
        )
    )
    parse_error, semantic_valid = _w2_decision_semantic_valid(
        decision_call.raw_completion
    )
    if parse_error != decision_call.parse_error:
        raise ValueError("W3 decision parse accounting is inconsistent")
    results.append(
        LocalModelPreflightW3CaseResult(
            case_id="D1",
            role="decision",
            parse_error=parse_error,
            semantic_valid=semantic_valid,
            usage=decision_call.usage,
            usage_complete=decision_call.usage_complete,
            cost_complete=decision_call.cost_complete,
            latency_ms=decision_call.latency_ms,
        )
    )

    active_residency_probe = residency_probe or (
        lambda: probe_loaded_local_model(model.name)
    )
    probe_started = perf_counter()
    try:
        loaded_model = await asyncio.to_thread(active_residency_probe)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError):
        loaded_model = None
    probe_latency_ms = max(0.0, (perf_counter() - probe_started) * 1000)
    same_model = compiler.name == model.name and compiler._model is model
    passed = bool(
        loaded_model is not None
        and same_model
        and all(
            not result.parse_error
            and result.semantic_valid
            and result.usage_complete
            and result.cost_complete
            and result.usage.cost_usd == 0.0
            for result in results
        )
    )
    return LocalModelPreflightW3Result(
        model=model.name,
        runtime=model.runtime_attestation,
        loaded_model=loaded_model,
        same_model_for_compiler_and_decision=same_model,
        cases=results,
        residency_probe_latency_ms=probe_latency_ms,
        fixture_sha256=LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256,
        protocol_sha256=LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
        passed=passed,
    )


class _TaskLocalPreflight:
    """Concurrency-safe live gate performed before any scenario model call."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._result: LocalModelPreflightResult | None = None

    async def ensure(
        self,
        model: LocalInspectDecisionModel,
    ) -> tuple[LocalModelPreflightResult, bool]:
        if self._result is not None:
            return self._result, False
        async with self._lock:
            if self._result is None:
                result = await run_local_model_preflight(model)
                if not result.passed:
                    raise ValueError("live local semantic preflight failed")
                self._result = result
                return result, True
        return self._result, False


class _TaskLocalW2Preflight:
    """Concurrency-safe four-call W2 gate run once per scenario task."""

    def __init__(self, fixture: Mapping[str, Any]) -> None:
        _validate_local_w2_preflight_fixture(fixture)
        self._fixture = fixture
        self._lock = asyncio.Lock()
        self._result: LocalModelPreflightW2Result | None = None

    async def ensure(
        self,
        model: LocalInspectDecisionModel,
    ) -> tuple[LocalModelPreflightW2Result, bool]:
        if self._result is not None:
            return self._result, False
        async with self._lock:
            if self._result is None:
                result = await run_local_model_preflight_w2(
                    model,
                    fixture=self._fixture,
                )
                if not result.passed:
                    raise ValueError("live local W2 semantic preflight failed")
                self._result = result
                return result, True
        return self._result, False


class _TaskLocalW3Preflight:
    """Concurrency-safe nine-call W3 gate run once per scenario task."""

    def __init__(self, fixture: Mapping[str, Any]) -> None:
        _validate_local_w3_preflight_fixture(fixture)
        self._fixture = fixture
        self._lock = asyncio.Lock()
        self._result: LocalModelPreflightW3Result | None = None

    async def ensure(
        self,
        model: LocalInspectDecisionModel,
    ) -> tuple[LocalModelPreflightW3Result, bool]:
        if self._result is not None:
            return self._result, False
        async with self._lock:
            if self._result is None:
                result = await run_local_model_preflight_w3(
                    model,
                    fixture=self._fixture,
                )
                if not result.passed:
                    raise ValueError("live local W3 semantic preflight failed")
                self._result = result
                return result, True
        return self._result, False


def local_model_preflight_sample() -> Sample:
    """Return one synthetic local compatibility case, never benchmark data."""

    return Sample(
        id="local-model-preflight-v0",
        input="Check local structured output, semantics, residency and accounting.",
        target="pass",
    )


def local_model_preflight_w2_sample() -> Sample:
    """Return one synthetic four-call W2 compatibility sample."""

    return Sample(
        id=LOCAL_MODEL_PREFLIGHT_W2_SAMPLE_ID,
        input="Check the frozen local W2 compiler and decision protocol.",
        target="pass",
    )


def local_model_preflight_w3_sample() -> Sample:
    """Return one synthetic nine-call W3 compatibility sample."""

    return Sample(
        id=LOCAL_MODEL_PREFLIGHT_W3_SAMPLE_ID,
        input="Check the frozen local W3 compiler and decision protocol.",
        target="pass",
    )


def local_model_preflight_w3_m2_sample() -> Sample:
    """Return the model-only W3 compatibility sample, never scenario data."""

    return Sample(
        id=LOCAL_MODEL_PREFLIGHT_W3_M2_SAMPLE_ID,
        input="Check W3 with the separately pinned M2 local model.",
        target="pass",
    )


def local_model_preflight_w3_m2_t1_sample() -> Sample:
    """Return the transport-only M2 compatibility sample, never scenario data."""

    return Sample(
        id=LOCAL_MODEL_PREFLIGHT_W3_M2_T1_SAMPLE_ID,
        input="Check W3-M2 with the frozen no-thinking transport intervention.",
        target="pass",
    )


@solver
def local_model_preflight_solver() -> Solver:
    """Standalone diagnostic gate that exposes failure instead of raising."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = LocalInspectDecisionModel(state, generate)
        result = await run_local_model_preflight(model)
        state = model.state
        serialized = result.model_dump(mode="json")
        state.metadata[LOCAL_PREFLIGHT_METADATA_KEY] = serialized
        state.store.set(LOCAL_PREFLIGHT_STORE_KEY, serialized)
        state.output = ModelOutput.from_content(
            model=model.name,
            content=result.model_dump_json(),
        )
        return state

    return solve


@solver
def local_model_preflight_w2_solver(
    fixture_path: str,
) -> Solver:
    """Standalone frozen W2 gate that exposes failure without repair."""

    fixture = load_local_w2_preflight_fixture(fixture_path)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = LocalInspectDecisionModel(state, generate)
        result = await run_local_model_preflight_w2(model, fixture=fixture)
        state = model.state
        serialized = result.model_dump(mode="json")
        state.metadata[LOCAL_PREFLIGHT_W2_METADATA_KEY] = serialized
        state.store.set(LOCAL_PREFLIGHT_W2_STORE_KEY, serialized)
        state.output = ModelOutput.from_content(
            model=model.name,
            content=result.model_dump_json(),
        )
        return state

    return solve


@solver
def local_model_preflight_w3_solver(fixture_path: str) -> Solver:
    """Standalone frozen W3 gate that exposes failure without repair."""

    fixture = load_local_w3_preflight_fixture(fixture_path)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = LocalInspectDecisionModel(state, generate)
        result = await run_local_model_preflight_w3(model, fixture=fixture)
        state = model.state
        serialized = result.model_dump(mode="json")
        state.metadata[LOCAL_PREFLIGHT_W3_METADATA_KEY] = serialized
        state.store.set(LOCAL_PREFLIGHT_W3_STORE_KEY, serialized)
        state.output = ModelOutput.from_content(
            model=model.name,
            content=result.model_dump_json(),
        )
        return state

    return solve


@solver
def local_model_preflight_w3_m2_solver(fixture_path: str) -> Solver:
    """Standalone W3-M2 gate using the unchanged W3 fixture and prompts."""

    fixture = load_local_w3_preflight_fixture(fixture_path)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if str(state.model) != LOCAL_W3_M2_OLLAMA_MODEL:
            raise ValueError("W3-M2 preflight requires the pinned M2 model")
        model = LocalInspectDecisionModel(state, generate)
        result = await run_local_model_preflight_w3(model, fixture=fixture)
        state = model.state
        serialized = result.model_dump(mode="json")
        state.metadata[LOCAL_PREFLIGHT_W3_M2_METADATA_KEY] = serialized
        state.store.set(LOCAL_PREFLIGHT_W3_M2_STORE_KEY, serialized)
        state.output = ModelOutput.from_content(
            model=model.name,
            content=result.model_dump_json(),
        )
        return state

    return solve


@solver
def local_model_preflight_w3_m2_t1_solver(fixture_path: str) -> Solver:
    """Run the frozen W3 gate under the T1 transport; expose failure exactly."""

    fixture = load_local_w3_preflight_fixture(fixture_path)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if str(state.model) != LOCAL_W3_M2_OLLAMA_MODEL:
            raise ValueError("W3-M2-T1 preflight requires the pinned M2 model")
        model = LocalInspectDecisionModel(state, generate)
        result = await run_local_model_preflight_w3(model, fixture=fixture)
        state = model.state
        serialized = result.model_dump(mode="json")
        state.metadata[LOCAL_PREFLIGHT_W3_M2_T1_METADATA_KEY] = serialized
        state.store.set(LOCAL_PREFLIGHT_W3_M2_T1_STORE_KEY, serialized)
        state.output = ModelOutput.from_content(
            model=model.name,
            content=result.model_dump_json(),
        )
        return state

    return solve


@scorer(metrics=[])
def local_model_preflight_scorer() -> Scorer:
    """Score local compatibility only, never reminder quality."""

    async def score(state: TaskState, target: Target) -> Score:
        result = LocalModelPreflightResult.model_validate_json(state.output.completion)
        return Score(
            value=1 if result.passed else 0,
            answer=result.model,
            explanation=(
                "local compiler semantics, no-action decision, residency and "
                "zero-cost accounting passed"
                if result.passed
                else "local model failed one or more semantic preflight checks"
            ),
        )

    return score


@scorer(metrics=[])
def local_model_preflight_w2_scorer() -> Scorer:
    """Score only frozen W2 compatibility, never scenario quality."""

    async def score(state: TaskState, target: Target) -> Score:
        result = LocalModelPreflightW2Result.model_validate_json(
            state.output.completion
        )
        return Score(
            value=1 if result.passed else 0,
            answer=result.model,
            explanation=(
                "local W2 C1,C2,C3,D1 semantics, residency and zero-cost "
                "accounting passed"
                if result.passed
                else "local model failed one or more frozen W2 preflight checks"
            ),
        )

    return score


@scorer(metrics=[])
def local_model_preflight_w3_scorer() -> Scorer:
    """Score only frozen W3 compatibility, never scenario quality."""

    async def score(state: TaskState, target: Target) -> Score:
        result = LocalModelPreflightW3Result.model_validate_json(
            state.output.completion
        )
        return Score(
            value=1 if result.passed else 0,
            answer=result.model,
            explanation=(
                "local W3 C1-C8,D1 semantics, residency and zero-cost accounting passed"
                if result.passed
                else "local model failed one or more frozen W3 preflight checks"
            ),
        )

    return score


@scorer(metrics=[])
def local_model_preflight_w3_m2_scorer() -> Scorer:
    """Score only W3 compatibility for the frozen M2 model-only cell."""

    async def score(state: TaskState, target: Target) -> Score:
        result = LocalModelPreflightW3Result.model_validate_json(
            state.output.completion
        )
        if result.model != LOCAL_W3_M2_OLLAMA_MODEL:
            raise ValueError("W3-M2 scorer received a different model")
        return Score(
            value=1 if result.passed else 0,
            answer=result.model,
            explanation=(
                "W3 C1-C8,D1 passed with the frozen M2 local model"
                if result.passed
                else "the frozen M2 model failed one or more W3 checks"
            ),
        )

    return score


@scorer(metrics=[])
def local_model_preflight_w3_m2_t1_scorer() -> Scorer:
    """Score only the frozen W3 semantics under the T1 transport cell."""

    async def score(state: TaskState, target: Target) -> Score:
        result = LocalModelPreflightW3Result.model_validate_json(
            state.output.completion
        )
        if result.model != LOCAL_W3_M2_OLLAMA_MODEL:
            raise ValueError("W3-M2-T1 scorer received a different model")
        return Score(
            value=1 if result.passed else 0,
            answer=result.model,
            explanation=(
                "W3 C1-C8,D1 passed with the frozen no-thinking transport"
                if result.passed
                else "the frozen T1 transport failed one or more W3 checks"
            ),
        )

    return score


def local_system_config_sha256(
    *,
    system: LocalSystemName,
    top_k: int = 5,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_repository: str = "qdrant/bge-small-en-v1.5-onnx-q",
    embedding_revision: str | None = None,
    embedding_artifact_sha256: str | None = None,
    pricing_config_sha256: str | None = None,
    oracle_annotations_sha256: str | None = None,
    compiler_prompt_variant: Literal["w1", "w2", "w3"] = "w1",
    w3_preflight_fixture_sha256: str | None = None,
    w3_preflight_protocol_sha256: str | None = None,
    w3_dataset_sha256: str | None = None,
    w3_reference_sha256: str | None = None,
) -> str:
    """Fingerprint all local knobs that can change a diagnostic result."""

    payload: dict[str, object] = {
        "runtime_version": LOCAL_RUNTIME_VERSION,
        "model": LOCAL_OLLAMA_MODEL,
        "model_manifest_sha256": LOCAL_OLLAMA_MANIFEST_SHA256,
        "model_blob_sha256": LOCAL_OLLAMA_MODEL_BLOB_SHA256,
        "ollama_version": LOCAL_OLLAMA_VERSION,
        "model_family": LOCAL_OLLAMA_FAMILY,
        "model_parameter_size": LOCAL_OLLAMA_PARAMETER_SIZE,
        "model_quantization": LOCAL_OLLAMA_QUANTIZATION,
        "base_url": LOCAL_OLLAMA_BASE_URL,
        "no_cloud": "1",
        "server_bind": "127.0.0.1:11434",
        "context_length": LOCAL_OLLAMA_CONTEXT_LENGTH,
        "num_parallel": 1,
        "max_loaded_models": 1,
        "temperature": 0.0,
        "system": system,
        "decision_contract_sha256": hashlib.sha256(
            local_decision_contract().encode()
        ).hexdigest(),
        "pricing_config_sha256": pricing_config_sha256,
    }
    if system == "vector_rag":
        payload.update(
            top_k=top_k,
            embedding_model=embedding_model,
            embedding_repository=embedding_repository,
            embedding_revision=embedding_revision,
            embedding_artifact_sha256=embedding_artifact_sha256,
        )
    if system in {"anamnesis", ORACLE_SYSTEM_NAME}:
        payload["deterministic_memory"] = anamnesis_runtime_contract()
    if system == "anamnesis":
        compiler_contract = {
            "w1": local_memory_compiler_transport_contract,
            "w2": local_memory_compiler_w2_transport_contract,
            "w3": local_memory_compiler_w3_transport_contract,
        }[compiler_prompt_variant]()
        payload["memory_compiler_sha256"] = hashlib.sha256(
            compiler_contract.encode()
        ).hexdigest()
        w3_pins = {
            "preflight_fixture_sha256": w3_preflight_fixture_sha256,
            "preflight_protocol_sha256": w3_preflight_protocol_sha256,
            "dataset_sha256": w3_dataset_sha256,
            "reference_sha256": w3_reference_sha256,
        }
        if compiler_prompt_variant == "w3":
            expected_w3_pins = {
                "preflight_fixture_sha256": (LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256),
                "preflight_protocol_sha256": (
                    LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256
                ),
                "dataset_sha256": LOCAL_WRITER_W3_DATASET_SHA256,
                "reference_sha256": LOCAL_WRITER_W3_REFERENCE_SHA256,
            }
            if w3_pins != expected_w3_pins:
                raise ValueError("W3 system hash requires all exact frozen W3 pins")
            payload["writer_w3_protocol"] = {
                **w3_pins,
                "intervention": "bundled-repair",
                "setup_policy": "frozen_w3_semantic_gate_c1_to_c8_d1",
                "scenario_compiler_calls": 39,
                "scenario_checkpoints": 62,
            }
        elif any(value is not None for value in w3_pins.values()):
            raise ValueError("W3 pins require compiler_prompt_variant=w3")
    elif compiler_prompt_variant != "w1":
        raise ValueError("compiler_prompt_variant=w2/w3 requires system=anamnesis")
    elif any(
        value is not None
        for value in (
            w3_preflight_fixture_sha256,
            w3_preflight_protocol_sha256,
            w3_dataset_sha256,
            w3_reference_sha256,
        )
    ):
        raise ValueError("W3 pins require system=anamnesis")
    if system == ORACLE_SYSTEM_NAME:
        if oracle_annotations_sha256 is None or (
            len(oracle_annotations_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in oracle_annotations_sha256
            )
        ):
            raise ValueError(
                "oracle system hash requires a valid oracle_annotations_sha256"
            )
        payload.update(
            compiler_mode="oracle",
            oracle_compiler_version=ORACLE_COMPILER_VERSION,
            oracle_annotations_sha256=oracle_annotations_sha256,
            same_model_for_compiler_and_decision=False,
            shared_decision_contract_version=LOCAL_DECISION_VERSION,
        )
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _validate_oracle_scenario_run(
    scenario: RuntimeScenario,
    run: ScenarioRun,
) -> None:
    """Fail closed unless oracle compilation stayed accepted and token-free."""

    if run.system != ORACLE_SYSTEM_NAME:
        raise ValueError("oracle runtime produced the wrong system identity")
    if len(run.checkpoints) != len(scenario.events):
        raise ValueError("oracle runtime checkpoint coverage differs from scenario")
    if run.compiler_parse_errors:
        raise ValueError("oracle runtime recorded a compiler parse error")
    if not run.usage_complete or not run.cost_complete:
        raise ValueError("oracle runtime usage and cost must be complete")
    zero_usage = Usage(cost_usd=0.0)
    if run.compiler_usage != zero_usage:
        raise ValueError("oracle compiler usage must be exactly zero and complete")
    if run.usage != run.decision_usage:
        raise ValueError("oracle total usage must equal decision-only usage")
    for event, checkpoint in zip(scenario.events, run.checkpoints, strict=True):
        expected_call = event.kind != "clock_tick"
        if checkpoint.compiler_called is not expected_call:
            raise ValueError("oracle compiler call coverage differs from scenario")
        if not expected_call:
            continue
        if checkpoint.compiler_parse_error:
            raise ValueError("oracle checkpoint recorded a compiler parse error")
        if checkpoint.memory_delta_accepted is not True:
            raise ValueError("oracle memory delta was not accepted")
        if checkpoint.compiler_usage != zero_usage:
            raise ValueError("oracle checkpoint compiler usage must be exactly zero")


@solver
def local_scenario_solver(
    system: LocalSystemName,
    *,
    repetition: int = 1,
    seed: int | None = None,
    top_k: int = 5,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_repository: str = "qdrant/bge-small-en-v1.5-onnx-q",
    embedding_revision: str | None = None,
    embedding_snapshot_path: str | None = None,
    expected_model: str = LOCAL_OLLAMA_MODEL,
    expected_system_config_sha256: str | None = None,
    expected_embedding_artifact_sha256: str | None = None,
    manifest_sha256: str | None = None,
    pricing_config_sha256: str | None = None,
    oracle_annotations_path: str | None = None,
    oracle_annotations_sha256: str | None = None,
    compiler_prompt_variant: Literal["w1", "w2", "w3"] = "w1",
    w2_preflight_fixture_path: str | None = None,
    w3_preflight_fixture_path: str | None = None,
    w3_preflight_protocol_sha256: str | None = None,
    w3_dataset_sha256: str | None = None,
    w3_reference_sha256: str | None = None,
) -> Solver:
    """Run one local diagnostic system after a live semantic preflight."""

    if repetition < 1:
        raise ValueError("repetition must be at least 1")
    if expected_model != LOCAL_OLLAMA_MODEL:
        raise ValueError("local solver expected_model must be the pinned Ollama tag")
    if system == ORACLE_SYSTEM_NAME:
        if oracle_annotations_path is None or oracle_annotations_sha256 is None:
            raise ValueError("local oracle solver requires a pinned oracle artifact")
        oracle_path = Path(oracle_annotations_path)
        if not oracle_path.is_absolute() or not oracle_path.is_file():
            raise ValueError("local oracle solver requires an absolute artifact path")
    elif oracle_annotations_path is not None or oracle_annotations_sha256 is not None:
        raise ValueError("oracle inputs are only valid for the oracle system")
    if compiler_prompt_variant == "w2":
        if system != "anamnesis":
            raise ValueError("W2 compiler prompt requires system=anamnesis")
        if w2_preflight_fixture_path is None:
            raise ValueError("W2 compiler prompt requires its frozen fixture")
        fixture_path = Path(w2_preflight_fixture_path)
        if not fixture_path.is_absolute() or not fixture_path.is_file():
            raise ValueError("W2 preflight fixture path must be an absolute file")
        w2_fixture = load_local_w2_preflight_fixture(fixture_path)
        if any(
            value is not None
            for value in (
                w3_preflight_fixture_path,
                w3_preflight_protocol_sha256,
                w3_dataset_sha256,
                w3_reference_sha256,
            )
        ):
            raise ValueError("W3 inputs are invalid for the W2 compiler")
        w3_fixture = None
    elif compiler_prompt_variant == "w3":
        if system != "anamnesis":
            raise ValueError("W3 compiler prompt requires system=anamnesis")
        if w2_preflight_fixture_path is not None:
            raise ValueError("W2 preflight fixture is invalid for the W3 compiler")
        if w3_preflight_fixture_path is None:
            raise ValueError("W3 compiler prompt requires its frozen fixture")
        fixture_path = Path(w3_preflight_fixture_path)
        if not fixture_path.is_absolute() or not fixture_path.is_file():
            raise ValueError("W3 preflight fixture path must be an absolute file")
        w3_fixture = load_local_w3_preflight_fixture(fixture_path)
        if {
            "protocol": w3_preflight_protocol_sha256,
            "dataset": w3_dataset_sha256,
            "reference": w3_reference_sha256,
        } != {
            "protocol": LOCAL_WRITER_W3_PREFLIGHT_PROTOCOL_SHA256,
            "dataset": LOCAL_WRITER_W3_DATASET_SHA256,
            "reference": LOCAL_WRITER_W3_REFERENCE_SHA256,
        }:
            raise ValueError("W3 compiler prompt requires all exact frozen W3 pins")
        w2_fixture = None
    else:
        if (
            w2_preflight_fixture_path is not None
            or w3_preflight_fixture_path is not None
        ):
            raise ValueError("W2/W3 preflight fixture is invalid for the W1 compiler")
        if any(
            value is not None
            for value in (
                w3_preflight_protocol_sha256,
                w3_dataset_sha256,
                w3_reference_sha256,
            )
        ):
            raise ValueError("W3 pins are invalid for the W1 compiler")
        w2_fixture = None
        w3_fixture = None

    shared_vectorizer: Vectorizer | None = None
    if system == "vector_rag":
        if embedding_revision is None:
            raise ValueError("local vector_rag requires an exact embedding revision")
        shared_vectorizer = FastEmbedVectorizer(
            model_name=embedding_model,
            repository=embedding_repository,
            revision=embedding_revision,
            snapshot_path=embedding_snapshot_path,
        )
    task_preflight = _TaskLocalPreflight()
    task_w2_preflight = (
        _TaskLocalW2Preflight(w2_fixture) if w2_fixture is not None else None
    )
    task_w3_preflight = (
        _TaskLocalW3Preflight(w3_fixture) if w3_fixture is not None else None
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        decision_model = LocalInspectDecisionModel(state, generate)
        if decision_model.name != expected_model:
            raise ValueError("active model differs from the frozen local manifest")
        if task_w3_preflight is not None:
            preflight, performed_here = await task_w3_preflight.ensure(decision_model)
            preflight_metadata_key = LOCAL_PREFLIGHT_W3_METADATA_KEY
            preflight_store_key = LOCAL_PREFLIGHT_W3_STORE_KEY
        elif task_w2_preflight is not None:
            preflight, performed_here = await task_w2_preflight.ensure(decision_model)
            preflight_metadata_key = LOCAL_PREFLIGHT_W2_METADATA_KEY
            preflight_store_key = LOCAL_PREFLIGHT_W2_STORE_KEY
        else:
            preflight, performed_here = await task_preflight.ensure(decision_model)
            preflight_metadata_key = LOCAL_PREFLIGHT_METADATA_KEY
            preflight_store_key = LOCAL_PREFLIGHT_STORE_KEY
        state = decision_model.state
        serialized_preflight = preflight.model_dump(mode="json")
        state.metadata[preflight_metadata_key] = serialized_preflight
        state.store.set(preflight_store_key, serialized_preflight)
        setup_latency_ms = preflight.setup_latency_ms if performed_here else 0.0

        scenario = _scenario_from_state(state)
        if isinstance(shared_vectorizer, FastEmbedVectorizer):
            setup_latency_ms += shared_vectorizer.warmup()
            if (
                expected_embedding_artifact_sha256 is not None
                and shared_vectorizer.artifact_sha256
                != expected_embedding_artifact_sha256
            ):
                raise ValueError(
                    "local embedding snapshot differs from the frozen manifest"
                )
        oracle_compiler: OracleCompiler | None = None
        if system == "anamnesis":
            prompt_builder = {
                "w1": build_local_memory_compiler_prompt,
                "w2": build_local_memory_compiler_w2_prompt,
                "w3": build_local_memory_compiler_w3_prompt,
            }[compiler_prompt_variant]
            strategy = AnamnesisMemoryStrategy(
                compiler=LocalInspectMemoryCompiler(
                    decision_model,
                    prompt_builder=prompt_builder,
                )
            )
        elif system == ORACLE_SYSTEM_NAME:
            if oracle_annotations_path is None or oracle_annotations_sha256 is None:
                raise ValueError("oracle artifact was not bound at task construction")
            oracle_bytes = Path(oracle_annotations_path).read_bytes()
            if hashlib.sha256(oracle_bytes).hexdigest() != oracle_annotations_sha256:
                raise ValueError(
                    "oracle artifact bytes changed after task construction"
                )
            oracle_artifact = OracleCompilerArtifact.model_validate_json(oracle_bytes)
            oracle_compiler = OracleCompiler(oracle_artifact, scenario)
            strategy = OracleAnamnesisMemoryStrategy(compiler=oracle_compiler)
        else:
            strategy = create_strategy(
                system,
                vectorizer=shared_vectorizer,
                top_k=top_k,
            )
        system_config_sha256 = local_system_config_sha256(
            system=system,
            top_k=top_k,
            embedding_model=embedding_model,
            embedding_repository=embedding_repository,
            embedding_revision=embedding_revision,
            embedding_artifact_sha256=(
                shared_vectorizer.artifact_sha256
                if isinstance(shared_vectorizer, FastEmbedVectorizer)
                else None
            ),
            pricing_config_sha256=pricing_config_sha256,
            oracle_annotations_sha256=oracle_annotations_sha256,
            compiler_prompt_variant=compiler_prompt_variant,
            w3_preflight_fixture_sha256=(
                LOCAL_WRITER_W3_PREFLIGHT_FIXTURE_SHA256
                if w3_fixture is not None
                else None
            ),
            w3_preflight_protocol_sha256=w3_preflight_protocol_sha256,
            w3_dataset_sha256=w3_dataset_sha256,
            w3_reference_sha256=w3_reference_sha256,
        )
        if (
            expected_system_config_sha256 is not None
            and system_config_sha256 != expected_system_config_sha256
        ):
            raise ValueError(
                "runtime system configuration differs from the local manifest"
            )
        run = await run_scenario(
            scenario=scenario,
            strategy=strategy,
            model=decision_model,
            repetition=repetition,
            seed=seed,
            scenario_sha256_override=_scenario_sha256_from_state(state),
            system_config_sha256=system_config_sha256,
            manifest_sha256=manifest_sha256,
            pricing_config_sha256=pricing_config_sha256,
            setup_latency_ms=setup_latency_ms,
            decision_prompt_builder=build_local_decision_prompt,
            decision_prompt_contract=local_decision_contract(),
            decision_prompt_version=LOCAL_DECISION_VERSION,
        )
        if oracle_compiler is not None:
            oracle_compiler.assert_complete()
            _validate_oracle_scenario_run(scenario, run)
        if run.usage.cost_usd != 0.0 or not run.cost_complete:
            raise ValueError("local run did not preserve complete zero-cost accounting")
        state = decision_model.state
        state.store.set(SCENARIO_RUN_STORE_KEY, run.model_dump(mode="json"))
        state.output = ModelOutput.from_content(
            model=decision_model.name,
            content=run.model_dump_json(),
        )
        return state

    return solve


def verify_zero_local_pricing(path: str | Path) -> str:
    """Verify the tracked pricing file declares exactly one all-zero local entry."""

    pricing_path = Path(path)
    content = pricing_path.read_bytes()
    raw = json.loads(content)
    if not isinstance(raw, dict) or set(raw) != {LOCAL_OLLAMA_MODEL}:
        raise ValueError("local pricing config must contain exactly the pinned model")
    entry = raw[LOCAL_OLLAMA_MODEL]
    expected_fields = {
        "input",
        "output",
        "input_cache_write",
        "input_cache_read",
    }
    if not isinstance(entry, dict) or set(entry) != expected_fields:
        raise ValueError("local pricing config has an invalid Inspect cost shape")
    if any(value != 0 and value != 0.0 for value in entry.values()):
        raise ValueError("local pricing config contains a non-zero API price")
    return hashlib.sha256(content).hexdigest()


__all__ = [
    "LOCAL_CONTEXT_ENV",
    "LOCAL_DECISION_VERSION",
    "LOCAL_STRUCTURED_MEMORY_PRECEDENCE",
    "LOCAL_MODEL_PREFLIGHT_PURPOSE",
    "LOCAL_MODEL_PREFLIGHT_TASK_VERSION",
    "LOCAL_MODEL_PREFLIGHT_W2_PURPOSE",
    "LOCAL_MODEL_PREFLIGHT_W2_SAMPLE_ID",
    "LOCAL_MODEL_PREFLIGHT_W2_TASK_VERSION",
    "LOCAL_NO_CLOUD_ENV",
    "LOCAL_OLLAMA_BASE_URL",
    "LOCAL_OLLAMA_CONTEXT_LENGTH",
    "LOCAL_OLLAMA_MANIFEST_SHA256",
    "LOCAL_OLLAMA_MODEL",
    "LOCAL_OLLAMA_MODEL_BLOB_SHA256",
    "LOCAL_PREFLIGHT_METADATA_KEY",
    "LOCAL_PREFLIGHT_W2_METADATA_KEY",
    "LOCAL_PREFLIGHT_W2_STORE_KEY",
    "LOCAL_SCENARIO_TASK_VERSION",
    "LocalDecisionWire",
    "LocalInspectDecisionModel",
    "LocalInspectMemoryCompiler",
    "LocalSystemName",
    "LocalLoadedModelAttestation",
    "LocalModelPreflightResult",
    "LocalModelPreflightW2CaseResult",
    "LocalModelPreflightW2Result",
    "build_local_decision_prompt",
    "local_decision_contract",
    "local_decision_prompt_contract",
    "local_decision_schema_contract",
    "local_memory_compiler_prompt_contract",
    "local_memory_compiler_schema_contract",
    "local_memory_compiler_transport_contract",
    "local_memory_compiler_w2_prompt_contract",
    "local_memory_compiler_w2_transport_contract",
    "local_model_preflight_sample",
    "local_model_preflight_scorer",
    "local_model_preflight_solver",
    "local_model_preflight_w2_sample",
    "local_model_preflight_w2_scorer",
    "local_model_preflight_w2_solver",
    "local_w2_preflight_prompts",
    "load_local_w2_preflight_fixture",
    "local_scenario_solver",
    "local_system_config_sha256",
    "probe_loaded_local_model",
    "run_local_model_preflight",
    "run_local_model_preflight_w2",
    "verify_zero_local_pricing",
]
