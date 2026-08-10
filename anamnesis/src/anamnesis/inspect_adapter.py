"""Inspect AI bridge for the provider-neutral Anamnesis evaluation core."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageUser,
    ModelCost,
    ModelOutput,
    ResponseSchema,
    get_model,
    get_model_info,
)
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    metric,
    scorer,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import json_schema
from pydantic import ValidationError

from anamnesis.baselines import (
    AnamnesisMemoryStrategy,
    FastEmbedVectorizer,
    Vectorizer,
    create_strategy,
)
from anamnesis.io import canonical_sha256
from anamnesis.memory import CompilerCall, CompilerRequest
from anamnesis.prompts import (
    build_decision_prompt,
    build_memory_compiler_prompt,
    memory_compiler_contract,
)
from anamnesis.runner import DecisionCall, DecisionModel, DecisionRequest, run_scenario
from anamnesis.runtime_contract import anamnesis_runtime_contract
from anamnesis.schema import (
    Decision,
    HostedWarmupAttestation,
    ObservableEvent,
    RuntimeScenario,
    Scenario,
    ScenarioRun,
    StrictModel,
    Usage,
)
from anamnesis.scoring import ScenarioScore, score_scenario
from anamnesis.wire import DecisionWire, MemoryDeltaWire

BaselineName = Literal["no_memory", "full_context", "vector_rag"]
SystemName = Literal["no_memory", "full_context", "vector_rag", "anamnesis"]

SCENARIO_METADATA_KEY = "scenario"
SCENARIO_SHA256_METADATA_KEY = "scenario_sha256"
SCENARIO_RUN_STORE_KEY = "anamnesis.scenario_run"
SCENARIO_SCORE_METADATA_KEY = "scenario_score"
HOSTED_WARMUP_PROMPT = (
    "Synthetic Anamnesis setup warmup. This is not a user event or benchmark "
    'checkpoint. Return exactly {"actions":[]} using the supplied schema.'
)
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"


def hosted_warmup_prompt_sha256() -> str:
    return hashlib.sha256(HOSTED_WARMUP_PROMPT.encode()).hexdigest()


def hosted_warmup_schema_sha256() -> str:
    serialized = json.dumps(
        DecisionWire.model_json_schema(), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _verify_official_openai_endpoint(active_model: object, model_name: str) -> None:
    """Reject explicit and environment-provided OpenAI endpoint overrides."""

    if not model_name.startswith("openai/"):
        raise ValueError("v0 requires the direct OpenAI provider")
    api = getattr(active_model, "api", None)
    client = getattr(api, "client", None)
    base_url = getattr(client, "base_url", None)
    if str(base_url).rstrip("/") != OFFICIAL_OPENAI_BASE_URL:
        raise ValueError("active model is not using the official OpenAI endpoint")


class ModelPreflightResult(StrictModel):
    """Synthetic structured-output compatibility check; never a benchmark."""

    model: str
    strict_schema_supported: bool
    compiler_parse_error: bool
    decision_parse_error: bool
    compiler_usage_complete: bool
    decision_usage_complete: bool
    compiler_cost_complete: bool
    decision_cost_complete: bool
    passed: bool


def scenario_record_to_sample(record: dict[str, Any]) -> Sample:
    """Create an Inspect sample without placing hidden gold in metadata."""

    scenario = Scenario.model_validate(record)
    return Sample(
        id=scenario.id,
        input=f"Run the seven-day simulated scenario {scenario.id}.",
        target=scenario.model_dump_json(),
        metadata={
            SCENARIO_METADATA_KEY: scenario.to_runtime().model_dump(mode="json"),
            SCENARIO_SHA256_METADATA_KEY: canonical_sha256(scenario),
        },
    )


def _supports_strict_schema(model_name: str) -> bool:
    """Return whether Inspect documents strict JSON schemas for the provider."""

    provider = model_name.split("/", maxsplit=1)[0].casefold()
    return provider in {"openai", "mistral"}


def _decision_schema(model_name: str) -> ResponseSchema:
    return ResponseSchema(
        name="anamnesis_decision",
        json_schema=json_schema(DecisionWire),
        strict=True if _supports_strict_schema(model_name) else None,
    )


def _memory_delta_schema(model_name: str) -> ResponseSchema:
    return ResponseSchema(
        name="anamnesis_memory_delta",
        json_schema=json_schema(MemoryDeltaWire),
        strict=True if _supports_strict_schema(model_name) else None,
    )


def _logical_input_tokens(model_usage: object) -> int:
    input_tokens = int(getattr(model_usage, "input_tokens", 0))
    cache_read = int(getattr(model_usage, "input_tokens_cache_read", 0) or 0)
    cache_write = int(getattr(model_usage, "input_tokens_cache_write", 0) or 0)
    return input_tokens + cache_read + cache_write


def _usage_from_output(output: ModelOutput) -> Usage:
    model_usage = output.usage
    if model_usage is None:
        return Usage()
    uncached = int(model_usage.input_tokens)
    cache_read = int(getattr(model_usage, "input_tokens_cache_read", 0) or 0)
    cache_write = int(getattr(model_usage, "input_tokens_cache_write", 0) or 0)
    return Usage(
        input_tokens=uncached + cache_read + cache_write,
        uncached_input_tokens=uncached,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        output_tokens=model_usage.output_tokens,
        cost_usd=model_usage.total_cost,
    )


def _usage_is_complete(output: ModelOutput) -> bool:
    return output.usage is not None


def _cost_is_complete(output: ModelOutput) -> bool:
    return output.usage is not None and output.usage.total_cost is not None


def _configured_model_cost(path: str, model_name: str) -> ModelCost:
    """Read the exact cost entry accepted by Inspect's model-cost config."""

    raw_text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        import yaml

        raw = yaml.safe_load(raw_text)
    if not isinstance(raw, dict) or model_name not in raw:
        raise ValueError(f"pricing config has no entry for {model_name}")
    return ModelCost.model_validate(raw[model_name])


def _verify_effective_model_cost(path: str, model_name: str) -> None:
    """Bind provider-reported total cost to the pinned Inspect cost table."""

    expected = _configured_model_cost(path, model_name)
    info = get_model_info(get_model())
    if info is None or info.cost is None:
        raise ValueError("active Inspect model has no effective pricing configuration")
    if info.cost != expected:
        raise ValueError("active Inspect pricing differs from the frozen config")


class InspectDecisionModel(DecisionModel):
    """Adapt Inspect's active model and Generate callback to DecisionModel."""

    def __init__(self, state: TaskState, generate: Generate) -> None:
        active_model = get_model()  # Fail early without an active model context.
        self.name = str(state.model)
        _verify_official_openai_endpoint(active_model, self.name)
        self.state = state
        self._generate = generate
        self._response_schema = _decision_schema(self.name)

    async def complete_structured(
        self,
        *,
        prompt: str,
        response_schema: ResponseSchema,
    ) -> tuple[ModelOutput, float]:
        """Make one no-retry structured call while sharing the Inspect state."""

        self.state.messages = [ChatMessageUser(content=prompt)]
        started = perf_counter()
        self.state = await self._generate(
            self.state,
            tool_calls="none",
            response_schema=response_schema,
        )
        return self.state.output, max(0.0, (perf_counter() - started) * 1000)

    async def decide(self, request: DecisionRequest) -> DecisionCall:
        # Each checkpoint is an independent decision. The selected memory context is
        # already rendered into request.prompt by the provider-neutral runner.
        output, measured_latency_ms = await self.complete_structured(
            prompt=request.prompt,
            response_schema=self._response_schema,
        )

        try:
            decision = DecisionWire.model_validate_json(output.completion).to_domain()
            parse_error = False
        except ValidationError:
            # Invalid output is recorded but never executed as an action.
            decision = Decision()
            parse_error = True

        return DecisionCall(
            decision=decision,
            usage=_usage_from_output(output),
            latency_ms=measured_latency_ms,
            parse_error=parse_error,
            raw_completion=output.completion,
            usage_complete=_usage_is_complete(output),
            cost_complete=_cost_is_complete(output),
        )


async def _run_hosted_model_warmup(
    model: InspectDecisionModel,
) -> HostedWarmupAttestation:
    """Make the one task-level synthetic setup call without a retry."""

    output, latency_ms = await model.complete_structured(
        prompt=HOSTED_WARMUP_PROMPT,
        response_schema=_decision_schema(model.name),
    )
    try:
        decision = DecisionWire.model_validate_json(output.completion).to_domain()
        parse_error = bool(decision.actions)
    except ValidationError:
        parse_error = True
    attestation = HostedWarmupAttestation(
        model=model.name,
        prompt_sha256=hosted_warmup_prompt_sha256(),
        response_schema_sha256=hosted_warmup_schema_sha256(),
        raw_completion=output.completion,
        usage=_usage_from_output(output),
        usage_complete=_usage_is_complete(output),
        cost_complete=_cost_is_complete(output),
        parse_error=parse_error,
        latency_ms=latency_ms,
    )
    if (
        attestation.parse_error
        or not attestation.usage_complete
        or not attestation.cost_complete
    ):
        raise ValueError("hosted-model warmup failed structured output or accounting")
    return attestation


class _TaskHostedWarmup:
    """Concurrency-safe one-shot warmup shared by every sample in one task."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._attestation: HostedWarmupAttestation | None = None

    async def ensure(
        self, model: InspectDecisionModel
    ) -> tuple[HostedWarmupAttestation, bool]:
        if self._attestation is not None:
            return self._attestation, False
        async with self._lock:
            if self._attestation is None:
                self._attestation = await _run_hosted_model_warmup(model)
                return self._attestation, True
        return self._attestation, False


class InspectMemoryCompiler:
    """Compile observable events with the same active Inspect model session."""

    def __init__(self, model: InspectDecisionModel) -> None:
        self.name = model.name
        self._model = model
        self._response_schema = _memory_delta_schema(self.name)

    async def compile(self, request: CompilerRequest) -> CompilerCall:
        output, latency_ms = await self._model.complete_structured(
            prompt=build_memory_compiler_prompt(
                event=request.event,
                active_state=request.active_state,
            ),
            response_schema=self._response_schema,
        )
        try:
            delta = MemoryDeltaWire.model_validate_json(output.completion).to_domain()
            parse_error = False
        except ValidationError:
            delta = None
            parse_error = True
        return CompilerCall(
            delta=delta,
            usage=_usage_from_output(output),
            latency_ms=latency_ms,
            parse_error=parse_error,
            raw_completion=output.completion,
            usage_complete=_usage_is_complete(output),
            cost_complete=_cost_is_complete(output),
        )


def _scenario_from_state(state: TaskState) -> RuntimeScenario:
    raw_scenario = state.metadata.get(SCENARIO_METADATA_KEY)
    if raw_scenario is None:
        raise ValueError(f"sample metadata is missing {SCENARIO_METADATA_KEY!r}")
    return RuntimeScenario.model_validate(raw_scenario)


def _scenario_sha256_from_state(state: TaskState) -> str:
    scenario_sha256 = state.metadata.get(SCENARIO_SHA256_METADATA_KEY)
    if not isinstance(scenario_sha256, str):
        raise ValueError(f"sample metadata is missing {SCENARIO_SHA256_METADATA_KEY!r}")
    return scenario_sha256


def _system_config_sha256(
    *,
    system: str,
    model: str,
    top_k: int,
    embedding_model: str,
    embedding_repository: str = "qdrant/bge-small-en-v1.5-onnx-q",
    embedding_revision: str | None = None,
    embedding_artifact_sha256: str | None = None,
    pricing_config_sha256: str | None = None,
) -> str:
    """Fingerprint the runtime knobs that can change a system's result."""

    payload: dict[str, object] = {
        "model": model,
        "system": system,
        "temperature": 0.0,
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
    if system == "anamnesis":
        payload["memory_compiler_sha256"] = hashlib.sha256(
            memory_compiler_contract().encode()
        ).hexdigest()
        payload["deterministic_memory"] = anamnesis_runtime_contract()
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


@solver
def scenario_solver(
    baseline: SystemName,
    *,
    repetition: int = 1,
    seed: int | None = None,
    top_k: int = 5,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_repository: str = "qdrant/bge-small-en-v1.5-onnx-q",
    embedding_revision: str | None = None,
    embedding_snapshot_path: str | None = None,
    expected_model: str | None = None,
    expected_system_config_sha256: str | None = None,
    expected_embedding_artifact_sha256: str | None = None,
    manifest_sha256: str | None = None,
    pricing_config_sha256: str | None = None,
    pricing_config_path: str | None = None,
) -> Solver:
    """Run one baseline over each complete scenario sample."""

    if repetition < 1:
        raise ValueError("repetition must be at least 1")

    shared_vectorizer: Vectorizer | None = None
    if baseline == "vector_rag":
        if embedding_revision is None:
            raise ValueError("measured vector_rag requires an exact embedding revision")
        # The embedding model is immutable and expensive to load. Selector state is
        # still created below per sample, so scenario memories remain isolated.
        shared_vectorizer = FastEmbedVectorizer(
            model_name=embedding_model,
            repository=embedding_repository,
            revision=embedding_revision,
            snapshot_path=embedding_snapshot_path,
        )

    task_warmup = _TaskHostedWarmup()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        scenario = _scenario_from_state(state)
        setup_latency_ms = 0.0
        if isinstance(shared_vectorizer, FastEmbedVectorizer):
            setup_latency_ms = shared_vectorizer.warmup()
            if (
                expected_embedding_artifact_sha256 is not None
                and shared_vectorizer.artifact_sha256
                != expected_embedding_artifact_sha256
            ):
                raise ValueError(
                    "local embedding snapshot differs from the frozen manifest"
                )
        decision_model = InspectDecisionModel(state, generate)
        if pricing_config_path is not None:
            _verify_effective_model_cost(pricing_config_path, decision_model.name)
        if expected_model is not None and decision_model.name != expected_model:
            raise ValueError("active model differs from the frozen final manifest")
        hosted_warmup: HostedWarmupAttestation | None = None
        if manifest_sha256 is not None:
            hosted_warmup, performed_here = await task_warmup.ensure(decision_model)
            if performed_here:
                setup_latency_ms += hosted_warmup.latency_ms
        if baseline == "anamnesis":
            strategy = AnamnesisMemoryStrategy(
                compiler=InspectMemoryCompiler(decision_model)
            )
        else:
            strategy = create_strategy(
                baseline,
                vectorizer=shared_vectorizer,
                top_k=top_k,
            )
        system_config_sha256 = _system_config_sha256(
            system=baseline,
            model=decision_model.name,
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
        )
        if (
            expected_system_config_sha256 is not None
            and system_config_sha256 != expected_system_config_sha256
        ):
            raise ValueError(
                "runtime system configuration differs from the final manifest"
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
            hosted_warmup=hosted_warmup,
        )

        state = decision_model.state
        serialized_run = run.model_dump_json()
        state.store.set(
            SCENARIO_RUN_STORE_KEY,
            run.model_dump(mode="json"),
        )
        state.output = ModelOutput.from_content(
            model=decision_model.name,
            content=serialized_run,
        )
        return state

    return solve


def model_preflight_sample() -> Sample:
    """Return one synthetic case that is not part of the 50 scenarios."""

    return Sample(
        id="model-preflight-v0",
        input="Check strict structured output and usage accounting.",
        target="pass",
    )


@solver
def model_preflight_solver() -> Solver:
    """Make exactly one MemoryDelta call and one Decision call."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = InspectDecisionModel(state, generate)
        compiler = InspectMemoryCompiler(model)
        event = ObservableEvent(
            id="preflight-event",
            at=datetime.fromisoformat("2026-01-05T09:00:00+00:00"),
            kind="user_message",
            text="At 17:00 today remind me to run the compatibility check.",
        )
        compiler_call = await compiler.compile(
            CompilerRequest(
                event=event,
                active_state='{"facts":[],"intents":[]}',
            )
        )
        decision_call = await model.decide(
            DecisionRequest(
                event=event,
                prompt=build_decision_prompt(
                    now=event.at.isoformat(),
                    current_event_id=event.id,
                    context_events=[event],
                    decision_history=[],
                    memory_view=None,
                ),
            )
        )
        strict_supported = _supports_strict_schema(model.name)
        passed = all(
            (
                strict_supported,
                not compiler_call.parse_error,
                not decision_call.parse_error,
                compiler_call.usage_complete,
                decision_call.usage_complete,
                compiler_call.cost_complete,
                decision_call.cost_complete,
            )
        )
        result = ModelPreflightResult(
            model=model.name,
            strict_schema_supported=strict_supported,
            compiler_parse_error=compiler_call.parse_error,
            decision_parse_error=decision_call.parse_error,
            compiler_usage_complete=compiler_call.usage_complete,
            decision_usage_complete=decision_call.usage_complete,
            compiler_cost_complete=compiler_call.cost_complete,
            decision_cost_complete=decision_call.cost_complete,
            passed=passed,
        )
        state = model.state
        state.output = ModelOutput.from_content(
            model=model.name,
            content=result.model_dump_json(),
        )
        return state

    return solve


@scorer(metrics=[])
def model_preflight_scorer() -> Scorer:
    """Score only compatibility, never reminder quality."""

    async def score(state: TaskState, target: Target) -> Score:
        result = ModelPreflightResult.model_validate_json(state.output.completion)
        return Score(
            value=1 if result.passed else 0,
            answer=result.model,
            explanation=(
                "strict compiler/decision schemas and accounting passed"
                if result.passed
                else "model failed one or more compatibility checks"
            ),
        )

    return score


def _validated_scores(scores: list[SampleScore]) -> list[ScenarioScore]:
    validated: list[ScenarioScore] = []
    for sample_score in scores:
        metadata = sample_score.score.metadata or {}
        raw_score = metadata.get(SCENARIO_SCORE_METADATA_KEY)
        if raw_score is None:
            raise ValueError(
                "Inspect score is missing deterministic ScenarioScore metadata"
            )
        validated.append(ScenarioScore.model_validate(raw_score))
    return validated


def _sum(scores: list[SampleScore], field: str) -> int:
    return sum(int(getattr(item, field)) for item in _validated_scores(scores))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@metric
def true_positives() -> Metric:
    def calculate(scores: list[SampleScore]) -> int:
        return _sum(scores, "tp")

    return calculate


@metric
def false_positives() -> Metric:
    def calculate(scores: list[SampleScore]) -> int:
        return _sum(scores, "fp")

    return calculate


@metric
def false_negatives() -> Metric:
    def calculate(scores: list[SampleScore]) -> int:
        return _sum(scores, "fn")

    return calculate


@metric
def precision() -> Metric:
    def calculate(scores: list[SampleScore]) -> float:
        tp = _sum(scores, "tp")
        return _ratio(tp, tp + _sum(scores, "fp"))

    return calculate


@metric
def recall() -> Metric:
    def calculate(scores: list[SampleScore]) -> float:
        tp = _sum(scores, "tp")
        return _ratio(tp, tp + _sum(scores, "fn"))

    return calculate


@metric
def f1() -> Metric:
    def calculate(scores: list[SampleScore]) -> float:
        tp = _sum(scores, "tp")
        fp = _sum(scores, "fp")
        fn = _sum(scores, "fn")
        return _ratio(2 * tp, 2 * tp + fp + fn)

    return calculate


@metric
def false_reminders() -> Metric:
    def calculate(scores: list[SampleScore]) -> int:
        return _sum(scores, "false_reminders")

    return calculate


@metric
def false_alarm_rate() -> Metric:
    def calculate(scores: list[SampleScore]) -> float:
        return _ratio(
            _sum(scores, "false_alarm_checkpoints"),
            _sum(scores, "negative_checkpoints"),
        )

    return calculate


@metric
def obsolete_errors() -> Metric:
    def calculate(scores: list[SampleScore]) -> int:
        return _sum(scores, "obsolete_errors")

    return calculate


@metric
def obsolete_trap_rate() -> Metric:
    def calculate(scores: list[SampleScore]) -> float:
        return _ratio(
            _sum(scores, "obsolete_traps_triggered"),
            _sum(scores, "obsolete_traps"),
        )

    return calculate


@metric
def duplicate_errors() -> Metric:
    def calculate(scores: list[SampleScore]) -> int:
        return _sum(scores, "duplicate_errors")

    return calculate


@metric
def provenance_exact_accuracy() -> Metric:
    def calculate(scores: list[SampleScore]) -> float:
        return _ratio(
            _sum(scores, "provenance_exact"),
            _sum(scores, "tp"),
        )

    return calculate


@metric
def invalid_outputs() -> Metric:
    def calculate(scores: list[SampleScore]) -> int:
        return _sum(scores, "invalid_outputs")

    return calculate


@scorer(
    metrics=[
        true_positives(),
        false_positives(),
        false_negatives(),
        precision(),
        recall(),
        f1(),
        false_reminders(),
        false_alarm_rate(),
        obsolete_errors(),
        obsolete_trap_rate(),
        duplicate_errors(),
        provenance_exact_accuracy(),
        invalid_outputs(),
    ]
)
def scenario_run_scorer() -> Scorer:
    """Score a serialized ScenarioRun without any model-based judging."""

    async def score(state: TaskState, target: Target) -> Score:
        scenario = Scenario.model_validate_json(target.text)
        run = ScenarioRun.model_validate_json(state.output.completion)
        if run.scenario_sha256 != canonical_sha256(scenario):
            raise ValueError("ScenarioRun was produced from a different scenario")
        scenario_score = score_scenario(scenario, run)
        denominator = 2 * scenario_score.tp + scenario_score.fp + scenario_score.fn
        scenario_f1 = _ratio(2 * scenario_score.tp, denominator) if denominator else 1.0
        return Score(
            value=scenario_f1,
            answer=run.scenario_id,
            metadata={
                SCENARIO_SCORE_METADATA_KEY: scenario_score.model_dump(mode="json")
            },
        )

    return score
