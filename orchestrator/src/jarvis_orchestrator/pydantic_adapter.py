"""Pinned Pydantic AI adapter boundary.

The dependency-free lifecycle does not import this module. When Pydantic AI
2.27.0 is installed, the wrapper below ensures that every Model interface call
has a persisted reservation; logical request hooks remain telemetry only.
"""

from __future__ import annotations

import contextvars
import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import asdict, dataclass
from dataclasses import replace
from typing import Any

from .common_schema import JarvisCommonSchemaTransformer, route_decision_schema
from .cloud_routing import ContextScope, OutputContract, TaskKind, TaskProfileV1
from .domain import ModelChunk, ModelRequest, ModelUsage, RouteDecision, canonical_json
from .history import canonical_messages_to_pydantic
from .model import DispatchPermit, GuardedAsyncTransport, _dispatch_reservation
from .store import OrchestratorStore
from .research import (
    NormalizedDocumentV1,
    RankedPassageV1,
    SearchFacetV1,
    SearchPlanV1,
    SpanRelation,
    SpanSelectionV1,
)

try:  # pragma: no cover - verified in the pinned integration environment
    from pydantic_ai.models.wrapper import WrapperModel
except ImportError:  # pragma: no cover
    WrapperModel = object  # type: ignore[assignment,misc]


class BudgetedPydanticModel(WrapperModel):  # type: ignore[misc]
    """Pydantic ``WrapperModel`` with graph-independent request accounting."""

    def __init__(self, wrapped: Any, store: OrchestratorStore, context_provider: Any) -> None:
        if WrapperModel is object:
            raise RuntimeError("pydantic-ai-slim[openai]==2.27.0 is required")
        super().__init__(wrapped)
        self._jarvis_store = store
        self._jarvis_context_provider = context_provider

    @asynccontextmanager
    async def _reservation(self, model_settings: Any):
        context = self._jarvis_context_provider()
        allowance = int(context.output_allowance)
        reservation_id = self._jarvis_store.reserve_model(
            context.task_id, context.epoch, purpose=context.purpose, output_allowance=allowance
        )
        self._jarvis_store.mark_model_dispatched(reservation_id, context.task_id, context.epoch)
        token = _dispatch_reservation.set(DispatchPermit(reservation_id))
        try:
            if model_settings is None:
                model_settings = {}
            model_settings = dict(model_settings)
            model_settings["max_tokens"] = allowance
            if context.timeout_seconds is not None:
                model_settings["timeout"] = context.timeout_seconds
            yield model_settings, context, reservation_id
        except BaseException:
            with suppress(Exception):
                self._jarvis_store.reconcile_model(reservation_id, context.task_id, context.epoch, None)
            raise
        finally:
            _dispatch_reservation.reset(token)

    async def request(self, messages: Any, model_settings: Any, model_request_parameters: Any) -> Any:
        async with self._reservation(model_settings) as (settings, context, reservation_id):
            response = await self.wrapped.request(messages, settings, model_request_parameters)
            usage = response.usage
            output = getattr(usage, "output_tokens", None)
            self._jarvis_store.reconcile_model(reservation_id, context.task_id, context.epoch, output)
            return response

    @asynccontextmanager
    async def request_stream(self, messages: Any, model_settings: Any, model_request_parameters: Any, run_context: Any = None):
        async with self._reservation(model_settings) as (settings, context, reservation_id):
            async with self.wrapped.request_stream(messages, settings, model_request_parameters, run_context) as response:
                yield response
                usage = response.usage
                output = getattr(usage, "output_tokens", None)
                self._jarvis_store.reconcile_model(reservation_id, context.task_id, context.epoch, output)


@dataclass(frozen=True, slots=True)
class PydanticDispatchContext:
    task_id: str
    epoch: int
    purpose: str
    output_allowance: int
    timeout_seconds: float | None


_pydantic_context: contextvars.ContextVar[PydanticDispatchContext | None] = contextvars.ContextVar(
    "jarvis_pydantic_dispatch_context", default=None
)


def _current_context() -> PydanticDispatchContext:
    result = _pydantic_context.get()
    if result is None:
        raise RuntimeError("Pydantic model dispatch has no orchestrator context")
    return result


class PydanticAgentModelPort:
    """Text ModelPort backed by the pinned Pydantic AI Agent runtime."""

    def __init__(
        self,
        agent: Any,
        budgeted_model: Any,
        store: OrchestratorStore,
        *,
        tools_capable: bool,
        json_schema_capable: bool,
        registered_tools: tuple[str, ...],
    ) -> None:
        self._agent = agent
        self.__budgeted_model = budgeted_model
        self._store = store
        self._tools_capable = tools_capable
        self._json_schema_capable = json_schema_capable
        self._registered_tools = frozenset(registered_tools)

    async def stream(self, request: ModelRequest):
        from pydantic_ai.usage import UsageLimits

        history, prompt = canonical_messages_to_pydantic(request.messages)
        context = PydanticDispatchContext(
            request.task_id,
            request.epoch,
            request.purpose,
            request.output_allowance,
            request.timeout_seconds,
        )
        token = _pydantic_context.set(context)
        try:
            settings = {
                "max_tokens": request.output_allowance,
                "temperature": 0,
                "top_p": 1,
                "timeout": request.timeout_seconds,
                "parallel_tool_calls": False,
            }
            async with self._agent.run_stream(
                prompt,
                message_history=history,
                instructions=request.system_prompt,
                model_settings=settings,
                usage_limits=UsageLimits(request_limit=1),
            ) as result:
                async for text in result.stream_text(delta=True, debounce_by=None):
                    yield ModelChunk(text)
                yield ModelChunk("", ModelUsage(output_tokens=result.usage.output_tokens, input_tokens=result.usage.input_tokens))
        finally:
            _pydantic_context.reset(token)

    async def plan(self, request: ModelRequest) -> RouteDecision:
        """Run the strict M1B planner with native schema or a synthetic output tool."""

        from enum import Enum
        from typing import Annotated, Literal

        from pydantic import ConfigDict, Field, create_model
        from pydantic_ai import Agent
        from pydantic_ai.output import NativeOutput, ToolOutput
        from pydantic_ai.usage import UsageLimits

        if not self._json_schema_capable and not self._tools_capable:
            raise RuntimeError("hybrid planner is unavailable without schema or tool capability")
        tool_values = ("none", *sorted(self._registered_tools))
        tool_enum = Enum(
            "JarvisToolName",
            {f"value_{index}": value for index, value in enumerate(tool_values)},
            type=str,
        )
        model_type = create_model(
            "JarvisRouteDecision",
            __config__=ConfigDict(extra="forbid", use_enum_values=True),
            route=(Literal["local", "memory_local", "research_local", "cloud", "tool_local"], ...),
            memory_query=(Annotated[str, Field(max_length=4096)], ...),
            research_query=(Annotated[str, Field(max_length=4096)], ...),
            tool_name=(tool_enum, ...),
            response_language=(Literal["el", "en", "default"], ...),
        )
        output = (
            NativeOutput(model_type, name="route_decision", strict=True)
            if self._json_schema_capable
            else ToolOutput(model_type, name="route_decision", strict=True, max_retries=0)
        )
        planner = Agent(
            self.__budgeted_model,
            output_type=output,
            retries={"tools": 0, "output": 0},
        )
        history, prompt = canonical_messages_to_pydantic(request.messages)
        context = PydanticDispatchContext(
            request.task_id,
            request.epoch,
            request.purpose,
            request.output_allowance,
            request.timeout_seconds,
        )
        token = _pydantic_context.set(context)
        try:
            result = await planner.run(
                prompt,
                message_history=history,
                instructions=request.system_prompt,
                model_settings={
                    "max_tokens": request.output_allowance,
                    "temperature": 0,
                    "top_p": 1,
                    "parallel_tool_calls": False,
                    "timeout": request.timeout_seconds,
                },
                usage_limits=UsageLimits(request_limit=1),
            )
        finally:
            _pydantic_context.reset(token)
        return RouteDecision(**result.output.model_dump()).validate(self._registered_tools)

    async def profile_task(
        self,
        request: ModelRequest,
        *,
        context_scope: ContextScope,
        output_contract: OutputContract,
        response_language: str,
    ) -> TaskProfileV1:
        """Retry-free local classification; lifecycle/egress facts stay code-owned."""

        from typing import Annotated, Literal

        from pydantic import BaseModel, ConfigDict, Field
        from pydantic_ai import Agent
        from pydantic_ai.output import NativeOutput, ToolOutput
        from pydantic_ai.usage import UsageLimits

        class ProfileOutput(BaseModel):
            model_config = ConfigDict(extra="forbid")
            task_kind: Literal[
                "summary", "classification", "extraction", "tool_decision",
                "qa", "analysis", "planning", "coding",
            ]
            reasoning_depth: Annotated[int, Field(ge=0, le=3)]
            constraint_count: Annotated[int, Field(ge=0, le=8)]
            domains: Annotated[
                list[Literal[
                    "general", "software", "mathematics", "science", "business",
                    "finance", "legal", "medical", "creative",
                ]],
                Field(min_length=1, max_length=4),
            ]

        if not self._json_schema_capable and not self._tools_capable:
            raise RuntimeError("task profiler requires qualified structured output")
        output = (
            NativeOutput(ProfileOutput, name="task_profile", strict=True)
            if self._json_schema_capable
            else ToolOutput(ProfileOutput, name="task_profile", strict=True, max_retries=0)
        )
        profiler = Agent(self.__budgeted_model, output_type=output, retries={"tools": 0, "output": 0})
        history, prompt = canonical_messages_to_pydantic(request.messages)
        context = PydanticDispatchContext(
            request.task_id,
            request.epoch,
            request.purpose,
            request.output_allowance,
            request.timeout_seconds,
        )
        token = _pydantic_context.set(context)
        try:
            result = await profiler.run(
                prompt,
                message_history=history,
                instructions=(
                    "Classify only task kind, required reasoning depth 0-3, explicit constraint count 0-8, "
                    "and 1-4 unique domains. Do not choose a provider, model, privacy policy, budget or approval."
                ),
                model_settings={
                    "max_tokens": request.output_allowance,
                    "temperature": 0,
                    "top_p": 1,
                    "parallel_tool_calls": False,
                    "timeout": request.timeout_seconds,
                },
                usage_limits=UsageLimits(request_limit=1),
            )
        finally:
            _pydantic_context.reset(token)
        raw = result.output
        return TaskProfileV1(
            version="task-profile/1",
            task_kind=TaskKind(raw.task_kind),
            reasoning_depth=raw.reasoning_depth,
            constraint_count=raw.constraint_count,
            domains=tuple(raw.domains),
            context_scope=context_scope,
            output_contract=output_contract,
            response_language=response_language,
        ).validate()

    async def profile_task_with_correction(
        self,
        request: ModelRequest,
        *,
        context_scope: ContextScope,
        output_contract: OutputContract,
        response_language: str,
    ) -> TaskProfileV1:
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        try:
            return await self.profile_task(
                request,
                context_scope=context_scope,
                output_contract=output_contract,
                response_language=response_language,
            )
        except asyncio.CancelledError:
            raise
        except (UnexpectedModelBehavior, ValueError):
            self._store.mark_correction(request.task_id, request.epoch)
        remaining = self._jarvis_remaining(request)
        correction = replace(
            request,
            purpose="task_profile_correction",
            output_allowance=min(256, remaining),
            messages=(*request.messages, {"role": "user", "content": "Return one corrected task profile matching the exact schema."}),
        )
        return await self.profile_task(
            correction,
            context_scope=context_scope,
            output_contract=output_contract,
            response_language=response_language,
        )

    async def plan_research(
        self,
        *,
        task_id: str,
        epoch: int,
        query: str,
        language: str,
        timezone: str,
        browser_fallback_allowed: bool,
        output_allowance: int,
    ) -> SearchPlanV1 | None:
        """One retry-free structured call combining route and SearchPlanV1."""

        from typing import Annotated, Literal

        from pydantic import BaseModel, ConfigDict, Field
        from pydantic_ai import Agent
        from pydantic_ai.output import NativeOutput, ToolOutput
        from pydantic_ai.usage import UsageLimits

        if output_allowance != 384:
            raise ValueError("research planning output allowance must be 384")
        if not self._json_schema_capable and not self._tools_capable:
            raise RuntimeError("research planner requires qualified structured output")

        class FacetOutput(BaseModel):
            model_config = ConfigDict(extra="forbid")
            facet_id: Annotated[str, Field(min_length=1, max_length=96)]
            label: Annotated[str, Field(min_length=1, max_length=160)]

        class PlanOutput(BaseModel):
            model_config = ConfigDict(extra="forbid")
            route: Literal["local", "research_local"]
            canonical_query: Annotated[str, Field(max_length=4096)]
            facets: Annotated[list[FacetOutput], Field(max_length=6)]
            subqueries: Annotated[list[Annotated[str, Field(min_length=1, max_length=512)]], Field(max_length=4)]
            freshness: Annotated[str, Field(min_length=1, max_length=64)]

        output = (
            NativeOutput(PlanOutput, name="research_route_plan", strict=True)
            if self._json_schema_capable
            else ToolOutput(PlanOutput, name="research_route_plan", strict=True, max_retries=0)
        )
        planner = Agent(
            self.__budgeted_model,
            output_type=output,
            retries={"tools": 0, "output": 0},
        )
        context = PydanticDispatchContext(task_id, epoch, "research_route_plan", 384, None)
        token = _pydantic_context.set(context)
        try:
            result = await planner.run(
                query + "\n/no_think",
                instructions=(
                    "Decide whether the request requires current, externally verifiable web evidence. "
                    "For local, return empty canonical_query/facets/subqueries. For research_local, return "
                    "a concise canonical query, 1-6 ordered facets with stable ASCII IDs, 1-4 ordered "
                    "subqueries, and a deterministic freshness label. Return only the schema output."
                ),
                model_settings={
                    "max_tokens": 384,
                    "temperature": 0,
                    "top_p": 1,
                    "parallel_tool_calls": False,
                },
                usage_limits=UsageLimits(request_limit=1),
            )
        finally:
            _pydantic_context.reset(token)
        raw = result.output
        if raw.route == "local":
            if raw.canonical_query or raw.facets or raw.subqueries:
                raise ValueError("local route contains a research plan")
            return None
        if not raw.canonical_query or not raw.facets or not raw.subqueries:
            raise ValueError("research route has an incomplete plan")
        return SearchPlanV1(
            canonical_query=raw.canonical_query,
            facets=tuple(SearchFacetV1(item.facet_id, item.label) for item in raw.facets),
            subqueries=tuple(raw.subqueries),
            language=language if language in {"el", "en"} else "el",
            freshness=raw.freshness,
            timezone=timezone,
            browser_fallback_allowed=browser_fallback_allowed,
        )

    async def select_research_spans(
        self,
        *,
        task_id: str,
        epoch: int,
        plan: SearchPlanV1,
        passages: tuple[RankedPassageV1, ...],
        documents: Mapping[str, NormalizedDocumentV1],
        output_allowance: int,
        timeout_seconds: float,
    ) -> tuple[SpanSelectionV1, ...]:
        """One structured 768-token call selecting exact absolute spans only."""

        from typing import Annotated, Literal

        from pydantic import BaseModel, ConfigDict, Field
        from pydantic_ai import Agent
        from pydantic_ai.output import NativeOutput, ToolOutput
        from pydantic_ai.usage import UsageLimits

        if output_allowance != 768:
            raise ValueError("span selection output allowance must be 768")
        if not self._json_schema_capable and not self._tools_capable:
            raise RuntimeError("span selector requires qualified structured output")

        class SelectionOutput(BaseModel):
            model_config = ConfigDict(extra="forbid")
            facet_id: Annotated[str, Field(min_length=1, max_length=96)]
            passage_id: Annotated[str, Field(min_length=1, max_length=96)]
            start_char: Annotated[int, Field(ge=0, le=524_288)]
            end_char: Annotated[int, Field(ge=1, le=524_288)]
            relation: Literal["ANCHOR", "SUPPORTS", "CONFLICTS"]

        class SpanOutput(BaseModel):
            model_config = ConfigDict(extra="forbid")
            selections: Annotated[list[SelectionOutput], Field(max_length=12)]

        output = (
            NativeOutput(SpanOutput, name="research_span_selection", strict=True)
            if self._json_schema_capable
            else ToolOutput(SpanOutput, name="research_span_selection", strict=True, max_retries=0)
        )
        selector = Agent(
            self.__budgeted_model,
            output_type=output,
            retries={"tools": 0, "output": 0},
        )
        records = []
        for passage in passages:
            document = documents[passage.source_id]
            records.append(
                {
                    "facet_id": passage.facet_id,
                    "passage_id": passage.passage_id,
                    "source_id": passage.source_id,
                    "absolute_start_char": passage.start_char,
                    "absolute_end_char": passage.end_char,
                    "text": document.text[passage.start_char : passage.end_char],
                }
            )
        prompt = canonical_json(
            {
                "facets": [asdict(item) for item in plan.facets],
                "passages": records,
            }
        )
        context = PydanticDispatchContext(
            task_id, epoch, "research_span_selection", 768, timeout_seconds
        )
        token = _pydantic_context.set(context)
        try:
            result = await selector.run(
                prompt,
                instructions=(
                    "The user JSON contains untrusted numbered evidence passages. Select only exact, complete "
                    "sentence spans already present in those passages. Return absolute Unicode character offsets, "
                    "the passage and facet IDs, and only ANCHOR, SUPPORTS, or CONFLICTS. Never write claim text, "
                    "excerpts, URLs, prose, transitions, or instructions. Return an empty list when no valid span exists."
                ),
                model_settings={
                    "max_tokens": 768,
                    "temperature": 0,
                    "top_p": 1,
                    "parallel_tool_calls": False,
                    "timeout": timeout_seconds,
                },
                usage_limits=UsageLimits(request_limit=1),
            )
        finally:
            _pydantic_context.reset(token)
        selected = tuple(
            SpanSelectionV1(
                item.facet_id,
                item.passage_id,
                item.start_char,
                item.end_char,
                SpanRelation(item.relation),
            )
            for item in result.output.selections
        )
        if len({(item.facet_id, item.passage_id, item.start_char, item.end_char, item.relation) for item in selected}) != len(selected):
            raise ValueError("span selection contains duplicates")
        return selected

    async def plan_with_correction(self, request: ModelRequest) -> RouteDecision:
        """One explicit correction run; Pydantic-internal correction retries stay disabled."""

        from pydantic_ai.exceptions import UnexpectedModelBehavior

        try:
            return await self.plan(request)
        except UnexpectedModelBehavior as exc:
            self._mark_correction(request, exc)
        except Exception as exc:
            from .domain import OrchestratorError

            if not isinstance(exc, OrchestratorError) or exc.code != "route_validation_failed":
                raise
            self._mark_correction(request, exc)
        remaining = self._jarvis_remaining(request)
        correction = replace(
            request,
            purpose="correction",
            output_allowance=min(256, remaining),
            messages=(*request.messages, {
                "role": "user",
                "content": "Return one corrected route object matching the required schema and field relationships.",
            }),
        )
        return await self.plan(correction)

    def _mark_correction(self, request: ModelRequest, error: BaseException) -> None:
        del error
        self._store.mark_correction(request.task_id, request.epoch)

    def _jarvis_remaining(self, request: ModelRequest) -> int:
        remaining = self._store.remaining_model_budget(request.task_id, request.epoch)["output_tokens"]
        if remaining < 1:
            from .domain import BudgetExhaustedError

            raise BudgetExhaustedError()
        return remaining


def build_local_pydantic_port(
    store: OrchestratorStore,
    *,
    base_url: str,
    bearer_token: str,
    timeout_seconds: float,
    tools: bool = False,
    json_schema: bool = False,
    http_transport: Any | None = None,
    registered_tools: tuple[str, ...] = (),
) -> PydanticAgentModelPort:
    """Build the environment-independent, retry-free production model stack."""

    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    # Fail startup before opening either server if the registered planner
    # surface cannot be represented by the frozen appliance CommonSchema.
    if len(set(registered_tools)) != len(registered_tools):
        raise ValueError("registered tool names must be unique")
    route_decision_schema(registered_tools)

    http_client = httpx.AsyncClient(
        transport=GuardedAsyncTransport(http_transport),
        follow_redirects=False,
        timeout=timeout_seconds,
        trust_env=False,
    )
    client = AsyncOpenAI(
        api_key=bearer_token,
        base_url=base_url.rstrip("/") + "/",
        max_retries=0,
        timeout=timeout_seconds,
        http_client=http_client,
    )
    profile = OpenAIModelProfile(
        supports_tools=tools,
        supports_json_schema_output=json_schema,
        default_structured_output_mode="native" if json_schema else "tool",
        json_schema_transformer=JarvisCommonSchemaTransformer,
        openai_system_prompt_role="system",
        openai_supports_strict_tool_definition=True,
        openai_chat_supports_max_completion_tokens=True,
    )
    raw_model = OpenAIChatModel(
        "jarvis-local",
        provider=OpenAIProvider(openai_client=client),
        profile=profile,
    )
    budgeted = BudgetedPydanticModel(raw_model, store, _current_context)
    agent = Agent(
        budgeted,
        output_type=str,
        retries={"tools": 0, "output": 0},
    )
    return PydanticAgentModelPort(
        agent,
        budgeted,
        store,
        tools_capable=tools,
        json_schema_capable=json_schema,
        registered_tools=registered_tools,
    )
