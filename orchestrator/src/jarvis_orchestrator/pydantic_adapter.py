"""Pinned Pydantic AI adapter boundary.

The dependency-free lifecycle does not import this module. When Pydantic AI
2.27.0 is installed, the wrapper below ensures that every Model interface call
has a persisted reservation; logical request hooks remain telemetry only.
"""

from __future__ import annotations

import contextvars
import asyncio
import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import asdict, dataclass
from dataclasses import replace
from typing import Any

from .common_schema import JarvisCommonSchemaTransformer, route_decision_schema
from .cloud_routing import ContextScope, OutputContract, TaskKind, TaskProfileV1
from .domain import (
    ModelChunk,
    ModelRequest,
    ModelUsage,
    OrchestratorError,
    RouteDecision,
    canonical_json,
)
from .history import canonical_messages_to_pydantic
from .model import (
    BudgetedModelPort,
    DispatchPermit,
    GuardedAsyncTransport,
    LocalInferenceModelPort,
    _dispatch_reservation,
)
from .store import OrchestratorStore
from .tool_contracts import ToolDefinitionV1
from .research import (
    NormalizedDocumentV1,
    RankedPassageV1,
    SearchFacetV1,
    SearchPlanV1,
    SpanRelation,
    SpanSelectionV1,
)

class BudgetedPydanticModel:
    """Mixin for a lazily constructed Pydantic ``WrapperModel``.

    The runtime subclass is created only by ``build_local_pydantic_port``.
    Importing the Orchestrator core or CLI therefore does not import the model
    SDK and still works in dependency-light operational environments.
    """

    def __init__(self, wrapped: Any, store: OrchestratorStore, context_provider: Any) -> None:
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
        except asyncio.CancelledError:
            with suppress(Exception):
                self._jarvis_store.reconcile_model(reservation_id, context.task_id, context.epoch, None)
            raise
        except OrchestratorError:
            with suppress(Exception):
                self._jarvis_store.reconcile_model(
                    reservation_id, context.task_id, context.epoch, None
                )
            raise
        except Exception as exc:
            # Pydantic AI, OpenAI and HTTPX use different exception classes for
            # the same unavailable/timeout/stream failure. Normalize them at
            # the authoritative physical-dispatch boundary.
            with suppress(Exception):
                self._jarvis_store.reconcile_model(
                    reservation_id, context.task_id, context.epoch, None
                )
            raise OrchestratorError(
                "backend_failure", status=503, retryable=True
            ) from exc
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


class _DirectLocalTextStreamPort:
    """Incremental, cancellation-safe final text stream for the local appliance."""

    def __init__(
        self,
        *,
        base_url: str,
        bearer_token: str,
        timeout_seconds: float,
        http_transport: Any | None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._bearer_token = bearer_token
        self._timeout_seconds = timeout_seconds
        self._http_transport = http_transport
        self._payload_builder = LocalInferenceModelPort(
            base_url=base_url,
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
        )

    async def stream(self, request: ModelRequest):
        import httpx

        payload = self._payload_builder._payload(request)
        timeout = self._timeout_seconds
        if request.timeout_seconds is not None:
            timeout = min(timeout, request.timeout_seconds)
        client = httpx.AsyncClient(
            transport=GuardedAsyncTransport(self._http_transport),
            follow_redirects=False,
            timeout=timeout,
            trust_env=False,
        )
        try:
            async with client.stream(
                "POST",
                self._url,
                content=canonical_json(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._bearer_token}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    parsed = json.loads(data)
                    usage_raw = parsed.get("usage")
                    usage = None
                    if isinstance(usage_raw, dict):
                        usage = ModelUsage(
                            usage_raw.get("completion_tokens"),
                            usage_raw.get("prompt_tokens"),
                        )
                    choices = parsed.get("choices") or []
                    text = ""
                    if choices:
                        text = choices[0].get("delta", {}).get("content") or ""
                    yield ModelChunk(text, usage)
        except asyncio.CancelledError:
            raise
        except OrchestratorError:
            raise
        except (httpx.HTTPError, json.JSONDecodeError, UnicodeError) as exc:
            raise OrchestratorError(
                "backend_failure", status=503, retryable=True
            ) from exc
        finally:
            await client.aclose()


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
        stream_runtime_factory: Any | None = None,
        final_stream_port: Any | None = None,
    ) -> None:
        self._agent = agent
        self.__budgeted_model = budgeted_model
        self._store = store
        self._tools_capable = tools_capable
        self._json_schema_capable = json_schema_capable
        self._registered_tools = frozenset(registered_tools)
        self._stream_runtime_factory = stream_runtime_factory
        self._final_stream_port = final_stream_port

    async def stream(self, request: ModelRequest):
        from pydantic_ai.usage import UsageLimits

        if self._final_stream_port is not None and request.purpose == "final":
            async for chunk in self._final_stream_port.stream(request):
                yield chunk
            return

        agent = self._agent
        stream_http_client = None
        if self._stream_runtime_factory is not None:
            agent, stream_http_client = self._stream_runtime_factory()
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
            async with agent.run_stream(
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
            if stream_http_client is not None:
                await stream_http_client.aclose()

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

    async def plan_tool_arguments(
        self,
        request: ModelRequest,
        *,
        definition: ToolDefinitionV1,
    ) -> Mapping[str, Any]:
        """Generate one strict argument object for one already-selected tool."""

        from enum import Enum
        from typing import Annotated

        from pydantic import ConfigDict, Field, create_model
        from pydantic_ai import Agent
        from pydantic_ai.output import NativeOutput, ToolOutput
        from pydantic_ai.usage import UsageLimits

        if definition.tool_id not in self._registered_tools:
            raise RuntimeError("tool is not registered in the local planner")

        counter = 0

        def build_type(schema: Mapping[str, Any], name: str) -> Any:
            nonlocal counter
            selected = schema.get("type")
            if selected == "string":
                values = schema.get("enum")
                if values is not None:
                    counter += 1
                    return Enum(
                        f"JarvisToolEnum{counter}",
                        {f"value_{index}": value for index, value in enumerate(values)},
                        type=str,
                    )
                return Annotated[
                    str,
                    Field(
                        min_length=schema.get("minLength"),
                        max_length=schema.get("maxLength"),
                    ),
                ]
            if selected == "integer":
                return Annotated[
                    int,
                    Field(ge=schema.get("minimum"), le=schema.get("maximum")),
                ]
            if selected == "boolean":
                return bool
            if selected == "array":
                item_type = build_type(schema["items"], f"{name}Item")
                return Annotated[
                    list[item_type],
                    Field(
                        min_length=schema.get("minItems"),
                        max_length=schema.get("maxItems"),
                    ),
                ]
            if selected == "object":
                fields = {
                    field_name: (build_type(field_schema, f"{name}{field_name.title()}"), ...)
                    for field_name, field_schema in schema["properties"].items()
                }
                return create_model(
                    name,
                    __config__=ConfigDict(extra="forbid", use_enum_values=True),
                    **fields,
                )
            raise RuntimeError("unsupported frozen tool schema type")

        model_type = build_type(definition.input_schema, "JarvisToolArguments")
        output = (
            NativeOutput(model_type, name="tool_arguments", strict=True)
            if self._json_schema_capable
            else ToolOutput(model_type, name="tool_arguments", strict=True, max_retries=0)
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
        return definition.validate_input(result.output.model_dump(mode="json"))

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

    try:
        import httpx
        from openai import AsyncOpenAI
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.models.wrapper import WrapperModel
        from pydantic_ai.profiles.openai import OpenAIModelProfile
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:  # pragma: no cover - exercised by dependency gate
        raise RuntimeError(
            "pydantic-ai-slim[openai]==2.27.0 and its locked dependencies are required"
        ) from exc

    # Fail startup before opening either server if the registered planner
    # surface cannot be represented by the frozen appliance CommonSchema.
    if len(set(registered_tools)) != len(registered_tools):
        raise ValueError("registered tool names must be unique")
    route_decision_schema(registered_tools)

    profile = OpenAIModelProfile(
        supports_tools=tools,
        supports_json_schema_output=json_schema,
        default_structured_output_mode="native" if json_schema else "tool",
        json_schema_transformer=JarvisCommonSchemaTransformer,
        openai_system_prompt_role="system",
        openai_supports_strict_tool_definition=True,
        openai_chat_supports_max_completion_tokens=True,
    )
    class _RuntimeBudgetedPydanticModel(BudgetedPydanticModel, WrapperModel):
        pass

    def make_runtime() -> tuple[Any, Any, Any]:
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
        raw_model = OpenAIChatModel(
            "jarvis-local",
            provider=OpenAIProvider(openai_client=client),
            profile=profile,
        )
        budgeted = _RuntimeBudgetedPydanticModel(raw_model, store, _current_context)
        agent = Agent(
            budgeted,
            output_type=str,
            retries={"tools": 0, "output": 0},
        )
        return agent, budgeted, http_client

    agent, budgeted, _shared_http_client = make_runtime()

    def make_stream_runtime() -> tuple[Any, Any]:
        stream_agent, _stream_budgeted, stream_http_client = make_runtime()
        return stream_agent, stream_http_client

    final_stream_port = BudgetedModelPort(
        store,
        _DirectLocalTextStreamPort(
            base_url=base_url,
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
            http_transport=http_transport,
        ),
    )
    return PydanticAgentModelPort(
        agent,
        budgeted,
        store,
        tools_capable=tools,
        json_schema_capable=json_schema,
        registered_tools=registered_tools,
        stream_runtime_factory=make_stream_runtime,
        final_stream_port=final_stream_port,
    )
