"""Pinned Pydantic AI adapter boundary.

The dependency-free lifecycle does not import this module. When Pydantic AI
2.27.0 is installed, the wrapper below ensures that every Model interface call
has a persisted reservation; logical request hooks remain telemetry only.
"""

from __future__ import annotations

import contextvars
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from .common_schema import JarvisCommonSchemaTransformer, route_decision_schema
from .domain import ModelChunk, ModelRequest, ModelUsage, RouteDecision
from .history import canonical_messages_to_pydantic
from .model import DispatchPermit, GuardedAsyncTransport, _dispatch_reservation
from .store import OrchestratorStore

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
        from typing import Literal

        from pydantic import ConfigDict, create_model
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
            memory_query=(str, ...),
            research_query=(str, ...),
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
