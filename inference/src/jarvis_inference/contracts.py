from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Any

from .errors import ApplianceError, backend_error, invalid, unsupported
from .limits import (
    ARGUMENTS_BYTES,
    CONVERSATIONAL_BYTES,
    DESCRIPTION_BYTES,
    MESSAGES,
    MESSAGE_BYTES,
    OUTPUT_TOKENS,
    STOP_BYTES,
    STOP_COUNT,
    TOOLS,
    TOOLS_PAYLOAD_BYTES,
)
from .schema_subset import normalize_schema, validate_instance
from .strict_json import canonical_bytes, canonical_text, parse_text

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$", re.ASCII)
CALL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$", re.ASCII)
ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", re.ASCII)


@dataclass(frozen=True, slots=True)
class Capabilities:
    text: bool = True
    stream: bool = True
    tools: bool = False
    json_object: bool = False
    json_schema: bool = False


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str | None
    schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    type: str
    name: str | None = None
    description: str | None = None
    schema: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class NormalizedRequest:
    model: str
    messages: tuple[Message, ...]
    max_completion_tokens: int
    stream: bool
    include_usage: bool
    temperature: float
    top_p: float
    stop: tuple[str, ...]
    tools: tuple[ToolDefinition, ...]
    tool_choice: str | tuple[str, str]
    response_format: ResponseFormat


TOP_LEVEL = {
    "model", "messages", "max_completion_tokens", "max_tokens", "stream",
    "stream_options", "n", "parallel_tool_calls", "temperature", "top_p",
    "stop", "tools", "tool_choice", "response_format",
}


def _exact(obj: Any, required: set[str], optional: set[str], parameter: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise invalid("invalid_parameter", "The parameter has an invalid object shape.", parameter)
    missing = required - set(obj)
    unknown = set(obj) - required - optional
    if missing:
        raise invalid("missing_required_parameter", "A required nested parameter is missing.", parameter)
    if unknown:
        raise unsupported(parameter + "." + sorted(unknown)[0])
    return obj


def _string(value: Any, parameter: str, limit: int = MESSAGE_BYTES) -> str:
    if not isinstance(value, str):
        raise invalid("invalid_parameter", "The parameter must be a string.", parameter)
    if len(value.encode("utf-8")) > limit:
        raise invalid("parameter_too_large", "The parameter exceeds its size limit.", parameter)
    return value


def _number(value: Any, parameter: str, low: float, high: float, *, low_open: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise invalid("invalid_parameter", "The parameter must be a number.", parameter)
    result = float(value)
    if not math.isfinite(result):
        raise invalid("invalid_parameter", "The numeric parameter must be finite.", parameter)
    if (result <= low if low_open else result < low) or result > high:
        raise invalid("invalid_parameter", "The numeric parameter is outside its supported range.", parameter)
    return result


def _tools(value: Any, capabilities: Capabilities) -> tuple[ToolDefinition, ...]:
    if value is None:
        return ()
    if not capabilities.tools:
        raise unsupported("tools")
    if not isinstance(value, list) or not value or len(value) > TOOLS:
        raise invalid("invalid_parameter", "Tools must be a non-empty bounded array.", "tools")
    if len(canonical_bytes(value)) > TOOLS_PAYLOAD_BYTES:
        raise invalid("parameter_too_large", "The tools payload exceeds its size limit.", "tools")
    result: list[ToolDefinition] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        tool = _exact(raw, {"type", "function"}, set(), f"tools.{index}")
        if tool["type"] != "function":
            raise unsupported(f"tools.{index}.type")
        function = _exact(
            tool["function"],
            {"name", "parameters", "strict"},
            {"description"},
            f"tools.{index}.function",
        )
        name = function["name"]
        if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name in seen:
            raise invalid("invalid_function_name", "Function names must be unique valid ASCII identifiers.", f"tools.{index}.function.name")
        if function["strict"] is not True:
            raise invalid("invalid_parameter", "Strict function tools require strict=true.", f"tools.{index}.function.strict")
        description = function.get("description")
        if description is not None:
            description = _string(description, f"tools.{index}.function.description", DESCRIPTION_BYTES)
        schema = normalize_schema(function["parameters"])
        seen.add(name)
        result.append(ToolDefinition(name, description, schema))
    return tuple(result)


def _response_format(value: Any, capabilities: Capabilities) -> ResponseFormat:
    if value is None:
        return ResponseFormat("text")
    outer = _exact(value, {"type"}, {"json_schema"}, "response_format")
    kind = outer["type"]
    if kind == "text":
        if set(outer) != {"type"}:
            raise unsupported("response_format.json_schema")
        return ResponseFormat("text")
    if kind == "json_object":
        if not capabilities.json_object:
            raise unsupported("response_format")
        if set(outer) != {"type"}:
            raise unsupported("response_format.json_schema")
        return ResponseFormat("json_object")
    if kind != "json_schema" or not capabilities.json_schema:
        raise unsupported("response_format.type")
    nested = _exact(
        outer.get("json_schema"), {"name", "schema", "strict"}, {"description"}, "response_format.json_schema"
    )
    name = nested["name"]
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise invalid("invalid_parameter", "The response format name is invalid.", "response_format.json_schema.name")
    if nested["strict"] is not True:
        raise invalid("invalid_parameter", "json_schema requires strict=true.", "response_format.json_schema.strict")
    description = nested.get("description")
    if description is not None:
        description = _string(description, "response_format.json_schema.description", DESCRIPTION_BYTES)
    return ResponseFormat("json_schema", name, description, normalize_schema(nested["schema"]))


def _tool_choice(value: Any, tools: tuple[ToolDefinition, ...]) -> str | tuple[str, str]:
    if value is None:
        return "auto" if tools else "none"
    if isinstance(value, str):
        if value not in {"none", "auto", "required"}:
            raise invalid("invalid_parameter", "The tool choice is invalid.", "tool_choice")
        if not tools and value != "none":
            raise invalid("invalid_parameter", "A tool choice requires declared tools.", "tool_choice")
        return value
    choice = _exact(value, {"type", "function"}, set(), "tool_choice")
    if choice["type"] != "function":
        raise unsupported("tool_choice.type")
    function = _exact(choice["function"], {"name"}, set(), "tool_choice.function")
    name = function["name"]
    if not isinstance(name, str) or name not in {tool.name for tool in tools}:
        raise invalid("invalid_parameter", "The named tool choice is not declared.", "tool_choice.function.name")
    return ("named", name)


def _historical_call(raw: Any, index: int, tools: dict[str, ToolDefinition]) -> ToolCall:
    call = _exact(raw, {"id", "type", "function"}, set(), f"messages.{index}.tool_calls.0")
    call_id = call["id"]
    if not isinstance(call_id, str) or not CALL_ID_RE.fullmatch(call_id):
        raise invalid("invalid_tool_call_id", "The tool call identifier is invalid.", f"messages.{index}.tool_calls.0.id")
    if call["type"] != "function":
        raise unsupported(f"messages.{index}.tool_calls.0.type")
    fn = _exact(call["function"], {"name", "arguments"}, set(), f"messages.{index}.tool_calls.0.function")
    name = fn["name"]
    if not isinstance(name, str) or name not in tools:
        raise invalid("invalid_parameter", "A historical tool call references an undeclared function.", f"messages.{index}.tool_calls.0.function.name")
    arguments = _string(fn["arguments"], f"messages.{index}.tool_calls.0.function.arguments", ARGUMENTS_BYTES)
    try:
        value = parse_text(arguments, nested=True)
    except RecursionError as exc:
        raise invalid("invalid_tool_arguments", "Function arguments exceed the nesting limit.") from exc
    if not isinstance(value, dict):
        raise invalid("invalid_tool_arguments", "Function arguments must encode a JSON object.", f"messages.{index}.tool_calls.0.function.arguments")
    validate_instance(value, tools[name].schema)
    return ToolCall(call_id, name, canonical_text(value), value)


def _messages(value: Any, tools: tuple[ToolDefinition, ...]) -> tuple[Message, ...]:
    if not isinstance(value, list) or not value or len(value) > MESSAGES:
        raise invalid("invalid_parameter", "Messages must be a non-empty bounded array.", "messages")
    tool_map = {tool.name: tool for tool in tools}
    seen_ids: set[str] = set()
    unresolved: str | None = None
    total = 0
    result: list[Message] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or "role" not in raw:
            raise invalid("invalid_parameter", "Every message requires a role.", f"messages.{index}")
        role = raw["role"]
        if role == "developer":
            raise unsupported(f"messages.{index}.role")
        if not isinstance(role, str) or role not in {"system", "user", "assistant", "tool"}:
            raise unsupported(f"messages.{index}.role")
        if unresolved is not None and role != "tool":
            raise invalid("invalid_tool_history", "An unresolved tool call must be followed by its tool result.", f"messages.{index}")
        if role in {"system", "user"}:
            message = _exact(raw, {"role", "content"}, set(), f"messages.{index}")
            content = _string(message["content"], f"messages.{index}.content")
            item = Message(role, content)
        elif role == "assistant":
            message = _exact(raw, {"role", "content"}, {"tool_calls"}, f"messages.{index}")
            calls = message.get("tool_calls")
            if calls is None:
                content = _string(message["content"], f"messages.{index}.content")
                item = Message(role, content)
            else:
                if message["content"] is not None or not isinstance(calls, list) or len(calls) != 1:
                    raise invalid("invalid_tool_history", "Assistant tool messages require null content and exactly one tool call.", f"messages.{index}")
                call = _historical_call(calls[0], index, tool_map)
                if call.id in seen_ids:
                    raise invalid("invalid_tool_history", "Tool call identifiers must be unique.", f"messages.{index}.tool_calls.0.id")
                seen_ids.add(call.id)
                unresolved = call.id
                total += len(call.arguments.encode())
                item = Message(role, None, (call,))
        else:
            message = _exact(raw, {"role", "content", "tool_call_id"}, set(), f"messages.{index}")
            content = _string(message["content"], f"messages.{index}.content")
            call_id = message["tool_call_id"]
            if not isinstance(call_id, str) or not CALL_ID_RE.fullmatch(call_id) or call_id != unresolved:
                raise invalid("invalid_tool_history", "The tool result does not resolve the outstanding tool call.", f"messages.{index}.tool_call_id")
            unresolved = None
            item = Message(role, content, tool_call_id=call_id)
        total += len((item.content or "").encode())
        if total > CONVERSATIONAL_BYTES:
            raise invalid("parameter_too_large", "The conversation exceeds its size limit.", "messages")
        result.append(item)
    if unresolved is not None:
        raise invalid("invalid_tool_history", "The final historical tool call is unresolved.", "messages")
    return tuple(result)


def normalize_request(raw: dict[str, Any], *, alias: str, capabilities: Capabilities, context_window: int) -> NormalizedRequest:
    unknown = set(raw) - TOP_LEVEL
    if unknown:
        raise unsupported(sorted(unknown)[0])
    if "model" not in raw or "messages" not in raw:
        raise invalid("missing_required_parameter", "model and messages are required.")
    if not isinstance(raw["model"], str):
        raise invalid("invalid_parameter", "model must be a string.", "model")
    if raw["model"] != alias:
        raise ApplianceError(404, "model_not_found", "The requested logical model alias does not exist.", "model")
    if "max_tokens" in raw and "max_completion_tokens" in raw:
        raise invalid("invalid_parameter", "max_tokens and max_completion_tokens cannot be combined.", "max_tokens")
    maximum = raw.get("max_completion_tokens", raw.get("max_tokens", min(1024, context_window)))
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= min(OUTPUT_TOKENS, context_window):
        raise invalid("invalid_parameter", "max_completion_tokens is outside the supported range.", "max_completion_tokens")
    stream = raw.get("stream", False)
    if not isinstance(stream, bool) or (stream and not capabilities.stream):
        raise unsupported("stream")
    stream_options = raw.get("stream_options")
    include_usage = False
    if stream_options is not None:
        if not stream:
            raise invalid("invalid_parameter", "stream_options requires stream=true.", "stream_options")
        options = _exact(stream_options, {"include_usage"}, set(), "stream_options")
        if not isinstance(options["include_usage"], bool):
            raise invalid("invalid_parameter", "include_usage must be boolean.", "stream_options.include_usage")
        include_usage = options["include_usage"]
    if "n" in raw and (type(raw["n"]) is not int or raw["n"] != 1):
        raise unsupported("n")
    if raw.get("parallel_tool_calls", False) is not False:
        raise unsupported("parallel_tool_calls")
    temperature = _number(raw.get("temperature", 1), "temperature", 0, 2)
    top_p = _number(raw.get("top_p", 1), "top_p", 0, 1, low_open=True)
    stop_raw = raw.get("stop", [])
    if isinstance(stop_raw, str):
        stop_raw = [stop_raw]
    if not isinstance(stop_raw, list) or len(stop_raw) > STOP_COUNT:
        raise invalid("invalid_parameter", "stop must contain at most four strings.", "stop")
    stop = tuple(_string(item, "stop", STOP_BYTES) for item in stop_raw)
    if any(not item for item in stop):
        raise invalid("invalid_parameter", "Stop strings cannot be empty.", "stop")
    tools = _tools(raw.get("tools"), capabilities)
    response_format = _response_format(raw.get("response_format"), capabilities)
    if tools and response_format.type != "text":
        raise invalid("invalid_parameter", "Tools and structured response formats are mutually exclusive.")
    if response_format.type != "text" and stream:
        raise unsupported("stream")
    tool_choice = _tool_choice(raw.get("tool_choice"), tools)
    messages = _messages(raw["messages"], tools)
    return NormalizedRequest(alias, messages, maximum, stream, include_usage, temperature, top_p, stop, tools, tool_choice, response_format)


def validate_generated_tool(name: str, arguments: str, request: NormalizedRequest) -> str:
    definitions = {tool.name: tool for tool in request.tools}
    if not isinstance(name, str) or name not in definitions or not isinstance(arguments, str):
        raise backend_error()
    if len(arguments.encode()) > ARGUMENTS_BYTES:
        raise backend_error()
    try:
        value = parse_text(arguments, nested=True)
        if not isinstance(value, dict):
            raise backend_error()
        validate_instance(value, definitions[name].schema)
    except (ApplianceError, RecursionError) as exc:
        if isinstance(exc, ApplianceError) and exc.status >= 500:
            raise
        raise backend_error() from exc
    choice = request.tool_choice
    if choice == "none" or (isinstance(choice, tuple) and name != choice[1]):
        raise backend_error()
    return canonical_text(value)


def validate_structured_content(content: str, request: NormalizedRequest) -> str:
    if request.response_format.type == "text":
        return content
    try:
        value = parse_text(content, nested=True)
        if not isinstance(value, dict):
            raise backend_error()
        if request.response_format.schema is not None:
            validate_instance(value, request.response_format.schema)
    except (ApplianceError, RecursionError) as exc:
        if isinstance(exc, ApplianceError) and exc.status >= 500:
            raise
        raise backend_error() from exc
    return canonical_text(value)
