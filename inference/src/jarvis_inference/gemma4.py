from __future__ import annotations

import hashlib
import re
from decimal import Decimal
from typing import Any

from .contracts import NAME_RE, NormalizedRequest, ToolDefinition
from .limits import JSON_DEPTH
from .strict_json import canonical_text, parse_text


PROFILE_ID = "google-gemma4-12b-it-qat-q4_0-v1"
MODEL_REPOSITORY = "google/gemma-4-12B-it-qat-q4_0-gguf"
MODEL_REVISION = "29d097773436b69ff9feafd636ab4cf873786537"
MODEL_FILENAME = "gemma-4-12b-it-qat-q4_0.gguf"
MODEL_SIZE_BYTES = 6_975_879_296
MODEL_SHA256 = "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"

# Hugging Face exposes these values from the GGUF metadata at MODEL_REVISION.
# The model artifact hash above cryptographically binds the same embedded
# tokenizer.chat_template. The appliance never evaluates that Jinja template
# at serving time; this module is its reviewed, deterministic transcription.
EMBEDDED_TEMPLATE_SIZE_BYTES = 18_683
EMBEDDED_TEMPLATE_SHA256 = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
BOS_TOKEN = "<bos>"

RENDERER_ID = "cyrus.gemma4-appliance-renderer/1"
TOOL_PARSER_ID = "cyrus.gemma4-single-tool-parser/1"
_RENDERER_CONTRACT = (
    b"cyrus.gemma4-appliance-renderer/1\n"
    b"model=google/gemma-4-12B-it-qat-q4_0-gguf\n"
    b"revision=29d097773436b69ff9feafd636ab4cf873786537\n"
    b"embedded-chat-template-sha256=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4\n"
    b"bos=<bos>\nadd-generation-prompt=true\nenable-thinking=false\n"
    b"tool-choice=auto,none,required,named\n"
)
_TOOL_PARSER_CONTRACT = (
    b"cyrus.gemma4-single-tool-parser/1\n"
    b"envelope=<|tool_call>call:NAME{ARGS}<tool_call|>\n"
    b"string-delimiter=<|\"|>\n"
    b"single-call=true\nprose=reject\nduplicate-keys=reject\ntrailing-bytes=reject\n"
)
RENDERER_SHA256 = hashlib.sha256(_RENDERER_CONTRACT).hexdigest()
TOOL_PARSER_SHA256 = hashlib.sha256(_TOOL_PARSER_CONTRACT).hexdigest()

GEMMA4_PROMPT_PROFILE: dict[str, Any] = {
    "id": PROFILE_ID,
    "model_repository": MODEL_REPOSITORY,
    "model_revision": MODEL_REVISION,
    "model_artifact": {
        "filename": MODEL_FILENAME,
        "size_bytes": MODEL_SIZE_BYTES,
        "sha256": MODEL_SHA256,
    },
    "embedded_chat_template": {
        "metadata_key": "tokenizer.chat_template",
        "size_bytes": EMBEDDED_TEMPLATE_SIZE_BYTES,
        "sha256": EMBEDDED_TEMPLATE_SHA256,
    },
    "renderer": {"id": RENDERER_ID, "sha256": RENDERER_SHA256},
    "tool_parser": {"id": TOOL_PARSER_ID, "sha256": TOOL_PARSER_SHA256},
    "thinking": False,
}

_STRING_DELIMITER = '<|"|>'
_TOOL_CALL_PREFIX = "<|tool_call>call:"
_TOOL_CALL_SUFFIX = "<tool_call|>"
_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", re.ASCII)
_BARE_KEY_RE = re.compile(r"^[^\s{}\[\],:]+$", re.UNICODE)


class GemmaToolCallError(ValueError):
    """The managed backend did not emit one exact Gemma 4 tool envelope."""


def uses_gemma4_profile(manifest: Any) -> bool:
    return getattr(manifest, "prompt_profile", None) == GEMMA4_PROMPT_PROFILE


def _dictsort(value: dict[str, Any]) -> list[tuple[str, Any]]:
    # Jinja's dictsort defaults to case-insensitive key ordering and preserves
    # insertion order for case-fold ties.
    return sorted(value.items(), key=lambda item: item[0].casefold())


def _safe_delimited(value: str) -> str:
    if _STRING_DELIMITER in value:
        raise ValueError("Gemma string values cannot contain the model string delimiter")
    return _STRING_DELIMITER + value + _STRING_DELIMITER


def _format_argument(value: Any, *, escape_keys: bool = True, depth: int = 1) -> str:
    if depth > JSON_DEPTH:
        raise ValueError("Gemma argument nesting exceeds the appliance limit")
    if value is None:
        return "null"
    if isinstance(value, str):
        return _safe_delimited(value)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return canonical_text(value)
    if isinstance(value, list):
        return "[" + ",".join(
            _format_argument(item, escape_keys=escape_keys, depth=depth + 1)
            for item in value
        ) + "]"
    if isinstance(value, dict):
        fields: list[str] = []
        for key, child in _dictsort(value):
            if not isinstance(key, str):
                raise ValueError("Gemma object keys must be strings")
            rendered_key = _safe_delimited(key) if escape_keys else _bare_key(key)
            fields.append(
                rendered_key + ":" + _format_argument(
                    child, escape_keys=escape_keys, depth=depth + 1
                )
            )
        return "{" + ",".join(fields) + "}"
    raise ValueError("Gemma arguments must stay inside the strict JSON domain")


def _bare_key(value: str) -> str:
    if not value or _BARE_KEY_RE.fullmatch(value) is None:
        raise ValueError("Gemma tool keys must be unambiguous bare names")
    return value


def _format_item_schema(schema: dict[str, Any]) -> str:
    fields: list[str] = []
    for key, value in _dictsort(schema):
        if value is None:
            continue
        if key == "properties":
            if not isinstance(value, dict):
                raise ValueError("Gemma schema properties must be an object")
            fields.append(
                "properties:{"
                + _format_parameters(value)
                + "}"
            )
        elif key == "required":
            fields.append(
                "required:["
                + ",".join(_safe_delimited(item) for item in value)
                + "]"
            )
        elif key == "type":
            rendered = value.upper() if isinstance(value, str) else [item.upper() for item in value]
            fields.append("type:" + _format_argument(rendered))
        else:
            fields.append(key + ":" + _format_argument(value))
    return ",".join(fields)


def _format_property(name: str, schema: dict[str, Any]) -> str:
    _bare_key(name)
    fields: list[str] = []
    description = schema.get("description")
    if description:
        fields.append("description:" + _safe_delimited(description))
    kind = schema["type"]
    if kind.upper() == "STRING" and schema.get("enum"):
        fields.append("enum:" + _format_argument(schema["enum"]))
    elif kind.upper() == "ARRAY" and isinstance(schema.get("items"), dict) and schema["items"]:
        fields.append("items:{" + _format_item_schema(schema["items"]) + "}")
    if schema.get("nullable"):
        fields.append("nullable:true")
    if kind.upper() == "OBJECT":
        properties = schema.get("properties")
        if isinstance(properties, dict):
            fields.append(
                "properties:{" + _format_parameters(properties) + "}"
            )
        required = schema.get("required", [])
        if required:
            fields.append("required:[" + ",".join(_safe_delimited(item) for item in required) + "]")
    fields.append("type:" + _safe_delimited(kind.upper()))
    return name + ":{" + ",".join(fields) + "}"


def _format_parameters(properties: dict[str, Any]) -> str:
    return ",".join(_format_property(name, schema) for name, schema in _dictsort(properties))


def _format_tool(tool: ToolDefinition) -> str:
    schema = tool.schema
    fields = ["description:" + _safe_delimited(tool.description or "")]
    parameters: list[str] = []
    if schema.get("properties"):
        parameters.append("properties:{" + _format_parameters(schema["properties"]) + "}")
    if schema.get("required"):
        parameters.append(
            "required:[" + ",".join(_safe_delimited(item) for item in schema["required"]) + "]"
        )
    if schema.get("type"):
        parameters.append("type:" + _safe_delimited(schema["type"].upper()))
    if parameters:
        fields.append("parameters:{" + ",".join(parameters) + "}")
    return (
        "<|tool>declaration:"
        + tool.name
        + "{"
        + ",".join(fields)
        + "}<tool|>"
    )


def _tool_policy(request: NormalizedRequest) -> tuple[tuple[ToolDefinition, ...], str | None]:
    if not request.tools:
        return (), None
    choice = request.tool_choice
    if choice == "auto":
        return request.tools, None
    if choice == "none":
        return (), "Do not call a function in this turn; answer directly."
    if choice == "required":
        return request.tools, "Call exactly one declared function in this turn."
    return request.tools, f'Call only the declared function "{choice[1]}" in this turn.'


def _response_policy(request: NormalizedRequest) -> str | None:
    response = request.response_format
    if response.type == "json_object":
        return "Return exactly one JSON object. Do not use Markdown or add prose."
    if response.type == "json_schema":
        return (
            "Return exactly one JSON object that validates against this JSON Schema. "
            "Do not use Markdown or add prose.\n" + canonical_text(response.schema)
        )
    return None


def _strip_thinking(text: str) -> str:
    result = ""
    for part in text.split("<channel|>"):
        result += part.split("<|channel>", 1)[0] if "<|channel>" in part else part
    return result.strip()


def _format_historical_call(name: str, arguments: dict[str, Any]) -> str:
    fields = ",".join(
        _bare_key(key) + ":" + _format_argument(value, escape_keys=False)
        for key, value in _dictsort(arguments)
    )
    return _TOOL_CALL_PREFIX + name + "{" + fields + "}" + _TOOL_CALL_SUFFIX


def _format_tool_response(name: str, content: str) -> str:
    return (
        "<|tool_response>response:"
        + name
        + "{value:"
        + _format_argument(content, escape_keys=False)
        + "}<tool_response|>"
    )


def render_gemma4_prompt(request: NormalizedRequest) -> str:
    """Render the exact pinned Gemma 4 text protocol without runtime Jinja."""

    tools, tool_policy = _tool_policy(request)
    policies = [item for item in (tool_policy, _response_policy(request)) if item]
    messages = request.messages
    start = 0
    system_content: str | None = None
    if messages and messages[0].role == "system":
        system_content = (messages[0].content or "").strip()
        start = 1
    if policies:
        policy = "\n\n".join(policies)
        system_content = system_content + "\n\n" + policy if system_content else policy

    parts = [BOS_TOKEN]
    if system_content is not None or tools:
        parts.append("<|turn>system\n")
        if system_content is not None:
            parts.append(system_content)
        parts.extend(_format_tool(tool) for tool in tools)
        parts.append("<turn|>\n")

    loop_messages = messages[start:]
    previous_non_tool_role: str | None = None
    last_message_type: str | None = None
    for index, message in enumerate(loop_messages):
        if message.role == "tool":
            continue
        last_message_type = None
        role = "model" if message.role == "assistant" else message.role
        continue_model_turn = role == "model" and previous_non_tool_role == "assistant"
        if not continue_model_turn:
            parts.append("<|turn>" + role + "\n")

        has_tool_response = False
        if message.tool_calls:
            call = message.tool_calls[0]
            parts.append(_format_historical_call(call.name, call.value))
            last_message_type = "tool_call"
            if index + 1 < len(loop_messages):
                response = loop_messages[index + 1]
                if response.role == "tool" and response.tool_call_id == call.id:
                    parts.append(_format_tool_response(call.name, response.content or ""))
                    last_message_type = "tool_response"
                    has_tool_response = True

        content = ""
        if message.content is not None:
            content = _strip_thinking(message.content) if role == "model" else message.content.strip()
            parts.append(content)

        next_non_tool_role: str | None = None
        for following in loop_messages[index + 1:]:
            if following.role != "tool":
                next_non_tool_role = following.role
                break
        continues_into_next = (
            role == "model"
            and next_non_tool_role == "assistant"
            and (not message.tool_calls or has_tool_response)
        )
        if last_message_type == "tool_call" and not has_tool_response:
            parts.append("<|tool_response>")
        elif continues_into_next:
            pass
        elif not (has_tool_response and not content and next_non_tool_role is None):
            parts.append("<turn|>\n")
        previous_non_tool_role = message.role

    if last_message_type not in {"tool_response", "tool_call"}:
        parts.append("<|turn>model\n<|channel>thought\n<channel|>")
    return "".join(parts)


class _ArgumentParser:
    def __init__(self, text: str):
        self.text = text
        self.position = 0

    def parse_object(self, depth: int = 1) -> dict[str, Any]:
        if depth > JSON_DEPTH:
            raise GemmaToolCallError("Gemma tool arguments exceed the nesting limit")
        self._expect("{")
        result: dict[str, Any] = {}
        if self._consume("}"):
            return result
        while True:
            colon = self.text.find(":", self.position)
            if colon < 0:
                raise GemmaToolCallError("Gemma tool argument key has no value")
            key = self.text[self.position:colon]
            if _BARE_KEY_RE.fullmatch(key) is None or key in result:
                raise GemmaToolCallError("Gemma tool argument key is invalid or duplicated")
            self.position = colon + 1
            result[key] = self.parse_value(depth + 1)
            if self._consume("}"):
                return result
            self._expect(",")

    def parse_value(self, depth: int) -> Any:
        if depth > JSON_DEPTH:
            raise GemmaToolCallError("Gemma tool arguments exceed the nesting limit")
        if self.text.startswith(_STRING_DELIMITER, self.position):
            self.position += len(_STRING_DELIMITER)
            end = self.text.find(_STRING_DELIMITER, self.position)
            if end < 0:
                raise GemmaToolCallError("Gemma tool string is unterminated")
            value = self.text[self.position:end]
            self.position = end + len(_STRING_DELIMITER)
            return value
        if self.text.startswith("true", self.position):
            self.position += 4
            return True
        if self.text.startswith("false", self.position):
            self.position += 5
            return False
        if self.text.startswith("null", self.position):
            self.position += 4
            return None
        if self.text.startswith("{", self.position):
            return self.parse_object(depth)
        if self._consume("["):
            values: list[Any] = []
            if self._consume("]"):
                return values
            while True:
                values.append(self.parse_value(depth + 1))
                if self._consume("]"):
                    return values
                self._expect(",")
        match = _NUMBER_RE.match(self.text, self.position)
        if match is None:
            raise GemmaToolCallError("Gemma tool argument value is invalid")
        self.position = match.end()
        try:
            return parse_text(match.group(0), nested=True)
        except Exception as exc:
            raise GemmaToolCallError("Gemma tool number is outside the strict JSON domain") from exc

    def _consume(self, value: str) -> bool:
        if self.text.startswith(value, self.position):
            self.position += len(value)
            return True
        return False

    def _expect(self, value: str) -> None:
        if not self._consume(value):
            raise GemmaToolCallError("Gemma tool arguments have invalid punctuation")


def contains_gemma4_tool_marker(text: str) -> bool:
    return "<|tool_call" in text or "<tool_call|>" in text


def parse_gemma4_tool_call(text: str) -> tuple[str, str]:
    """Parse exactly one native Gemma call, rejecting prose and extra calls."""

    if not text.startswith(_TOOL_CALL_PREFIX) or not text.endswith(_TOOL_CALL_SUFFIX):
        raise GemmaToolCallError("Gemma tool output is not one complete envelope")
    body = text[len(_TOOL_CALL_PREFIX):-len(_TOOL_CALL_SUFFIX)]
    brace = body.find("{")
    if brace <= 0:
        raise GemmaToolCallError("Gemma tool output has no function arguments")
    name = body[:brace]
    if NAME_RE.fullmatch(name) is None:
        raise GemmaToolCallError("Gemma tool output has an invalid function name")
    parser = _ArgumentParser(body[brace:])
    arguments = parser.parse_object()
    if parser.position != len(parser.text):
        raise GemmaToolCallError("Gemma tool output contains trailing bytes or another call")
    return name, canonical_text(arguments)
