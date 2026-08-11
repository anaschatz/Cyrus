from __future__ import annotations

from .contracts import NormalizedRequest
from .strict_json import canonical_text


def render_prompt(request: NormalizedRequest) -> str:
    """Render the appliance-owned, qualification-pinned chat template.

    This exact returned string is tokenized by the managed backend and is then
    passed unchanged to generation. Any renderer change changes the gateway
    artifact digest and therefore invalidates qualification.
    """
    parts: list[str] = []
    if request.tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    **({"description": tool.description} if tool.description is not None else {}),
                    "parameters": tool.schema,
                    "strict": True,
                },
            }
            for tool in request.tools
        ]
        parts.append("<|jarvis_tools|>\n" + canonical_text(tools) + "\n<|end|>\n")
        choice = request.tool_choice[1] if isinstance(request.tool_choice, tuple) else request.tool_choice
        parts.append(
            "<|jarvis_tool_policy|>\n"
            + canonical_text({"choice": choice, "output": {"name": "function name", "arguments": {}}})
            + "\n<|end|>\n"
        )
    if request.response_format.type == "json_object":
        parts.append("<|jarvis_response_format|>\n{\"type\":\"json_object\"}\n<|end|>\n")
    elif request.response_format.type == "json_schema":
        parts.append(
            "<|jarvis_response_format|>\n"
            + canonical_text({
                "type": "json_schema",
                "name": request.response_format.name,
                "schema": request.response_format.schema,
                "strict": True,
            })
            + "\n<|end|>\n"
        )
    for message in request.messages:
        parts.append(f"<|{message.role}|>\n")
        if message.tool_calls:
            call = message.tool_calls[0]
            parts.append(canonical_text({
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.value},
            }))
        else:
            parts.append(message.content or "")
            if message.tool_call_id is not None:
                parts.append("\n<|tool_call_id|>" + message.tool_call_id)
        parts.append("\n<|end|>\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def render_mlx_qwen3_prompt(request: NormalizedRequest) -> str:
    """Render the pinned Qwen3 MLX profile without executing model Jinja.

    This is an appliance-owned transcription of the qualified Qwen3 ChatML
    boundary. Tool definitions and structured-output constraints are rendered
    deterministically, while Outlines independently constrains the generated
    token sequence. A renderer change changes the gateway wheel hash and thus
    invalidates native qualification.
    """

    system_parts: list[str] = []
    message_start = 0
    if request.messages and request.messages[0].role == "system":
        system_parts.append(request.messages[0].content or "")
        message_start = 1

    if request.tools:
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    **({"description": tool.description} if tool.description is not None else {}),
                    "parameters": tool.schema,
                    "strict": True,
                },
            }
            for tool in request.tools
        ]
        choice = request.tool_choice[1] if isinstance(request.tool_choice, tuple) else request.tool_choice
        system_parts.append(
            "# Tools\n\nAvailable strict function definitions:\n<tools>\n"
            + "\n".join(canonical_text(tool) for tool in definitions)
            + "\n</tools>\n\n"
            + "Tool policy: " + choice + ". When calling a tool, output only one JSON object "
            + 'with exactly the keys "name" and "arguments". Do not add XML, Markdown, or prose.'
        )
    elif request.response_format.type == "json_object":
        system_parts.append("Output only one valid JSON object. Do not add Markdown or prose.")
    elif request.response_format.type == "json_schema":
        system_parts.append(
            "Output only one JSON object conforming exactly to this schema. Do not add Markdown or prose.\n"
            + canonical_text(request.response_format.schema)
        )

    parts: list[str] = []
    if system_parts:
        parts.append("<|im_start|>system\n" + "\n\n".join(system_parts) + "<|im_end|>\n")

    for message in request.messages[message_start:]:
        if message.role in {"system", "user"}:
            if message.content is None or message.tool_calls or message.tool_call_id is not None:
                raise ValueError("the qualified MLX Qwen3 profile received invalid message history")
            parts.append(f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n")
        elif message.role == "assistant":
            if message.tool_calls:
                call = message.tool_calls[0]
                payload = canonical_text({"name": call.name, "arguments": call.value})
                parts.append(f"<|im_start|>assistant\n<tool_call>\n{payload}\n</tool_call><|im_end|>\n")
            elif message.content is not None and message.tool_call_id is None:
                parts.append(f"<|im_start|>assistant\n{message.content}<|im_end|>\n")
            else:
                raise ValueError("the qualified MLX Qwen3 profile received invalid assistant history")
        elif message.role == "tool":
            if message.content is None or message.tool_call_id is None or message.tool_calls:
                raise ValueError("the qualified MLX Qwen3 profile received invalid tool history")
            parts.append(
                "<|im_start|>user\n<tool_response>\n"
                + message.content
                + "\n</tool_response><|im_end|>\n"
            )
        else:
            raise ValueError("the qualified MLX Qwen3 profile received an unsupported role")

    parts.append("<|im_start|>assistant\n")
    if not request.tools and request.response_format.type == "text":
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts)
