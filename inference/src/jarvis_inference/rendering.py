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

