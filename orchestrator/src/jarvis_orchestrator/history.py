"""Canonical-history adapters; framework messages are never durable authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def canonical_messages_to_pydantic(messages: Sequence[Mapping[str, str]]) -> tuple[list[Any], str]:
    """Convert completed pairs and return the final user prompt separately."""

    try:
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    except ImportError as exc:  # pragma: no cover - production dependency gate
        raise RuntimeError("pydantic-ai-slim[openai]==2.27.0 is required") from exc
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("canonical messages must end with a user prompt")
    history: list[Any] = []
    for message in messages[:-1]:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("canonical message content must be text")
        if role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content)]))
        elif role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content)]))
        else:
            raise ValueError("canonical history role is invalid")
    return history, messages[-1]["content"]
