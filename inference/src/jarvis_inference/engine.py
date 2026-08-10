from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, Protocol

from .contracts import NormalizedRequest


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    token_ids: tuple[int, ...]
    rendered_prompt: str
    request: NormalizedRequest


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    kind: str  # text, tool, final
    text: str = ""
    tool_name: str | None = None
    tool_arguments: str | None = None
    finish_reason: str | None = None
    completion_tokens: int | None = None


class Engine(Protocol):
    def ready(self) -> bool: ...
    def prepare(self, request: NormalizedRequest, deadline: float) -> PreparedPrompt: ...
    def generate(self, prompt: PreparedPrompt, cancel: threading.Event, deadline: float) -> Iterable[GenerationEvent]: ...
    def cleanup(self, deadline: float) -> None: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


class EngineFailure(RuntimeError):
    pass


class EngineProtocolFailure(EngineFailure):
    pass


class EngineTimeout(EngineFailure):
    pass

