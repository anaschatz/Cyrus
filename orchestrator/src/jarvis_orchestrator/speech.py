"""Deterministic, chunk-boundary-independent speech unit assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .domain import nfc


TERMINATORS = frozenset({".", "!", "?", ";", "\u037e", "\u0387", "\u2026"})
_WHITESPACE = re.compile(r"\s+")


def resolve_locale(language: str, default: str = "el-GR") -> str:
    primary = language.strip().lower().split("-", 1)[0]
    if primary == "el":
        return "el-GR"
    if primary == "en":
        return "en-US"
    return default


def _speakable(value: str) -> str:
    return _WHITESPACE.sub(" ", nfc(value)).strip()


def _hard_split(value: str, maximum: int) -> list[str]:
    pending = value.strip()
    result: list[str] = []
    while len(pending) > maximum:
        split_at = max((index for index, char in enumerate(pending[: maximum + 1]) if char.isspace()), default=-1)
        if split_at <= 0:
            split_at = maximum
        piece = pending[:split_at].strip()
        if piece:
            result.append(piece)
        pending = pending[split_at:].strip()
    if pending:
        result.append(pending)
    return result


@dataclass(slots=True)
class SpeechSegmenterV1:
    maximum_characters: int = 240
    _buffer: str = ""

    def feed(self, fragment: str) -> tuple[str, ...]:
        if not fragment:
            return ()
        self._buffer += fragment
        completed: list[str] = []
        start = 0
        index = 0
        while index < len(self._buffer):
            char = self._buffer[index]
            paragraph = char == "\n" and index + 1 < len(self._buffer) and self._buffer[index + 1] == "\n"
            boundary = paragraph
            end = index + (2 if paragraph else 1)
            if char in TERMINATORS:
                if char == "." and index > 0 and index + 1 < len(self._buffer):
                    if self._buffer[index - 1].isascii() and self._buffer[index - 1].isdigit() and self._buffer[index + 1].isascii() and self._buffer[index + 1].isdigit():
                        index += 1
                        continue
                end = index + 1
                while end < len(self._buffer) and self._buffer[end] in TERMINATORS:
                    end += 1
                boundary = end < len(self._buffer) and self._buffer[end].isspace()
            if boundary:
                value = _speakable(self._buffer[start:end])
                if value:
                    completed.extend(_hard_split(value, self.maximum_characters))
                start = end
                while start < len(self._buffer) and self._buffer[start].isspace():
                    start += 1
                index = start
                continue
            index += 1
        self._buffer = self._buffer[start:]
        return tuple(completed)

    def finish(self) -> tuple[str, ...]:
        value = _speakable(self._buffer)
        self._buffer = ""
        if not value:
            return ()
        return tuple(_hard_split(value, self.maximum_characters))


def segment_text(value: str, maximum_characters: int = 240) -> tuple[str, ...]:
    assembler = SpeechSegmenterV1(maximum_characters)
    return assembler.feed(value + "\n\n") + assembler.finish()
