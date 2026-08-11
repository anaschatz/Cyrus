"""Closed, deterministic admission gate for explicit reminder requests.

This module decides only whether the dedicated local schedule compiler may see
the current transcript.  It never parses time, selects a database target, or
creates authority.  Those responsibilities remain with schedule-proposal/1,
the approval arbiter, and SchedulerStore respectively.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


SCHEDULE_ADMISSION_CONTRACT = "schedule-natural-language-admission/1"


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value).casefold()
    without_marks = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_marks.split())


_CREATE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:θυμισε|υπενθυμισε)\s+(?:μου|με)\b",
        r"\b(?:βαλε|ορισε|δημιουργησε|φτιαξε)(?:\s+μου)?\s+(?:μια\s+|ενα\s+)?(?:υπενθυμιση|χρονομετρο|ξυπνητηρι)\b",
        r"\bremind\s+me\b",
        r"\b(?:set|create|add)\s+(?:an?\s+)?(?:reminder|timer|alarm)\b",
    )
)


@dataclass(frozen=True, slots=True)
class ScheduleAdmission:
    admitted: bool
    operation: str
    reason_code: str


def classify_schedule_request(text: str) -> ScheduleAdmission:
    """Admit only explicit create intent; cancel/snooze require selected IDs."""

    if not isinstance(text, str) or not text.strip():
        return ScheduleAdmission(False, "NONE", "empty_request")
    folded = _fold(text)
    if any(pattern.search(folded) for pattern in _CREATE_PATTERNS):
        return ScheduleAdmission(True, "CREATE_JOB", "explicit_create_intent")
    return ScheduleAdmission(False, "NONE", "no_explicit_schedule_intent")
