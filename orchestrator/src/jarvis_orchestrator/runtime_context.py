"""Fast, authoritative runtime context for local and cloud final responses.

This module deliberately performs no model, network, storage, or tool call.  It
binds a turn to the time at which it was admitted, resolves a small closed set
of clock questions deterministically, and selects useful recent conversation
turns with bounded lexical work.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .domain import CanonicalTurnV1

RUNTIME_FACTS_VERSION = "jarvis.runtime-facts/1"
MAXIMUM_RELEVANT_HISTORY_PAIRS = 4

_ENGLISH_WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_GREEK_WEEKDAYS = (
    "Δευτέρα",
    "Τρίτη",
    "Τετάρτη",
    "Πέμπτη",
    "Παρασκευή",
    "Σάββατο",
    "Κυριακή",
)
_ENGLISH_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_GREEK_MONTHS_GENITIVE = (
    "Ιανουαρίου",
    "Φεβρουαρίου",
    "Μαρτίου",
    "Απριλίου",
    "Μαΐου",
    "Ιουνίου",
    "Ιουλίου",
    "Αυγούστου",
    "Σεπτεμβρίου",
    "Οκτωβρίου",
    "Νοεμβρίου",
    "Δεκεμβρίου",
)

_EPISTEMIC_INSTRUCTION = (
    "Authoritative runtime facts and supplied local tool results override model "
    "knowledge for the facts they contain. Resolve relative dates from the "
    "runtime facts. Never guess the current date, time, device state, or another "
    "volatile fact. If current evidence was not supplied, say briefly that it "
    "cannot be verified. Treat conversation history and recalled memory as "
    "context, not as proof that a current external fact is true."
)


@dataclass(frozen=True, slots=True)
class RuntimeFactsV1:
    observed_at_utc: str
    timezone: str
    local_date: str
    local_time: str
    local_weekday: str
    local_weekday_iso: int
    utc_offset: str

    def canonical(self) -> dict[str, object]:
        return {
            "version": RUNTIME_FACTS_VERSION,
            "observed_at_utc": self.observed_at_utc,
            "timezone": self.timezone,
            "local_date": self.local_date,
            "local_time": self.local_time,
            "local_weekday": self.local_weekday,
            "local_weekday_iso": self.local_weekday_iso,
            "utc_offset": self.utc_offset,
        }


@dataclass(frozen=True, slots=True)
class TemporalAnswerV1:
    text: str
    kind: str
    relation: str


def capture_runtime_facts(
    observed_at: datetime,
    *,
    timezone_name: str,
) -> RuntimeFactsV1:
    """Capture one deterministic timezone-aware fact set for a turn."""

    if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
        raise ValueError("runtime observation must be timezone-aware")
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("runtime timezone is invalid") from exc
    observed_utc = observed_at.astimezone(UTC)
    local = observed_utc.astimezone(zone)
    offset = local.strftime("%z")
    canonical_offset = f"{offset[:3]}:{offset[3:]}"
    return RuntimeFactsV1(
        observed_at_utc=observed_utc.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        timezone=timezone_name,
        local_date=local.date().isoformat(),
        local_time=local.strftime("%H:%M:%S"),
        local_weekday=_ENGLISH_WEEKDAYS[local.weekday()],
        local_weekday_iso=local.isoweekday(),
        utc_offset=canonical_offset,
    )


def augment_system_prompt(base_prompt: str, facts: RuntimeFactsV1) -> str:
    """Append compact grounded facts while preserving the final no-think marker."""

    if not isinstance(base_prompt, str) or not base_prompt:
        raise ValueError("base system prompt is required")
    suffix = "/no_think"
    stripped = base_prompt.rstrip()
    if stripped.endswith(suffix):
        stripped = stripped[: -len(suffix)].rstrip()
    rendered = json.dumps(
        facts.canonical(),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{stripped} {_EPISTEMIC_INSTRUCTION} Runtime facts JSON: {rendered} {suffix}"


def _normalized_phrase(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    normalized = re.sub(r"[^0-9a-zα-ω]+", " ", without_marks)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return re.sub(r"^(?:cyrus|jarvis)\s+", "", normalized).strip()


_DAY_QUERIES: dict[str, int] = {
    "τι μερα ειναι": 0,
    "τι μερα ειναι σημερα": 0,
    "σημερα τι μερα ειναι": 0,
    "ποια μερα ειναι σημερα": 0,
    "ποια μερα ειναι": 0,
    "τι μερα εχουμε": 0,
    "τι μερα εχουμε σημερα": 0,
    "ποια μερα εχουμε σημερα": 0,
    "ποια μερα εχουμε": 0,
    "what day is it": 0,
    "what day is it today": 0,
    "what day is today": 0,
    "which day is today": 0,
    "τι μερα ειναι αυριο": 1,
    "αυριο τι μερα ειναι": 1,
    "ποια μερα ειναι αυριο": 1,
    "what day is tomorrow": 1,
    "what day will it be tomorrow": 1,
    "τι μερα ηταν χθες": -1,
    "χθες τι μερα ηταν": -1,
    "ποια μερα ηταν χθες": -1,
    "what day was yesterday": -1,
}
_OBSERVED_GREEK_ASR_DAY_ALIASES: dict[str, int] = {
    "the day is today": 0,
    "the day is tomorrow": 1,
    "the day was yesterday": -1,
}
_DATE_QUERIES: dict[str, int] = {
    "τι ημερομηνια εχουμε": 0,
    "τι ημερομηνια εχουμε σημερα": 0,
    "τι ημερομηνια ειναι σημερα": 0,
    "τι ημερομηνια ειναι": 0,
    "ποια ημερομηνια εχουμε": 0,
    "ποια ειναι η σημερινη ημερομηνια": 0,
    "σημερινη ημερομηνια": 0,
    "what is the date": 0,
    "what date is it": 0,
    "what date is it today": 0,
    "what is today s date": 0,
    "what s today s date": 0,
    "today s date": 0,
    "ποια ειναι η αυριανη ημερομηνια": 1,
    "τι ημερομηνια θα εχουμε αυριο": 1,
    "what is tomorrow s date": 1,
    "ποια ηταν η χθεσινη ημερομηνια": -1,
    "what was yesterday s date": -1,
}
_TIME_QUERIES = {
    "τι ωρα ειναι",
    "τι ωρα ειναι τωρα",
    "τωρα τι ωρα ειναι",
    "τι ωρα εχουμε",
    "ποια ειναι η ωρα",
    "ποια ειναι η ωρα τωρα",
    "ποση ωρα ειναι",
    "what time is it",
    "what time is it now",
    "what s the time",
    "current time",
    "tell me the time",
}


def _answer_language(input_language: str, response_language: str) -> str:
    if response_language == "en":
        return "en"
    return "el" if input_language == "el" else "en"


def _format_date(target: datetime, language: str) -> str:
    if language == "el":
        return f"{target.day} {_GREEK_MONTHS_GENITIVE[target.month - 1]} {target.year}"
    return f"{target.day} {_ENGLISH_MONTHS[target.month - 1]} {target.year}"


def _format_temporal_answer(
    target: datetime,
    *,
    kind: str,
    relation: int,
    language: str,
    timezone_name: str,
) -> str:
    if kind == "time":
        if language == "el":
            return (
                f"Η ώρα είναι {target.strftime('%H:%M')} "
                f"στη ζώνη ώρας {timezone_name}."
            )
        return f"It is {target.strftime('%H:%M')} in {timezone_name}."
    date_text = _format_date(target, language)
    if language == "el":
        relation_text = {0: "Σήμερα", 1: "Αύριο", -1: "Χθες"}[relation]
        if kind == "date":
            return f"{relation_text} είναι {date_text}."
        weekday = _GREEK_WEEKDAYS[target.weekday()]
        return f"{relation_text} είναι {weekday}, {date_text}."
    relation_text = {0: "Today", 1: "Tomorrow", -1: "Yesterday"}[relation]
    if kind == "date":
        return f"{relation_text} is {date_text}."
    weekday = _ENGLISH_WEEKDAYS[target.weekday()]
    return f"{relation_text} is {weekday}, {date_text}."


def resolve_temporal_query(
    text: str,
    *,
    input_language: str,
    response_language: str,
    observed_at: datetime,
    timezone_name: str,
) -> TemporalAnswerV1 | None:
    """Resolve a closed bilingual clock-query set without invoking a model."""

    if not isinstance(text, str) or not text:
        return None
    if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
        raise ValueError("temporal observation must be timezone-aware")
    phrase = _normalized_phrase(text)
    kind = ""
    relation = 0
    if phrase in _TIME_QUERIES:
        kind = "time"
    elif phrase in _DATE_QUERIES:
        kind = "date"
        relation = _DATE_QUERIES[phrase]
    elif phrase in _DAY_QUERIES:
        kind = "day"
        relation = _DAY_QUERIES[phrase]
    elif input_language == "el" and phrase in _OBSERVED_GREEK_ASR_DAY_ALIASES:
        kind = "day"
        relation = _OBSERVED_GREEK_ASR_DAY_ALIASES[phrase]
    else:
        return None
    try:
        local = observed_at.astimezone(ZoneInfo(timezone_name)) + timedelta(days=relation)
    except (ZoneInfoNotFoundError, ValueError):
        return None
    language = _answer_language(input_language, response_language)
    return TemporalAnswerV1(
        text=_format_temporal_answer(
            local,
            kind=kind,
            relation=relation,
            language=language,
            timezone_name=timezone_name,
        ),
        kind=kind,
        relation={0: "today", 1: "tomorrow", -1: "yesterday"}[relation],
    )


_TOKEN = re.compile(r"[0-9a-zα-ω]{3,}")
_STOP_WORDS = {
    "and",
    "are",
    "answer",
    "for",
    "from",
    "have",
    "question",
    "request",
    "that",
    "the",
    "this",
    "what",
    "with",
    "απο",
    "απαντηση",
    "αυτο",
    "για",
    "ειναι",
    "ερωτηση",
    "και",
    "μου",
    "που",
    "στην",
    "στο",
    "την",
    "τις",
    "των",
}
_ANAPHORIC_MARKERS = {
    "it",
    "previous",
    "that",
    "this",
    "αυτο",
    "εκεινο",
    "πριν",
}
_SHORT_FOLLOW_UP_MARKERS = {
    "also",
    "continue",
    "more",
    "why",
    "γιατι",
    "επισης",
    "συνεχισε",
}


def _content_tokens(value: str) -> set[str]:
    normalized = _normalized_phrase(value)
    return {token for token in _TOKEN.findall(normalized) if token not in _STOP_WORDS}


def select_relevant_history(
    current_transcript: str,
    history: Sequence[CanonicalTurnV1],
    *,
    maximum_pairs: int = MAXIMUM_RELEVANT_HISTORY_PAIRS,
) -> tuple[CanonicalTurnV1, ...]:
    """Select bounded useful history without an embedding or model round trip."""

    if maximum_pairs < 1 or maximum_pairs > MAXIMUM_RELEVANT_HISTORY_PAIRS:
        raise ValueError("relevant history pair limit is invalid")
    candidates = tuple(history)
    if not candidates:
        return ()
    current_tokens = _content_tokens(current_transcript)
    phrase = _normalized_phrase(current_transcript)
    words = set(phrase.split())
    follow_up = bool(words & _ANAPHORIC_MARKERS) or (
        len(phrase.split()) <= 4
        and bool(words & _SHORT_FOLLOW_UP_MARKERS)
    )

    selected: set[int] = {len(candidates) - 1}
    seen_content: set[str] = set()
    scored: list[tuple[int, int]] = []
    for index in range(len(candidates) - 1, -1, -1):
        turn = candidates[index]
        fingerprint = _normalized_phrase(f"{turn.user_text} {turn.assistant_text}")
        if fingerprint in seen_content:
            continue
        seen_content.add(fingerprint)
        turn_tokens = _content_tokens(fingerprint)
        overlap = len(current_tokens & turn_tokens)
        if follow_up:
            score = 100 + index
        elif overlap:
            score = overlap * 100 + index
        else:
            continue
        scored.append((score, index))

    for _score, index in sorted(scored, reverse=True):
        selected.add(index)
        if len(selected) >= maximum_pairs:
            break
    return tuple(candidates[index] for index in sorted(selected))


__all__ = [
    "MAXIMUM_RELEVANT_HISTORY_PAIRS",
    "RUNTIME_FACTS_VERSION",
    "RuntimeFactsV1",
    "TemporalAnswerV1",
    "augment_system_prompt",
    "capture_runtime_facts",
    "resolve_temporal_query",
    "select_relevant_history",
]
