"""Current audit-only lifecycle gate; frozen v1-v3 remain reproducible."""

from __future__ import annotations

import re

from .anamnesis_shadow import (
    DETERMINISTIC_LIFECYCLE_GATE_V3_CONTRACT_SHA256,
    LifecycleCandidate, LifecycleDecision, LifecycleObservation, LifecycleOperation,
    _AMBIGUOUS_LIFECYCLE_PREFIXES, _V2_CREATE_PREFIXES, _V2_REPLACE_PREFIXES,
    _V2_REPLACE_VERBS_EN, _V2_REPLACE_VERBS_EL, _V2_REPLACE_MARKERS_EN,
    _V2_REPLACE_MARKERS_EL, _V2_CANCEL_PREFIXES, _V2_CANCEL_VERBS_EN,
    _V2_CANCEL_VERBS_EL, _V2_HEDGE_WORDS, _v3_normalized_text,
)
from .domain import canonical_sha256
from .memory import (
    MEMORY_NORMALIZER_VERSION, _EXPLICIT_REMEMBER_PREFIX, _EXPLICIT_FORGET_PREFIX,
    parse_explicit_memory_command,
)


DETERMINISTIC_LIFECYCLE_GATE_V4_CONTRACT_SHA256 = canonical_sha256({
    "name": "jarvis.deterministic-lifecycle-gate", "version": "4.0",
    "parent_gate_contract_sha256": DETERMINISTIC_LIFECYCLE_GATE_V3_CONTRACT_SHA256,
    "explicit_command_parser": {
        "normalizer": MEMORY_NORMALIZER_VERSION,
        "remember_pattern": _EXPLICIT_REMEMBER_PREFIX.pattern,
        "forget_pattern": _EXPLICIT_FORGET_PREFIX.pattern,
    },
    "prefix_matching": "whole-final-word; punctuation-ended-prefix-exact",
    "command_remainder": "at-least-one-alphanumeric-character",
    "hedges": "whole-Unicode-words-with-punctuation-boundaries",
    "markers": "English-whole-word; Greek-word-prefix",
    "target_policy": "zero=ignore,one=exact-index,multiple=fallback",
    "authoritative": False,
})
DETERMINISTIC_ONLY_LIFECYCLE_WRITER_V4_CONTRACT_SHA256 = canonical_sha256({
    "name": "jarvis.deterministic-only-lifecycle-writer", "version": "1.0",
    "gate_contract_sha256": DETERMINISTIC_LIFECYCLE_GATE_V4_CONTRACT_SHA256,
    "ambiguous_policy": "ignore", "model_calls": 0, "authoritative": False,
})


def _remainder(value: str, prefixes: tuple[str, ...]) -> str | None:
    for literal in prefixes:
        prefix = _v3_normalized_text(literal)
        if not value.startswith(prefix):
            continue
        remainder = value[len(prefix):]
        if prefix[-1].isalnum() and remainder and (remainder[0].isalnum() or remainder[0] == "_"):
            continue
        return remainder.strip()
    return None


def _words(value: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", value))


def _family(value: str, verbs: tuple[str, ...], markers: tuple[str, ...], *, stems: bool = False) -> str | None:
    remainder = _remainder(value, verbs)
    if remainder is None:
        return None
    words = _words(remainder)
    normalized = tuple(_v3_normalized_text(marker) for marker in markers)
    if any(word.startswith(marker) if stems else word == marker for word in words for marker in normalized):
        return remainder
    return None


class DeterministicLifecycleGateV4:
    @property
    def contract_sha256(self) -> str:
        return DETERMINISTIC_LIFECYCLE_GATE_V4_CONTRACT_SHA256

    def decide(self, observation: LifecycleObservation,
               active_memories: tuple[LifecycleCandidate, ...]) -> LifecycleDecision | None:
        text = _v3_normalized_text(observation.text)
        explicit = parse_explicit_memory_command(observation.text)
        operation = None
        remainder = None
        if explicit is not None:
            operation = LifecycleOperation.CREATE if explicit[0] == "remember" else LifecycleOperation.CANCEL
            remainder = explicit[1]
        else:
            for candidate, prefixes in (
                (LifecycleOperation.CREATE, _V2_CREATE_PREFIXES),
                (LifecycleOperation.REPLACE, _V2_REPLACE_PREFIXES),
                (LifecycleOperation.CANCEL, _V2_CANCEL_PREFIXES),
            ):
                remainder = _remainder(text, prefixes)
                if remainder is not None:
                    operation = candidate
                    break
            if operation is None:
                for candidate, verbs, markers, stems in (
                    (LifecycleOperation.REPLACE, _V2_REPLACE_VERBS_EN, _V2_REPLACE_MARKERS_EN, False),
                    (LifecycleOperation.REPLACE, _V2_REPLACE_VERBS_EL, _V2_REPLACE_MARKERS_EL, True),
                    (LifecycleOperation.CANCEL, _V2_CANCEL_VERBS_EN, ("memory",), False),
                    (LifecycleOperation.CANCEL, _V2_CANCEL_VERBS_EL, ("μνήμη",), True),
                ):
                    remainder = _family(text, verbs, markers, stems=stems)
                    if remainder is not None:
                        operation = candidate
                        break
        if operation is None:
            if _remainder(text, _AMBIGUOUS_LIFECYCLE_PREFIXES) is not None:
                return None
            return LifecycleDecision(LifecycleOperation.IGNORE)
        if (not any(character.isalnum() for character in remainder or "")
                or _words(text).intersection(_v3_normalized_text(word) for word in _V2_HEDGE_WORDS)):
            return None
        if operation is LifecycleOperation.CREATE:
            return LifecycleDecision(operation)
        if not active_memories:
            return LifecycleDecision(LifecycleOperation.IGNORE)
        if len(active_memories) == 1:
            return LifecycleDecision(operation, active_memories[0].index)
        return None


class DeterministicOnlyLifecycleWriterV4:
    @property
    def contract_sha256(self) -> str:
        return DETERMINISTIC_ONLY_LIFECYCLE_WRITER_V4_CONTRACT_SHA256

    async def decide(self, observation: LifecycleObservation,
                     active_memories: tuple[LifecycleCandidate, ...]) -> LifecycleDecision:
        decision = DeterministicLifecycleGateV4().decide(observation, active_memories)
        return decision or LifecycleDecision(LifecycleOperation.IGNORE)
