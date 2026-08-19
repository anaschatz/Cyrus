"""Framework-independent memory contracts for active recall and shadow evaluation."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class MemoryOperation(StrEnum):
    IGNORE = "ignore"
    UPSERT = "upsert"
    CANCEL = "cancel"


class ExplicitMemoryCommandStatus(StrEnum):
    """Closed committed outcomes for deterministic explicit memory commands."""

    STORED = "stored"
    ALREADY_EXISTS = "already_exists"
    DELETED = "deleted"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    NOT_ENABLED = "not_enabled"
    UNAVAILABLE = "unavailable"
    LIMIT_EXCEEDED = "limit_exceeded"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_CONTENT = "invalid_content"


_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_MEMORY_KEY = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPLICIT_REMEMBER_PREFIX = re.compile(
    r"^\s*(?:jarvis[\s,:-]+)?(?:"
    r"να\s+θυμασαι(?:\s+(?:οτι|πωσ))?"
    r"|θυμησου(?:\s+(?:οτι|πωσ))?"
    r"|κρατα\s+στη\s+μνημη(?:\s+(?:οτι|πωσ))?"
    r"|please\s+remember(?:\s+that)?"
    r"|remember(?:\s+that)?"
    r")\s*[:,;-]?\s*",
)
_EXPLICIT_FORGET_PREFIX = re.compile(
    r"^\s*(?:jarvis[\s,:-]+)?(?:"
    r"ξεχασε(?:\s+(?:οτι|πωσ))?"
    r"|διεγραψε\s+απο\s+τη\s+μνημη(?:\s+(?:οτι|πωσ))?"
    r"|please\s+forget(?:\s+that)?"
    r"|forget(?:\s+that)?"
    r")\s*[:,;-]?\s*",
)

# Retained for callers that imported the earlier bounded-model contract.
# Explicit commands now use code-owned responses and consume no model tokens.
EXPLICIT_MEMORY_RESPONSE_TOKENS = 32
MEMORY_NORMALIZER_VERSION = "jarvis.memory-normalizer/1"
MAX_MEMORY_RECORD_BYTES = 4 * 1024
MAX_MEMORY_QUERY_BYTES = 4 * 1024
MAX_MEMORY_QUERY_TERMS = 32
MAX_MEMORY_RECORD_TERMS = 128
MAX_MEMORY_PRINCIPAL_RECORDS = 2_000
MAX_MEMORY_PRINCIPAL_BYTES = 8 * 1024 * 1024
MEMORY_RECORD_TTL_DAYS = 365

_MEMORY_TERM = re.compile(r"[^\W_]+", re.UNICODE)


def _accent_folded_prefix_view(value: str) -> tuple[str, tuple[int, ...]]:
    """Return a folded matching view and its offsets into the original text.

    The folded text is used only to locate the end of a closed command prefix.
    Callers slice the target from ``value`` using the returned boundary map, so
    target accents, case, and characters never come from the folded view.
    """

    folded: list[str] = []
    original_ends: list[int] = []
    for index, character in enumerate(value):
        for candidate in unicodedata.normalize("NFKD", character.casefold()):
            if unicodedata.combining(candidate):
                continue
            folded.append(candidate)
            original_ends.append(index + 1)
    return "".join(folded), tuple(original_ends)


def parse_explicit_memory_command(value: str) -> tuple[str, str] | None:
    """Return one closed explicit operation/target without model interpretation."""

    normalized = unicodedata.normalize("NFC", value)
    prefix_view, original_ends = _accent_folded_prefix_view(normalized)
    for operation, pattern in (
        ("remember", _EXPLICIT_REMEMBER_PREFIX),
        ("forget", _EXPLICIT_FORGET_PREFIX),
    ):
        match = pattern.match(prefix_view)
        if match is None:
            continue
        prefix_end = 0 if match.end() == 0 else original_ends[match.end() - 1]
        target = " ".join(normalized[prefix_end:].split()).strip(" .")
        # Size is deliberately enforced by the transactional command writer.
        # Returning the parsed command here ensures an oversized explicit
        # request receives a truthful code-owned limit outcome and can never
        # fall through to a model-generated acknowledgement.
        if not target:
            return None
        return operation, target
    return None


def normalize_explicit_memory_target(value: str) -> str:
    """Return the exact-match key shared by remember deduplication and forget."""

    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(without_marks.split()).strip(" .")


def memory_normalized_sha256(value: str) -> str:
    """Hash the versioned accent/case-insensitive saved-memory identity."""

    return hashlib.sha256(
        normalize_explicit_memory_target(value).encode("utf-8")
    ).hexdigest()


def memory_search_terms(value: str) -> tuple[str, ...]:
    """Return stable unique normalized terms without exposing them to SQLite."""

    return tuple(
        sorted(set(_MEMORY_TERM.findall(normalize_explicit_memory_target(value))))
    )


def explicit_memory_response(
    operation: str,
    status: ExplicitMemoryCommandStatus,
    language: str,
) -> str:
    """Render a short code-owned response from a committed command outcome."""

    valid = {
        "remember": {
            ExplicitMemoryCommandStatus.STORED,
            ExplicitMemoryCommandStatus.ALREADY_EXISTS,
            ExplicitMemoryCommandStatus.NOT_ENABLED,
            ExplicitMemoryCommandStatus.UNAVAILABLE,
            ExplicitMemoryCommandStatus.LIMIT_EXCEEDED,
            ExplicitMemoryCommandStatus.QUOTA_EXCEEDED,
            ExplicitMemoryCommandStatus.INVALID_CONTENT,
        },
        "forget": {
            ExplicitMemoryCommandStatus.DELETED,
            ExplicitMemoryCommandStatus.NOT_FOUND,
            ExplicitMemoryCommandStatus.AMBIGUOUS,
            ExplicitMemoryCommandStatus.NOT_ENABLED,
            ExplicitMemoryCommandStatus.UNAVAILABLE,
            ExplicitMemoryCommandStatus.LIMIT_EXCEEDED,
            ExplicitMemoryCommandStatus.INVALID_CONTENT,
        },
    }
    if operation not in valid or status not in valid[operation]:
        raise ValueError("explicit memory command outcome is invalid")
    greek = language.casefold().split("-", 1)[0] == "el"
    messages = {
        ExplicitMemoryCommandStatus.STORED: (
            "Το αποθήκευσα στην αποθηκευμένη μνήμη.",
            "Stored in saved memory.",
        ),
        ExplicitMemoryCommandStatus.ALREADY_EXISTS: (
            "Υπήρχε ήδη στην αποθηκευμένη μνήμη.",
            "It was already in saved memory.",
        ),
        ExplicitMemoryCommandStatus.DELETED: (
            "Το αφαίρεσα από την αποθηκευμένη μνήμη· το ιστορικό συνομιλίας δεν άλλαξε.",
            "Removed from saved memory; chat history is unchanged.",
        ),
        ExplicitMemoryCommandStatus.NOT_FOUND: (
            "Δεν βρήκα ακριβώς αυτή την εγγραφή στην αποθηκευμένη μνήμη.",
            "I could not find that exact item in saved memory.",
        ),
        ExplicitMemoryCommandStatus.AMBIGUOUS: (
            "Βρήκα πολλές αντιστοιχίες στην αποθηκευμένη μνήμη· δεν αφαίρεσα καμία.",
            "I found multiple matches in saved memory and removed none.",
        ),
        ExplicitMemoryCommandStatus.NOT_ENABLED: (
            "Η αποθηκευμένη μνήμη δεν είναι ενεργοποιημένη· δεν έκανα καμία αλλαγή.",
            "Saved memory is not enabled; I made no change.",
        ),
        ExplicitMemoryCommandStatus.UNAVAILABLE: (
            "Η αποθηκευμένη μνήμη δεν είναι προσωρινά διαθέσιμη· δεν έκανα καμία αλλαγή.",
            "Saved memory is temporarily unavailable; I made no change.",
        ),
        ExplicitMemoryCommandStatus.LIMIT_EXCEEDED: (
            "Η εγγραφή ξεπερνά το όριο μεγέθους της αποθηκευμένης μνήμης· δεν έκανα αλλαγή.",
            "That item exceeds the saved-memory size limit; I made no change.",
        ),
        ExplicitMemoryCommandStatus.QUOTA_EXCEEDED: (
            "Η αποθηκευμένη μνήμη είναι πλήρης· δεν έκανα αλλαγή.",
            "Saved memory is full; I made no change.",
        ),
        ExplicitMemoryCommandStatus.INVALID_CONTENT: (
            "Η εγγραφή δεν μπορεί να ευρετηριαστεί με ασφάλεια· δεν έκανα αλλαγή.",
            "That item cannot be safely indexed; I made no change.",
        ),
    }
    return messages[status][0 if greek else 1]


@dataclass(frozen=True, slots=True)
class ExplicitMemoryCommandResult:
    """Content-free durable result reconstructed from the command audit."""

    operation: str
    status: ExplicitMemoryCommandStatus
    matched_count: int
    changed: bool
    target_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExplicitMemoryCommandStatus):
            raise TypeError("explicit memory status is invalid")
        expected_change = self.status in {
            ExplicitMemoryCommandStatus.STORED,
            ExplicitMemoryCommandStatus.DELETED,
        }
        if self.operation not in {"remember", "forget"}:
            raise ValueError("explicit memory operation is invalid")
        if not _SHA256.fullmatch(self.target_sha256):
            raise ValueError("explicit memory target hash is invalid")
        if (
            isinstance(self.matched_count, bool)
            or not isinstance(self.matched_count, int)
            or self.matched_count < 0
        ):
            raise ValueError("explicit memory match count is invalid")
        if not isinstance(self.changed, bool):
            raise TypeError("explicit memory changed flag must be boolean")
        if self.changed != expected_change:
            raise ValueError("explicit memory changed flag is inconsistent")
        expected_matches = {
            ExplicitMemoryCommandStatus.STORED: lambda count: count == 0,
            ExplicitMemoryCommandStatus.ALREADY_EXISTS: lambda count: count >= 1,
            ExplicitMemoryCommandStatus.DELETED: lambda count: count == 1,
            ExplicitMemoryCommandStatus.NOT_FOUND: lambda count: count == 0,
            ExplicitMemoryCommandStatus.AMBIGUOUS: lambda count: count > 1,
            ExplicitMemoryCommandStatus.NOT_ENABLED: lambda count: count == 0,
            ExplicitMemoryCommandStatus.UNAVAILABLE: lambda count: count == 0,
            ExplicitMemoryCommandStatus.LIMIT_EXCEEDED: lambda count: count == 0,
            ExplicitMemoryCommandStatus.QUOTA_EXCEEDED: lambda count: count == 0,
            ExplicitMemoryCommandStatus.INVALID_CONTENT: lambda count: count == 0,
        }
        if not expected_matches[self.status](self.matched_count):
            raise ValueError("explicit memory match count is inconsistent")
        explicit_memory_response(self.operation, self.status, "en")


@dataclass(frozen=True, slots=True)
class MemoryObservation:
    task_id: str
    conversation_id: str
    turn_id: str
    source_event_id: str
    text: str
    language: str
    epoch: int = 0

    def __post_init__(self) -> None:
        for value in (
            self.task_id,
            self.conversation_id,
            self.turn_id,
            self.source_event_id,
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError("memory observation identifier is invalid")
        if not self.text or len(self.text.encode("utf-8")) > 32_768:
            raise ValueError("memory observation text is empty or too large")
        if not self.language or len(self.language.encode("utf-8")) > 32:
            raise ValueError("memory observation language is invalid")
        if (
            isinstance(self.epoch, bool)
            or not isinstance(self.epoch, int)
            or self.epoch < 0
        ):
            raise ValueError("memory observation epoch is invalid")


@dataclass(frozen=True, slots=True)
class MemoryRecallRequest:
    task_id: str
    conversation_id: str
    epoch: int
    query: str
    limit: int

    def __post_init__(self) -> None:
        for value in (self.task_id, self.conversation_id):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError("memory recall identifier is invalid")
        if (
            isinstance(self.epoch, bool)
            or not isinstance(self.epoch, int)
            or self.epoch < 0
        ):
            raise ValueError("memory recall epoch is invalid")
        if not self.query or len(self.query.encode("utf-8")) > MAX_MEMORY_QUERY_BYTES:
            raise ValueError("memory recall query is empty or too large")
        if len(memory_search_terms(self.query)) > MAX_MEMORY_QUERY_TERMS:
            raise ValueError("memory recall query has too many unique terms")
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or not 1 <= self.limit <= 20
        ):
            raise ValueError("memory recall limit is invalid")


@dataclass(frozen=True, slots=True)
class MemoryItem:
    text: str
    score: float

    def __post_init__(self) -> None:
        if not self.text or len(self.text.encode("utf-8")) > MAX_MEMORY_RECORD_BYTES:
            raise ValueError("memory item text is empty or too large")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
            or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("memory item score is invalid")


@dataclass(frozen=True, slots=True)
class MemoryRecall:
    items: tuple[MemoryItem, ...] = ()
    provider: str = "disabled"

    def __post_init__(self) -> None:
        if len(self.items) > 20:
            raise ValueError("memory recall returned too many items")
        if not _IDENTIFIER.fullmatch(self.provider):
            raise ValueError("memory recall provider is invalid")


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    stored: bool
    provider: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.provider):
            raise ValueError("memory writer provider is invalid")
        if not _SHA256.fullmatch(self.content_sha256):
            raise ValueError("memory writer content hash is invalid")


@dataclass(frozen=True, slots=True)
class MemoryDeleteResult:
    """Content-free outcome of one explicit retrospective forget command."""

    requested: bool
    deleted: bool
    matched_count: int
    provider: str
    target_sha256: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.provider):
            raise ValueError("memory delete provider is invalid")
        if not _SHA256.fullmatch(self.target_sha256):
            raise ValueError("memory delete target hash is invalid")
        if (
            isinstance(self.matched_count, bool)
            or not isinstance(self.matched_count, int)
            or self.matched_count < 0
        ):
            raise ValueError("memory delete match count is invalid")
        if not self.requested and (self.deleted or self.matched_count):
            raise ValueError("unrequested memory delete outcome is not empty")
        if self.deleted != (self.requested and self.matched_count == 1):
            raise ValueError("memory delete outcome is inconsistent")


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    operation: MemoryOperation
    memory_key: str | None
    supersedes_source_event_ids: tuple[str, ...]
    summary: str
    writer_contract_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation, MemoryOperation):
            raise TypeError("memory proposal operation is invalid")
        if self.operation is MemoryOperation.IGNORE:
            if self.memory_key is not None or self.supersedes_source_event_ids:
                raise ValueError("ignored memory proposal must be empty")
        elif self.memory_key is None or not _MEMORY_KEY.fullmatch(self.memory_key):
            raise ValueError("memory proposal key is invalid")
        if len(self.supersedes_source_event_ids) > 1:
            raise ValueError("memory proposal may supersede at most one source")
        if len(set(self.supersedes_source_event_ids)) != len(
            self.supersedes_source_event_ids
        ):
            raise ValueError("memory proposal superseded sources must be unique")
        if any(
            not _IDENTIFIER.fullmatch(value)
            for value in self.supersedes_source_event_ids
        ):
            raise ValueError("memory proposal superseded source is invalid")
        if (
            self.operation is MemoryOperation.CANCEL
            and len(self.supersedes_source_event_ids) != 1
        ):
            raise ValueError("memory cancellation must identify one active source")
        if not self.summary or len(self.summary.encode("utf-8")) > 512:
            raise ValueError("memory proposal summary is empty or too large")
        if not _SHA256.fullmatch(self.writer_contract_sha256):
            raise ValueError("memory writer contract hash is invalid")


class ShadowMemoryServicePort(Protocol):
    """Proposes memory changes but cannot mutate authoritative state."""

    async def observe(self, observation: MemoryObservation) -> MemoryProposal: ...


class DisabledShadowMemoryService:
    async def observe(self, observation: MemoryObservation) -> MemoryProposal:
        del observation
        raise RuntimeError("memory_shadow_service_disabled")


__all__ = [
    "EXPLICIT_MEMORY_RESPONSE_TOKENS",
    "MAX_MEMORY_PRINCIPAL_BYTES",
    "MAX_MEMORY_PRINCIPAL_RECORDS",
    "MAX_MEMORY_QUERY_BYTES",
    "MAX_MEMORY_QUERY_TERMS",
    "MAX_MEMORY_RECORD_BYTES",
    "MAX_MEMORY_RECORD_TERMS",
    "MEMORY_NORMALIZER_VERSION",
    "MEMORY_RECORD_TTL_DAYS",
    "DisabledShadowMemoryService",
    "ExplicitMemoryCommandResult",
    "ExplicitMemoryCommandStatus",
    "MemoryDeleteResult",
    "MemoryItem",
    "MemoryObservation",
    "MemoryOperation",
    "MemoryProposal",
    "MemoryRecall",
    "MemoryRecallRequest",
    "MemoryWriteResult",
    "ShadowMemoryServicePort",
    "explicit_memory_response",
    "memory_normalized_sha256",
    "memory_search_terms",
    "normalize_explicit_memory_target",
    "parse_explicit_memory_command",
]
