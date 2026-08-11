"""Frozen, framework-independent Research System v1 contracts and algorithms.

The module intentionally contains no network or model SDK code.  Adapters may
produce candidates, but only these validators and deterministic functions are
allowed to turn them into durable or publishable research evidence.
"""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from .domain import OrchestratorError, canonical_json, canonical_sha256, nfc, parse_utc

TEXT_NORMALIZATION_VERSION = "jarvis-research-text/1"
DATE_METADATA_VERSION = "date-metadata/1"
ROBOTS_POLICY_VERSION = "robots-policy/1"
CORROBORATION_POLICY_VERSION = "jarvis-corroboration/1"
MAX_PLAN_SUBQUERIES = 4
MAX_DOCUMENTS = 8
MAX_DOCUMENT_BYTES = 64 * 1024
MAX_RUN_DOCUMENT_BYTES = 512 * 1024
MAX_PASSAGES = 12
MAX_PASSAGE_CHARS = 1_000
MAX_PASSAGES_PER_FACET = 3
MAX_CLAIMS = 6
MAX_BUNDLE_BYTES = 32 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,95}$")
_UNSAFE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_INLINE_SPACE = re.compile(r"[^\S\n]+")
_MANY_BLANKS = re.compile(r"\n{3,}")
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_VALUE = re.compile(r"(?<!\w)(?:\d{1,4}(?:[.,]\d+)?%?|\d{4}-\d{2}-\d{2})(?!\w)")
_NEGATION = frozenset({"not", "no", "never", "without", "δεν", "μη", "χωρίς", "ουδέποτε"})
_MARKDOWN_ESCAPE = re.compile(r"([\\`*_{}\[\]()<>#+.!|~-])")
_CREDENTIAL_QUERY_KEYS = frozenset(
    {"token", "access_token", "api_key", "apikey", "key", "secret", "password", "passwd", "auth", "signature", "sig"}
)
_INVALID_PERCENT = re.compile(r"%(?![0-9a-fA-F]{2})")


def _require_text(value: str, name: str, *, minimum: int = 1, maximum: int = 4096) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    normalized = nfc(value)
    if normalized != value or not minimum <= len(value) <= maximum or _UNSAFE_CONTROL.search(value):
        raise ValueError(f"{name} is invalid")
    return value


def _require_id(value: str, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _require_hash(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


class SourceRole(StrEnum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    UNKNOWN = "UNKNOWN"


class DateProvenance(StrEnum):
    JSON_LD = "JSON_LD"
    ARTICLE_PUBLISHED_TIME = "ARTICLE_PUBLISHED_TIME"
    APPROVED_META = "APPROVED_META"
    ARTICLE_TIME = "ARTICLE_TIME"
    CONFLICT = "CONFLICT"
    NONE = "NONE"


class FacetState(StrEnum):
    COVERED = "COVERED"
    CONFLICTED = "CONFLICTED"
    UNCOVERED = "UNCOVERED"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


class ResearchStatus(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    FAILED = "FAILED"


class SpanRelation(StrEnum):
    ANCHOR = "ANCHOR"
    SUPPORTS = "SUPPORTS"
    CONFLICTS = "CONFLICTS"


@dataclass(frozen=True, slots=True)
class SearchFacetV1:
    facet_id: str
    label: str

    def __post_init__(self) -> None:
        _require_id(self.facet_id, "facet_id")
        _require_text(self.label, "facet label", maximum=160)


@dataclass(frozen=True, slots=True)
class SearchPlanV1:
    canonical_query: str
    facets: tuple[SearchFacetV1, ...]
    subqueries: tuple[str, ...]
    language: str
    freshness: str
    timezone: str
    page_limit: int = 8
    deadline_seconds: int = 45
    browser_fallback_allowed: bool = False
    version: str = "search-plan/1"

    def __post_init__(self) -> None:
        _require_text(self.canonical_query, "canonical query", maximum=4096)
        if not 1 <= len(self.facets) <= MAX_CLAIMS or len({f.facet_id for f in self.facets}) != len(self.facets):
            raise ValueError("facets are invalid")
        if not 1 <= len(self.subqueries) <= MAX_PLAN_SUBQUERIES:
            raise ValueError("subqueries are invalid")
        for query in self.subqueries:
            _require_text(query, "subquery", maximum=512)
        if len(set(self.subqueries)) != len(self.subqueries):
            raise ValueError("subqueries must be unique")
        if self.language not in {"el", "en"}:
            raise ValueError("plan language is unsupported")
        _require_text(self.freshness, "freshness", maximum=64)
        _require_text(self.timezone, "timezone", maximum=64)
        if isinstance(self.page_limit, bool) or not 1 <= self.page_limit <= MAX_DOCUMENTS:
            raise ValueError("page_limit is invalid")
        if self.deadline_seconds != 45:
            raise ValueError("deadline_seconds must be 45")
        if not isinstance(self.browser_fallback_allowed, bool):
            raise ValueError("browser_fallback_allowed must be boolean")
        if self.version != "search-plan/1":
            raise ValueError("unsupported search plan version")

    @property
    def plan_hash(self) -> str:
        return canonical_sha256(asdict(self))

    @property
    def approval_arguments(self) -> dict[str, Any]:
        return {
            "query": self.canonical_query,
            "subqueries": list(self.subqueries),
            "public_services": True,
            "page_limit": self.page_limit,
            "deadline_seconds": self.deadline_seconds,
            "browser_fallback_allowed": self.browser_fallback_allowed,
            "plan_hash": self.plan_hash,
        }


@dataclass(frozen=True, slots=True)
class NormalizedDocumentV1:
    text: str
    sha256: str
    character_length: int
    byte_length: int
    truncated: bool
    normalization_version: str = TEXT_NORMALIZATION_VERSION

    def __post_init__(self) -> None:
        if self.normalization_version != TEXT_NORMALIZATION_VERSION:
            raise ValueError("unsupported normalization version")
        _require_hash(self.sha256, "document hash")
        if self.text != normalize_research_text(self.text, limit_bytes=MAX_DOCUMENT_BYTES)[0]:
            raise ValueError("document text is not normalized")
        encoded = self.text.encode("utf-8")
        if self.character_length != len(self.text) or self.byte_length != len(encoded):
            raise ValueError("document lengths do not match")
        if self.byte_length > MAX_DOCUMENT_BYTES or hashlib.sha256(encoded).hexdigest() != self.sha256:
            raise ValueError("document hash or size is invalid")
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated must be boolean")

    @classmethod
    def from_text(cls, value: str) -> "NormalizedDocumentV1":
        normalized, truncated = normalize_research_text(value, limit_bytes=MAX_DOCUMENT_BYTES)
        encoded = normalized.encode("utf-8")
        return cls(normalized, hashlib.sha256(encoded).hexdigest(), len(normalized), len(encoded), truncated)


@dataclass(frozen=True, slots=True)
class EvidenceSourceV1:
    source_id: str
    requested_url: str
    final_url: str
    accepted_canonical_url: str
    title: str
    publisher: str
    published_at: str | None
    date_provenance: DateProvenance
    retrieved_at: str
    source_role: SourceRole
    authority_registry_version: str
    registrable_domain: str
    independence_group_id: str
    normalized_document_hash: str
    language_code: str
    language_detector_version: str
    language_confidence: float

    def __post_init__(self) -> None:
        _require_id(self.source_id, "source_id")
        for name in ("requested_url", "final_url", "accepted_canonical_url"):
            validate_public_url(getattr(self, name))
        _require_text(self.title, "title", maximum=512)
        _require_text(self.publisher, "publisher", minimum=0, maximum=256)
        if self.published_at is not None:
            parse_utc(self.published_at)
            if self.date_provenance in {DateProvenance.NONE, DateProvenance.CONFLICT}:
                raise ValueError("publication provenance contradicts date")
        elif self.date_provenance not in {DateProvenance.NONE, DateProvenance.CONFLICT}:
            raise ValueError("publication provenance requires a date")
        parse_utc(self.retrieved_at)
        _require_text(self.authority_registry_version, "authority registry version", maximum=64)
        _require_text(self.registrable_domain, "registrable domain", maximum=253)
        _require_id(self.independence_group_id, "independence_group_id")
        _require_hash(self.normalized_document_hash, "normalized document hash")
        if self.language_code not in {"el", "en"} or not 0.80 <= self.language_confidence <= 1.0:
            raise ValueError("source language is ineligible")
        _require_text(self.language_detector_version, "language detector version", maximum=128)


@dataclass(frozen=True, slots=True)
class RankedPassageV1:
    passage_id: str
    source_id: str
    document_hash: str
    facet_id: str
    start_char: int
    end_char: int
    section_heading: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.passage_id, "passage_id")
        _require_id(self.source_id, "source_id")
        _require_hash(self.document_hash, "document hash")
        _require_id(self.facet_id, "facet_id")
        if isinstance(self.start_char, bool) or not 0 <= self.start_char < self.end_char:
            raise ValueError("passage offsets are invalid")
        if self.end_char - self.start_char > MAX_PASSAGE_CHARS:
            raise ValueError("passage is too long")
        if self.section_heading is not None:
            _require_text(self.section_heading, "section heading", maximum=200)


@dataclass(frozen=True, slots=True)
class SpanSelectionV1:
    facet_id: str
    passage_id: str
    start_char: int
    end_char: int
    relation: SpanRelation

    def __post_init__(self) -> None:
        _require_id(self.facet_id, "facet_id")
        _require_id(self.passage_id, "passage_id")
        if isinstance(self.start_char, bool) or not 0 <= self.start_char < self.end_char:
            raise ValueError("span offsets are invalid")


@dataclass(frozen=True, slots=True)
class EvidenceLocatorV1:
    source_id: str
    document_hash: str
    normalization_version: str
    passage_id: str
    start_char: int
    end_char: int
    section_heading: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.source_id, "source_id")
        _require_hash(self.document_hash, "document hash")
        if self.normalization_version != TEXT_NORMALIZATION_VERSION:
            raise ValueError("unsupported normalization version")
        _require_id(self.passage_id, "passage_id")
        if isinstance(self.start_char, bool) or not 0 <= self.start_char < self.end_char:
            raise ValueError("locator offsets are invalid")
        if self.end_char - self.start_char > 400:
            raise ValueError("locator is too long")


@dataclass(frozen=True, slots=True)
class EvidenceClaimV1:
    claim_id: str
    facet_id: str
    statement: str
    anchor: EvidenceLocatorV1
    corroborating: EvidenceLocatorV1 | None = None
    corroboration_policy_version: str = CORROBORATION_POLICY_VERSION

    def __post_init__(self) -> None:
        _require_id(self.claim_id, "claim_id")
        _require_id(self.facet_id, "facet_id")
        _require_text(self.statement, "statement", minimum=40, maximum=400)
        if self.corroboration_policy_version != CORROBORATION_POLICY_VERSION:
            raise ValueError("unsupported corroboration policy")


@dataclass(frozen=True, slots=True)
class FacetResultV1:
    facet_id: str
    state: FacetState
    claim_ids: tuple[str, ...] = ()
    reason_code: str = ""

    def __post_init__(self) -> None:
        _require_id(self.facet_id, "facet_id")
        if len(self.claim_ids) > MAX_CLAIMS or len(set(self.claim_ids)) != len(self.claim_ids):
            raise ValueError("facet claim IDs are invalid")
        for claim_id in self.claim_ids:
            _require_id(claim_id, "claim_id")
        _require_text(self.reason_code, "reason code", minimum=0, maximum=96)


@dataclass(frozen=True, slots=True)
class EvidenceBundleV1:
    plan_hash: str
    status: ResearchStatus
    facets: tuple[FacetResultV1, ...]
    sources: tuple[EvidenceSourceV1, ...]
    claims: tuple[EvidenceClaimV1, ...]
    bundle_version: str = "evidence-bundle/1"

    def __post_init__(self) -> None:
        _require_hash(self.plan_hash, "plan hash")
        if self.bundle_version != "evidence-bundle/1":
            raise ValueError("unsupported bundle version")
        if not self.facets or len({item.facet_id for item in self.facets}) != len(self.facets):
            raise ValueError("bundle facets are invalid")
        if len(self.sources) > MAX_DOCUMENTS or len({item.source_id for item in self.sources}) != len(self.sources):
            raise ValueError("bundle sources are invalid")
        if len(self.claims) > MAX_CLAIMS or len({item.claim_id for item in self.claims}) != len(self.claims):
            raise ValueError("bundle claims are invalid")
        source_ids = {item.source_id for item in self.sources}
        claim_ids = {item.claim_id for item in self.claims}
        facet_ids = {item.facet_id for item in self.facets}
        for claim in self.claims:
            if claim.facet_id not in facet_ids or claim.anchor.source_id not in source_ids:
                raise ValueError("claim cross-reference is invalid")
            if claim.corroborating is not None and claim.corroborating.source_id not in source_ids:
                raise ValueError("corroborating source is missing")
        for facet in self.facets:
            if not set(facet.claim_ids) <= claim_ids:
                raise ValueError("facet claim cross-reference is invalid")
        if resolve_research_status(item.state for item in self.facets) != self.status:
            raise ValueError("overall status does not match facets")
        if len(self.to_json().encode("utf-8")) > MAX_BUNDLE_BYTES:
            raise ValueError("evidence bundle is too large")

    def to_json(self) -> str:
        return canonical_json(asdict(self))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResearchRequestV1:
    task_id: str
    conversation_id: str
    epoch: int
    plan: SearchPlanV1
    deadline_seconds: int = 45
    max_documents: int = MAX_DOCUMENTS
    max_passages: int = MAX_PASSAGES
    max_claims: int = MAX_CLAIMS
    span_selection_output_tokens: int = 768

    def __post_init__(self) -> None:
        _require_id(self.task_id, "task_id")
        _require_id(self.conversation_id, "conversation_id")
        if isinstance(self.epoch, bool) or self.epoch < 0:
            raise ValueError("epoch is invalid")
        if self.deadline_seconds != 45 or self.max_documents != 8 or self.max_passages != 12 or self.max_claims != 6:
            raise ValueError("research execution bounds are frozen")
        if self.span_selection_output_tokens != 768:
            raise ValueError("span selection output budget is frozen")


@dataclass(frozen=True, slots=True)
class ResearchCorpusV1:
    plan_hash: str
    sources: tuple[EvidenceSourceV1, ...]
    documents: Mapping[str, NormalizedDocumentV1]
    passages: tuple[RankedPassageV1, ...]
    technical_failure_facets: tuple[str, ...] = ()
    network_retry_count: int = 0

    def __post_init__(self) -> None:
        _require_hash(self.plan_hash, "plan hash")
        if len(self.sources) > MAX_DOCUMENTS or len({item.source_id for item in self.sources}) != len(self.sources):
            raise ValueError("research corpus sources are invalid")
        source_ids = {item.source_id for item in self.sources}
        if set(self.documents) != source_ids:
            raise ValueError("research corpus must include one document per source")
        if sum(document.byte_length for document in self.documents.values()) > MAX_RUN_DOCUMENT_BYTES:
            raise ValueError("research document budget exceeded")
        for source in self.sources:
            if self.documents[source.source_id].sha256 != source.normalized_document_hash:
                raise ValueError("research corpus document hash mismatch")
        if len(self.passages) > MAX_PASSAGES or len({item.passage_id for item in self.passages}) != len(self.passages):
            raise ValueError("research corpus passages are invalid")
        per_facet = Counter(item.facet_id for item in self.passages)
        if any(count > MAX_PASSAGES_PER_FACET for count in per_facet.values()):
            raise ValueError("research corpus exceeds per-facet passage budget")
        for passage in self.passages:
            document = self.documents.get(passage.source_id)
            if document is None or passage.document_hash != document.sha256 or passage.end_char > document.character_length:
                raise ValueError("research corpus passage is not reconstructable")
        if len(set(self.technical_failure_facets)) != len(self.technical_failure_facets):
            raise ValueError("technical failure facets are invalid")
        for facet_id in self.technical_failure_facets:
            _require_id(facet_id, "facet_id")
        if self.network_retry_count not in {0, 1}:
            raise ValueError("network retry budget exceeded")


@dataclass(frozen=True, slots=True)
class ResearchResultV1:
    bundle: EvidenceBundleV1
    cited_documents: Mapping[str, NormalizedDocumentV1]

    def __post_init__(self) -> None:
        if set(self.cited_documents) != {source.source_id for source in self.bundle.sources}:
            raise ValueError("result must include exactly the cited documents")
        if sum(document.byte_length for document in self.cited_documents.values()) > MAX_RUN_DOCUMENT_BYTES:
            raise ValueError("research document budget exceeded")


@dataclass(frozen=True, slots=True)
class SourceFingerprintV1:
    source_id: str
    registrable_domain: str
    title: str
    publisher: str
    published_at: str | None
    document: NormalizedDocumentV1

    def __post_init__(self) -> None:
        _require_id(self.source_id, "source_id")
        _require_text(self.registrable_domain, "registrable domain", maximum=253)
        _require_text(self.title, "title", maximum=512)
        _require_text(self.publisher, "publisher", minimum=0, maximum=256)
        if self.published_at is not None:
            parse_utc(self.published_at)


def normalize_research_text(value: str, *, limit_bytes: int = MAX_DOCUMENT_BYTES) -> tuple[str, bool]:
    if not isinstance(value, str):
        raise ValueError("research text must be a string")
    text = html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", _UNSAFE_CONTROL.sub("", text))
    text = "\n".join(_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    text = _MANY_BLANKS.sub("\n\n", text).strip()
    encoded = text.encode("utf-8")
    if len(encoded) <= limit_bytes:
        return text, False
    clipped = encoded[:limit_bytes]
    while True:
        try:
            text = clipped.decode("utf-8")
            break
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return text.rstrip(), True


def validate_public_url(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 4096
        or _UNSAFE_CONTROL.search(value)
        or any(character.isspace() for character in value)
        or "\\" in value
        or _INVALID_PERCENT.search(value)
    ):
        raise ValueError("URL is invalid")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("URL authority is invalid")
    if parsed.fragment or parsed.port not in {None, 80, 443}:
        raise ValueError("URL target is forbidden")
    if parsed.hostname.endswith(".") or any(ord(ch) > 127 for ch in parsed.hostname):
        raise ValueError("URL host must be canonical ASCII")
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        raise ValueError("URL IP literal is forbidden")
    if literal is None and "." not in parsed.hostname:
        raise ValueError("URL host must be a public DNS name")
    try:
        decoded_target = unquote(parsed.path + "?" + parsed.query, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("URL encoding is invalid") from exc
    if _UNSAFE_CONTROL.search(decoded_target) or any(character in "\r\n" for character in decoded_target):
        raise ValueError("URL contains encoded controls")
    for pair in parsed.query.split("&") if parsed.query else ():
        key = unquote(pair.partition("=")[0], errors="strict").casefold()
        if key in _CREDENTIAL_QUERY_KEYS:
            raise ValueError("URL contains a credential parameter")
    host = f"[{parsed.hostname}]" if literal is not None and literal.version == 6 else parsed.hostname
    default_port = 80 if parsed.scheme == "http" else 443
    netloc = host if parsed.port in {None, default_port} else f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc.lower(), parsed.path or "/", parsed.query, ""))


def accepted_canonical_url(final_url: str, canonical_url: str | None, *, same_registrable_domain: bool, content_matches: bool) -> str:
    safe_final = validate_public_url(final_url)
    if not canonical_url:
        return safe_final
    try:
        safe_canonical = validate_public_url(canonical_url)
    except ValueError:
        return safe_final
    return safe_canonical if same_registrable_domain or content_matches else safe_final


def make_locator(selection: SpanSelectionV1, passage: RankedPassageV1, document: NormalizedDocumentV1) -> EvidenceLocatorV1:
    if selection.passage_id != passage.passage_id or selection.facet_id != passage.facet_id:
        raise ValueError("selection does not match passage")
    if passage.document_hash != document.sha256 or passage.end_char > document.character_length:
        raise ValueError("passage does not match document")
    start = selection.start_char
    end = selection.end_char
    if start < passage.start_char or end > passage.end_char:
        raise ValueError("selection is outside passage")
    statement = document.text[start:end]
    _require_text(statement, "statement", minimum=40, maximum=400)
    if statement != statement.strip() or statement[-1] not in ".!?;;…":
        raise ValueError("span must contain complete sentence boundaries")
    raw_prefix = document.text[:start]
    prefix = raw_prefix.rstrip()
    if prefix and not raw_prefix.endswith("\n\n") and prefix[-1] not in ".!?;;…":
        raise ValueError("span must start at a complete sentence boundary")
    return EvidenceLocatorV1(
        passage.source_id,
        document.sha256,
        document.normalization_version,
        passage.passage_id,
        start,
        end,
        passage.section_heading,
    )


def reconstruct(locator: EvidenceLocatorV1, document: NormalizedDocumentV1) -> str:
    if locator.document_hash != document.sha256 or locator.normalization_version != document.normalization_version:
        raise ValueError("locator document mismatch")
    if locator.end_char > document.character_length:
        raise ValueError("locator out of bounds")
    return document.text[locator.start_char : locator.end_char]


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN.finditer(unicodedata.normalize("NFKC", text)))


def weighted_token_jaccard(left: str, right: str) -> float:
    first, second = Counter(_tokens(left)), Counter(_tokens(right))
    vocabulary = set(first) | set(second)
    if not vocabulary:
        return 0.0
    weights = {token: 1.0 + min(len(token), 12) / 12 for token in vocabulary}
    intersection = sum(min(first[token], second[token]) * weights[token] for token in vocabulary)
    union = sum(max(first[token], second[token]) * weights[token] for token in vocabulary)
    return intersection / union if union else 0.0


def assign_independence_groups(
    fingerprints: Sequence[SourceFingerprintV1], *, similarity_threshold: float = 0.90
) -> dict[str, str]:
    """Collapse exact/syndicated near duplicates even across different domains."""

    if not 0.90 <= similarity_threshold <= 1.0:
        raise ValueError("independence threshold is outside v1 policy")
    if len(fingerprints) > MAX_DOCUMENTS or len({item.source_id for item in fingerprints}) != len(fingerprints):
        raise ValueError("source fingerprints are invalid")
    parent = list(range(len(fingerprints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        first, second = find(left), find(right)
        if first != second:
            parent[max(first, second)] = min(first, second)

    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            first, second = fingerprints[left], fingerprints[right]
            exact = first.document.sha256 == second.document.sha256
            near = (
                weighted_token_jaccard(first.document.text, second.document.text) >= similarity_threshold
                and weighted_token_jaccard(first.title, second.title) >= 0.75
            )
            matching_signals = (
                not first.publisher
                or not second.publisher
                or first.publisher.casefold() == second.publisher.casefold()
            ) and (
                first.published_at is None
                or second.published_at is None
                or first.published_at == second.published_at
            )
            if exact or (near and matching_signals):
                union(left, right)
    members: dict[int, list[SourceFingerprintV1]] = {}
    for index, item in enumerate(fingerprints):
        members.setdefault(find(index), []).append(item)
    result: dict[str, str] = {}
    for cluster in members.values():
        identity = hashlib.sha256(
            canonical_json(
                sorted(
                    (item.document.sha256, item.source_id) for item in cluster
                )
            ).encode("utf-8")
        ).hexdigest()[:24]
        for item in cluster:
            result[item.source_id] = f"ig-{identity}"
    return result


def _polarity(text: str) -> bool:
    return any(token in _NEGATION for token in _tokens(text))


def _values(text: str) -> frozenset[str]:
    return frozenset(item.replace(",", ".") for item in _VALUE.findall(text.casefold()))


def corroborates(
    anchor: str,
    candidate: str,
    *,
    relation: SpanRelation,
    anchor_entities: Iterable[str],
    candidate_entities: Iterable[str],
    different_independence_group: bool,
    conflict_candidate_present: bool,
) -> bool:
    left_entities = frozenset(nfc(item).casefold() for item in anchor_entities)
    right_entities = frozenset(nfc(item).casefold() for item in candidate_entities)
    return all(
        (
            relation == SpanRelation.SUPPORTS,
            bool(left_entities) and left_entities == right_entities,
            _values(anchor) == _values(candidate),
            _polarity(anchor) == _polarity(candidate),
            weighted_token_jaccard(anchor, candidate) >= 0.70,
            different_independence_group,
            not conflict_candidate_present,
        )
    )


_ENTITY_STOP = frozenset(
    {"the", "a", "an", "this", "that", "according", "official", "current", "η", "ο", "το", "οι", "τα", "ένα", "μία", "επίσημη"}
)


def critical_entities(value: str) -> tuple[str, ...]:
    """Pinned alias-free lexical entity extractor used by the v1 heuristic."""

    tokens = [match.group(0) for match in _TOKEN.finditer(value)]
    entities: list[str] = []
    current: list[str] = []
    for token in tokens:
        eligible = len(token) > 1 and (token[0].isupper() or token.isupper()) and token.casefold() not in _ENTITY_STOP
        if eligible:
            current.append(token.casefold())
        elif current:
            entities.append(" ".join(current))
            current = []
    if current:
        entities.append(" ".join(current))
    return tuple(sorted(set(entities)))


def _conflicts(anchor: str, candidate: str, relation: SpanRelation) -> bool:
    entities = critical_entities(anchor)
    if relation != SpanRelation.CONFLICTS or not entities or tuple(entities) != critical_entities(candidate):
        return False
    return _values(anchor) != _values(candidate) or _polarity(anchor) != _polarity(candidate)


def build_research_result(
    plan: SearchPlanV1,
    corpus: ResearchCorpusV1,
    selections: Sequence[SpanSelectionV1],
) -> ResearchResultV1:
    """Validate model spans, derive extractive claims, and resolve statuses."""

    if corpus.plan_hash != plan.plan_hash:
        raise ValueError("research corpus does not match plan")
    if len(selections) > MAX_PASSAGES or len(
        {(item.facet_id, item.passage_id, item.start_char, item.end_char, item.relation) for item in selections}
    ) != len(selections):
        raise ValueError("span selections are invalid")
    facet_ids = {item.facet_id for item in plan.facets}
    if any(item.facet_id not in facet_ids for item in selections):
        raise ValueError("span selection references unknown facet")
    passages = {item.passage_id: item for item in corpus.passages}
    sources = {item.source_id: item for item in corpus.sources}
    valid: dict[str, list[tuple[SpanSelectionV1, EvidenceLocatorV1, str]]] = {
        item.facet_id: [] for item in plan.facets
    }
    for selection in selections:
        passage = passages.get(selection.passage_id)
        if passage is None:
            continue
        try:
            locator = make_locator(selection, passage, corpus.documents[passage.source_id])
            statement = reconstruct(locator, corpus.documents[passage.source_id])
        except (KeyError, ValueError):
            continue
        valid[selection.facet_id].append((selection, locator, statement))

    claims: list[EvidenceClaimV1] = []
    facets: list[FacetResultV1] = []
    cited_source_ids: set[str] = set()
    technical = set(corpus.technical_failure_facets)
    for facet in plan.facets:
        candidates = sorted(
            valid[facet.facet_id],
            key=lambda item: (item[1].passage_id, item[1].start_char, item[1].end_char, item[0].relation.value),
        )
        anchors = [item for item in candidates if item[0].relation == SpanRelation.ANCHOR]
        related = [item for item in candidates if item[0].relation != SpanRelation.ANCHOR]
        selected_claim: EvidenceClaimV1 | None = None
        conflict = False
        for _, anchor_locator, statement in anchors:
            conflict = any(_conflicts(statement, other, selection.relation) for selection, _, other in related)
            if conflict:
                break
            anchor_source = sources[anchor_locator.source_id]
            support_locator: EvidenceLocatorV1 | None = None
            for selection, locator, support_text in related:
                support_source = sources[locator.source_id]
                if corroborates(
                    statement,
                    support_text,
                    relation=selection.relation,
                    anchor_entities=critical_entities(statement),
                    candidate_entities=critical_entities(support_text),
                    different_independence_group=(
                        anchor_source.independence_group_id != support_source.independence_group_id
                    ),
                    conflict_candidate_present=False,
                ):
                    support_locator = locator
                    break
            covered = anchor_source.source_role == SourceRole.PRIMARY or (
                anchor_source.source_role == SourceRole.SECONDARY
                and support_locator is not None
                and sources[support_locator.source_id].source_role == SourceRole.SECONDARY
            )
            if not covered:
                continue
            identity = canonical_sha256(
                {
                    "plan_hash": plan.plan_hash,
                    "facet_id": facet.facet_id,
                    "anchor": asdict(anchor_locator),
                    "support": asdict(support_locator) if support_locator else None,
                }
            )[:24]
            selected_claim = EvidenceClaimV1(
                f"claim-{identity}",
                facet.facet_id,
                statement,
                anchor_locator,
                support_locator,
            )
            break
        if conflict:
            facets.append(FacetResultV1(facet.facet_id, FacetState.CONFLICTED, (), "conflict"))
        elif selected_claim is not None:
            claims.append(selected_claim)
            cited_source_ids.add(selected_claim.anchor.source_id)
            if selected_claim.corroborating is not None:
                cited_source_ids.add(selected_claim.corroborating.source_id)
            facets.append(FacetResultV1(facet.facet_id, FacetState.COVERED, (selected_claim.claim_id,)))
        elif facet.facet_id in technical:
            facets.append(FacetResultV1(facet.facet_id, FacetState.TECHNICAL_FAILURE, (), "retrieval_failure"))
        else:
            facets.append(FacetResultV1(facet.facet_id, FacetState.UNCOVERED, (), "no_qualified_evidence"))
    cited_sources = tuple(item for item in corpus.sources if item.source_id in cited_source_ids)
    cited_documents = {source.source_id: corpus.documents[source.source_id] for source in cited_sources}
    bundle = EvidenceBundleV1(
        plan.plan_hash,
        resolve_research_status(item.state for item in facets),
        tuple(facets),
        cited_sources,
        tuple(claims),
    )
    return ResearchResultV1(bundle, cited_documents)


def resolve_research_status(states: Iterable[FacetState]) -> ResearchStatus:
    selected = tuple(states)
    if not selected:
        raise ValueError("at least one facet is required")
    covered = sum(item == FacetState.COVERED for item in selected)
    if covered == len(selected):
        return ResearchStatus.SUFFICIENT
    if covered:
        return ResearchStatus.PARTIAL
    if any(item == FacetState.TECHNICAL_FAILURE for item in selected):
        return ResearchStatus.FAILED
    return ResearchStatus.INSUFFICIENT


def coverage_state(*, conflict: bool, primary_anchor: bool, corroborating_secondary_groups: int, technical_failure: bool = False) -> FacetState:
    if conflict:
        return FacetState.CONFLICTED
    if primary_anchor or corroborating_secondary_groups >= 2:
        return FacetState.COVERED
    if technical_failure:
        return FacetState.TECHNICAL_FAILURE
    return FacetState.UNCOVERED


def markdown_escape(value: str) -> str:
    return _MARKDOWN_ESCAPE.sub(r"\\\1", value.replace("&", "&amp;"))


def _citation(source: EvidenceSourceV1) -> str:
    title = markdown_escape(source.title)
    url = quote(validate_public_url(source.accepted_canonical_url), safe=":/?#[]@!$&'*+,;=%")
    date = f", {source.published_at[:10]}" if source.published_at else ""
    publisher = f" — {markdown_escape(source.publisher)}" if source.publisher else ""
    return f"[{title}]({url}){publisher}{date}"


_LEADS_EL = {
    ResearchStatus.SUFFICIENT: "Βρέθηκαν επαρκή τεκμήρια.",
    ResearchStatus.PARTIAL: "Βρέθηκαν μερικά επαρκή τεκμήρια.",
    ResearchStatus.INSUFFICIENT: "Δεν βρήκα επαρκή τεκμήρια.",
    ResearchStatus.FAILED: "Η έρευνα απέτυχε πριν προκύψει αξιολογήσιμο αποτέλεσμα.",
}
_LEADS_EN = {
    ResearchStatus.SUFFICIENT: "Sufficient evidence was found.",
    ResearchStatus.PARTIAL: "Some sufficient evidence was found.",
    ResearchStatus.INSUFFICIENT: "I did not find sufficient evidence.",
    ResearchStatus.FAILED: "Research failed before an evaluable result was available.",
}


def _lead(language: str, status: ResearchStatus) -> str:
    return (_LEADS_EL if language == "el" else _LEADS_EN)[status]


def render_bundle(plan: SearchPlanV1, bundle: EvidenceBundleV1) -> str:
    if plan.plan_hash != bundle.plan_hash or tuple(item.facet_id for item in plan.facets) != tuple(item.facet_id for item in bundle.facets):
        raise ValueError("bundle does not match approved plan")
    sources = {item.source_id: item for item in bundle.sources}
    claims = {item.claim_id: item for item in bundle.claims}
    lines = [_lead(plan.language, bundle.status)]
    for planned, result in zip(plan.facets, bundle.facets, strict=True):
        lines.append(f"\n{markdown_escape(planned.label)}")
        ordered = sorted(
            (claims[item] for item in result.claim_ids),
            key=lambda claim: (0 if sources[claim.anchor.source_id].source_role == SourceRole.PRIMARY else 1, claim.claim_id),
        )
        for claim in ordered:
            citations = [_citation(sources[claim.anchor.source_id])]
            if claim.corroborating is not None:
                citations.append(_citation(sources[claim.corroborating.source_id]))
            lines.append(f"- {markdown_escape(claim.statement)} ({'; '.join(citations)})")
        if result.state == FacetState.CONFLICTED:
            lines.append("- Υπάρχουν αντικρουόμενα τεκμήρια για αυτό το σκέλος." if plan.language == "el" else "- Conflicting evidence exists for this facet.")
        elif result.state == FacetState.UNCOVERED:
            lines.append("- Δεν βρέθηκαν επαρκή τεκμήρια για αυτό το σκέλος." if plan.language == "el" else "- Sufficient evidence was not found for this facet.")
        elif result.state == FacetState.TECHNICAL_FAILURE:
            lines.append("- Τεχνική αποτυχία εμπόδισε την αξιολόγηση αυτού του σκέλους." if plan.language == "el" else "- A technical failure prevented evaluation of this facet.")
    return "\n".join(lines)


def validate_syntactic_citation_coverage(
    bundle: EvidenceBundleV1,
    documents: Mapping[str, NormalizedDocumentV1],
) -> None:
    """Prove locator/cross-reference coverage, not semantic correctness."""

    sources = {item.source_id: item for item in bundle.sources}
    if set(documents) != set(sources):
        raise ValueError("citation documents do not match cited sources")
    for source_id, source in sources.items():
        if documents[source_id].sha256 != source.normalized_document_hash:
            raise ValueError("citation source document hash mismatch")
    referenced = {claim_id for facet in bundle.facets for claim_id in facet.claim_ids}
    if referenced != {claim.claim_id for claim in bundle.claims}:
        raise ValueError("not every claim is rendered by a facet")
    for claim in bundle.claims:
        if reconstruct(claim.anchor, documents[claim.anchor.source_id]) != claim.statement:
            raise ValueError("claim anchor is not reconstructable")
        if claim.corroborating is not None:
            reconstruct(claim.corroborating, documents[claim.corroborating.source_id])


def speech_units(plan: SearchPlanV1, bundle: EvidenceBundleV1) -> tuple[tuple[str, str], ...]:
    claims = {item.claim_id: item for item in bundle.claims}
    sources = {item.source_id: item for item in bundle.sources}
    default_locale = "el-GR" if plan.language == "el" else "en-US"
    result: list[tuple[str, str]] = [(default_locale, _lead(plan.language, bundle.status))]
    for facet in bundle.facets:
        ordered = sorted(
            (claims[claim_id] for claim_id in facet.claim_ids),
            key=lambda claim: (0 if sources[claim.anchor.source_id].source_role == SourceRole.PRIMARY else 1, claim.claim_id),
        )
        for claim in ordered:
            source = sources[claim.anchor.source_id]
            result.append(("el-GR" if source.language_code == "el" else "en-US", claim.statement))
        if facet.state == FacetState.CONFLICTED:
            result.append((default_locale, "Υπάρχουν αντικρουόμενα τεκμήρια." if plan.language == "el" else "There is conflicting evidence."))
        elif facet.state == FacetState.UNCOVERED:
            result.append((default_locale, "Δεν βρέθηκαν επαρκή τεκμήρια." if plan.language == "el" else "Sufficient evidence was not found."))
        elif facet.state == FacetState.TECHNICAL_FAILURE:
            result.append((default_locale, "Τεχνική αποτυχία εμπόδισε την αξιολόγηση." if plan.language == "el" else "A technical failure prevented evaluation."))
    return tuple(result)


def select_ranked_passages(
    documents: Mapping[str, NormalizedDocumentV1], facets: Sequence[SearchFacetV1]
) -> tuple[RankedPassageV1, ...]:
    """Small deterministic BM25-style passage selector used before model spans."""

    candidates: list[tuple[str, int, int, str]] = []
    for source_id, document in sorted(documents.items()):
        cursor = 0
        for paragraph in document.text.split("\n\n"):
            start = document.text.find(paragraph, cursor)
            end = start + len(paragraph)
            cursor = end
            if 40 <= len(paragraph) <= MAX_PASSAGE_CHARS:
                candidates.append((source_id, start, end, paragraph))
    if not candidates:
        return ()
    tokenized = [_tokens(item[3]) for item in candidates]
    document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
    result: list[RankedPassageV1] = []
    for facet in facets:
        query = set(_tokens(facet.label))
        scored: list[tuple[float, str, int, int]] = []
        for candidate, tokens in zip(candidates, tokenized, strict=True):
            counts = Counter(tokens)
            score = 0.0
            for token in query:
                tf = counts[token]
                if tf:
                    score += tf / (tf + 1.2) * math.log(1 + len(candidates) / document_frequency[token])
            source_id, start, end, _ = candidate
            scored.append((-score, source_id, start, end))
        for _, source_id, start, end in sorted(scored)[:MAX_PASSAGES_PER_FACET]:
            document = documents[source_id]
            result.append(
                RankedPassageV1(
                    f"p{len(result) + 1:02d}", source_id, document.sha256, facet.facet_id, start, end
                )
            )
            if len(result) == MAX_PASSAGES:
                return tuple(result)
    return tuple(result)


def bundle_from_json(value: str) -> EvidenceBundleV1:
    raw = json.loads(value)
    sources = tuple(
        EvidenceSourceV1(
            **{
                **item,
                "date_provenance": DateProvenance(item["date_provenance"]),
                "source_role": SourceRole(item["source_role"]),
            }
        )
        for item in raw["sources"]
    )
    claims = tuple(
        EvidenceClaimV1(
            **{
                **item,
                "anchor": EvidenceLocatorV1(**item["anchor"]),
                "corroborating": EvidenceLocatorV1(**item["corroborating"]) if item.get("corroborating") else None,
            }
        )
        for item in raw["claims"]
    )
    facets = tuple(FacetResultV1(**{**item, "state": FacetState(item["state"]), "claim_ids": tuple(item["claim_ids"])}) for item in raw["facets"])
    return EvidenceBundleV1(raw["plan_hash"], ResearchStatus(raw["status"]), facets, sources, claims, raw["bundle_version"])


def corpus_from_json(value: str) -> ResearchCorpusV1:
    raw = json.loads(value)
    if not isinstance(raw, dict) or set(raw) != {
        "plan_hash",
        "sources",
        "documents",
        "passages",
        "technical_failure_facets",
        "network_retry_count",
    }:
        raise ValueError("research corpus fields are invalid")
    source_fields = set(EvidenceSourceV1.__dataclass_fields__)
    passage_fields = set(RankedPassageV1.__dataclass_fields__)
    document_fields = set(NormalizedDocumentV1.__dataclass_fields__)
    if (
        not isinstance(raw["sources"], list)
        or any(not isinstance(item, dict) or set(item) != source_fields for item in raw["sources"])
        or not isinstance(raw["documents"], dict)
        or any(not isinstance(item, dict) or set(item) != document_fields for item in raw["documents"].values())
        or not isinstance(raw["passages"], list)
        or any(not isinstance(item, dict) or set(item) != passage_fields for item in raw["passages"])
    ):
        raise ValueError("research corpus nested fields are invalid")
    sources = tuple(
        EvidenceSourceV1(
            **{
                **item,
                "date_provenance": DateProvenance(item["date_provenance"]),
                "source_role": SourceRole(item["source_role"]),
            }
        )
        for item in raw["sources"]
    )
    documents = {
        source_id: NormalizedDocumentV1(**document)
        for source_id, document in raw["documents"].items()
    }
    passages = tuple(RankedPassageV1(**item) for item in raw["passages"])
    return ResearchCorpusV1(
        raw["plan_hash"],
        sources,
        documents,
        passages,
        tuple(raw["technical_failure_facets"]),
        raw["network_retry_count"],
    )


def plan_from_json(value: str) -> SearchPlanV1:
    raw = json.loads(value)
    return SearchPlanV1(
        raw["canonical_query"],
        tuple(SearchFacetV1(**item) for item in raw["facets"]),
        tuple(raw["subqueries"]),
        raw["language"],
        raw["freshness"],
        raw["timezone"],
        raw["page_limit"],
        raw["deadline_seconds"],
        raw["browser_fallback_allowed"],
        raw["version"],
    )


def document_from_json(value: str) -> NormalizedDocumentV1:
    return NormalizedDocumentV1(**json.loads(value))


def fail_closed(code: str) -> OrchestratorError:
    return OrchestratorError(code, status=409, retryable=False)
