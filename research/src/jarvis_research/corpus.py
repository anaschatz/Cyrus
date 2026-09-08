"""Deterministic ResearchCorpusV1 construction from fetched documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from publicsuffixlist import PublicSuffixList

from jarvis_orchestrator.research import (
    EvidenceSourceV1,
    ResearchCorpusV1,
    SearchPlanV1,
    SourceFingerprintV1,
    SourceRole,
    accepted_canonical_url,
    assign_independence_groups,
    select_ranked_passages,
    validate_public_url,
)
from jarvis_orchestrator.research_security import language_eligibility

from .adapters import RetrievalCorpusV1
from .config import ASSET_ROOT, file_sha256, load_json_object
from .extraction import extract_main_content

PSL_VERSION = "publicsuffixlist/1.0.2.20260726"
PSL_SHA256 = "40f3bcfa9aef508abb8f879fbe6b2177c9e09dc8c67c01a4971b7020504a0a80"
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_GREEK = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True, slots=True)
class LanguageResultV1:
    code: str
    confidence: float


class PinnedLanguageDetectorV1:
    def __init__(self, artifact: Path = ASSET_ROOT / "language-detector-v1.json") -> None:
        config = load_json_object(artifact)
        self.version = str(config["version"])
        self.sha256 = file_sha256(artifact)
        self._english = frozenset(str(item) for item in config["english_markers"])
        self._unsupported = frozenset(str(item) for item in config["unsupported_markers"])

    def detect(self, text: str) -> LanguageResultV1:
        # ponytail: script/marker heuristic, not a calibrated language model;
        # replace only with a qualified detector and new artifact identity.
        sample = text.encode("utf-8")[:4096].decode("utf-8", errors="ignore")
        greek = len(_GREEK.findall(sample))
        latin = len(_LATIN.findall(sample))
        total = greek + latin
        if total == 0:
            return LanguageResultV1("und", 0.0)
        if greek / total >= 0.80:
            return LanguageResultV1("el", greek / total)
        tokens = [match.group(0).casefold() for match in _WORD.finditer(sample)]
        english = sum(token in self._english for token in tokens)
        unsupported = sum(token in self._unsupported for token in tokens)
        if unsupported == 0 and latin / total >= 0.80:
            return LanguageResultV1("en", min(1.0, 0.80 + 0.025 * english))
        return LanguageResultV1("und", 0.0)


class PublicSuffixRegistryV1:
    def __init__(self) -> None:
        self.version = PSL_VERSION
        package_root = Path(__import__("publicsuffixlist").__file__).parent
        artifacts = tuple(sorted(package_root.glob("*.dat")))
        if len(artifacts) != 1:
            raise ValueError("pinned PSL artifact is unavailable")
        artifact = artifacts[0]
        self.sha256 = file_sha256(artifact)
        if self.sha256 != PSL_SHA256:
            raise ValueError("pinned PSL digest mismatch")
        self._psl = PublicSuffixList(artifact.read_bytes())

    def registrable_domain(self, url: str) -> str:
        host = urlsplit(validate_public_url(url)).hostname or ""
        result = self._psl.privatesuffix(host.casefold())
        if result is None:
            raise ValueError("registrable domain is unavailable")
        return result


class AuthorityRegistryV1:
    def __init__(self, artifact: Path = ASSET_ROOT / "authority-registry-v1.json") -> None:
        value = load_json_object(artifact)
        if set(value) != {"default_role", "primary_rules", "version"} or value["default_role"] != "SECONDARY":
            raise ValueError("authority registry is invalid")
        rules = value["primary_rules"]
        if not isinstance(rules, list) or any(not isinstance(item, dict) or set(item) != {"domain"} for item in rules):
            raise ValueError("authority registry rules are invalid")
        self.version = str(value["version"])
        self.sha256 = file_sha256(artifact)
        self._primary = frozenset(str(item["domain"]).casefold() for item in rules)

    def role(self, registrable_domain: str) -> SourceRole:
        primary = any(
            registrable_domain == rule or registrable_domain.endswith("." + rule)
            for rule in self._primary
        )
        return SourceRole.PRIMARY if primary else SourceRole.SECONDARY


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class CorpusBuilderV1:
    def __init__(
        self,
        *,
        detector: PinnedLanguageDetectorV1 | None = None,
        psl: PublicSuffixRegistryV1 | None = None,
        authority: AuthorityRegistryV1 | None = None,
        clock=_utc_now,
    ) -> None:
        self.detector = detector or PinnedLanguageDetectorV1()
        self.psl = psl or PublicSuffixRegistryV1()
        self.authority = authority or AuthorityRegistryV1()
        self._clock = clock

    def build(self, plan: SearchPlanV1, retrieval: RetrievalCorpusV1, *, network_retry_count: int) -> ResearchCorpusV1:
        staged: list[tuple[str, object, object, str, str]] = []
        total_bytes = 0
        exact_by_hash: dict[str, str] = {}
        fetched_hashes: dict[str, set[str]] = {}
        for fetched in retrieval.fetched[: plan.page_limit]:
            try:
                content = extract_main_content(fetched.body, fetched.mime_type)
                if content.document.character_length < 200 or total_bytes + content.document.byte_length > 512 * 1024:
                    continue
                detected = self.detector.detect(content.document.text)
                eligible, _ = language_eligibility(
                    sample=content.document.text, detected_code=detected.code, confidence=detected.confidence
                )
                if not eligible:
                    continue
                domain = self.psl.registrable_domain(fetched.final_url)
                fetched_hashes.setdefault(content.document.sha256, set()).add(domain)
                if content.document.sha256 in exact_by_hash:
                    continue
                source_id = "src-" + hashlib.sha256(
                    (fetched.final_url + "\n" + content.document.sha256).encode("utf-8")
                ).hexdigest()[:24]
                exact_by_hash[content.document.sha256] = source_id
                staged.append((source_id, fetched, content, domain, detected.code + ":" + format(detected.confidence, ".8f")))
                total_bytes += content.document.byte_length
            except (UnicodeError, ValueError):
                continue
        fingerprints = tuple(
            SourceFingerprintV1(source_id, domain, content.title, content.publisher, content.published_at, content.document)
            for source_id, _, content, domain, _ in staged
        )
        groups = assign_independence_groups(fingerprints)
        sources: list[EvidenceSourceV1] = []
        documents = {}
        for source_id, fetched, content, domain, language in staged:
            code, confidence = language.split(":", 1)
            canonical: str | None = None
            if content.canonical_url:
                try:
                    canonical = validate_public_url(urljoin(fetched.final_url, content.canonical_url))
                except ValueError:
                    canonical = None
            canonical_domain = None
            if canonical:
                try:
                    canonical_domain = self.psl.registrable_domain(canonical)
                except ValueError:
                    canonical = None
            accepted = accepted_canonical_url(
                fetched.final_url,
                canonical,
                same_registrable_domain=canonical_domain == domain,
                content_matches=bool(canonical_domain and canonical_domain in fetched_hashes.get(content.document.sha256, set())),
            )
            sources.append(
                EvidenceSourceV1(
                    source_id=source_id,
                    requested_url=fetched.requested_url,
                    final_url=fetched.final_url,
                    accepted_canonical_url=accepted,
                    title=content.title,
                    publisher=content.publisher,
                    published_at=content.published_at,
                    date_provenance=content.date_provenance,
                    retrieved_at=self._clock(),
                    source_role=self.authority.role(domain),
                    authority_registry_version=self.authority.version,
                    registrable_domain=domain,
                    independence_group_id=groups[source_id],
                    normalized_document_hash=content.document.sha256,
                    language_code=code,
                    language_detector_version=self.detector.version,
                    language_confidence=float(confidence),
                )
            )
            documents[source_id] = content.document
        passages = select_ranked_passages(documents, plan.facets)
        technical = tuple(facet.facet_id for facet in plan.facets) if not sources and retrieval.technical_failures else ()
        return ResearchCorpusV1(plan.plan_hash, tuple(sources), documents, passages, technical, network_retry_count)
