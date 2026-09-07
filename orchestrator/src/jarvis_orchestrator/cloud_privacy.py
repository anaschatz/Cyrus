"""Deterministic local-only privacy classification for cloud egress."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from .jcs import domain_sha256


class PrivacyClass(StrEnum):
    PUBLIC = "PUBLIC"
    PERSONAL_LOW_RISK = "PERSONAL_LOW_RISK"
    SENSITIVE_IDENTITY = "SENSITIVE_IDENTITY"
    SENSITIVE_CONTACT_LOCATION = "SENSITIVE_CONTACT_LOCATION"
    SENSITIVE_HEALTH = "SENSITIVE_HEALTH"
    SENSITIVE_FINANCIAL = "SENSITIVE_FINANCIAL"
    SENSITIVE_LEGAL = "SENSITIVE_LEGAL"
    SENSITIVE_INTIMATE_OR_MINOR = "SENSITIVE_INTIMATE_OR_MINOR"
    BIOMETRIC = "BIOMETRIC"
    SECRET_CREDENTIAL = "SECRET_CREDENTIAL"
    MEMORY_DERIVED = "MEMORY_DERIVED"
    RESEARCH_EVIDENCE = "RESEARCH_EVIDENCE"
    UNKNOWN = "UNKNOWN"


CLOUD_ELIGIBLE_CLASSES = frozenset({PrivacyClass.PUBLIC, PrivacyClass.PERSONAL_LOW_RISK})


@dataclass(frozen=True, slots=True)
class PrivacyResultV1:
    version: str
    classification: PrivacyClass
    normalized_sha256: str
    policy_sha256: str

    @property
    def cloud_eligible(self) -> bool:
        return self.classification in CLOUD_ELIGIBLE_CLASSES


def normalize_egress(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _rule(pattern: str) -> re.Pattern[str]:
    """Compile literal-language rules under the same Unicode policy as input."""

    return re.compile(unicodedata.normalize("NFC", pattern).casefold())


_RULES: tuple[tuple[PrivacyClass, re.Pattern[str]], ...] = (
    (PrivacyClass.SECRET_CREDENTIAL, _rule(r"\b(?:api[_ -]?key|secret|password|passwd|bearer|private[_ -]?key|otp|pin|κωδικ(?:[οό]σ?|ο[υύ]σ?|ο[ιί]|[ωώ]ν)|συνθηματικ\w*|ιδιωτικ\w* κλειδ\w*)\b|\bsk-[a-z0-9_-]{8,}\b")),
    (PrivacyClass.SENSITIVE_FINANCIAL, _rule(r"\b(?:iban|swift|credit card|debit card|bank account|τραπεζ\w*|πιστωτικ\w*|χρεωστικ\w*)\b|\b\d{13,19}\b")),
    (PrivacyClass.SENSITIVE_HEALTH, _rule(r"\b(?:diagnosis|disease|medication|patient|therapy|health record|διάγνωση|ασθεν\w*|φάρμακ\w*|θεραπε\w*)\b")),
    (PrivacyClass.SENSITIVE_LEGAL, _rule(r"\b(?:lawsuit|criminal|attorney-client|legal case|δικογραφ\w*|ποινικ\w*|δικηγορ\w*)\b")),
    (PrivacyClass.BIOMETRIC, _rule(r"\b(?:fingerprint|faceprint|retina|biometric|δακτυλικ\w*|βιομετρ\w*)\b")),
    (PrivacyClass.SENSITIVE_INTIMATE_OR_MINOR, _rule(r"\b(?:sexual|intimate|minor child|underage|ανήλικ\w*|σεξουαλ\w*)\b")),
    (PrivacyClass.SENSITIVE_IDENTITY, _rule(r"\b(?:passport|social security|national id|tax id|αφμ|αμκα|ταυτότητ\w*|διαβατήρ\w*)\b")),
    (PrivacyClass.SENSITIVE_CONTACT_LOCATION, _rule(r"\b(?:home address|postal address|phone number|gps|διεύθυνση κατοικίας|τηλέφωνο|τοποθεσία)\b|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")),
)


class JarvisPrivacyClassifierV1:
    version = "jarvis-privacy/1"

    def __init__(self) -> None:
        self.policy_sha256 = domain_sha256(
            "privacy-policy/1",
            {
                "eligible": sorted(item.value for item in CLOUD_ELIGIBLE_CLASSES),
                "normalization": "NFC+casefold+whitespace",
                "rules": [
                    {"classification": classification.value, "pattern": pattern.pattern}
                    for classification, pattern in _RULES
                ],
                "version": self.version,
            },
        )

    def classify(
        self,
        text: str,
        *,
        memory_derived: bool = False,
        research_evidence: bool = False,
    ) -> PrivacyResultV1:
        normalized = normalize_egress(text)
        digest = domain_sha256("combined-egress-text/1", {"text": normalized})
        if memory_derived:
            classification = PrivacyClass.MEMORY_DERIVED
        elif research_evidence:
            classification = PrivacyClass.RESEARCH_EVIDENCE
        elif not normalized:
            classification = PrivacyClass.UNKNOWN
        else:
            classification = PrivacyClass.PUBLIC
            for candidate, pattern in _RULES:
                if pattern.search(normalized):
                    classification = candidate
                    break
        return PrivacyResultV1(self.version, classification, digest, self.policy_sha256)

    async def classify_bounded(
        self,
        parts: Iterable[str],
        *,
        memory_derived: bool = False,
        research_evidence: bool = False,
        timeout_seconds: float = 0.05,
    ) -> PrivacyResultV1:
        combined = "\n".join(parts)
        try:
            async with asyncio.timeout(timeout_seconds):
                return await asyncio.to_thread(
                    self.classify,
                    combined,
                    memory_derived=memory_derived,
                    research_evidence=research_evidence,
                )
        except Exception:
            # The closed taxonomy requires fail-closed UNKNOWN for every local
            # classifier failure.  Do not expose exception content.
            normalized = normalize_egress(combined)
            return PrivacyResultV1(
                self.version,
                PrivacyClass.UNKNOWN,
                domain_sha256("combined-egress-text/1", {"text": normalized}),
                self.policy_sha256,
            )
