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


_RULES: tuple[tuple[PrivacyClass, re.Pattern[str]], ...] = (
    (PrivacyClass.SECRET_CREDENTIAL, re.compile(r"\b(?:api[_ -]?key|secret|password|passwd|bearer|private[_ -]?key|otp|pin)\b|\bsk-[a-z0-9_-]{8,}\b")),
    (PrivacyClass.SENSITIVE_FINANCIAL, re.compile(r"\b(?:iban|swift|credit card|debit card|bank account|τραπεζ|πιστωτικ|χρεωστικ)\b|\b\d{13,19}\b")),
    (PrivacyClass.SENSITIVE_HEALTH, re.compile(r"\b(?:diagnosis|disease|medication|patient|therapy|health record|διάγνωση|ασθεν|φάρμακ|θεραπε)\b")),
    (PrivacyClass.SENSITIVE_LEGAL, re.compile(r"\b(?:lawsuit|criminal|attorney-client|legal case|δικογραφ|ποινικ|δικηγορ)\b")),
    (PrivacyClass.BIOMETRIC, re.compile(r"\b(?:fingerprint|faceprint|retina|biometric|δακτυλικ|βιομετρ)\b")),
    (PrivacyClass.SENSITIVE_INTIMATE_OR_MINOR, re.compile(r"\b(?:sexual|intimate|minor child|underage|ανήλικ|σεξουαλ)\b")),
    (PrivacyClass.SENSITIVE_IDENTITY, re.compile(r"\b(?:passport|social security|national id|tax id|αφμ|αμκα|ταυτότητ|διαβατήρ)\b")),
    (PrivacyClass.SENSITIVE_CONTACT_LOCATION, re.compile(r"\b(?:home address|postal address|phone number|gps|διεύθυνση κατοικίας|τηλέφωνο|τοποθεσία)\b|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")),
)


class JarvisPrivacyClassifierV1:
    version = "jarvis-privacy/1"

    def __init__(self) -> None:
        self.policy_sha256 = domain_sha256(
            "privacy-policy/1",
            {
                "eligible": sorted(item.value for item in CLOUD_ELIGIBLE_CLASSES),
                "normalization": "NFC+casefold+whitespace",
                "rules": [classification.value for classification, _ in _RULES],
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
