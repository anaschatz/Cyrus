"""Code-owned provenance and egress policy for tool-derived content."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .jcs import domain_sha256, jcs_bytes


PROVENANCE_CONTRACT = "jarvis.content-provenance/1"


class ProvenanceError(ValueError):
    pass


class ContentSourceKind(StrEnum):
    USER_INPUT = "USER_INPUT"
    LOCAL_MODEL = "LOCAL_MODEL"
    CLOUD_MODEL = "CLOUD_MODEL"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_TEMPLATE = "TOOL_TEMPLATE"
    MEMORY_RECALL = "MEMORY_RECALL"
    RESEARCH_EVIDENCE = "RESEARCH_EVIDENCE"
    CODE_TEMPLATE = "CODE_TEMPLATE"
    LEGACY_UNVERIFIED = "LEGACY_UNVERIFIED"


class PolicyLabel(StrEnum):
    STANDARD = "STANDARD"
    LOCAL_PRIVATE = "LOCAL_PRIVATE"


class EgressDisposition(StrEnum):
    REVIEWABLE = "REVIEWABLE"
    NO_EGRESS = "NO_EGRESS"


class EgressDestination(StrEnum):
    LOCAL_MODEL = "LOCAL_MODEL"
    LOCAL_UI_VOICE = "LOCAL_UI_VOICE"
    CLOUD_MODEL = "CLOUD_MODEL"
    PUBLIC_RESEARCH = "PUBLIC_RESEARCH"
    LONG_TERM_MEMORY = "LONG_TERM_MEMORY"
    CLOUD_COMPACTION = "CLOUD_COMPACTION"


_PRIVATE_SOURCES = frozenset(
    {
        ContentSourceKind.TOOL_RESULT,
        ContentSourceKind.TOOL_TEMPLATE,
        ContentSourceKind.MEMORY_RECALL,
        ContentSourceKind.RESEARCH_EVIDENCE,
        ContentSourceKind.LEGACY_UNVERIFIED,
    }
)
_LOCAL_DESTINATIONS = frozenset({EgressDestination.LOCAL_MODEL, EgressDestination.LOCAL_UI_VOICE})


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ProvenanceError("invalid SHA-256")
    return value


@dataclass(frozen=True, slots=True, order=True)
class ProvenanceRefV1:
    kind: ContentSourceKind
    source_id: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not self.source_id or len(self.source_id.encode("utf-8")) > 256:
            raise ProvenanceError("invalid provenance source id")
        _sha256(self.content_sha256)

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "source_id": self.source_id, "content_sha256": self.content_sha256}


@dataclass(frozen=True, slots=True)
class ContentProvenanceV1:
    sources: tuple[ProvenanceRefV1, ...]
    policy_label: PolicyLabel
    egress: EgressDisposition

    def __post_init__(self) -> None:
        if not self.sources or self.sources != tuple(sorted(set(self.sources))):
            raise ProvenanceError("provenance sources must be non-empty, unique, and sorted")
        contains_private = any(source.kind in _PRIVATE_SOURCES for source in self.sources)
        if contains_private and (
            self.policy_label is not PolicyLabel.LOCAL_PRIVATE
            or self.egress is not EgressDisposition.NO_EGRESS
        ):
            raise ProvenanceError("private provenance cannot be downgraded")

    @classmethod
    def source(cls, kind: ContentSourceKind, source_id: str, content: str) -> "ContentProvenanceV1":
        private = kind in _PRIVATE_SOURCES
        return cls(
            (ProvenanceRefV1(kind, source_id, hashlib.sha256(content.encode("utf-8")).hexdigest()),),
            PolicyLabel.LOCAL_PRIVATE if private else PolicyLabel.STANDARD,
            EgressDisposition.NO_EGRESS if private else EgressDisposition.REVIEWABLE,
        )

    @classmethod
    def tool_task_output(
        cls,
        *,
        task_id: str,
        content: str,
        inputs: Iterable["ContentProvenanceV1"] = (),
        deterministic: bool = False,
    ) -> "ContentProvenanceV1":
        kind = ContentSourceKind.TOOL_TEMPLATE if deterministic else ContentSourceKind.TOOL_RESULT
        own = cls.source(kind, task_id, content)
        return cls.derive((*tuple(inputs), own))

    @classmethod
    def derive(cls, inputs: Iterable["ContentProvenanceV1"]) -> "ContentProvenanceV1":
        selected = tuple(inputs)
        if not selected:
            raise ProvenanceError("derived provenance requires inputs")
        sources = tuple(sorted({source for item in selected for source in item.sources}))
        private = any(
            item.policy_label is PolicyLabel.LOCAL_PRIVATE
            or item.egress is EgressDisposition.NO_EGRESS
            or any(source.kind in _PRIVATE_SOURCES for source in item.sources)
            for item in selected
        )
        return cls(
            sources,
            PolicyLabel.LOCAL_PRIVATE if private else PolicyLabel.STANDARD,
            EgressDisposition.NO_EGRESS if private else EgressDisposition.REVIEWABLE,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ContentProvenanceV1":
        if set(raw) != {"sources", "policy_label", "egress"} or not isinstance(raw["sources"], list):
            raise ProvenanceError("invalid provenance object")
        try:
            sources = tuple(
                ProvenanceRefV1(
                    ContentSourceKind(item["kind"]),
                    item["source_id"],
                    item["content_sha256"],
                )
                for item in raw["sources"]
                if isinstance(item, Mapping) and set(item) == {"kind", "source_id", "content_sha256"}
            )
            if len(sources) != len(raw["sources"]):
                raise ProvenanceError("invalid provenance source")
            return cls(sources, PolicyLabel(raw["policy_label"]), EgressDisposition(raw["egress"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProvenanceError("invalid provenance object") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [source.to_dict() for source in self.sources],
            "policy_label": self.policy_label.value,
            "egress": self.egress.value,
        }

    @property
    def sha256(self) -> str:
        return domain_sha256(PROVENANCE_CONTRACT, self.to_dict())


@dataclass(frozen=True, slots=True)
class LabeledTextV1:
    text: str
    provenance: ContentProvenanceV1

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class EgressGuardV1:
    @staticmethod
    def assert_allowed(destination: EgressDestination, provenance: ContentProvenanceV1) -> None:
        if destination in _LOCAL_DESTINATIONS:
            return
        if provenance.egress is EgressDisposition.NO_EGRESS:
            raise ProvenanceError("content_no_egress")
        if any(source.kind in _PRIVATE_SOURCES for source in provenance.sources):
            raise ProvenanceError("content_no_egress")


def assert_side_effect_intent_binding(
    *,
    argument_provenance: ContentProvenanceV1,
    user_intent_binding_sha256: str,
    confirmed_physical_user_turn_sha256: str,
) -> None:
    """Block tool-read prompt injection from becoming write authority.

    A side effect whose material arguments contain TOOL_RESULT provenance is
    authorized only by a distinct, hash-bound physical user confirmation.  An
    approval click cannot substitute for that new utterance.
    """

    _sha256(user_intent_binding_sha256)
    if confirmed_physical_user_turn_sha256:
        _sha256(confirmed_physical_user_turn_sha256)
    derived_from_tool = any(
        source.kind is ContentSourceKind.TOOL_RESULT
        for source in argument_provenance.sources
    )
    if derived_from_tool and (
        not confirmed_physical_user_turn_sha256
        or confirmed_physical_user_turn_sha256 == user_intent_binding_sha256
    ):
        raise ProvenanceError("tool_explicit_user_confirmation_required")


def provenance_manifest_sha256(messages: Iterable[LabeledTextV1]) -> str:
    value = [
        {
            "content_sha256": item.content_sha256,
            "provenance_sha256": item.provenance.sha256,
        }
        for item in messages
    ]
    return hashlib.sha256(jcs_bytes(value)).hexdigest()
