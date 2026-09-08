"""Immutable JARVIS v1 memory release boundary.

The general prospective-memory writer remains an experiment.  This module is
the code-owned authority that keeps model-produced proposals audit-only and
prevents them from becoming action evidence or automatic side effects.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any

from .domain import canonical_sha256

MEMORY_RELEASE_POLICY_VERSION = "jarvis.memory-release-policy.v1"
_MEMORY_RELEASE_POLICY_DOCUMENT = {
    "version": MEMORY_RELEASE_POLICY_VERSION,
    "production_v1": {
        "retrospective_memory_writes": "explicit_commands_only",
        "prospective_reminders": "explicit_deterministic_only",
        "generative_compilers": "not_qualified",
        "automatic_side_effects": False,
        "action_evidence_from_memory": False,
    },
    "experimental": {
        "generative_output": "shadow_audit_only",
        "authoritative_writes": False,
        "automatic_side_effects": False,
        "human_review_required": True,
    },
    "promotion_requires": (
        "new_writer_or_stricter_constrained_generation",
        "fresh_unseen_evaluation_sets",
        "multi_week_shadow_trial",
        "very_high_precision",
        "complete_trigger_provenance",
        "pending_trigger_and_cancellation_ui",
    ),
}
MEMORY_RELEASE_POLICY_SHA256 = canonical_sha256(_MEMORY_RELEASE_POLICY_DOCUMENT)


def _freeze_policy(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_policy(item) for key, item in value.items()}
        )
    if isinstance(value, tuple):
        return tuple(_freeze_policy(item) for item in value)
    return value


MEMORY_RELEASE_POLICY: Mapping[str, Any] = _freeze_policy(
    _MEMORY_RELEASE_POLICY_DOCUMENT
)
del _MEMORY_RELEASE_POLICY_DOCUMENT

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMON_SAFETY_FIELDS = {
    "release_policy_version": MEMORY_RELEASE_POLICY_VERSION,
    "release_policy_sha256": MEMORY_RELEASE_POLICY_SHA256,
    "authoritative_state_changed": False,
    "automatic_side_effect_requested": False,
    "action_evidence_emitted": False,
}


class ShadowHistoryReaderFacade:
    """Expose only a defensive, read-only shadow-history projection.

    The production composition root must pass this facade to experimental
    writers instead of giving them the authoritative ``OrchestratorStore``.
    """

    __slots__ = ("__read_history",)

    def __init__(
        self,
        read_history: Callable[[str], list[dict[str, Any]]],
    ) -> None:
        if not callable(read_history):
            raise TypeError("shadow history reader must be callable")
        self.__read_history = read_history

    def memory_shadow_history(self, conversation_id: str) -> list[dict[str, Any]]:
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("shadow history conversation is invalid")
        history = self.__read_history(conversation_id)
        if not isinstance(history, list) or any(
            not isinstance(item, dict) for item in history
        ):
            raise TypeError("shadow history projection is invalid")
        return deepcopy(history)


def shadow_safety_fields() -> dict[str, object]:
    """Return a fresh copy of the mandatory audit-only envelope fields."""

    return dict(_COMMON_SAFETY_FIELDS)


def validate_shadow_audit_payload(
    event_type: str,
    payload: Mapping[str, Any],
) -> None:
    """Reject any shadow result that weakens the v1 no-side-effect boundary."""

    for key, expected in _COMMON_SAFETY_FIELDS.items():
        if type(payload.get(key)) is not type(expected) or payload.get(key) != expected:
            raise ValueError(f"memory shadow safety field is invalid: {key}")
    source_event_id = payload.get("source_event_id")
    text_sha256 = payload.get("text_sha256")
    if not isinstance(source_event_id, str) or not source_event_id:
        raise ValueError("memory shadow source event is invalid")
    if not isinstance(text_sha256, str) or not _SHA256.fullmatch(text_sha256):
        raise ValueError("memory shadow source hash is invalid")

    if event_type == "memory.shadow.failed":
        expected_keys = set(_COMMON_SAFETY_FIELDS) | {
            "status",
            "source_event_id",
            "text_sha256",
            "error_code",
            "authority_state_before_sha256",
            "authority_state_after_sha256",
        }
        if set(payload) != expected_keys or payload.get("status") != "failed":
            raise ValueError("memory shadow failure shape is invalid")
        before = payload.get("authority_state_before_sha256")
        after = payload.get("authority_state_after_sha256")
        if (
            not isinstance(before, str)
            or not _SHA256.fullmatch(before)
            or after != before
        ):
            raise ValueError("memory shadow authority fingerprint changed")
        error_code = payload.get("error_code")
        if error_code not in ("timeout", "service_error"):
            raise ValueError("memory shadow failure code is invalid")
        return

    if event_type != "memory.shadow.proposed":
        raise ValueError("memory shadow audit event type is invalid")
    expected_keys = set(_COMMON_SAFETY_FIELDS) | {
        "status",
        "source_event_id",
        "text_sha256",
        "proposal",
        "proposal_sha256",
        "provenance",
        "authority_state_before_sha256",
        "authority_state_after_sha256",
    }
    if set(payload) != expected_keys or payload.get("status") != "proposed":
        raise ValueError("memory shadow proposal envelope is invalid")
    before = payload.get("authority_state_before_sha256")
    after = payload.get("authority_state_after_sha256")
    if not isinstance(before, str) or not _SHA256.fullmatch(before) or after != before:
        raise ValueError("memory shadow authority fingerprint changed")
    proposal = payload.get("proposal")
    proposal_sha256 = payload.get("proposal_sha256")
    if not isinstance(proposal, Mapping):
        raise TypeError("memory shadow proposal is invalid")
    if proposal_sha256 != canonical_sha256(proposal):
        raise ValueError("memory shadow proposal hash is invalid")
    writer_contract = proposal.get("writer_contract_sha256")
    supersedes = proposal.get("supersedes_source_event_ids")
    if not isinstance(writer_contract, str) or not _SHA256.fullmatch(writer_contract):
        raise ValueError("memory shadow writer contract is invalid")
    if not isinstance(supersedes, list) or any(
        not isinstance(value, str) or not value for value in supersedes
    ):
        raise ValueError("memory shadow supersession evidence is invalid")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "current_source_event_id",
        "source_text_sha256",
        "superseded_source_event_ids",
        "writer_contract_sha256",
        "proposal_sha256",
    }:
        raise ValueError("memory shadow provenance shape is invalid")
    if provenance != {
        "current_source_event_id": source_event_id,
        "source_text_sha256": text_sha256,
        "superseded_source_event_ids": supersedes,
        "writer_contract_sha256": writer_contract,
        "proposal_sha256": proposal_sha256,
    }:
        raise ValueError("memory shadow provenance binding is invalid")


__all__ = [
    "MEMORY_RELEASE_POLICY",
    "MEMORY_RELEASE_POLICY_SHA256",
    "MEMORY_RELEASE_POLICY_VERSION",
    "ShadowHistoryReaderFacade",
    "shadow_safety_fields",
    "validate_shadow_audit_payload",
]
