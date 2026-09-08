"""Fail-closed release qualification for the real OpenRouter cloud path."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cloud_contract import DEEPSEEK_MODEL, TERRA_MODEL, QualifiedEndpointV1
from .cloud_privacy import JarvisPrivacyClassifierV1, PrivacyClass
from .jcs import domain_sha256
from .config import _parse_strict_json


CLOUD_RELEASE_VERSION = "cloud-release-qualification/1"
SHADOW_MINIMUM_SECONDS = 30 * 24 * 60 * 60
QUALIFICATION_MAXIMUM_VALIDITY = timedelta(days=31)
MONTHLY_KEY_LIMIT_MICROUSD = 5_000_000


class CloudQualificationError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise CloudQualificationError(code)


def _exact(value: Any, keys: set[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(code)
    return value


def _digest(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(code)
    return value


def _utc(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        selected = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail(code)
    if selected.tzinfo is None or selected.utcoffset() != timedelta(0):
        _fail(code)
    return selected.astimezone(UTC)


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def dedicated_key_sha256(token: str) -> str:
    try:
        if not isinstance(token, str) or not token or len(token.encode("utf-8")) > 512:
            _fail("cloud_key_invalid")
        return domain_sha256("openrouter-dedicated-key/1", {"token": token})
    except (ValueError, UnicodeError) as exc:
        raise CloudQualificationError("cloud_key_invalid") from exc


def _read_release(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        with open(path, "rb", opener=lambda name, flags: os.open(
            name, flags | getattr(os, "O_NONBLOCK", 0)
        )) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                _fail("cloud_release_evidence_unavailable")
            raw = stream.read(1024 * 1024 + 1)
        if not raw or len(raw) > 1024 * 1024:
            _fail("cloud_release_evidence_unavailable")
        document = _parse_strict_json(raw)
        if not isinstance(document, Mapping):
            _fail("cloud_release_evidence_invalid")
        return document, domain_sha256(CLOUD_RELEASE_VERSION, document)
    except (OSError, ValueError, RecursionError) as exc:
        raise CloudQualificationError("cloud_release_evidence_unavailable") from exc


def cloud_release_sha256(path: str | Path) -> str:
    return _read_release(Path(path))[1]


_PRIVACY_FIXTURES: Mapping[str, tuple[tuple[str, PrivacyClass], ...]] = {
    "english": (
        ("Public product documentation", PrivacyClass.PUBLIC),
        ("My password is not for external processing", PrivacyClass.SECRET_CREDENTIAL),
        ("The patient diagnosis is private", PrivacyClass.SENSITIVE_HEALTH),
        ("My home address is private", PrivacyClass.SENSITIVE_CONTACT_LOCATION),
        ("My passport number is private", PrivacyClass.SENSITIVE_IDENTITY),
    ),
    "greek": (
        ("Δημόσια τεκμηρίωση προϊόντος", PrivacyClass.PUBLIC),
        ("Το ΑΦΜ μου είναι ιδιωτικό", PrivacyClass.SENSITIVE_IDENTITY),
        ("Η διάγνωση του ασθενούς είναι ιδιωτική", PrivacyClass.SENSITIVE_HEALTH),
        ("Το password μου είναι ιδιωτικό", PrivacyClass.SECRET_CREDENTIAL),
        ("Η διεύθυνση κατοικίας μου είναι ιδιωτική", PrivacyClass.SENSITIVE_CONTACT_LOCATION),
    ),
    "mixed": (
        ("Public τεκμηρίωση προϊόντος", PrivacyClass.PUBLIC),
        ("My API key είναι ιδιωτικό", PrivacyClass.SECRET_CREDENTIAL),
        ("The patient έχει διάγνωση", PrivacyClass.SENSITIVE_HEALTH),
        ("My home διεύθυνση κατοικίας", PrivacyClass.SENSITIVE_CONTACT_LOCATION),
        ("Το passport number είναι ιδιωτικό", PrivacyClass.SENSITIVE_IDENTITY),
    ),
}


def run_cloud_privacy_qualification() -> dict[str, Any]:
    """Run frozen Greek, English and mixed combined-egress fixtures locally."""

    classifier = JarvisPrivacyClassifierV1()
    counts: dict[str, int] = {}
    failures = 0
    for language, fixtures in _PRIVACY_FIXTURES.items():
        counts[f"{language}_case_count"] = len(fixtures)
        for text, expected in fixtures:
            if classifier.classify(text).classification != expected:
                failures += 1
    combined = classifier.classify(
        "Public product documentation\nMy API key είναι ιδιωτικό"
    )
    counts["mixed_case_count"] += 1
    if combined.classification != PrivacyClass.SECRET_CREDENTIAL:
        failures += 1
    return {
        "classifier_version": classifier.version,
        **counts,
        "failure_count": failures,
        "policy_sha256": classifier.policy_sha256,
        "ready": failures == 0,
        "version": "cloud-privacy-qualification/1",
    }


@dataclass(frozen=True, slots=True)
class CloudReleaseQualificationV1:
    sha256: str
    qualified_at: datetime
    valid_until: datetime
    active_endpoints: tuple[QualifiedEndpointV1, ...]


def load_cloud_release_qualification(
    path: str | Path,
    *,
    expected_sha256: str,
    registry: Sequence[QualifiedEndpointV1],
    bearer_token: str,
    terra_enabled: bool,
    now: datetime | None = None,
) -> CloudReleaseQualificationV1:
    """Validate all non-live evidence before any cloud client is constructed."""

    selected = Path(path)
    if not selected.is_absolute():
        _fail("cloud_release_evidence_unavailable")
    document, actual_sha256 = _read_release(selected)
    top = _exact(
        document,
        {
            "endpoint_evidence",
            "faults",
            "key",
            "privacy",
            "qualified_at",
            "shadow",
            "valid_until",
            "version",
            "voice_approval",
        },
        "cloud_release_evidence_invalid",
    )
    if top["version"] != CLOUD_RELEASE_VERSION:
        _fail("cloud_release_evidence_invalid")
    if _digest(expected_sha256, "cloud_release_digest_invalid") != actual_sha256:
        _fail("cloud_release_digest_mismatch")
    qualified_at = _utc(top["qualified_at"], "cloud_release_time_invalid")
    valid_until = _utc(top["valid_until"], "cloud_release_time_invalid")
    selected_now = (now or datetime.now(UTC)).astimezone(UTC)
    if not qualified_at <= selected_now < valid_until:
        _fail("cloud_release_expired")
    if valid_until - qualified_at > QUALIFICATION_MAXIMUM_VALIDITY:
        _fail("cloud_release_validity_invalid")

    key = _exact(
        top["key"],
        {
            "dedicated",
            "include_byok_in_limit",
            "is_management_key",
            "is_provisioning_key",
            "key_sha256",
            "limit_microusd",
            "limit_reset",
            "observed_at",
        },
        "cloud_key_evidence_invalid",
    )
    if (
        key["dedicated"] is not True
        or key["include_byok_in_limit"] is not False
        or key["is_management_key"] is not False
        or key["is_provisioning_key"] is not False
        or key["limit_microusd"] != MONTHLY_KEY_LIMIT_MICROUSD
        or key["limit_reset"] != "monthly"
        or _digest(key["key_sha256"], "cloud_key_evidence_invalid")
        != dedicated_key_sha256(bearer_token)
    ):
        _fail("cloud_key_evidence_invalid")
    key_observed = _utc(key["observed_at"], "cloud_key_evidence_invalid")
    if abs(key_observed - qualified_at) > timedelta(minutes=10):
        _fail("cloud_key_evidence_invalid")

    voice = _exact(
        top["voice_approval"],
        {
            "duplicate_dispatch_count",
            "f10_applied_count",
            "f11_denied_count",
            "real_voice_client",
        },
        "cloud_voice_evidence_invalid",
    )
    if (
        voice["real_voice_client"] is not True
        or not _integer(voice["f10_applied_count"], minimum=1)
        or not _integer(voice["f11_denied_count"], minimum=1)
        or not _integer(voice["duplicate_dispatch_count"])
        or voice["duplicate_dispatch_count"] != 0
    ):
        _fail("cloud_voice_evidence_invalid")

    privacy = _exact(
        top["privacy"],
        {
            "classifier_version",
            "english_case_count",
            "failure_count",
            "greek_case_count",
            "mixed_case_count",
            "policy_sha256",
        },
        "cloud_privacy_evidence_invalid",
    )
    if (
        privacy["classifier_version"] != "jarvis-privacy/1"
        or any(
            not _integer(privacy[f"{language}_case_count"], minimum=len(fixtures) + (language == "mixed"))
            for language, fixtures in _PRIVACY_FIXTURES.items()
        )
        or not _integer(privacy["failure_count"])
        or privacy["failure_count"] != 0
    ):
        _fail("cloud_privacy_evidence_invalid")
    if (
        _digest(privacy["policy_sha256"], "cloud_privacy_evidence_invalid")
        != JarvisPrivacyClassifierV1().policy_sha256
    ):
        _fail("cloud_privacy_evidence_invalid")

    faults = _exact(
        top["faults"],
        {
            "disconnect_passed",
            "missing_usage_passed",
            "no_duplicate_dispatch_passed",
            "stale_catalog_passed",
            "timeout_passed",
        },
        "cloud_fault_evidence_invalid",
    )
    if any(faults[name] is not True for name in faults):
        _fail("cloud_fault_evidence_invalid")

    shadow = _exact(
        top["shadow"],
        {
            "budget_violation_count",
            "ended_at",
            "external_dispatch_count",
            "fallback_count",
            "maximum_ledger_microusd",
            "projected_reserved_microusd",
            "started_at",
            "turn_count",
        },
        "cloud_shadow_evidence_invalid",
    )
    shadow_started = _utc(shadow["started_at"], "cloud_shadow_evidence_invalid")
    shadow_ended = _utc(shadow["ended_at"], "cloud_shadow_evidence_invalid")
    if (
        (shadow_ended - shadow_started).total_seconds() < SHADOW_MINIMUM_SECONDS
        or not _integer(shadow["turn_count"], minimum=1)
        or not _integer(shadow["budget_violation_count"])
        or not _integer(shadow["external_dispatch_count"])
        or not _integer(shadow["fallback_count"])
        or shadow["budget_violation_count"] != 0
        or shadow["external_dispatch_count"] != 0
        or shadow["fallback_count"] != 0
        or not _integer(shadow["maximum_ledger_microusd"])
        or not 0 <= shadow["maximum_ledger_microusd"] <= MONTHLY_KEY_LIMIT_MICROUSD
        or not _integer(shadow["projected_reserved_microusd"])
        or not 0 <= shadow["projected_reserved_microusd"] <= MONTHLY_KEY_LIMIT_MICROUSD
        or shadow_ended > qualified_at
    ):
        _fail("cloud_shadow_evidence_invalid")

    raw_endpoint_evidence = top["endpoint_evidence"]
    if not isinstance(raw_endpoint_evidence, list):
        _fail("cloud_endpoint_evidence_invalid")
    evidence_by_identity: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
    endpoint_keys = {
        "actual_dispatch_count",
        "cost_reconciled",
        "model_canonical_id",
        "no_model_fallback",
        "no_provider_fallback",
        "parameter_profile_sha256",
        "provider_endpoint_tag",
        "qualification_sha256",
        "qualification_version",
        "quantization",
        "receipt_identity_verified",
        "request_snapshot_sha256",
        "response_snapshot_sha256",
        "zdr_catalog_verified",
    }
    for raw in raw_endpoint_evidence:
        evidence = _exact(raw, endpoint_keys, "cloud_endpoint_evidence_invalid")
        if any(not isinstance(evidence[name], str) for name in (
            "model_canonical_id", "provider_endpoint_tag", "quantization", "qualification_version"
        )):
            _fail("cloud_endpoint_evidence_invalid")
        identity = (
            evidence["model_canonical_id"],
            evidence["provider_endpoint_tag"],
            evidence["quantization"],
            _digest(evidence["parameter_profile_sha256"], "cloud_endpoint_evidence_invalid"),
            evidence["qualification_version"],
        )
        if identity in evidence_by_identity:
            _fail("cloud_endpoint_evidence_invalid")
        for name in (
            "qualification_sha256",
            "request_snapshot_sha256",
            "response_snapshot_sha256",
        ):
            _digest(evidence[name], "cloud_endpoint_evidence_invalid")
        if (
            not _integer(evidence["actual_dispatch_count"], minimum=1)
            or any(
                evidence[name] is not True
                for name in (
                    "cost_reconciled",
                    "no_model_fallback",
                    "no_provider_fallback",
                    "receipt_identity_verified",
                    "zdr_catalog_verified",
                )
            )
        ):
            _fail("cloud_endpoint_evidence_invalid")
        evidence_by_identity[identity] = evidence

    registry_by_model: dict[str, list[QualifiedEndpointV1]] = {}
    for endpoint in registry:
        endpoint.validate()
        if endpoint.model_canonical_id not in {DEEPSEEK_MODEL, TERRA_MODEL}:
            _fail("cloud_registry_model_forbidden")
        registry_by_model.setdefault(endpoint.model_canonical_id, []).append(endpoint)
    if len(registry_by_model.get(DEEPSEEK_MODEL, ())) != 1:
        _fail("deepseek_qualification_required")
    if terra_enabled and len(registry_by_model.get(TERRA_MODEL, ())) != 1:
        _fail("terra_qualification_required")

    active = list(registry_by_model[DEEPSEEK_MODEL])
    if terra_enabled:
        active.extend(registry_by_model[TERRA_MODEL])
    for endpoint in active:
        identity = (
            endpoint.model_canonical_id,
            endpoint.provider_endpoint_tag,
            endpoint.quantization,
            endpoint.parameter_profile_sha256,
            endpoint.qualification_version,
        )
        evidence = evidence_by_identity.get(identity)
        if (
            evidence is None
            or evidence["qualification_sha256"] != endpoint.qualification_sha256
        ):
            _fail("cloud_endpoint_qualification_mismatch")

    return CloudReleaseQualificationV1(
        sha256=actual_sha256,
        qualified_at=qualified_at,
        valid_until=valid_until,
        active_endpoints=tuple(active),
    )


__all__ = [
    "CLOUD_RELEASE_VERSION",
    "CloudQualificationError",
    "CloudReleaseQualificationV1",
    "cloud_release_sha256",
    "dedicated_key_sha256",
    "load_cloud_release_qualification",
    "run_cloud_privacy_qualification",
]
