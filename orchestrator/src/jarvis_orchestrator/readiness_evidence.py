"""Signed, identity-bound operational-readiness evidence.

The evidence file contains no trust root.  Callers must inject a verifier that
was created from an independently pinned Ed25519 public key.  A legacy or
self-signed JSON document is therefore never eligible for a ready decision.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import platform
import re
import stat
import sys
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .operations import OperationalError, ProcessLock
from .private_state import (
    PrivateStateError,
    create_or_replace_private_file,
    ensure_private_directory,
)
from .windows_private_file import (
    WindowsPrivateFileError,
    harden_windows_private_path,
    validate_windows_private_descriptor,
    validate_windows_private_path,
)

READINESS_EVIDENCE_VERSION = "jarvis.readiness-evidence/2"
READINESS_AGGREGATE_VERSION = "operational-ready-gate/4"
MAXIMUM_EVIDENCE_BYTES = 1024 * 1024
MAXIMUM_KEY_FILE_BYTES = 16 * 1024
MAXIMUM_ARTIFACT_BYTES = 128 * 1024 * 1024 * 1024
MAXIMUM_FRESHNESS_SECONDS = 7 * 24 * 60 * 60
MAXIMUM_CLOCK_SKEW_SECONDS = 30
MAXIMUM_QUALIFICATION_WINDOW_SECONDS = 72 * 60 * 60
MAXIMUM_COHORT_RECEIPT_BYTES = 64 * 1024

SYNTHETIC_KIND = "synthetic_lifecycle"
LIVE_KIND = "sequential_local_model"
RESTART_KIND = "restart_generation_playback"
RECOVERY_KIND = "sqlite_recovery"
SOAK_KIND = "twenty_four_hour_soak"
NATIVE_KIND = "native_voice_playback"
AGGREGATE_KIND = "operational_ready"

LEAF_EVIDENCE_KINDS = (
    SYNTHETIC_KIND,
    LIVE_KIND,
    RESTART_KIND,
    RECOVERY_KIND,
    SOAK_KIND,
    NATIVE_KIND,
)
EVIDENCE_KINDS = (*LEAF_EVIDENCE_KINDS, AGGREGATE_KIND)
QUALIFICATION_SEQUENCE_BY_KIND = MappingProxyType(
    {
        SYNTHETIC_KIND: 1,
        LIVE_KIND: 2,
        RESTART_KIND: 3,
        RECOVERY_KIND: 4,
        SOAK_KIND: 5,
        NATIVE_KIND: 6,
        AGGREGATE_KIND: 7,
    }
)
_QUALIFICATION_KIND_BY_SEQUENCE = MappingProxyType(
    {sequence: kind for kind, sequence in QUALIFICATION_SEQUENCE_BY_KIND.items()}
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_MODEL_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
_UTC_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SIGNATURE_DOMAIN = b"JARVIS\x00readiness-evidence-signature/2\x00"
_DIGEST_DOMAIN = b"JARVIS\x00readiness-evidence/2\x00"
_DATABASE_DIAGNOSTICS_DOMAIN = b"JARVIS\x00readiness-database-diagnostics/1\x00"
_CONFIG_PROFILE_DOMAIN = b"JARVIS\x00readiness-config-profile/1\x00"
_BEARER_VALUE_DOMAIN = b"JARVIS\x00readiness-redacted-bearer/1\x00"
_COHORT_IDENTITY_DOMAIN = b"JARVIS\x00readiness-cohort-identity/1\x00"
_COHORT_DESTINATION_DOMAIN = b"JARVIS\x00readiness-cohort-destination/1\x00"
MEMORY_CRYPTO_DISABLED_STATE_SHA256 = hashlib.sha256(
    b"JARVIS\x00memory-crypto-readiness/1\x00disabled"
).hexdigest()
_COHORT_RECEIPT_VERSION = "jarvis.readiness-cohort-receipt/1"
_COHORT_RECEIPT_KEYS = {
    "cohort_id",
    "identity_sha256",
    "signer_key_id",
    "slots",
    "version",
    "window_expires_at_utc",
    "window_started_at_utc",
}
_COHORT_SLOT_KEYS = {
    "destination_sha256",
    "evidence_kind",
    "evidence_sha256",
}
_BEARER_FIELDS = frozenset(
    {
        "control_bearer_token",
        "inference_bearer_token",
        "research_bearer_token",
    }
)

_ENVELOPE_KEYS = {
    "captured_at_utc",
    "config_profile_sha256",
    "database_diagnostics",
    "database_diagnostics_sha256",
    "database_schema_version",
    "evidence_kind",
    "evidence_sha256",
    "expires_at_utc",
    "freshness_seconds",
    "host_class",
    "inference_manifest_sha256",
    "model_digest_sha256",
    "model_identity",
    "memory_crypto_state_sha256",
    "inference_runtime_identity_sha256",
    "parent_evidence_sha256s",
    "payload",
    "payload_sha256",
    "signature",
    "signer_key_id",
    "source_build_sha256",
    "qualification",
    "version",
}

_DATABASE_DIAGNOSTICS_KEYS = {
    "active_task_count",
    "claimed_outbox_count",
    "conversation_count",
    "database_bytes",
    "enabled_cloud_endpoint_count",
    "extension_outbox_count",
    "freelist_count",
    "integrity_issue_count",
    "integrity_ok",
    "invariant_counts",
    "invariant_violation_count",
    "lifecycle_event_count",
    "operational_log_bytes",
    "outbox_count",
    "page_count",
    "pending_approval_count",
    "pending_capability_reservation_count",
    "pending_model_reservation_count",
    "ready",
    "replayable_extension_outbox_count",
    "replayable_outbox_count",
    "schema_version",
    "shm_bytes",
    "task_count",
    "version",
    "wal_bytes",
}

_DATABASE_INVARIANT_KEYS = {
    "cloud_dispatch_overflow",
    "completed_without_done",
    "duplicate_speech_units",
    "duplicate_text_chunks",
    "multiple_active_conversations",
    "multiple_active_tasks",
    "mutually_exclusive_terminals",
    "output_after_terminal",
}

_PAYLOAD_KEYS = {
    SYNTHETIC_KIND: {
        "cancellation_stage_counts",
        "completed_turn_count",
        "elapsed_ms",
        "invariant_check_count",
        "invariant_violation_count",
        "maximum_database_bytes",
        "ready",
        "turn_count",
        "version",
    },
    LIVE_KIND: {
        "database_bounded",
        "duplicate_publication_count",
        "elapsed_ms",
        "maximum_turn_ms",
        "physical_speaker_playback_observed",
        "playback_completion_mode",
        "ready",
        "response_count",
        "speech_unit_count",
        "turn_count",
        "voice_auth_protocol",
        "version",
    },
    RESTART_KIND: {
        "duplicate_publication_count",
        "generation_interrupted",
        "playback_replay_exact",
        "ready",
        "voice_auth_protocol",
        "version",
    },
    RECOVERY_KIND: {
        "backup_integrity_ok",
        "corruption_detected",
        "elapsed_ms",
        "production_database_mutated",
        "ready",
        "restore_integrity_ok",
        "schema_version",
        "version",
    },
    SOAK_KIND: {
        "batch_count",
        "database_bounded",
        "duration_seconds",
        "invariant_violation_count",
        "maximum_database_bytes",
        "maximum_operational_log_bytes",
        "operational_log_bounded",
        "physical_speaker_playback_observed",
        "playback_completion_mode",
        "ready",
        "turn_count",
        "voice_auth_protocol",
        "version",
    },
    NATIVE_KIND: {
        "checks",
        "code",
        "content_captured",
        "network_used",
        "platform",
        "product_version",
        "protocol_version",
        "ready",
        "runtime_mode",
        "runtime_profile_sha256",
        "schema_version",
        "voice_auth_protocol",
    },
    AGGREGATE_KIND: {
        "checks",
        "content_captured",
        "ready",
        "version",
    },
}

_PAYLOAD_VERSIONS = {
    SYNTHETIC_KIND: "synthetic-lifecycle-ready-gate/1",
    LIVE_KIND: "live-sequential-ready-gate/2",
    RESTART_KIND: "live-restart-ready-gate/2",
    RECOVERY_KIND: "sqlite-recovery-ready-gate/1",
    SOAK_KIND: "live-soak-ready-gate/2",
    AGGREGATE_KIND: READINESS_AGGREGATE_VERSION,
}

_CANCELLATION_STAGES = {
    "ADMITTED",
    "ROUTING",
    "LOADING_CONTEXT",
    "EXECUTING",
    "RESPONDING",
    "AWAITING_APPROVAL",
    "DECISION_WAITING_RECONNECT",
}

_NATIVE_CHECK_SHAPES = {
    "native_provenance": set(),
    "runtime_cases": set(),
    "turn_integrity": {
        "duplicate_transcripts",
        "duplicate_tts_playbacks",
        "minimum_real_voice_turns",
        "real_voice_turns",
        "stuck_ptt",
    },
    "acoustic_barge_in": {"maximum_ms", "samples", "threshold_ms"},
    "automatic_recovery": set(),
    "soak": {
        "crashes",
        "duration_seconds",
        "memory_stable",
        "minimum_duration_seconds",
        "rss_samples",
    },
    "runtime_profile": set(),
    "privacy": set(),
}

_NATIVE_CHECK_CODES = {
    "native_provenance": (
        "qualification.native_hardware_verified",
        "qualification.native_hardware_required",
    ),
    "runtime_cases": (
        "qualification.native_cases_passed",
        "qualification.native_cases_incomplete",
    ),
    "turn_integrity": (
        "qualification.native_turn_integrity_passed",
        "qualification.native_turn_integrity_failed",
    ),
    "acoustic_barge_in": (
        "qualification.native_acoustic_barge_in_passed",
        "qualification.native_acoustic_barge_in_required",
    ),
    "automatic_recovery": (
        "qualification.native_recovery_passed",
        "qualification.native_recovery_required",
    ),
    "soak": (
        "qualification.native_soak_passed",
        "qualification.native_soak_required",
    ),
    "runtime_profile": (
        "qualification.native_profile_stable",
        "qualification.native_profile_required",
        "qualification.native_profile_mismatch",
    ),
    "privacy": (
        "qualification.native_privacy_passed",
        "qualification.native_privacy_failed",
    ),
}


def _error(code: str) -> OperationalError:
    return OperationalError(code)


def _digest(value: Any, code: str = "readiness_evidence_invalid") -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _error(code)
    return value


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_key_id(value: Any) -> bool:
    return isinstance(value, str) and _KEY_ID.fullmatch(value) is not None


def _as_utc(value: datetime, *, error_code: str = "readiness_evidence_invalid") -> datetime:
    try:
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValueError
        return value.astimezone(UTC)
    except (OverflowError, TypeError, ValueError):
        raise _error(error_code) from None


def _utc_text(value: datetime, *, offset_seconds: int = 0) -> str:
    try:
        selected = _as_utc(value).replace(microsecond=0) + timedelta(seconds=offset_seconds)
    except OverflowError:
        raise _error("readiness_evidence_invalid") from None
    return selected.isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or _UTC_SECONDS.fullmatch(value) is None:
        raise _error("readiness_evidence_invalid")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        raise _error("readiness_evidence_invalid") from None
    if _utc_text(parsed) != value:
        raise _error("readiness_evidence_invalid")
    return parsed


def _validate_json(
    value: Any, *, depth: int = 0, budget: list[int] | None = None
) -> None:
    remaining = [100_000] if budget is None else budget
    remaining[0] -= 1
    if remaining[0] < 0 or depth > 64:
        raise _error("readiness_evidence_invalid")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error("readiness_evidence_invalid")
        return
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise _error("readiness_evidence_invalid")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise _error("readiness_evidence_invalid") from None
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _error("readiness_evidence_invalid")
            _validate_json(key, depth=depth + 1, budget=remaining)
            _validate_json(item, depth=depth + 1, budget=remaining)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _validate_json(item, depth=depth + 1, budget=remaining)
        return
    raise _error("readiness_evidence_invalid")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    _validate_json(value)
    try:
        return json.dumps(
            _plain_json(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (RecursionError, TypeError, UnicodeEncodeError, ValueError):
        raise _error("readiness_evidence_invalid") from None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _error("readiness_evidence_invalid")
        result[key] = value
    return result


def _reject_constant(_: str) -> Any:
    raise _error("readiness_evidence_invalid")


def _is_windows_reparse(value: os.stat_result) -> bool:
    return bool(
        os.name == "nt"
        and getattr(value, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _snapshot(value: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(value, name, 0))
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_file_attributes",
        )
    )


def _validate_private_key_descriptor(metadata: os.stat_result, descriptor: int) -> None:
    if os.name == "posix":
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) not in {
            0o400,
            0o600,
        }:
            raise _error("readiness_signing_key_not_private")
        return
    if os.name == "nt":  # pragma: no cover - exercised by native Windows CI
        try:
            validate_windows_private_descriptor(descriptor)
        except WindowsPrivateFileError:
            raise _error("readiness_signing_key_not_private") from None
        return
    raise _error("readiness_signing_key_not_private")


def _validate_private_state_descriptor(
    metadata: os.stat_result,
    descriptor: int,
) -> None:
    if os.name == "posix":
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise _error("readiness_cohort_receipt_invalid")
        return
    if os.name == "nt":  # pragma: no cover - exercised by native Windows CI
        try:
            validate_windows_private_descriptor(descriptor)
        except WindowsPrivateFileError:
            raise _error("readiness_cohort_receipt_invalid") from None
        return
    raise _error("readiness_cohort_receipt_invalid")


def _read_regular_file(
    path: str | Path,
    *,
    maximum_bytes: int,
    require_private_key: bool = False,
    require_private_state: bool = False,
) -> bytes:
    if require_private_key and require_private_state:
        raise _error("readiness_evidence_file_invalid")
    selected = Path(path).absolute()
    descriptor = -1
    try:
        before_path = os.lstat(selected)
        if (
            stat.S_ISLNK(before_path.st_mode)
            or _is_windows_reparse(before_path)
            or not stat.S_ISREG(before_path.st_mode)
        ):
            raise _error("readiness_evidence_file_invalid")
        if before_path.st_size > maximum_bytes:
            raise _error("readiness_evidence_file_invalid")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(selected, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _snapshot(before) != _snapshot(
            before_path
        ):
            raise _error("readiness_evidence_file_invalid")
        if require_private_key:
            _validate_private_key_descriptor(before, descriptor)
        if require_private_state:
            _validate_private_state_descriptor(before, descriptor)
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        after_path = os.lstat(selected)
        content = b"".join(chunks)
        if (
            len(content) > maximum_bytes
            or len(content) != before.st_size
            or _snapshot(after) != _snapshot(before)
            or _snapshot(after_path) != _snapshot(before)
            or _is_windows_reparse(after)
            or _is_windows_reparse(after_path)
        ):
            raise _error("readiness_evidence_file_invalid")
        if require_private_key:
            _validate_private_key_descriptor(after, descriptor)
        if require_private_state:
            _validate_private_state_descriptor(after, descriptor)
        return content
    except OperationalError:
        raise
    except OSError:
        raise _error("readiness_evidence_file_invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parse_canonical_document(content: bytes) -> dict[str, Any]:
    raw = _parse_strict_json_object(content)
    if raw.get("version") != READINESS_EVIDENCE_VERSION:
        raise _error("readiness_evidence_legacy_ineligible")
    if canonical_json_bytes(raw) != content:
        raise _error("readiness_evidence_invalid")
    return raw


def _parse_strict_json_object(content: bytes) -> dict[str, Any]:
    if (
        not content
        or len(content) > MAXIMUM_EVIDENCE_BYTES
        or content.startswith(b"\xef\xbb\xbf")
    ):
        raise _error("readiness_evidence_invalid")
    try:
        raw = json.loads(
            content.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except OperationalError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError):
        raise _error("readiness_evidence_invalid") from None
    if not isinstance(raw, dict):
        raise _error("readiness_evidence_invalid")
    canonical_json_bytes(raw)
    return raw


@dataclass(frozen=True, slots=True)
class HostClassV1:
    os_family: str
    architecture: str
    runtime_mode: str

    def __post_init__(self) -> None:
        if self.os_family not in ("linux", "macos", "windows"):
            raise _error("readiness_identity_invalid")
        if self.architecture not in ("arm64", "x86_64"):
            raise _error("readiness_identity_invalid")
        if self.runtime_mode not in ("source", "signed_package"):
            raise _error("readiness_identity_invalid")

    def canonical(self) -> dict[str, str]:
        return {
            "architecture": self.architecture,
            "os_family": self.os_family,
            "runtime_mode": self.runtime_mode,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> HostClassV1:
        if not isinstance(value, Mapping) or set(value) != {
            "architecture",
            "os_family",
            "runtime_mode",
        }:
            raise _error("readiness_evidence_invalid")
        return cls(
            os_family=value["os_family"],
            architecture=value["architecture"],
            runtime_mode=value["runtime_mode"],
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedInferenceIdentityV2:
    model_alias: str
    manifest_digest: str
    runtime_identity_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_alias, str)
            or _MODEL_IDENTITY.fullmatch(self.model_alias) is None
        ):
            raise _error("readiness_runtime_identity_invalid")
        _digest(self.manifest_digest, "readiness_runtime_identity_invalid")
        _digest(
            self.runtime_identity_sha256,
            "readiness_runtime_identity_invalid",
        )


@dataclass(frozen=True, slots=True)
class QualificationBindingV2:
    cohort_id: str
    sequence: int
    window_started_at_utc: str
    window_expires_at_utc: str

    def __post_init__(self) -> None:
        try:
            cohort = uuid.UUID(self.cohort_id)
        except (AttributeError, TypeError, ValueError):
            raise _error("readiness_qualification_invalid") from None
        started = _parse_utc(self.window_started_at_utc)
        expires = _parse_utc(self.window_expires_at_utc)
        if (
            cohort.version != 4
            or str(cohort) != self.cohort_id
            or isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence not in QUALIFICATION_SEQUENCE_BY_KIND.values()
            or expires - started
            != timedelta(seconds=MAXIMUM_QUALIFICATION_WINDOW_SECONDS)
        ):
            raise _error("readiness_qualification_invalid")

    def canonical(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "sequence": self.sequence,
            "window_expires_at_utc": self.window_expires_at_utc,
            "window_started_at_utc": self.window_started_at_utc,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> QualificationBindingV2:
        if not isinstance(value, Mapping) or set(value) != {
            "cohort_id",
            "sequence",
            "window_expires_at_utc",
            "window_started_at_utc",
        }:
            raise _error("readiness_qualification_invalid")
        return cls(
            cohort_id=value["cohort_id"],
            sequence=value["sequence"],
            window_started_at_utc=value["window_started_at_utc"],
            window_expires_at_utc=value["window_expires_at_utc"],
        )

    def validate_for(self, evidence_kind: str, captured_at_utc: str) -> None:
        captured = _parse_utc(captured_at_utc)
        started = _parse_utc(self.window_started_at_utc)
        expires = _parse_utc(self.window_expires_at_utc)
        if (
            evidence_kind not in EVIDENCE_KINDS
            or QUALIFICATION_SEQUENCE_BY_KIND.get(evidence_kind) != self.sequence
            or captured < started
            or captured >= expires
        ):
            raise _error("readiness_qualification_invalid")


def qualification_binding_for_kind(
    evidence_kind: str,
    *,
    cohort_id: str,
    window_started_at: datetime | str,
) -> QualificationBindingV2:
    if evidence_kind not in EVIDENCE_KINDS:
        raise _error("readiness_qualification_invalid")
    if isinstance(window_started_at, datetime):
        started = _as_utc(window_started_at, error_code="readiness_qualification_invalid")
    elif isinstance(window_started_at, str):
        started = _parse_utc(window_started_at)
    else:
        raise _error("readiness_qualification_invalid")
    return QualificationBindingV2(
        cohort_id=cohort_id,
        sequence=QUALIFICATION_SEQUENCE_BY_KIND[evidence_kind],
        window_started_at_utc=_utc_text(started),
        window_expires_at_utc=_utc_text(
            started, offset_seconds=MAXIMUM_QUALIFICATION_WINDOW_SECONDS
        ),
    )


@dataclass(frozen=True, slots=True)
class ReadinessIdentityV2:
    source_build_sha256: str
    config_profile_sha256: str
    model_identity: str
    model_digest_sha256: str
    inference_manifest_sha256: str
    inference_runtime_identity_sha256: str
    database_schema_version: int
    host_class: HostClassV1
    memory_crypto_state_sha256: str = MEMORY_CRYPTO_DISABLED_STATE_SHA256

    def __post_init__(self) -> None:
        _digest(self.source_build_sha256, "readiness_identity_invalid")
        _digest(self.config_profile_sha256, "readiness_identity_invalid")
        _digest(self.model_digest_sha256, "readiness_identity_invalid")
        _digest(self.inference_manifest_sha256, "readiness_identity_invalid")
        _digest(
            self.inference_runtime_identity_sha256,
            "readiness_identity_invalid",
        )
        _digest(self.memory_crypto_state_sha256, "readiness_identity_invalid")
        if (
            not isinstance(self.model_identity, str)
            or _MODEL_IDENTITY.fullmatch(self.model_identity) is None
            or isinstance(self.database_schema_version, bool)
            or not isinstance(self.database_schema_version, int)
            or self.database_schema_version < 1
            or not isinstance(self.host_class, HostClassV1)
        ):
            raise _error("readiness_identity_invalid")

    def canonical(self) -> dict[str, Any]:
        return {
            "config_profile_sha256": self.config_profile_sha256,
            "database_schema_version": self.database_schema_version,
            "host_class": self.host_class.canonical(),
            "inference_manifest_sha256": self.inference_manifest_sha256,
            "inference_runtime_identity_sha256": (
                self.inference_runtime_identity_sha256
            ),
            "model_digest_sha256": self.model_digest_sha256,
            "model_identity": self.model_identity,
            "memory_crypto_state_sha256": self.memory_crypto_state_sha256,
            "source_build_sha256": self.source_build_sha256,
        }


def local_host_class() -> HostClassV1:
    """Return a coarse non-identifying host class for the current runtime."""

    os_family = {
        "darwin": "macos",
        "linux": "linux",
        "windows": "windows",
    }.get(platform.system().lower())
    architecture = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "x86_64",
        "x86_64": "x86_64",
    }.get(platform.machine().lower())
    if os_family is None or architecture is None:
        raise _error("readiness_identity_invalid")
    return HostClassV1(
        os_family=os_family,
        architecture=architecture,
        runtime_mode="signed_package" if getattr(sys, "frozen", False) else "source",
    )


class ReadinessEvidenceSigner(Protocol):
    @property
    def key_id(self) -> str: ...

    def sign(self, message: bytes) -> bytes: ...


class ReadinessEvidenceVerifier(Protocol):
    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool: ...


class Ed25519EvidenceSigner:
    __slots__ = ("_key", "_key_id")

    def __init__(self, key_id: str, key: Ed25519PrivateKey) -> None:
        if not _is_key_id(key_id) or not isinstance(key, Ed25519PrivateKey):
            raise _error("readiness_signer_invalid")
        self._key_id = key_id
        self._key = key

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, message: bytes) -> bytes:
        return self._key.sign(message)


class Ed25519EvidenceVerifier:
    __slots__ = ("_keys",)

    def __init__(self, keys: Mapping[str, Ed25519PublicKey]) -> None:
        if not keys or any(
            not _is_key_id(key_id) or not isinstance(key, Ed25519PublicKey)
            for key_id, key in keys.items()
        ):
            raise _error("readiness_verifier_invalid")
        self._keys = MappingProxyType(dict(keys))

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        key = self._keys.get(key_id)
        if key is None:
            return False
        try:
            key.verify(signature, message)
        except InvalidSignature:
            return False
        return True


def _signature_text(signature: bytes) -> str:
    return base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def _signature_bytes(value: Any) -> bytes:
    if not isinstance(value, str) or not value or "=" in value or len(value) > 128:
        raise _error("readiness_evidence_invalid")
    try:
        encoded = value.encode("ascii", errors="strict")
        decoded = base64.b64decode(
            encoded + b"=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, UnicodeEncodeError, ValueError):
        raise _error("readiness_evidence_invalid") from None
    if len(decoded) != 64 or _signature_text(decoded) != value:
        raise _error("readiness_evidence_invalid")
    return decoded


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_nonnegative_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        selected = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(selected) and selected >= 0


def _require_nonnegative_integers(payload: Mapping[str, Any], names: set[str]) -> None:
    if any(not _nonnegative_integer(payload.get(name)) for name in names):
        raise _error("readiness_payload_invalid")


def _closed_database_diagnostics(
    value: Any,
    *,
    expected_schema_version: int,
) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict()
        except Exception:  # noqa: BLE001 - injected diagnostics fail closed
            raise _error("readiness_database_diagnostics_invalid") from None
    if not isinstance(value, Mapping):
        raise _error("readiness_database_diagnostics_invalid")
    try:
        normalized = json.loads(canonical_json_bytes(dict(value)))
    except (RecursionError, TypeError, ValueError):
        raise _error("readiness_database_diagnostics_invalid") from None
    if set(normalized) != _DATABASE_DIAGNOSTICS_KEYS:
        raise _error("readiness_database_diagnostics_invalid")
    invariants = normalized.get("invariant_counts")
    if (
        normalized.get("version") != "operational-diagnostics/1"
        or not isinstance(invariants, dict)
        or set(invariants) != _DATABASE_INVARIANT_KEYS
        or any(not _nonnegative_integer(item) for item in invariants.values())
    ):
        raise _error("readiness_database_diagnostics_invalid")
    integer_fields = _DATABASE_DIAGNOSTICS_KEYS - {
        "integrity_ok",
        "invariant_counts",
        "ready",
        "version",
    }
    if any(not _nonnegative_integer(normalized.get(name)) for name in integer_fields):
        raise _error("readiness_database_diagnostics_invalid")
    if (
        normalized["schema_version"] != expected_schema_version
        or normalized.get("ready") is not True
        or normalized.get("integrity_ok") is not True
        or normalized.get("integrity_issue_count") != 0
        or normalized.get("invariant_violation_count") != 0
        or sum(invariants.values()) != normalized["invariant_violation_count"]
    ):
        raise _error("readiness_database_unhealthy")
    return normalized


def _database_diagnostics_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _DATABASE_DIAGNOSTICS_DOMAIN + canonical_json_bytes(value)
    ).hexdigest()


def validated_readiness_database_diagnostics(
    value: Any,
    *,
    expected_schema_version: int,
) -> dict[str, Any]:
    """Return the closed healthy diagnostics projection used by evidence."""

    return _closed_database_diagnostics(
        value,
        expected_schema_version=expected_schema_version,
    )


def _validate_native_payload(payload: Mapping[str, Any]) -> None:
    try:
        from jarvis_voice import PROTOCOL_VERSION, __version__
        from jarvis_voice.native_qualification import SCHEMA_VERSION
        from jarvis_voice.voice_auth import AUTH_PROTOCOL
    except ImportError:
        raise _error("readiness_payload_invalid") from None
    checks = payload.get("checks")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("product_version") != __version__
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("voice_auth_protocol") != AUTH_PROTOCOL
        or payload.get("platform") not in ("macos", "windows")
        or payload.get("runtime_mode") not in ("source", "signed_package")
        or not _is_digest(payload.get("runtime_profile_sha256"))
        or not isinstance(payload.get("ready"), bool)
        or payload.get("code")
        not in ("qualification.native_ready", "qualification.native_blocked")
        or payload.get("content_captured") is not False
        or payload.get("network_used") is not False
        or not isinstance(checks, Mapping)
        or set(checks) != set(_NATIVE_CHECK_SHAPES)
    ):
        raise _error("readiness_payload_invalid")
    for name, details in _NATIVE_CHECK_SHAPES.items():
        check = checks[name]
        if (
            not isinstance(check, Mapping)
            or set(check) != {"code", "status", *details}
            or check.get("status") not in ("ready", "blocked")
        ):
            raise _error("readiness_payload_invalid")
        codes = _NATIVE_CHECK_CODES[name]
        expected_codes = codes[:1] if check["status"] == "ready" else codes[1:]
        if check.get("code") not in expected_codes:
            raise _error("readiness_payload_invalid")
    integer_details = {
        "duplicate_transcripts",
        "duplicate_tts_playbacks",
        "minimum_real_voice_turns",
        "real_voice_turns",
        "stuck_ptt",
    }
    if any(
        not _nonnegative_integer(checks["turn_integrity"].get(name))
        for name in integer_details
    ):
        raise _error("readiness_payload_invalid")
    acoustic = checks["acoustic_barge_in"]
    if (
        not _nonnegative_integer(acoustic.get("samples"))
        or acoustic.get("maximum_ms") is not None
        and not _finite_nonnegative_number(acoustic.get("maximum_ms"))
        or not _finite_nonnegative_number(acoustic.get("threshold_ms"))
    ):
        raise _error("readiness_payload_invalid")
    soak = checks["soak"]
    if any(
        not _nonnegative_integer(soak.get(name))
        for name in (
            "crashes",
            "duration_seconds",
            "minimum_duration_seconds",
            "rss_samples",
        )
    ) or not isinstance(soak.get("memory_stable"), bool):
        raise _error("readiness_payload_invalid")
    all_ready = all(check["status"] == "ready" for check in checks.values())
    if payload["ready"] is not all_ready or payload["code"] != (
        "qualification.native_ready" if all_ready else "qualification.native_blocked"
    ):
        raise _error("readiness_payload_invalid")


def _validate_payload_identity(
    evidence_kind: str,
    payload: Mapping[str, Any],
    identity: ReadinessIdentityV2,
) -> None:
    if evidence_kind != NATIVE_KIND:
        return
    if (
        payload.get("platform") != identity.host_class.os_family
        or payload.get("runtime_mode") != identity.host_class.runtime_mode
    ):
        raise _error("readiness_native_identity_mismatch")


def _validate_closed_leaf_payload(
    evidence_kind: str, payload: Mapping[str, Any]
) -> None:
    if evidence_kind == SYNTHETIC_KIND:
        counts = payload.get("cancellation_stage_counts")
        if (
            not isinstance(counts, Mapping)
            or set(counts) != _CANCELLATION_STAGES
            or any(not _nonnegative_integer(value) for value in counts.values())
        ):
            raise _error("readiness_payload_invalid")
        _require_nonnegative_integers(
            payload,
            {
                "completed_turn_count",
                "elapsed_ms",
                "invariant_check_count",
                "invariant_violation_count",
                "maximum_database_bytes",
                "turn_count",
            },
        )
        return
    if evidence_kind == LIVE_KIND:
        _require_nonnegative_integers(
            payload,
            {
                "duplicate_publication_count",
                "elapsed_ms",
                "maximum_turn_ms",
                "response_count",
                "speech_unit_count",
                "turn_count",
            },
        )
        if (
            not isinstance(payload.get("database_bounded"), bool)
            or payload.get("physical_speaker_playback_observed") is not False
            or payload.get("playback_completion_mode") != "protocol_probe"
            or payload.get("voice_auth_protocol") != "jarvis.voice-auth/1"
        ):
            raise _error("readiness_payload_invalid")
        return
    if evidence_kind == RESTART_KIND:
        _require_nonnegative_integers(payload, {"duplicate_publication_count"})
        if (
            not isinstance(payload.get("generation_interrupted"), bool)
            or not isinstance(payload.get("playback_replay_exact"), bool)
            or payload.get("voice_auth_protocol") != "jarvis.voice-auth/1"
        ):
            raise _error("readiness_payload_invalid")
        return
    if evidence_kind == RECOVERY_KIND:
        _require_nonnegative_integers(payload, {"elapsed_ms", "schema_version"})
        if (
            payload.get("schema_version", 0) < 1
            or not isinstance(payload.get("backup_integrity_ok"), bool)
            or not isinstance(payload.get("corruption_detected"), bool)
            or not isinstance(payload.get("production_database_mutated"), bool)
            or not isinstance(payload.get("restore_integrity_ok"), bool)
        ):
            raise _error("readiness_payload_invalid")
        return
    if evidence_kind == SOAK_KIND:
        _require_nonnegative_integers(
            payload,
            {
                "batch_count",
                "duration_seconds",
                "invariant_violation_count",
                "maximum_database_bytes",
                "maximum_operational_log_bytes",
                "turn_count",
            },
        )
        if (
            not isinstance(payload.get("database_bounded"), bool)
            or not isinstance(payload.get("operational_log_bounded"), bool)
            or payload.get("physical_speaker_playback_observed") is not False
            or payload.get("playback_completion_mode") != "protocol_probe"
            or payload.get("voice_auth_protocol") != "jarvis.voice-auth/1"
        ):
            raise _error("readiness_payload_invalid")
        return
    if evidence_kind == NATIVE_KIND:
        _validate_native_payload(payload)


def _closed_payload(evidence_kind: str, payload: Any) -> dict[str, Any]:
    if evidence_kind not in EVIDENCE_KINDS or not isinstance(payload, Mapping):
        raise _error("readiness_payload_invalid")
    try:
        normalized = json.loads(canonical_json_bytes(dict(payload)))
    except (RecursionError, ValueError):
        raise _error("readiness_payload_invalid") from None
    expected = _PAYLOAD_KEYS[evidence_kind]
    if set(normalized) != expected:
        raise _error("readiness_payload_invalid")
    if (
        evidence_kind != NATIVE_KIND
        and normalized.get("version") != _PAYLOAD_VERSIONS[evidence_kind]
    ):
        raise _error("readiness_payload_invalid")
    if not isinstance(normalized.get("ready"), bool):
        raise _error("readiness_payload_invalid")
    _validate_closed_leaf_payload(evidence_kind, normalized)
    if evidence_kind == AGGREGATE_KIND:
        if normalized.get("content_captured") is not False:
            raise _error("readiness_payload_invalid")
        checks = normalized.get("checks")
        if not isinstance(checks, dict) or set(checks) != set(LEAF_EVIDENCE_KINDS):
            raise _error("readiness_payload_invalid")
        if any(
            not isinstance(value, dict)
            or set(value) != {"ready"}
            or not isinstance(value["ready"], bool)
            for value in checks.values()
        ):
            raise _error("readiness_payload_invalid")
        if normalized["ready"] is not all(value["ready"] for value in checks.values()):
            raise _error("readiness_payload_invalid")
    return normalized


def _closed_parents(evidence_kind: str, parent_evidence_sha256s: Any) -> dict[str, str]:
    if not isinstance(parent_evidence_sha256s, Mapping):
        raise _error("readiness_parent_evidence_invalid")
    parents = dict(parent_evidence_sha256s)
    expected_kinds = (
        set() if evidence_kind in LEAF_EVIDENCE_KINDS else set(LEAF_EVIDENCE_KINDS)
    )
    if (
        evidence_kind not in EVIDENCE_KINDS
        or set(parents) != expected_kinds
        or any(not _is_digest(value) for value in parents.values())
        or len(set(parents.values())) != len(parents)
    ):
        raise _error("readiness_parent_evidence_invalid")
    return dict(sorted(parents.items()))


def _core_document(
    *,
    evidence_kind: str,
    captured_at_utc: str,
    expires_at_utc: str,
    freshness_seconds: int,
    identity: ReadinessIdentityV2,
    database_diagnostics: Mapping[str, Any],
    database_diagnostics_sha256: str,
    qualification: QualificationBindingV2,
    payload: Mapping[str, Any],
    payload_sha256: str,
    parent_evidence_sha256s: Mapping[str, str],
    signer_key_id: str,
) -> dict[str, Any]:
    return {
        "captured_at_utc": captured_at_utc,
        **identity.canonical(),
        "database_diagnostics": dict(database_diagnostics),
        "database_diagnostics_sha256": database_diagnostics_sha256,
        "evidence_kind": evidence_kind,
        "expires_at_utc": expires_at_utc,
        "freshness_seconds": freshness_seconds,
        "parent_evidence_sha256s": dict(parent_evidence_sha256s),
        "payload": dict(payload),
        "payload_sha256": payload_sha256,
        "qualification": qualification.canonical(),
        "signer_key_id": signer_key_id,
        "version": READINESS_EVIDENCE_VERSION,
    }


@dataclass(frozen=True, slots=True)
class ReadinessEvidenceV2:
    evidence_kind: str
    captured_at_utc: str
    expires_at_utc: str
    freshness_seconds: int
    identity: ReadinessIdentityV2
    database_diagnostics: Mapping[str, Any]
    database_diagnostics_sha256: str
    qualification: QualificationBindingV2
    payload: Mapping[str, Any]
    payload_sha256: str
    parent_evidence_sha256s: Mapping[str, str]
    signer_key_id: str
    evidence_sha256: str
    signature: str

    def __post_init__(self) -> None:
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise _error("readiness_evidence_invalid")
        if not isinstance(self.identity, ReadinessIdentityV2) or not isinstance(
            self.qualification, QualificationBindingV2
        ):
            raise _error("readiness_evidence_invalid")
        captured = _parse_utc(self.captured_at_utc)
        expires = _parse_utc(self.expires_at_utc)
        if (
            isinstance(self.freshness_seconds, bool)
            or not isinstance(self.freshness_seconds, int)
            or not 60 <= self.freshness_seconds <= MAXIMUM_FRESHNESS_SECONDS
            or expires - captured != timedelta(seconds=self.freshness_seconds)
            or not _is_key_id(self.signer_key_id)
        ):
            raise _error("readiness_evidence_invalid")
        self.qualification.validate_for(self.evidence_kind, self.captured_at_utc)
        payload = _closed_payload(self.evidence_kind, self.payload)
        _validate_payload_identity(self.evidence_kind, payload, self.identity)
        diagnostics = _closed_database_diagnostics(
            self.database_diagnostics,
            expected_schema_version=self.identity.database_schema_version,
        )
        if _database_diagnostics_sha256(diagnostics) != _digest(
            self.database_diagnostics_sha256
        ):
            raise _error("readiness_database_diagnostics_invalid")
        parents = _closed_parents(self.evidence_kind, self.parent_evidence_sha256s)
        if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != _digest(
            self.payload_sha256
        ):
            raise _error("readiness_evidence_invalid")
        core = _core_document(
            evidence_kind=self.evidence_kind,
            captured_at_utc=self.captured_at_utc,
            expires_at_utc=self.expires_at_utc,
            freshness_seconds=self.freshness_seconds,
            identity=self.identity,
            database_diagnostics=diagnostics,
            database_diagnostics_sha256=self.database_diagnostics_sha256,
            qualification=self.qualification,
            payload=payload,
            payload_sha256=self.payload_sha256,
            parent_evidence_sha256s=parents,
            signer_key_id=self.signer_key_id,
        )
        expected_evidence_sha256 = hashlib.sha256(
            _DIGEST_DOMAIN + canonical_json_bytes(core)
        ).hexdigest()
        if expected_evidence_sha256 != _digest(self.evidence_sha256):
            raise _error("readiness_evidence_invalid")
        _signature_bytes(self.signature)
        object.__setattr__(
            self,
            "database_diagnostics",
            MappingProxyType(diagnostics),
        )
        object.__setattr__(self, "payload", MappingProxyType(payload))
        object.__setattr__(self, "parent_evidence_sha256s", MappingProxyType(parents))

    def canonical(self) -> dict[str, Any]:
        return {
            **_core_document(
                evidence_kind=self.evidence_kind,
                captured_at_utc=self.captured_at_utc,
                expires_at_utc=self.expires_at_utc,
                freshness_seconds=self.freshness_seconds,
                identity=self.identity,
                database_diagnostics=self.database_diagnostics,
                database_diagnostics_sha256=self.database_diagnostics_sha256,
                qualification=self.qualification,
                payload=self.payload,
                payload_sha256=self.payload_sha256,
                parent_evidence_sha256s=self.parent_evidence_sha256s,
                signer_key_id=self.signer_key_id,
            ),
            "evidence_sha256": self.evidence_sha256,
            "signature": self.signature,
        }


def build_readiness_evidence(
    *,
    evidence_kind: str,
    payload: Mapping[str, Any],
    identity: ReadinessIdentityV2,
    database_diagnostics: Any,
    qualification: QualificationBindingV2,
    signer: ReadinessEvidenceSigner,
    captured_at: datetime | None = None,
    freshness_seconds: int = 24 * 60 * 60,
    parent_evidence_sha256s: Mapping[str, str] | None = None,
) -> ReadinessEvidenceV2:
    try:
        signer_key_id = signer.key_id
    except (AttributeError, TypeError):
        raise _error("readiness_signer_invalid") from None
    if (
        isinstance(freshness_seconds, bool)
        or not isinstance(freshness_seconds, int)
        or not 60 <= freshness_seconds <= MAXIMUM_FRESHNESS_SECONDS
        or not _is_key_id(signer_key_id)
    ):
        raise _error("readiness_evidence_invalid")
    normalized_payload = _closed_payload(evidence_kind, payload)
    if not isinstance(identity, ReadinessIdentityV2):
        raise _error("readiness_identity_invalid")
    _validate_payload_identity(evidence_kind, normalized_payload, identity)
    diagnostics = _closed_database_diagnostics(
        database_diagnostics,
        expected_schema_version=identity.database_schema_version,
    )
    diagnostics_sha256 = _database_diagnostics_sha256(diagnostics)
    parents = _closed_parents(
        evidence_kind,
        {} if parent_evidence_sha256s is None else parent_evidence_sha256s,
    )
    selected = _as_utc(datetime.now(UTC) if captured_at is None else captured_at)
    captured_text = _utc_text(selected)
    if not isinstance(qualification, QualificationBindingV2):
        raise _error("readiness_qualification_invalid")
    qualification.validate_for(evidence_kind, captured_text)
    expires_text = _utc_text(selected, offset_seconds=freshness_seconds)
    payload_sha256 = hashlib.sha256(
        canonical_json_bytes(normalized_payload)
    ).hexdigest()
    core = _core_document(
        evidence_kind=evidence_kind,
        captured_at_utc=captured_text,
        expires_at_utc=expires_text,
        freshness_seconds=freshness_seconds,
        identity=identity,
        database_diagnostics=diagnostics,
        database_diagnostics_sha256=diagnostics_sha256,
        qualification=qualification,
        payload=normalized_payload,
        payload_sha256=payload_sha256,
        parent_evidence_sha256s=parents,
        signer_key_id=signer_key_id,
    )
    evidence_sha256 = hashlib.sha256(
        _DIGEST_DOMAIN + canonical_json_bytes(core)
    ).hexdigest()
    try:
        signature = signer.sign(_SIGNATURE_DOMAIN + bytes.fromhex(evidence_sha256))
    except Exception:  # noqa: BLE001 - injected signer failures are fail-closed
        raise _error("readiness_signer_invalid") from None
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise _error("readiness_signer_invalid")
    return ReadinessEvidenceV2(
        evidence_kind=evidence_kind,
        captured_at_utc=captured_text,
        expires_at_utc=expires_text,
        freshness_seconds=freshness_seconds,
        identity=identity,
        database_diagnostics=MappingProxyType(diagnostics),
        database_diagnostics_sha256=diagnostics_sha256,
        qualification=qualification,
        payload=MappingProxyType(normalized_payload),
        payload_sha256=payload_sha256,
        parent_evidence_sha256s=parents,
        signer_key_id=signer_key_id,
        evidence_sha256=evidence_sha256,
        signature=_signature_text(signature),
    )


def _envelope_from_mapping(value: Mapping[str, Any]) -> ReadinessEvidenceV2:
    if set(value) != _ENVELOPE_KEYS:
        raise _error("readiness_evidence_invalid")
    identity = ReadinessIdentityV2(
        source_build_sha256=value["source_build_sha256"],
        config_profile_sha256=value["config_profile_sha256"],
        model_identity=value["model_identity"],
        model_digest_sha256=value["model_digest_sha256"],
        inference_manifest_sha256=value["inference_manifest_sha256"],
        inference_runtime_identity_sha256=value["inference_runtime_identity_sha256"],
        database_schema_version=value["database_schema_version"],
        host_class=HostClassV1.from_mapping(value["host_class"]),
        memory_crypto_state_sha256=value["memory_crypto_state_sha256"],
    )
    qualification = QualificationBindingV2.from_mapping(value["qualification"])
    parents = value["parent_evidence_sha256s"]
    if (
        value["evidence_kind"] not in EVIDENCE_KINDS
        or not isinstance(value["freshness_seconds"], int)
        or isinstance(value["freshness_seconds"], bool)
        or not isinstance(parents, dict)
        or not isinstance(value["signer_key_id"], str)
        or not _is_key_id(value["signer_key_id"])
    ):
        raise _error("readiness_evidence_invalid")
    payload = _closed_payload(value["evidence_kind"], value["payload"])
    return ReadinessEvidenceV2(
        evidence_kind=value["evidence_kind"],
        captured_at_utc=value["captured_at_utc"],
        expires_at_utc=value["expires_at_utc"],
        freshness_seconds=value["freshness_seconds"],
        identity=identity,
        database_diagnostics=MappingProxyType(
            _closed_database_diagnostics(
                value["database_diagnostics"],
                expected_schema_version=identity.database_schema_version,
            )
        ),
        database_diagnostics_sha256=_digest(value["database_diagnostics_sha256"]),
        qualification=qualification,
        payload=MappingProxyType(payload),
        payload_sha256=_digest(value["payload_sha256"]),
        parent_evidence_sha256s=MappingProxyType(
            _closed_parents(value["evidence_kind"], parents)
        ),
        signer_key_id=value["signer_key_id"],
        evidence_sha256=_digest(value["evidence_sha256"]),
        signature=value["signature"],
    )


def _verify_evidence_signature(
    evidence: ReadinessEvidenceV2,
    verifier: ReadinessEvidenceVerifier,
) -> None:
    if verifier is None:
        raise _error("readiness_verifier_required")
    signature = _signature_bytes(evidence.signature)
    try:
        valid = verifier.verify(
            evidence.signer_key_id,
            _SIGNATURE_DOMAIN + bytes.fromhex(evidence.evidence_sha256),
            signature,
        )
    except Exception:  # noqa: BLE001 - injected verifier failures are fail-closed
        valid = False
    if valid is not True:
        raise _error("readiness_evidence_signature_invalid")


def verify_readiness_evidence(
    path: str | Path,
    *,
    verifier: ReadinessEvidenceVerifier,
    expected_kind: str,
    expected_identity: ReadinessIdentityV2,
    now: datetime | None = None,
    expected_parent_evidence_sha256s: Mapping[str, str] | None = None,
) -> ReadinessEvidenceV2:
    if verifier is None:
        raise _error("readiness_verifier_required")
    if now is not None:
        now = _as_utc(now)
    raw = _parse_canonical_document(
        _read_regular_file(path, maximum_bytes=MAXIMUM_EVIDENCE_BYTES)
    )
    evidence = _envelope_from_mapping(raw)
    if (
        evidence.evidence_kind != expected_kind
        or evidence.identity != expected_identity
    ):
        raise _error("readiness_evidence_identity_mismatch")
    captured = _parse_utc(evidence.captured_at_utc)
    expires = _parse_utc(evidence.expires_at_utc)
    if (
        not 60 <= evidence.freshness_seconds <= MAXIMUM_FRESHNESS_SECONDS
        or expires - captured != timedelta(seconds=evidence.freshness_seconds)
    ):
        raise _error("readiness_evidence_invalid")
    selected_now = datetime.now(UTC) if now is None else now
    if captured - selected_now > timedelta(seconds=MAXIMUM_CLOCK_SKEW_SECONDS):
        raise _error("readiness_evidence_not_yet_valid")
    if selected_now >= expires:
        raise _error("readiness_evidence_stale")
    expected_parents = _closed_parents(
        evidence.evidence_kind,
        (
            evidence.parent_evidence_sha256s
            if expected_parent_evidence_sha256s is None
            else expected_parent_evidence_sha256s
        ),
    )
    if dict(evidence.parent_evidence_sha256s) != expected_parents:
        raise _error("readiness_parent_evidence_invalid")
    _verify_evidence_signature(evidence, verifier)
    return evidence


def _cohort_identity_sha256(identity: ReadinessIdentityV2) -> str:
    return hashlib.sha256(
        _COHORT_IDENTITY_DOMAIN + canonical_json_bytes(identity.canonical())
    ).hexdigest()


def _canonical_evidence_destination(path: str | Path) -> Path:
    absolute = os.path.abspath(os.fspath(path))
    parent = os.path.realpath(os.path.dirname(absolute))
    return Path(os.path.normcase(os.path.join(parent, os.path.basename(absolute))))


def _cohort_destination_sha256(path: str | Path) -> str:
    absolute = os.fspath(_canonical_evidence_destination(path))
    return hashlib.sha256(
        _COHORT_DESTINATION_DOMAIN + os.fsencode(absolute)
    ).hexdigest()


def _cohort_receipt_paths(
    state_directory: str | Path,
    cohort_id: str,
) -> tuple[Path, Path]:
    selected = Path(os.path.abspath(os.fspath(state_directory)))
    return (
        selected / f"readiness-cohort-{cohort_id}.json",
        selected / f".readiness-cohort-{cohort_id}.lock",
    )


def _closed_cohort_receipt(value: Any) -> dict[str, Any]:
    try:
        if not isinstance(value, Mapping) or set(value) != _COHORT_RECEIPT_KEYS:
            raise _error("readiness_cohort_receipt_invalid")
        cohort = uuid.UUID(value["cohort_id"])
        started = _parse_utc(value["window_started_at_utc"])
        expires = _parse_utc(value["window_expires_at_utc"])
        slots = value["slots"]
        if (
            value["version"] != _COHORT_RECEIPT_VERSION
            or cohort.version != 4
            or str(cohort) != value["cohort_id"]
            or expires - started
            != timedelta(seconds=MAXIMUM_QUALIFICATION_WINDOW_SECONDS)
            or not _is_digest(value["identity_sha256"])
            or not _is_key_id(value["signer_key_id"])
            or not isinstance(slots, Mapping)
            or not 1 <= len(slots) <= len(EVIDENCE_KINDS)
        ):
            raise _error("readiness_cohort_receipt_invalid")
        normalized_slots: dict[str, dict[str, str]] = {}
        for raw_sequence, raw_slot in slots.items():
            if (
                not isinstance(raw_sequence, str)
                or not raw_sequence.isascii()
                or not raw_sequence.isdecimal()
                or str(int(raw_sequence)) != raw_sequence
                or not isinstance(raw_slot, Mapping)
                or set(raw_slot) != _COHORT_SLOT_KEYS
            ):
                raise _error("readiness_cohort_receipt_invalid")
            sequence = int(raw_sequence)
            expected_kind = _QUALIFICATION_KIND_BY_SEQUENCE.get(sequence)
            if (
                expected_kind is None
                or raw_slot["evidence_kind"] != expected_kind
                or not _is_digest(raw_slot["evidence_sha256"])
                or not _is_digest(raw_slot["destination_sha256"])
            ):
                raise _error("readiness_cohort_receipt_invalid")
            normalized_slots[raw_sequence] = {
                "destination_sha256": raw_slot["destination_sha256"],
                "evidence_kind": raw_slot["evidence_kind"],
                "evidence_sha256": raw_slot["evidence_sha256"],
            }
        return {
            "cohort_id": value["cohort_id"],
            "identity_sha256": value["identity_sha256"],
            "signer_key_id": value["signer_key_id"],
            "slots": normalized_slots,
            "version": _COHORT_RECEIPT_VERSION,
            "window_expires_at_utc": value["window_expires_at_utc"],
            "window_started_at_utc": value["window_started_at_utc"],
        }
    except OperationalError as exc:
        if exc.code == "readiness_cohort_receipt_invalid":
            raise
        raise _error("readiness_cohort_receipt_invalid") from None
    except (AttributeError, TypeError, ValueError):
        raise _error("readiness_cohort_receipt_invalid") from None


def _ensure_private_cohort_directory(path: str | Path) -> Path:
    try:
        selected = ensure_private_directory(
            os.path.realpath(os.path.abspath(os.fspath(path))),
            create=True,
        )
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            harden_windows_private_path(selected)
            validate_windows_private_path(selected)
        return selected
    except (PrivateStateError, WindowsPrivateFileError) as exc:
        raise _error("readiness_cohort_state_invalid") from exc


def _read_cohort_receipt(path: Path) -> dict[str, Any]:
    try:
        content = _read_regular_file(
            path,
            maximum_bytes=MAXIMUM_COHORT_RECEIPT_BYTES,
            require_private_state=True,
        )
        raw = _parse_strict_json_object(content)
        if canonical_json_bytes(raw) != content:
            raise _error("readiness_cohort_receipt_invalid")
        return _closed_cohort_receipt(raw)
    except OperationalError as exc:
        if exc.code == "readiness_cohort_receipt_invalid":
            raise
        raise _error("readiness_cohort_receipt_invalid") from None


def _atomic_write_cohort_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(_closed_cohort_receipt(receipt))
    if len(encoded) > MAXIMUM_COHORT_RECEIPT_BYTES:
        raise _error("readiness_cohort_receipt_invalid")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    directory_descriptor = -1
    try:
        create_or_replace_private_file(temporary, encoded)
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            harden_windows_private_path(temporary)
            validate_windows_private_path(temporary)
        os.replace(temporary, path)
        if os.name == "posix":
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0),
            )
            os.fsync(directory_descriptor)
        content = _read_regular_file(
            path,
            maximum_bytes=MAXIMUM_COHORT_RECEIPT_BYTES,
            require_private_state=True,
        )
        if content != encoded:
            raise _error("readiness_cohort_receipt_invalid")
    except OperationalError:
        raise
    except (OSError, PrivateStateError, WindowsPrivateFileError) as exc:
        raise _error("readiness_cohort_receipt_write_failed") from exc
    finally:
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        try:
            metadata = os.lstat(temporary)
            if stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                os.unlink(temporary)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _atomic_write_evidence(path: Path, encoded: bytes) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    directory_descriptor = -1
    try:
        ensure_private_directory(path.parent, create=True)
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            harden_windows_private_path(path.parent)
            validate_windows_private_path(path.parent)
        create_or_replace_private_file(temporary, encoded)
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            harden_windows_private_path(temporary)
            validate_windows_private_path(temporary)
        os.replace(temporary, path)
        if os.name == "posix":
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0),
            )
            os.fsync(directory_descriptor)
        content = _read_regular_file(
            path,
            maximum_bytes=MAXIMUM_EVIDENCE_BYTES,
            require_private_state=True,
        )
        if content != encoded:
            raise _error("readiness_evidence_write_failed")
    except OperationalError:
        raise
    except (OSError, PrivateStateError, WindowsPrivateFileError) as exc:
        raise _error("readiness_evidence_write_failed") from exc
    finally:
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        try:
            metadata = os.lstat(temporary)
            if stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                os.unlink(temporary)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _cohort_receipt_header(evidence: ReadinessEvidenceV2) -> dict[str, str]:
    return {
        "cohort_id": evidence.qualification.cohort_id,
        "identity_sha256": _cohort_identity_sha256(evidence.identity),
        "signer_key_id": evidence.signer_key_id,
        "version": _COHORT_RECEIPT_VERSION,
        "window_expires_at_utc": evidence.qualification.window_expires_at_utc,
        "window_started_at_utc": evidence.qualification.window_started_at_utc,
    }


def _cohort_slot(
    path: str | Path,
    evidence: ReadinessEvidenceV2,
) -> dict[str, str]:
    return {
        "destination_sha256": _cohort_destination_sha256(path),
        "evidence_kind": evidence.evidence_kind,
        "evidence_sha256": evidence.evidence_sha256,
    }


def _require_receipt_header(
    receipt: Mapping[str, Any],
    evidence: ReadinessEvidenceV2,
) -> None:
    header = _cohort_receipt_header(evidence)
    if any(receipt.get(key) != value for key, value in header.items()):
        raise _error("readiness_qualification_cohort_claim_mismatch")


def _claim_readiness_cohort_slot(
    state_directory: str | Path,
    destination: Path,
    evidence: ReadinessEvidenceV2,
) -> None:
    selected_state = _ensure_private_cohort_directory(state_directory)
    receipt_path, lock_path = _cohort_receipt_paths(
        selected_state,
        evidence.qualification.cohort_id,
    )
    try:
        with ProcessLock(lock_path):
            try:
                receipt = _read_cohort_receipt(receipt_path)
            except OperationalError as exc:
                if exc.code != "readiness_cohort_receipt_invalid":
                    raise
                try:
                    os.lstat(receipt_path)
                except FileNotFoundError:
                    header = _cohort_receipt_header(evidence)
                    receipt = {**header, "slots": {}}
                else:
                    raise
            _require_receipt_header(receipt, evidence)
            slots = dict(receipt["slots"])
            if evidence.evidence_kind == AGGREGATE_KIND:
                for kind in LEAF_EVIDENCE_KINDS:
                    existing_parent = slots.get(
                        str(QUALIFICATION_SEQUENCE_BY_KIND[kind])
                    )
                    if (
                        existing_parent is None
                        or existing_parent["evidence_sha256"]
                        != evidence.parent_evidence_sha256s[kind]
                    ):
                        raise _error("readiness_parent_claim_invalid")
            sequence = str(evidence.qualification.sequence)
            desired_slot = _cohort_slot(destination, evidence)
            existing_slot = slots.get(sequence)
            if existing_slot is not None:
                if existing_slot != desired_slot:
                    raise _error("readiness_qualification_sequence_claimed")
                return
            if any(
                slot["destination_sha256"] == desired_slot["destination_sha256"]
                for slot in slots.values()
            ):
                raise _error("readiness_qualification_destination_claimed")
            slots[sequence] = desired_slot
            _atomic_write_cohort_receipt(
                receipt_path,
                {**receipt, "slots": slots},
            )
    except OperationalError as exc:
        if exc.code == "service_already_running":
            raise _error("readiness_cohort_receipt_busy") from None
        raise


def validate_readiness_cohort_receipt(
    state_directory: str | Path,
    claims: Mapping[str, tuple[str | Path, ReadinessEvidenceV2]],
) -> None:
    if not isinstance(claims, Mapping) or not claims:
        raise _error("readiness_qualification_claim_missing")
    first = next(iter(claims.values()), None)
    if (
        first is None
        or not isinstance(first, tuple)
        or len(first) != 2
        or not isinstance(first[1], ReadinessEvidenceV2)
    ):
        raise _error("readiness_qualification_claim_missing")
    reference = first[1]
    selected_state = _ensure_private_cohort_directory(state_directory)
    receipt_path, lock_path = _cohort_receipt_paths(
        selected_state,
        reference.qualification.cohort_id,
    )
    try:
        with ProcessLock(lock_path):
            receipt = _read_cohort_receipt(receipt_path)
            _require_receipt_header(receipt, reference)
            slots = receipt["slots"]
            for kind, claim in claims.items():
                if (
                    not isinstance(kind, str)
                    or not isinstance(claim, tuple)
                    or len(claim) != 2
                    or not isinstance(claim[1], ReadinessEvidenceV2)
                    or claim[1].evidence_kind != kind
                ):
                    raise _error("readiness_qualification_claim_missing")
                evidence = claim[1]
                _require_receipt_header(receipt, evidence)
                if slots.get(str(evidence.qualification.sequence)) != _cohort_slot(
                    claim[0], evidence
                ):
                    raise _error("readiness_qualification_claim_mismatch")
    except OperationalError as exc:
        if exc.code == "service_already_running":
            raise _error("readiness_cohort_receipt_busy") from None
        raise


def write_readiness_evidence(
    path: str | Path,
    evidence: ReadinessEvidenceV2,
    *,
    verifier: ReadinessEvidenceVerifier,
    cohort_state_directory: str | Path,
    current_state_provider: (
        Callable[[], tuple[ReadinessIdentityV2, Any]] | None
    ) = None,
    final_time_provider: Callable[[], datetime] | None = None,
) -> Path:
    if not isinstance(evidence, ReadinessEvidenceV2):
        raise _error("readiness_evidence_invalid")
    validated = _envelope_from_mapping(evidence.canonical())
    _verify_evidence_signature(validated, verifier)
    if validated.evidence_kind == AGGREGATE_KIND:
        if current_state_provider is None:
            raise _error("readiness_current_state_required")
        try:
            current_identity, current_diagnostics = current_state_provider()
        except OperationalError:
            raise
        except Exception:  # noqa: BLE001 - injected current-state failures fail closed
            raise _error("readiness_current_state_unavailable") from None
        if current_identity != validated.identity:
            raise _error("readiness_identity_changed_during_gate")
        normalized_diagnostics = _closed_database_diagnostics(
            current_diagnostics,
            expected_schema_version=current_identity.database_schema_version,
        )
        if (
            normalized_diagnostics != dict(validated.database_diagnostics)
            or _database_diagnostics_sha256(normalized_diagnostics)
            != validated.database_diagnostics_sha256
        ):
            raise _error("readiness_database_state_changed")
        try:
            final_now = (
                final_time_provider()
                if final_time_provider is not None
                else datetime.now(UTC)
            )
        except Exception:  # noqa: BLE001 - injected clocks fail closed
            raise _error("readiness_final_time_unavailable") from None
        final_now = _as_utc(final_now, error_code="readiness_final_time_invalid")
        if _parse_utc(validated.captured_at_utc) > final_now or final_now >= _parse_utc(
            validated.qualification.window_expires_at_utc
        ):
            raise _error("readiness_qualification_window_invalid")
        if final_now >= _parse_utc(validated.expires_at_utc):
            raise _error("readiness_evidence_stale")
    destination = _canonical_evidence_destination(path)
    encoded = canonical_json_bytes(validated.canonical())
    if len(encoded) > MAXIMUM_EVIDENCE_BYTES:
        raise _error("readiness_report_too_large")
    _claim_readiness_cohort_slot(
        cohort_state_directory,
        destination,
        validated,
    )
    _atomic_write_evidence(destination, encoded)
    return destination


def sha256_regular_artifact(path: str | Path) -> str:
    selected = Path(path).absolute()
    descriptor = -1
    try:
        before_path = os.lstat(selected)
        if (
            stat.S_ISLNK(before_path.st_mode)
            or _is_windows_reparse(before_path)
            or not stat.S_ISREG(before_path.st_mode)
            or before_path.st_size > MAXIMUM_ARTIFACT_BYTES
        ):
            raise _error("readiness_artifact_invalid")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(selected, flags)
        before = os.fstat(descriptor)
        if _snapshot(before) != _snapshot(before_path):
            raise _error("readiness_artifact_invalid")
        digest = hashlib.sha256()
        read_bytes = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            read_bytes += len(chunk)
            if read_bytes > MAXIMUM_ARTIFACT_BYTES:
                raise _error("readiness_artifact_invalid")
            digest.update(chunk)
        after = os.fstat(descriptor)
        after_path = os.lstat(selected)
        if (
            read_bytes != before.st_size
            or _snapshot(after) != _snapshot(before)
            or _snapshot(after_path) != _snapshot(before)
            or _is_windows_reparse(after)
            or _is_windows_reparse(after_path)
        ):
            raise _error("readiness_artifact_invalid")
        return digest.hexdigest()
    except OperationalError:
        raise
    except OSError:
        raise _error("readiness_artifact_invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _config_profile_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _error("readiness_identity_invalid")
        return {key: _config_profile_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_config_profile_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise _error("readiness_identity_invalid")


def readiness_config_profile_sha256(config: Any) -> str:
    """Hash a canonical config projection without exposing bearer values."""

    try:
        from .config import OrchestratorConfig

        if not isinstance(config, OrchestratorConfig):
            raise TypeError
        projection: dict[str, Any] = {}
        for field in fields(config):
            value = getattr(config, field.name)
            if field.name in _BEARER_FIELDS:
                if not isinstance(value, str):
                    raise TypeError
                projection[field.name] = {
                    "redacted_sha256": hashlib.sha256(
                        _BEARER_VALUE_DOMAIN + value.encode("utf-8", errors="strict")
                    ).hexdigest()
                }
            else:
                projection[field.name] = _config_profile_value(value)
    except (TypeError, UnicodeEncodeError):
        raise _error("readiness_identity_invalid") from None
    document = {
        "configuration": projection,
        "version": "jarvis.readiness-config-profile/1",
    }
    return hashlib.sha256(
        _CONFIG_PROFILE_DOMAIN + canonical_json_bytes(document)
    ).hexdigest()


def readiness_identity_from_config(
    config: Any,
    *,
    database_schema_version: int,
    inference_identity: AuthenticatedInferenceIdentityV2,
    host_class: HostClassV1 | None = None,
    memory_crypto_state_sha256: str | None = None,
) -> ReadinessIdentityV2:
    """Resolve pinned identities and cross-check both configured artifacts."""

    required = (
        "readiness_build_artifact_file",
        "readiness_build_artifact_sha256",
        "readiness_model_artifact_file",
        "readiness_model_artifact_sha256",
        "readiness_inference_manifest_sha256",
        "readiness_inference_runtime_identity_sha256",
    )
    try:
        values = {name: getattr(config, name) for name in required}
        model_identity = config.model_alias
    except AttributeError:
        raise _error("readiness_authority_required") from None
    if any(value is None or value == "" for value in values.values()):
        raise _error("readiness_authority_required")
    build_digest = sha256_regular_artifact(values["readiness_build_artifact_file"])
    if build_digest != values["readiness_build_artifact_sha256"]:
        raise _error("readiness_build_identity_mismatch")
    model_digest = sha256_regular_artifact(values["readiness_model_artifact_file"])
    if model_digest != values["readiness_model_artifact_sha256"]:
        raise _error("readiness_model_identity_mismatch")
    if not isinstance(inference_identity, AuthenticatedInferenceIdentityV2):
        raise _error("readiness_runtime_identity_invalid")
    if (
        inference_identity.model_alias != model_identity
        or inference_identity.manifest_digest
        != values["readiness_inference_manifest_sha256"]
        or inference_identity.runtime_identity_sha256
        != values["readiness_inference_runtime_identity_sha256"]
    ):
        raise _error("readiness_runtime_identity_mismatch")
    try:
        memory_enabled = config.memory_enabled
    except AttributeError:
        raise _error("readiness_authority_required") from None
    if not isinstance(memory_enabled, bool):
        raise _error("readiness_identity_invalid")
    if memory_enabled:
        if memory_crypto_state_sha256 is None:
            raise _error("readiness_memory_crypto_required")
        selected_memory_state = _digest(
            memory_crypto_state_sha256,
            "readiness_memory_crypto_invalid",
        )
    else:
        if memory_crypto_state_sha256 not in {
            None,
            MEMORY_CRYPTO_DISABLED_STATE_SHA256,
        }:
            raise _error("readiness_memory_crypto_invalid")
        selected_memory_state = MEMORY_CRYPTO_DISABLED_STATE_SHA256
    return ReadinessIdentityV2(
        source_build_sha256=build_digest,
        config_profile_sha256=readiness_config_profile_sha256(config),
        model_identity=model_identity,
        model_digest_sha256=model_digest,
        inference_manifest_sha256=inference_identity.manifest_digest,
        inference_runtime_identity_sha256=(inference_identity.runtime_identity_sha256),
        database_schema_version=database_schema_version,
        host_class=host_class or local_host_class(),
        memory_crypto_state_sha256=selected_memory_state,
    )


def load_native_gate_payload(
    path: str | Path,
    *,
    expected_runtime_profile_sha256: str,
    expected_host_class: HostClassV1,
) -> dict[str, Any]:
    """Strictly parse raw native evidence and project its content-free result."""

    raw = _parse_strict_json_object(
        _read_regular_file(path, maximum_bytes=MAXIMUM_EVIDENCE_BYTES)
    )
    if not _is_digest(expected_runtime_profile_sha256) or not isinstance(
        expected_host_class, HostClassV1
    ):
        raise _error("native_voice_evidence_identity_mismatch")
    try:
        from jarvis_voice.errors import BackendError
        from jarvis_voice.native_qualification import evaluate_native_evidence

        evaluated = evaluate_native_evidence(
            raw,
            expected_runtime_profile_sha256=expected_runtime_profile_sha256,
        )
    except (BackendError, ImportError, TypeError, ValueError):
        raise _error("native_voice_evidence_invalid") from None
    runtime_profile = raw.get("runtime_profile")
    if (
        not isinstance(runtime_profile, Mapping)
        or raw.get("platform") != expected_host_class.os_family
        or runtime_profile.get("mode") != expected_host_class.runtime_mode
        or runtime_profile.get("fingerprint_sha256") != expected_runtime_profile_sha256
    ):
        raise _error("native_voice_evidence_identity_mismatch")
    projected = {
        **evaluated,
        "runtime_mode": runtime_profile["mode"],
        "runtime_profile_sha256": expected_runtime_profile_sha256,
    }
    return _closed_payload(NATIVE_KIND, projected)


def load_ed25519_signer(
    path: str | Path,
    *,
    key_id: str,
    expected_public_key_sha256: str,
) -> Ed25519EvidenceSigner:
    _digest(expected_public_key_sha256, "readiness_signer_invalid")
    content = _read_regular_file(
        Path(path).absolute(),
        maximum_bytes=MAXIMUM_KEY_FILE_BYTES,
        require_private_key=True,
    )
    try:
        key = serialization.load_pem_private_key(content, password=None)
    except (TypeError, ValueError):
        raise _error("readiness_signer_invalid") from None
    if not isinstance(key, Ed25519PrivateKey):
        raise _error("readiness_signer_invalid")
    raw_public = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    if hashlib.sha256(raw_public).hexdigest() != expected_public_key_sha256:
        raise _error("readiness_signer_pin_mismatch")
    return Ed25519EvidenceSigner(key_id, key)


def load_pinned_ed25519_verifier(
    path: str | Path,
    *,
    key_id: str,
    expected_public_key_sha256: str,
) -> Ed25519EvidenceVerifier:
    _digest(expected_public_key_sha256, "readiness_verifier_invalid")
    content = _read_regular_file(path, maximum_bytes=MAXIMUM_KEY_FILE_BYTES)
    try:
        key = serialization.load_pem_public_key(content)
    except (TypeError, ValueError):
        raise _error("readiness_verifier_invalid") from None
    if not isinstance(key, Ed25519PublicKey):
        raise _error("readiness_verifier_invalid")
    raw_public = key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    if hashlib.sha256(raw_public).hexdigest() != expected_public_key_sha256:
        raise _error("readiness_verifier_pin_mismatch")
    return Ed25519EvidenceVerifier({key_id: key})


__all__ = [
    "AGGREGATE_KIND",
    "LEAF_EVIDENCE_KINDS",
    "LIVE_KIND",
    "MEMORY_CRYPTO_DISABLED_STATE_SHA256",
    "NATIVE_KIND",
    "READINESS_AGGREGATE_VERSION",
    "READINESS_EVIDENCE_VERSION",
    "RECOVERY_KIND",
    "RESTART_KIND",
    "SOAK_KIND",
    "SYNTHETIC_KIND",
    "AuthenticatedInferenceIdentityV2",
    "Ed25519EvidenceSigner",
    "Ed25519EvidenceVerifier",
    "HostClassV1",
    "QualificationBindingV2",
    "ReadinessEvidenceV2",
    "ReadinessIdentityV2",
    "build_readiness_evidence",
    "canonical_json_bytes",
    "load_ed25519_signer",
    "load_native_gate_payload",
    "load_pinned_ed25519_verifier",
    "local_host_class",
    "qualification_binding_for_kind",
    "readiness_config_profile_sha256",
    "readiness_identity_from_config",
    "sha256_regular_artifact",
    "validate_readiness_cohort_receipt",
    "validated_readiness_database_diagnostics",
    "verify_readiness_evidence",
    "write_readiness_evidence",
]
