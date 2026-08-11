"""Fail-closed qualification gate for S2 native scheduler delivery."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .scheduler_compatibility import SchedulerCompatibilityManifest


SCHEMA_VERSION = "scheduler-native-qualification/1"
REQUIRED_CHECKS = frozenset(
    {
        "permission_probe",
        "post",
        "replace",
        "remove",
        "silent",
        "callback_idempotency",
        "locked_privacy",
        "unknown_privacy",
        "restart_recovery",
    }
)


@dataclass(frozen=True, slots=True)
class NativeSchedulerQualification:
    sha256: str
    platform: str
    notification_adapter_sha256: str
    lock_state_provider_sha256: str
    value: Mapping[str, Any]


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} invalid")
    return value


def load_native_scheduler_qualification(
    path: Path,
    expected_sha256: str,
    *,
    platform_name: str,
    compatibility: SchedulerCompatibilityManifest,
) -> NativeSchedulerQualification:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("scheduler native qualification digest mismatch")
    value = json.loads(raw)
    fields = {
        "schema_version", "platform", "ready", "permission", "capabilities",
        "notification_adapter_sha256", "lock_state_provider_sha256", "checks",
        "qualified_at_utc", "artifact_signature_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("scheduler native qualification fields invalid")
    expected_platform = "windows" if platform_name == "win32" else "macos" if platform_name == "darwin" else ""
    if value["schema_version"] != SCHEMA_VERSION or value["platform"] != expected_platform:
        raise ValueError("scheduler native qualification platform invalid")
    if value["ready"] is not True or value["permission"] != "granted":
        raise ValueError("scheduler native qualification is not ready")
    if value["capabilities"] != ["post", "replace", "remove", "silent", "lock_state"]:
        raise ValueError("scheduler native capabilities invalid")
    checks = value["checks"]
    if not isinstance(checks, Mapping) or set(checks) != REQUIRED_CHECKS or any(
        checks[name] != "pass" for name in REQUIRED_CHECKS
    ):
        raise ValueError("scheduler native qualification checks invalid")
    notification_digest = _sha(value["notification_adapter_sha256"], "notification adapter digest")
    lock_digest = _sha(value["lock_state_provider_sha256"], "lock-state provider digest")
    artifact_digest = _sha(value["artifact_signature_sha256"], "artifact signature digest")
    try:
        qualified_at = datetime.fromisoformat(str(value["qualified_at_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("scheduler qualification timestamp invalid") from exc
    if qualified_at.tzinfo is None or qualified_at.utcoffset() != UTC.utcoffset(qualified_at):
        raise ValueError("scheduler qualification timestamp invalid")
    manifest = compatibility.value
    if (
        manifest["notification_adapter_qualification_sha256"] != digest
        or manifest["lock_state_provider_sha256"] != lock_digest
        or manifest["artifact_signature_sha256"] != artifact_digest
    ):
        raise ValueError("scheduler native qualification binding mismatch")
    return NativeSchedulerQualification(
        digest, expected_platform, notification_digest, lock_digest, value
    )
