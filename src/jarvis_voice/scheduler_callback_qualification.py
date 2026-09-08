"""Pinned qualification contract for native scheduler action callbacks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .scheduler_compatibility import load_pinned_scheduler_json


SCHEMA_VERSION = "scheduler-native-callback-qualification/1"
REQUIRED_CHECKS = frozenset({
    "signed_identity", "permission_probe", "post", "replace", "remove", "silent",
    "snooze_callback", "cancel_callback", "callback_authentication", "reconnect_replay",
})


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} invalid")
    return value


@dataclass(frozen=True, slots=True)
class SchedulerCallbackQualification:
    sha256: str
    helper_sha256: str
    helper_code_identity_sha256: str
    adapter_sha256: str
    value: Mapping[str, Any]


def load_scheduler_callback_qualification(
    path: Path, expected_sha256: str, *, platform_name: str
) -> SchedulerCallbackQualification:
    value, digest = load_pinned_scheduler_json(path, expected_sha256)
    fields = {
        "schema_version", "platform", "ready", "native_helper_sha256",
        "native_helper_code_identity_sha256", "notification_adapter_sha256",
        "checks", "qualified_at_utc", "artifact_manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("scheduler callback qualification fields invalid")
    expected_platform = "macos" if platform_name == "darwin" else "windows" if platform_name == "win32" else ""
    if (
        not expected_platform
        or value["schema_version"] != SCHEMA_VERSION
        or value["platform"] != expected_platform
        or value["ready"] is not True
    ):
        raise ValueError("scheduler callback qualification is not ready")
    checks = value["checks"]
    if not isinstance(checks, Mapping) or set(checks) != REQUIRED_CHECKS or any(
        checks[name] != "pass" for name in REQUIRED_CHECKS
    ):
        raise ValueError("scheduler callback qualification checks invalid")
    try:
        qualified_at = datetime.fromisoformat(str(value["qualified_at_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("scheduler callback qualification timestamp invalid") from exc
    if qualified_at.tzinfo is None or qualified_at.utcoffset() != UTC.utcoffset(qualified_at):
        raise ValueError("scheduler callback qualification timestamp invalid")
    _sha(value["artifact_manifest_sha256"], "artifact manifest digest")
    return SchedulerCallbackQualification(
        digest,
        _sha(value["native_helper_sha256"], "native helper digest"),
        _sha(value["native_helper_code_identity_sha256"], "native helper identity"),
        _sha(value["notification_adapter_sha256"], "notification adapter digest"),
        value,
    )
