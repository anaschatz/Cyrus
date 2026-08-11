"""Strict Voice compatibility manifest gate for scheduler advertisement."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FIELDS = {
    "schema_version", "supported_protocols", "capabilities", "sequence_namespace",
    "cpython", "unicode_db", "aead_contract", "key_provider_contract",
    "scheduler_journal_schema", "scheduler_playback_isolation", "lock_state_provider_sha256",
    "notification_adapter_qualification_sha256", "tts_adapter_qualification_sha256",
    "maximum_frame_bytes", "artifact_signature_sha256",
}


@dataclass(frozen=True, slots=True)
class SchedulerCompatibilityManifest:
    sha256: str
    capabilities: tuple[str, ...]
    value: Mapping[str, Any]


def load_scheduler_compatibility(path: Path, expected_sha256: str) -> SchedulerCompatibilityManifest:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("scheduler compatibility manifest digest mismatch")
    value = json.loads(raw)
    if not isinstance(value, Mapping) or set(value) != FIELDS:
        raise ValueError("scheduler compatibility manifest fields invalid")
    if value["schema_version"] != "voice-scheduler-compatibility/1":
        raise ValueError("scheduler compatibility manifest version invalid")
    if value["supported_protocols"] != ["jarvis.scheduler/1", "jarvis.approval-arbiter/1"]:
        raise ValueError("scheduler protocols invalid")
    if value["sequence_namespace"] != "protocol_direction_connection":
        raise ValueError("scheduler sequence namespace invalid")
    if value["cpython"] != "3.11.14" or value["unicode_db"] != "14.0.0":
        raise ValueError("scheduler unicode contract invalid")
    if sys.version_info[:3] != (3, 11, 14) or unicodedata.unidata_version != "14.0.0":
        raise ValueError("scheduler runtime mismatch")
    if value["aead_contract"] != "scheduler-content-aead/1":
        raise ValueError("scheduler AEAD contract invalid")
    if value["scheduler_journal_schema"] != "voice-scheduler-journal/1":
        raise ValueError("scheduler journal contract invalid")
    if value["scheduler_playback_isolation"] is not True or value["maximum_frame_bytes"] != 16384:
        raise ValueError("scheduler playback/frame contract invalid")
    for name in (
        "lock_state_provider_sha256", "notification_adapter_qualification_sha256",
        "tts_adapter_qualification_sha256", "artifact_signature_sha256",
    ):
        item = value[name]
        if item is not None and not re.fullmatch(r"[0-9a-f]{64}", item):
            raise ValueError(f"{name} invalid")
    capabilities = tuple(value["capabilities"])
    allowed = {"jarvis.scheduler/1", "jarvis.approval-arbiter/1", "scheduler_notification_speech", "scheduler_notification_actions"}
    if set(capabilities) - allowed or not {"jarvis.scheduler/1", "jarvis.approval-arbiter/1"}.issubset(capabilities):
        raise ValueError("scheduler manifest capabilities invalid")
    return SchedulerCompatibilityManifest(digest, capabilities, value)
