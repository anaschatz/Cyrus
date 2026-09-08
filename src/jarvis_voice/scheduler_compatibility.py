"""Strict Voice compatibility manifest gate for scheduler advertisement."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MAX_SCHEDULER_DOCUMENT_BYTES = 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate scheduler JSON key")
        value[key] = item
    return value


def load_pinned_scheduler_json(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    """Runtime gates always require an independently supplied digest."""
    if expected_sha256 is None:
        raise ValueError("scheduler document digest invalid")
    return load_scheduler_json(path, expected_sha256=expected_sha256)


def load_scheduler_json(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    """Read one bounded snapshot; qualification producers may omit the pin."""
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("scheduler document digest invalid")
    try:
        with open(path, "rb", opener=lambda name, flags: os.open(
            name, flags | getattr(os, "O_NONBLOCK", 0)
        )) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("scheduler document must be a regular file")
            raw = stream.read(MAX_SCHEDULER_DOCUMENT_BYTES + 1)
    except OSError as exc:
        raise ValueError("scheduler document unreadable") from exc
    if not raw or len(raw) > MAX_SCHEDULER_DOCUMENT_BYTES:
        raise ValueError("scheduler document size invalid")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("scheduler document digest mismatch")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        # Check finite numbers and Unicode scalar values without changing pinned bytes.
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, RecursionError) as exc:
        raise ValueError("scheduler document JSON invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("scheduler document JSON must be an object")
    return value, digest


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
    value, digest = load_pinned_scheduler_json(path, expected_sha256)
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
    if (value["scheduler_playback_isolation"] is not True
            or type(value["maximum_frame_bytes"]) is not int or value["maximum_frame_bytes"] != 16384):
        raise ValueError("scheduler playback/frame contract invalid")
    if not isinstance(value["key_provider_contract"], str) or not value["key_provider_contract"]:
        raise ValueError("scheduler key provider contract invalid")
    for name in (
        "lock_state_provider_sha256", "notification_adapter_qualification_sha256",
        "tts_adapter_qualification_sha256", "artifact_signature_sha256",
    ):
        item = value[name]
        if name == "tts_adapter_qualification_sha256" and item is None:
            continue
        if not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None:
            raise ValueError(f"{name} invalid")
    capabilities = value["capabilities"]
    if (not isinstance(capabilities, list) or any(not isinstance(item, str) for item in capabilities)
            or len(set(capabilities)) != len(capabilities)):
        raise ValueError("scheduler manifest capabilities invalid")
    capabilities = tuple(capabilities)
    allowed = {"jarvis.scheduler/1", "jarvis.approval-arbiter/1", "scheduler_notification_speech", "scheduler_notification_actions"}
    if set(capabilities) - allowed or not {"jarvis.scheduler/1", "jarvis.approval-arbiter/1"}.issubset(capabilities):
        raise ValueError("scheduler manifest capabilities invalid")
    if "scheduler_notification_speech" in capabilities and value["tts_adapter_qualification_sha256"] is None:
        raise ValueError("scheduler TTS qualification missing")
    return SchedulerCompatibilityManifest(digest, capabilities, value)
