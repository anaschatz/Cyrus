"""Bind scheduler speech advertisement to strict native voice evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .native_qualification import evaluate_native_evidence
from .scheduler_compatibility import SchedulerCompatibilityManifest, load_pinned_scheduler_json


@dataclass(frozen=True, slots=True)
class SchedulerTtsQualification:
    sha256: str
    platform: str
    runtime_profile_sha256: str


def load_scheduler_tts_qualification(
    path: Path,
    expected_sha256: str,
    *,
    platform_name: str,
    compatibility: SchedulerCompatibilityManifest,
) -> SchedulerTtsQualification:
    evidence, digest = load_pinned_scheduler_json(path, expected_sha256)
    report = evaluate_native_evidence(evidence)
    expected_platform = "windows" if platform_name == "win32" else "macos" if platform_name == "darwin" else ""
    if report["ready"] is not True or report["platform"] != expected_platform:
        raise ValueError("scheduler TTS native qualification is not ready")
    if compatibility.value["tts_adapter_qualification_sha256"] != digest:
        raise ValueError("scheduler TTS qualification binding mismatch")
    runtime_profile = evidence["runtime_profile"]["fingerprint_sha256"]
    if not isinstance(runtime_profile, str):
        raise ValueError("scheduler TTS runtime profile is missing")
    return SchedulerTtsQualification(digest, expected_platform, runtime_profile)
