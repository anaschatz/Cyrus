"""Strict, content-free native hardware acceptance evidence.

Unlike deterministic smoke tests, this gate can become ready only from a real
machine run with physical PTT, capture, playback, acoustic silence detection,
recovery exercises, and an eight-hour soak.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import BackendError
from .qualification import write_report


SCHEMA_VERSION = 1
MINIMUM_REAL_VOICE_TURNS = 200
MAX_PTT_TO_ACOUSTIC_SILENCE_MS = 200.0
MINIMUM_ACOUSTIC_SAMPLES = 20
MINIMUM_SOAK_SECONDS = 8 * 60 * 60
MINIMUM_SOAK_RSS_SAMPLES = 97
MAX_RSS_SLOPE_BYTES_PER_HOUR = 1 * 1024 * 1024
MAX_RSS_END_GROWTH_BYTES = 32 * 1024 * 1024

COMMON_CASES = (
    "microphone_disconnect_recovery",
    "sleep_resume_recovery",
    "acoustic_barge_in",
)
WINDOWS_CASES = (
    "raw_input",
    "wasapi_capture",
    "sapi5_playback",
    "tray",
    "suspend_resume",
    "lock_unlock",
    "microphone_change",
    "missed_key_up",
    "packaged_executable",
    *COMMON_CASES,
)
MACOS_CASES = (
    "coreaudio_capture",
    "accessibility_permission",
    "ptt_event_tap",
    "selected_tts_backend",
    "suspend_resume",
    "microphone_change",
    "stable_runtime_profile",
    *COMMON_CASES,
)
PLATFORMS = {"windows": WINDOWS_CASES, "macos": MACOS_CASES}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _error(code: str) -> BackendError:
    return BackendError(code, "Native qualification evidence is invalid", subsystem="qualification")


def _pending_case() -> dict[str, object]:
    return {"status": "pending", "attempts": 0}


def native_evidence_template(platform_name: str, *, run_id: str | None = None) -> dict[str, object]:
    """Create a strict pending template; it is never evidence of a pass."""

    if platform_name not in PLATFORMS:
        raise _error("qualification.native_platform_invalid")
    identifier = str(uuid.uuid4()) if run_id is None else run_id
    try:
        parsed = uuid.UUID(identifier)
    except (ValueError, AttributeError) as exc:
        raise _error("qualification.native_run_id_invalid") from exc
    if parsed.version != 4 or str(parsed) != identifier:
        raise _error("qualification.native_run_id_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": identifier,
        "platform": platform_name,
        "provenance": {
            "native_machine": False,
            "voice_layer_mocks_used": True,
            "synthetic_voice_turns": True,
            "hardware_capture": False,
            "physical_ptt": False,
            "physical_playback": False,
        },
        "cases": {name: _pending_case() for name in PLATFORMS[platform_name]},
        "counters": {
            "real_voice_turns": 0,
            "duplicate_transcripts": 0,
            "duplicate_tts_playbacks": 0,
            "stuck_ptt": 0,
            "microphone_disconnect_recoveries": 0,
            "sleep_resume_recoveries": 0,
            "crashes": 0,
        },
        "acoustic_barge_in": {
            "method": "pending",
            "audio_path": "pending",
            "samples": 0,
            "max_ptt_to_silence_ms": None,
            "silence_hold_ms": 0,
        },
        "soak": {
            "duration_seconds": 0,
            "rss_samples": 0,
            "rss_start_bytes": None,
            "rss_end_bytes": None,
            "rss_peak_bytes": None,
            "rss_slope_bytes_per_hour": None,
        },
        "runtime_profile": {
            "mode": "pending",
            "fingerprint_sha256": None,
            "stable": False,
            "signed": False,
        },
        "privacy": {
            "audio_persisted": False,
            "transcript_text_persisted": False,
            "device_identifiers_persisted": False,
            "content_captured": False,
        },
    }


def _exact_keys(value: Any, expected: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == expected


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def validate_native_evidence(raw: Any) -> dict[str, object]:
    top_keys = {
        "schema_version",
        "run_id",
        "platform",
        "provenance",
        "cases",
        "counters",
        "acoustic_barge_in",
        "soak",
        "runtime_profile",
        "privacy",
    }
    if not _exact_keys(raw, top_keys) or raw.get("schema_version") != SCHEMA_VERSION:
        raise _error("qualification.native_schema_invalid")
    platform_name = raw.get("platform")
    if platform_name not in PLATFORMS:
        raise _error("qualification.native_platform_invalid")
    try:
        identifier = uuid.UUID(raw.get("run_id"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise _error("qualification.native_run_id_invalid") from exc
    if identifier.version != 4 or str(identifier) != raw.get("run_id"):
        raise _error("qualification.native_run_id_invalid")

    provenance_keys = {
        "native_machine",
        "voice_layer_mocks_used",
        "synthetic_voice_turns",
        "hardware_capture",
        "physical_ptt",
        "physical_playback",
    }
    provenance = raw.get("provenance")
    if not _exact_keys(provenance, provenance_keys) or any(
        not isinstance(provenance[name], bool) for name in provenance_keys
    ):
        raise _error("qualification.native_provenance_invalid")

    cases = raw.get("cases")
    expected_cases = set(PLATFORMS[str(platform_name)])
    if not _exact_keys(cases, expected_cases):
        raise _error("qualification.native_cases_invalid")
    for value in cases.values():
        if (
            not _exact_keys(value, {"status", "attempts"})
            or value.get("status") not in {"pending", "pass", "fail"}
            or not _nonnegative_integer(value.get("attempts"))
            or (value.get("status") == "pass" and value.get("attempts") < 1)
        ):
            raise _error("qualification.native_cases_invalid")

    counter_keys = {
        "real_voice_turns",
        "duplicate_transcripts",
        "duplicate_tts_playbacks",
        "stuck_ptt",
        "microphone_disconnect_recoveries",
        "sleep_resume_recoveries",
        "crashes",
    }
    counters = raw.get("counters")
    if not _exact_keys(counters, counter_keys) or any(
        not _nonnegative_integer(counters[name]) for name in counter_keys
    ):
        raise _error("qualification.native_counters_invalid")

    acoustic_keys = {
        "method",
        "audio_path",
        "samples",
        "max_ptt_to_silence_ms",
        "silence_hold_ms",
    }
    acoustic = raw.get("acoustic_barge_in")
    if (
        not _exact_keys(acoustic, acoustic_keys)
        or acoustic.get("method") not in {"pending", "acoustic_microphone"}
        or acoustic.get("audio_path") not in {"pending", "wired", "built_in"}
        or not _nonnegative_integer(acoustic.get("samples"))
        or not _nonnegative_integer(acoustic.get("silence_hold_ms"))
        or (
            acoustic.get("max_ptt_to_silence_ms") is not None
            and not _finite_nonnegative(acoustic.get("max_ptt_to_silence_ms"))
        )
    ):
        raise _error("qualification.native_acoustic_invalid")

    soak_keys = {
        "duration_seconds",
        "rss_samples",
        "rss_start_bytes",
        "rss_end_bytes",
        "rss_peak_bytes",
        "rss_slope_bytes_per_hour",
    }
    soak = raw.get("soak")
    if not _exact_keys(soak, soak_keys):
        raise _error("qualification.native_soak_invalid")
    if not _nonnegative_integer(soak.get("duration_seconds")) or not _nonnegative_integer(soak.get("rss_samples")):
        raise _error("qualification.native_soak_invalid")
    for name in ("rss_start_bytes", "rss_end_bytes", "rss_peak_bytes"):
        if soak.get(name) is not None and not _nonnegative_integer(soak.get(name)):
            raise _error("qualification.native_soak_invalid")
    slope = soak.get("rss_slope_bytes_per_hour")
    if slope is not None and (
        not isinstance(slope, (int, float)) or isinstance(slope, bool) or not math.isfinite(float(slope))
    ):
        raise _error("qualification.native_soak_invalid")

    profile = raw.get("runtime_profile")
    if (
        not _exact_keys(profile, {"mode", "fingerprint_sha256", "stable", "signed"})
        or profile.get("mode") not in {"pending", "source", "signed_package"}
        or not isinstance(profile.get("stable"), bool)
        or not isinstance(profile.get("signed"), bool)
        or (
            profile.get("fingerprint_sha256") is not None
            and (not isinstance(profile.get("fingerprint_sha256"), str) or _SHA256.fullmatch(profile["fingerprint_sha256"]) is None)
        )
    ):
        raise _error("qualification.native_profile_invalid")

    privacy = raw.get("privacy")
    privacy_keys = {"audio_persisted", "transcript_text_persisted", "device_identifiers_persisted", "content_captured"}
    if not _exact_keys(privacy, privacy_keys) or any(not isinstance(privacy[name], bool) for name in privacy_keys):
        raise _error("qualification.native_privacy_invalid")
    return dict(raw)


def _gate(ready: bool, ready_code: str, blocked_code: str, **details: object) -> dict[str, object]:
    return {"status": "ready" if ready else "blocked", "code": ready_code if ready else blocked_code, **details}


def evaluate_native_evidence(raw: Any) -> dict[str, object]:
    evidence = validate_native_evidence(raw)
    provenance = evidence["provenance"]
    cases = evidence["cases"]
    counters = evidence["counters"]
    acoustic = evidence["acoustic_barge_in"]
    soak = evidence["soak"]
    profile = evidence["runtime_profile"]
    privacy = evidence["privacy"]
    platform_name = str(evidence["platform"])

    native = (
        provenance["native_machine"] is True
        and provenance["voice_layer_mocks_used"] is False
        and provenance["synthetic_voice_turns"] is False
        and provenance["hardware_capture"] is True
        and provenance["physical_ptt"] is True
        and provenance["physical_playback"] is True
    )
    runtime_cases_ready = all(item["status"] == "pass" and item["attempts"] >= 1 for item in cases.values())
    turn_integrity = (
        counters["real_voice_turns"] >= MINIMUM_REAL_VOICE_TURNS
        and counters["duplicate_transcripts"] == 0
        and counters["duplicate_tts_playbacks"] == 0
        and counters["stuck_ptt"] == 0
    )
    acoustic_ready = (
        acoustic["method"] == "acoustic_microphone"
        and acoustic["audio_path"] in {"wired", "built_in"}
        and acoustic["samples"] >= MINIMUM_ACOUSTIC_SAMPLES
        and acoustic["max_ptt_to_silence_ms"] is not None
        and float(acoustic["max_ptt_to_silence_ms"]) < MAX_PTT_TO_ACOUSTIC_SILENCE_MS
        and acoustic["silence_hold_ms"] >= 50
    )
    recovery_ready = (
        counters["microphone_disconnect_recoveries"] >= 1
        and counters["sleep_resume_recoveries"] >= 1
        and cases["microphone_disconnect_recovery"]["status"] == "pass"
        and cases["sleep_resume_recovery"]["status"] == "pass"
    )
    rss_values_ready = all(soak[name] is not None for name in ("rss_start_bytes", "rss_end_bytes", "rss_peak_bytes", "rss_slope_bytes_per_hour"))
    memory_stable = False
    if rss_values_ready:
        memory_stable = (
            soak["rss_peak_bytes"] >= max(soak["rss_start_bytes"], soak["rss_end_bytes"])
            and soak["rss_end_bytes"] <= soak["rss_start_bytes"] + MAX_RSS_END_GROWTH_BYTES
            and soak["rss_slope_bytes_per_hour"] <= MAX_RSS_SLOPE_BYTES_PER_HOUR
        )
    soak_ready = (
        soak["duration_seconds"] >= MINIMUM_SOAK_SECONDS
        and soak["rss_samples"] >= MINIMUM_SOAK_RSS_SAMPLES
        and counters["crashes"] == 0
        and memory_stable
    )
    profile_ready = (
        profile["stable"] is True
        and isinstance(profile["fingerprint_sha256"], str)
        and (
            (platform_name == "windows" and profile["mode"] == "signed_package" and profile["signed"] is True)
            or (
                platform_name == "macos"
                and profile["mode"] in {"source", "signed_package"}
                and (profile["mode"] != "signed_package" or profile["signed"] is True)
            )
        )
    )
    privacy_ready = all(privacy[name] is False for name in privacy)

    checks = {
        "native_provenance": _gate(native, "qualification.native_hardware_verified", "qualification.native_hardware_required"),
        "runtime_cases": _gate(runtime_cases_ready, "qualification.native_cases_passed", "qualification.native_cases_incomplete"),
        "turn_integrity": _gate(
            turn_integrity,
            "qualification.native_turn_integrity_passed",
            "qualification.native_turn_integrity_failed",
            real_voice_turns=counters["real_voice_turns"],
            minimum_real_voice_turns=MINIMUM_REAL_VOICE_TURNS,
            duplicate_transcripts=counters["duplicate_transcripts"],
            duplicate_tts_playbacks=counters["duplicate_tts_playbacks"],
            stuck_ptt=counters["stuck_ptt"],
        ),
        "acoustic_barge_in": _gate(
            acoustic_ready,
            "qualification.native_acoustic_barge_in_passed",
            "qualification.native_acoustic_barge_in_required",
            samples=acoustic["samples"],
            maximum_ms=acoustic["max_ptt_to_silence_ms"],
            threshold_ms=MAX_PTT_TO_ACOUSTIC_SILENCE_MS,
        ),
        "automatic_recovery": _gate(recovery_ready, "qualification.native_recovery_passed", "qualification.native_recovery_required"),
        "soak": _gate(
            soak_ready,
            "qualification.native_soak_passed",
            "qualification.native_soak_required",
            duration_seconds=soak["duration_seconds"],
            minimum_duration_seconds=MINIMUM_SOAK_SECONDS,
            rss_samples=soak["rss_samples"],
            memory_stable=memory_stable,
            crashes=counters["crashes"],
        ),
        "runtime_profile": _gate(profile_ready, "qualification.native_profile_stable", "qualification.native_profile_required"),
        "privacy": _gate(privacy_ready, "qualification.native_privacy_passed", "qualification.native_privacy_failed"),
    }
    ready = all(check["status"] == "ready" for check in checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "ready": ready,
        "code": "qualification.native_ready" if ready else "qualification.native_blocked",
        "platform": platform_name,
        "checks": checks,
        "content_captured": False,
        "network_used": False,
    }


def load_native_evidence(path: str | Path) -> dict[str, object]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error("qualification.native_evidence_unreadable") from exc
    return validate_native_evidence(raw)


def write_native_evidence_template(path: str | Path, platform_name: str) -> Path:
    return write_report(path, native_evidence_template(platform_name))
