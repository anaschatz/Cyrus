"""Strict aggregation for bounded, content-free native qualification runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from . import PROTOCOL_VERSION, __version__
from .errors import BackendError
from .native_qualification import (
    MAX_PTT_TO_ACOUSTIC_SILENCE_MS,
    MINIMUM_ACOUSTIC_SAMPLES,
    evaluate_native_evidence,
    native_evidence_template,
    validate_native_evidence,
)
from .voice_auth import AUTH_PROTOCOL


BUNDLE_SCHEMA_VERSION = 1
MAXIMUM_SOURCE_RUNS = 256


def _error(code: str) -> BackendError:
    return BackendError(
        code,
        "Native qualification evidence bundle is invalid",
        subsystem="qualification",
    )


def _semantic_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _merged_case(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    attempts = sum(int(value["attempts"]) for value in values)
    statuses = {str(value["status"]) for value in values}
    status = "fail" if "fail" in statuses else "pass" if "pass" in statuses else "pending"
    return {"status": status, "attempts": attempts}


def aggregate_native_evidence(runs: Sequence[Any]) -> dict[str, object]:
    """Aggregate bounded runs without treating disjoint runs as one soak."""

    if (
        not isinstance(runs, Sequence)
        or isinstance(runs, (str, bytes, bytearray))
        or not 1 <= len(runs) <= MAXIMUM_SOURCE_RUNS
    ):
        raise _error("qualification.native_bundle_source_count_invalid")
    evidence = [validate_native_evidence(value) for value in runs]
    evidence.sort(key=lambda value: str(value["run_id"]))
    run_ids = [str(value["run_id"]) for value in evidence]
    if len(set(run_ids)) != len(run_ids):
        raise _error("qualification.native_bundle_duplicate_run")

    first = evidence[0]
    platform_name = str(first["platform"])
    profile = dict(first["runtime_profile"])
    for value in evidence:
        if value["platform"] != platform_name:
            raise _error("qualification.native_bundle_platform_mismatch")
        if value["runtime_profile"] != profile:
            raise _error("qualification.native_bundle_profile_mismatch")
        provenance = value["provenance"]
        if (
            provenance["native_machine"] is not True
            or provenance["voice_layer_mocks_used"] is not False
            or provenance["synthetic_voice_turns"] is not False
        ):
            raise _error("qualification.native_bundle_provenance_invalid")
        acoustic = value["acoustic_barge_in"]
        if acoustic["samples"] > 0 and (
            acoustic["method"] == "pending"
            or acoustic["audio_path"] == "pending"
            or acoustic["max_ptt_to_silence_ms"] is None
        ):
            raise _error("qualification.native_bundle_acoustic_invalid")
        if any(value["privacy"].values()):
            raise _error("qualification.native_bundle_privacy_invalid")

    fingerprint = profile.get("fingerprint_sha256")
    if not isinstance(fingerprint, str):
        raise _error("qualification.native_bundle_profile_mismatch")
    aggregate = native_evidence_template(
        platform_name,
        runtime_profile_sha256=fingerprint,
    )
    aggregate["runtime_profile"] = profile
    aggregate["provenance"] = {
        "native_machine": True,
        "voice_layer_mocks_used": False,
        "synthetic_voice_turns": False,
        "hardware_capture": any(value["provenance"]["hardware_capture"] for value in evidence),
        "physical_ptt": any(value["provenance"]["physical_ptt"] for value in evidence),
        "physical_playback": any(value["provenance"]["physical_playback"] for value in evidence),
    }
    aggregate["privacy"] = dict(first["privacy"])

    aggregate["cases"] = {
        name: _merged_case([value["cases"][name] for value in evidence])
        for name in first["cases"]
    }
    aggregate["counters"] = {
        name: sum(int(value["counters"][name]) for value in evidence)
        for name in first["counters"]
    }

    acoustic_profiles = {
        (
            value["acoustic_barge_in"]["method"],
            value["acoustic_barge_in"]["audio_path"],
            value["acoustic_barge_in"]["silence_hold_ms"],
        )
        for value in evidence
        if value["acoustic_barge_in"]["method"] != "pending"
    }
    if len(acoustic_profiles) > 1:
        raise _error("qualification.native_bundle_acoustic_profile_mismatch")
    acoustic_samples = sum(int(value["acoustic_barge_in"]["samples"]) for value in evidence)
    acoustic_maxima = [
        float(value["acoustic_barge_in"]["max_ptt_to_silence_ms"])
        for value in evidence
        if value["acoustic_barge_in"]["max_ptt_to_silence_ms"] is not None
    ]
    if acoustic_profiles:
        method, audio_path, silence_hold_ms = next(iter(acoustic_profiles))
    else:
        method, audio_path, silence_hold_ms = "pending", "pending", 0
    aggregate["acoustic_barge_in"] = {
        "method": method,
        "audio_path": audio_path,
        "samples": acoustic_samples,
        "max_ptt_to_silence_ms": max(acoustic_maxima) if acoustic_maxima else None,
        "silence_hold_ms": silence_hold_ms,
    }
    if (
        acoustic_samples >= MINIMUM_ACOUSTIC_SAMPLES
        and acoustic_maxima
        and max(acoustic_maxima) < MAX_PTT_TO_ACOUSTIC_SILENCE_MS
        and aggregate["cases"]["acoustic_barge_in"]["status"] != "fail"
    ):
        aggregate["cases"]["acoustic_barge_in"] = {
            "status": "pass",
            "attempts": acoustic_samples,
        }

    # The soak gate requires one continuous run. Selecting the longest source
    # preserves that property; summing durations would create false evidence.
    longest = max(
        evidence,
        key=lambda value: (
            int(value["soak"]["duration_seconds"]),
            int(value["soak"]["rss_samples"]),
        ),
    )
    aggregate["soak"] = dict(longest["soak"])

    sources = [
        {
            "run_id": str(value["run_id"]),
            "evidence_sha256": _semantic_sha256(value),
        }
        for value in evidence
    ]
    report = evaluate_native_evidence(aggregate)
    return {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "product_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "voice_auth_protocol": AUTH_PROTOCOL,
        "platform": platform_name,
        "sources": sources,
        "aggregation": {
            "source_count": len(sources),
            "counters": "sum",
            "acoustic": "sum_samples_max_latency",
            "soak": "longest_continuous_run",
        },
        "aggregate": aggregate,
        "report": report,
    }
