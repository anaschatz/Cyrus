from __future__ import annotations

import copy
import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.native_evidence_bundle import aggregate_native_evidence
from jarvis_voice.native_qualification import native_evidence_template


def partial_run(*, turns: int, samples: int, latency_ms: float, duration_seconds: int) -> dict[str, object]:
    evidence = native_evidence_template("macos", runtime_profile_sha256="a" * 64)
    evidence["provenance"] = {
        "native_machine": True,
        "voice_layer_mocks_used": False,
        "synthetic_voice_turns": False,
        "hardware_capture": True,
        "physical_ptt": True,
        "physical_playback": True,
    }
    for name in (
        "coreaudio_capture",
        "accessibility_permission",
        "ptt_event_tap",
        "selected_tts_backend",
        "stable_runtime_profile",
    ):
        evidence["cases"][name] = {"status": "pass", "attempts": 1}
    evidence["counters"]["real_voice_turns"] = turns
    evidence["acoustic_barge_in"] = {
        "method": "acoustic_microphone",
        "audio_path": "built_in",
        "samples": samples,
        "max_ptt_to_silence_ms": latency_ms if samples else None,
        "silence_hold_ms": 50,
    }
    evidence["soak"] = {
        "duration_seconds": duration_seconds,
        "rss_samples": 2,
        "rss_start_bytes": 100,
        "rss_end_bytes": 100,
        "rss_peak_bytes": 100,
        "rss_slope_bytes_per_hour": 0.0,
    }
    evidence["runtime_profile"] = {
        "mode": "source",
        "fingerprint_sha256": "a" * 64,
        "stable": True,
        "signed": False,
    }
    return evidence


class NativeEvidenceBundleTests(unittest.TestCase):
    def test_sums_turns_and_acoustics_but_keeps_longest_continuous_soak(self) -> None:
        first = partial_run(turns=2, samples=1, latency_ms=22.684, duration_seconds=90)
        second = partial_run(turns=3, samples=1, latency_ms=50.394, duration_seconds=120)

        bundle = aggregate_native_evidence([second, first])

        self.assertEqual(bundle["aggregation"]["source_count"], 2)
        self.assertEqual(bundle["aggregate"]["counters"]["real_voice_turns"], 5)
        self.assertEqual(bundle["aggregate"]["acoustic_barge_in"]["samples"], 2)
        self.assertEqual(bundle["aggregate"]["acoustic_barge_in"]["max_ptt_to_silence_ms"], 50.394)
        self.assertEqual(bundle["aggregate"]["soak"]["duration_seconds"], 120)
        self.assertEqual(bundle["aggregation"]["soak"], "longest_continuous_run")
        self.assertFalse(bundle["report"]["ready"])
        self.assertFalse(bundle["report"]["content_captured"])

    def test_twenty_bounded_samples_satisfy_only_the_acoustic_gate(self) -> None:
        runs = [partial_run(turns=2, samples=1, latency_ms=25.0 + index, duration_seconds=30) for index in range(20)]

        bundle = aggregate_native_evidence(runs)

        self.assertEqual(bundle["aggregate"]["cases"]["acoustic_barge_in"], {"status": "pass", "attempts": 20})
        self.assertEqual(bundle["report"]["checks"]["acoustic_barge_in"]["status"], "ready")
        self.assertEqual(bundle["report"]["checks"]["soak"]["status"], "blocked")

    def test_failed_acoustic_case_dominates_successful_samples(self) -> None:
        runs = [partial_run(turns=2, samples=1, latency_ms=25.0, duration_seconds=30) for _ in range(20)]
        runs[0]["cases"]["acoustic_barge_in"] = {"status": "fail", "attempts": 1}

        bundle = aggregate_native_evidence(runs)

        self.assertEqual(bundle["aggregate"]["cases"]["acoustic_barge_in"]["status"], "fail")
        self.assertEqual(bundle["report"]["checks"]["runtime_cases"]["status"], "blocked")

    def test_rejects_duplicate_profile_mismatch_and_content_capture(self) -> None:
        first = partial_run(turns=1, samples=1, latency_ms=20.0, duration_seconds=10)
        with self.assertRaisesRegex(BackendError, "qualification.native_bundle_duplicate_run"):
            aggregate_native_evidence([first, first])

        second = partial_run(turns=1, samples=1, latency_ms=21.0, duration_seconds=10)
        second["runtime_profile"]["fingerprint_sha256"] = "b" * 64
        with self.assertRaisesRegex(BackendError, "qualification.native_bundle_profile_mismatch"):
            aggregate_native_evidence([first, second])

        captured = copy.deepcopy(first)
        captured["run_id"] = partial_run(turns=0, samples=0, latency_ms=0.0, duration_seconds=1)["run_id"]
        captured["privacy"]["content_captured"] = True
        with self.assertRaisesRegex(BackendError, "qualification.native_bundle_privacy_invalid"):
            aggregate_native_evidence([first, captured])

    def test_rejects_mixed_acoustic_profiles(self) -> None:
        first = partial_run(turns=1, samples=1, latency_ms=20.0, duration_seconds=10)
        second = partial_run(turns=1, samples=1, latency_ms=21.0, duration_seconds=10)
        second["acoustic_barge_in"]["audio_path"] = "wired"

        with self.assertRaisesRegex(BackendError, "qualification.native_bundle_acoustic_profile_mismatch"):
            aggregate_native_evidence([first, second])


if __name__ == "__main__":
    unittest.main()
