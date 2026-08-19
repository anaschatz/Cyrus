from __future__ import annotations

import copy
import unittest

import _bootstrap  # noqa: F401

from jarvis_voice import PROTOCOL_VERSION, __version__
from jarvis_voice.errors import BackendError
from jarvis_voice.native_qualification import (
    MAX_PTT_TO_ACOUSTIC_SILENCE_MS,
    MINIMUM_REAL_VOICE_TURNS,
    MINIMUM_SOAK_SECONDS,
    evaluate_native_evidence,
    native_evidence_template,
    validate_native_evidence,
)
from jarvis_voice.voice_auth import AUTH_PROTOCOL


def ready_evidence(platform_name: str) -> dict[str, object]:
    evidence = native_evidence_template(platform_name)
    evidence["provenance"] = {
        "native_machine": True,
        "voice_layer_mocks_used": False,
        "synthetic_voice_turns": False,
        "hardware_capture": True,
        "physical_ptt": True,
        "physical_playback": True,
    }
    for value in evidence["cases"].values():
        value.update({"status": "pass", "attempts": 1})
    evidence["counters"] = {
        "real_voice_turns": MINIMUM_REAL_VOICE_TURNS,
        "duplicate_transcripts": 0,
        "duplicate_tts_playbacks": 0,
        "stuck_ptt": 0,
        "microphone_disconnect_recoveries": 1,
        "sleep_resume_recoveries": 1,
        "crashes": 0,
    }
    evidence["acoustic_barge_in"] = {
        "method": "acoustic_microphone",
        "audio_path": "wired",
        "samples": 20,
        "max_ptt_to_silence_ms": 199.9,
        "silence_hold_ms": 50,
    }
    evidence["soak"] = {
        "duration_seconds": MINIMUM_SOAK_SECONDS,
        "rss_samples": 97,
        "rss_start_bytes": 400_000_000,
        "rss_end_bytes": 402_000_000,
        "rss_peak_bytes": 450_000_000,
        "rss_slope_bytes_per_hour": 250_000.0,
    }
    evidence["runtime_profile"] = {
        "mode": "signed_package" if platform_name == "windows" else "source",
        "fingerprint_sha256": "a" * 64,
        "stable": True,
        "signed": platform_name == "windows",
    }
    return evidence


class NativeQualificationTests(unittest.TestCase):
    def test_pending_template_is_strict_and_never_ready(self) -> None:
        evidence = native_evidence_template("macos")
        self.assertEqual(evidence["schema_version"], 2)
        self.assertEqual(evidence["product_version"], __version__)
        self.assertEqual(evidence["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(evidence["voice_auth_protocol"], AUTH_PROTOCOL)
        report = evaluate_native_evidence(evidence)
        self.assertFalse(report["ready"])
        self.assertEqual(report["checks"]["acoustic_barge_in"]["status"], "blocked")
        self.assertEqual(report["checks"]["soak"]["status"], "blocked")
        self.assertFalse(report["content_captured"])

    def test_complete_real_windows_and_mac_evidence_pass(self) -> None:
        for platform_name in ("windows", "macos"):
            with self.subTest(platform=platform_name):
                report = evaluate_native_evidence(ready_evidence(platform_name))
                self.assertTrue(report["ready"])
                self.assertEqual(report["code"], "qualification.native_ready")

    def test_legacy_or_wrong_protocol_identity_is_rejected(self) -> None:
        legacy = ready_evidence("windows")
        legacy["schema_version"] = 1
        for field in ("product_version", "protocol_version", "voice_auth_protocol"):
            legacy.pop(field)
        with self.assertRaisesRegex(BackendError, "qualification.native_schema_invalid"):
            validate_native_evidence(legacy)

        for field, value in (
            ("product_version", "0.7.0"),
            ("protocol_version", "jarvis.voice/0"),
            ("voice_auth_protocol", "jarvis.voice-auth/0"),
        ):
            evidence = ready_evidence("windows")
            evidence[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                BackendError, "qualification.native_schema_invalid"
            ):
                validate_native_evidence(evidence)

    def test_expected_runtime_profile_must_match(self) -> None:
        evidence = ready_evidence("windows")
        report = evaluate_native_evidence(
            evidence,
            expected_runtime_profile_sha256="b" * 64,
        )
        self.assertFalse(report["ready"])
        self.assertEqual(
            report["checks"]["runtime_profile"]["code"],
            "qualification.native_profile_mismatch",
        )

    def test_duplicate_stuck_and_exact_200ms_each_block(self) -> None:
        for field in ("duplicate_transcripts", "duplicate_tts_playbacks", "stuck_ptt"):
            evidence = ready_evidence("macos")
            evidence["counters"][field] = 1
            self.assertEqual(
                evaluate_native_evidence(evidence)["checks"]["turn_integrity"]["status"],
                "blocked",
            )
        evidence = ready_evidence("macos")
        evidence["acoustic_barge_in"]["max_ptt_to_silence_ms"] = MAX_PTT_TO_ACOUSTIC_SILENCE_MS
        self.assertEqual(
            evaluate_native_evidence(evidence)["checks"]["acoustic_barge_in"]["status"],
            "blocked",
        )

    def test_software_callback_and_short_or_growing_soak_block(self) -> None:
        evidence = ready_evidence("macos")
        evidence["acoustic_barge_in"]["method"] = "pending"
        self.assertFalse(evaluate_native_evidence(evidence)["ready"])

        evidence = ready_evidence("macos")
        evidence["soak"]["duration_seconds"] = MINIMUM_SOAK_SECONDS - 1
        self.assertEqual(evaluate_native_evidence(evidence)["checks"]["soak"]["status"], "blocked")

        evidence = ready_evidence("macos")
        evidence["soak"]["rss_end_bytes"] = evidence["soak"]["rss_start_bytes"] + 64 * 1024 * 1024
        evidence["soak"]["rss_peak_bytes"] = evidence["soak"]["rss_end_bytes"]
        self.assertEqual(evaluate_native_evidence(evidence)["checks"]["soak"]["status"], "blocked")

    def test_unknown_or_content_field_is_rejected(self) -> None:
        evidence = ready_evidence("windows")
        evidence["notes"] = "transcript text must not be stored"
        with self.assertRaisesRegex(BackendError, "qualification.native_schema_invalid"):
            validate_native_evidence(evidence)

        evidence = copy.deepcopy(ready_evidence("windows"))
        evidence["cases"]["raw_input"]["note"] = "physical key details"
        with self.assertRaisesRegex(BackendError, "qualification.native_cases_invalid"):
            validate_native_evidence(evidence)


if __name__ == "__main__":
    unittest.main()
