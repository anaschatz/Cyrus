from __future__ import annotations

import time
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.native_collector import (
    AcousticCalibrationSampler,
    AcousticResult,
    AcousticSilenceMonitor,
    MacNativeTelemetryCollector,
    derive_acoustic_threshold,
    mac_runtime_profile_fingerprint,
)
from jarvis_voice.macos.native_run import (
    current_process_rss_bytes,
    macos_native_collection_preflight,
    run_macos_native_collection,
)
from jarvis_voice.native_qualification import evaluate_native_evidence


class _FakeAcousticMonitor:
    def __init__(self, results: list[AcousticResult]) -> None:
        self.results = results
        self.armed: list[float] = []

    def arm(self, timestamp: float) -> None:
        self.armed.append(timestamp)

    def take_results(self) -> tuple[AcousticResult, ...]:
        values = tuple(self.results)
        self.results.clear()
        return values


def _event(name: str, timestamp: float, **details: object) -> dict[str, object]:
    return {"event": name, "timestamp": timestamp, **details}


class AcousticSilenceMonitorTests(unittest.TestCase):
    def test_worker_requires_audible_preroll_and_reports_sustained_silence(self) -> None:
        monitor = AcousticSilenceMonitor(silence_threshold_dbfs=-40.0)
        sample_rate = 48_000
        loud = array("f", [0.1] * 960).tobytes()
        silence = array("f", [0.0] * 960).tobytes()
        try:
            monitor.publish_audio(loud, sample_rate, 0.98)
            monitor.arm(1.0)
            monitor.publish_audio(silence, sample_rate, 1.02)
            monitor.publish_audio(silence, sample_rate, 1.04)
            monitor.publish_audio(silence, sample_rate, 1.06)
            deadline = time.monotonic() + 1.0
            results: tuple[AcousticResult, ...] = ()
            while time.monotonic() < deadline and not results:
                results = monitor.take_results()
                time.sleep(0.005)
        finally:
            monitor.close()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].valid)
        self.assertAlmostEqual(results[0].latency_ms or -1, 0.0, delta=1.0)

    def test_silence_without_audible_preroll_is_rejected(self) -> None:
        monitor = AcousticSilenceMonitor(silence_threshold_dbfs=-40.0)
        silence = array("f", [0.0] * 4_800).tobytes()
        try:
            monitor.arm(1.0)
            monitor.publish_audio(silence, 48_000, 1.1)
            monitor.publish_audio(silence, 48_000, 1.3)
            deadline = time.monotonic() + 1.0
            results: tuple[AcousticResult, ...] = ()
            while time.monotonic() < deadline and not results:
                results = monitor.take_results()
                time.sleep(0.005)
        finally:
            monitor.close()
        self.assertFalse(results[0].valid)
        self.assertEqual(results[0].code, "qualification.acoustic_preroll_not_audible")

    def test_calibration_keeps_only_levels_and_derives_separated_threshold(self) -> None:
        sampler = AcousticCalibrationSampler()
        sampler.begin("ambient")
        for _ in range(10):
            sampler.publish_audio(array("f", [0.001] * 480).tobytes(), 48_000, 1.0)
        sampler.begin("playback")
        for _ in range(10):
            sampler.publish_audio(array("f", [0.1] * 480).tobytes(), 48_000, 2.0)
        sampler.end()
        levels = sampler.levels()
        calibration = derive_acoustic_threshold(levels["ambient"], levels["playback"])
        self.assertEqual(calibration.ambient_samples, 10)
        self.assertEqual(calibration.playback_samples, 10)
        self.assertGreaterEqual(calibration.separation_db, 39.0)
        self.assertLess(calibration.silence_threshold_dbfs, calibration.playback_p20_dbfs)


class MacNativeTelemetryCollectorTests(unittest.TestCase):
    def test_real_event_aggregation_can_satisfy_strict_gate(self) -> None:
        acoustic = _FakeAcousticMonitor(
            [AcousticResult(True, 120.0 + index, "qualification.acoustic_sample_valid") for index in range(20)]
        )
        collector = MacNativeTelemetryCollector(
            profile_fingerprint="a" * 64,
            audio_path="wired",
            acoustic_monitor=acoustic,  # type: ignore[arg-type]
            platform_name="darwin",
            clock=lambda: 0.0,
        )
        collector.feed(_event("audio_stream_started", 0.1))
        collector.feed(_event("ptt_source_started", 0.2))
        collector.feed(_event("ptt_down_received", 0.3, output_active=True, ptt_cycle=1))
        collector.feed(_event("ptt_up_received", 0.4, ptt_cycle=1, forced=False, reason="raw_input"))
        for cycle in range(1, 201):
            collector.feed(_event("transcript_final_created", float(cycle), ptt_cycle=cycle, language="el"))
        collector.feed(_event("tts_unit_started", 201.0, response_token=1, unit_index=1, locale="el-GR"))
        collector.feed(_event("playback_completed_created", 202.0, response_token=1))
        collector.record_microphone_recovery(passed=True)
        collector.record_sleep_recovery(passed=True)
        for index in range(97):
            collector.sample_rss(400_000_000 + index * 10_000, timestamp=index * 300.0)
        evidence = collector.evidence(now=28_800.0, final=True)
        report = evaluate_native_evidence(evidence)
        self.assertTrue(report["ready"])
        self.assertEqual(acoustic.armed, [0.3])
        self.assertEqual(evidence["counters"]["real_voice_turns"], 200)
        self.assertEqual(evidence["acoustic_barge_in"]["samples"], 20)

    def test_runtime_recovery_events_update_native_gate_without_manual_marking(self) -> None:
        collector = MacNativeTelemetryCollector(
            profile_fingerprint="c" * 64,
            audio_path="built_in",
            platform_name="darwin",
            clock=lambda: 0.0,
        )
        collector.feed(_event("microphone_disconnect_recovered", 1.0))
        collector.feed(_event("sleep_resume_recovered", 2.0))
        evidence = collector.evidence(now=3.0)
        self.assertEqual(evidence["counters"]["microphone_disconnect_recoveries"], 1)
        self.assertEqual(evidence["counters"]["sleep_resume_recoveries"], 1)
        self.assertEqual(evidence["cases"]["microphone_disconnect_recovery"]["status"], "pass")
        self.assertEqual(evidence["cases"]["sleep_resume_recovery"]["status"], "pass")

    def test_duplicate_and_stuck_state_are_counted_without_ids_in_evidence(self) -> None:
        collector = MacNativeTelemetryCollector(
            profile_fingerprint="b" * 64,
            audio_path="built_in",
            platform_name="darwin",
            clock=lambda: 0.0,
        )
        collector.feed(_event("ptt_down_received", 1.0, output_active=False, ptt_cycle=1))
        collector.feed(_event("transcript_final_created", 2.0, ptt_cycle=1, language="el"))
        collector.feed(_event("transcript_final_created", 2.1, ptt_cycle=1, language="el"))
        collector.feed(_event("tts_unit_started", 3.0, response_token=1, unit_index=1, locale="el-GR"))
        collector.feed(_event("tts_unit_started", 3.1, response_token=1, unit_index=1, locale="el-GR"))
        evidence = collector.evidence(now=4.0, final=True)
        self.assertEqual(evidence["counters"]["duplicate_transcripts"], 1)
        self.assertEqual(evidence["counters"]["duplicate_tts_playbacks"], 1)
        self.assertEqual(evidence["counters"]["stuck_ptt"], 1)
        encoded = str(evidence)
        self.assertNotIn("response_id", encoded)
        self.assertNotIn("turn_id", encoded)

    def test_profile_fingerprint_does_not_expose_local_paths_or_voice_names(self) -> None:
        config = voice_config_from_mapping(
            {
                "vad": {"artifact": {"path": "/private/vad.onnx", "size": 1, "sha256": "1" * 64}},
                "macos": {"system_voices": {"el-GR": "Private Voice", "en-US": "Another Voice"}},
            }
        )
        fingerprint = mac_runtime_profile_fingerprint(config)
        self.assertEqual(len(fingerprint), 64)
        self.assertNotIn("Private", fingerprint)
        self.assertNotIn("/private", fingerprint)


class MacNativeRunTests(unittest.TestCase):
    def test_preflight_is_passive_and_requires_explicit_calibration(self) -> None:
        config = voice_config_from_mapping(
            {
                "vad": {"artifact": {"path": "/private/vad.onnx", "size": 1, "sha256": "1" * 64}},
                "macos": {"system_voices": {"el-GR": "Voice A", "en-US": "Voice B"}},
            }
        )
        blocked = macos_native_collection_preflight(
            config,
            silence_threshold_dbfs=-40.0,
            audio_path="wired",
            doctor_probe=lambda _: {"ready": True, "network_used": False},
        )
        ready = macos_native_collection_preflight(
            config,
            silence_threshold_dbfs=-40.0,
            audio_path="wired",
            threshold_calibrated=True,
            doctor_probe=lambda _: {"ready": True, "network_used": False},
        )
        self.assertFalse(blocked["ready_to_start"])
        self.assertTrue(ready["ready_to_start"])
        self.assertFalse(ready["microphone_opened"])
        self.assertFalse(ready["ptt_started"])

    def test_current_rss_parses_ps_kib_and_run_refuses_uncalibrated_threshold(self) -> None:
        rss = current_process_rss_bytes(
            pid=123,
            runner=lambda *args, **kwargs: SimpleNamespace(stdout=" 4096\n", returncode=0),
        )
        self.assertEqual(rss, 4096 * 1024)
        config = voice_config_from_mapping({})
        with mock.patch("jarvis_voice.macos.native_run.sys.platform", "darwin"), self.assertRaisesRegex(
            BackendError,
            "qualification.acoustic_calibration_required",
        ):
            run_macos_native_collection(
                config,
                output="/tmp/unused-native-evidence.json",
                silence_threshold_dbfs=-40.0,
                audio_path="wired",
                duration_seconds=1.0,
            )


if __name__ == "__main__":
    unittest.main()
