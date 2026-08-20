from __future__ import annotations

import time
import tempfile
import threading
import unittest
from array import array
from dataclasses import replace
from pathlib import Path
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
    mac_source_runtime_sha256,
)
from jarvis_voice.macos.greektts15_runtime import IsolatedSoundDevicePcmPlayback
from jarvis_voice.macos.native_run import (
    calibrate_macos_acoustic_threshold,
    create_macos_calibration_tts,
    current_process_rss_bytes,
    macos_native_collection_preflight,
    run_macos_acoustic_barge_in_sample,
    run_macos_native_collection,
)
from jarvis_voice.native_qualification import evaluate_native_evidence
from jarvis_voice.ptt import PttEvent, PttTransition
from jarvis_voice.voice_auth import create_voice_auth_key_file


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


def _authenticated_config(root: Path):
    key_path = root / "voice-auth.json"
    create_voice_auth_key_file(key_path, key=b"k" * 32)
    return voice_config_from_mapping({"voice_auth_key_file": str(key_path)})


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
            sampler.publish_audio(array("f", [0.001] * 960).tobytes(), 48_000, 1.0)
        sampler.begin("playback")
        for _ in range(10):
            sampler.publish_audio(array("f", [0.1] * 960).tobytes(), 48_000, 2.0)
        sampler.end()
        sampler.close()
        levels = sampler.levels()
        calibration = derive_acoustic_threshold(levels["ambient"], levels["playback"])
        self.assertEqual(calibration.ambient_samples, 10)
        self.assertEqual(calibration.playback_samples, 10)
        self.assertGreaterEqual(calibration.separation_db, 39.0)
        self.assertLess(calibration.silence_threshold_dbfs, calibration.playback_p80_dbfs)

    def test_calibration_uses_audible_speech_frames_not_natural_pauses(self) -> None:
        calibration = derive_acoustic_threshold(
            [-60.0] * 10,
            [-70.0, -70.0, -70.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0],
        )
        self.assertEqual(calibration.playback_p80_dbfs, -30.0)
        self.assertEqual(calibration.silence_threshold_dbfs, -57.0)
        self.assertEqual(calibration.audible_margin_db, 3.0)

    def test_calibration_worker_drains_small_queue_during_capture(self) -> None:
        sampler = AcousticCalibrationSampler(capacity=4)
        try:
            for phase, amplitude in (("ambient", 0.001), ("playback", 0.1)):
                sampler.begin(phase)
                for index in range(12):
                    sampler.publish_audio(
                        array("f", [amplitude] * 480).tobytes(),
                        48_000,
                        float(index),
                    )
                    time.sleep(0.005)
                sampler.end()
        finally:
            sampler.close()
        levels = sampler.levels()
        self.assertEqual(len(levels["ambient"]), 6)
        self.assertEqual(len(levels["playback"]), 6)

    def test_calibration_separation_failure_exposes_only_aggregate_levels(self) -> None:
        with self.assertRaises(BackendError) as raised:
            derive_acoustic_threshold([-30.0] * 10, [-26.0] * 10)
        self.assertEqual(raised.exception.code, "qualification.acoustic_calibration_separation")
        self.assertEqual(
            raised.exception.safe_context,
            {
                "ambient_p95_dbfs": -30.0,
                "playback_p20_dbfs": -26.0,
                "playback_p50_dbfs": -26.0,
                "playback_p80_dbfs": -26.0,
                "playback_p95_dbfs": -26.0,
                "separation_db": 4.0,
                "required_separation_db": 6.0,
                "ambient_samples": 10,
                "playback_samples": 10,
            },
        )


class MacSourceIdentityTests(unittest.TestCase):
    def make_source_tree(self, root: Path) -> None:
        package = root / "src" / "jarvis_voice" / "macos"
        scripts = root / "scripts"
        package.mkdir(parents=True)
        scripts.mkdir()
        (root / "src" / "jarvis_voice" / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
        (package / "runtime.py").write_text("RUNTIME = 1\n", encoding="utf-8")
        (package / "runtime 2.py").write_text("IGNORED = True\n", encoding="utf-8")
        (scripts / "run_greektts15_sidecar_worker.py").write_text("WORKER = 1\n", encoding="utf-8")
        (scripts / "run_macos_native_collection.py").write_text("COLLECTOR = 1\n", encoding="utf-8")

    def test_source_identity_changes_with_runtime_but_ignores_duplicate_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source_tree(root)
            initial = mac_source_runtime_sha256(root)

            duplicate = root / "src" / "jarvis_voice" / "macos" / "runtime 2.py"
            duplicate.write_text("IGNORED = False\n", encoding="utf-8")
            self.assertEqual(mac_source_runtime_sha256(root), initial)

            runtime = root / "src" / "jarvis_voice" / "macos" / "runtime.py"
            runtime.write_text("RUNTIME = 2\n", encoding="utf-8")
            self.assertNotEqual(mac_source_runtime_sha256(root), initial)

    def test_source_identity_fails_closed_on_missing_launch_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source_tree(root)
            (root / "scripts" / "run_macos_native_collection.py").unlink()

            with self.assertRaisesRegex(BackendError, "qualification.native_source_identity_invalid"):
                mac_source_runtime_sha256(root)


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
        self.assertEqual(collector.real_voice_turn_count, 200)
        self.assertEqual(collector.completed_playback_count, 1)

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

    def test_greektts_profile_fingerprint_uses_approval_digest_not_paths(self) -> None:
        config = voice_config_from_mapping({
            "vad": {"artifact": {"path": "/private/vad.onnx", "size": 1, "sha256": "1" * 64}},
            "macos": {
                "tts_backend": "greektts15_f1",
                "greektts15": {
                    "model_root": "/private/model",
                    "approval_manifest_path": "/private/approved.json",
                    "approval_manifest_sha256": "2" * 64,
                    "repository_root": "/private/repository",
                },
            },
        })
        fingerprint = mac_runtime_profile_fingerprint(config)
        self.assertEqual(len(fingerprint), 64)
        self.assertNotIn("/private", fingerprint)


class MacNativeRunTests(unittest.TestCase):
    def test_focused_acoustic_sample_cancels_on_physical_ptt_and_returns_latency(self) -> None:
        state: dict[str, object] = {}

        class Monitor:
            def __init__(self, **kwargs): state["threshold"] = kwargs["silence_threshold_dbfs"]
            def publish_audio(self, *args): pass
            def arm(self, received_at):
                state["armed_at"] = received_at
                state["result"] = AcousticResult(True, 123.0, "qualification.acoustic_sample_valid")
            def take_results(self):
                result = state.pop("result", None)
                return () if result is None else (result,)
            def close(self): pass

        class Capture:
            def __init__(self, **kwargs): del kwargs
            def start(self): state["capture_started"] = True
            def stop(self): state["capture_stopped"] = True

        class Ptt:
            def __init__(self, publisher, **kwargs): self.publisher = publisher; del kwargs
            def start(self): state["publisher"] = self.publisher
            def stop(self): state["ptt_stopped"] = True

        class Tts:
            def __init__(self, on_started): self.on_started = on_started; self.last_error = None
            def enqueue(self, unit):
                self.on_started(unit)
                state["publisher"].events.put_nowait(PttEvent(PttTransition.DOWN, time.perf_counter()))
            def cancel(self, epoch, reason): state.setdefault("cancels", []).append((epoch, reason))
            def wait_idle(self, timeout): del timeout; return True
            def close(self): state["tts_closed"] = True

        def build_tts(_config, *, on_started, on_completed):
            del on_completed
            return Tts(on_started)

        config = voice_config_from_mapping({"ptt_key": "CONTROL_OPTION"})
        with mock.patch("jarvis_voice.macos.native_run.sys.platform", "darwin"):
            report = run_macos_acoustic_barge_in_sample(
                config,
                silence_threshold_dbfs=-53.303,
                audio_path="built_in",
                capture_factory=Capture,
                tts_builder=build_tts,
                ptt_source_factory=Ptt,
                monitor_factory=Monitor,
            )
        self.assertTrue(report["ready"])
        self.assertEqual(report["ptt_to_acoustic_silence_ms"], 123.0)
        self.assertEqual(state["cancels"], [(2, "ptt_down"), (3, "qualification_end")])
        self.assertEqual(state["threshold"], -53.303)
        self.assertTrue(state["capture_stopped"])
        self.assertTrue(state["ptt_stopped"])
        self.assertTrue(state["tts_closed"])

    def test_focused_acoustic_sample_exercises_pocket_primary_in_english(self) -> None:
        state: dict[str, object] = {}

        class Monitor:
            def __init__(self, **kwargs): del kwargs
            def publish_audio(self, *args): pass
            def arm(self, received_at):
                del received_at
                state["result"] = AcousticResult(True, 75.0, "qualification.acoustic_sample_valid")
            def take_results(self):
                result = state.pop("result", None)
                return () if result is None else (result,)
            def close(self): pass

        class Capture:
            def __init__(self, **kwargs): del kwargs
            def start(self): pass
            def stop(self): pass

        class Ptt:
            def __init__(self, publisher, **kwargs): state["publisher"] = publisher; del kwargs
            def start(self): pass
            def stop(self): pass

        class Tts:
            last_error = None
            def __init__(self, on_started): self.on_started = on_started
            def enqueue(self, unit):
                state["unit"] = unit
                self.on_started(unit)
                state["publisher"].events.put_nowait(PttEvent(PttTransition.DOWN, time.perf_counter()))
            def cancel(self, epoch, reason): del epoch, reason
            def wait_idle(self, timeout): del timeout; return True
            def close(self): pass

        config = voice_config_from_mapping({"ptt_key": "CONTROL_OPTION"})
        config = replace(config, macos=replace(config.macos, tts_backend="pocket_tts"))
        with mock.patch("jarvis_voice.macos.native_run.sys.platform", "darwin"):
            report = run_macos_acoustic_barge_in_sample(
                config,
                silence_threshold_dbfs=-53.303,
                audio_path="built_in",
                capture_factory=Capture,
                tts_builder=lambda _config, *, on_started, on_completed: Tts(on_started),
                ptt_source_factory=Ptt,
                monitor_factory=Monitor,
            )

        self.assertTrue(report["ready"])
        unit = state["unit"]
        self.assertEqual(unit.locale, "en-US")
        self.assertIn("push to talk", unit.text)

    def test_focused_acoustic_batch_reuses_runtime_and_advances_epochs(self) -> None:
        state: dict[str, object] = {"arms": 0}

        class Monitor:
            def __init__(self, **kwargs): del kwargs
            def publish_audio(self, *args): pass
            def arm(self, received_at):
                del received_at
                state["arms"] = int(state["arms"]) + 1
                state["result"] = AcousticResult(
                    True,
                    100.0 + int(state["arms"]),
                    "qualification.acoustic_sample_valid",
                )
            def take_results(self):
                result = state.pop("result", None)
                return () if result is None else (result,)
            def close(self): pass

        class Capture:
            def __init__(self, **kwargs): del kwargs
            def start(self): pass
            def stop(self): pass

        class Ptt:
            def __init__(self, publisher, **kwargs): state["publisher"] = publisher; del kwargs
            def start(self): pass
            def stop(self): pass

        class Tts:
            last_error = None
            def __init__(self, on_started): self.on_started = on_started
            def enqueue(self, unit):
                self.on_started(unit)
                state["publisher"].events.put_nowait(PttEvent(PttTransition.DOWN, time.perf_counter()))
            def cancel(self, epoch, reason): state.setdefault("cancels", []).append((epoch, reason))
            def wait_idle(self, timeout): del timeout; return True
            def close(self): pass

        def build_tts(_config, *, on_started, on_completed):
            del on_completed
            return Tts(on_started)

        progress: list[dict[str, object]] = []
        config = voice_config_from_mapping({"ptt_key": "CONTROL_OPTION"})
        with mock.patch("jarvis_voice.macos.native_run.sys.platform", "darwin"):
            report = run_macos_acoustic_barge_in_sample(
                config,
                silence_threshold_dbfs=-53.303,
                audio_path="built_in",
                sample_count=2,
                progress_sink=progress.append,
                capture_factory=Capture,
                tts_builder=build_tts,
                ptt_source_factory=Ptt,
                monitor_factory=Monitor,
            )
        self.assertTrue(report["ready"])
        self.assertEqual(report["valid_samples"], 2)
        self.assertEqual(report["maximum_ptt_to_acoustic_silence_ms"], 102.0)
        self.assertEqual([item["sample_index"] for item in progress], [1, 2])
        self.assertEqual(state["cancels"], [(2, "ptt_down"), (4, "ptt_down"), (5, "qualification_end")])

    def test_focused_acoustic_batch_stops_when_latency_reaches_threshold(self) -> None:
        state: dict[str, object] = {}

        class Monitor:
            def __init__(self, **kwargs): del kwargs
            def publish_audio(self, *args): pass
            def arm(self, received_at): del received_at
            def take_results(self): return (AcousticResult(True, 200.0, "qualification.acoustic_sample_valid"),)
            def close(self): pass

        class Capture:
            def __init__(self, **kwargs): del kwargs
            def start(self): pass
            def stop(self): pass

        class Ptt:
            def __init__(self, publisher, **kwargs): state["publisher"] = publisher; del kwargs
            def start(self): pass
            def stop(self): pass

        class Tts:
            last_error = None
            def __init__(self, on_started): self.on_started = on_started
            def enqueue(self, unit):
                self.on_started(unit)
                state["publisher"].events.put_nowait(PttEvent(PttTransition.DOWN, time.perf_counter()))
            def cancel(self, epoch, reason): del epoch, reason
            def wait_idle(self, timeout): del timeout; return True
            def close(self): pass

        config = voice_config_from_mapping({"ptt_key": "CONTROL_OPTION"})
        with mock.patch("jarvis_voice.macos.native_run.sys.platform", "darwin"):
            report = run_macos_acoustic_barge_in_sample(
                config,
                silence_threshold_dbfs=-53.303,
                audio_path="built_in",
                sample_count=2,
                capture_factory=Capture,
                tts_builder=lambda _config, *, on_started, on_completed: Tts(on_started),
                ptt_source_factory=Ptt,
                monitor_factory=Monitor,
            )
        self.assertFalse(report["ready"])
        self.assertEqual(report["valid_samples"], 0)
        self.assertEqual(len(report["samples"]), 1)
        self.assertEqual(report["samples"][0]["code"], "qualification.acoustic_sample_threshold_exceeded")

    def test_calibration_uses_a_short_bounded_phrase(self) -> None:
        observer = None
        spoken: list[str] = []

        class Capture:
            def __init__(self, *, frame_observer, **kwargs):
                nonlocal observer
                del kwargs
                observer = frame_observer
            def start(self): pass
            def stop(self): pass

        class Tts:
            def __init__(self, *, on_started, on_completed, **kwargs):
                del kwargs
                self.on_started = on_started
                self.on_completed = on_completed
                self.last_error = None
            def enqueue(self, unit):
                spoken.append(unit.text)
                self.on_started(unit)
                assert observer is not None
                for index in range(10):
                    observer(array("f", [0.1] * 960).tobytes(), 48_000, 2.0 + index * 0.02)
                self.on_completed(unit)
            def close(self): pass

        def capture_ambient(_seconds: float) -> None:
            assert observer is not None
            for index in range(10):
                observer(array("f", [0.001] * 960).tobytes(), 48_000, 1.0 + index * 0.02)

        config = voice_config_from_mapping({"macos": {"system_voices": {"el-GR": "A", "en-US": "B"}}})
        with mock.patch("jarvis_voice.macos.native_run.sys.platform", "darwin"):
            report = calibrate_macos_acoustic_threshold(
                config,
                audio_path="built_in",
                capture_factory=Capture,
                tts_factory=Tts,
                sleeper=capture_ambient,
            )
        self.assertTrue(report["ready"])
        self.assertEqual(spoken, ["Η φωνή λειτουργεί καθαρά."])

    def test_calibration_uses_the_digest_bound_selected_f1_backend(self) -> None:
        config = voice_config_from_mapping({
            "vad": {"artifact": {"path": "/private/vad.onnx", "size": 1, "sha256": "1" * 64}},
            "macos": {
                "tts_backend": "greektts15_f1",
                "greektts15": {
                    "model_root": "/private/model",
                    "approval_manifest_path": "/private/approved.json",
                    "approval_manifest_sha256": "2" * 64,
                    "repository_root": "/private/repository",
                    "warm_on_start": True,
                },
            },
        })
        candidate = object()
        engine = object()
        backend = object()
        started = lambda _unit: None
        completed = lambda _unit: None
        with (
            mock.patch("jarvis_voice.macos.native_run.load_digest_bound_greektts_candidate", return_value=candidate) as load,
            mock.patch("jarvis_voice.macos.native_run.MlxGreekTtsStreamingEngine", return_value=engine) as make_engine,
            mock.patch("jarvis_voice.macos.native_run.GreekTtsF1Backend", return_value=backend) as make_backend,
        ):
            selected = create_macos_calibration_tts(config, on_started=started, on_completed=completed)
        self.assertIs(selected, backend)
        load.assert_called_once_with("/private/approved.json", "2" * 64, "/private/repository")
        make_engine.assert_called_once_with(model_root="/private/model", candidate=candidate)
        self.assertIs(make_backend.call_args.kwargs["engine"], engine)
        self.assertIs(make_backend.call_args.kwargs["playback_factory"], IsolatedSoundDevicePcmPlayback)
        self.assertIs(make_backend.call_args.kwargs["on_started"], started)
        self.assertIs(make_backend.call_args.kwargs["on_completed"], completed)
        self.assertTrue(make_backend.call_args.kwargs["warm_on_start"])

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

    def test_native_collection_stops_only_after_bounded_turn_playback_completes(self) -> None:
        class Runtime:
            def __init__(self, _config, **kwargs):
                del kwargs
                self.stopped = threading.Event()
                self.timings = (
                    _event("audio_stream_started", 1.0),
                    _event("ptt_source_started", 1.1),
                    _event("ptt_down_received", 1.2, output_active=False, ptt_cycle=1),
                    _event("ptt_up_received", 1.3, ptt_cycle=1, forced=False, reason="event_tap"),
                    _event("transcript_final_created", 1.4, ptt_cycle=1, language="el"),
                    _event("tts_unit_started", 1.5, response_token=1, unit_index=1, locale="el-GR"),
                    _event("playback_completed_created", 1.6, response_token=1),
                )

            def run(self):
                self.stopped.wait(timeout=2.0)

            def request_stop(self):
                self.stopped.set()

            def timing_snapshot(self):
                return self.timings

        class Server:
            def __init__(self, _url, _auth_key): pass
            def start(self): pass
            def stop(self): pass

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "jarvis_voice.macos.native_run.sys.platform",
            "darwin",
        ):
            root = Path(directory)
            config = _authenticated_config(root)
            report = run_macos_native_collection(
                config,
                output=root / "evidence.json",
                silence_threshold_dbfs=-40.0,
                audio_path="built_in",
                duration_seconds=2.0,
                rss_interval_seconds=1.0,
                maximum_real_voice_turns=1,
                threshold_calibrated=True,
                runtime_factory=Runtime,
                server_factory=Server,
                rss_probe=lambda: 100_000_000,
            )
        self.assertTrue(report["turn_batch_completed"])
        self.assertEqual(report["collection_stop_reason"], "turn_batch_complete")
        self.assertEqual(report["maximum_real_voice_turns"], 1)
        self.assertEqual(report["runtime_error_codes"], [])
        self.assertFalse(report["ready"])

    def test_native_collection_rejects_oversized_turn_batch(self) -> None:
        config = voice_config_from_mapping({})
        with mock.patch("jarvis_voice.macos.native_run.sys.platform", "darwin"), self.assertRaisesRegex(
            BackendError,
            "qualification.native_turn_batch_invalid",
        ):
            run_macos_native_collection(
                config,
                output="/tmp/unused-native-evidence.json",
                silence_threshold_dbfs=-40.0,
                audio_path="built_in",
                duration_seconds=1.0,
                maximum_real_voice_turns=6,
                threshold_calibrated=True,
            )

    def test_native_collection_reports_content_free_runtime_error_code(self) -> None:
        class Runtime:
            def __init__(self, _config, **kwargs): del kwargs
            def run(self): raise BackendError("runtime.test_failure", "private details", subsystem="runtime")
            def request_stop(self): pass
            def timing_snapshot(self): return ()

        class Server:
            def __init__(self, _url, _auth_key): pass
            def start(self): pass
            def stop(self): pass

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "jarvis_voice.macos.native_run.sys.platform",
            "darwin",
        ):
            root = Path(directory)
            report = run_macos_native_collection(
                _authenticated_config(root),
                output=root / "evidence.json",
                silence_threshold_dbfs=-40.0,
                audio_path="built_in",
                duration_seconds=1.0,
                rss_interval_seconds=1.0,
                threshold_calibrated=True,
                runtime_factory=Runtime,
                server_factory=Server,
                rss_probe=lambda: 100_000_000,
            )
        self.assertEqual(report["collection_stop_reason"], "runtime_failed")
        self.assertEqual(report["runtime_error_codes"], ["runtime.test_failure"])
        self.assertNotIn("private details", str(report))


if __name__ == "__main__":
    unittest.main()
