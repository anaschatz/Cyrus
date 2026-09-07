from __future__ import annotations

import time
import contextlib
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
from jarvis_voice.macos import native_run
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

    def test_tiny_coreaudio_callbacks_use_same_20ms_rms_as_calibration(self) -> None:
        monitor = AcousticSilenceMonitor(silence_threshold_dbfs=-65.16)
        sample_rate = 48_000
        loud = array("f", [0.1] * 960).tobytes()
        quiet = array("f", [0.0002] * 64).tobytes()
        transient = array("f", [0.0008] * 64).tobytes()
        try:
            monitor.publish_audio(loud, sample_rate, 0.98)
            monitor.arm(1.0)
            timestamp = 1.0
            for callback_index in range(45):
                timestamp += 64 / sample_rate
                monitor.publish_audio(
                    transient if callback_index % 15 == 0 else quiet,
                    sample_rate,
                    timestamp,
                )
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

    def test_soak_duration_never_rounds_up_uncompleted_seconds(self) -> None:
        collector = MacNativeTelemetryCollector(
            profile_fingerprint="a" * 64, audio_path="built_in", clock=lambda: 0.0,
        )
        for elapsed, seconds in ((0.9, 0), (28_799.9, 28_799), (28_800.0, 28_800)):
            with self.subTest(elapsed=elapsed):
                self.assertEqual(collector.evidence(now=elapsed)["soak"]["duration_seconds"], seconds)

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

    def test_selected_backend_artifacts_and_settings_change_profile_identity(self) -> None:
        artifact = {"path": "/private/model", "size": 1, "sha256": "1" * 64}
        kokoro = voice_config_from_mapping({
            "macos": {"tts_backend": "kokoro_onnx", "kokoro_onnx": {
                "model": artifact, "voices": {**artifact, "path": "/private/voices"},
            }},
        })
        cpu = voice_config_from_mapping({"macos": {"asr_backend": "faster_whisper_cpu_smoke"}})
        greek = voice_config_from_mapping({"macos": {"tts_backend": "greektts15_f1", "greektts15": {
            "model_root": "/private/model", "approval_manifest_path": "/private/approval.json",
            "approval_manifest_sha256": "2" * 64, "repository_root": "/private/repository",
        }}})
        with mock.patch("jarvis_voice.macos.native_collector.mac_source_runtime_sha256", return_value="a" * 64):
            for field in ("model", "voices", "warm_on_start"):
                with self.subTest(backend="kokoro", field=field):
                    old = kokoro.macos.kokoro_onnx
                    value = False if field == "warm_on_start" else replace(getattr(old, field), sha256="3" * 64)
                    changed = replace(kokoro, macos=replace(kokoro.macos, kokoro_onnx=replace(old, **{field: value})))
                    self.assertNotEqual(mac_runtime_profile_fingerprint(kokoro), mac_runtime_profile_fingerprint(changed))
            for field, value in (("model_profile", "large-v3-turbo"), ("cpu_compute_type", "float32")):
                with self.subTest(backend="cpu", field=field):
                    changed = replace(cpu, asr=replace(cpu.asr, **{field: value}))
                    self.assertNotEqual(mac_runtime_profile_fingerprint(cpu), mac_runtime_profile_fingerprint(changed))
            changed = replace(greek, macos=replace(greek.macos, greektts15=replace(greek.macos.greektts15, warm_on_start=False)))
            with self.subTest(backend="greek", field="warm_on_start"):
                self.assertNotEqual(mac_runtime_profile_fingerprint(greek), mac_runtime_profile_fingerprint(changed))
            relocated = replace(kokoro, macos=replace(kokoro.macos, kokoro_onnx=replace(
                kokoro.macos.kokoro_onnx,
                model=replace(kokoro.macos.kokoro_onnx.model, path="/elsewhere/model"),
                voices=replace(kokoro.macos.kokoro_onnx.voices, path="/elsewhere/voices"),
            )))
            self.assertEqual(mac_runtime_profile_fingerprint(kokoro), mac_runtime_profile_fingerprint(relocated))

    def test_capture_vad_and_selected_whisper_settings_are_profile_bound(self) -> None:
        config = voice_config_from_mapping({})
        with mock.patch("jarvis_voice.macos.native_collector.mac_source_runtime_sha256", return_value="a" * 64):
            original = mac_runtime_profile_fingerprint(config)
            for field, value in (
                ("ptt_queue_capacity", 32), ("audio_queue_capacity", 128),
                ("maximum_hold_seconds", 15.0), ("maximum_utterance_seconds", 15.0),
            ):
                with self.subTest(section="capture", field=field):
                    self.assertNotEqual(original, mac_runtime_profile_fingerprint(replace(config, **{field: value})))
            for field, value in (("threshold", 0.8), ("minimum_speech_ms", 120), ("padding_ms", 100)):
                with self.subTest(section="vad", field=field):
                    changed = replace(config, vad=replace(config.vad, **{field: value}))
                    self.assertNotEqual(original, mac_runtime_profile_fingerprint(changed))
            for field, value in (
                ("language", "el"), ("threads", 8),
                ("qualified_audio_context_minimum_seconds", 4.0),
                ("qualified_audio_context_maximum_seconds", 10.0),
                ("startup_timeout_seconds", 40.0), ("request_timeout_seconds", 90.0),
                ("maximum_response_bytes", 524_288),
            ):
                with self.subTest(section="whisper", field=field):
                    changed = replace(config, macos=replace(config.macos, whisper_cpp=replace(config.macos.whisper_cpp, **{field: value})))
                    self.assertNotEqual(original, mac_runtime_profile_fingerprint(changed))

    def test_recovery_failure_remains_visible_after_later_success(self) -> None:
        for kind, recovered, cases, counter in (
            ("microphone_disconnect", "microphone_disconnect_recovered", ("microphone_disconnect_recovery", "microphone_change"), "microphone_disconnect_recoveries"),
            ("sleep_resume", "sleep_resume_recovered", ("sleep_resume_recovery", "suspend_resume"), "sleep_resume_recoveries"),
        ):
            with self.subTest(kind=kind):
                collector = MacNativeTelemetryCollector(
                    profile_fingerprint="a" * 64, audio_path="built_in",
                    platform_name="darwin", clock=lambda: 0.0,
                )
                collector.feed(_event("native_io_recovery_failed", 1.0, recovery_kind=kind))
                collector.feed(_event(recovered, 2.0))
                evidence = collector.evidence(now=3.0)
                for case in cases:
                    self.assertEqual(evidence["cases"][case], {"status": "fail", "attempts": 2})
                self.assertEqual(evidence["counters"][counter], 1)
                self.assertEqual(evaluate_native_evidence(evidence)["checks"]["automatic_recovery"]["status"], "blocked")

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
    def test_calibration_factory_transfers_or_closes_fallback_ownership(self) -> None:
        base = voice_config_from_mapping({})
        for backend, engine_name, backend_name in (
            ("kokoro_onnx", "KokoroOnnxCpuEngine", "KokoroOnnxTtsBackend"),
            ("pocket_tts", "PocketTtsCpuEngine", "PocketTtsBackend"),
        ):
            for fails in (True, False):
                with self.subTest(backend=backend, fails=fails):
                    fallback, primary = mock.Mock(), mock.Mock()
                    failure = RuntimeError("private primary startup failure")
                    with (
                        mock.patch.object(native_run, "MacSystemTtsBackend", return_value=fallback),
                        mock.patch.object(native_run, engine_name, return_value=object()),
                        mock.patch.object(native_run, backend_name, return_value=primary,
                                          side_effect=failure if fails else None) as factory,
                    ):
                        config = replace(base, macos=replace(base.macos, tts_backend=backend))
                        if fails:
                            with self.assertRaises(RuntimeError) as raised:
                                create_macos_calibration_tts(config, on_started=lambda _: None, on_completed=lambda _: None)
                            self.assertIs(raised.exception, failure)
                            fallback.close.assert_called_once_with()
                        else:
                            selected = create_macos_calibration_tts(config, on_started=lambda _: None, on_completed=lambda _: None)
                            self.assertIs(selected, primary)
                            self.assertIs(factory.call_args.kwargs["fallback_backend"], fallback)
                            fallback.close.assert_not_called()

    def test_calibration_releases_every_acquired_resource_after_failure(self) -> None:
        for stage in ("capture_factory", "tts_factory", "capture_start", "capture_stop", "tts_close", "sampler_end"):
            with self.subTest(stage=stage):
                sampler = mock.Mock()
                capture = mock.Mock()
                tts = mock.Mock(last_error=None)
                failure = RuntimeError("private calibration failure")
                capture_factory = mock.Mock(return_value=capture)
                tts_factory = mock.Mock(return_value=tts)
                if stage == "capture_factory":
                    capture_factory.side_effect = failure
                elif stage == "tts_factory":
                    tts_factory.side_effect = failure
                else:
                    operation = {"capture_start": capture.start, "capture_stop": capture.stop,
                                 "tts_close": tts.close, "sampler_end": sampler.end}[stage]
                    operation.side_effect = failure
                def enqueue(unit):
                    callbacks = tts_factory.call_args.kwargs
                    callbacks["on_started"](unit)
                    callbacks["on_completed"](unit)
                tts.enqueue.side_effect = enqueue
                with (
                    mock.patch.object(native_run, "sys", SimpleNamespace(platform="darwin")),
                    mock.patch.object(native_run, "AcousticCalibrationSampler", return_value=sampler),
                ):
                    with self.assertRaises(RuntimeError) as raised:
                        calibrate_macos_acoustic_threshold(
                            voice_config_from_mapping({}), audio_path="built_in",
                            capture_factory=capture_factory, tts_factory=tts_factory,
                            sleeper=lambda _: None,
                        )
                self.assertIs(raised.exception, failure)
                for name, operation, acquired in (
                    ("sampler", sampler.close, True),
                    ("capture", capture.stop, stage != "capture_factory"),
                    ("tts", tts.close, stage not in ("capture_factory", "tts_factory")),
                ):
                    with self.subTest(resource=name):
                        self.assertEqual(operation.call_count, int(acquired))

    def test_acoustic_sample_releases_resources_on_startup_and_cleanup_failures(self) -> None:
        for stage in ("ptt_factory", "capture_factory", "tts_factory", "capture_start", "ptt_start", "ptt_stop", "capture_stop", "tts_close", "cancel"):
            with self.subTest(stage=stage):
                monitor = mock.Mock()
                monitor.take_results.return_value = (AcousticResult(True, 75.0, "qualification.acoustic_sample_valid"),)
                ptt, capture, tts = mock.Mock(), mock.Mock(), mock.Mock(last_error=None)
                failure = KeyboardInterrupt() if stage == "ptt_stop" else RuntimeError("private sample failure")
                factories = {name: mock.Mock(return_value=value) for name, value in (("ptt", ptt), ("capture", capture), ("tts", tts))}
                if stage.endswith("_factory"):
                    factories[stage.removesuffix("_factory")].side_effect = failure
                elif stage == "cancel":
                    def cancel(_epoch, reason):
                        if reason == "qualification_end":
                            raise failure
                    tts.cancel.side_effect = cancel
                else:
                    {"capture_start": capture.start, "ptt_start": ptt.start, "ptt_stop": ptt.stop,
                     "capture_stop": capture.stop, "tts_close": tts.close}[stage].side_effect = failure
                def enqueue(unit):
                    factories["tts"].call_args.kwargs["on_started"](unit)
                    factories["ptt"].call_args.args[0].events.put_nowait(PttEvent(PttTransition.DOWN, time.perf_counter()))
                tts.enqueue.side_effect = enqueue
                tts.wait_idle.return_value = True
                with mock.patch.object(native_run, "sys", SimpleNamespace(platform="darwin")), self.assertRaises(type(failure)) as raised:
                    run_macos_acoustic_barge_in_sample(
                        voice_config_from_mapping({}), silence_threshold_dbfs=-40.0,
                        audio_path="built_in", monitor_factory=lambda **_: monitor,
                        ptt_source_factory=factories["ptt"], capture_factory=factories["capture"],
                        tts_builder=factories["tts"],
                    )
                self.assertIs(raised.exception, failure)
                for name, operation, acquired in (
                    ("monitor", monitor.close, True),
                    ("ptt", ptt.stop, stage != "ptt_factory"),
                    ("capture", capture.stop, stage not in ("ptt_factory", "capture_factory")),
                    ("tts", tts.close, stage not in ("ptt_factory", "capture_factory", "tts_factory")),
                ):
                    with self.subTest(resource=name):
                        self.assertEqual(operation.call_count, int(acquired))

    def test_collection_cleans_up_initialization_thread_and_shutdown_failures(self) -> None:
        thread_class = threading.Thread
        for stage in ("profile", "auth", "server_factory", "runtime_factory", "thread_factory", "thread_start", "server_start", "request_stop", "server_stop", "write", "progress"):
            with self.subTest(stage=stage), contextlib.ExitStack() as patches:
                monitor = mock.Mock()
                monitor.take_results.return_value = ()
                runtime, server = mock.Mock(), mock.Mock()
                stopped = threading.Event()
                entered = threading.Event()
                failure = RuntimeError("private collection failure")
                acquired = set()
                workers = []
                def make_runtime(*args, **kwargs):
                    if stage == "runtime_factory":
                        raise failure
                    acquired.add("runtime")
                    return runtime
                def make_server(*args, **kwargs):
                    if stage == "server_factory":
                        raise failure
                    acquired.add("server")
                    return server
                def make_thread(*args, **kwargs):
                    if stage == "thread_factory":
                        raise failure
                    worker = thread_class(*args, **kwargs)
                    workers.append(worker)
                    return worker
                def run():
                    entered.set()
                    try:
                        stopped.wait(timeout=2.0)
                    finally:
                        runtime.close()
                def request_stop():
                    stopped.set()
                    if stage == "request_stop":
                        raise failure
                def interrupt(_seconds):
                    self.assertTrue(entered.wait(timeout=1.0))
                    raise KeyboardInterrupt
                runtime.run.side_effect = run
                runtime.request_stop.side_effect = request_stop
                runtime.timing_snapshot.return_value = ()
                patches.enter_context(mock.patch.object(native_run, "sys", SimpleNamespace(platform="darwin")))
                patches.enter_context(mock.patch.object(native_run.threading, "Thread", side_effect=make_thread))
                patches.enter_context(mock.patch.object(native_run, "AcousticSilenceMonitor", return_value=monitor))
                profile = patches.enter_context(mock.patch.object(native_run, "mac_runtime_profile_fingerprint", return_value="a" * 64))
                auth = patches.enter_context(mock.patch.object(native_run, "configured_voice_auth_key", return_value=object()))
                writer = patches.enter_context(mock.patch.object(native_run, "write_report"))
                if stage in ("profile", "auth", "server_start", "server_stop", "write"):
                    {"profile": profile, "auth": auth, "server_start": server.start,
                     "server_stop": server.stop, "write": writer}[stage].side_effect = failure
                elif stage == "thread_start":
                    patches.enter_context(mock.patch.object(thread_class, "start", side_effect=failure))
                progress = mock.Mock(side_effect=failure if stage == "progress" else None)
                try:
                    with self.assertRaises(RuntimeError) as raised:
                        run_macos_native_collection(
                            voice_config_from_mapping({}), output="/tmp/unused-native-cleanup.json",
                            silence_threshold_dbfs=-40.0, audio_path="built_in",
                            duration_seconds=1.0, threshold_calibrated=True,
                            runtime_factory=make_runtime, server_factory=make_server,
                            rss_probe=lambda: 100_000_000, sleeper=interrupt, progress_sink=progress,
                        )
                    self.assertIs(raised.exception, failure)
                    for name, operation, owned in (
                        ("monitor", monitor.close, True),
                        ("server", server.stop, "server" in acquired),
                        ("runtime", runtime.close, "runtime" in acquired),
                    ):
                        with self.subTest(resource=name):
                            self.assertEqual(operation.call_count, int(owned))
                    if stage in ("request_stop", "server_stop", "progress"):
                        self.assertGreaterEqual(writer.call_args.args[1]["counters"]["crashes"], 1)
                finally:
                    stopped.set()
                    for worker in workers:
                        if worker.ident is not None:
                            worker.join(timeout=1.0)
                            self.assertFalse(worker.is_alive())

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
