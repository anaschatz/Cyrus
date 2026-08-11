"""Guided long-running native macOS qualification collection."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..backends import SpeechUnit
from ..config import VoiceConfig
from ..errors import BackendError
from ..native_qualification import evaluate_native_evidence
from ..qualification import write_report
from .doctor import macos_doctor_report
from .interactive_smoke import MacSmokeLoopbackServer
from .audio_capture import CoreAudioCapture
from .native_collector import (
    AcousticCalibrationSampler,
    AcousticSilenceMonitor,
    MacNativeTelemetryCollector,
    derive_acoustic_threshold,
    mac_runtime_profile_fingerprint,
)
from .runtime import MacVoiceRuntime
from .system_tts import MacSystemTtsBackend


def calibrate_macos_acoustic_threshold(
    config: VoiceConfig,
    *,
    audio_path: str,
    ambient_seconds: float = 3.0,
    playback_timeout_seconds: float = 20.0,
    capture_factory: Callable[..., Any] = CoreAudioCapture,
    tts_factory: Callable[..., Any] = MacSystemTtsBackend,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Measure ambient/playback separation through the real microphone path."""

    if sys.platform != "darwin":
        raise BackendError("qualification.native_macos_required", "Acoustic calibration requires macOS", subsystem="qualification")
    if audio_path not in {"wired", "built_in"} or not 1.0 <= ambient_seconds <= 15.0:
        raise BackendError("qualification.native_collector_config_invalid", "Acoustic calibration settings are invalid", subsystem="qualification")
    sampler = AcousticCalibrationSampler()
    started = threading.Event()
    completed = threading.Event()
    capture = capture_factory(
        queue_capacity=config.audio_queue_capacity,
        maximum_utterance_seconds=config.maximum_utterance_seconds,
        device=config.audio_input_device,
        frame_observer=sampler.publish_audio,
    )
    tts = tts_factory(
        voice_mapping=config.macos.system_voices,
        on_started=lambda _unit: started.set(),
        on_completed=lambda _unit: completed.set(),
    )
    try:
        capture.start()
        sampler.begin("ambient")
        sleeper(ambient_seconds)
        sampler.end()
        sampler.begin("playback")
        tts.enqueue(
            SpeechUnit(
                response_id="acoustic-calibration",
                epoch=1,
                unit_index=1,
                locale="el-GR",
                text="Η βαθμονόμηση της φωνής ολοκληρώνεται με καθαρή και σταθερή ομιλία.",
            )
        )
        if not started.wait(timeout=playback_timeout_seconds):
            raise BackendError("qualification.acoustic_calibration_tts_start", "Calibration playback did not start", subsystem="qualification")
        if not completed.wait(timeout=playback_timeout_seconds):
            raise BackendError("qualification.acoustic_calibration_tts_timeout", "Calibration playback did not complete", subsystem="qualification")
        sampler.end()
    finally:
        sampler.end()
        capture.stop()
        tts.close()
    levels = sampler.levels()
    calibration = derive_acoustic_threshold(levels["ambient"], levels["playback"])
    return {
        "ready": True,
        "code": "qualification.acoustic_calibration_completed",
        "profile_fingerprint_sha256": mac_runtime_profile_fingerprint(config),
        "audio_path": audio_path,
        "threshold_calibrated": True,
        "calibration": asdict(calibration),
        "microphone_opened": True,
        "physical_playback": True,
        "raw_audio_persisted": False,
        "network_used": False,
        "installation_performed": False,
    }


def current_process_rss_bytes(
    *,
    pid: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    """Read current, not peak, RSS through the native ps interface."""

    process_id = os.getpid() if pid is None else pid
    try:
        result = runner(
            ["/bin/ps", "-o", "rss=", "-p", str(process_id)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
            stdin=subprocess.DEVNULL,
        )
        kib = int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise BackendError("qualification.native_rss_probe_failed", "Current RSS probe failed", subsystem="qualification") from exc
    if kib <= 0:
        raise BackendError("qualification.native_rss_probe_failed", "Current RSS probe failed", subsystem="qualification")
    return kib * 1024


def macos_native_collection_preflight(
    config: VoiceConfig,
    *,
    silence_threshold_dbfs: float,
    audio_path: str,
    threshold_calibrated: bool = False,
    doctor_probe: Callable[[VoiceConfig], dict[str, object]] = macos_doctor_report,
) -> dict[str, object]:
    """Passive start-readiness check; it never opens audio or starts PTT."""

    monitor = AcousticSilenceMonitor(silence_threshold_dbfs=silence_threshold_dbfs)
    monitor.close()
    if audio_path not in {"wired", "built_in"}:
        raise BackendError("qualification.native_collector_config_invalid", "Audio path is invalid", subsystem="qualification")
    doctor = doctor_probe(config)
    ready_to_start = doctor.get("ready") is True and threshold_calibrated
    return {
        "ready_to_start": ready_to_start,
        "ready": False,
        "code": (
            "qualification.macos_native_collection_can_start"
            if ready_to_start
            else "qualification.macos_native_collection_preflight_blocked"
        ),
        "native_qualification_completed": False,
        "doctor": doctor,
        "profile_fingerprint_sha256": mac_runtime_profile_fingerprint(config),
        "acoustic": {
            "method": "acoustic_microphone",
            "audio_path": audio_path,
            "silence_threshold_dbfs": silence_threshold_dbfs,
            "silence_hold_ms": 50,
            "threshold_calibrated": threshold_calibrated,
        },
        "microphone_opened": False,
        "ptt_started": False,
        "network_used": False,
        "installation_performed": False,
    }


def run_macos_native_collection(
    config: VoiceConfig,
    *,
    output: str | Path,
    silence_threshold_dbfs: float,
    audio_path: str,
    duration_seconds: float = 28_800.0,
    rss_interval_seconds: float = 300.0,
    threshold_calibrated: bool = False,
    runtime_factory: Callable[..., MacVoiceRuntime] = MacVoiceRuntime,
    server_factory: Callable[[str], MacSmokeLoopbackServer] = MacSmokeLoopbackServer,
    rss_probe: Callable[[], int] = current_process_rss_bytes,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    progress_sink: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Collect a partial or complete real-hardware run without user content."""

    if sys.platform != "darwin":
        raise BackendError("qualification.native_macos_required", "Native collection requires macOS", subsystem="qualification")
    if not threshold_calibrated:
        raise BackendError("qualification.acoustic_calibration_required", "Acoustic threshold calibration is required", subsystem="qualification")
    if duration_seconds < 1.0 or duration_seconds > 24 * 60 * 60:
        raise BackendError("qualification.native_duration_invalid", "Native collection duration is invalid", subsystem="qualification")
    if rss_interval_seconds < 1.0 or rss_interval_seconds > 600.0:
        raise BackendError("qualification.native_rss_interval_invalid", "RSS interval is invalid", subsystem="qualification")
    monitor = AcousticSilenceMonitor(silence_threshold_dbfs=silence_threshold_dbfs)
    collector = MacNativeTelemetryCollector(
        profile_fingerprint=mac_runtime_profile_fingerprint(config),
        profile_mode="source",
        audio_path=audio_path,
        acoustic_monitor=monitor,
        clock=clock,
    )
    runtime = runtime_factory(config, audio_frame_observer=monitor.publish_audio)
    server = server_factory(config.orchestrator_url)
    runtime_errors: list[BaseException] = []

    def run_runtime() -> None:
        try:
            runtime.run()
        except BaseException as exc:
            runtime_errors.append(exc)

    thread = threading.Thread(target=run_runtime, name="jarvis-native-qualification-runtime", daemon=True)
    cursor = 0
    started = clock()
    deadline = started + duration_seconds
    next_rss = started
    next_progress = started
    interrupted = False
    try:
        server.start()
        thread.start()
        while clock() < deadline and not runtime_errors:
            snapshot = runtime.timing_snapshot()
            collector.feed_many(snapshot[cursor:])
            cursor = len(snapshot)
            collector.poll_acoustic()
            now = clock()
            if now >= next_rss:
                collector.sample_rss(rss_probe(), timestamp=now)
                next_rss += rss_interval_seconds
            if now >= next_progress:
                partial = collector.evidence(now=now)
                write_report(output, partial)
                if progress_sink is not None:
                    counters = partial["counters"]
                    acoustic = partial["acoustic_barge_in"]
                    progress_sink(
                        {
                            "elapsed_seconds": partial["soak"]["duration_seconds"],
                            "real_voice_turns": counters["real_voice_turns"],
                            "acoustic_samples": acoustic["samples"],
                        }
                    )
                next_progress += 60.0
            sleeper(0.02)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        runtime.request_stop()
        if thread.is_alive():
            thread.join(timeout=config.application.shutdown_timeout_seconds)
        server.stop()
        snapshot = runtime.timing_snapshot()
        collector.feed_many(snapshot[cursor:])
        collector.poll_acoustic()
        if runtime_errors or thread.is_alive():
            collector.record_crash()
        try:
            collector.sample_rss(rss_probe(), timestamp=clock())
        except BackendError:
            collector.record_crash()
        evidence = collector.evidence(now=clock(), final=True)
        write_report(output, evidence)
        monitor.close()

    report = evaluate_native_evidence(evidence)
    return {
        **report,
        "interrupted": interrupted,
        "evidence_written": True,
        "orchestrator_loopback_only": True,
        "voice_layer_mocks_used": False,
    }
