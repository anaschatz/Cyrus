"""Content-free native macOS telemetry and acoustic barge-in collection."""

from __future__ import annotations

import hashlib
import json
import math
import queue
import stat
import sys
import threading
import time
from array import array
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import PROTOCOL_VERSION, __version__
from ..config import VoiceConfig
from ..errors import BackendError
from ..native_qualification import (
    MAX_PTT_TO_ACOUSTIC_SILENCE_MS,
    MINIMUM_ACOUSTIC_SAMPLES,
    native_evidence_template,
)
from ..pocket_tts import (
    POCKET_TTS_DISTRIBUTION_VERSION,
    POCKET_TTS_PYYAML_VERSION,
    POCKET_TTS_TORCH_VERSION,
)
from ..voice_auth import AUTH_PROTOCOL


ACOUSTIC_AUDIBLE_MARGIN_DB = 3.0
ACOUSTIC_MINIMUM_SEPARATION_DB = 6.0


def mac_source_runtime_sha256(repository_root: str | Path | None = None) -> str:
    """Bind source qualification to importable Voice code and launch scripts."""

    root = Path(__file__).resolve().parents[3] if repository_root is None else Path(repository_root).resolve()
    package_root = root / "src" / "jarvis_voice"
    candidates = [
        path
        for path in package_root.rglob("*.py")
        if path.stem.isidentifier()
        and all(part.isidentifier() for part in path.relative_to(package_root).parent.parts)
    ]
    candidates.extend(
        root / "scripts" / name
        for name in (
            "run_greektts15_sidecar_worker.py",
            "run_macos_native_collection.py",
        )
    )
    candidates = sorted(candidates, key=lambda path: path.relative_to(root).as_posix())
    if not candidates:
        raise BackendError("qualification.native_source_identity_invalid", "Runtime source identity is unavailable", subsystem="qualification")
    digest = hashlib.sha256()
    for path in candidates:
        try:
            metadata = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise OSError
            content = path.read_bytes()
            relative = path.relative_to(root).as_posix().encode("utf-8")
        except (OSError, ValueError):
            raise BackendError(
                "qualification.native_source_identity_invalid",
                "Runtime source identity is unavailable",
                subsystem="qualification",
            ) from None
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def mac_runtime_profile_fingerprint(config: VoiceConfig) -> str:
    """Hash the stable runtime selection without persisting local paths."""

    profile = {
        "product_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "voice_auth_protocol": AUTH_PROTOCOL,
        "source_runtime_sha256": mac_source_runtime_sha256(),
        "ptt_key": config.ptt_key,
        "asr_backend": config.macos.asr_backend,
        "asr_model_sha256": config.macos.whisper_cpp.model.sha256,
        "asr_executable_sha256": config.macos.whisper_cpp.executable.sha256,
        "asr_audio_context": config.macos.whisper_cpp.qualified_audio_context,
        "asr_coreml_revision": config.macos.whisper_cpp.coreml_encoder.revision,
        "asr_coreml_files": [
            {
                "path": artifact.path,
                "size": artifact.size,
                "sha256": artifact.sha256,
            }
            for artifact in sorted(
                config.macos.whisper_cpp.coreml_encoder.files,
                key=lambda item: str(item.path),
            )
        ],
        "vad_sha256": config.vad.artifact.sha256,
        "tts_backend": config.macos.tts_backend,
        "tts_profile": (
            (
                {
                    "approval_manifest_sha256": config.macos.greektts15.approval_manifest_sha256,
                    "worker_python_sha256": config.macos.greektts15.worker_python.sha256,
                    "fallback_voice_mapping": dict(sorted(config.macos.system_voices.items())),
                }
                if config.macos.greektts15.worker_python.path is not None
                else {
                    "approval_manifest_sha256": config.macos.greektts15.approval_manifest_sha256,
                    "fallback_voice_mapping": dict(sorted(config.macos.system_voices.items())),
                }
            )
            if config.macos.tts_backend == "greektts15_f1"
            else (
                {
                    "distribution_version": POCKET_TTS_DISTRIBUTION_VERSION,
                    "torch_version": POCKET_TTS_TORCH_VERSION,
                    "pyyaml_version": POCKET_TTS_PYYAML_VERSION,
                    "model_sha256": config.pocket_tts.model.sha256,
                    "tokenizer_sha256": config.pocket_tts.tokenizer.sha256,
                    "voice_state_sha256": config.pocket_tts.voice_state.sha256,
                    "temperature": config.pocket_tts.temperature,
                    "max_tokens": config.pocket_tts.max_tokens,
                    "warm_on_start": config.pocket_tts.warm_on_start,
                    "fallback_voice_mapping": dict(
                        sorted(config.macos.system_voices.items())
                    ),
                }
                if config.macos.tts_backend == "pocket_tts"
                else {"voice_mapping": dict(sorted(config.macos.system_voices.items()))}
            )
        ),
    }
    encoded = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class AcousticResult:
    valid: bool
    latency_ms: float | None
    code: str


@dataclass(frozen=True, slots=True)
class AcousticCalibration:
    silence_threshold_dbfs: float
    ambient_p95_dbfs: float
    playback_p80_dbfs: float
    separation_db: float
    audible_margin_db: float
    ambient_samples: int
    playback_samples: int


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise BackendError("qualification.acoustic_calibration_empty", "Acoustic calibration had no samples", subsystem="qualification")
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def derive_acoustic_threshold(
    ambient_levels_dbfs: Sequence[float],
    playback_levels_dbfs: Sequence[float],
) -> AcousticCalibration:
    """Derive a conservative silence threshold from microphone observations."""

    if len(ambient_levels_dbfs) < 10 or len(playback_levels_dbfs) < 10:
        raise BackendError(
            "qualification.acoustic_calibration_samples",
            "Acoustic calibration needs more samples",
            subsystem="qualification",
            safe_context={
                "ambient_samples": len(ambient_levels_dbfs),
                "playback_samples": len(playback_levels_dbfs),
            },
        )
    ambient_p95 = _percentile(ambient_levels_dbfs, 0.95)
    # Natural speech contains short pauses. A low playback percentile measures
    # those pauses rather than the audible signal, so use a robust high speech
    # percentile while retaining a separate ambient margin.
    playback_p80 = _percentile(playback_levels_dbfs, 0.80)
    separation = playback_p80 - ambient_p95
    if separation < ACOUSTIC_MINIMUM_SEPARATION_DB:
        raise BackendError(
            "qualification.acoustic_calibration_separation",
            "Playback was not sufficiently audible at the microphone",
            subsystem="qualification",
            safe_context={
                "ambient_p95_dbfs": round(ambient_p95, 3),
                "playback_p20_dbfs": round(_percentile(playback_levels_dbfs, 0.20), 3),
                "playback_p50_dbfs": round(_percentile(playback_levels_dbfs, 0.50), 3),
                "playback_p80_dbfs": round(playback_p80, 3),
                "playback_p95_dbfs": round(_percentile(playback_levels_dbfs, 0.95), 3),
                "separation_db": round(separation, 3),
                "required_separation_db": ACOUSTIC_MINIMUM_SEPARATION_DB,
                "ambient_samples": len(ambient_levels_dbfs),
                "playback_samples": len(playback_levels_dbfs),
            },
        )
    threshold = min(
        ambient_p95 + ACOUSTIC_AUDIBLE_MARGIN_DB,
        playback_p80 - ACOUSTIC_AUDIBLE_MARGIN_DB,
    )
    threshold = max(-90.0, min(-10.0, threshold))
    return AcousticCalibration(
        silence_threshold_dbfs=round(threshold, 3),
        ambient_p95_dbfs=round(ambient_p95, 3),
        playback_p80_dbfs=round(playback_p80, 3),
        separation_db=round(separation, 3),
        audible_margin_db=ACOUSTIC_AUDIBLE_MARGIN_DB,
        ambient_samples=len(ambient_levels_dbfs),
        playback_samples=len(playback_levels_dbfs),
    )


class AcousticCalibrationSampler:
    """Bounded callback sink whose report retains levels, never PCM or text."""

    def __init__(self, *, capacity: int = 4096) -> None:
        if capacity < 1:
            raise BackendError(
                "qualification.acoustic_calibration_capacity",
                "Acoustic calibration queue capacity is invalid",
                subsystem="qualification",
            )
        self._frames: queue.Queue[tuple[str, bytes, int] | None] = queue.Queue(maxsize=capacity)
        self._phase: str | None = None
        self._overflow = threading.Event()
        self._closed = threading.Event()
        self._levels: dict[str, list[float]] = {"ambient": [], "playback": []}
        self._levels_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="jarvis-acoustic-calibration",
            daemon=True,
        )
        self._thread.start()

    def begin(self, phase: str) -> None:
        if phase not in {"ambient", "playback"}:
            raise BackendError("qualification.acoustic_calibration_phase", "Acoustic calibration phase is invalid", subsystem="qualification")
        self._phase = phase

    def end(self) -> None:
        self._phase = None

    def publish_audio(self, pcm_float32: bytes, sample_rate: int, received_at: float) -> None:
        del received_at
        phase = self._phase
        if (
            self._closed.is_set()
            or phase is None
            or sample_rate < 8_000
            or not pcm_float32
            or len(pcm_float32) % 4
        ):
            return
        try:
            self._frames.put_nowait((phase, bytes(pcm_float32), sample_rate))
        except queue.Full:
            self._overflow.set()

    def _run(self) -> None:
        pending = {"ambient": bytearray(), "playback": bytearray()}
        rates: dict[str, int] = {}

        def consume(phase: str, *, final: bool = False) -> None:
            rate = rates.get(phase)
            if rate is None:
                return
            window_bytes = max(1, round(rate * 0.02)) * 4
            buffer = pending[phase]
            while len(buffer) >= window_bytes or (final and buffer):
                take = window_bytes if len(buffer) >= window_bytes else len(buffer)
                pcm = bytes(buffer[:take])
                del buffer[:take]
                level, _sample_count = AcousticSilenceMonitor._level_dbfs(pcm)
                with self._levels_lock:
                    self._levels[phase].append(level)

        while True:
            item = self._frames.get()
            try:
                if item is None:
                    consume("ambient", final=True)
                    consume("playback", final=True)
                    return
                phase, pcm, sample_rate = item
                previous_rate = rates.setdefault(phase, sample_rate)
                if previous_rate != sample_rate:
                    raise ValueError("sample rate changed during acoustic calibration")
                pending[phase].extend(pcm)
                consume(phase)
            except Exception:
                self._overflow.set()
            finally:
                self._frames.task_done()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self.end()
        self._closed.set()
        try:
            self._frames.put(None, timeout=2.0)
        except queue.Full:
            self._overflow.set()
            return
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self._overflow.set()

    def levels(self) -> dict[str, list[float]]:
        self.close()
        if self._overflow.is_set():
            raise BackendError("qualification.acoustic_calibration_overflow", "Acoustic calibration queue overflowed", subsystem="qualification")
        with self._levels_lock:
            return {phase: list(values) for phase, values in self._levels.items()}


class AcousticSilenceMonitor:
    """Measure microphone-observed silence after physical PTT_DOWN.

    The CoreAudio callback only copies into a bounded non-blocking queue. RMS
    work and the 50 ms sustained-silence decision happen on a worker thread.
    """

    def __init__(
        self,
        *,
        silence_threshold_dbfs: float,
        silence_hold_ms: int = 50,
        audible_margin_db: float = ACOUSTIC_AUDIBLE_MARGIN_DB,
        pre_roll_ms: int = 250,
        capacity: int = 256,
    ) -> None:
        if not -90.0 <= silence_threshold_dbfs <= -10.0:
            raise BackendError("qualification.acoustic_threshold_invalid", "Acoustic threshold is invalid", subsystem="qualification")
        if not 50 <= silence_hold_ms <= 500 or not 50 <= pre_roll_ms <= 1_000 or capacity < 8:
            raise BackendError("qualification.acoustic_monitor_invalid", "Acoustic monitor settings are invalid", subsystem="qualification")
        self.silence_threshold_dbfs = float(silence_threshold_dbfs)
        self.silence_hold_seconds = silence_hold_ms / 1000.0
        self.audible_threshold_dbfs = self.silence_threshold_dbfs + audible_margin_db
        self.pre_roll_seconds = pre_roll_ms / 1000.0
        self._commands: queue.Queue[tuple[object, ...]] = queue.Queue(maxsize=capacity)
        self._results: queue.Queue[AcousticResult] = queue.Queue()
        self._stop = threading.Event()
        self._overflow = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jarvis-acoustic-silence", daemon=True)
        self._thread.start()

    def publish_audio(self, pcm_float32: bytes, sample_rate: int, received_at: float) -> None:
        if sample_rate < 8_000 or not pcm_float32 or len(pcm_float32) % 4:
            return
        try:
            self._commands.put_nowait(("audio", pcm_float32, sample_rate, received_at))
        except queue.Full:
            self._overflow.set()

    def arm(self, ptt_down_at: float) -> None:
        try:
            self._commands.put_nowait(("arm", ptt_down_at))
        except queue.Full:
            self._overflow.set()

    def take_results(self) -> tuple[AcousticResult, ...]:
        results: list[AcousticResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return tuple(results)

    @staticmethod
    def _level_dbfs(pcm: bytes) -> tuple[float, int]:
        samples = array("f")
        samples.frombytes(pcm)
        if not samples:
            return -120.0, 0
        mean_square = sum(float(value) * float(value) for value in samples) / len(samples)
        return (-120.0 if mean_square <= 1e-12 else 20.0 * math.log10(math.sqrt(mean_square))), len(samples)

    def _run(self) -> None:
        history: deque[tuple[float, float]] = deque()
        armed_at: float | None = None
        audible = False
        silence_started: float | None = None
        pending = bytearray()
        pending_rate: int | None = None
        pending_start: float | None = None

        def process_window(pcm: bytes, frame_start: float, frame_end: float) -> None:
            nonlocal armed_at, audible, silence_started
            level, _sample_count = self._level_dbfs(pcm)
            history.append((frame_end, level))
            while history and history[0][0] < frame_end - self.pre_roll_seconds:
                history.popleft()
            if armed_at is None:
                return
            if self._overflow.is_set():
                self._results.put(AcousticResult(False, None, "qualification.acoustic_queue_overflow"))
                self._overflow.clear()
                armed_at = None
                return
            if not audible and level >= self.audible_threshold_dbfs and frame_end <= armed_at + 0.05:
                audible = True
            if not audible:
                if frame_end - armed_at > 0.25:
                    self._results.put(AcousticResult(False, None, "qualification.acoustic_preroll_not_audible"))
                    armed_at = None
                return
            if level <= self.silence_threshold_dbfs:
                silence_started = frame_start if silence_started is None else silence_started
                if frame_end - silence_started >= self.silence_hold_seconds:
                    latency = max(0.0, (silence_started - armed_at) * 1000.0)
                    self._results.put(AcousticResult(True, round(latency, 3), "qualification.acoustic_sample_valid"))
                    armed_at = None
                    silence_started = None
            else:
                silence_started = None
            if armed_at is not None and frame_end - armed_at > 2.0:
                self._results.put(AcousticResult(False, None, "qualification.acoustic_silence_timeout"))
                armed_at = None

        while not self._stop.is_set():
            try:
                command = self._commands.get(timeout=0.1)
            except queue.Empty:
                continue
            kind = command[0]
            if kind == "stop":
                return
            if kind == "arm":
                armed_at = float(command[1])
                audible = any(
                    timestamp >= armed_at - self.pre_roll_seconds and level >= self.audible_threshold_dbfs
                    for timestamp, level in history
                )
                silence_started = None
                if self._overflow.is_set():
                    self._results.put(AcousticResult(False, None, "qualification.acoustic_queue_overflow"))
                    self._overflow.clear()
                    armed_at = None
                continue
            _kind, pcm, sample_rate, received_at = command
            sample_rate = int(sample_rate)
            sample_count = len(pcm) // 4
            if pending_rate is not None and pending_rate != sample_rate:
                self._overflow.set()
                pending.clear()
                pending_start = None
            pending_rate = sample_rate
            if pending_start is None:
                pending_start = float(received_at) - sample_count / sample_rate
            pending.extend(pcm)
            window_samples = max(1, round(sample_rate * 0.02))
            window_bytes = window_samples * 4
            window_seconds = window_samples / sample_rate
            while len(pending) >= window_bytes and pending_start is not None:
                window = bytes(pending[:window_bytes])
                del pending[:window_bytes]
                window_end = pending_start + window_seconds
                process_window(window, pending_start, window_end)
                pending_start = window_end
            if not pending:
                pending_start = None

    def close(self) -> None:
        self._stop.set()
        try:
            self._commands.put_nowait(("stop",))
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)


class MacNativeTelemetryCollector:
    """Aggregate real runtime telemetry into the strict native schema."""

    def __init__(
        self,
        *,
        profile_fingerprint: str,
        profile_mode: str = "source",
        audio_path: str,
        acoustic_monitor: AcousticSilenceMonitor | None = None,
        platform_name: str | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        selected_platform = sys.platform if platform_name is None else platform_name
        if len(profile_fingerprint) != 64 or any(character not in "0123456789abcdef" for character in profile_fingerprint):
            raise BackendError("qualification.native_profile_invalid", "Runtime profile fingerprint is invalid", subsystem="qualification")
        if profile_mode not in {"source", "signed_package"} or audio_path not in {"wired", "built_in"}:
            raise BackendError("qualification.native_collector_config_invalid", "Native collector configuration is invalid", subsystem="qualification")
        self.clock = clock
        self.started_at = float(clock())
        self.profile_fingerprint = profile_fingerprint
        self.profile_mode = profile_mode
        self.audio_path = audio_path
        self.acoustic_monitor = acoustic_monitor
        self.native_machine = selected_platform == "darwin"
        initial = native_evidence_template("macos")
        self.run_id = str(initial["run_id"])
        self.cases = {name: {"status": "pending", "attempts": 0} for name in initial["cases"]}
        self.transcript_cycles: set[int] = set()
        self.started_units: set[tuple[int, int]] = set()
        self.completed_responses: set[int] = set()
        self.duplicate_transcripts = 0
        self.duplicate_playbacks = 0
        self.stuck_ptt = 0
        self.microphone_recoveries = 0
        self.sleep_recoveries = 0
        self.crashes = 0
        self._ptt_active = False
        self._stuck_counted = False
        self._hardware_capture = False
        self._physical_ptt = False
        self._physical_playback = False
        self._rss: list[tuple[float, int]] = []
        self._acoustic_latencies: list[float] = []
        self._acoustic_invalid = 0
        self._pass("stable_runtime_profile")

    @property
    def real_voice_turn_count(self) -> int:
        return len(self.transcript_cycles)

    @property
    def completed_playback_count(self) -> int:
        return len(self.completed_responses)

    def _pass(self, name: str) -> None:
        self.cases[name]["status"] = "pass"
        self.cases[name]["attempts"] += 1

    def _fail(self, name: str) -> None:
        self.cases[name]["status"] = "fail"
        self.cases[name]["attempts"] += 1

    def feed(self, entry: Mapping[str, object]) -> None:
        name = entry.get("event")
        timestamp = entry.get("timestamp")
        if not isinstance(name, str) or not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            raise BackendError("qualification.native_telemetry_invalid", "Native telemetry is invalid", subsystem="qualification")
        if name == "audio_stream_started":
            self._hardware_capture = True
            self._pass("coreaudio_capture")
        elif name == "ptt_source_started":
            self._pass("accessibility_permission")
            self._pass("ptt_event_tap")
        elif name == "ptt_down_received":
            self._physical_ptt = True
            if self._ptt_active:
                self.stuck_ptt += 1
            self._ptt_active = True
            self._stuck_counted = False
            if entry.get("output_active") is True and self.acoustic_monitor is not None:
                self.acoustic_monitor.arm(float(timestamp))
        elif name == "ptt_up_received":
            self._ptt_active = False
            self._stuck_counted = False
        elif name == "transcript_final_created":
            cycle = entry.get("ptt_cycle")
            if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 1:
                raise BackendError("qualification.native_telemetry_invalid", "Transcript cycle is invalid", subsystem="qualification")
            if cycle in self.transcript_cycles:
                self.duplicate_transcripts += 1
            else:
                self.transcript_cycles.add(cycle)
        elif name == "tts_unit_started":
            response = entry.get("response_token")
            unit = entry.get("unit_index")
            if (
                not isinstance(response, int)
                or isinstance(response, bool)
                or response < 1
                or not isinstance(unit, int)
                or isinstance(unit, bool)
                or unit < 1
            ):
                raise BackendError("qualification.native_telemetry_invalid", "TTS token is invalid", subsystem="qualification")
            identity = (response, unit)
            if identity in self.started_units:
                self.duplicate_playbacks += 1
            else:
                self.started_units.add(identity)
            self._physical_playback = True
            self._pass("selected_tts_backend")
        elif name == "playback_completed_created":
            response = entry.get("response_token")
            if not isinstance(response, int) or isinstance(response, bool) or response < 1:
                raise BackendError("qualification.native_telemetry_invalid", "Playback token is invalid", subsystem="qualification")
            if response in self.completed_responses:
                self.duplicate_playbacks += 1
            else:
                self.completed_responses.add(response)
        elif name == "microphone_disconnect_recovered":
            self.record_microphone_recovery(passed=True)
        elif name == "sleep_resume_recovered":
            self.record_sleep_recovery(passed=True)
        elif name == "native_io_recovery_failed":
            recovery_kind = entry.get("recovery_kind")
            if recovery_kind == "microphone_disconnect":
                self.record_microphone_recovery(passed=False)
            elif recovery_kind == "sleep_resume":
                self.record_sleep_recovery(passed=False)

    def feed_many(self, entries: Sequence[Mapping[str, object]]) -> None:
        for entry in entries:
            self.feed(entry)

    def poll_acoustic(self) -> None:
        if self.acoustic_monitor is None:
            return
        for result in self.acoustic_monitor.take_results():
            if result.valid and result.latency_ms is not None:
                self._acoustic_latencies.append(result.latency_ms)
            else:
                self._acoustic_invalid += 1

    def sample_rss(self, rss_bytes: int, *, timestamp: float | None = None) -> None:
        if not isinstance(rss_bytes, int) or isinstance(rss_bytes, bool) or rss_bytes <= 0:
            raise BackendError("qualification.native_rss_invalid", "RSS sample is invalid", subsystem="qualification")
        selected = float(self.clock()) if timestamp is None else float(timestamp)
        if self._rss and selected < self._rss[-1][0]:
            raise BackendError("qualification.native_rss_time_invalid", "RSS timestamp moved backwards", subsystem="qualification")
        self._rss.append((selected, rss_bytes))

    def record_microphone_recovery(self, *, passed: bool) -> None:
        (self._pass if passed else self._fail)("microphone_disconnect_recovery")
        (self._pass if passed else self._fail)("microphone_change")
        if passed:
            self.microphone_recoveries += 1

    def record_sleep_recovery(self, *, passed: bool) -> None:
        (self._pass if passed else self._fail)("sleep_resume_recovery")
        (self._pass if passed else self._fail)("suspend_resume")
        if passed:
            self.sleep_recoveries += 1

    def record_crash(self) -> None:
        self.crashes += 1

    @staticmethod
    def _rss_slope(samples: list[tuple[float, int]]) -> float | None:
        if len(samples) < 2:
            return None
        origin = samples[0][0]
        hours = [(timestamp - origin) / 3600.0 for timestamp, _rss in samples]
        values = [float(rss) for _timestamp, rss in samples]
        mean_x = sum(hours) / len(hours)
        mean_y = sum(values) / len(values)
        denominator = sum((value - mean_x) ** 2 for value in hours)
        if denominator <= 0.0:
            return 0.0
        return sum((x - mean_x) * (y - mean_y) for x, y in zip(hours, values, strict=True)) / denominator

    def evidence(self, *, now: float | None = None, final: bool = False) -> dict[str, object]:
        self.poll_acoustic()
        selected_now = float(self.clock()) if now is None else float(now)
        if final and self._ptt_active and not self._stuck_counted:
            self.stuck_ptt += 1
            self._stuck_counted = True
        acoustic_pass = (
            self._acoustic_invalid == 0
            and len(self._acoustic_latencies) >= MINIMUM_ACOUSTIC_SAMPLES
            and max(self._acoustic_latencies, default=MAX_PTT_TO_ACOUSTIC_SILENCE_MS) < MAX_PTT_TO_ACOUSTIC_SILENCE_MS
        )
        if acoustic_pass:
            self.cases["acoustic_barge_in"] = {"status": "pass", "attempts": len(self._acoustic_latencies)}
        elif self._acoustic_invalid:
            self.cases["acoustic_barge_in"] = {"status": "fail", "attempts": self._acoustic_invalid}
        rss_values = [value for _timestamp, value in self._rss]
        template = native_evidence_template("macos", run_id=self.run_id)
        template["provenance"] = {
            "native_machine": self.native_machine,
            "voice_layer_mocks_used": False,
            "synthetic_voice_turns": False,
            "hardware_capture": self._hardware_capture,
            "physical_ptt": self._physical_ptt,
            "physical_playback": self._physical_playback,
        }
        template["cases"] = {name: dict(value) for name, value in self.cases.items()}
        template["counters"] = {
            "real_voice_turns": len(self.transcript_cycles),
            "duplicate_transcripts": self.duplicate_transcripts,
            "duplicate_tts_playbacks": self.duplicate_playbacks,
            "stuck_ptt": self.stuck_ptt,
            "microphone_disconnect_recoveries": self.microphone_recoveries,
            "sleep_resume_recoveries": self.sleep_recoveries,
            "crashes": self.crashes,
        }
        template["acoustic_barge_in"] = {
            "method": "acoustic_microphone" if self.acoustic_monitor is not None else "pending",
            "audio_path": self.audio_path,
            "samples": len(self._acoustic_latencies),
            "max_ptt_to_silence_ms": max(self._acoustic_latencies) if self._acoustic_latencies else None,
            "silence_hold_ms": 50 if self.acoustic_monitor is not None else 0,
        }
        template["soak"] = {
            "duration_seconds": max(0, round(selected_now - self.started_at)),
            "rss_samples": len(self._rss),
            "rss_start_bytes": rss_values[0] if rss_values else None,
            "rss_end_bytes": rss_values[-1] if rss_values else None,
            "rss_peak_bytes": max(rss_values) if rss_values else None,
            "rss_slope_bytes_per_hour": self._rss_slope(self._rss),
        }
        template["runtime_profile"] = {
            "mode": self.profile_mode,
            "fingerprint_sha256": self.profile_fingerprint,
            "stable": True,
            "signed": self.profile_mode == "signed_package",
        }
        return template
