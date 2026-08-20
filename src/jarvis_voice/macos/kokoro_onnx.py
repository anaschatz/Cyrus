"""CPU-only Kokoro ONNX English TTS with epoch-safe system fallback."""

from __future__ import annotations

import hashlib
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..backends import SpeechUnit
from ..config import ArtifactConfig, KokoroOnnxConfig
from ..errors import BackendError


SAMPLE_RATE = 24_000
PCM_BLOCK_SAMPLES = 480
CPU_INTRA_OP_THREADS = 2
CPU_INTER_OP_THREADS = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_kokoro_artifact(artifact: ArtifactConfig, name: str) -> Path:
    """Resolve and verify a digest-bound local Kokoro artifact."""

    if (
        not isinstance(artifact.path, str)
        or not isinstance(artifact.size, int)
        or isinstance(artifact.size, bool)
        or artifact.size < 1
        or not isinstance(artifact.sha256, str)
        or len(artifact.sha256) != 64
    ):
        raise BackendError(
            "tts.kokoro.artifact_manifest_invalid",
            "Kokoro artifact manifest is invalid",
            subsystem="tts",
            safe_context={"artifact": name},
        )
    path = Path(artifact.path).expanduser()
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != artifact.size
        or _sha256(path) != artifact.sha256
    ):
        raise BackendError(
            "tts.kokoro.artifact_changed",
            "Kokoro artifact failed integrity validation",
            subsystem="tts",
            safe_context={"artifact": name},
        )
    return path


@dataclass(frozen=True, slots=True)
class KokoroAudio:
    pcm16: bytes
    sample_rate: int = SAMPLE_RATE
    channels: int = 1


class KokoroEngine(Protocol):
    def warm(self) -> float: ...
    def synthesize(self, text: str, cancelled: threading.Event) -> KokoroAudio: ...
    def cancel(self) -> None: ...


class Playback(Protocol):
    def start(self, sample_rate: int) -> None: ...
    def write(self, pcm16: bytes) -> None: ...
    def finish(self) -> None: ...
    def abort(self) -> None: ...


class RoutedFallback(Protocol):
    def enqueue(self, unit: SpeechUnit) -> None: ...
    def cancel(self, new_epoch: int, reason: str) -> None: ...
    @property
    def is_idle(self) -> bool: ...
    @property
    def last_error(self) -> BackendError | None: ...
    def close(self) -> None: ...


class KokoroOnnxCpuEngine:
    """Lazy, network-free ONNX Runtime engine pinned to CPUExecutionProvider."""

    def __init__(self, config: KokoroOnnxConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._runtime: tuple[Any, Any] | None = None
        self._warm_load_ms: float | None = None
        self.last_synthesis_ms: float | None = None
        self.last_audio_seconds: float | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        with self._lock:
            if self._runtime is not None:
                return self._runtime
            started = time.perf_counter()
            model = validate_kokoro_artifact(self.config.model, "model")
            voices = validate_kokoro_artifact(self.config.voices, "voices")
            try:
                import numpy as np
                import onnxruntime as ort
                from kokoro_onnx import Kokoro
            except ImportError as exc:
                raise BackendError(
                    "tts.kokoro.dependencies_missing",
                    "Kokoro ONNX runtime dependencies are missing",
                    subsystem="tts",
                ) from exc
            try:
                session_options = ort.SessionOptions()
                session_options.intra_op_num_threads = CPU_INTRA_OP_THREADS
                session_options.inter_op_num_threads = CPU_INTER_OP_THREADS
                session = ort.InferenceSession(
                    str(model),
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
                if session.get_providers() != ["CPUExecutionProvider"]:
                    raise BackendError(
                        "tts.kokoro.cpu_provider_required",
                        "Kokoro must run on the CPU provider",
                        subsystem="tts",
                    )
                kokoro = Kokoro.from_session(session, str(voices))
                george = kokoro.get_voice_style("bm_george")
                fable = kokoro.get_voice_style("bm_fable")
                blend = np.asarray(george * 0.70 + fable * 0.30, dtype=np.float32)
            except BackendError:
                raise
            except Exception as exc:
                raise BackendError(
                    "tts.kokoro.load_failed",
                    "Kokoro ONNX model could not be loaded",
                    subsystem="tts",
                ) from exc
            self._runtime = kokoro, blend
            self._warm_load_ms = (time.perf_counter() - started) * 1000.0
            return self._runtime

    def warm(self) -> float:
        self._ensure_loaded()
        return float(self._warm_load_ms or 0.0)

    def synthesize(self, text: str, cancelled: threading.Event) -> KokoroAudio:
        self._cancelled.clear()
        if cancelled.is_set():
            return KokoroAudio(b"")
        kokoro, blend = self._ensure_loaded()
        if cancelled.is_set() or self._cancelled.is_set():
            return KokoroAudio(b"")
        started = time.perf_counter()
        try:
            samples, sample_rate = kokoro.create(
                text,
                voice=blend,
                speed=self.config.speed,
                lang=self.config.language,
                trim=True,
            )
            import numpy as np

            audio = np.asarray(samples, dtype=np.float32).reshape(-1)
            if sample_rate != SAMPLE_RATE or audio.size == 0 or not np.isfinite(audio).all():
                raise ValueError("invalid Kokoro audio")
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        except Exception as exc:
            raise BackendError(
                "tts.kokoro.synthesis_failed",
                "Kokoro ONNX synthesis failed",
                subsystem="tts",
            ) from exc
        self.last_synthesis_ms = (time.perf_counter() - started) * 1000.0
        self.last_audio_seconds = len(pcm16) / (2.0 * SAMPLE_RATE)
        if cancelled.is_set() or self._cancelled.is_set():
            return KokoroAudio(b"")
        return KokoroAudio(pcm16)

    def cancel(self) -> None:
        self._cancelled.set()


class SoundDevicePlayback:
    """Low-latency CoreAudio playback whose abort discards pending buffers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stream: Any | None = None
        self._aborted = False

    def start(self, sample_rate: int) -> None:
        try:
            import sounddevice as sd

            stream = sd.RawOutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=0,
                latency="low",
            )
            stream.start()
        except Exception as exc:
            raise BackendError(
                "tts.kokoro.playback_open_failed",
                "Kokoro playback could not be opened",
                subsystem="tts",
            ) from exc
        with self._lock:
            if self._aborted:
                try:
                    stream.abort()
                finally:
                    stream.close()
                return
            self._stream = stream

    def write(self, pcm16: bytes) -> None:
        with self._lock:
            if self._aborted:
                return
            stream = self._stream
        if stream is None:
            raise BackendError("tts.kokoro.playback_not_started", "Kokoro playback is not ready", subsystem="tts")
        stream.write(pcm16)

    def finish(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.abort()
        finally:
            stream.close()


class KokoroOnnxTtsBackend:
    """Serialize English units, fence stale output, and fall back before audio."""

    def __init__(
        self,
        *,
        engine: KokoroEngine,
        fallback_backend: RoutedFallback,
        playback_factory: Callable[[], Playback] = SoundDevicePlayback,
        on_started: Callable[[SpeechUnit], None] | None = None,
        on_completed: Callable[[SpeechUnit], None] | None = None,
        queue_capacity: int = 16,
        maximum_text_bytes: int = 16_384,
        warm_on_start: bool = True,
        warm_timeout_seconds: float = 30.0,
    ) -> None:
        if queue_capacity < 1 or maximum_text_bytes < 1 or warm_timeout_seconds <= 0:
            raise ValueError("Kokoro runtime bounds must be positive")
        self.engine = engine
        self.fallback_backend = fallback_backend
        self.playback_factory = playback_factory
        self.on_started = on_started or (lambda unit: None)
        self.on_completed = on_completed or (lambda unit: None)
        self.maximum_text_bytes = maximum_text_bytes
        self._queue: queue.Queue[SpeechUnit] = queue.Queue(maxsize=queue_capacity)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._warm_ready = threading.Event()
        self._cancelled_through_epoch = 0
        self._active: SpeechUnit | None = None
        self._active_cancel: threading.Event | None = None
        self._playback: Playback | None = None
        self._error: BackendError | None = None
        self._primary_disabled = False
        self._warm_on_start = warm_on_start
        self._warm_load_ms: float | None = None
        self._thread = threading.Thread(target=self._worker, name="jarvis-kokoro-onnx", daemon=True)
        self._thread.start()
        if warm_on_start and not self._warm_ready.wait(warm_timeout_seconds):
            self._primary_disabled = True

    @property
    def warm_load_ms(self) -> float | None:
        return self._warm_load_ms

    @property
    def degraded_to_fallback(self) -> bool:
        return self._primary_disabled

    def enqueue(self, unit: SpeechUnit) -> None:
        size = len(unit.text.encode("utf-8"))
        if size == 0 or size > self.maximum_text_bytes or unit.locale != "en-US":
            raise BackendError("tts.unit_invalid", "Kokoro speech unit is invalid", subsystem="tts")
        with self._lock:
            if unit.epoch <= self._cancelled_through_epoch:
                return
            try:
                self._queue.put_nowait(unit)
            except queue.Full as exc:
                raise BackendError("tts.queue_overflow", "Kokoro sentence queue is full", subsystem="tts") from exc

    def cancel(self, new_epoch: int, reason: str) -> None:
        with self._lock:
            self._cancelled_through_epoch = max(self._cancelled_through_epoch, new_epoch)
            retained: list[SpeechUnit] = []
            while True:
                try:
                    unit = self._queue.get_nowait()
                except queue.Empty:
                    break
                if unit.epoch > self._cancelled_through_epoch:
                    retained.append(unit)
            for unit in retained:
                self._queue.put_nowait(unit)
            cancelled = self._active_cancel
            playback = self._playback
        if cancelled is not None:
            cancelled.set()
        self.engine.cancel()
        if playback is not None:
            playback.abort()
        self.fallback_backend.cancel(new_epoch, reason)

    @property
    def is_idle(self) -> bool:
        with self._lock:
            return self._active is None and self._queue.empty() and self.fallback_backend.is_idle

    @property
    def last_error(self) -> BackendError | None:
        return self._error or self.fallback_backend.last_error

    def wait_idle(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_idle:
                return True
            time.sleep(0.005)
        return self.is_idle

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self.cancel(2**63 - 1, "application_close")
        self._thread.join(timeout=2.0)
        self.fallback_backend.close()

    def _is_stale(self, unit: SpeechUnit, cancelled: threading.Event) -> bool:
        with self._lock:
            return self._stop.is_set() or cancelled.is_set() or unit.epoch <= self._cancelled_through_epoch

    def _use_fallback(self, unit: SpeechUnit, cancelled: threading.Event) -> None:
        if self._is_stale(unit, cancelled):
            return
        self.fallback_backend.enqueue(unit)
        while not self._is_stale(unit, cancelled) and not self.fallback_backend.is_idle:
            time.sleep(0.005)
        error = self.fallback_backend.last_error
        if error is not None and not self._is_stale(unit, cancelled):
            raise error

    def _worker(self) -> None:
        if self._warm_on_start:
            try:
                self._warm_load_ms = self.engine.warm()
            except Exception:
                self._primary_disabled = True
            finally:
                self._warm_ready.set()
        while not self._stop.is_set():
            try:
                unit = self._queue.get(timeout=0.02)
            except queue.Empty:
                continue
            cancelled = threading.Event()
            with self._lock:
                if unit.epoch <= self._cancelled_through_epoch:
                    continue
                self._active = unit
                self._active_cancel = cancelled
            playback: Playback | None = None
            started = False
            try:
                if self._primary_disabled:
                    self._use_fallback(unit, cancelled)
                    continue
                try:
                    audio = self.engine.synthesize(unit.text, cancelled)
                    if self._is_stale(unit, cancelled):
                        continue
                    if not audio.pcm16 or audio.sample_rate != SAMPLE_RATE or audio.channels != 1:
                        raise BackendError("tts.kokoro.audio_invalid", "Kokoro emitted invalid audio", subsystem="tts")
                    playback = self.playback_factory()
                    with self._lock:
                        self._playback = playback
                    playback.start(audio.sample_rate)
                    if self._is_stale(unit, cancelled):
                        playback.abort()
                        continue
                    self.on_started(unit)
                    started = True
                    block_bytes = PCM_BLOCK_SAMPLES * 2
                    for offset in range(0, len(audio.pcm16), block_bytes):
                        if self._is_stale(unit, cancelled):
                            break
                        playback.write(audio.pcm16[offset : offset + block_bytes])
                    if self._is_stale(unit, cancelled):
                        playback.abort()
                        continue
                    playback.finish()
                    self.on_completed(unit)
                except Exception as exc:
                    if playback is not None:
                        playback.abort()
                    if self._is_stale(unit, cancelled):
                        continue
                    if started:
                        raise exc
                    self._primary_disabled = True
                    self._use_fallback(unit, cancelled)
            except BackendError as exc:
                self._error = exc
            except Exception as exc:
                self._error = BackendError("tts.kokoro.failed", "Kokoro TTS failed", subsystem="tts")
                self._error.__cause__ = exc
            finally:
                with self._lock:
                    if self._playback is playback:
                        self._playback = None
                    if self._active is unit:
                        self._active = None
                    if self._active_cancel is cancelled:
                        self._active_cancel = None
