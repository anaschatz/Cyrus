"""Offline Pocket TTS streaming with epoch-safe local playback and fallback."""

from __future__ import annotations

import hashlib
import importlib.resources
import queue
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from .backends import SpeechUnit
from .config import ArtifactConfig, PocketTtsConfig
from .errors import BackendError


POCKET_TTS_DISTRIBUTION_VERSION = "2.1.0"
POCKET_TTS_TORCH_VERSION = "2.13.0"
POCKET_TTS_PYYAML_VERSION = "6.0.3"
POCKET_TTS_SAMPLE_RATE = 24_000
POCKET_TTS_PREBUFFER_CHUNKS = 3
PRIMARY_LOCALES = frozenset({"en-US"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pocket_tts_artifact(artifact: ArtifactConfig, name: str) -> Path:
    """Resolve one immutable, local Pocket TTS artifact without following links."""

    if (
        not isinstance(artifact.path, str)
        or not isinstance(artifact.size, int)
        or isinstance(artifact.size, bool)
        or artifact.size < 1
        or not isinstance(artifact.sha256, str)
        or len(artifact.sha256) != 64
    ):
        raise BackendError(
            "tts.pocket.artifact_manifest_invalid",
            "Pocket TTS artifact manifest is invalid",
            subsystem="tts",
            safe_context={"artifact": name},
        )
    path = Path(artifact.path).expanduser()
    try:
        metadata = path.lstat()
    except OSError:
        metadata = None
    if (
        not path.is_absolute()
        or path.is_symlink()
        or metadata is None
        or not path.is_file()
        or metadata.st_size != artifact.size
        or _sha256(path) != artifact.sha256
    ):
        raise BackendError(
            "tts.pocket.artifact_changed",
            "Pocket TTS artifact failed integrity validation",
            subsystem="tts",
            safe_context={"artifact": name},
        )
    return path


@dataclass(frozen=True, slots=True)
class PocketAudioChunk:
    pcm16: bytes
    sample_rate: int = POCKET_TTS_SAMPLE_RATE
    channels: int = 1


class PocketEngine(Protocol):
    def warm(self) -> float: ...
    def stream(
        self, text: str, cancelled: threading.Event
    ) -> Iterator[PocketAudioChunk]: ...
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


def _contains_remote_reference(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith(("hf://", "http://", "https://"))
    if isinstance(value, dict):
        return any(
            _contains_remote_reference(key) or _contains_remote_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_remote_reference(item) for item in value)
    return False


class PocketTtsCpuEngine:
    """Pinned Pocket TTS 2.1 engine that can resolve only verified local files."""

    def __init__(self, config: PocketTtsConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._runtime: tuple[Any, dict[str, Any], Any] | None = None
        self._warm_load_ms: float | None = None
        self.last_first_chunk_ms: float | None = None
        self.last_synthesis_ms: float | None = None
        self.last_audio_seconds: float | None = None

    def _ensure_loaded(self) -> tuple[Any, dict[str, Any], Any]:
        with self._lock:
            if self._runtime is not None:
                return self._runtime
            started = time.perf_counter()
            model_path = validate_pocket_tts_artifact(self.config.model, "model")
            tokenizer_path = validate_pocket_tts_artifact(
                self.config.tokenizer, "tokenizer"
            )
            voice_path = validate_pocket_tts_artifact(
                self.config.voice_state, "voice_state"
            )
            required_distributions = {
                "pocket-tts": POCKET_TTS_DISTRIBUTION_VERSION,
                "torch": POCKET_TTS_TORCH_VERSION,
                "PyYAML": POCKET_TTS_PYYAML_VERSION,
            }
            try:
                installed_versions = {
                    name: version(name) for name in required_distributions
                }
            except PackageNotFoundError as exc:
                raise BackendError(
                    "tts.pocket.dependencies_missing",
                    "Pocket TTS runtime dependencies are missing",
                    subsystem="tts",
                ) from exc
            if installed_versions != required_distributions:
                raise BackendError(
                    "tts.pocket.version_mismatch",
                    "Pocket TTS runtime versions do not match the pinned profile",
                    subsystem="tts",
                )
            try:
                import torch
                import yaml
                from pocket_tts import TTSModel

                template_resource = importlib.resources.files("pocket_tts").joinpath(
                    "config", "english.yaml"
                )
                template = yaml.safe_load(template_resource.read_text(encoding="utf-8"))
                if not isinstance(template, dict):
                    raise ValueError("Pocket TTS configuration template is invalid")
                template["weights_path"] = str(model_path)
                template["weights_path_without_voice_cloning"] = str(model_path)
                template["flow_lm"]["lookup_table"]["tokenizer_path"] = str(
                    tokenizer_path
                )
                if _contains_remote_reference(template):
                    raise ValueError("Pocket TTS configuration retained a remote reference")
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix="jarvis-pocket-tts-",
                    suffix=".yaml",
                    delete=False,
                ) as stream:
                    yaml.safe_dump(template, stream, sort_keys=True)
                    generated_config = Path(stream.name)
                try:
                    model = TTSModel.load_model(
                        config=generated_config,
                        temp=self.config.temperature,
                        lsd_decode_steps=1,
                        quantize=False,
                    )
                finally:
                    generated_config.unlink(missing_ok=True)
                parameter = next(model.parameters())
                if parameter.device.type != "cpu":
                    raise ValueError("Pocket TTS must remain on CPU")
                if int(model.config.mimi.sample_rate) != POCKET_TTS_SAMPLE_RATE:
                    raise ValueError("Pocket TTS sample rate is invalid")
                voice_state = model.get_state_for_audio_prompt(str(voice_path))
            except BackendError:
                raise
            except Exception as exc:
                raise BackendError(
                    "tts.pocket.load_failed",
                    "Pocket TTS model could not be loaded",
                    subsystem="tts",
                ) from exc
            self._runtime = model, voice_state, torch
            self._warm_load_ms = (time.perf_counter() - started) * 1000.0
            return self._runtime

    def warm(self) -> float:
        self._ensure_loaded()
        return float(self._warm_load_ms or 0.0)

    def stream(
        self, text: str, cancelled: threading.Event
    ) -> Iterator[PocketAudioChunk]:
        self._cancelled.clear()
        if cancelled.is_set():
            return
        model, voice_state, torch = self._ensure_loaded()
        if cancelled.is_set() or self._cancelled.is_set():
            return
        started = time.perf_counter()
        first_chunk_at: float | None = None
        sample_count = 0
        generator = None
        try:
            generator = model.generate_audio_stream(
                model_state=voice_state,
                text_to_generate=text,
                max_tokens=self.config.max_tokens,
                copy_state=True,
            )
            for raw_chunk in generator:
                if cancelled.is_set() or self._cancelled.is_set():
                    break
                audio = raw_chunk.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
                if audio.numel() == 0 or not bool(torch.isfinite(audio).all().item()):
                    raise ValueError("Pocket TTS emitted invalid audio")
                pcm16 = (
                    audio.clamp(-1.0, 1.0)
                    .mul(32767.0)
                    .to(dtype=torch.int16)
                    .contiguous()
                    .numpy()
                    .tobytes()
                )
                if not pcm16:
                    raise ValueError("Pocket TTS emitted an empty audio chunk")
                now = time.perf_counter()
                if first_chunk_at is None:
                    first_chunk_at = now
                sample_count += len(pcm16) // 2
                yield PocketAudioChunk(pcm16)
        except Exception as exc:
            raise BackendError(
                "tts.pocket.synthesis_failed",
                "Pocket TTS synthesis failed",
                subsystem="tts",
            ) from exc
        finally:
            if generator is not None:
                close = getattr(generator, "close", None)
                if callable(close):
                    close()
            finished = time.perf_counter()
            self.last_first_chunk_ms = (
                None if first_chunk_at is None else (first_chunk_at - started) * 1000.0
            )
            self.last_synthesis_ms = (finished - started) * 1000.0
            self.last_audio_seconds = sample_count / POCKET_TTS_SAMPLE_RATE

    def cancel(self) -> None:
        self._cancelled.set()


class SoundDeviceStreamingPlayback:
    """Bounded asynchronous PCM output with immediate buffer abort."""

    def __init__(self, *, queue_capacity: int = 32) -> None:
        if queue_capacity < 1:
            raise ValueError("Playback queue capacity must be positive")
        self._lock = threading.RLock()
        self._stream: Any | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=queue_capacity)
        self._writer: threading.Thread | None = None
        self._writer_error: BaseException | None = None
        self._stream_started = False
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
        except Exception as exc:
            raise BackendError(
                "tts.pocket.playback_open_failed",
                "Pocket TTS playback could not be opened",
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
            self._stream_started = False

    def _ensure_writer(self, *, force: bool = False) -> None:
        with self._lock:
            if (
                self._aborted
                or self._stream is None
                or self._writer is not None
                or (not force and self._queue.qsize() < POCKET_TTS_PREBUFFER_CHUNKS)
            ):
                return
            writer = threading.Thread(
                target=self._run_writer,
                name="jarvis-pocket-playback",
                daemon=True,
            )
            self._writer = writer
            writer.start()

    def _run_writer(self) -> None:
        try:
            while True:
                pcm16 = self._queue.get()
                if pcm16 is None:
                    return
                with self._lock:
                    if self._aborted:
                        return
                    stream = self._stream
                    stream_started = self._stream_started
                if stream is None:
                    return
                if not stream_started:
                    stream.start()
                    with self._lock:
                        if self._aborted:
                            return
                        self._stream_started = True
                stream.write(pcm16)
        except BaseException as exc:
            with self._lock:
                if not self._aborted:
                    self._writer_error = exc

    def _raise_writer_error(self) -> None:
        with self._lock:
            error = self._writer_error
        if error is not None:
            raise BackendError(
                "tts.pocket.playback_write_failed",
                "Pocket TTS playback failed while writing audio",
                subsystem="tts",
            ) from error

    def write(self, pcm16: bytes) -> None:
        with self._lock:
            if self._aborted:
                return
            ready = self._stream is not None
        if not ready:
            raise BackendError(
                "tts.pocket.playback_not_started",
                "Pocket TTS playback is not ready",
                subsystem="tts",
            )
        self._raise_writer_error()
        try:
            self._queue.put(pcm16, timeout=1.0)
        except queue.Full as exc:
            raise BackendError(
                "tts.pocket.playback_queue_timeout",
                "Pocket TTS playback queue stopped draining",
                subsystem="tts",
            ) from exc
        self._ensure_writer()
        self._raise_writer_error()

    def finish(self) -> None:
        self._ensure_writer(force=True)
        with self._lock:
            stream = self._stream
            writer = self._writer
        if stream is None:
            return
        error: BaseException | None = None
        try:
            self._queue.put(None, timeout=1.0)
            if writer is not None:
                writer.join(timeout=10.0)
                if writer.is_alive():
                    raise BackendError(
                        "tts.pocket.playback_finish_timeout",
                        "Pocket TTS playback did not finish",
                        subsystem="tts",
                    )
            self._raise_writer_error()
        except BaseException as exc:
            error = exc
        with self._lock:
            aborted = self._aborted
            owns_stream = self._stream is stream
            if owns_stream:
                self._stream = None
                self._writer = None
                self._stream_started = False
        if owns_stream and not aborted:
            try:
                if error is None:
                    stream.stop()
            finally:
                stream.close()
        if error is not None and not aborted:
            raise error

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            stream, self._stream = self._stream, None
            self._writer = None
            self._stream_started = False
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(None)
        if stream is None:
            return
        try:
            stream.abort()
        finally:
            stream.close()


class PocketTtsBackend:
    """Serialize authoritative units, stream English, and route safe fallbacks."""

    def __init__(
        self,
        *,
        engine: PocketEngine,
        fallback_backend: RoutedFallback,
        playback_factory: Callable[[], Playback] = SoundDeviceStreamingPlayback,
        on_started: Callable[[SpeechUnit], None] | None = None,
        on_completed: Callable[[SpeechUnit], None] | None = None,
        queue_capacity: int = 16,
        maximum_text_bytes: int = 2_048,
        warm_on_start: bool = True,
        warm_timeout_seconds: float = 30.0,
    ) -> None:
        if queue_capacity < 1 or maximum_text_bytes < 1 or warm_timeout_seconds <= 0:
            raise ValueError("Pocket TTS runtime bounds must be positive")
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
        self._thread = threading.Thread(
            target=self._worker, name="jarvis-pocket-tts", daemon=True
        )
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
        if size == 0 or size > self.maximum_text_bytes:
            raise BackendError(
                "tts.unit_invalid", "Pocket TTS speech unit is invalid", subsystem="tts"
            )
        with self._lock:
            if unit.epoch <= self._cancelled_through_epoch:
                return
            try:
                self._queue.put_nowait(unit)
            except queue.Full as exc:
                raise BackendError(
                    "tts.queue_overflow",
                    "Pocket TTS sentence queue is full",
                    subsystem="tts",
                ) from exc

    def cancel(self, new_epoch: int, reason: str) -> None:
        with self._lock:
            self._cancelled_through_epoch = max(
                self._cancelled_through_epoch, new_epoch
            )
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
            return (
                self._active is None
                and self._queue.empty()
                and self.fallback_backend.is_idle
            )

    @property
    def last_error(self) -> BackendError | None:
        return self._error or self.fallback_backend.last_error

    def wait_idle(self, timeout: float = 10.0) -> bool:
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
            return (
                self._stop.is_set()
                or cancelled.is_set()
                or unit.epoch <= self._cancelled_through_epoch
            )

    def _use_fallback(self, unit: SpeechUnit, cancelled: threading.Event) -> None:
        if self._is_stale(unit, cancelled):
            return
        self.fallback_backend.enqueue(unit)
        while not self._is_stale(unit, cancelled) and not self.fallback_backend.is_idle:
            time.sleep(0.005)
        error = self.fallback_backend.last_error
        if error is not None and not self._is_stale(unit, cancelled):
            raise error

    def _retire(
        self,
        unit: SpeechUnit,
        cancelled: threading.Event,
        playback: Playback | None,
    ) -> None:
        with self._lock:
            if self._playback is playback:
                self._playback = None
            if self._active is unit:
                self._active = None
            if self._active_cancel is cancelled:
                self._active_cancel = None

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
            pending_audio: list[PocketAudioChunk] = []
            started = False
            retired = False
            try:
                if unit.locale not in PRIMARY_LOCALES or self._primary_disabled:
                    self._use_fallback(unit, cancelled)
                    continue
                try:
                    for audio in self.engine.stream(unit.text, cancelled):
                        if self._is_stale(unit, cancelled):
                            break
                        if (
                            not audio.pcm16
                            or audio.sample_rate != POCKET_TTS_SAMPLE_RATE
                            or audio.channels != 1
                        ):
                            raise BackendError(
                                "tts.pocket.audio_invalid",
                                "Pocket TTS emitted invalid audio",
                                subsystem="tts",
                            )
                        if playback is None:
                            pending_audio.append(audio)
                            if len(pending_audio) < POCKET_TTS_PREBUFFER_CHUNKS:
                                continue
                            playback = self.playback_factory()
                            with self._lock:
                                self._playback = playback
                            playback.start(audio.sample_rate)
                            if self._is_stale(unit, cancelled):
                                playback.abort()
                                break
                            for buffered in pending_audio:
                                playback.write(buffered.pcm16)
                            pending_audio.clear()
                            self.on_started(unit)
                            started = True
                            continue
                        playback.write(audio.pcm16)
                    if playback is None and pending_audio and not self._is_stale(unit, cancelled):
                        playback = self.playback_factory()
                        with self._lock:
                            self._playback = playback
                        playback.start(pending_audio[0].sample_rate)
                        for buffered in pending_audio:
                            playback.write(buffered.pcm16)
                        pending_audio.clear()
                        self.on_started(unit)
                        started = True
                    if self._is_stale(unit, cancelled):
                        if playback is not None:
                            playback.abort()
                        continue
                    if playback is None or not started:
                        raise BackendError(
                            "tts.pocket.audio_invalid",
                            "Pocket TTS emitted no playable audio",
                            subsystem="tts",
                        )
                    playback.finish()
                    if self._is_stale(unit, cancelled):
                        continue
                    self._retire(unit, cancelled, playback)
                    retired = True
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
                self._error = BackendError(
                    "tts.pocket.failed", "Pocket TTS failed", subsystem="tts"
                )
                self._error.__cause__ = exc
            finally:
                if not retired:
                    self._retire(unit, cancelled, playback)


__all__ = [
    "POCKET_TTS_DISTRIBUTION_VERSION",
    "POCKET_TTS_PYYAML_VERSION",
    "POCKET_TTS_SAMPLE_RATE",
    "POCKET_TTS_TORCH_VERSION",
    "PocketAudioChunk",
    "PocketTtsBackend",
    "PocketTtsCpuEngine",
    "SoundDeviceStreamingPlayback",
    "validate_pocket_tts_artifact",
]
