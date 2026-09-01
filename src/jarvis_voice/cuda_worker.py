"""Supervised, local-only faster-whisper CUDA subprocess boundary."""

from __future__ import annotations

import multiprocessing
import threading
import time
import unicodedata
import uuid
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Protocol

from .asr import decode_bilingual_transcript, weighted_logprob
from .backends import AsrScore, AudioChunk, LanguageMetadata, TranscriptResult
from .errors import BackendError
from .gpu import AdmissionDecision, GpuAdmissionPolicy, GpuInventory, require_admission
from .manifest import BASE_MODEL_MANIFEST, ModelManifest, validate_model_directory


@dataclass(frozen=True, slots=True)
class CudaWorkerConfig:
    model_directory: str
    device_index: int
    compute_type: str
    worker_start_timeout_seconds: float
    transcription_timeout_seconds: float
    maximum_audio_bytes: int
    manifest: ModelManifest = BASE_MODEL_MANIFEST


class CudaWorkerSession(Protocol):
    metrics: list[dict[str, float | int | str]]
    @property
    def alive(self) -> bool: ...
    def start(self) -> None: ...
    def transcribe(self, audio: AudioChunk) -> TranscriptResult: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


def _safe_worker_error(exc: BaseException, phase: str) -> dict[str, object]:
    description = f"{type(exc).__name__} {exc}".lower()
    if "out of memory" in description or "cuda_error_out_of_memory" in description:
        code = "cuda.oom"
    elif isinstance(exc, BackendError):
        code = exc.code
    elif phase == "load":
        code = "cuda.model_load_failed"
    else:
        code = "cuda.transcription_failed"
    return {"type": "error", "code": code, "retryable": True}


def _validate_parent_manifest(config: CudaWorkerConfig) -> None:
    validation = validate_model_directory(config.model_directory, config.manifest)
    if not validation.ok:
        context = {"artifact_name": validation.artifact_name} if validation.artifact_name else None
        raise BackendError(validation.code, "Whisper model manifest validation failed", subsystem="asr", safe_context=context)


def run_cuda_worker(
    connection: Connection,
    config: CudaWorkerConfig,
    *,
    model_factory: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    """Child entrypoint. All frames are strict dictionaries and never logged."""

    try:
        _validate_parent_manifest(config)  # Deliberate second validation in child.
        factory = model_factory
        if factory is None:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
            factory = WhisperModel
        started = clock()
        model = factory(
            config.model_directory,
            device="cuda",
            device_index=config.device_index,
            compute_type=config.compute_type,
            local_files_only=True,
        )
        connection.send({"type": "ready", "load_duration_seconds": clock() - started})
    except BaseException as exc:
        connection.send(_safe_worker_error(exc, "load"))
        connection.close()
        return

    try:
        while True:
            request = connection.recv()
            if not isinstance(request, dict) or request.get("type") not in {"transcribe", "shutdown"}:
                connection.send({"type": "error", "code": "cuda.worker_protocol_error", "retryable": False})
                return
            if request["type"] == "shutdown":
                if set(request) != {"type"}:
                    connection.send({"type": "error", "code": "cuda.worker_protocol_error", "retryable": False})
                    return
                connection.send({"type": "shutdown.complete"})
                return
            if set(request) != {"type", "request_id", "sample_rate", "audio"}:
                connection.send({"type": "error", "code": "cuda.worker_protocol_error", "retryable": False})
                return
            request_id = request["request_id"]
            audio_bytes = request["audio"]
            if (
                not isinstance(request_id, str)
                or not isinstance(audio_bytes, bytes)
                or request["sample_rate"] != 16_000
                or len(audio_bytes) > config.maximum_audio_bytes
                or len(audio_bytes) % 4
            ):
                connection.send({"type": "error", "code": "cuda.worker_invalid_audio", "retryable": False})
                continue
            try:
                try:
                    import numpy  # type: ignore[import-not-found]
                except ImportError:
                    samples: Any = array("f")
                    samples.frombytes(audio_bytes)
                else:
                    samples = numpy.frombuffer(audio_bytes, dtype=numpy.float32)
                inference_started = clock()
                text, detected_language, detected_score, segments = (
                    decode_bilingual_transcript(model, samples)
                )
                if not text:
                    raise BackendError("asr.empty_transcript", "ASR produced no transcript", subsystem="asr")
                raw, score = weighted_logprob(segments)
                connection.send(
                    {
                        "type": "result",
                        "request_id": request_id,
                        "text": text,
                        "language_code": detected_language,
                        "language_score": detected_score,
                        "asr_score": score,
                        "raw_avg_logprob": raw,
                        "inference_duration_seconds": clock() - inference_started,
                    }
                )
            except BaseException as exc:
                error = _safe_worker_error(exc, "transcribe")
                error["request_id"] = request_id
                connection.send(error)
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        connection.close()


def _process_entry(connection: Connection, config: CudaWorkerConfig) -> None:
    run_cuda_worker(connection, config)


class ProcessCudaWorkerSession:
    """Parent-side supervisor using Windows-compatible spawn semantics."""

    def __init__(self, config: CudaWorkerConfig, *, context: Any | None = None, clock: Callable[[], float] = time.perf_counter) -> None:
        self.config = config
        self.context = context or multiprocessing.get_context("spawn")
        self.clock = clock
        self.metrics: list[dict[str, float | int | str]] = []
        self._connection: Connection | None = None
        self._process: Any | None = None

    @property
    def alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def _receive(self, timeout: float, timeout_code: str) -> dict[str, Any]:
        if self._connection is None or self._process is None:
            raise BackendError("cuda.worker_not_started", "CUDA worker is not started", subsystem="cuda")
        if not self._connection.poll(timeout):
            if not self._process.is_alive():
                raise BackendError("cuda.worker_crashed", "CUDA worker exited unexpectedly", subsystem="cuda", retryable=True)
            raise BackendError(timeout_code, "CUDA worker timed out", subsystem="cuda", retryable=True)
        frame = self._connection.recv()
        if not isinstance(frame, dict) or not isinstance(frame.get("type"), str):
            raise BackendError("cuda.worker_protocol_error", "CUDA worker returned an invalid frame", subsystem="cuda")
        return frame

    @staticmethod
    def _raise_error(frame: dict[str, Any]) -> None:
        code = frame.get("code")
        if not isinstance(code, str):
            code = "cuda.worker_protocol_error"
        raise BackendError(code, "CUDA worker failed", subsystem="cuda", retryable=bool(frame.get("retryable", False)))

    def start(self) -> None:
        if self.alive:
            return
        started = self.clock()
        parent, child = self.context.Pipe(duplex=True)
        process = self.context.Process(target=_process_entry, args=(child, self.config), name="jarvis-cuda-asr", daemon=True)
        self._connection, self._process = parent, process
        process.start()
        child.close()
        try:
            frame = self._receive(self.config.worker_start_timeout_seconds, "cuda.worker_start_timeout")
            if frame["type"] == "error":
                self._raise_error(frame)
            if frame["type"] != "ready" or not isinstance(frame.get("load_duration_seconds"), (int, float)):
                raise BackendError("cuda.worker_protocol_error", "CUDA worker did not send ready", subsystem="cuda")
            self.metrics.append({"event": "worker_spawn_to_ready", "duration_seconds": self.clock() - started})
            self.metrics.append({"event": "model_load", "duration_seconds": float(frame["load_duration_seconds"])})
        except BaseException:
            self.cancel()
            raise

    def transcribe(self, audio: AudioChunk) -> TranscriptResult:
        if audio.sample_rate != 16_000:
            raise BackendError("asr.invalid_audio", "ASR requires mono 16-kHz audio", subsystem="asr")
        payload = audio.samples.tobytes()
        if len(payload) > self.config.maximum_audio_bytes:
            raise BackendError("cuda.ipc_audio_too_large", "CUDA IPC audio exceeds its bound", subsystem="cuda")
        self.start()
        assert self._connection is not None
        request_id = str(uuid.uuid4())
        self._connection.send({"type": "transcribe", "request_id": request_id, "sample_rate": 16_000, "audio": payload})
        try:
            frame = self._receive(self.config.transcription_timeout_seconds, "cuda.transcription_timeout")
        except BackendError:
            self.cancel()
            raise
        if frame["type"] == "error":
            self._raise_error(frame)
        expected = {
            "type", "request_id", "text", "language_code", "language_score",
            "asr_score", "raw_avg_logprob", "inference_duration_seconds",
        }
        if set(frame) != expected or frame.get("request_id") != request_id:
            raise BackendError("cuda.worker_protocol_error", "CUDA worker result did not match the request", subsystem="cuda")
        if not isinstance(frame["text"], str) or not frame["text"]:
            raise BackendError("cuda.worker_protocol_error", "CUDA worker returned invalid text", subsystem="cuda")
        numeric = ("language_score", "asr_score", "raw_avg_logprob", "inference_duration_seconds")
        if any(not isinstance(frame[name], (int, float)) or isinstance(frame[name], bool) for name in numeric):
            raise BackendError("cuda.worker_protocol_error", "CUDA worker returned invalid metrics", subsystem="cuda")
        self.metrics.append({"event": "cuda_inference", "duration_seconds": float(frame["inference_duration_seconds"])})
        return TranscriptResult(
            unicodedata.normalize("NFC", frame["text"]),
            LanguageMetadata(str(frame["language_code"]), float(frame["language_score"])),
            AsrScore(float(frame["asr_score"]), float(frame["raw_avg_logprob"])),
        )

    def cancel(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
        self._close_handles()

    def close(self) -> None:
        started = self.clock()
        process, connection = self._process, self._connection
        if process is not None and process.is_alive() and connection is not None:
            try:
                connection.send({"type": "shutdown"})
                frame = self._receive(min(5.0, self.config.worker_start_timeout_seconds), "cuda.worker_shutdown_timeout")
                if frame.get("type") != "shutdown.complete":
                    raise BackendError("cuda.worker_protocol_error", "CUDA worker shutdown was not acknowledged", subsystem="cuda")
            except (BackendError, BrokenPipeError, EOFError, OSError):
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate(); process.join(timeout=5)
        self._close_handles()
        self.metrics.append({"event": "worker_shutdown", "duration_seconds": self.clock() - started})

    def _close_handles(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._process = None


FALLBACKABLE_CUDA_ERRORS = frozenset(
    {
        "cuda.inventory_unavailable",
        "cuda.inventory_malformed",
        "cuda.no_devices",
        "cuda.invalid_device_index",
        "cuda.insufficient_free_vram",
        "cuda.worker_start_timeout",
        "cuda.worker_crashed",
        "cuda.model_load_failed",
        "cuda.transcription_timeout",
        "cuda.transcription_failed",
        "cuda.oom",
        "cuda.warm_restart_limit",
    }
)


class CudaFasterWhisperBackend:
    """Admission-gated CUDA backend with turn-scoped or bounded warm residency."""

    def __init__(
        self,
        worker_config: CudaWorkerConfig,
        *,
        inventory: GpuInventory,
        residency: str,
        minimum_free_vram_mb: int,
        reserve_for_other_workloads_mb: int,
        warm_restart_limit: int,
        session_factory: Callable[[CudaWorkerConfig], CudaWorkerSession] = ProcessCudaWorkerSession,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        _validate_parent_manifest(worker_config)
        if residency not in {"turn_scoped", "warm"}:
            raise ValueError("Unsupported CUDA residency")
        self.worker_config = worker_config
        self.inventory = inventory
        self.policy = GpuAdmissionPolicy(inventory, clock)
        self.residency = residency
        self.minimum_free_vram_mb = minimum_free_vram_mb
        self.reserve_for_other_workloads_mb = reserve_for_other_workloads_mb
        self.warm_restart_limit = warm_restart_limit
        self.session_factory = session_factory
        self.clock = clock
        self.metrics: list[dict[str, float | int | str]] = []
        self.last_admission: AdmissionDecision | None = None
        self._warm_session: CudaWorkerSession | None = None
        self._active_session: CudaWorkerSession | None = None
        self._warm_restarts = 0
        self._warm_started_once = False
        self._lock = threading.RLock()

    def _admit(self) -> None:
        decision = self.policy.evaluate(
            device_index=self.worker_config.device_index,
            minimum_free_vram_mb=self.minimum_free_vram_mb,
            reserve_for_other_workloads_mb=self.reserve_for_other_workloads_mb,
        )
        self.last_admission = decision
        self.metrics.append({"event": "gpu_admission", "duration_seconds": decision.duration_seconds})
        if decision.free_vram_mb is not None:
            self.metrics.append({"event": "vram_before_worker", "value_mb": decision.free_vram_mb})
        require_admission(decision)

    def _new_session(self) -> CudaWorkerSession:
        self._admit()
        session = self.session_factory(self.worker_config)
        try:
            session.start()
        except BaseException:
            session.cancel()
            self.metrics.extend(session.metrics)
            raise
        self._record_vram("vram_after_model_load")
        return session

    def _record_vram(self, event: str) -> None:
        try:
            device = next(item for item in self.inventory.devices() if item.index == self.worker_config.device_index)
        except (BackendError, StopIteration):
            return
        self.metrics.append({"event": event, "value_mb": device.free_vram_mb})

    def transcribe(self, audio: object) -> TranscriptResult:
        if not isinstance(audio, AudioChunk) or audio.sample_rate != 16_000:
            raise BackendError("asr.invalid_audio", "ASR requires mono 16-kHz audio", subsystem="asr")
        if self.residency == "turn_scoped":
            session = self._new_session()
            with self._lock:
                self._active_session = session
            try:
                result = session.transcribe(audio)
            finally:
                session.close()  # Process exit precedes result publication to the caller.
                self.metrics.extend(session.metrics)
                self._record_vram("vram_after_worker_exit")
                with self._lock:
                    if self._active_session is session:
                        self._active_session = None
            return result

        with self._lock:
            session = self._warm_session
            if session is None:
                if self._warm_started_once:
                    if self._warm_restarts >= self.warm_restart_limit:
                        raise BackendError("cuda.warm_restart_limit", "CUDA warm worker restart limit reached", subsystem="cuda", retryable=True)
                    self._warm_restarts += 1
                session = self._new_session()
                self._warm_session = session
                self._warm_started_once = True
            self._active_session = session
        try:
            return session.transcribe(audio)
        except BackendError:
            session.cancel()
            with self._lock:
                if self._warm_session is session:
                    self._warm_session = None
            raise
        finally:
            self.metrics.extend(session.metrics)
            session.metrics.clear()
            with self._lock:
                if self._active_session is session:
                    self._active_session = None

    def cancel_current(self) -> None:
        with self._lock:
            session = self._active_session
            if session is not None and self._warm_session is session:
                self._warm_session = None
            self._active_session = None
        if session is not None:
            session.cancel()

    def close(self) -> None:
        with self._lock:
            sessions = {id(item): item for item in (self._active_session, self._warm_session) if item is not None}
            self._active_session = None
            self._warm_session = None
        for session in sessions.values():
            session.close()
