"""Supervised, local-only faster-whisper CUDA subprocess boundary."""

from __future__ import annotations

import multiprocessing
import sys
import threading
import time
import unicodedata
import uuid
from array import array
from collections.abc import Callable
from concurrent.futures import Future
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


def _finite_metric(value: Any) -> bool:
    return type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max


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
            return

        while True:
            request = connection.recv()
            if not isinstance(request, dict) or request.get("type") not in ("transcribe", "shutdown"):
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
                connection.send({"type": "error", "code": "cuda.worker_invalid_audio", "retryable": False, "request_id": request_id})
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
    except (EOFError, OSError):
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
        self._lock = threading.RLock()
        self._cancelled = False

    @property
    def alive(self) -> bool:
        with self._lock:
            return bool(self._process is not None and self._process.is_alive())

    def _receive(self, timeout: float, timeout_code: str) -> dict[str, Any]:
        with self._lock:
            connection, process = self._connection, self._process
        if connection is None or process is None:
            raise BackendError("cuda.worker_not_started", "CUDA worker is not started", subsystem="cuda")
        received: Future[Any] = Future()
        def receive() -> None:
            try:
                if not connection.poll(timeout):
                    if not process.is_alive():
                        raise BackendError("cuda.worker_crashed", "CUDA worker exited unexpectedly", subsystem="cuda", retryable=True)
                    raise BackendError(timeout_code, "CUDA worker timed out", subsystem="cuda", retryable=True)
                received.set_result(connection.recv())
            except BaseException as exc:
                received.set_exception(exc)
        receiver = threading.Thread(target=receive, name="jarvis-cuda-receive", daemon=True)
        try:
            receiver.start()
            frame = received.result(timeout=timeout)
        except TimeoutError as exc:
            self.cancel()
            raise BackendError(timeout_code, "CUDA worker timed out", subsystem="cuda", retryable=True) from exc
        except (EOFError, OSError, ValueError, RuntimeError) as exc:
            raise BackendError("cuda.worker_crashed", "CUDA worker connection failed", subsystem="cuda", retryable=True) from exc
        finally:
            if receiver.ident is not None:
                receiver.join(timeout=5)
        if not isinstance(frame, dict) or not isinstance(frame.get("type"), str):
            raise BackendError("cuda.worker_protocol_error", "CUDA worker returned an invalid frame", subsystem="cuda")
        return frame

    @staticmethod
    def _raise_error(frame: dict[str, Any], request_id: str | None = None) -> None:
        code = frame.get("code")
        expected = {"type", "code", "retryable"}
        if request_id is not None:
            expected.add("request_id")
        if (set(frame) != expected or frame.get("request_id") != request_id
                or not isinstance(code, str) or not code or type(frame.get("retryable")) is not bool):
            raise BackendError("cuda.worker_protocol_error", "CUDA worker returned an invalid error", subsystem="cuda")
        raise BackendError(code, "CUDA worker failed", subsystem="cuda", retryable=frame["retryable"])

    def start(self) -> None:
        try:
            with self._lock:
                if self._cancelled:
                    raise BackendError("cuda.worker_cancelled", "CUDA worker was cancelled", subsystem="cuda")
                if self.alive:
                    return
                self._close_handles()
                started = self.clock()
                parent, child = self.context.Pipe(duplex=True)
                self._connection = parent
                try:
                    self._process = self.context.Process(target=_process_entry, args=(child, self.config), name="jarvis-cuda-asr", daemon=True)
                    self._process.start()
                finally:
                    child.close()
            frame = self._receive(self.config.worker_start_timeout_seconds, "cuda.worker_start_timeout")
            if frame["type"] == "error":
                self._raise_error(frame)
            if (set(frame) != {"type", "load_duration_seconds"} or frame["type"] != "ready"
                    or not _finite_metric(frame["load_duration_seconds"]) or frame["load_duration_seconds"] < 0):
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
        with self._lock:
            connection = self._connection
        if connection is None:
            raise BackendError("cuda.worker_cancelled", "CUDA worker was cancelled", subsystem="cuda")
        request_id = str(uuid.uuid4())
        try:
            connection.send({"type": "transcribe", "request_id": request_id, "sample_rate": 16_000, "audio": payload})
            frame = self._receive(self.config.transcription_timeout_seconds, "cuda.transcription_timeout")
        except (EOFError, OSError) as exc:
            self.cancel()
            raise BackendError("cuda.worker_crashed", "CUDA worker connection failed", subsystem="cuda", retryable=True) from exc
        except BackendError:
            self.cancel()
            raise
        if frame["type"] == "error":
            self._raise_error(frame, request_id)
        expected = {
            "type", "request_id", "text", "language_code", "language_score",
            "asr_score", "raw_avg_logprob", "inference_duration_seconds",
        }
        if frame["type"] != "result" or set(frame) != expected or frame.get("request_id") != request_id:
            raise BackendError("cuda.worker_protocol_error", "CUDA worker result did not match the request", subsystem="cuda")
        if not isinstance(frame["text"], str) or not frame["text"]:
            raise BackendError("cuda.worker_protocol_error", "CUDA worker returned invalid text", subsystem="cuda")
        if not isinstance(frame["language_code"], str) or not frame["language_code"]:
            raise BackendError("cuda.worker_protocol_error", "CUDA worker returned invalid language", subsystem="cuda")
        numeric = ("language_score", "asr_score", "raw_avg_logprob", "inference_duration_seconds")
        if (any(not _finite_metric(frame[name]) for name in numeric)
                or not 0 <= frame["language_score"] <= 1 or not 0 <= frame["asr_score"] <= 1
                or frame["raw_avg_logprob"] > 0 or frame["inference_duration_seconds"] < 0):
            raise BackendError("cuda.worker_protocol_error", "CUDA worker returned invalid metrics", subsystem="cuda")
        self.metrics.append({"event": "cuda_inference", "duration_seconds": float(frame["inference_duration_seconds"])})
        return TranscriptResult(
            unicodedata.normalize("NFC", frame["text"]),
            LanguageMetadata(frame["language_code"], float(frame["language_score"])),
            AsrScore(float(frame["asr_score"]), float(frame["raw_avg_logprob"])),
        )

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            process = self._process
            if process is not None:
                for method in ("terminate", "kill"):
                    if not process.is_alive():
                        break
                    getattr(process, method)()
                    process.join(timeout=5)
                if process.is_alive():
                    raise BackendError("cuda.worker_shutdown_timeout", "CUDA worker did not exit", subsystem="cuda")
            self._close_handles()

    def close(self) -> None:
        with self._lock:
            self._cancelled = True
            started = self.clock()
            process, connection = self._process, self._connection
            if process is not None and process.is_alive() and connection is not None:
                try:
                    connection.send({"type": "shutdown"})
                    frame = self._receive(min(5.0, self.config.worker_start_timeout_seconds), "cuda.worker_shutdown_timeout")
                    if frame != {"type": "shutdown.complete"}:
                        raise BackendError("cuda.worker_protocol_error", "CUDA worker shutdown was not acknowledged", subsystem="cuda")
                except (BackendError, EOFError, OSError):
                    pass
                else:
                    process.join(timeout=5)
            self.cancel()
            self.metrics.append({"event": "worker_shutdown", "duration_seconds": self.clock() - started})

    def _close_handles(self) -> None:
        try:
            if self._connection is not None:
                self._connection.close()
        finally:
            self._connection = None
            if self._process is not None:
                self._process.close()
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
        self._cancel_generation = 0
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

    def _new_session(self, generation: int) -> CudaWorkerSession:
        self._admit()
        session = self.session_factory(self.worker_config)
        try:
            with self._lock:
                if generation != self._cancel_generation:
                    raise BackendError("cuda.worker_cancelled", "CUDA worker was cancelled", subsystem="cuda")
                self._active_session = session
                if self.residency == "warm":
                    self._warm_session = session
            session.start()
            with self._lock:
                if generation != self._cancel_generation:
                    raise BackendError("cuda.worker_cancelled", "CUDA worker was cancelled", subsystem="cuda")
                if self.residency == "warm":
                    self._warm_started_once = True
            self._record_vram("vram_after_model_load")
            return session
        except BaseException:
            session.cancel()
            with self._lock:
                if self._active_session is session:
                    self._active_session = None
                if self._warm_session is session:
                    self._warm_session = None
            self.metrics.extend(session.metrics)
            raise

    def _record_vram(self, event: str) -> None:
        try:
            device = next(item for item in self.inventory.devices() if item.index == self.worker_config.device_index)
        except (BackendError, StopIteration):
            return
        self.metrics.append({"event": event, "value_mb": device.free_vram_mb})

    def transcribe(self, audio: object) -> TranscriptResult:
        if not isinstance(audio, AudioChunk) or audio.sample_rate != 16_000:
            raise BackendError("asr.invalid_audio", "ASR requires mono 16-kHz audio", subsystem="asr")
        with self._lock:
            generation = self._cancel_generation
            session = self._warm_session if self.residency == "warm" else None
            if session is None and self.residency == "warm" and self._warm_started_once:
                if self._warm_restarts >= self.warm_restart_limit:
                    raise BackendError("cuda.warm_restart_limit", "CUDA warm worker restart limit reached", subsystem="cuda", retryable=True)
                self._warm_restarts += 1
            if session is not None:
                self._active_session = session
        if session is None:
            session = self._new_session(generation)
        try:
            return session.transcribe(audio)
        except BackendError:
            session.cancel()
            with self._lock:
                if self._warm_session is session:
                    self._warm_session = None
            raise
        finally:
            if self.residency == "turn_scoped":
                session.close()  # Process exit precedes result publication to the caller.
            self.metrics.extend(session.metrics)
            session.metrics.clear()
            if self.residency == "turn_scoped":
                self._record_vram("vram_after_worker_exit")
            with self._lock:
                if self._active_session is session:
                    self._active_session = None

    def cancel_current(self) -> None:
        with self._lock:
            self._cancel_generation += 1
            session = self._active_session
        if session is not None:
            session.cancel()
            with self._lock:
                if self._warm_session is session:
                    self._warm_session = None
                if self._active_session is session:
                    self._active_session = None

    def close(self) -> None:
        with self._lock:
            self._cancel_generation += 1
            sessions = {id(item): item for item in (self._active_session, self._warm_session) if item is not None}
        failure: BaseException | None = None
        for session in sessions.values():
            try:
                session.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
            else:
                with self._lock:
                    if self._active_session is session:
                        self._active_session = None
                    if self._warm_session is session:
                        self._warm_session = None
        if failure is not None:
            raise failure
