"""Explicit CPU/CUDA ASR selection with one-shot CPU fallback."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .asr import FasterWhisperCpuBackend
from .backends import AsrBackend, TranscriptResult
from .config import VoiceConfig
from .errors import BackendError
from .gpu import GpuInventory, NvidiaSmiInventory


class FallbackAsrBackend:
    """Try CUDA once, then the same validated audio on the persistent CPU backend."""

    def __init__(self, cuda_backend: AsrBackend, cpu_backend: AsrBackend, *, enabled: bool, fallbackable_codes: frozenset[str], clock: Callable[[], float] = time.perf_counter) -> None:
        self.cuda_backend = cuda_backend
        self.cpu_backend = cpu_backend
        self.enabled = enabled
        self.fallbackable_codes = fallbackable_codes
        self.clock = clock
        self.metrics: list[dict[str, float | int | str]] = []
        self.fallbacks = 0

    def transcribe(self, audio: object) -> TranscriptResult:
        try:
            return self.cuda_backend.transcribe(audio)
        except BackendError as exc:
            if not self.enabled or exc.code not in self.fallbackable_codes:
                raise
            started = self.clock()
            self.fallbacks += 1
            self.metrics.append({"event": "cuda_cpu_fallback", "code": exc.code})
            try:
                return self.cpu_backend.transcribe(audio)
            finally:
                self.metrics.append({"event": "cpu_fallback", "duration_seconds": self.clock() - started})

    def cancel_current(self) -> None:
        cancel = getattr(self.cuda_backend, "cancel_current", None)
        if cancel is not None:
            cancel()

    def close(self) -> None:
        close = getattr(self.cuda_backend, "close", None)
        if close is not None:
            close()


class AsrBackendFactory:
    """CPU construction is CUDA-import-free; CUDA modules are loaded on demand."""

    def __init__(
        self,
        *,
        inventory_factory: Callable[[], GpuInventory] = NvidiaSmiInventory,
        cpu_factory: Callable[..., AsrBackend] = FasterWhisperCpuBackend,
        cuda_backend_factory: Callable[..., AsrBackend] | None = None,
        cuda_session_factory: Callable[[Any], Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.inventory_factory = inventory_factory
        self.cpu_factory = cpu_factory
        self.cuda_backend_factory = cuda_backend_factory
        self.cuda_session_factory = cuda_session_factory
        self.clock = clock

    def create(self, config: VoiceConfig) -> AsrBackend:
        model_directory = config.whisper_model_directory
        if not model_directory:
            raise BackendError("asr.model_directory_required", "Whisper model directory is required", subsystem="asr")
        cpu = self.cpu_factory(model_directory, compute_type=config.asr.cpu_compute_type)
        if config.asr.execution_profile == "cpu":
            return cpu

        # Deliberately lazy: choosing CPU never imports or initializes CUDA code.
        from .cuda_worker import (
            FALLBACKABLE_CUDA_ERRORS,
            CudaFasterWhisperBackend,
            CudaWorkerConfig,
        )

        cuda = config.asr.cuda
        if cuda.device_index is None:
            raise BackendError("config.cuda_device_required", "CUDA profile requires an explicit device index", subsystem="config")
        worker_config = CudaWorkerConfig(
            model_directory=model_directory,
            device_index=cuda.device_index,
            compute_type=cuda.compute_type,
            worker_start_timeout_seconds=cuda.worker_start_timeout_seconds,
            transcription_timeout_seconds=cuda.transcription_timeout_seconds,
            maximum_audio_bytes=round(config.maximum_utterance_seconds * 16_000 * 4),
        )
        backend_factory = self.cuda_backend_factory or CudaFasterWhisperBackend
        kwargs: dict[str, object] = {
            "inventory": self.inventory_factory(),
            "residency": cuda.residency,
            "minimum_free_vram_mb": cuda.minimum_free_vram_mb,
            "reserve_for_other_workloads_mb": cuda.reserve_for_other_workloads_mb,
            "warm_restart_limit": cuda.warm_restart_limit,
            "clock": self.clock,
        }
        if self.cuda_session_factory is not None:
            kwargs["session_factory"] = self.cuda_session_factory
        gpu_backend = backend_factory(worker_config, **kwargs)
        return FallbackAsrBackend(
            gpu_backend,
            cpu,
            enabled=cuda.fallback_to_cpu,
            fallbackable_codes=FALLBACKABLE_CUDA_ERRORS,
            clock=self.clock,
        )
