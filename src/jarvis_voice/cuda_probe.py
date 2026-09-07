"""Explicit, local-only active CUDA readiness probe."""

from __future__ import annotations

import threading
import time
from array import array
from collections.abc import Callable

from .backends import AudioChunk
from .config import VoiceConfig
from .cuda_worker import CudaFasterWhisperBackend, CudaWorkerConfig
from .errors import BackendError
from .gpu import GpuInventory, NvidiaSmiInventory
from .manifest import model_manifest_for_profile


def run_active_cuda_probe(
    config: VoiceConfig,
    *,
    inventory: GpuInventory | None = None,
    backend_factory: Callable[..., CudaFasterWhisperBackend] = CudaFasterWhisperBackend,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    if config.asr.execution_profile != "cuda" or config.asr.cuda.device_index is None:
        raise BackendError("cuda.active_probe_requires_cuda_profile", "Active CUDA probe requires an explicit CUDA profile", subsystem="cuda")
    if not config.whisper_model_directory:
        raise BackendError("asr.model_directory_required", "Whisper model directory is required", subsystem="asr")
    cuda = config.asr.cuda
    inventory = inventory or NvidiaSmiInventory()
    worker_config = CudaWorkerConfig(
        config.whisper_model_directory,
        cuda.device_index,
        cuda.compute_type,
        cuda.worker_start_timeout_seconds,
        cuda.transcription_timeout_seconds,
        round(config.maximum_utterance_seconds * 16_000 * 4),
        model_manifest_for_profile(config.asr.model_profile),
    )
    backend = backend_factory(
        worker_config,
        inventory=inventory,
        residency="turn_scoped",
        minimum_free_vram_mb=cuda.minimum_free_vram_mb,
        reserve_for_other_workloads_mb=cuda.reserve_for_other_workloads_mb,
        warm_restart_limit=0,
        clock=clock,
    )
    samples: list[int] = []
    stop = threading.Event()

    def sample_vram() -> None:
        interval = cuda.vram_poll_interval_ms / 1000
        while not stop.wait(interval):
            try:
                device = next(item for item in inventory.devices() if item.index == cuda.device_index)
            except (BackendError, StopIteration):
                continue
            samples.append(device.free_vram_mb)

    sampler = threading.Thread(target=sample_vram, name="jarvis-vram-probe", daemon=True)
    empty_transcript = False
    try:
        sampler.start()
        started = clock()
        try:
            backend.transcribe(AudioChunk(array("f", [0.0] * 16_000), 16_000))
        except BackendError as exc:
            if exc.code != "asr.empty_transcript":
                raise
            empty_transcript = True
    finally:
        try:
            backend.close()
        finally:
            stop.set()
            if sampler.ident is not None:
                sampler.join(timeout=2)
    values = [int(item["value_mb"]) for item in backend.metrics if "value_mb" in item]
    return {
        "status": "ready",
        "code": "cuda.active_probe_succeeded",
        "duration_seconds": clock() - started,
        "empty_probe_transcript": empty_transcript,
        "vram_free_samples_mb": {
            "first": values[0] if values else None,
            "minimum_observed": min(samples + values) if samples or values else None,
            "after_worker_exit": next((int(item["value_mb"]) for item in reversed(backend.metrics) if item.get("event") == "vram_after_worker_exit"), None),
        },
        "vram_sampling_interval_ms": cuda.vram_poll_interval_ms,
        "peak_is_sampled_not_absolute": True,
        "metrics": list(backend.metrics),
        "transcript_exposed": False,
        "network_used": False,
    }
