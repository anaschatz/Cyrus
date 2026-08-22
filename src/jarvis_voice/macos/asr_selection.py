"""Explicit macOS ASR selection without implicit model conversion or download."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ..asr import FasterWhisperCpuBackend
from ..backends import AsrBackend
from ..config import VoiceConfig
from ..errors import BackendError
from ..manifest import model_manifest_for_profile
from .whisper_cpp import WhisperCppSidecarBackend


def with_faster_whisper_cpu_smoke(config: VoiceConfig, model_directory: str) -> VoiceConfig:
    """Return an in-memory, explicit CPU-smoke configuration.

    The helper never copies, converts, downloads, or loads the model. Requiring an
    absolute directory prevents a later source-run from depending on its cwd.
    """

    path = Path(model_directory).expanduser()
    if not path.is_absolute():
        raise BackendError(
            "asr.absolute_model_directory_required",
            "The macOS CPU-smoke model directory must be absolute",
            subsystem="asr",
        )
    return replace(
        config,
        whisper_model_directory=str(path),
        asr=replace(config.asr, execution_profile="cpu"),
        macos=replace(config.macos, asr_backend="faster_whisper_cpu_smoke"),
    )


def create_macos_asr_backend(
    config: VoiceConfig,
    *,
    cpu_factory: Callable[..., AsrBackend] = FasterWhisperCpuBackend,
    whisper_cpp_factory: Callable[..., AsrBackend] = WhisperCppSidecarBackend,
) -> AsrBackend:
    """Construct only the explicitly selected local ASR backend."""

    if config.macos.asr_backend == "faster_whisper_cpu_smoke":
        if not config.whisper_model_directory:
            raise BackendError(
                "asr.model_directory_required",
                "The faster-whisper CPU-smoke profile requires a local model directory",
                subsystem="asr",
            )
        return cpu_factory(
            config.whisper_model_directory,
            compute_type=config.asr.cpu_compute_type,
            manifest=model_manifest_for_profile(config.asr.model_profile),
        )
    if config.macos.asr_backend == "whisper_cpp":
        return whisper_cpp_factory(
            config.macos.whisper_cpp,
            maximum_audio_seconds=config.maximum_utterance_seconds,
        )
    raise BackendError("asr.macos_backend_unsupported", "The configured macOS ASR backend is unsupported", subsystem="asr")
