"""Manifest-gated, CPU-only faster-whisper backend."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable

from .backends import AsrScore, AudioChunk, LanguageMetadata, TranscriptResult
from .errors import BackendError
from .manifest import BASE_MODEL_MANIFEST, ModelManifest, validate_model_directory


def weighted_logprob(segments: Iterable[Any]) -> tuple[float, float]:
    weighted_sum = 0.0
    token_count = 0
    for segment in segments:
        count = max(1, len(getattr(segment, "tokens", ()) or ()))
        weighted_sum += float(segment.avg_logprob) * count
        token_count += count
    if token_count == 0:
        return float("-inf"), 0.0
    raw = weighted_sum / token_count
    return raw, max(0.0, min(1.0, math.exp(raw)))


def choose_cpu_compute_type(requested: str, supported: Iterable[str]) -> str:
    available = set(supported)
    if requested in available:
        return requested
    for candidate in ("int8", "int8_float32", "float32", "int16"):
        if candidate in available:
            return candidate
    raise BackendError("asr.cpu_compute_type_unsupported", "No supported CPU compute type is available", subsystem="asr")


class FasterWhisperCpuBackend:
    def __init__(self, model_directory: str, *, compute_type: str = "int8", model_factory: Callable[..., Any] | None = None, manifest: ModelManifest = BASE_MODEL_MANIFEST) -> None:
        validation = validate_model_directory(model_directory, manifest)
        if not validation.ok:
            context = {"artifact_name": validation.artifact_name} if validation.artifact_name else None
            raise BackendError(validation.code, "Whisper model manifest validation failed", subsystem="asr", safe_context=context)
        self.model_directory = str(Path(model_directory).expanduser())
        self.compute_type = compute_type
        self._model_factory = model_factory
        self._model: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        factory = self._model_factory
        if factory is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore[import-not-found]
            except ImportError as exc:
                raise BackendError("asr.runtime_missing", "faster-whisper is not installed", subsystem="asr") from exc
            factory = WhisperModel
            try:
                import ctranslate2  # type: ignore[import-not-found]
                self.compute_type = choose_cpu_compute_type(self.compute_type, ctranslate2.get_supported_compute_types("cpu"))
            except ImportError:
                pass
        self._model = factory(self.model_directory, device="cpu", compute_type=self.compute_type, local_files_only=True)

    def transcribe(self, audio: object) -> TranscriptResult:
        if not isinstance(audio, AudioChunk) or audio.sample_rate != 16_000:
            raise BackendError("asr.invalid_audio", "ASR requires mono 16-kHz audio", subsystem="asr")
        self.load()
        try:
            import numpy  # type: ignore[import-not-found]
        except ImportError as exc:
            if self._model_factory is None:
                raise BackendError("asr.numpy_missing", "NumPy is not installed", subsystem="asr") from exc
            samples: Any = audio.samples
        else:
            samples = numpy.asarray(audio.samples, dtype=numpy.float32)
        segments_iter, info = self._model.transcribe(samples, vad_filter=False, task="transcribe")
        segments = list(segments_iter)
        text = unicodedata.normalize("NFC", "".join(str(segment.text) for segment in segments).strip())
        if not text:
            raise BackendError("asr.empty_transcript", "ASR produced no transcript", subsystem="asr")
        raw, score = weighted_logprob(segments)
        return TranscriptResult(
            text=text,
            language=LanguageMetadata(str(getattr(info, "language", "und")), float(getattr(info, "language_probability", 0.0))),
            asr_score=AsrScore(score, raw),
        )
