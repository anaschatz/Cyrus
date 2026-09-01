"""Manifest-gated, CPU-only faster-whisper backend."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable

from .backends import AsrScore, AudioChunk, LanguageMetadata, TranscriptResult
from .errors import BackendError
from .manifest import BASE_MODEL_MANIFEST, ModelManifest, validate_model_directory

BILINGUAL_LANGUAGE_CANDIDATES = ("el", "en")
BILINGUAL_RESOLUTION_THRESHOLD = 0.8
ISOLATED_TURN_OPTIONS = {
    "vad_filter": False,
    "task": "transcribe",
    "beam_size": 5,
    "condition_on_previous_text": False,
}


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


def decode_bilingual_transcript(
    model: Any,
    samples: Any,
) -> tuple[str, str, float, list[Any]]:
    """Decode one isolated PTT turn with a deterministic Greek/English tie-break."""

    segments_iter, info = model.transcribe(samples, **ISOLATED_TURN_OPTIONS)
    segments = list(segments_iter)
    detected_language = str(getattr(info, "language", "und"))
    detected_score = float(getattr(info, "language_probability", 0.0))
    text = unicodedata.normalize(
        "NFC",
        "".join(str(segment.text) for segment in segments).strip(),
    )
    if detected_score < BILINGUAL_RESOLUTION_THRESHOLD:
        candidates: list[tuple[float, str, list[Any], str]] = []
        for language in BILINGUAL_LANGUAGE_CANDIDATES:
            candidate_iter, _candidate_info = model.transcribe(
                samples,
                **ISOLATED_TURN_OPTIONS,
                language=language,
            )
            candidate_segments = list(candidate_iter)
            candidate_text = unicodedata.normalize(
                "NFC",
                "".join(
                    str(segment.text) for segment in candidate_segments
                ).strip(),
            )
            candidate_raw, _candidate_score = weighted_logprob(candidate_segments)
            if candidate_text and math.isfinite(candidate_raw):
                candidates.append(
                    (candidate_raw, language, candidate_segments, candidate_text)
                )
        if candidates:
            chosen = max(candidates, key=lambda item: item[0])
            maximum_raw = max(item[0] for item in candidates)
            weights = [math.exp(item[0] - maximum_raw) for item in candidates]
            total_weight = sum(weights)
            chosen_index = candidates.index(chosen)
            detected_language = chosen[1]
            detected_score = weights[chosen_index] / total_weight
            segments = chosen[2]
            text = chosen[3]
    return text, detected_language, detected_score, segments


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
        text, detected_language, detected_score, segments = (
            decode_bilingual_transcript(self._model, samples)
        )
        if not text:
            raise BackendError("asr.empty_transcript", "ASR produced no transcript", subsystem="asr")
        raw, score = weighted_logprob(segments)
        return TranscriptResult(
            text=text,
            language=LanguageMetadata(detected_language, detected_score),
            asr_score=AsrScore(score, raw),
        )

    def close(self) -> None:
        """Release the persistent local model reference on runtime shutdown."""

        self._model = None
