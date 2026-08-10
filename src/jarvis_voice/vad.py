"""VAD boundaries, deterministic fake, and manifest-gated Silero adapter."""

from __future__ import annotations

import hashlib
from array import array
from pathlib import Path
from typing import Callable, Iterable

from .backends import AudioChunk, VadResult
from .errors import BackendError


def validate_artifact(path: str | None, size: int | None, sha256: str | None) -> tuple[bool, str]:
    if not path or size is None or not sha256:
        return False, "vad.artifact_manifest_required"
    artifact = Path(path).expanduser()
    if not artifact.is_file():
        return False, "vad.artifact_missing"
    try:
        if artifact.stat().st_size != size:
            return False, "vad.artifact_size_mismatch"
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False, "vad.artifact_unreadable"
    if digest.hexdigest() != sha256.lower():
        return False, "vad.artifact_hash_mismatch"
    return True, "vad.artifact_valid"


class FakeVadBackend:
    def __init__(self, *, start_sample: int | None = 0, end_sample: int | None = None) -> None:
        self.start_sample = start_sample
        self.end_sample = end_sample

    def trim(self, audio: AudioChunk) -> VadResult:
        if self.start_sample is None:
            return VadResult(False, AudioChunk(array("f"), audio.sample_rate))
        end = len(audio.samples) if self.end_sample is None else self.end_sample
        trimmed = array("f", audio.samples[self.start_sample:end])
        return VadResult(bool(trimmed), AudioChunk(trimmed, audio.sample_rate), self.start_sample, end)


class SileroOnnxVad:
    """The ONNX tensor contract remains blocked until native Windows verification."""

    def __init__(self, *, artifact_path: str, size: int, sha256: str, threshold: float, minimum_speech_ms: int, padding_ms: int, detector: Callable[[AudioChunk], Iterable[tuple[int, int, float]]] | None = None) -> None:
        ok, code = validate_artifact(artifact_path, size, sha256)
        if not ok:
            raise BackendError(code, "Silero ONNX artifact validation failed", subsystem="vad")
        self.artifact_path = str(Path(artifact_path).expanduser())
        self.threshold = threshold
        self.minimum_speech_ms = minimum_speech_ms
        self.padding_ms = padding_ms
        self._detector = detector
        self._session = None
        self._numpy = None

    def _load_runtime(self):
        if self._session is not None and self._numpy is not None:
            return self._numpy, self._session
        try:
            import numpy  # type: ignore[import-not-found]
            import onnxruntime  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendError("vad.runtime_missing", "ONNX Runtime is not installed", subsystem="vad") from exc
        self._numpy = numpy
        self._session = onnxruntime.InferenceSession(self.artifact_path, providers=["CPUExecutionProvider"])
        return self._numpy, self._session

    def _detect(self, audio: AudioChunk) -> list[tuple[int, int, float]]:
        if self._detector is not None:
            return list(self._detector(audio))
        numpy, session = self._load_runtime()
        input_names = {item.name for item in session.get_inputs()}
        state = numpy.zeros((2, 1, 128), dtype=numpy.float32)
        hidden = numpy.zeros((2, 1, 64), dtype=numpy.float32)
        cell = numpy.zeros((2, 1, 64), dtype=numpy.float32)
        sample_rate = numpy.array(audio.sample_rate, dtype=numpy.int64)
        regions: list[tuple[int, int, float]] = []
        active_start: int | None = None
        active_peak = 0.0
        chunk_size = 512 if audio.sample_rate == 16_000 else 256
        context_size = 64 if audio.sample_rate == 16_000 else 32
        context = numpy.zeros((1, context_size), dtype=numpy.float32)
        values = numpy.asarray(audio.samples, dtype=numpy.float32)
        for start in range(0, len(values), chunk_size):
            chunk = values[start : start + chunk_size]
            if len(chunk) < chunk_size:
                chunk = numpy.pad(chunk, (0, chunk_size - len(chunk)))
            # The pinned Silero 6.2.1 ONNX wrapper prepends 64 samples of
            # rolling context at 16 kHz (32 at 8 kHz) to every model frame.
            # State alone is not sufficient; omitting context suppresses real
            # speech probabilities while still producing structurally valid
            # model output.
            model_input = numpy.concatenate((context, chunk.reshape(1, -1)), axis=1)
            feeds: dict[str, object] = {"input": model_input, "sr": sample_rate}
            if "state" in input_names:
                feeds["state"] = state
            elif {"h", "c"}.issubset(input_names):
                feeds.update({"h": hidden, "c": cell})
            else:
                raise BackendError("vad.unsupported_model_contract", "Silero ONNX inputs are unsupported", subsystem="vad")
            outputs = session.run(None, feeds)
            probability = float(numpy.asarray(outputs[0]).reshape(-1)[0])
            if "state" in input_names:
                state = outputs[1]
            else:
                hidden, cell = outputs[1], outputs[2]
            context = model_input[:, -context_size:]
            if probability >= self.threshold:
                if active_start is None:
                    active_start = start
                active_peak = max(active_peak, probability)
            elif active_start is not None:
                regions.append((active_start, min(start, len(values)), active_peak))
                active_start, active_peak = None, 0.0
        if active_start is not None:
            regions.append((active_start, len(values), active_peak))
        return regions

    def trim(self, audio: AudioChunk) -> VadResult:
        regions = [item for item in self._detect(audio) if item[2] >= self.threshold]
        minimum = round(audio.sample_rate * self.minimum_speech_ms / 1000)
        regions = [item for item in regions if item[1] - item[0] >= minimum]
        if not regions:
            return VadResult(False, AudioChunk(array("f"), audio.sample_rate))
        padding = round(audio.sample_rate * self.padding_ms / 1000)
        start = max(0, regions[0][0] - padding)
        end = min(len(audio.samples), regions[-1][1] + padding)
        return VadResult(True, AudioChunk(array("f", audio.samples[start:end]), audio.sample_rate), start, end)
