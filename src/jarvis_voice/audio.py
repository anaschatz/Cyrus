"""Bounded callback buffering and worker-side audio conversion."""

from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .backends import AudioChunk
from .errors import BackendError


@dataclass(frozen=True, slots=True)
class CaptureStats:
    accepted_frames: int
    discarded_frames: int
    overflowed: bool


class BoundedCaptureBuffer:
    """A callback only copies bytes or fails immediately; it never processes audio."""

    def __init__(self, *, capacity: int, sample_rate: int, maximum_samples: int | None = None) -> None:
        if capacity < 1 or sample_rate < 1:
            raise ValueError("capacity and sample_rate must be positive")
        self.capacity = capacity
        self.sample_rate = sample_rate
        self.maximum_samples = maximum_samples
        self._frames: deque[bytes] = deque()
        self._capturing = False
        self._overflowed = False
        self._accepted = 0
        self._discarded = 0
        self._sample_count = 0

    def begin_turn(self) -> None:
        self._frames.clear()
        self._overflowed = False
        self._accepted = 0
        self._sample_count = 0
        self._capturing = True

    def publish_from_callback(self, frames: bytes) -> bool:
        if not self._capturing:
            self._discarded += 1
            return False
        incoming_samples = len(frames) // 4
        if (
            self._overflowed
            or len(self._frames) >= self.capacity
            or (self.maximum_samples is not None and self._sample_count + incoming_samples > self.maximum_samples)
        ):
            self._overflowed = True
            self._capturing = False
            self._frames.clear()
            return False
        self._frames.append(bytes(frames))
        self._sample_count += incoming_samples
        self._accepted += 1
        return True

    def end_turn(self) -> AudioChunk:
        self._capturing = False
        if self._overflowed:
            self._frames.clear()
            raise BackendError("audio.capture_overflow", "Audio capture queue overflowed", subsystem="audio")
        samples = array("f")
        for frame in self._frames:
            samples.frombytes(frame)
        self._frames.clear()
        return AudioChunk(samples, self.sample_rate)

    def abort_turn(self) -> None:
        self._capturing = False
        self._frames.clear()

    @property
    def stats(self) -> CaptureStats:
        return CaptureStats(self._accepted, self._discarded, self._overflowed)


def resample_audio(audio: AudioChunk, *, target_rate: int = 16_000, resampler: Callable[[array, int, int], array] | None = None) -> AudioChunk:
    if audio.sample_rate == target_rate:
        return audio
    if resampler is not None:
        return AudioChunk(resampler(audio.samples, audio.sample_rate, target_rate), target_rate)
    try:
        import numpy  # type: ignore[import-not-found]
        import soxr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BackendError("audio.resampler_missing", "SoXR runtime is not installed", subsystem="audio") from exc
    converted = soxr.resample(numpy.asarray(audio.samples, dtype=numpy.float32), audio.sample_rate, target_rate)
    return AudioChunk(array("f", converted.tolist()), target_rate)
