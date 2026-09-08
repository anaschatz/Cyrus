"""Warm WASAPI capture with a minimal PortAudio callback."""

from __future__ import annotations

import sys
from typing import Any

from ..audio import BoundedCaptureBuffer
from ..backends import AudioChunk
from ..errors import BackendError


class WasapiAudioCapture:
    def __init__(self, *, queue_capacity: int, maximum_utterance_seconds: float, device: int | str | None = None) -> None:
        self.queue_capacity = queue_capacity
        self.maximum_utterance_seconds = maximum_utterance_seconds
        self.device = device
        self._stream: Any | None = None
        self._buffer: BoundedCaptureBuffer | None = None

    def start(self) -> None:
        if self._stream is not None:
            return
        if sys.platform != "win32":
            raise BackendError("audio.wasapi_windows_only", "WASAPI requires Windows", subsystem="audio")
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendError("audio.runtime_missing", "sounddevice is not installed", subsystem="audio") from exc
        try:
            device_info = sd.query_devices(self.device, "input")
            sample_rate = int(device_info["default_samplerate"])
            self._buffer = BoundedCaptureBuffer(
                capacity=self.queue_capacity,
                sample_rate=sample_rate,
                maximum_samples=round(sample_rate * self.maximum_utterance_seconds),
            )
            settings = sd.WasapiSettings(auto_convert=True)
            self._stream = sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=0,
                device=self.device,
                channels=1,
                dtype="float32",
                latency="low",
                extra_settings=settings,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self.stop()
            raise BackendError("audio.wasapi_start_failed", "WASAPI input could not start", subsystem="audio") from exc

    def _callback(self, indata: bytes, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info, status
        buffer = self._buffer
        if buffer is not None:
            buffer.publish_from_callback(indata)

    def begin_turn(self) -> None:
        if self._buffer is None:
            raise BackendError("audio.not_started", "Audio capture is not started", subsystem="audio")
        self._buffer.begin_turn()

    def end_turn(self) -> AudioChunk:
        if self._buffer is None:
            raise BackendError("audio.not_started", "Audio capture is not started", subsystem="audio")
        return self._buffer.end_turn()

    def abort_turn(self) -> None:
        if self._buffer is not None:
            self._buffer.abort_turn()

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        self._buffer = None
        if stream is not None:
            for operation in (stream.stop, stream.close):
                try:
                    operation()
                except Exception:
                    pass
