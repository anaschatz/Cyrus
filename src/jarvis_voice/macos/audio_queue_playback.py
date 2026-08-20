"""Native macOS AudioQueue playback for Pocket TTS PCM streaming."""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

from ..errors import BackendError


_AUDIO_TOOLBOX = "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox"
_LINEAR_PCM = int.from_bytes(b"lpcm", "big")
_SIGNED_INTEGER = 1 << 2
_PACKED = 1 << 3


class _AudioStreamBasicDescription(ctypes.Structure):
    _fields_ = [
        ("mSampleRate", ctypes.c_double),
        ("mFormatID", ctypes.c_uint32),
        ("mFormatFlags", ctypes.c_uint32),
        ("mBytesPerPacket", ctypes.c_uint32),
        ("mFramesPerPacket", ctypes.c_uint32),
        ("mBytesPerFrame", ctypes.c_uint32),
        ("mChannelsPerFrame", ctypes.c_uint32),
        ("mBitsPerChannel", ctypes.c_uint32),
        ("mReserved", ctypes.c_uint32),
    ]


class _AudioQueueBuffer(ctypes.Structure):
    _fields_ = [
        ("mAudioDataBytesCapacity", ctypes.c_uint32),
        ("mAudioData", ctypes.c_void_p),
        ("mAudioDataByteSize", ctypes.c_uint32),
        ("mUserData", ctypes.c_void_p),
        ("mPacketDescriptionCapacity", ctypes.c_uint32),
        ("mPacketDescriptions", ctypes.c_void_p),
        ("mPacketDescriptionCount", ctypes.c_uint32),
    ]


_AudioQueueRef = ctypes.c_void_p
_AudioQueueBufferRef = ctypes.POINTER(_AudioQueueBuffer)
_OutputCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, _AudioQueueRef, _AudioQueueBufferRef
)


class AudioQueueApi(Protocol):
    def create(self, sample_rate: int, callback: Callable[[Any], None]) -> Any: ...
    def allocate(self, queue: Any, size: int) -> Any: ...
    def enqueue(self, queue: Any, buffer: Any, pcm16: bytes) -> None: ...
    def prime(self, queue: Any) -> None: ...
    def start(self, queue: Any) -> None: ...
    def stop(self, queue: Any, *, immediate: bool) -> None: ...
    def dispose(self, queue: Any, *, immediate: bool) -> None: ...


class _NativeAudioQueueApi:
    """Small ctypes boundary around the dependency-free AudioToolbox API."""

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise BackendError(
                "tts.pocket.audio_queue_macos_only",
                "Native AudioQueue playback requires macOS",
                subsystem="tts",
            )
        self._toolbox = ctypes.CDLL(_AUDIO_TOOLBOX)
        self._callbacks: dict[int, Any] = {}
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        toolbox = self._toolbox
        toolbox.AudioQueueNewOutput.argtypes = [
            ctypes.POINTER(_AudioStreamBasicDescription),
            _OutputCallback,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_AudioQueueRef),
        ]
        toolbox.AudioQueueNewOutput.restype = ctypes.c_int32
        toolbox.AudioQueueAllocateBuffer.argtypes = [
            _AudioQueueRef,
            ctypes.c_uint32,
            ctypes.POINTER(_AudioQueueBufferRef),
        ]
        toolbox.AudioQueueAllocateBuffer.restype = ctypes.c_int32
        toolbox.AudioQueueEnqueueBuffer.argtypes = [
            _AudioQueueRef,
            _AudioQueueBufferRef,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        toolbox.AudioQueueEnqueueBuffer.restype = ctypes.c_int32
        toolbox.AudioQueuePrime.argtypes = [
            _AudioQueueRef,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        toolbox.AudioQueuePrime.restype = ctypes.c_int32
        toolbox.AudioQueueStart.argtypes = [_AudioQueueRef, ctypes.c_void_p]
        toolbox.AudioQueueStart.restype = ctypes.c_int32
        toolbox.AudioQueueStop.argtypes = [_AudioQueueRef, ctypes.c_ubyte]
        toolbox.AudioQueueStop.restype = ctypes.c_int32
        toolbox.AudioQueueDispose.argtypes = [_AudioQueueRef, ctypes.c_ubyte]
        toolbox.AudioQueueDispose.restype = ctypes.c_int32

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status:
            raise OSError(int(status), f"AudioQueue {operation} failed")

    def create(self, sample_rate: int, callback: Callable[[Any], None]) -> Any:
        queue = _AudioQueueRef()

        @_OutputCallback
        def native_callback(
            _user_data: Any, _queue: Any, buffer: _AudioQueueBufferRef
        ) -> None:
            callback(buffer)

        description = _AudioStreamBasicDescription(
            float(sample_rate),
            _LINEAR_PCM,
            _SIGNED_INTEGER | _PACKED,
            2,
            1,
            2,
            1,
            16,
            0,
        )
        self._check(
            self._toolbox.AudioQueueNewOutput(
                ctypes.byref(description),
                native_callback,
                None,
                None,
                None,
                0,
                ctypes.byref(queue),
            ),
            "create",
        )
        self._callbacks[int(queue.value)] = native_callback
        return queue

    def allocate(self, queue: Any, size: int) -> Any:
        buffer = _AudioQueueBufferRef()
        self._check(
            self._toolbox.AudioQueueAllocateBuffer(
                queue, size, ctypes.byref(buffer)
            ),
            "allocate buffer",
        )
        return buffer

    def enqueue(self, queue: Any, buffer: Any, pcm16: bytes) -> None:
        ctypes.memmove(buffer.contents.mAudioData, pcm16, len(pcm16))
        buffer.contents.mAudioDataByteSize = len(pcm16)
        self._check(
            self._toolbox.AudioQueueEnqueueBuffer(queue, buffer, 0, None),
            "enqueue buffer",
        )

    def prime(self, queue: Any) -> None:
        self._check(self._toolbox.AudioQueuePrime(queue, 0, None), "prime")

    def start(self, queue: Any) -> None:
        self._check(self._toolbox.AudioQueueStart(queue, None), "start")

    def stop(self, queue: Any, *, immediate: bool) -> None:
        self._check(
            self._toolbox.AudioQueueStop(queue, int(immediate)), "stop"
        )

    def dispose(self, queue: Any, *, immediate: bool) -> None:
        try:
            self._check(
                self._toolbox.AudioQueueDispose(queue, int(immediate)), "dispose"
            )
        finally:
            if getattr(queue, "value", None):
                self._callbacks.pop(int(queue.value), None)


class MacAudioQueueStreamingPlayback:
    """Bounded native PCM16 playback with synchronous hard cancellation."""

    def __init__(
        self,
        *,
        api: AudioQueueApi | None = None,
        buffer_count: int = 4,
        buffer_bytes: int = 8_192,
        maximum_staging_bytes: int = 384_000,
        write_timeout_seconds: float = 1.0,
        finish_timeout_seconds: float = 15.0,
    ) -> None:
        if (
            buffer_count < 2
            or buffer_bytes < 512
            or buffer_bytes % 2
            or maximum_staging_bytes < buffer_bytes
            or write_timeout_seconds <= 0
            or finish_timeout_seconds <= 0
        ):
            raise ValueError("AudioQueue playback bounds are invalid")
        self._api = api or _NativeAudioQueueApi()
        self._buffer_count = buffer_count
        self._buffer_bytes = buffer_bytes
        self._maximum_staging_bytes = maximum_staging_bytes
        self._write_timeout_seconds = write_timeout_seconds
        self._finish_timeout_seconds = finish_timeout_seconds
        self._condition = threading.Condition(threading.RLock())
        self._queue: Any | None = None
        self._available: deque[Any] = deque()
        self._staging = bytearray()
        self._inflight = 0
        self._started = False
        self._finishing = False
        self._aborted = False
        self._disposed = False
        self._callback_error: BaseException | None = None

    def start(self, sample_rate: int) -> None:
        if sample_rate < 8_000 or sample_rate > 192_000:
            raise BackendError(
                "tts.pocket.playback_sample_rate_invalid",
                "Pocket TTS playback sample rate is invalid",
                subsystem="tts",
            )
        try:
            queue = self._api.create(sample_rate, self._on_buffer_complete)
            buffers = [
                self._api.allocate(queue, self._buffer_bytes)
                for _ in range(self._buffer_count)
            ]
        except Exception as exc:
            raise BackendError(
                "tts.pocket.playback_open_failed",
                "Pocket TTS native playback could not be opened",
                subsystem="tts",
            ) from exc
        with self._condition:
            if self._aborted:
                dispose = True
            else:
                self._queue = queue
                self._available.extend(buffers)
                dispose = False
        if dispose:
            self._api.dispose(queue, immediate=True)

    def _raise_callback_error_locked(self) -> None:
        if self._callback_error is not None:
            raise BackendError(
                "tts.pocket.playback_write_failed",
                "Pocket TTS native playback failed while streaming audio",
                subsystem="tts",
            ) from self._callback_error

    def _pump_locked(self) -> None:
        queue = self._queue
        if queue is None or self._aborted:
            return
        while self._available and (
            len(self._staging) >= self._buffer_bytes
            or (self._finishing and self._staging)
        ):
            size = min(len(self._staging), self._buffer_bytes)
            size -= size % 2
            if not size:
                break
            pcm16 = bytes(self._staging[:size])
            del self._staging[:size]
            buffer = self._available.popleft()
            try:
                self._api.enqueue(queue, buffer, pcm16)
            except BaseException:
                self._available.appendleft(buffer)
                raise
            self._inflight += 1
        if not self._started and self._inflight and (
            self._inflight >= self._buffer_count or self._finishing
        ):
            self._api.prime(queue)
            self._api.start(queue)
            self._started = True
        self._condition.notify_all()

    def _on_buffer_complete(self, buffer: Any) -> None:
        with self._condition:
            if self._aborted or self._disposed:
                return
            self._inflight = max(0, self._inflight - 1)
            self._available.append(buffer)
            try:
                self._pump_locked()
            except BaseException as exc:
                self._callback_error = exc
            self._condition.notify_all()

    def write(self, pcm16: bytes) -> None:
        if not pcm16 or len(pcm16) % 2:
            raise BackendError(
                "tts.pocket.playback_audio_invalid",
                "Pocket TTS playback received invalid PCM16 audio",
                subsystem="tts",
            )
        deadline = time.monotonic() + self._write_timeout_seconds
        with self._condition:
            if self._queue is None and not self._aborted:
                raise BackendError(
                    "tts.pocket.playback_not_started",
                    "Pocket TTS native playback is not ready",
                    subsystem="tts",
                )
            while (
                not self._aborted
                and len(self._staging) + len(pcm16) > self._maximum_staging_bytes
            ):
                self._raise_callback_error_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendError(
                        "tts.pocket.playback_queue_timeout",
                        "Pocket TTS native playback queue stopped draining",
                        subsystem="tts",
                    )
                self._condition.wait(remaining)
            if self._aborted:
                return
            self._raise_callback_error_locked()
            self._staging.extend(pcm16)
            try:
                self._pump_locked()
            except Exception as exc:
                raise BackendError(
                    "tts.pocket.playback_write_failed",
                    "Pocket TTS native playback failed while writing audio",
                    subsystem="tts",
                ) from exc

    def finish(self) -> None:
        deadline = time.monotonic() + self._finish_timeout_seconds
        with self._condition:
            if self._queue is None or self._aborted:
                return
            self._finishing = True
            try:
                self._pump_locked()
            except Exception as exc:
                raise BackendError(
                    "tts.pocket.playback_finish_failed",
                    "Pocket TTS native playback could not finish",
                    subsystem="tts",
                ) from exc
            while not self._aborted and (self._staging or self._inflight):
                self._raise_callback_error_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendError(
                        "tts.pocket.playback_finish_timeout",
                        "Pocket TTS native playback did not finish",
                        subsystem="tts",
                    )
                self._condition.wait(remaining)
            if self._aborted:
                return
            self._raise_callback_error_locked()
            queue = self._queue
            self._queue = None
            self._disposed = True
        if queue is not None:
            try:
                self._api.stop(queue, immediate=True)
            finally:
                self._api.dispose(queue, immediate=True)

    def abort(self) -> None:
        with self._condition:
            if self._aborted:
                return
            self._aborted = True
            self._staging.clear()
            queue = self._queue
            self._queue = None
            self._condition.notify_all()
        if queue is None:
            return
        try:
            self._api.stop(queue, immediate=True)
        finally:
            self._api.dispose(queue, immediate=True)
            with self._condition:
                self._disposed = True
                self._condition.notify_all()


__all__ = ["MacAudioQueueStreamingPlayback"]
