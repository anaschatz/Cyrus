"""Non-blocking ASR coordination for cancellable CUDA turns."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .backends import AsrBackend, AudioChunk, TranscriptResult
from .errors import BackendError


@dataclass(frozen=True, slots=True)
class TranscriptionOutcome:
    turn_id: str
    result: TranscriptResult | None = None
    error: BackendError | None = None


class AsyncTranscriptionCoordinator:
    def __init__(
        self,
        backend: AsrBackend,
        *,
        executor_factory: Callable[[], Any] = lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-asr"),
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.backend = backend
        self.executor = executor_factory()
        self.clock = clock
        self.metrics: list[dict[str, float | str]] = []
        self._future: Future[TranscriptResult] | None = None
        self._turn_id: str | None = None
        self._started_at: float | None = None
        self._retired: list[Future[TranscriptResult]] = []

    @property
    def active(self) -> bool:
        return self._future is not None

    def submit(self, turn_id: str, audio: AudioChunk) -> None:
        if self._future is not None:
            raise BackendError("asr.transcription_busy", "A transcription is already active", subsystem="asr", retryable=True)
        self._turn_id = turn_id
        self._started_at = self.clock()
        self._future = self.executor.submit(self.backend.transcribe, audio)

    def cancel_current(self) -> None:
        future = self._future
        if future is None:
            return
        self._future = None
        self._turn_id = None
        self._started_at = None
        future.cancel()
        self._retired.append(future)
        cancel = getattr(self.backend, "cancel_current", None)
        if cancel is not None:
            cancel()

    def poll(self) -> TranscriptionOutcome | None:
        self._retired = [future for future in self._retired if not future.done()]
        future, turn_id = self._future, self._turn_id
        if future is None or turn_id is None or not future.done():
            return None
        started_at = self._started_at
        self._future = None
        self._turn_id = None
        self._started_at = None
        if started_at is not None:
            self.metrics.append({"event": "ptt_up_to_asr_result", "duration_seconds": self.clock() - started_at})
        try:
            return TranscriptionOutcome(turn_id, result=future.result())
        except BackendError as exc:
            return TranscriptionOutcome(turn_id, error=exc)
        except BaseException as exc:
            error = BackendError("asr.backend_failed", "ASR backend failed", subsystem="asr")
            error.__cause__ = exc
            return TranscriptionOutcome(turn_id, error=error)

    def close(self) -> None:
        try:
            self.cancel_current()
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
            close = getattr(self.backend, "close", None)
            if close is not None:
                close()
