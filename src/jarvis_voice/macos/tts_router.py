"""Ordered locale routing across the Greek F1 and macOS system TTS backends."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Collection
from typing import Protocol

from ..backends import SpeechUnit
from ..errors import BackendError


class RoutedBackend(Protocol):
    def enqueue(self, unit: SpeechUnit) -> None: ...
    def cancel(self, new_epoch: int, reason: str) -> None: ...
    @property
    def is_idle(self) -> bool: ...
    @property
    def last_error(self) -> BackendError | None: ...
    def close(self) -> None: ...


class MacLocaleRoutingTtsBackend:
    """Serialize units while routing Greek to F1 and fallbacks to ``say``."""

    def __init__(
        self,
        *,
        greek_backend: RoutedBackend,
        fallback_backend: RoutedBackend,
        fallback_locales: Collection[str],
        queue_capacity: int = 64,
    ) -> None:
        if queue_capacity < 1 or "el-GR" in fallback_locales:
            raise BackendError("tts.router_config_invalid", "TTS locale router configuration is invalid", subsystem="tts")
        self.greek_backend = greek_backend
        self.fallback_backend = fallback_backend
        self.fallback_locales = frozenset(fallback_locales)
        self._queue: queue.Queue[SpeechUnit] = queue.Queue(maxsize=queue_capacity)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._cancelled_through_epoch = 0
        self._active: SpeechUnit | None = None
        self._error: BackendError | None = None
        self._thread = threading.Thread(target=self._worker, name="jarvis-macos-tts-router", daemon=True)
        self._thread.start()

    @property
    def warm_load_ms(self) -> float | None:
        value = getattr(self.greek_backend, "warm_load_ms", None)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def _backend_for(self, locale: str) -> RoutedBackend:
        if locale == "el-GR":
            return self.greek_backend
        if locale in self.fallback_locales:
            return self.fallback_backend
        raise BackendError("tts.locale_unsupported", "TTS locale is unsupported", subsystem="tts")

    def enqueue(self, unit: SpeechUnit) -> None:
        self._backend_for(unit.locale)
        if not unit.text or len(unit.text.encode("utf-8")) > 16_384:
            raise BackendError("tts.unit_size_invalid", "Speech unit is empty or too large", subsystem="tts")
        with self._lock:
            if self._error is not None:
                raise self._error
            if unit.epoch <= self._cancelled_through_epoch:
                return
            try:
                self._queue.put_nowait(unit)
            except queue.Full as exc:
                raise BackendError("tts.queue_overflow", "TTS locale router queue is full", subsystem="tts") from exc

    def cancel(self, new_epoch: int, reason: str) -> None:
        with self._lock:
            self._cancelled_through_epoch = max(self._cancelled_through_epoch, new_epoch)
            retained: list[SpeechUnit] = []
            while True:
                try:
                    unit = self._queue.get_nowait()
                except queue.Empty:
                    break
                if unit.epoch > self._cancelled_through_epoch:
                    retained.append(unit)
            for unit in retained:
                self._queue.put_nowait(unit)
        self.greek_backend.cancel(new_epoch, reason)
        self.fallback_backend.cancel(new_epoch, reason)

    @property
    def is_idle(self) -> bool:
        with self._lock:
            return (
                self._active is None
                and self._queue.empty()
                and self.greek_backend.is_idle
                and self.fallback_backend.is_idle
            )

    @property
    def last_error(self) -> BackendError | None:
        with self._lock:
            return self._error or self.greek_backend.last_error or self.fallback_backend.last_error

    def wait_idle(self, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_idle:
                return True
            time.sleep(0.005)
        return self.is_idle

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self.cancel(2**63 - 1, "application_close")
        self._thread.join(timeout=2.0)
        first_error: Exception | None = None
        for backend in (self.greek_backend, self.fallback_backend):
            try:
                backend.close()
            except Exception as exc:  # pragma: no cover - defensive native cleanup
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                unit = self._queue.get(timeout=0.02)
            except queue.Empty:
                continue
            with self._lock:
                if unit.epoch <= self._cancelled_through_epoch:
                    continue
                self._active = unit
            backend = self._backend_for(unit.locale)
            try:
                backend.enqueue(unit)
                while not self._stop.wait(0.005):
                    with self._lock:
                        stale = unit.epoch <= self._cancelled_through_epoch
                    if stale or backend.is_idle:
                        break
            except BackendError as exc:
                with self._lock:
                    self._error = exc
            finally:
                with self._lock:
                    if self._active is unit:
                        self._active = None
