"""Epoch-aware SAPI5 worker. COM is owned exclusively by its STA thread."""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from ..backends import SpeechUnit
from ..errors import BackendError

SVSFLAGS_ASYNC = 1
SVSFPURGE_BEFORE_SPEAK = 2


class SapiAdapter(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def speak_async(self, text: str, voice_description: str) -> None: ...
    def wait_until_done(self, timeout_ms: int) -> bool: ...
    def purge(self) -> None: ...
    def voices(self) -> list[str]: ...


class Win32SapiAdapter:
    def __init__(self) -> None:
        self._voice = None
        self._pythoncom = None

    def open(self) -> None:  # pragma: no cover - native Windows only
        if sys.platform != "win32":
            raise BackendError("tts.sapi_windows_only", "SAPI requires Windows", subsystem="tts")
        try:
            import pythoncom  # type: ignore[import-not-found]
            import win32com.client  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendError("tts.runtime_missing", "pywin32 is not installed", subsystem="tts") from exc
        pythoncom.CoInitialize()
        self._pythoncom = pythoncom
        self._voice = win32com.client.Dispatch("SAPI.SpVoice")

    def close(self) -> None:  # pragma: no cover - native Windows only
        self._voice = None
        if self._pythoncom is not None:
            self._pythoncom.CoUninitialize(); self._pythoncom = None

    def voices(self) -> list[str]:  # pragma: no cover - native Windows only
        return [self._voice.GetVoices().Item(index).GetDescription() for index in range(self._voice.GetVoices().Count)]

    def _select_voice(self, description: str) -> None:  # pragma: no cover
        if not description:
            return
        voices = self._voice.GetVoices()
        for index in range(voices.Count):
            token = voices.Item(index)
            if token.GetDescription() == description:
                self._voice.Voice = token
                return
        raise BackendError("tts.voice_missing", "Configured SAPI voice is not installed", subsystem="tts")

    def speak_async(self, text: str, voice_description: str) -> None:  # pragma: no cover
        self._select_voice(voice_description)
        self._voice.Speak(text, SVSFLAGS_ASYNC)

    def wait_until_done(self, timeout_ms: int) -> bool:  # pragma: no cover
        return bool(self._voice.WaitUntilDone(timeout_ms))

    def purge(self) -> None:  # pragma: no cover
        self._voice.Speak("", SVSFLAGS_ASYNC | SVSFPURGE_BEFORE_SPEAK)


@dataclass(frozen=True, slots=True)
class SapiTiming:
    event: str
    timestamp: float
    epoch: int


class SapiTtsBackend:
    def __init__(self, *, voice_mapping: Mapping[str, str], adapter_factory: Callable[[], SapiAdapter] = Win32SapiAdapter, on_started: Callable[[SpeechUnit], None] | None = None, on_completed: Callable[[SpeechUnit], None] | None = None) -> None:
        self.voice_mapping = dict(voice_mapping)
        self.adapter_factory = adapter_factory
        self.on_started = on_started or (lambda unit: None)
        self.on_completed = on_completed or (lambda unit: None)
        self.timings: list[SapiTiming] = []
        self._queue: queue.Queue[SpeechUnit] = queue.Queue()
        self._cancel_event = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._cancelled_through_epoch = 0
        self._active: SpeechUnit | None = None
        self._error: BackendError | None = None
        self._thread = threading.Thread(target=self._worker, name="jarvis-sapi", daemon=True)
        self._thread.start()

    def enqueue(self, unit: SpeechUnit) -> None:
        with self._lock:
            if unit.epoch <= self._cancelled_through_epoch:
                return
            self._queue.put_nowait(unit)

    def cancel(self, new_epoch: int, reason: str) -> None:
        del reason
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
            self._cancel_event.set()

    @property
    def is_idle(self) -> bool:
        with self._lock:
            return self._active is None and self._queue.empty()

    @property
    def last_error(self) -> BackendError | None:
        return self._error

    def wait_idle(self, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_idle:
                return True
            time.sleep(0.005)
        return self.is_idle

    def close(self) -> None:
        self._stop_event.set(); self._cancel_event.set(); self._thread.join(timeout=2)

    def _worker(self) -> None:
        adapter = self.adapter_factory()
        try:
            adapter.open()
            while not self._stop_event.is_set():
                try:
                    unit = self._queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                with self._lock:
                    if unit.epoch <= self._cancelled_through_epoch:
                        continue
                    self._active = unit
                    self._cancel_event.clear()
                self.timings.append(SapiTiming("speak_dispatch", time.perf_counter(), unit.epoch))
                adapter.speak_async(unit.text, self.voice_mapping.get(unit.locale, ""))
                self.on_started(unit)
                cancelled = False
                while not adapter.wait_until_done(10):
                    if self._cancel_event.is_set() or self._stop_event.is_set():
                        self.timings.append(SapiTiming("purge_dispatch", time.perf_counter(), unit.epoch))
                        adapter.purge()
                        self.timings.append(SapiTiming("purge_complete", time.perf_counter(), unit.epoch))
                        cancelled = True
                        break
                with self._lock:
                    stale = unit.epoch <= self._cancelled_through_epoch
                    self._active = None
                if not cancelled and not stale:
                    self.timings.append(SapiTiming("stream_complete", time.perf_counter(), unit.epoch))
                    self.on_completed(unit)
        except BackendError as exc:
            self._error = exc
        except Exception as exc:
            self._error = BackendError("tts.worker_failed", "SAPI worker failed", subsystem="tts")
            self._error.__cause__ = exc
        finally:
            with self._lock:
                self._active = None
            adapter.close()
