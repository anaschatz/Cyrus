"""Thread-safe lifecycle around the blocking Windows voice runtime."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from .errors import VoiceError


class RuntimeProcess(Protocol):
    def run(self) -> None: ...
    def request_stop(self) -> None: ...


class LifecycleLogger(Protocol):
    def write(self, event: str, *, status: str | None = None, code: str | None = None) -> None: ...


class LifecycleStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LifecycleSnapshot:
    status: LifecycleStatus
    error_code: str | None = None


class VoiceLifecycleController:
    def __init__(self, runtime_factory: Callable[[], RuntimeProcess], *, logger: LifecycleLogger | None = None, shutdown_timeout: float = 5.0) -> None:
        self.runtime_factory = runtime_factory
        self.logger = logger
        self.shutdown_timeout = shutdown_timeout
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._runtime: RuntimeProcess | None = None
        self._stop_requested = threading.Event()
        self._status = LifecycleStatus.STOPPED
        self._error_code: str | None = None

    @property
    def snapshot(self) -> LifecycleSnapshot:
        with self._lock:
            return LifecycleSnapshot(self._status, self._error_code)

    def _record(self, event: str, *, status: LifecycleStatus, code: str | None = None) -> None:
        if self.logger is not None:
            try:
                self.logger.write(event, status=status.value, code=code)
            except Exception:
                # Local diagnostics are best-effort and must never break voice
                # cancellation or trap the application in a half-started state.
                pass

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if self._status in {LifecycleStatus.STARTING, LifecycleStatus.RUNNING, LifecycleStatus.STOPPING}:
                return False
            self._status = LifecycleStatus.STARTING
            self._error_code = None
            self._stop_requested.clear()
            try:
                self._thread = threading.Thread(target=self._worker, name="jarvis-voice-runtime", daemon=True)
                self._thread.start()
            except Exception:
                self._thread = None
                self._status = LifecycleStatus.ERROR
                self._error_code = "runtime.start_failed"
                self._record("voice.lifecycle.error", status=LifecycleStatus.ERROR, code=self._error_code)
                return False
        self._record("voice.lifecycle.start", status=LifecycleStatus.STARTING)
        return True

    def _worker(self) -> None:
        try:
            runtime = self.runtime_factory()
            with self._lock:
                self._runtime = runtime
                should_stop = self._stop_requested.is_set()
                self._status = LifecycleStatus.STOPPING if should_stop else LifecycleStatus.RUNNING
            if should_stop:
                runtime.request_stop()
            else:
                self._record("voice.lifecycle.running", status=LifecycleStatus.RUNNING)
            runtime.run()
        except VoiceError as exc:
            with self._lock:
                self._status = LifecycleStatus.ERROR
                self._error_code = exc.code
            self._record("voice.lifecycle.error", status=LifecycleStatus.ERROR, code=exc.code)
        except Exception:
            with self._lock:
                self._status = LifecycleStatus.ERROR
                self._error_code = "runtime.unhandled_failure"
            self._record("voice.lifecycle.error", status=LifecycleStatus.ERROR, code="runtime.unhandled_failure")
        else:
            with self._lock:
                self._status = LifecycleStatus.STOPPED
                self._error_code = None
            self._record("voice.lifecycle.stopped", status=LifecycleStatus.STOPPED)
        finally:
            with self._lock:
                self._runtime = None
                self._thread = None

    def stop(self) -> bool:
        with self._lock:
            if self._status in {LifecycleStatus.STOPPED, LifecycleStatus.ERROR} and (
                self._thread is None or not self._thread.is_alive()
            ):
                return False
            self._status = LifecycleStatus.STOPPING
            self._stop_requested.set()
            runtime = self._runtime
            thread = self._thread
        self._record("voice.lifecycle.stop", status=LifecycleStatus.STOPPING)
        if runtime is not None:
            try:
                runtime.request_stop()
            except Exception:
                with self._lock:
                    self._status = LifecycleStatus.ERROR
                    self._error_code = "runtime.stop_failed"
                self._record("voice.lifecycle.error", status=LifecycleStatus.ERROR, code="runtime.stop_failed")
                return False
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.shutdown_timeout)
            if thread.is_alive():
                with self._lock:
                    self._status = LifecycleStatus.ERROR
                    self._error_code = "runtime.shutdown_timeout"
                self._record("voice.lifecycle.error", status=LifecycleStatus.ERROR, code="runtime.shutdown_timeout")
                return False
        return True

    def shutdown(self) -> None:
        self.stop()
