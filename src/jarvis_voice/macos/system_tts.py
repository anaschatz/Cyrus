"""Epoch-aware macOS system TTS using content-private stdin."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from typing import BinaryIO, Protocol

from ..backends import SpeechUnit
from ..errors import BackendError


class SayProcess(Protocol):
    stdin: BinaryIO | None
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


def _start_say(argv: list[str]) -> SayProcess:
    return subprocess.Popen(  # type: ignore[return-value]
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        start_new_session=True,
    )


class MacSystemTtsBackend:
    def __init__(
        self,
        *,
        voice_mapping: Mapping[str, str],
        process_factory: Callable[[list[str]], SayProcess] = _start_say,
        on_started: Callable[[SpeechUnit], None] | None = None,
        on_completed: Callable[[SpeechUnit], None] | None = None,
        executable: str = "/usr/bin/say",
        maximum_text_bytes: int = 16_384,
    ) -> None:
        self.voice_mapping = dict(voice_mapping)
        self.process_factory = process_factory
        self.on_started = on_started or (lambda unit: None)
        self.on_completed = on_completed or (lambda unit: None)
        self.executable = executable
        self.maximum_text_bytes = maximum_text_bytes
        self._queue: queue.Queue[SpeechUnit] = queue.Queue()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._cancelled_through_epoch = 0
        self._active: SpeechUnit | None = None
        self._process: SayProcess | None = None
        self._error: BackendError | None = None
        self._thread = threading.Thread(target=self._worker, name="jarvis-macos-tts", daemon=True)
        self._thread.start()

    def enqueue(self, unit: SpeechUnit) -> None:
        encoded_size = len(unit.text.encode("utf-8"))
        if encoded_size == 0 or encoded_size > self.maximum_text_bytes:
            raise BackendError("tts.unit_size_invalid", "Speech unit is empty or too large", subsystem="tts")
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
            process = self._process
        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass

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
        self._stop_event.set()
        self.cancel(2**63 - 1, "application_close")
        self._thread.join(timeout=2.0)
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    @staticmethod
    def _stop_process(process: SayProcess) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                unit = self._queue.get(timeout=0.02)
            except queue.Empty:
                continue
            with self._lock:
                if unit.epoch <= self._cancelled_through_epoch:
                    continue
                self._active = unit
            voice = self.voice_mapping.get(unit.locale, "")
            argv = [self.executable]
            if voice:
                argv.extend(["-v", voice])
            argv.extend(["-f", "-"])
            process: SayProcess | None = None
            try:
                process = self.process_factory(argv)
                with self._lock:
                    self._process = process
                    stale = unit.epoch <= self._cancelled_through_epoch
                if stale:
                    self._stop_process(process)
                    continue
                if process.stdin is None:
                    raise OSError("say stdin unavailable")
                process.stdin.write(unit.text.encode("utf-8"))
                process.stdin.close()
                self.on_started(unit)
                while process.poll() is None and not self._stop_event.wait(0.005):
                    with self._lock:
                        stale = unit.epoch <= self._cancelled_through_epoch
                    if stale:
                        self._stop_process(process)
                        break
                exit_code = process.poll()
                with self._lock:
                    stale = unit.epoch <= self._cancelled_through_epoch
                if not stale and not self._stop_event.is_set():
                    if exit_code == 0:
                        self.on_completed(unit)
                    else:
                        self._error = BackendError("tts.say_failed", "macOS system TTS failed", subsystem="tts")
            except Exception as exc:
                self._error = BackendError("tts.say_failed", "macOS system TTS failed", subsystem="tts")
                self._error.__cause__ = exc
                if process is not None:
                    self._stop_process(process)
            finally:
                with self._lock:
                    self._process = None
                    self._active = None
