from __future__ import annotations

import threading
import time
from collections import deque


class RestartCircuitBreaker:
    def __init__(self, *, limit: int = 3, window: float = 300.0, cooldown: float = 300.0):
        self.limit = limit
        self.window = window
        self.cooldown = cooldown
        self._events: deque[float] = deque()
        self._open_since: float | None = None
        self._half_open_inflight = False
        self._lock = threading.Lock()

    def allow(self, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        with self._lock:
            self._prune(current)
            if self._open_since is None:
                if len(self._events) < self.limit:
                    return True
                self._open_since = current
                return False
            if current - self._open_since < self.cooldown or self._half_open_inflight:
                return False
            self._half_open_inflight = True
            return True

    def record_restart(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        with self._lock:
            self._prune(current)
            if self._half_open_inflight:
                self._open_since = current
                self._half_open_inflight = False
                return
            if len(self._events) >= self.limit:
                self._open_since = current
                self._half_open_inflight = False
                return
            self._events.append(current)

    def probe_succeeded(self) -> None:
        with self._lock:
            if self._half_open_inflight:
                self._events.clear()
                self._open_since = None
                self._half_open_inflight = False

    def probe_failed(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        with self._lock:
            self._open_since = current
            self._half_open_inflight = False

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0] > self.window:
            self._events.popleft()
