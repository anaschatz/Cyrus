"""LLM-independent bounded scheduler worker.

UTC wall time owns persisted due instants.  Monotonic time is used only for
bounded waits; every wake re-reads UTC so wall-clock jumps flow through the
same recovery path as restart.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from .scheduler_store import SchedulerStore


class SchedulerWorker:
    def __init__(
        self,
        store: SchedulerStore,
        *,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
        maximum_wait_seconds: float = 0.25,
    ) -> None:
        if not 0.01 <= maximum_wait_seconds <= 1.0:
            raise ValueError("maximum_wait_seconds must be between 10ms and 1s")
        self.store = store
        self.wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self.monotonic_ns = monotonic_ns or time.monotonic_ns
        self.maximum_wait_seconds = maximum_wait_seconds
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._last_wall_ms: int | None = None
        self._last_monotonic_ns: int | None = None
        self.worker_boot_id = str(uuid.uuid4())

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name="jarvis-deterministic-scheduler")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def notify(self) -> None:
        self._wake.set()

    def tick(self) -> dict[str, int | bool]:
        now = self.wall_clock()
        if now.tzinfo is None or now.utcoffset() != UTC.utcoffset(now):
            raise RuntimeError("scheduler wall clock must return UTC")
        wall_ms = int(now.timestamp() * 1000)
        mono = self.monotonic_ns()
        jumped = False
        if self._last_wall_ms is not None and self._last_monotonic_ns is not None:
            elapsed_ms = max(0, (mono - self._last_monotonic_ns) // 1_000_000)
            jumped = abs((wall_ms - self._last_wall_ms) - elapsed_ms) > 1000
        self._last_wall_ms = wall_ms
        self._last_monotonic_ns = mono
        expired_approvals = self.store.expire_approvals(now=now)
        key_restored = self.store.retry_key_blocked_deliveries(now=now)
        fired = self.store.fire_due(
            now=now,
            worker_boot_id=self.worker_boot_id,
            monotonic_lease_deadline_ns=mono + int(self.maximum_wait_seconds * 1_000_000_000),
            clock_jump_detected=jumped,
        )
        expired_deliveries = self.store.expire_deliveries(now=now)
        purged = self.store.purge_content(now=now)
        return {
            "clock_jump_detected": jumped,
            "expired_approval_count": expired_approvals,
            "key_restored_delivery_count": key_restored,
            "fired_count": len(fired),
            "expired_delivery_count": expired_deliveries,
            "purged_content_count": purged,
        }

    async def run(self) -> None:
        while True:
            self.tick()
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.maximum_wait_seconds)
            except TimeoutError:
                pass
