from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jarvis_orchestrator.scheduler_contract import PackagedTimezoneProvider
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.scheduler_store import SchedulerStore
from jarvis_orchestrator.scheduler_worker import SchedulerWorker
from jarvis_orchestrator.store import OrchestratorStore


class SchedulerClockTests(unittest.TestCase):
    def test_wall_jump_is_observed_but_monotonic_owns_wait_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = OrchestratorStore(Path(directory) / "db.sqlite3")
            try:
                store = SchedulerStore(
                    base, cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
                    zones=PackagedTimezoneProvider(),
                )
                walls = iter([
                    datetime(2030, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2030, 1, 1, 1, 0, tzinfo=UTC),
                    datetime(2030, 1, 1, 0, 30, tzinfo=UTC),
                ])
                monotonic = iter([1_000_000_000, 1_250_000_000, 1_500_000_000])
                worker = SchedulerWorker(
                    store, wall_clock=lambda: next(walls), monotonic_ns=lambda: next(monotonic),
                    maximum_wait_seconds=0.25,
                )
                first = worker.tick()
                forward = worker.tick()
                backward = worker.tick()
                self.assertFalse(first["clock_jump_detected"])
                self.assertTrue(forward["clock_jump_detected"])
                self.assertTrue(backward["clock_jump_detected"])
                self.assertNotEqual(worker.worker_boot_id, "")
            finally:
                base.close()


if __name__ == "__main__":
    unittest.main()
