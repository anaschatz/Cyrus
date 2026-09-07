from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import TaskState
from jarvis_orchestrator.scheduler_contract import PackagedTimezoneProvider
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.scheduler_store import SchedulerStore
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore
from test_planner_scheduler_tool_e2e import FinalModel, ScheduleCompiler, transcript


class ScheduleAdmissionDeadlineTests(unittest.TestCase):
    def test_task_deadline_stops_waiting_and_discards_late_compiler_result(self):
        for boundary in ("active", "absolute"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "store.sqlite3"
                config = OrchestratorConfig(
                    database_path=path, control_bearer_token="control",
                    inference_bearer_token="inference",
                    active_budget_seconds=0.05 if boundary == "active" else 120,
                )
                store = OrchestratorStore(path, active_budget_seconds=config.active_budget_seconds)
                try:
                    scheduler = SchedulerStore(
                        store, cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
                        zones=PackagedTimezoneProvider(),
                    )
                    compiler = ScheduleCompiler(scheduler)
                    service = OrchestratorService(config, store, FinalModel(),
                                                  scheduler=scheduler, schedule_compiler=compiler)
                    task = store.admit_transcript(
                        event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4()),
                        payload=transcript("Θύμισέ μου να κάνω διάλειμμα σε είκοσι λεπτά"),
                        absolute_lifetime=timedelta(seconds=0.05 if boundary == "absolute" else 120),
                    ).task
                    release = threading.Event()
                    compile_proposal = compiler.compile

                    def delayed_compile(value):
                        if not release.wait(2):
                            raise TimeoutError("test did not release compiler")
                        return compile_proposal(value)

                    async def run():
                        try:
                            completed = await asyncio.wait_for(service.execute_task(task.task_id), 0.5)
                            self.assertEqual(completed.state, TaskState.BUDGET_EXHAUSTED)
                            self.assertIsNone(scheduler.task_run(task.task_id, task.epoch))
                        finally:
                            release.set()

                    with patch.object(compiler, "compile", side_effect=delayed_compile) as call:
                        asyncio.run(run())  # Also joins the released compiler worker.
                        self.assertEqual(call.call_count, 1)
                    self.assertIsNone(scheduler.task_run(task.task_id, task.epoch))
                    self.assertEqual(scheduler.list_jobs(), [])
                finally:
                    store.close()


if __name__ == "__main__":
    unittest.main()
