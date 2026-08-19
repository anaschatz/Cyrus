from __future__ import annotations

import asyncio
import tempfile
import unittest
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.control import ControlApplication
from jarvis_orchestrator.domain import ModelChunk, ModelRequest, TaskState
from jarvis_orchestrator.schedule_admission import classify_schedule_request
from jarvis_orchestrator.scheduler_contract import PackagedTimezoneProvider, compile_proposal
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.scheduler_store import SchedulerStore
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore
from jarvis_voice.protocol import EventFactory

from test_scheduler_v5 import proposal_input, proposal_output


class CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.calls += 1
        yield ModelChunk("Κανονική απάντηση.")


class Compiler:
    def __init__(self, scheduler: SchedulerStore) -> None:
        self.scheduler = scheduler
        self.calls = 0
        self.qualification = SimpleNamespace(sha256="1" * 64)

    def compile(self, model_input):
        self.calls += 1
        return compile_proposal(
            model_input,
            proposal_output(model_input["request_id"], minutes=20),
            compiler_qualification_sha256="1" * 64,
            zones=self.scheduler.zones,
        )


class SchedulerAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        path = Path(self.temporary.name) / "orchestrator.sqlite3"
        self.store = OrchestratorStore(path)
        self.scheduler = SchedulerStore(
            self.store,
            cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            zones=PackagedTimezoneProvider(),
        )
        self.model = CountingModel()
        self.compiler = Compiler(self.scheduler)
        self.config = OrchestratorConfig.from_test_mapping({
            "database_path": str(path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
        })

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def admit(self, text: str, language: str = "el"):
        service = OrchestratorService(
            self.config,
            self.store,
            self.model,
            scheduler=self.scheduler,
            schedule_compiler=self.compiler,
        )
        event = EventFactory().create(
            "input.transcript.final",
            {
                "text": text,
                "language": {"code": language, "score": 0.99},
                "asr_score": {
                    "value": 0.9,
                    "method": "exp_weighted_avg_logprob",
                    "raw_avg_logprob": -0.1,
                    "calibrated": False,
                },
            },
            turn_id=str(uuid.uuid4()),
        )
        admission = service.admit_envelope(event)
        return service, admission

    def test_gate_is_explicit_bilingual_and_does_not_parse_time(self) -> None:
        for text in (
            "Θύμισέ μου να τηλεφωνήσω αύριο στις δέκα.",
            "Βάλε μου μια υπενθύμιση για αύριο.",
            "Βάλε μια υπενθύμιση για τις οκτώ.",
            "Remind me to take a break in twenty minutes.",
            "Set a timer for five minutes.",
        ):
            with self.subTest(text=text):
                self.assertTrue(classify_schedule_request(text).admitted)
        for text in ("Τι ώρα είναι;", "I should remember this.", "Ακύρωσε το προηγούμενο."):
            with self.subTest(text=text):
                self.assertFalse(classify_schedule_request(text).admitted)

    def test_explicit_request_uses_only_schedule_compiler_and_queues_approval(self) -> None:
        service, admission = self.admit(
            "Θύμισέ μου να κάνω διάλειμμα σε είκοσι λεπτά."
        )
        waiting = asyncio.run(service.execute_task(admission.task.task_id))
        self.assertEqual(waiting.state, TaskState.AWAITING_APPROVAL)
        self.assertEqual(self.compiler.calls, 1)
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(self.store._connection.execute(
            "SELECT COUNT(*) FROM scheduler_proposals"
        ).fetchone()[0], 1)
        self.assertEqual(self.store._connection.execute(
            "SELECT COUNT(*) FROM approval_arbiter_queue WHERE state='WAITING_FOR_CHANNEL'"
        ).fetchone()[0], 1)
        run = self.scheduler.task_run(admission.task.task_id, admission.task.epoch)
        self.scheduler.decide_approval(
            approval_id=run["approval_id"],
            proposal_id=run["proposal_id"],
            proposal_sha256=run["proposal_sha256"],
            decision="approve",
            cause_event_id=str(uuid.uuid4()),
        )
        completed = asyncio.run(service.execute_task(admission.task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        history = self.store.canonical_conversation(completed.conversation_id)
        self.assertEqual(history.turns[0].assistant_text, "Η υπενθύμιση δημιουργήθηκε.")

    def test_ordinary_request_uses_general_model(self) -> None:
        service, admission = self.admit("Πες μου μια σύντομη απάντηση.")
        completed = asyncio.run(service.execute_task(admission.task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual(self.compiler.calls, 0)
        self.assertEqual(self.model.calls, 1)

    def test_explicit_request_fails_closed_when_compiler_is_unavailable(self) -> None:
        service, admission = self.admit("Remind me to take a break in twenty minutes.", "en")
        service.schedule_compiler = None
        completed = asyncio.run(service.execute_task(admission.task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual(self.model.calls, 0)
        history = self.store.canonical_conversation(completed.conversation_id)
        self.assertIn("currently unavailable", history.turns[0].assistant_text)

    def test_capability_reports_native_delivery_only_for_qualified_connection(self) -> None:
        configured = OrchestratorConfig.from_test_mapping({
            "database_path": str(self.store.path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "voice_scheduler_compatibility_sha256": "9" * 64,
        })
        service = OrchestratorService(
            configured,
            self.store,
            self.model,
            scheduler=self.scheduler,
            schedule_compiler=self.compiler,
        )
        app = ControlApplication(configured, service)
        before = {item["name"]: item for item in app.get("/v1/capabilities")[1]["capabilities"]}
        self.assertEqual(before["notification_delivery"]["state"], "BLOCKED")
        service.attach_client(str(uuid.uuid4()), scheduler_session_id=str(uuid.uuid4()))
        after = {item["name"]: item for item in app.get("/v1/capabilities")[1]["capabilities"]}
        self.assertEqual(after["notification_delivery"]["state"], "READY")
        self.assertEqual(after["notification_actions"]["state"], "BLOCKED")
        self.assertEqual(after["reminders_overall"]["state"], "BLOCKED")

    def test_notification_actions_require_the_separately_qualified_connection(self) -> None:
        configured = OrchestratorConfig.from_test_mapping({
            "database_path": str(self.store.path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "voice_scheduler_compatibility_sha256": "9" * 64,
            "voice_scheduler_callback_qualification_sha256": "8" * 64,
        })
        service = OrchestratorService(
            configured, self.store, self.model,
            scheduler=self.scheduler, schedule_compiler=self.compiler,
        )
        app = ControlApplication(configured, service)
        connection = str(uuid.uuid4())
        session = str(uuid.uuid4())
        service.attach_client(
            connection,
            scheduler_session_id=session,
            scheduler_action_session_id=session,
        )
        capabilities = {
            item["name"]: item for item in app.get("/v1/capabilities")[1]["capabilities"]
        }
        self.assertEqual(capabilities["notification_actions"]["state"], "READY")
        self.assertEqual(
            capabilities["notification_actions"]["qualification_sha256"], "8" * 64
        )
        self.assertEqual(capabilities["reminders_overall"]["state"], "READY")


if __name__ == "__main__":
    unittest.main()
