from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import (
    ModelChunk,
    ModelRequest,
    RouteDecision,
    TaskState,
    parse_utc,
    utc_text,
)
from jarvis_orchestrator.scheduler_contract import PackagedTimezoneProvider, compile_proposal
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.scheduler_store import SchedulerStore
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore
from jarvis_orchestrator.tool_contracts import (
    ToolOutcomeV1,
    ToolRegistryV1,
    ToolStatus,
    VerificationStatus,
)
from jarvis_orchestrator.tool_crypto import InMemoryToolKeyProvider, ToolPayloadCipherV1
from jarvis_orchestrator.tool_store import ToolStoreV1


def transcript(text: str) -> dict[str, object]:
    return {
        "text": text,
        "language": {"code": "el", "score": 0.99},
        "asr_score": {
            "value": 0.9,
            "method": "exp_weighted_avg_logprob",
            "raw_avg_logprob": -0.1,
            "calibrated": False,
        },
    }


class FinalModel:
    async def stream(self, request: ModelRequest):
        self.request = request
        yield ModelChunk("Η σημείωση αποθηκεύτηκε.")


class ToolPlanner:
    def __init__(self) -> None:
        self.route_calls = 0
        self.argument_calls = 0

    async def plan_with_correction(self, request: ModelRequest) -> RouteDecision:
        self.route_calls += 1
        return RouteDecision("tool_local", "", "", "note.capture@1", "el")

    async def plan_tool_arguments(self, request: ModelRequest, *, definition):
        self.argument_calls += 1
        self.definition = definition
        return {"title": "Ιδέα", "body": "Να αγοράσω καφέ."}


class ToolExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.acks: list[tuple[str, str]] = []

    async def execute_typed(self, **kwargs):
        self.calls.append(kwargs)
        arguments = kwargs["arguments"]
        return (
            ToolOutcomeV1(ToolStatus.VERIFIED, VerificationStatus.MATCHED, "ok"),
            {
                "note_ref": str(uuid.uuid4()),
                "title": arguments["title"],
                "body_sha256": hashlib.sha256(arguments["body"].encode("utf-8")).hexdigest(),
                "created_at_utc": "2030-01-10T12:00:01.000Z",
                "storage_domain": "JARVIS_PRIVATE_NOTES",
            },
        )

    async def acknowledge(self, execution_id: str, idempotency_key: str) -> None:
        self.acks.append((execution_id, idempotency_key))


class ScheduleCompiler:
    def __init__(self, scheduler: SchedulerStore) -> None:
        self.scheduler = scheduler

    def compile(self, value):
        request_id = value["request_id"]
        none = {
            "kind": "NONE",
            "local_datetime": None,
            "year": None,
            "month": None,
            "day": None,
            "local_time": None,
            "timezone": None,
            "amount": None,
            "unit": None,
            "relation": None,
            "weekday": None,
        }
        output = {
            "schema_version": "schedule-proposal/1",
            "request_id": request_id,
            "outcome": "PROPOSAL",
            "operation": "CREATE_JOB",
            "target_job_id": None,
            "target_job_revision": None,
            "target_occurrence_id": None,
            "time_expression": {
                **none,
                "kind": "AFTER_DURATION",
                "amount": 20,
                "unit": "MINUTE",
            },
            "recurrence": {
                "kind": "NONE",
                "interval_amount": None,
                "interval_unit": None,
                "weekdays": [],
                "day_of_month": None,
                "count": None,
                "until": none,
            },
            "notification": {
                "title": "Διάλειμμα",
                "body": "Κάνε διάλειμμα.",
                "speakable_text": "Ώρα για διάλειμμα.",
                "locale": "el-GR",
                "visibility": "private",
            },
            "clarification_reason": "NONE",
            "clarification_question": None,
        }
        return compile_proposal(
            value,
            output,
            compiler_qualification_sha256="1" * 64,
            zones=self.scheduler.zones,
        )


class PlannerSchedulerToolE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "orchestrator.sqlite3"

    def admit(self, store: OrchestratorStore, text: str):
        result = store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload=transcript(text),
            now=parse_utc(utc_text()),
        )
        assert result.task is not None
        return result.task

    def test_scheduler_compiler_approval_executes_and_resumes_same_task(self) -> None:
        store = OrchestratorStore(self.path)
        self.addCleanup(store.close)
        scheduler = SchedulerStore(
            store,
            cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            zones=PackagedTimezoneProvider(),
        )
        config = OrchestratorConfig(
            database_path=self.path,
            control_bearer_token="control",
            inference_bearer_token="inference",
        )
        service = OrchestratorService(
            config,
            store,
            FinalModel(),
            scheduler=scheduler,
            schedule_compiler=ScheduleCompiler(scheduler),
        )
        task = self.admit(store, "Θύμισέ μου να κάνω διάλειμμα σε είκοσι λεπτά")
        waiting = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(waiting.state, TaskState.AWAITING_APPROVAL)
        run = scheduler.task_run(task.task_id, task.epoch)
        self.assertEqual(run["state"], "PENDING")

        scheduler.decide_approval(
            approval_id=run["approval_id"],
            proposal_id=run["proposal_id"],
            proposal_sha256=run["proposal_sha256"],
            decision="approve",
            cause_event_id=str(uuid.uuid4()),
            now=parse_utc(utc_text()),
        )
        self.assertEqual(service.startup_recovery(), 0)
        self.assertEqual(
            store.get_task(task.task_id).state,
            TaskState.DECISION_WAITING_RECONNECT,
        )
        service.attach_client("reconnected-scheduler-client")
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual(scheduler.list_jobs()[0]["state"], "DRAINING")
        answer = store.canonical_conversation(task.conversation_id).turns[0].assistant_text
        self.assertEqual(answer, "Η υπενθύμιση δημιουργήθηκε.")

    def test_tool_planner_approval_permit_execution_and_local_follow_up(self) -> None:
        store = OrchestratorStore(self.path, model_request_limit=6, tool_authority_enabled=True)
        self.addCleanup(store.close)
        registry = ToolRegistryV1.build(root_ids=("documents",), app_ids=("textedit",))
        tool_store = ToolStoreV1(
            store,
            cipher=ToolPayloadCipherV1(InMemoryToolKeyProvider()),
            registry=registry,
        )
        planner = ToolPlanner()
        executor = ToolExecutor()
        model = FinalModel()
        config = OrchestratorConfig(
            database_path=self.path,
            control_bearer_token="control",
            inference_bearer_token="inference",
            model_request_limit=6,
            tools_enabled=True,
            tool_policy_sha256="a" * 64,
            scheduler_contract_sha256="b" * 64,
        )
        service = OrchestratorService(
            config,
            store,
            model,
            tool_store=tool_store,
            tool_planner=planner,
            tool_executor=executor,
        )
        task = self.admit(store, "Κράτησε σημείωση να αγοράσω καφέ")
        waiting = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(waiting.state, TaskState.AWAITING_APPROVAL)
        run = tool_store.task_run(task.task_id, task.epoch)
        tool_store.decide_action_approval(
            approval_id=run["approval_id"],
            epoch=task.epoch,
            decision="approve",
        )
        self.assertEqual(service.startup_recovery(), 0)
        self.assertEqual(
            store.get_task(task.task_id).state,
            TaskState.DECISION_WAITING_RECONNECT,
        )
        service.attach_client("reconnected-tool-client")

        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual((planner.route_calls, planner.argument_calls, len(executor.calls)), (1, 1, 1))
        self.assertTrue(executor.calls[0]["permit_id"])
        self.assertEqual(
            executor.acks,
            [(executor.calls[0]["execution_id"], executor.calls[0]["idempotency_key"])],
        )
        effects = store.canonical_conversation(task.conversation_id).turns[0].effects
        self.assertEqual([(item.kind, item.status) for item in effects], [("tool", "VERIFIED")])
        self.assertIn("Local tool result JSON", model.request.messages[-1]["content"])

    def test_cancellation_while_tool_approval_pending_prevents_execution(self) -> None:
        store = OrchestratorStore(self.path, model_request_limit=6, tool_authority_enabled=True)
        self.addCleanup(store.close)
        registry = ToolRegistryV1.build(root_ids=("documents",), app_ids=("textedit",))
        tool_store = ToolStoreV1(
            store,
            cipher=ToolPayloadCipherV1(InMemoryToolKeyProvider()),
            registry=registry,
        )
        executor = ToolExecutor()
        service = OrchestratorService(
            OrchestratorConfig(
                database_path=self.path,
                control_bearer_token="control",
                inference_bearer_token="inference",
                model_request_limit=6,
                tools_enabled=True,
                tool_policy_sha256="a" * 64,
                scheduler_contract_sha256="b" * 64,
            ),
            store,
            FinalModel(),
            tool_store=tool_store,
            tool_planner=ToolPlanner(),
            tool_executor=executor,
        )
        task = self.admit(store, "Κράτησε σημείωση να αγοράσω καφέ")
        asyncio.run(service.execute_task(task.task_id))
        service.cancel_task(task.task_id, task.epoch)
        self.assertEqual(store.get_task(task.task_id).state, TaskState.CANCELLED)
        self.assertEqual(tool_store.task_run(task.task_id, task.epoch)["status"], "CANCELLED")
        self.assertEqual(executor.calls, [])


if __name__ == "__main__":
    unittest.main()
