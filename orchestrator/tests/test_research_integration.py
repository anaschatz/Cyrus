from __future__ import annotations

import asyncio
import tempfile
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401
from test_research_contract import corpus_fixture

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import ModelRequest, ReasonCode, TaskState, utc_now
from jarvis_orchestrator.research_adapter import FakeResearchAdapter
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore


class UnusedTextModel:
    calls = 0

    async def stream(self, request: ModelRequest):
        self.calls += 1
        raise AssertionError("research publication must not invoke the text model")
        yield  # pragma: no cover


class FakePlanner:
    def __init__(self, plan, selections) -> None:
        self.plan = plan
        self.selections = selections
        self.calls = []
        self.span_calls = []

    async def plan_research(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["output_allowance"] != 384:
            raise AssertionError("wrong planning budget")
        return self.plan

    async def select_research_spans(self, **kwargs):
        self.span_calls.append(kwargs)
        if kwargs["output_allowance"] != 768:
            raise AssertionError("wrong span-selection budget")
        return self.selections


class BlockingAdapter:
    def __init__(self, result) -> None:
        self.result = result
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def research(self, request, *, idempotency_key):
        del request, idempotency_key
        self.started.set()
        await self.release.wait()
        return self.result


def research_config(path: Path) -> OrchestratorConfig:
    return OrchestratorConfig.from_test_mapping(
        {
            "database_path": str(path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "research_enabled": True,
            "research_bearer_token": "research-private",
            "research_language_detector_version": "lid/1",
            "research_language_detector_sha256": "a" * 64,
            "research_model_sha256": "b" * 64,
            "research_profile_sha256": "c" * 64,
            "research_qualification_sha256": "d" * 64,
        }
    )


def admit(store: OrchestratorStore):
    result = store.admit_transcript(
        event_id=str(uuid.uuid4()),
        turn_id=str(uuid.uuid4()),
        payload={"text": "Βρες την τρέχουσα τιμή.", "language": {"code": "el", "score": 0.99}},
        now=utc_now(),
    )
    return result.task


class ResearchIntegrationTests(unittest.TestCase):
    def make_service(self, adapter=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "state.sqlite3"
        store = OrchestratorStore(path)
        self.addCleanup(store.close)
        plan, corpus, selection, _ = corpus_fixture()
        planner = FakePlanner(plan, (selection,))
        selected = adapter or FakeResearchAdapter(
            {plan.plan_hash: corpus}
        )
        model = UnusedTextModel()
        service = OrchestratorService(
            research_config(path),
            store,
            model,
            research_planner=planner,
            research=selected,
        )
        service.attach_client("test-client")
        return service, store, planner, selected, model

    def test_route_approval_single_reservation_bundle_and_deterministic_publication(self) -> None:
        service, store, planner, adapter, model = self.make_service()
        task = admit(store)
        waiting = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(waiting.state, TaskState.AWAITING_APPROVAL)
        approval = store.list_pending_approvals()[0]
        service.decide_approval(approval["approval_id"], task.epoch, "approve")
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(len(planner.span_calls), 1)
        self.assertEqual(adapter.calls[0][0].span_selection_output_tokens, 768)
        self.assertEqual(len(planner.calls) + len(planner.span_calls), 2)
        self.assertEqual(model.calls, 0)
        with store._lock:
            reservations = store._connection.execute(
                "SELECT COUNT(*) FROM capability_reservations WHERE task_id=? AND kind='research'",
                (task.task_id,),
            ).fetchone()[0]
        self.assertEqual(reservations, 1)
        answer = store.canonical_conversation(task.conversation_id).turns[0].assistant_text
        self.assertIn("https://example.com/product", answer)
        self.assertIn("Βρέθηκαν επαρκή τεκμήρια", answer)

    def test_denial_never_calls_retrieval_or_latent_model(self) -> None:
        service, store, planner, adapter, model = self.make_service()
        task = admit(store)
        asyncio.run(service.execute_task(task.task_id))
        approval = store.list_pending_approvals()[0]
        service.decide_approval(approval["approval_id"], task.epoch, "deny")
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual((len(adapter.calls), model.calls), (0, 0))
        self.assertIn("δεν εκτελέστηκε", store.canonical_conversation(task.conversation_id).turns[0].assistant_text)

    def test_late_adapter_completion_after_cancellation_is_discarded(self) -> None:
        _, corpus, _, _ = corpus_fixture()
        blocking = BlockingAdapter(corpus)
        service, store, _, _, _ = self.make_service(blocking)
        task = admit(store)
        asyncio.run(service.execute_task(task.task_id))
        approval = store.list_pending_approvals()[0]
        service.decide_approval(approval["approval_id"], task.epoch, "approve")

        async def race():
            execution = asyncio.create_task(service.execute_task(task.task_id))
            await blocking.started.wait()
            service.cancel_task(task.task_id, task.epoch)
            blocking.release.set()
            return await execution

        current = asyncio.run(race())
        self.assertEqual(current.state, TaskState.CANCELLED)
        self.assertIsNone(store.research_run(task.task_id)["bundle_json"])
        self.assertEqual(store.research_documents(task.task_id), {})
        self.assertEqual(store.canonical_conversation(task.conversation_id).turns, ())


if __name__ == "__main__":
    unittest.main()
