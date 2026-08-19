from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import ModelChunk, ModelRequest, TaskState
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.pydantic_adapter import build_local_pydantic_port
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore


class DownModel:
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.request = request
        raise RuntimeError("private provider diagnostic")
        yield  # pragma: no cover


class HangingModel:
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.request = request
        await asyncio.sleep(10)
        yield ModelChunk("late")  # pragma: no cover


class ModelFailureGateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "orchestrator.sqlite3"

    def config(self, *, timeout: float) -> OrchestratorConfig:
        return OrchestratorConfig.from_test_mapping(
            {
                "database_path": str(self.path),
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
                "model_timeout_seconds": timeout,
                "memory_enabled": False,
                "research_enabled": False,
                "cloud_enabled": False,
                "tools_enabled": False,
            }
        )

    def run_failure(self, model: object, *, timeout: float):
        store = OrchestratorStore(self.path)
        self.addCleanup(store.close)
        service = OrchestratorService(
            self.config(timeout=timeout),
            store,
            BudgetedModelPort(store, model),  # type: ignore[arg-type]
        )
        admitted = store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload={
                "text": "failure contract",
                "language": {"code": "en", "score": 1.0},
                "asr_score": {
                    "calibrated": False,
                    "method": "test",
                    "value": 1.0,
                },
            },
        ).task
        assert admitted is not None
        started = time.monotonic()
        terminal = asyncio.run(service.execute_task(admitted.task_id))
        elapsed = time.monotonic() - started
        events = store.claim_outbox(terminal.conversation_id, "failure-gate")
        with store._lock:
            reservations = store._connection.execute(
                "SELECT purpose,state,output_allowance,actual_output_tokens "
                "FROM reservations WHERE task_id=? ORDER BY created_at",
                (admitted.task_id,),
            ).fetchall()
            route = store._connection.execute(
                "SELECT payload_json FROM task_events WHERE task_id=? "
                "AND event_type='LOADING_CONTEXT'",
                (admitted.task_id,),
            ).fetchone()
        return store, terminal, elapsed, events, reservations, route

    def assert_deterministic_failure(self, model: object, *, timeout: float) -> None:
        _, terminal, elapsed, events, reservations, route = self.run_failure(
            model, timeout=timeout
        )
        self.assertEqual(terminal.state, TaskState.FAILED)
        self.assertLess(elapsed, timeout + 0.5)
        self.assertEqual(
            [event.event_type for event in events],
            ["server.ack", "server.error", "response.cancelled"],
        )
        self.assertEqual(events[1].payload["code"], "backend_failure")
        self.assertEqual(events[2].payload["reason_code"], "backend_failure")
        self.assertEqual(events[2].payload["partial_output"], False)
        self.assertFalse(
            {
                "response.started",
                "response.text.delta",
                "response.speech.commit",
                "response.speech.done",
            }
            & {event.event_type for event in events}
        )
        self.assertEqual(len({event.event_id for event in events}), 3)
        self.assertEqual(len(reservations), 1)
        self.assertEqual(reservations[0]["purpose"], "final")
        self.assertEqual(reservations[0]["state"], "RECONCILED")
        self.assertEqual(
            reservations[0]["actual_output_tokens"],
            reservations[0]["output_allowance"],
        )
        self.assertIsNotNone(route)
        self.assertEqual(route["payload_json"], '{"route":"local"}')

    def test_down_model_is_local_only_and_has_exact_terminal_wire_contract(self) -> None:
        self.assert_deterministic_failure(DownModel(), timeout=0.2)

    def test_hanging_model_obeys_bounded_timeout_and_same_terminal_contract(self) -> None:
        self.assert_deterministic_failure(HangingModel(), timeout=0.03)

    def test_failure_replay_has_same_logical_ids_without_row_growth(self) -> None:
        store, terminal, _, first, _, _ = self.run_failure(DownModel(), timeout=0.2)
        for event in first:
            store.finish_outbox_attempt(event.event_id, "failure-gate", complete=True)
        replay = store.claim_outbox(terminal.conversation_id, "reconnect")
        self.assertEqual([event.event_id for event in first], [event.event_id for event in replay])
        with store._lock:
            count = store._connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE response_id=?",
                (terminal.response_id,),
            ).fetchone()[0]
        self.assertEqual(count, 3)

    def test_production_pydantic_transport_failure_is_backend_failure(self) -> None:
        try:
            import httpx
        except ImportError:  # pragma: no cover - pinned production dependency
            self.skipTest("httpx is unavailable")

        async def unavailable(request):
            raise httpx.ConnectError("private transport diagnostic", request=request)

        store = OrchestratorStore(self.path)
        self.addCleanup(store.close)
        port = build_local_pydantic_port(
            store,
            base_url="http://127.0.0.1:9/v1",
            bearer_token="private",
            timeout_seconds=0.2,
            http_transport=httpx.MockTransport(unavailable),
        )
        service = OrchestratorService(self.config(timeout=0.2), store, port)
        admitted = store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload={
                "text": "production failure contract",
                "language": {"code": "en", "score": 1.0},
                "asr_score": {
                    "calibrated": False,
                    "method": "test",
                    "value": 1.0,
                },
            },
        ).task
        assert admitted is not None
        terminal = asyncio.run(service.execute_task(admitted.task_id))
        events = store.claim_outbox(terminal.conversation_id, "pydantic-failure")
        self.assertEqual(terminal.state, TaskState.FAILED)
        self.assertEqual(
            [event.event_type for event in events],
            ["server.ack", "server.error", "response.cancelled"],
        )
        self.assertEqual(events[1].payload["code"], "backend_failure")
        self.assertEqual(events[2].payload["reason_code"], "backend_failure")


if __name__ == "__main__":
    unittest.main()
