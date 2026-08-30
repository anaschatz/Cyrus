from __future__ import annotations

import asyncio
import tempfile
import unittest
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import (
    ModelChunk,
    ModelRequest,
    ModelUsage,
    TaskState,
    parse_utc,
)
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.runtime_context import resolve_temporal_query
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore


class NeverModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.requests.append(request)
        raise AssertionError("temporal fast path must not dispatch the model")
        yield  # pragma: no cover


class RecordingModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.requests.append(request)
        yield ModelChunk("Grounded.", ModelUsage(input_tokens=32, output_tokens=2))


class TemporalFastPathTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = OrchestratorStore(Path(temporary.name) / "orchestrator.sqlite3")
        self.addCleanup(self.store.close)
        self.model = NeverModel()
        self.config = OrchestratorConfig.from_test_mapping(
            {
                "database_path": str(self.store.path),
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
                "memory_enabled": False,
                "research_enabled": False,
                "cloud_enabled": False,
                "tools_enabled": False,
                "scheduler_default_timezone": "Europe/Athens",
            }
        )
        self.service = OrchestratorService(
            self.config,
            self.store,
            BudgetedModelPort(self.store, self.model),
        )

    def test_greek_day_query_completes_without_model_and_matches_admission(self) -> None:
        admitted = self.store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload={
                "text": "Τι μέρα είναι σήμερα;",
                "language": {"code": "el", "score": 0.99},
            },
        ).task
        assert admitted is not None
        expected = resolve_temporal_query(
            "Τι μέρα είναι σήμερα;",
            input_language="el",
            response_language="match_input",
            observed_at=parse_utc(admitted.admitted_at),
            timezone_name="Europe/Athens",
        )
        assert expected is not None

        terminal = asyncio.run(self.service.execute_task(admitted.task_id))
        events = self.store.claim_outbox(admitted.conversation_id, "temporal-test")

        self.assertEqual(terminal.state, TaskState.COMPLETED)
        self.assertEqual(self.model.requests, [])
        self.assertEqual(
            [
                event.payload["text"]
                for event in events
                if event.event_type == "response.text.delta"
            ],
            [expected.text],
        )
        speech = [
            event.payload
            for event in events
            if event.event_type == "response.speech.commit"
        ]
        self.assertEqual(speech[0]["locale"], "el-GR")
        self.assertEqual(speech[0]["text"], expected.text)
        with self.store._lock:
            route = self.store._connection.execute(
                "SELECT payload_json FROM task_events WHERE task_id=? "
                "AND event_type='LOADING_CONTEXT'",
                (admitted.task_id,),
            ).fetchone()
            reservations = self.store._connection.execute(
                "SELECT COUNT(*) FROM reservations WHERE task_id=?",
                (admitted.task_id,),
            ).fetchone()[0]
        self.assertEqual(route["payload_json"], '{"route":"runtime_temporal"}')
        self.assertEqual(reservations, 0)

    def test_known_greek_asr_distortion_uses_same_grounded_path(self) -> None:
        admitted = self.store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload={
                "text": "The day is today",
                "language": {"code": "el", "score": 0.0},
            },
        ).task
        assert admitted is not None

        terminal = asyncio.run(self.service.execute_task(admitted.task_id))

        self.assertEqual(terminal.state, TaskState.COMPLETED)
        self.assertEqual(self.model.requests, [])

    def test_ordinary_llm_request_receives_authoritative_runtime_snapshot(self) -> None:
        model = RecordingModel()
        service = OrchestratorService(
            self.config,
            self.store,
            BudgetedModelPort(self.store, model),
        )
        admitted = self.store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload={
                "text": "Explain why context grounding matters.",
                "language": {"code": "en", "score": 1.0},
            },
        ).task
        assert admitted is not None

        terminal = asyncio.run(service.execute_task(admitted.task_id))

        self.assertEqual(terminal.state, TaskState.COMPLETED)
        self.assertEqual(len(model.requests), 1)
        prompt = model.requests[0].system_prompt
        expected_observed_at = (
            parse_utc(admitted.admitted_at)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        self.assertIn('"timezone":"Europe/Athens"', prompt)
        self.assertIn(f'"observed_at_utc":"{expected_observed_at}"', prompt)
        self.assertIn("Never guess the current date, time", prompt)
        self.assertEqual(prompt.count("/no_think"), 1)
        self.assertTrue(prompt.endswith("/no_think"))


if __name__ == "__main__":
    unittest.main()
