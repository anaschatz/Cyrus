from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from jarvis_orchestrator.domain import TaskState
from jarvis_orchestrator.store import OrchestratorStore


NOW = datetime(2026, 8, 10, 9, 0, tzinfo=UTC)


def payload(text: str = "Γεια σου.") -> dict[str, object]:
    return {
        "text": text,
        "language": {"code": "el", "score": 0.99},
        "asr_score": {
            "value": 0.8,
            "method": "exp_weighted_avg_logprob",
            "raw_avg_logprob": -0.2,
            "calibrated": False,
        },
    }


class StoreCase:
    def make_store(self, **kwargs: object) -> OrchestratorStore:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)  # type: ignore[attr-defined]
        store = OrchestratorStore(Path(temporary.name) / "orchestrator.sqlite3", **kwargs)
        self.addCleanup(store.close)  # type: ignore[attr-defined]
        return store

    def admitted(self, store: OrchestratorStore, *, event_id: str, turn_id: str, now=NOW):
        result = store.admit_transcript(event_id=event_id, turn_id=turn_id, payload=payload(), now=now)
        assert result.task is not None
        return result.task

    @staticmethod
    def executing(store: OrchestratorStore, task, *, now=NOW):
        current = store.transition(task.task_id, task.epoch, TaskState.ADMITTED, TaskState.ROUTING, now=now)
        current = store.transition(task.task_id, task.epoch, TaskState.ROUTING, TaskState.LOADING_CONTEXT, now=now)
        return store.transition(task.task_id, task.epoch, TaskState.LOADING_CONTEXT, TaskState.EXECUTING, now=now)
