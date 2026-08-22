from __future__ import annotations

import asyncio
import tempfile
import unittest
import uuid
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.context_assembler import ContextAssemblyError
from jarvis_orchestrator.domain import CanonicalTurnV1, ModelChunk, ModelRequest, TaskState
from jarvis_orchestrator.memory import MemoryItem, MemoryRecall
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore
from jarvis_orchestrator.translation import contains_greek


class NeverModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.requests.append(request)
        raise AssertionError("model dispatch was not expected")
        yield  # pragma: no cover


class RecordingTranslator:
    def __init__(self, replacements: Mapping[str, str]) -> None:
        self.replacements = dict(replacements)
        self.calls: list[dict[str, str]] = []

    async def translate(self, *, task_id: str, epoch: int, items):
        del task_id, epoch
        self.calls.append(dict(items))
        return {
            component_id: self.replacements[component_id]
            for component_id in items
            if component_id in self.replacements
        }


class TranslationServiceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = OrchestratorStore(self.root / "orchestrator.sqlite3")
        self.addCleanup(self.store.close)

    def config(self, *, enabled: bool) -> OrchestratorConfig:
        document: dict[str, object] = {
            "database_path": str(self.store.path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "response_language": "en",
        }
        if enabled:
            document.update(
                {
                    "input_translation_mode": "opus_ct2",
                    "input_translation_model_root": str(self.root / "model"),
                    "input_translation_manifest_file": str(self.root / "manifest.json"),
                    "input_translation_manifest_sha256": "1" * 64,
                }
            )
        return OrchestratorConfig.from_test_mapping(document)

    def admit(self, text: str, language: str = "el"):
        result = self.store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload={"text": text, "language": {"code": language}},
        )
        assert result.task is not None
        return result.task

    def test_projects_current_history_and_recall_without_mutating_sources(self) -> None:
        translator = RecordingTranslator(
            {
                "current": "What is the system status?",
                "history.0.user": "Remember my important tasks.",
                "history.0.assistant": "I will remember them.",
                "recall.0": "The important task is tomorrow at 18:30.",
            }
        )
        service = OrchestratorService(
            self.config(enabled=True),
            self.store,
            BudgetedModelPort(self.store, NeverModel()),
            input_translator=translator,
        )
        task = self.admit("Ποια είναι η κατάσταση του συστήματος;")
        history = (
            CanonicalTurnV1(
                turn_id="prior-turn",
                user_text="Να θυμάσαι τις σημαντικές εργασίες μου.",
                user_language="el",
                assistant_text="Θα τις θυμάμαι.",
            ),
        )
        recall = MemoryRecall(
            items=(
                MemoryItem(
                    "Η σημαντική εργασία είναι αύριο στις 18:30.",
                    0.9,
                ),
            ),
            provider="test",
        )

        with patch.object(
            self.store,
            "recent_canonical_turns",
            return_value=history,
        ):
            context = asyncio.run(
                service._final_local_context(
                    task,
                    "Ποια είναι η κατάσταση του συστήματος;",
                    recall,
                    None,
                )
            )

        self.assertEqual(
            translator.calls,
            [
                {
                    "current": "Ποια είναι η κατάσταση του συστήματος;",
                    "history.0.user": "Να θυμάσαι τις σημαντικές εργασίες μου.",
                    "history.0.assistant": "Θα τις θυμάμαι.",
                    "recall.0": "Η σημαντική εργασία είναι αύριο στις 18:30.",
                }
            ],
        )
        rendered = str(tuple(dict(message) for message in context.messages))
        self.assertFalse(contains_greek(rendered))
        self.assertIn("What is the system status?", rendered)
        self.assertIn("tomorrow at 18:30", rendered)
        self.assertEqual(history[0].user_text, "Να θυμάσαι τις σημαντικές εργασίες μου.")
        self.assertEqual(
            self.store.task_input(task.task_id)["text"],
            "Ποια είναι η κατάσταση του συστήματος;",
        )

    def test_latin_only_input_bypasses_translator(self) -> None:
        translator = RecordingTranslator({})
        service = OrchestratorService(
            self.config(enabled=True),
            self.store,
            BudgetedModelPort(self.store, NeverModel()),
            input_translator=translator,
        )
        task = self.admit("yes", language="el")

        context = asyncio.run(
            service._final_local_context(task, "yes", MemoryRecall(), None)
        )

        self.assertEqual(translator.calls, [])
        self.assertIn("yes", str(context.messages))

    def test_invalid_projection_fails_closed_before_context_assembly(self) -> None:
        translator = RecordingTranslator({})
        service = OrchestratorService(
            self.config(enabled=True),
            self.store,
            BudgetedModelPort(self.store, NeverModel()),
            input_translator=translator,
        )
        task = self.admit("Απάντησε.")

        with self.assertRaises(ContextAssemblyError) as raised:
            asyncio.run(
                service._final_local_context(task, "Απάντησε.", MemoryRecall(), None)
            )

        self.assertEqual(raised.exception.code, "input_translation_invalid")

    def test_greek_without_enabled_translator_emits_no_model_or_speech(self) -> None:
        model = NeverModel()
        service = OrchestratorService(
            self.config(enabled=False),
            self.store,
            BudgetedModelPort(self.store, model),
        )
        task = self.admit("Απάντησε στα αγγλικά.")

        terminal = asyncio.run(service.execute_task(task.task_id))

        self.assertEqual(terminal.state, TaskState.FAILED)
        self.assertEqual(model.requests, [])
        events = self.store.claim_outbox(task.conversation_id, "translation-disabled")
        self.assertFalse(
            any(
                event.event_type in {"response.started", "response.speech.commit"}
                for event in events
            )
        )
        errors = [event for event in events if event.event_type == "server.error"]
        self.assertEqual([event.payload["code"] for event in errors], ["input_translation_unavailable"])
        cancelled = [
            event for event in events if event.event_type == "response.cancelled"
        ]
        self.assertEqual(len(cancelled), 1)
        self.assertFalse(cancelled[0].payload["partial_output"])
if __name__ == "__main__":
    unittest.main()
