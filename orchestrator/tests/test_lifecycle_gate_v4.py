from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.anamnesis_shadow import (
    DETERMINISTIC_LIFECYCLE_GATE_V3_CONTRACT_SHA256,
    AnamnesisShadowAdapter, DeterministicOnlyLifecycleWriterV3,
    DeterministicFirstLifecycleWriter, LifecycleCandidate, LifecycleDecision, reconstruct_active_shadow_memories,
    LifecycleObservation, LifecycleOperation, LocalStructuredLifecycleWriter,
)
from jarvis_orchestrator.cli import _run_components
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import ModelChunk, ModelUsage, TaskState
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.lifecycle_gate import (
    DETERMINISTIC_LIFECYCLE_GATE_V4_CONTRACT_SHA256,
    DETERMINISTIC_ONLY_LIFECYCLE_WRITER_V4_CONTRACT_SHA256,
    DeterministicLifecycleGateV4, DeterministicOnlyLifecycleWriterV4,
)
from jarvis_orchestrator.memory import MemoryObservation, MemoryOperation
from jarvis_orchestrator.memory_release import ShadowHistoryReaderFacade
from jarvis_orchestrator.store import OrchestratorStore


# Authored regression cases, not an unseen quality/qualification benchmark.
CASES = (
    ("Forgetfulness matters.", "en", "ignore"),
    ("Please forgetting is common.", "en", "ignore"),
    ("Remembering details takes practice.", "en", "ignore"),
    ("Ξεχάσετε αυτά.", "el", "ignore"),
    ("Update my memoryless example.", "en", "ignore"),
    ("Remove this memoryless function.", "en", "ignore"),
    ("Correction:", "en", "fallback"),
    ("Correction: ...", "en", "fallback"),
    ("Διόρθωση:", "el", "fallback"),
    ("Διορθωση: !!!", "el", "fallback"),
    ("Forget:", "en", "fallback"),
    ("Ξέχασε...", "el", "fallback"),
    ("Forget my preference, maybe.", "en", "fallback"),
    ("Correction: perhaps, use tea.", "en", "fallback"),
    ("Remember that (possibly) the key moved.", "en", "fallback"),
    ("Ξέχασε την προτίμησή μου, ίσως.", "el", "fallback"),
    ("Διόρθωση: μάλλον, προτιμώ νερό.", "el", "fallback"),
    ("Να θυμασαι οτι (ισως) αλλαξε η διευθυνση.", "el", "fallback"),
    ("Remember thatched roofs are traditional.", "en", "create"),
    ("Να θυμάσαι οτιδήποτε σου ζήτησα.", "el", "create"),
    ("Could you remember that the lift is out of service?", "en", "create"),
    ("Θα ήθελα να θυμάσαι ότι προτιμώ τον πρώτο όροφο.", "el", "create"),
    ("Correction:coffee replaces tea.", "en", "replace_0"),
    ("Διόρθωση:προτιμώ τον δεύτερο όροφο.", "el", "replace_0"),
    ("Forget my stored drink.", "en", "cancel_0"),
    ("Ξέχασε την προτίμησή μου.", "el", "cancel_0"),
    ("Delete my stored drink from memory.", "en", "cancel_0"),
    ("Διέγραψε το ρόφημα από τη μνήμη.", "el", "cancel_0"),
)


def encoded(decision):
    if decision is None:
        return "fallback"
    return decision.operation.value + ("" if decision.active_index is None else f"_{decision.active_index}")


class StaticModel:
    async def stream(self, request):
        yield ModelChunk("Done.", ModelUsage(input_tokens=3, output_tokens=2))


class LifecycleGateV4Tests(unittest.TestCase):
    def test_bilingual_counterexamples_and_valid_commands(self):
        gate = DeterministicLifecycleGateV4()
        for text, language, expected in CASES:
            with self.subTest(text=text):
                self.assertEqual(encoded(gate.decide(LifecycleObservation(text, language),
                                 (LifecycleCandidate(0, "stored preference", language),))), expected)

    def test_historical_48_case_matrix_is_a_read_only_compatibility_check(self):
        path = Path(__file__).resolve().parents[1] / "eval/lifecycle_shadow_natural_v3.jsonl"
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "6bd4ec1643666335e3f5cbcaaf4cb44af6ab04006c26520274bc236db39812be")
        for row in map(json.loads, raw.splitlines()):
            with self.subTest(case=row["id"]):
                candidates = tuple(LifecycleCandidate(i, text, row["language"]) for i, text in enumerate(row["active"]))
                self.assertEqual(encoded(DeterministicLifecycleGateV4().decide(
                    LifecycleObservation(row["text"], row["language"]), candidates)), row["expected"])

    def test_target_count_and_deterministic_only_ambiguity_preserve_authority_boundary(self):
        observation = LifecycleObservation("Correction: coffee.", "en")
        gate = DeterministicLifecycleGateV4()
        self.assertEqual(encoded(gate.decide(observation, ())), "ignore")
        active = (LifecycleCandidate(0, "tea", "en"), LifecycleCandidate(1, "city", "en"))
        self.assertIsNone(gate.decide(observation, active))
        writer = DeterministicOnlyLifecycleWriterV4()
        self.assertEqual(asyncio.run(writer.decide(observation, active)), LifecycleDecision(LifecycleOperation.IGNORE))
        self.assertEqual(writer.contract_sha256, DETERMINISTIC_ONLY_LIFECYCLE_WRITER_V4_CONTRACT_SHA256)
        self.assertEqual(gate.contract_sha256, DETERMINISTIC_LIFECYCLE_GATE_V4_CONTRACT_SHA256)
        self.assertEqual(gate.contract_sha256, "94cd53ac6f9d4d314d4521ac21d06658cce6bcd0b601b8fdf6211ba5adf52e56")
        self.assertNotEqual(gate.contract_sha256, DETERMINISTIC_LIFECYCLE_GATE_V3_CONTRACT_SHA256)

    def test_current_runtime_wires_v4_in_both_existing_shadow_modes(self):
        for mode in ("deterministic_only", "structured_fallback"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config = OrchestratorConfig.from_test_mapping({
                    "database_path": str(Path(directory) / "state.sqlite3"),
                    "control_bearer_token": "control-private", "inference_bearer_token": "inference-private",
                    "memory_shadow_enabled": True, "memory_shadow_mode": mode,
                })
                store = OrchestratorStore(config.database_path)
                try:
                    with patch("jarvis_orchestrator.cli.InferenceControlClient.capabilities", return_value=SimpleNamespace(
                        text=True, stream=True, tools=False, json_schema=True, json_object=True,
                    )), patch("jarvis_orchestrator.pydantic_adapter.build_local_pydantic_port", return_value=SimpleNamespace(aclose=AsyncMock())), patch(
                        "jarvis_orchestrator.cli._run_servers", new=AsyncMock()
                    ) as servers:
                        asyncio.run(_run_components(config, store, voice_auth_key=object()))
                    adapter = servers.call_args.kwargs["shadow_memory"]
                    observation = MemoryObservation(*(str(uuid.uuid4()) for _ in range(4)),
                                                    text="Forgetfulness matters.", language="en")
                    proposal = asyncio.run(adapter.observe(observation))
                    self.assertEqual(proposal.operation, MemoryOperation.IGNORE)
                    expected = (DETERMINISTIC_ONLY_LIFECYCLE_WRITER_V4_CONTRACT_SHA256 if mode == "deterministic_only"
                                else DeterministicFirstLifecycleWriter(LocalStructuredLifecycleWriter(None),
                                     gate=DeterministicLifecycleGateV4()).contract_sha256)
                    self.assertEqual(proposal.writer_contract_sha256, expected)
                    service = OrchestratorService(config, store, BudgetedModelPort(store, StaticModel()), memory=adapter)
                    legacy = OrchestratorService(config, store, BudgetedModelPort(store, StaticModel()), memory=AnamnesisShadowAdapter(
                        ShadowHistoryReaderFacade(store.memory_shadow_history), DeterministicOnlyLifecycleWriterV3(),
                    ))
                    for i, text in enumerate(("Remember that tea is preferred.", "Forgetfulness matters.")):
                        task = store.admit_transcript(event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4()),
                               payload={"text": text, "language": {"code": "en"}}).task
                        self.assertIsNotNone(task)
                        selected_service = legacy if i == 0 else service
                        self.assertEqual(asyncio.run(selected_service.execute_task(task.task_id)).state, TaskState.COMPLETED)
                    active = reconstruct_active_shadow_memories(store.memory_shadow_history(task.conversation_id))
                    self.assertEqual(len(active), 1)
                    self.assertEqual(active[0].source_text, "Remember that tea is preferred.")
                    audit = store.memory_shadow_audit(task.task_id)
                    self.assertEqual(audit["payload"]["proposal"]["operation"], "ignore")
                    self.assertEqual(audit["payload"]["proposal"]["writer_contract_sha256"], expected)
                    first_payload = store.memory_shadow_history(task.conversation_id)[0]["payload"]
                    self.assertEqual(first_payload["proposal"]["writer_contract_sha256"], DeterministicOnlyLifecycleWriterV3().contract_sha256)
                    self.assertEqual(store.memory_record_count(task.conversation_id), 0)
                finally:
                    store.close()


if __name__ == "__main__":
    unittest.main()
