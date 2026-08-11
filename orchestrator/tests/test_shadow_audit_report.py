from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import _bootstrap  # noqa: F401
from jarvis_orchestrator.anamnesis_shadow import (
    LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256,
    AnamnesisShadowAdapter,
    DeterministicFirstLifecycleWriter,
    DeterministicLifecycleGateV3,
    LifecycleDecision,
    LifecycleOperation,
)
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import ModelChunk, ModelUsage, canonical_sha256
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.memory_release import (
    MEMORY_RELEASE_POLICY_SHA256,
    ShadowHistoryReaderFacade,
)
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.shadow_audit import build_shadow_trial_report
from jarvis_orchestrator.store import OrchestratorStore


class _StaticModel:
    async def stream(self, request) -> AsyncIterator[ModelChunk]:
        del request
        yield ModelChunk("Έτοιμο.", ModelUsage(input_tokens=3, output_tokens=2))


class _RejectingFallback:
    contract_sha256 = LOCAL_LIFECYCLE_WRITER_CONTRACT_SHA256

    async def decide(self, observation, active_memories) -> LifecycleDecision:
        del observation, active_memories
        raise AssertionError("fallback was not expected")


class ShadowAuditReportTests(unittest.TestCase):
    def _database(self, directory: str) -> Path:
        path = Path(directory) / "orchestrator.sqlite3"
        store = OrchestratorStore(path)
        config = OrchestratorConfig.from_mapping(
            {
                "database_path": str(path),
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
                "memory_shadow_enabled": True,
            }
        )
        writer = DeterministicFirstLifecycleWriter(
            _RejectingFallback(),
            gate=DeterministicLifecycleGateV3(),
        )
        service = OrchestratorService(
            config,
            store,
            BudgetedModelPort(store, _StaticModel()),
            memory=AnamnesisShadowAdapter(
                ShadowHistoryReaderFacade(store.memory_shadow_history), writer
            ),
        )
        for language, text in (
            ("en", "Remember that I prefer tea."),
            ("el", "Διόρθωση: προτιμώ καφέ."),
            ("en", "Forget my drink preference."),
        ):
            task = store.admit_transcript(
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                payload={
                    "text": text,
                    "language": {"code": language, "score": 1.0},
                },
            ).task
            assert task is not None
            asyncio.run(service.execute_task(task.task_id))
        store.close()
        return path

    def test_report_is_read_only_content_free_and_integrity_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._database(directory)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            report = build_shadow_trial_report(path)
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertEqual(report["completed_turns"], 3)
            self.assertEqual(report["audited_turns"], 3)
            self.assertEqual(report["audited_conversations"], 1)
            self.assertEqual(
                report["event_counts"]["memory.shadow.proposed"],
                3,
            )
            self.assertEqual(
                report["operation_counts"],
                {"upsert": 2, "cancel": 1, "ignore": 0},
            )
            self.assertEqual(report["active_shadow_memories"], 0)
            self.assertEqual(report["authoritative_memory_records"], 0)
            self.assertEqual(report["authority_violations"], 0)
            self.assertEqual(
                report["release_policy_counts"],
                {MEMORY_RELEASE_POLICY_SHA256: 3},
            )
            self.assertTrue(report["integrity_passed"])
            self.assertFalse(report["sample_sufficient"])
            self.assertEqual(
                report["decision"],
                "insufficient_real_session_evidence",
            )
            encoded = json.dumps(report, ensure_ascii=False)
            self.assertNotIn("προτιμώ καφέ", encoded)
            self.assertNotIn("prefer tea", encoded)
            self.assertFalse(report["contains_user_content"])
            self.assertFalse(report["contains_stable_user_or_conversation_ids"])

    def test_authority_tamper_rejects_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._database(directory)
            connection = __import__("sqlite3").connect(path)
            row = connection.execute(
                """SELECT event_sequence,payload_json FROM task_events
                   WHERE event_type='memory.shadow.proposed'
                   ORDER BY event_sequence LIMIT 1"""
            ).fetchone()
            payload = json.loads(row[1])
            payload["authoritative_state_changed"] = True
            connection.execute(
                "UPDATE task_events SET payload_json=? WHERE event_sequence=?",
                (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    row[0],
                ),
            )
            connection.commit()
            connection.close()
            report = build_shadow_trial_report(path)
            self.assertFalse(report["integrity_passed"])
            self.assertEqual(report["authority_violations"], 1)
            self.assertEqual(report["decision"], "shadow_trial_rejected")

    def test_policy_and_malformed_proposal_tamper_fail_closed(self) -> None:
        for mutation in ("policy", "proposal"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = self._database(directory)
                connection = __import__("sqlite3").connect(path)
                row = connection.execute(
                    """SELECT event_sequence,payload_json FROM task_events
                       WHERE event_type='memory.shadow.proposed'
                       ORDER BY event_sequence LIMIT 1"""
                ).fetchone()
                payload = json.loads(row[1])
                if mutation == "policy":
                    payload["release_policy_sha256"] = "0" * 64
                else:
                    payload["proposal"] = "malformed"
                    payload["proposal_sha256"] = canonical_sha256("malformed")
                connection.execute(
                    "UPDATE task_events SET payload_json=? WHERE event_sequence=?",
                    (
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        row[0],
                    ),
                )
                connection.commit()
                connection.close()
                report = build_shadow_trial_report(path)
                self.assertFalse(report["integrity_passed"])
                self.assertGreaterEqual(report["invalid_rows"], 1)
                self.assertEqual(report["decision"], "shadow_trial_rejected")

    def test_explicit_retrospective_record_is_reported_but_not_shadow_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._database(directory)
            store = OrchestratorStore(path)
            try:
                with store._lock:
                    row = store._connection.execute(
                        """SELECT t.task_id,t.conversation_id,t.epoch,i.event_id
                           FROM tasks t JOIN inbox i USING(task_id)
                           WHERE t.state='COMPLETED'
                           ORDER BY t.epoch DESC LIMIT 1"""
                    ).fetchone()
                assert row is not None
                store.add_memory_record(
                    conversation_id=row["conversation_id"],
                    task_id=row["task_id"],
                    epoch=row["epoch"],
                    source_event_id="explicit-memory-source",
                    scoped_user_id="local-user",
                    content="explicit remembered fact",
                    metadata={"origin": "explicit_command"},
                )
            finally:
                store.close()
            report = build_shadow_trial_report(path)
            self.assertEqual(report["retrospective_memory_records"], 1)
            self.assertEqual(report["authority_violations"], 0)
            self.assertTrue(report["integrity_passed"])

    def test_missing_database_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.sqlite3"
            with self.assertRaises(FileNotFoundError):
                build_shadow_trial_report(missing)


if __name__ == "__main__":
    unittest.main()
