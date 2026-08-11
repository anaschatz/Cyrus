from __future__ import annotations

import sqlite3
import json
import tempfile
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from test_research_contract import fixture

from jarvis_orchestrator.domain import ReasonCode, StaleEpochError
from jarvis_orchestrator.store import OrchestratorStore


class ResearchStoreTests(StoreCase, unittest.TestCase):
    def test_plan_approval_bundle_documents_and_effect_are_hash_bound(self) -> None:
        store = self.make_store()
        task = self.executing(
            store,
            self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())),
        )
        plan, document, _, bundle = fixture()
        created = store.create_research_approval(task.task_id, task.epoch, plan, now=NOW)
        pending = store.list_pending_approvals(now=NOW)[0]
        self.assertEqual(pending["arguments"]["plan_hash"], plan.plan_hash)
        self.assertEqual(pending["arguments"]["deadline_seconds"], 45)
        self.assertTrue(pending["arguments"]["public_services"])
        store.decide_approval(created["approval_id"], task.epoch, "approve", client_connected=True, now=NOW)
        reservation = store.begin_research_run(task.task_id, task.epoch, request_key=plan.plan_hash, now=NOW)
        store.commit_research_result(
            reservation,
            task.task_id,
            task.epoch,
            bundle=bundle,
            cited_documents={"source-1": document},
            now=NOW,
        )
        run = store.research_run(task.task_id)
        self.assertEqual((run["status"], run["bundle_hash"]), ("SUFFICIENT", bundle.sha256))
        self.assertEqual(store.research_documents(task.task_id)["source-1"].text, document.text)
        with store._lock:
            effects = json.loads(
                store._connection.execute(
                    "SELECT effects_json FROM canonical_turns WHERE task_id=?", (task.task_id,)
                ).fetchone()[0]
            )
        self.assertEqual(effects[-1]["result_sha256"], bundle.sha256)
        self.assertEqual(effects[-1]["arguments_sha256"], plan.plan_hash)

    def test_late_research_result_after_tombstone_leaves_no_bundle_or_document(self) -> None:
        store = self.make_store()
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        plan, document, _, bundle = fixture()
        created = store.create_research_approval(task.task_id, task.epoch, plan, now=NOW)
        store.decide_approval(created["approval_id"], task.epoch, "approve", client_connected=True, now=NOW)
        reservation = store.begin_research_run(task.task_id, task.epoch, request_key=plan.plan_hash, now=NOW)
        store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
        with self.assertRaises(StaleEpochError):
            store.commit_research_result(
                reservation,
                task.task_id,
                task.epoch,
                bundle=bundle,
                cited_documents={"source-1": document},
                now=NOW,
            )
        self.assertIsNone(store.research_run(task.task_id)["bundle_json"])
        self.assertEqual(store.research_documents(task.task_id), {})

    def test_conversation_deletion_removes_research_rows(self) -> None:
        store = self.make_store()
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        plan, _, _, _ = fixture()
        store.create_research_approval(task.task_id, task.epoch, plan, now=NOW)
        cancelled = store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
        store.delete_conversation(task.conversation_id, cancelled["epoch"], now=NOW)
        with store._lock:
            self.assertEqual(store._connection.execute("SELECT COUNT(*) FROM research_runs").fetchone()[0], 0)
            self.assertEqual(store._connection.execute("SELECT COUNT(*) FROM research_documents").fetchone()[0], 0)

    def test_v2_database_migrates_through_research_to_current_schema_v7(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            first = OrchestratorStore(path)
            with first._lock:
                first._connection.execute("DROP TABLE research_documents")
                first._connection.execute("DROP TABLE research_runs")
                first._connection.execute("UPDATE schema_meta SET value='2' WHERE key='schema_version'")
            first.close()
            second = OrchestratorStore(path)
            self.addCleanup(second.close)
            self.assertEqual(second._connection.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0], "7")
            tables = {row[0] for row in second._connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"research_runs", "research_documents"} <= tables)


if __name__ == "__main__":
    unittest.main()
