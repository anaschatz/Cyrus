from __future__ import annotations

import sqlite3
import unittest
import uuid

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.domain import ReasonCode, utc_text
from jarvis_orchestrator.scheduler_contract import PackagedTimezoneProvider, compile_proposal
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.scheduler_store import SchedulerStore
from test_scheduler_v5 import proposal_input, proposal_output


class ConversationDeleteBoundaryTests(StoreCase, unittest.TestCase):
    def test_scheduler_link_does_not_block_history_deletion_or_delete_jobs(self):
        for decision in (None, "approve", "deny"):
            with self.subTest(decision=decision):
                store = self.make_store()
                scheduler = SchedulerStore(
                    store, cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
                    zones=PackagedTimezoneProvider(),
                )
                task = self.executing(store, self.admitted(
                    store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())
                ))
                request_id = str(uuid.uuid4())
                compiled = compile_proposal(
                    proposal_input(request_id, utc_text(NOW)), proposal_output(request_id),
                    compiler_qualification_sha256="1" * 64, zones=scheduler.zones,
                )
                created = scheduler.create_proposal(compiled, task_id=task.task_id, epoch=task.epoch, now=NOW)
                if decision:
                    scheduler.decide_approval(
                        approval_id=created["approval_id"], proposal_id=created["proposal_id"],
                        proposal_sha256=created["proposal_sha256"], decision=decision,
                        cause_event_id=str(uuid.uuid4()), now=NOW,
                    )
                cancelled = store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
                jobs = [tuple(row) for row in store._connection.execute("SELECT * FROM scheduler_jobs")]
                self.assertEqual(len(jobs), int(decision == "approve"))
                store.delete_conversation(task.conversation_id, cancelled["epoch"], now=NOW)
                self.assertEqual(store._connection.execute("SELECT COUNT(*) FROM scheduler_task_links").fetchone()[0], 0)
                self.assertEqual(store._connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0], 0)
                self.assertEqual([tuple(row) for row in store._connection.execute("SELECT * FROM scheduler_jobs")], jobs)
                self.assertEqual(store._connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_large_history_deletion_does_not_expand_one_sql_parameter_per_task(self):
        store = self.make_store()
        for _ in range(12):
            task = self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4()))
            cancelled = store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
        store._connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
        store.delete_conversation(task.conversation_id, cancelled["epoch"], now=NOW)
        for table in ("conversations", "tasks", "task_events", "inbox", "outbox"):
            self.assertEqual(store._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
