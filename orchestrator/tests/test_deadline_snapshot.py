from __future__ import annotations

import unittest
import uuid
from datetime import timedelta

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.domain import TaskState
from jarvis_orchestrator.store import OrchestratorStore


class DeadlineSnapshotTests(StoreCase, unittest.TestCase):
    def test_committed_reconnect_or_extension_prevents_stale_expiration(self):
        for action in ("reconnect", "extend"):
            with self.subTest(action=action):
                store = self.make_store()
                task = self.executing(
                    store,
                    self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())),
                )
                store.set_disconnected(
                    task.task_id, task.epoch, now=NOW, ttl=timedelta(seconds=1)
                )
                peer = OrchestratorStore(store.path)
                self.addCleanup(peer.close)
                interleaved = []
                errors = []
                selected = NOW + timedelta(seconds=2)

                def before_write_transaction(sql):
                    if sql != "BEGIN IMMEDIATE" or interleaved:
                        return
                    interleaved.append(action)
                    try:
                        if action == "reconnect":
                            peer.set_reconnected(task.task_id, task.epoch, now=selected)
                        else:
                            peer.set_disconnected(
                                task.task_id, task.epoch, now=selected,
                                ttl=timedelta(seconds=30),
                            )
                    except Exception as exc:
                        errors.append(exc)

                # Trace runs before SQLite executes BEGIN: the peer commits at
                # the boundary where the old implementation held a stale read.
                store._connection.set_trace_callback(before_write_transaction)
                try:
                    expired = store.expire_deadlines(now=selected)
                finally:
                    store._connection.set_trace_callback(None)
                self.assertEqual(errors, [])
                self.assertEqual(interleaved, [action])
                self.assertEqual(expired, 0)
                current = store.get_task(task.task_id)
                self.assertEqual(current.state, TaskState.EXECUTING)
                self.assertFalse(current.tombstoned)
                if action == "extend":
                    self.assertEqual(
                        store.expire_deadlines(now=selected + timedelta(seconds=30)), 1
                    )


if __name__ == "__main__":
    unittest.main()
