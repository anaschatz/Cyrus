from __future__ import annotations

import unittest
import uuid
from datetime import timedelta

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase, payload

from jarvis_orchestrator.domain import OrchestratorError, ReasonCode, StaleEpochError, TaskState


class StoreContractTests(StoreCase, unittest.TestCase):
    def test_admission_duplicate_ack_is_stable_and_does_not_grow(self) -> None:
        store = self.make_store()
        event_id, turn_id = str(uuid.uuid4()), str(uuid.uuid4())
        accepted = store.admit_transcript(event_id=event_id, turn_id=turn_id, payload=payload(), now=NOW)
        duplicate = store.admit_transcript(event_id=event_id, turn_id=turn_id, payload=payload(), now=NOW)
        again = store.admit_transcript(event_id=event_id, turn_id=turn_id, payload=payload(), now=NOW)
        self.assertEqual(duplicate.ack_event_id, again.ack_event_id)
        self.assertNotEqual(accepted.ack_event_id, duplicate.ack_event_id)
        with store._lock:
            bindings = store._connection.execute(
                "SELECT response_id FROM outbox WHERE acked_event_id=? ORDER BY ordinal",
                (event_id,),
            ).fetchall()
            count = store._connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE acked_event_id=?", (event_id,)
            ).fetchone()[0]
        self.assertEqual(count, 2)
        self.assertEqual(
            [row["response_id"] for row in bindings],
            [accepted.task.response_id, accepted.task.response_id],
        )

    def test_conflicting_duplicate_is_rejected_without_new_task(self) -> None:
        store = self.make_store()
        event_id, turn_id = str(uuid.uuid4()), str(uuid.uuid4())
        first = store.admit_transcript(event_id=event_id, turn_id=turn_id, payload=payload(), now=NOW)
        with self.assertRaisesRegex(OrchestratorError, "protocol.conflicting_duplicate"):
            store.admit_transcript(event_id=event_id, turn_id=turn_id, payload=payload("άλλο"), now=NOW)
        self.assertEqual(store.active_conversation()["active_task_id"], first.task.task_id)

    def test_cancellation_epoch_cas_rejects_late_history_and_outbox(self) -> None:
        store = self.make_store()
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        result = store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
        self.assertEqual(result["epoch"], task.epoch + 1)
        with self.assertRaises(StaleEpochError):
            store.start_response(task.task_id, task.epoch, "el-GR", now=NOW)
        events = store.claim_outbox(task.conversation_id, str(uuid.uuid4()), now=NOW)
        self.assertEqual(
            [event.event_type for event in events],
            ["server.ack", "response.cancelled"],
        )
        self.assertEqual(events[-1].payload["partial_output"], False)

    def test_claimed_output_is_rechecked_after_cancellation_commit(self) -> None:
        store = self.make_store()
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        store.start_response(task.task_id, task.epoch, "el-GR", now=NOW)
        store.append_response_event(
            task.task_id, task.epoch, "response.text.delta", {"chunk_index": 0, "text": "stale"}, now=NOW
        )
        claimed = store.claim_outbox(task.conversation_id, "writer", now=NOW)
        stale_event = next(event for event in claimed if event.event_type == "response.text.delta")
        store.cancel_task(task.task_id, task.epoch, ReasonCode.USER_CANCELLED, now=NOW)
        self.assertFalse(store.outbox_claim_is_current(stale_event.event_id, "writer", now=NOW))
        with self.assertRaises(StaleEpochError):
            store.finish_outbox_attempt(stale_event.event_id, "writer", complete=False)

    def test_completion_commits_history_and_both_done_rows_atomically(self) -> None:
        store = self.make_store()
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        store.start_response(task.task_id, task.epoch, "el-GR", now=NOW)
        store.append_response_event(
            task.task_id, task.epoch, "response.text.delta", {"chunk_index": 0, "text": "Έτοιμο."}, now=NOW
        )
        completed = store.complete_response(
            task.task_id,
            task.epoch,
            "Έτοιμο.",
            chunk_count=1,
            trailing_speech=[(1, "el-GR", "Έτοιμο.")],
            last_unit_index=1,
            now=NOW,
        )
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual(store.canonical_conversation(task.conversation_id).turns[0].assistant_text, "Έτοιμο.")
        events = store.claim_outbox(task.conversation_id, str(uuid.uuid4()), now=NOW)
        types = [event.event_type for event in events]
        self.assertIn("response.text.done", types)
        self.assertIn("response.speech.done", types)

    def test_output_indices_are_strictly_contiguous_and_terminal_is_immutable(self) -> None:
        store = self.make_store()
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        store.start_response(task.task_id, task.epoch, "el-GR", now=NOW)
        with self.assertRaises(OrchestratorError):
            store.append_response_event(
                task.task_id,
                task.epoch,
                "response.text.delta",
                {"chunk_index": 1, "text": "gap"},
                now=NOW,
            )
        store.append_response_event(
            task.task_id,
            task.epoch,
            "response.text.delta",
            {"chunk_index": 0, "text": "ok"},
            now=NOW,
        )
        completed = store.complete_response(
            task.task_id, task.epoch, "ok", chunk_count=1, trailing_speech=[], last_unit_index=0, now=NOW
        )
        with self.assertRaises(OrchestratorError):
            store.append_response_event(
                task.task_id,
                task.epoch,
                "response.text.delta",
                {"chunk_index": 1, "text": "late"},
                now=NOW,
            )
        self.assertEqual(completed.state, TaskState.COMPLETED)

    def test_failure_wire_mapping_differs_before_and_after_partial_output(self) -> None:
        store = self.make_store()
        before = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        store.terminate_failure(
            before.task_id,
            before.epoch,
            state=TaskState.FAILED,
            reason=ReasonCode.BACKEND_FAILURE,
            error_code="backend_failure",
            now=NOW,
        )
        events = store.claim_outbox(before.conversation_id, "before", now=NOW)
        self.assertEqual(
            [event.event_type for event in events],
            ["server.ack", "server.error", "response.cancelled"],
        )

        after = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        store.start_response(after.task_id, after.epoch, "el-GR", now=NOW)
        store.append_response_event(
            after.task_id, after.epoch, "response.text.delta", {"chunk_index": 0, "text": "partial"}, now=NOW
        )
        store.terminate_failure(
            after.task_id,
            after.epoch,
            state=TaskState.BUDGET_EXHAUSTED,
            reason=ReasonCode.ACTIVE_BUDGET_EXHAUSTED,
            error_code="active_budget_exhausted",
            now=NOW,
        )
        visible = store.claim_outbox(after.conversation_id, "after", now=NOW)
        self.assertEqual(
            [event.event_type for event in visible],
            ["server.ack", "response.cancelled"],
        )

    def test_internal_diagnostic_is_closed_and_content_free(self) -> None:
        store = self.make_store()
        task = self.executing(
            store,
            self.admitted(
                store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())
            ),
        )
        store.record_internal_diagnostic(task.task_id, task.epoch, "type_error", now=NOW)
        with store._lock:
            row = store._connection.execute(
                "SELECT event_type,payload_json FROM task_events WHERE task_id=? ORDER BY event_sequence DESC LIMIT 1",
                (task.task_id,),
            ).fetchone()
        self.assertEqual(row["event_type"], "diagnostic.internal_failure")
        self.assertEqual(row["payload_json"], '{"code":"type_error"}')
        with self.assertRaises(ValueError):
            store.record_internal_diagnostic(
                task.task_id, task.epoch, "private exception text", now=NOW
            )

    def test_outbox_replays_on_new_connection_until_exact_delivery_expiry(self) -> None:
        store = self.make_store()
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        store.start_response(task.task_id, task.epoch, "el-GR", now=NOW)
        store.complete_response(task.task_id, task.epoch, "", chunk_count=0, trailing_speech=[], last_unit_index=0, now=NOW)
        first = store.claim_outbox(task.conversation_id, "first", now=NOW)
        for event in first:
            store.finish_outbox_attempt(event.event_id, "first", complete=True)
        second = store.claim_outbox(task.conversation_id, "second", now=NOW + timedelta(seconds=29))
        self.assertEqual([event.event_id for event in first], [event.event_id for event in second])
        for event in second:
            store.finish_outbox_attempt(event.event_id, "second", complete=True)
        self.assertEqual(store.claim_outbox(task.conversation_id, "late", now=NOW + timedelta(seconds=30)), [])
        store.expire_delivery(now=NOW + timedelta(seconds=30))
        with store._lock:
            states = {row[0] for row in store._connection.execute("SELECT state FROM outbox WHERE response_id=?", (task.response_id,))}
        self.assertEqual(states, {"NON_REPLAYABLE"})

    def test_approval_pauses_active_clock_and_offline_decision_waits_for_reconnect(self) -> None:
        store = self.make_store(
            active_budget_seconds=120, tool_authority_enabled=True
        )
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())), now=NOW)
        approval = store.create_approval(
            task.task_id, task.epoch, kind="tool", name="lamp", arguments={"on": True}, now=NOW + timedelta(seconds=60)
        )
        paused = store.get_task(task.task_id)
        self.assertEqual(paused.state, TaskState.AWAITING_APPROVAL)
        self.assertEqual(paused.active_elapsed_seconds, 60)
        decision = store.decide_approval(
            approval, task.epoch, "approve", client_connected=False, now=NOW + timedelta(minutes=5)
        )
        self.assertEqual(decision["task_state"], "DECISION_WAITING_RECONNECT")
        resumed = store.set_reconnected(task.task_id, task.epoch, now=NOW + timedelta(minutes=5, seconds=10))
        self.assertEqual(resumed.state, TaskState.EXECUTING)
        self.assertEqual(resumed.remaining_active_seconds(now=NOW + timedelta(minutes=5, seconds=20)), 50)

    def test_awaiting_approval_ignores_disconnect_but_decision_reconnect_expires(self) -> None:
        store = self.make_store(tool_authority_enabled=True)
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        approval = store.create_approval(task.task_id, task.epoch, kind="tool", name="lamp", arguments={}, now=NOW)
        store.set_disconnected(task.task_id, task.epoch, now=NOW)
        self.assertIsNone(store.get_task(task.task_id).disconnect_expires_at)
        store.decide_approval(approval, task.epoch, "deny", client_connected=False, now=NOW)
        self.assertEqual(store.expire_deadlines(now=NOW + timedelta(seconds=30)), 1)
        cancelled = store.get_task(task.task_id)
        self.assertEqual(cancelled.state, TaskState.CANCELLED)
        events = store.claim_outbox(task.conversation_id, "decision-timeout", now=NOW + timedelta(seconds=30))
        self.assertEqual(events[-1].payload["reason_code"], "approval_reconnect_timeout")

    def test_offline_completion_replaces_disconnect_deadline_with_delivery_window(self) -> None:
        store = self.make_store()
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        store.set_disconnected(task.task_id, task.epoch, now=NOW)
        store.start_response(task.task_id, task.epoch, "el-GR", now=NOW + timedelta(seconds=10))
        completed = store.complete_response(
            task.task_id,
            task.epoch,
            "",
            chunk_count=0,
            trailing_speech=[],
            last_unit_index=0,
            now=NOW + timedelta(seconds=10),
        )
        self.assertIsNone(completed.disconnect_expires_at)
        self.assertEqual(completed.delivery_expires_at, "2026-08-10T09:00:40.000Z")

    def test_active_and_absolute_clocks_expire_independently(self) -> None:
        store = self.make_store(active_budget_seconds=120)
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())), now=NOW)
        self.assertEqual(store.expire_deadlines(now=NOW + timedelta(seconds=120)), 1)
        self.assertEqual(store.get_task(task.task_id).state, TaskState.BUDGET_EXHAUSTED)

    def test_approval_expiry_is_durable_and_cancels_with_new_epoch(self) -> None:
        store = self.make_store()
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        approval = store.create_approval(
            task.task_id,
            task.epoch,
            kind="research",
            name="search",
            arguments={},
            now=NOW,
            ttl=timedelta(seconds=10),
        )
        with self.assertRaisesRegex(OrchestratorError, "approval_expired"):
            store.decide_approval(
                approval,
                task.epoch,
                "approve",
                client_connected=True,
                now=NOW + timedelta(seconds=10),
            )
        self.assertEqual(store.get_task(task.task_id).state, TaskState.CANCELLED)
        with store._lock:
            status = store._connection.execute(
                "SELECT status FROM approvals WHERE approval_id=?", (approval,)
            ).fetchone()[0]
        self.assertEqual(status, "EXPIRED")
        self.assertEqual(store.active_conversation()["epoch"], task.epoch + 1)


if __name__ == "__main__":
    unittest.main()
