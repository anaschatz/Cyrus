from __future__ import annotations

import sqlite3
import unittest
from datetime import timedelta
import uuid

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase

from jarvis_orchestrator.domain import OrchestratorError, TaskState, utc_now, utc_text
from jarvis_orchestrator.store import SCHEMA_VERSION
from jarvis_orchestrator.tool_contracts import ToolRegistryV1
from jarvis_orchestrator.tool_crypto import InMemoryToolKeyProvider, ToolPayloadCipherV1
from jarvis_orchestrator.tool_store import ToolStoreV1
from jarvis_orchestrator.tool_state import ToolRunEvent, ToolRunState
from jarvis_voice.action_approval import (
    ACTION_APPROVAL_PROTOCOL,
    ActionApprovalEnvelope,
    ActionApprovalEventFactory,
)


class ToolStoreV6Tests(StoreCase, unittest.TestCase):
    def tool_store(self, store):
        return ToolStoreV1(
            store,
            cipher=ToolPayloadCipherV1(InMemoryToolKeyProvider()),
            registry=ToolRegistryV1.build(
                root_ids=("documents",),
                app_ids=("calendar", "finder", "mail", "preview", "reminders", "textedit"),
            ),
        )

    def test_schema_is_v6_and_scheduler_tables_survive(self) -> None:
        store = self.make_store(tool_authority_enabled=True)
        self.assertEqual(SCHEMA_VERSION, 10)
        with store._lock:
            version = store._connection.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
            names = {row[0] for row in store._connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(version, "10")
        self.assertIn("scheduler_jobs", names)
        self.assertIn("tool_sensitive_payloads", names)
        self.assertIn("content_provenance", names)

    def test_create_run_is_atomic_idempotent_and_preview_is_one_use(self) -> None:
        store = self.make_store(tool_authority_enabled=True)
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        tools = self.tool_store(store)
        tool_call_id = str(uuid.uuid4())
        idempotency_key = str(uuid.uuid4())
        created = tools.create_tool_run(
            task_id=task.task_id,
            epoch=task.epoch,
            tool_call_id=tool_call_id,
            tool_id="note.capture@1",
            arguments={"title": "Private", "body": "Body"},
            preview={"title": "Private", "body": "Body", "utf8_bytes": 4, "utf16_units": 4, "storage_domain": "JARVIS_PRIVATE_NOTES"},
            idempotency_key=idempotency_key,
            policy_sha256="a" * 64,
            scheduler_contract_sha256="b" * 64,
            user_intent_binding_sha256="c" * 64,
            now=NOW,
        )
        self.assertEqual(store.get_task(task.task_id).state, TaskState.AWAITING_APPROVAL)
        with store._lock:
            approval_count = store._connection.execute("SELECT COUNT(*) FROM approvals WHERE kind='tool'").fetchone()[0]
            raw_arguments = store._connection.execute("SELECT arguments_json FROM approvals WHERE approval_id=?", (created["approval_id"],)).fetchone()[0]
        self.assertEqual(approval_count, 1)
        self.assertNotIn("Body", raw_arguments)
        preview = tools.consume_preview(
            approval_id=created["approval_id"],
            presentation_id=created["presentation_id"],
            preview_nonce=created["preview_fetch_nonce"],
            epoch=task.epoch,
            now=NOW,
        )
        self.assertEqual(preview["body"], "Body")
        with self.assertRaises(Exception):
            tools.consume_preview(
                approval_id=created["approval_id"],
                presentation_id=created["presentation_id"],
                preview_nonce=created["preview_fetch_nonce"],
                epoch=task.epoch,
                now=NOW,
            )

    def test_global_approval_queue_limit_applies_to_action_producer(self) -> None:
        store = self.make_store(tool_authority_enabled=True)
        task = self.executing(
            store,
            self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())),
        )
        with store._transaction() as connection:
            for index in range(64):
                approval_id = str(uuid.uuid4())
                connection.execute(
                    """INSERT INTO approval_arbiter_queue(
                       approval_id,producer_protocol,binding_sha256,state,
                       created_at,expires_at,updated_at
                       ) VALUES(?,?,?,'WAITING_FOR_CHANNEL',?,?,?)""",
                    (
                        approval_id,
                        "test.approval/1",
                        f"{index:064x}",
                        "2030-01-01T00:00:00.000Z",
                        "2031-01-01T00:00:00.000Z",
                        "2030-01-01T00:00:00.000Z",
                    ),
                )
        tools = self.tool_store(store)
        with self.assertRaisesRegex(OrchestratorError, "APPROVAL_QUEUE_FULL"):
            tools.create_tool_run(
                task_id=task.task_id,
                epoch=task.epoch,
                tool_call_id=str(uuid.uuid4()),
                tool_id="file.open@1",
                arguments={"file_ref": str(uuid.uuid4())},
                preview={"display_name": "a.txt"},
                idempotency_key=str(uuid.uuid4()),
                policy_sha256="a" * 64,
                scheduler_contract_sha256="b" * 64,
                user_intent_binding_sha256="c" * 64,
                now=NOW,
            )
        with store._lock:
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM approvals WHERE kind='tool'"
                ).fetchone()[0],
                0,
            )

    def test_restart_after_dispatch_commit_becomes_ambiguous(self) -> None:
        store = self.make_store(tool_authority_enabled=True)
        task = self.executing(store, self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())))
        tools = self.tool_store(store)
        created = tools.create_tool_run(
            task_id=task.task_id,
            epoch=task.epoch,
            tool_call_id=str(uuid.uuid4()),
            tool_id="file.open@1",
            arguments={"file_ref": str(uuid.uuid4())},
            preview={"display_name": "a.txt"},
            idempotency_key=str(uuid.uuid4()),
            policy_sha256="a" * 64,
            scheduler_contract_sha256="b" * 64,
            user_intent_binding_sha256="c" * 64,
            now=NOW,
        )
        with store._transaction() as connection:
            connection.execute("UPDATE tool_executions SET status='APPROVED' WHERE execution_id=?", (created["execution_id"],))
        tools.transition(created["execution_id"], ToolRunEvent.PERMIT_COMMITTED, reason_code="ok", now=NOW)
        self.assertEqual(tools.recover_uncertain(now=NOW), 1)
        with store._lock:
            state = store._connection.execute("SELECT status FROM tool_executions WHERE execution_id=?", (created["execution_id"],)).fetchone()[0]
        self.assertEqual(state, ToolRunState.AMBIGUOUS.value)

    def test_tool_output_and_n_hop_local_answer_never_enter_cloud_history(self) -> None:
        store = self.make_store(tool_authority_enabled=True)
        first = self.executing(
            store,
            self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())),
        )
        tools = self.tool_store(store)
        created = tools.create_tool_run(
            task_id=first.task_id,
            epoch=first.epoch,
            tool_call_id=str(uuid.uuid4()),
            tool_id="note.capture@1",
            arguments={"title": "Private", "body": "Body"},
            preview={"title": "Private", "body": "Body"},
            idempotency_key=str(uuid.uuid4()),
            policy_sha256="a" * 64,
            scheduler_contract_sha256="b" * 64,
            user_intent_binding_sha256="c" * 64,
            now=NOW,
        )
        tools.decide_action_approval(
            approval_id=created["approval_id"], epoch=first.epoch, decision="deny", now=NOW
        )
        store.start_response(first.task_id, first.epoch, "el-GR", now=NOW)
        store.append_response_event(
            first.task_id,
            first.epoch,
            "response.text.delta",
            {"chunk_index": 0, "text": "Δεν έγινε ενέργεια."},
            now=NOW,
        )
        store.complete_response(
            first.task_id,
            first.epoch,
            "Δεν έγινε ενέργεια.",
            chunk_count=1,
            trailing_speech=[],
            last_unit_index=0,
            now=NOW,
        )
        self.assertEqual(store.cloud_safe_history(first.conversation_id), ())

        second = self.executing(
            store,
            self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())),
        )
        store.start_response(second.task_id, second.epoch, "el-GR", now=NOW)
        store.append_response_event(
            second.task_id,
            second.epoch,
            "response.text.delta",
            {"chunk_index": 0, "text": "Περίληψη."},
            now=NOW,
        )
        store.complete_response(
            second.task_id,
            second.epoch,
            "Περίληψη.",
            chunk_count=1,
            trailing_speech=[],
            last_unit_index=0,
            now=NOW,
        )
        self.assertEqual(store.cloud_safe_history(first.conversation_id), ())

    def test_action_transport_ack_is_separate_from_business_decision(self) -> None:
        store = self.make_store(tool_authority_enabled=True)
        task = self.executing(
            store,
            self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())),
        )
        tools = self.tool_store(store)
        created = tools.create_tool_run(
            task_id=task.task_id,
            epoch=task.epoch,
            tool_call_id=str(uuid.uuid4()),
            tool_id="note.capture@1",
            arguments={"title": "Private", "body": "Body"},
            preview={"title": "Private", "body": "Body"},
            idempotency_key=str(uuid.uuid4()),
            policy_sha256="a" * 64,
            scheduler_contract_sha256="b" * 64,
            user_intent_binding_sha256="c" * 64,
            now=NOW,
        )
        session_id = str(uuid.uuid4())
        promoted = tools.promote_next_action_approval(session_id, now=NOW)
        self.assertIsNotNone(promoted)
        requested = ActionApprovalEnvelope(
            ACTION_APPROVAL_PROTOCOL,
            "action.approval.requested",
            promoted["event_id"],
            session_id,
            1,
            "2026-08-10T09:00:00.000Z",
            promoted["payload"],
        )
        tools.acknowledge_action_extension(requested.event_id)
        decision = ActionApprovalEventFactory(session_id).create(
            "action.approval.decision",
            {
                "approval_id": created["approval_id"],
                "presentation_id": created["presentation_id"],
                "requested_event_id": requested.event_id,
                "approval_binding_sha256": created["approval_binding_sha256"],
                "decision_nonce": requested.payload["decision_nonce"],
                "decision": "deny",
            },
            sent_at="2026-08-10T09:00:00.000Z",
        )
        result = tools.decide_action_transport(
            decision_event_id=decision.event_id,
            semantic_sha256=decision.semantic_sha256,
            approval_id=decision.payload["approval_id"],
            presentation_id=decision.payload["presentation_id"],
            requested_event_id=decision.payload["requested_event_id"],
            approval_binding_sha256=decision.payload["approval_binding_sha256"],
            decision_nonce=decision.payload["decision_nonce"],
            decision=decision.payload["decision"],
            session_id=session_id,
            now=NOW,
        )
        self.assertEqual(result["decision"], "deny")
        event_types = {item["event_type"] for item in tools.claim_action_extension_events()}
        self.assertEqual(
            event_types,
            {"action.approval.decision_ack", "action.approval.decision_result"},
        )


    def prepared_action(self):
        store = self.make_store(tool_authority_enabled=True)
        task = self.executing(store, self.admitted(
            store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4())
        ))
        tools = self.tool_store(store)
        arguments = dict(
            task_id=task.task_id, epoch=task.epoch,
            tool_call_id=str(uuid.uuid4()), tool_id="note.capture@1",
            arguments={"title": "Private", "body": "Body"},
            preview={"title": "Private", "body": "Body"},
            idempotency_key=str(uuid.uuid4()), policy_sha256="a" * 64,
            scheduler_contract_sha256="b" * 64,
            user_intent_binding_sha256="c" * 64, now=NOW,
            approval_ttl=timedelta(seconds=2),
        )
        created = tools.create_tool_run(**arguments)
        session_id = str(uuid.uuid4())
        promoted = tools.promote_next_action_approval(session_id, now=NOW)
        decision = dict(
            decision_event_id=str(uuid.uuid4()), semantic_sha256="d" * 64,
            approval_id=created["approval_id"],
            presentation_id=created["presentation_id"],
            requested_event_id=promoted["event_id"],
            approval_binding_sha256=created["approval_binding_sha256"],
            decision_nonce=promoted["payload"]["decision_nonce"],
            decision="approve", session_id=session_id, now=NOW,
        )
        return store, task, tools, created, arguments, decision

    def test_action_decision_rolls_back_if_result_cannot_be_enqueued(self):
        store, task, tools, created, _, decision = self.prepared_action()
        tables = ("approvals", "tool_executions", "approval_arbiter_queue",
                  "action_approval_presentations", "action_approval_inbox",
                  "action_approval_outbox", "tasks", "action_audit")
        def snapshot():
            with store._lock:
                return {name: [tuple(row) for row in store._connection.execute(
                    f"SELECT * FROM {name} ORDER BY rowid"
                )] for name in tables}
        before = snapshot()
        with store._transaction() as connection:
            connection.execute("""CREATE TRIGGER reject_decision_result
                BEFORE INSERT ON action_approval_outbox
                WHEN NEW.event_type='action.approval.decision_result'
                BEGIN SELECT RAISE(ABORT, 'injected outbox failure'); END""")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected outbox failure"):
            tools.decide_action_transport(**decision)
        self.assertEqual(snapshot(), before)
        with store._transaction() as connection:
            connection.execute("DROP TRIGGER reject_decision_result")
        result = tools.decide_action_transport(**decision)
        self.assertEqual(result["decision"], "approve")
        after = snapshot()
        self.assertEqual(tools.decide_action_transport(**decision), {"duplicate": True})
        self.assertEqual(snapshot(), after)
        self.assertEqual(store.get_task(task.task_id).state, TaskState.EXECUTING)
        self.assertEqual(tools.task_run(task.task_id, task.epoch)["status"], "APPROVED")
        self.assertEqual({e["event_type"] for e in tools.claim_action_extension_events()},
                         {"action.approval.decision_ack", "action.approval.decision_result"})

    def test_late_action_decision_persists_expiration_and_releases_queue(self):
        for transport in (False, True):
            with self.subTest(transport=transport):
                store, task, tools, created, _, decision = self.prepared_action()
                late = NOW + timedelta(seconds=3)
                if transport:
                    result = tools.decide_action_transport(**(decision | {"now": late}))
                    self.assertTrue(result["expired"])
                    events = tools.claim_action_extension_events()
                    payload = next(e["payload"] for e in events
                                   if e["event_type"] == "action.approval.decision_result")
                    self.assertEqual(payload["result"], "rejected")
                    self.assertEqual(payload["reason_code"], "approval_expired")
                else:
                    with self.assertRaisesRegex(OrchestratorError, "tool.approval_expired"):
                        tools.decide_action_approval(approval_id=created["approval_id"],
                            epoch=task.epoch, decision="approve", now=late)
                with store._lock:
                    for table in ("approvals", "tool_executions"):
                        status = store._connection.execute(
                            f"SELECT status FROM {table} WHERE approval_id=?",
                            (created["approval_id"],)).fetchone()[0]
                        self.assertEqual(status, "EXPIRED")
                    row = store._connection.execute(
                        "SELECT state FROM approval_arbiter_queue WHERE approval_id=?",
                        (created["approval_id"],)).fetchone()
                    self.assertEqual(row[0], "EXPIRED")
                self.assertEqual(store.get_task(task.task_id).state, TaskState.CANCELLED)
                self.assertNotIn("action.approval.requested",
                                 {e["event_type"] for e in tools.claim_action_extension_events()})

    def test_unacknowledged_action_event_is_retried_after_one_second(self):
        store, _, tools, _, _, decision = self.prepared_action()
        event_id = decision["requested_event_id"]
        tools.finish_action_extension_write(event_id, complete=True)
        with store._transaction() as connection:
            connection.execute("UPDATE action_approval_outbox SET last_sent_at=? WHERE event_id=?",
                               (utc_text(utc_now() - timedelta(seconds=2)), event_id))
        self.assertIn(event_id, {e["event_id"] for e in tools.claim_action_extension_events()})
        tools.acknowledge_action_extension(event_id)
        self.assertNotIn(event_id, {e["event_id"] for e in tools.claim_action_extension_events()})

    def test_approved_action_cannot_dispatch_after_approval_deadline(self):
        store, task, tools, created, _, _ = self.prepared_action()
        tools.decide_action_approval(approval_id=created["approval_id"],
            epoch=task.epoch, decision="approve", now=NOW)
        with self.assertRaisesRegex(OrchestratorError, "tool.approval_expired"):
            tools.issue_execution_permit(created["execution_id"], task_id=task.task_id,
                epoch=task.epoch, policy_sha256="a" * 64,
                scheduler_contract_sha256="b" * 64, now=NOW + timedelta(seconds=3))
        self.assertEqual(tools.task_run(task.task_id, task.epoch)["status"], "APPROVED")
        with store._lock:
            self.assertIsNone(store._connection.execute(
                "SELECT permit_id FROM tool_executions WHERE execution_id=?",
                (created["execution_id"],)).fetchone()[0])

    def test_replayed_tool_run_requires_all_approval_bindings(self):
        _, _, tools, created, arguments, _ = self.prepared_action()
        self.assertEqual(tools.create_tool_run(**arguments)["execution_id"], created["execution_id"])
        for change in ({"policy_sha256": "e" * 64},
                       {"scheduler_contract_sha256": "e" * 64},
                       {"user_intent_binding_sha256": "e" * 64},
                       {"preview": {"title": "Different", "body": "Body"}}):
            with self.subTest(change=change), self.assertRaisesRegex(
                OrchestratorError, "tool.idempotency_conflict"
            ):
                tools.create_tool_run(**(arguments | change))

if __name__ == "__main__":
    unittest.main()
