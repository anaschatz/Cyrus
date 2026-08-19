from __future__ import annotations

import asyncio
import json
import threading
import unittest
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.control import ControlApplication
from jarvis_orchestrator.domain import (
    ModelChunk,
    ModelRequest,
    ModelUsage,
    OrchestratorError,
    TaskState,
    canonical_sha256,
    parse_utc,
    utc_now,
)
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.gateway import VoiceGateway
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import (
    ControlMutationKind,
    ControlMutationRequest,
    OrchestratorStore,
)
from jarvis_voice.voice_auth import VoiceAuthKey


def _config(path: Path) -> OrchestratorConfig:
    return OrchestratorConfig.from_test_mapping(
        {
            "database_path": str(path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
        }
    )


class UnusedModel:
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        del request
        yield ModelChunk("unused", ModelUsage(input_tokens=1, output_tokens=1))


def _application(store: OrchestratorStore) -> ControlApplication:
    service = OrchestratorService(
        _config(store.path),
        store,
        BudgetedModelPort(store, UnusedModel()),
    )
    return ControlApplication(service.config, service)


def _receipt_count(store: OrchestratorStore) -> int:
    with store._lock:
        return int(
            store._connection.execute(
                "SELECT COUNT(*) FROM idempotency_commands"
            ).fetchone()[0]
        )


class AtomicControlIdempotencyTests(StoreCase, unittest.TestCase):
    def test_mismatched_http_operation_binding_has_no_receipt_or_mutation(
        self,
    ) -> None:
        store = self.make_store()
        first_id = str(uuid.uuid4())
        second_id = str(uuid.uuid4())
        mismatches = (
            (
                ControlMutationRequest(
                    ControlMutationKind.CREATE_CONVERSATION
                ),
                "DELETE",
                "/v1/conversations",
            ),
            (
                ControlMutationRequest(
                    ControlMutationKind.CREATE_CONVERSATION
                ),
                "POST",
                f"/v1/conversations/{first_id}",
            ),
            (
                ControlMutationRequest(
                    ControlMutationKind.DELETE_CONVERSATION,
                    first_id,
                ),
                "DELETE",
                f"/v1/conversations/{second_id}",
            ),
            (
                ControlMutationRequest(
                    ControlMutationKind.DELETE_MEMORY_RECORD,
                    first_id,
                ),
                "DELETE",
                f"/v1/memory/records/{second_id}",
            ),
            (
                ControlMutationRequest(
                    ControlMutationKind.CANCEL_TASK,
                    first_id,
                ),
                "POST",
                f"/v1/approvals/{first_id}/decision",
            ),
            (
                ControlMutationRequest(
                    ControlMutationKind.DECIDE_APPROVAL,
                    first_id,
                ),
                "DELETE",
                f"/v1/approvals/{first_id}/decision",
            ),
        )
        statements: list[str] = []
        with store._lock:
            store._connection.set_trace_callback(statements.append)
        try:
            for request, method, path in mismatches:
                with self.subTest(kind=request.kind, method=method, path=path):
                    with self.assertRaisesRegex(
                        ValueError, "HTTP binding is invalid"
                    ):
                        store.execute_control_mutation(
                            str(uuid.uuid4()),
                            method,
                            path,
                            {},
                            0,
                            request,
                            client_connected=False,
                            now=NOW,
                        )
        finally:
            with store._lock:
                store._connection.set_trace_callback(None)
        self.assertEqual(statements, [])
        self.assertIsNone(store.active_conversation())
        self.assertEqual(_receipt_count(store), 0)

    def test_trace_has_one_begin_lookup_mutation_receipt_and_commit(self) -> None:
        store = self.make_store()
        app = _application(store)
        statements: list[str] = []
        with store._lock:
            store._connection.set_trace_callback(statements.append)
        try:
            status, _ = app.mutate(
                "POST",
                "/v1/conversations",
                {},
                str(uuid.uuid4()),
                0,
            )
        finally:
            with store._lock:
                store._connection.set_trace_callback(None)
        self.assertEqual(status, 201)
        normalized = [" ".join(statement.split()) for statement in statements]
        self.assertEqual(
            [statement for statement in normalized if statement == "BEGIN IMMEDIATE"],
            ["BEGIN IMMEDIATE"],
        )
        lookup = next(
            index
            for index, statement in enumerate(normalized)
            if statement.startswith("SELECT * FROM idempotency_commands")
        )
        mutation = next(
            index
            for index, statement in enumerate(normalized)
            if statement.startswith("INSERT INTO conversations")
        )
        receipt = next(
            index
            for index, statement in enumerate(normalized)
            if statement.startswith("INSERT INTO idempotency_commands")
        )
        self.assertLess(lookup, mutation)
        self.assertLess(mutation, receipt)
        self.assertLess(receipt, normalized.index("COMMIT"))

    def test_success_error_replay_and_conflicting_reuse_are_exact(self) -> None:
        store = self.make_store()
        app = _application(store)

        create_key = str(uuid.uuid4())
        created = app.mutate(
            "POST", "/v1/conversations", {}, create_key, 0
        )
        self.assertEqual(
            app.mutate("POST", "/v1/conversations", {}, create_key, 0),
            created,
        )
        with self.assertRaisesRegex(OrchestratorError, "idempotency_conflict"):
            app.mutate(
                "POST",
                "/v1/conversations",
                {"unexpected": True},
                create_key,
                0,
            )
        self.assertEqual(created[0], 201)
        self.assertEqual(_receipt_count(store), 1)

        task = self.admitted(
            store,
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            now=NOW,
        )
        busy_delete_key = str(uuid.uuid4())
        busy_path = f"/v1/conversations/{task.conversation_id}"
        busy = app.mutate(
            "DELETE", busy_path, None, busy_delete_key, task.epoch
        )
        self.assertEqual(
            busy,
            (
                409,
                {
                    "error": {
                        "code": "conversation_busy",
                        "retryable": True,
                    }
                },
            ),
        )

        cancel_key = str(uuid.uuid4())
        cancel_path = f"/v1/tasks/{task.task_id}/cancel"
        cancelled = app.mutate(
            "POST",
            cancel_path,
            {"reason_code": "user_cancelled"},
            cancel_key,
            task.epoch,
        )
        self.assertEqual(
            app.mutate(
                "POST",
                cancel_path,
                {"reason_code": "user_cancelled"},
                cancel_key,
                task.epoch,
            ),
            cancelled,
        )
        self.assertEqual(cancelled[1]["epoch"], task.epoch + 1)
        self.assertEqual(
            app.mutate(
                "DELETE", busy_path, None, busy_delete_key, task.epoch
            ),
            busy,
        )
        self.assertIsNotNone(store.active_conversation())

        delete_key = str(uuid.uuid4())
        deleted = app.mutate(
            "DELETE",
            busy_path,
            None,
            delete_key,
            task.epoch + 1,
        )
        self.assertEqual(deleted, (204, None))
        self.assertEqual(
            app.mutate(
                "DELETE",
                busy_path,
                None,
                delete_key,
                task.epoch + 1,
            ),
            deleted,
        )
        self.assertIsNone(store.active_conversation())
        self.assertEqual(_receipt_count(store), 4)

    def test_research_decision_is_atomic_and_tool_decision_is_route_blocked(
        self,
    ) -> None:
        store = self.make_store(tool_authority_enabled=True)
        app = _application(store)
        selected = utc_now()
        research_task = self.executing(
            store,
            self.admitted(
                store,
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                now=selected,
            ),
            now=selected,
        )
        research_approval = store.create_approval(
            research_task.task_id,
            research_task.epoch,
            kind="research",
            name="bounded_research",
            arguments={},
            now=selected,
        )
        key = str(uuid.uuid4())
        path = f"/v1/approvals/{research_approval}/decision"
        decided = app.mutate(
            "POST", path, {"decision": "approve"}, key, research_task.epoch
        )
        self.assertEqual(
            app.mutate(
                "POST", path, {"decision": "approve"}, key, research_task.epoch
            ),
            decided,
        )
        self.assertEqual(decided[0], 200)
        with store._lock:
            approval_status = store._connection.execute(
                "SELECT status FROM approvals WHERE approval_id=?",
                (research_approval,),
            ).fetchone()[0]
            decision_events = store._connection.execute(
                "SELECT COUNT(*) FROM task_events "
                "WHERE task_id=? AND event_type='DECISION_WAITING_RECONNECT'",
                (research_task.task_id,),
            ).fetchone()[0]
            effects = json.loads(
                store._connection.execute(
                    "SELECT effects_json FROM canonical_turns WHERE task_id=?",
                    (research_task.task_id,),
                ).fetchone()[0]
            )
        self.assertEqual(approval_status, "APPROVED")
        self.assertEqual(decision_events, 1)
        self.assertEqual(len(effects), 1)

        tool_task = self.executing(
            store,
            self.admitted(
                store,
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                now=selected + timedelta(seconds=1),
            ),
            now=selected + timedelta(seconds=1),
        )
        tool_approval = store.create_approval(
            tool_task.task_id,
            tool_task.epoch,
            kind="tool",
            name="local_action",
            arguments={},
            now=selected + timedelta(seconds=1),
        )
        tool_key = str(uuid.uuid4())
        tool_path = f"/v1/approvals/{tool_approval}/decision"
        blocked = app.mutate(
            "POST",
            tool_path,
            {"decision": "approve"},
            tool_key,
            tool_task.epoch,
        )
        self.assertEqual(
            blocked,
            (
                409,
                {
                    "error": {
                        "code": "tool.approval_route_required",
                        "retryable": False,
                    }
                },
            ),
        )
        self.assertEqual(
            app.mutate(
                "POST",
                tool_path,
                {"decision": "approve"},
                tool_key,
                tool_task.epoch,
            ),
            blocked,
        )
        with store._lock:
            tool_row = store._connection.execute(
                "SELECT status FROM approvals WHERE approval_id=?",
                (tool_approval,),
            ).fetchone()
            task_state = store._connection.execute(
                "SELECT state FROM tasks WHERE task_id=?", (tool_task.task_id,)
            ).fetchone()[0]
            effects_json = store._connection.execute(
                "SELECT effects_json FROM canonical_turns WHERE task_id=?",
                (tool_task.task_id,),
            ).fetchone()[0]
        self.assertEqual(tool_row["status"], "PENDING")
        self.assertEqual(task_state, TaskState.AWAITING_APPROVAL.value)
        self.assertEqual(effects_json, "[]")

    def test_crash_after_each_mutation_before_receipt_rolls_back_everything(
        self,
    ) -> None:
        create_store = self.make_store()
        create_app = _application(create_store)
        with patch.object(
            OrchestratorStore,
            "_store_command_tx",
            side_effect=RuntimeError("crash_after_mutation"),
        ):
            with self.assertRaisesRegex(RuntimeError, "crash_after_mutation"):
                create_app.mutate(
                    "POST",
                    "/v1/conversations",
                    {},
                    str(uuid.uuid4()),
                    0,
                )
        self.assertIsNone(create_store.active_conversation())
        self.assertEqual(_receipt_count(create_store), 0)

        cancel_store = self.make_store()
        cancel_app = _application(cancel_store)
        cancel_task = self.admitted(
            cancel_store,
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            now=NOW,
        )
        with patch.object(
            OrchestratorStore,
            "_store_command_tx",
            side_effect=RuntimeError("crash_after_mutation"),
        ):
            with self.assertRaisesRegex(RuntimeError, "crash_after_mutation"):
                cancel_app.mutate(
                    "POST",
                    f"/v1/tasks/{cancel_task.task_id}/cancel",
                    {"reason_code": "user_cancelled"},
                    str(uuid.uuid4()),
                    cancel_task.epoch,
                )
        self.assertEqual(
            cancel_store.get_task(cancel_task.task_id).state,
            TaskState.ADMITTED,
        )
        self.assertEqual(
            cancel_store.active_conversation()["epoch"], cancel_task.epoch
        )
        self.assertEqual(_receipt_count(cancel_store), 0)

        delete_store = self.make_store()
        delete_app = _application(delete_store)
        conversation = delete_store.create_conversation(0, now=NOW)
        with patch.object(
            OrchestratorStore,
            "_store_command_tx",
            side_effect=RuntimeError("crash_after_mutation"),
        ):
            with self.assertRaisesRegex(RuntimeError, "crash_after_mutation"):
                delete_app.mutate(
                    "DELETE",
                    f"/v1/conversations/{conversation['conversation_id']}",
                    None,
                    str(uuid.uuid4()),
                    0,
                )
        self.assertEqual(
            delete_store.active_conversation()["conversation_id"],
            conversation["conversation_id"],
        )
        self.assertEqual(_receipt_count(delete_store), 0)

        approval_store = self.make_store()
        approval_app = _application(approval_store)
        approval_task = self.executing(
            approval_store,
            self.admitted(
                approval_store,
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                now=NOW,
            ),
            now=NOW,
        )
        approval_id = approval_store.create_approval(
            approval_task.task_id,
            approval_task.epoch,
            kind="research",
            name="bounded_research",
            arguments={},
            now=NOW,
        )
        with patch.object(
            OrchestratorStore,
            "_store_command_tx",
            side_effect=RuntimeError("crash_after_mutation"),
        ):
            with self.assertRaisesRegex(RuntimeError, "crash_after_mutation"):
                approval_app.mutate(
                    "POST",
                    f"/v1/approvals/{approval_id}/decision",
                    {"decision": "deny"},
                    str(uuid.uuid4()),
                    approval_task.epoch,
                )
        with approval_store._lock:
            approval_status = approval_store._connection.execute(
                "SELECT status FROM approvals WHERE approval_id=?",
                (approval_id,),
            ).fetchone()[0]
            effects_json = approval_store._connection.execute(
                "SELECT effects_json FROM canonical_turns WHERE task_id=?",
                (approval_task.task_id,),
            ).fetchone()[0]
        self.assertEqual(approval_status, "PENDING")
        self.assertEqual(
            approval_store.get_task(approval_task.task_id).state,
            TaskState.AWAITING_APPROVAL,
        )
        self.assertEqual(effects_json, "[]")
        self.assertEqual(_receipt_count(approval_store), 0)

    def test_two_store_connections_same_key_commit_one_effect(self) -> None:
        first_store = self.make_store()
        second_store = OrchestratorStore(first_store.path)
        self.addCleanup(second_store.close)
        first_app = _application(first_store)
        second_app = _application(second_store)
        key = str(uuid.uuid4())
        barrier = threading.Barrier(2)

        def create(app: ControlApplication) -> tuple[int, object | None]:
            barrier.wait(timeout=5)
            return app.mutate("POST", "/v1/conversations", {}, key, 0)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(create, first_app)
            second_future = executor.submit(create, second_app)
            first = first_future.result(timeout=5)
            second = second_future.result(timeout=5)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 201)
        with first_store._lock:
            conversation_count = first_store._connection.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
        self.assertEqual(conversation_count, 1)
        self.assertEqual(_receipt_count(first_store), 1)
        with self.assertRaisesRegex(OrchestratorError, "idempotency_conflict"):
            second_app.mutate(
                "POST",
                "/v1/conversations",
                {"different": True},
                key,
                0,
            )

    def test_receipt_expires_at_exactly_twenty_four_hours(self) -> None:
        store = self.make_store()
        key = str(uuid.uuid4())
        request = ControlMutationRequest(
            ControlMutationKind.CREATE_CONVERSATION
        )
        first = store.execute_control_mutation(
            key,
            "POST",
            "/v1/conversations",
            {},
            0,
            request,
            client_connected=False,
            now=NOW,
        )
        replay = store.execute_control_mutation(
            key,
            "POST",
            "/v1/conversations",
            {},
            0,
            request,
            client_connected=False,
            now=NOW + timedelta(hours=24, microseconds=-1),
        )
        self.assertEqual(replay, first)
        replacement = store.execute_control_mutation(
            key,
            "POST",
            "/v1/conversations",
            {},
            0,
            request,
            client_connected=False,
            now=NOW + timedelta(hours=24),
        )
        self.assertNotEqual(replacement, first)
        with store._lock:
            rows = store._connection.execute(
                "SELECT * FROM idempotency_commands"
            ).fetchall()
            conversation_count = store._connection.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
        self.assertEqual(len(rows), 1)
        self.assertEqual(conversation_count, 2)
        self.assertEqual(
            parse_utc(rows[0]["expires_at"]) - parse_utc(rows[0]["created_at"]),
            timedelta(hours=24),
        )
        self.assertEqual(rows[0]["method"], "POST")
        self.assertEqual(rows[0]["canonical_path"], "/v1/conversations")
        self.assertEqual(rows[0]["request_sha256"], canonical_sha256({}))
        self.assertEqual(rows[0]["expected_epoch"], 0)

    def test_expired_approval_and_410_receipt_commit_and_replay_together(
        self,
    ) -> None:
        store = self.make_store()
        task = self.executing(
            store,
            self.admitted(
                store,
                event_id=str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                now=NOW,
            ),
            now=NOW,
        )
        approval_id = store.create_approval(
            task.task_id,
            task.epoch,
            kind="research",
            name="expiring_research",
            arguments={},
            now=NOW,
            ttl=timedelta(seconds=1),
        )
        key = str(uuid.uuid4())
        path = f"/v1/approvals/{approval_id}/decision"
        request = ControlMutationRequest(
            ControlMutationKind.DECIDE_APPROVAL,
            approval_id,
        )
        first = store.execute_control_mutation(
            key,
            "POST",
            path,
            {"decision": "approve"},
            task.epoch,
            request,
            client_connected=True,
            now=NOW + timedelta(seconds=2),
        )
        self.assertEqual(
            first,
            (
                410,
                '{"error":{"code":"approval_expired","retryable":false}}',
            ),
        )
        with store._lock:
            approval_status = store._connection.execute(
                "SELECT status FROM approvals WHERE approval_id=?",
                (approval_id,),
            ).fetchone()[0]
            event_count = store._connection.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id=?",
                (task.task_id,),
            ).fetchone()[0]
        self.assertEqual(approval_status, "EXPIRED")
        self.assertEqual(store.get_task(task.task_id).state, TaskState.CANCELLED)
        self.assertEqual(store.active_conversation()["epoch"], task.epoch + 1)
        self.assertEqual(_receipt_count(store), 1)
        replay = store.execute_control_mutation(
            key,
            "POST",
            path,
            {"decision": "approve"},
            task.epoch,
            request,
            client_connected=True,
            now=NOW + timedelta(seconds=3),
        )
        self.assertEqual(replay, first)
        with store._lock:
            replay_event_count = store._connection.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id=?",
                (task.task_id,),
            ).fetchone()[0]
        self.assertEqual(replay_event_count, event_count)
        self.assertEqual(_receipt_count(store), 1)

    def test_replayed_schedule_is_deduplicated_by_gateway_task_id(self) -> None:
        class BlockingService:
            def __init__(self) -> None:
                self.calls = 0
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def execute_task(self, task_id: str) -> str:
                self.calls += 1
                self.started.set()
                await self.release.wait()
                return task_id

        async def scenario() -> None:
            service = BlockingService()
            gateway = VoiceGateway(
                service,  # type: ignore[arg-type]
                auth_key=VoiceAuthKey(1, b"a" * 32),
                business_admission=lambda: True,
            )
            gateway._track_execution("same-task")
            await service.started.wait()
            first = gateway._execution_tasks["same-task"]
            gateway._track_execution("same-task")
            self.assertIs(gateway._execution_tasks["same-task"], first)
            self.assertEqual(service.calls, 1)
            service.release.set()
            self.assertEqual(await first, "same-task")

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
