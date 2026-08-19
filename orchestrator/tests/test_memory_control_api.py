from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from helpers import NOW
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.control import ControlApplication, handler_factory
from jarvis_orchestrator.domain import (
    ModelChunk,
    ModelRequest,
    ModelUsage,
    OrchestratorError,
    TaskState,
)
from jarvis_orchestrator.memory_crypto import (
    InMemoryMemoryEpochKeyProvider,
    MemoryEnvelopeCipherV1,
)
from jarvis_orchestrator.memory_runtime import InjectedMemoryCryptoRuntime
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import (
    ControlMutationKind,
    ControlMutationRequest,
    OrchestratorStore,
)


class UnusedModel:
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        del request
        yield ModelChunk("unused", ModelUsage(input_tokens=1, output_tokens=1))


class TransactionSpyProvider(InMemoryMemoryEpochKeyProvider):
    def __init__(self) -> None:
        super().__init__()
        self.stores: list[OrchestratorStore] = []
        self.calls: list[str] = []
        self.transaction_violations: list[str] = []

    def _observe(self, name: str) -> None:
        self.calls.append(name)
        if any(store._connection.in_transaction for store in self.stores):
            self.transaction_violations.append(name)

    def bootstrap(self):
        self._observe("bootstrap")
        return super().bootstrap()

    def active_version(self):
        self._observe("active_version")
        return super().active_version()

    def load(self, version: int):
        self._observe("load")
        return super().load(version)

    def exists(self, version: int):
        self._observe("exists")
        return super().exists(version)


class MemoryControlApiTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "state.sqlite3"
        self.provider = TransactionSpyProvider()
        self.runtime = InjectedMemoryCryptoRuntime(
            self.provider.install_id,
            MemoryEnvelopeCipherV1(self.provider),
        )
        self.store = self._open_store()
        self.store.prepare_memory_storage(now=NOW)
        self.application = self._application(self.store)

    def _open_store(self, *, prepared: bool = False) -> OrchestratorStore:
        store = OrchestratorStore(self.path, memory_crypto=self.runtime)
        self.provider.stores.append(store)
        self.addCleanup(store.close)
        if prepared:
            store.prepare_memory_storage(now=NOW)
        return store

    @staticmethod
    def _config(path: Path) -> OrchestratorConfig:
        return OrchestratorConfig.from_test_mapping(
            {
                "database_path": str(path),
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
            }
        )

    @classmethod
    def _application(cls, store: OrchestratorStore) -> ControlApplication:
        config = cls._config(store.path)
        service = OrchestratorService(
            config,
            store,
            BudgetedModelPort(store, UnusedModel()),
        )
        return ControlApplication(config, service)

    @staticmethod
    def _remember(
        store: OrchestratorStore,
        content: str,
        *,
        now=NOW,
    ) -> str:
        admitted = store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload={
                "text": f"Remember that {content}",
                "language": {"code": "en"},
            },
            now=now,
        )
        assert admitted.task is not None
        task = store.transition(
            admitted.task.task_id,
            admitted.task.epoch,
            TaskState.ADMITTED,
            TaskState.ROUTING,
            now=now,
        )
        task = store.transition(
            task.task_id,
            task.epoch,
            TaskState.ROUTING,
            TaskState.LOADING_CONTEXT,
            now=now,
        )
        task = store.transition(
            task.task_id,
            task.epoch,
            TaskState.LOADING_CONTEXT,
            TaskState.EXECUTING,
            now=now,
        )
        _, result = store.complete_explicit_memory_command(
            task.task_id,
            task.epoch,
            enabled=True,
            locale="en-US",
            now=now,
        )
        if result.status.value != "stored":
            raise AssertionError("memory record was not stored")
        matches = [
            record
            for record in store.control_memory_record_page(limit=50, now=now)["items"]
            if record["content"] == content
        ]
        if len(matches) != 1:
            raise AssertionError("stored memory record is unavailable")
        return matches[0]["record_id"]

    def _handler(self, headers: list[tuple[str, str]]):
        handler_type = handler_factory(self.application)
        result = handler_type.__new__(handler_type)
        result.client_address = ("127.0.0.1", 12345)
        message = Message()
        for name, value in headers:
            message[name] = value
        result.headers = message
        return result

    def test_list_get_cursor_projection_expiry_and_revision_binding(self) -> None:
        first_id = self._remember(self.store, "alpha is saved", now=NOW)
        second_id = self._remember(
            self.store,
            "beta is saved",
            now=NOW + timedelta(seconds=1),
        )
        self._remember(
            self.store,
            "gamma is saved",
            now=NOW + timedelta(seconds=2),
        )

        status, first_page = self.application.get("/v1/memory/records?limit=2")
        self.assertEqual(status, 200)
        self.assertEqual(len(first_page["records"]), 2)
        self.assertIsInstance(first_page["next_cursor"], str)
        for record in first_page["records"]:
            self.assertEqual(
                set(record),
                {"record_id", "content", "created_at", "expires_at"},
            )
            self.assertNotIn("key", str(record).lower())
            self.assertNotIn("hmac", str(record).lower())
            self.assertNotIn("envelope", str(record).lower())

        status, second_page = self.application.get(
            "/v1/memory/records?limit=2&after=" + first_page["next_cursor"]
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(second_page["records"]), 1)
        self.assertIsNone(second_page["next_cursor"])
        self.assertEqual(second_page["memory_revision"], first_page["memory_revision"])

        status, item = self.application.get(f"/v1/memory/records/{first_id}")
        self.assertEqual(status, 200)
        self.assertEqual(item["record"]["record_id"], first_id)
        self.assertEqual(
            set(item["record"]),
            {"record_id", "content", "created_at", "expires_at"},
        )

        token = first_page["next_cursor"]
        offset = len(token) // 2
        replacement = "A" if token[offset] != "A" else "B"
        with self.assertRaisesRegex(OrchestratorError, "invalid_cursor"):
            self.application.get(
                "/v1/memory/records?after="
                + token[:offset]
                + replacement
                + token[offset + 1 :]
            )
        with self.assertRaisesRegex(OrchestratorError, "invalid_cursor"):
            self.application.get("/v1/memory/records?after=")
        with self.assertRaisesRegex(OrchestratorError, "invalid_request"):
            self.application.get("/v1/memory/records?limit=1&limit=2")
        for invalid_limit in ("0", "51", "true"):
            with (
                self.subTest(limit=invalid_limit),
                self.assertRaisesRegex(
                    OrchestratorError,
                    "invalid_request",
                ),
            ):
                self.application.get(f"/v1/memory/records?limit={invalid_limit}")

        self._remember(
            self.store,
            "delta changes the revision",
            now=NOW + timedelta(seconds=3),
        )
        with self.assertRaisesRegex(OrchestratorError, "memory_revision_conflict"):
            self.application.get("/v1/memory/records?limit=2&after=" + token)

        after_expiry = NOW + timedelta(days=366)
        with patch("jarvis_orchestrator.store.utc_now", return_value=after_expiry):
            _, expired_page = self.application.get("/v1/memory/records")
            self.assertEqual(expired_page["records"], [])
            with self.assertRaisesRegex(OrchestratorError, "not_found"):
                self.application.get(f"/v1/memory/records/{second_id}")
            expired_delete = self.application.memory_mutate(
                "DELETE",
                f"/v1/memory/records/{second_id}",
                None,
                str(uuid.uuid4()),
                expired_page["memory_revision"],
            )
            self.assertEqual(
                expired_delete,
                (
                    404,
                    {"error": {"code": "not_found", "retryable": False}},
                ),
            )

    def test_delete_exact_replay_and_all_request_bindings(self) -> None:
        first_id = self._remember(self.store, "first delete target")
        second_id = self._remember(
            self.store,
            "second delete target",
            now=NOW + timedelta(seconds=1),
        )
        page = self.store.control_memory_record_page(limit=50, now=NOW)
        revision = page["memory_revision"]
        key = str(uuid.uuid4())
        path = f"/v1/memory/records/{first_id}"

        missing_key = str(uuid.uuid4())
        missing_path = f"/v1/memory/records/{uuid.uuid4()}"
        missing = self.application.memory_mutate(
            "DELETE",
            missing_path,
            None,
            missing_key,
            revision,
        )
        self.assertEqual(
            missing,
            (
                404,
                {"error": {"code": "not_found", "retryable": False}},
            ),
        )
        self.assertEqual(
            self.application.memory_mutate(
                "DELETE",
                missing_path,
                None,
                missing_key,
                revision,
            ),
            missing,
        )

        invalid_body_key = str(uuid.uuid4())
        invalid_body = {"unexpected_body_sha256": "0" * 64}
        rejected = self.application.memory_mutate(
            "DELETE",
            path,
            invalid_body,
            invalid_body_key,
            revision,
        )
        self.assertEqual(
            rejected,
            (
                400,
                {"error": {"code": "invalid_request", "retryable": False}},
            ),
        )
        self.assertIsNotNone(self.store.memory_record(first_id))
        with self.assertRaisesRegex(OrchestratorError, "idempotency_conflict"):
            self.application.memory_mutate(
                "DELETE",
                path,
                None,
                invalid_body_key,
                revision,
            )

        first = self.application.memory_mutate("DELETE", path, None, key, revision)
        self.assertEqual(first[0], 200)
        self.assertEqual(
            first[1],
            {
                "record_id": first_id,
                "status": "removed_from_saved_memory",
                "chat_history_changed": False,
                "physical_erasure_claimed": False,
                "memory_revision": revision + 1,
            },
        )
        self.assertEqual(
            self.application.memory_mutate("DELETE", path, None, key, revision),
            first,
        )
        prepared_session = self.store._memory_epoch_session
        self.store._memory_epoch_session = None
        try:
            self.assertEqual(
                self.application.memory_mutate(
                    "DELETE",
                    path,
                    None,
                    key,
                    revision,
                ),
                first,
            )
        finally:
            self.store._memory_epoch_session = prepared_session
        for changed_path, changed_body, changed_revision in (
            (f"/v1/memory/records/{second_id}", None, revision),
            (path, {"unexpected": True}, revision),
            (path, None, revision + 1),
        ):
            with (
                self.subTest(
                    path=changed_path,
                    body=changed_body,
                    revision=changed_revision,
                ),
                self.assertRaisesRegex(
                    OrchestratorError,
                    "idempotency_conflict",
                ),
            ):
                self.application.memory_mutate(
                    "DELETE",
                    changed_path,
                    changed_body,
                    key,
                    changed_revision,
                )
        with self.assertRaisesRegex(OrchestratorError, "idempotency_conflict"):
            self.application.mutate(
                "POST",
                "/v1/conversations",
                {},
                key,
                0,
            )

        stale_key = str(uuid.uuid4())
        stale_path = f"/v1/memory/records/{second_id}"
        stale = self.application.memory_mutate(
            "DELETE",
            stale_path,
            None,
            stale_key,
            revision,
        )
        self.assertEqual(
            stale,
            (
                409,
                {
                    "error": {
                        "code": "memory_revision_conflict",
                        "retryable": False,
                    }
                },
            ),
        )
        self.assertEqual(
            self.application.memory_mutate(
                "DELETE",
                stale_path,
                None,
                stale_key,
                revision,
            ),
            stale,
        )
        with self.store._lock:
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM memory_record_tombstones WHERE record_id=?",
                    (first_id,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM memory_record_audits "
                    "WHERE record_id=? AND action='CONTROL_DELETED'",
                    (first_id,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM idempotency_commands WHERE idempotency_key=?",
                    (key,),
                ).fetchone()[0],
                1,
            )
        self.assertIsNotNone(self.store.memory_record(second_id))

    def test_delete_receipt_expiry_keeps_durable_revision_binding(self) -> None:
        record_id = self._remember(self.store, "durable replay target")
        revision = self.store.control_memory_record_page(
            limit=1,
            now=NOW,
        )["memory_revision"]
        key = str(uuid.uuid4())
        path = f"/v1/memory/records/{record_id}"
        request = ControlMutationRequest(
            ControlMutationKind.DELETE_MEMORY_RECORD,
            record_id,
        )
        first = self.store.execute_control_mutation(
            key,
            "DELETE",
            path,
            None,
            revision,
            request,
            client_connected=False,
            now=NOW,
        )
        replay = self.store.execute_control_mutation(
            key,
            "DELETE",
            path,
            None,
            revision,
            request,
            client_connected=False,
            now=NOW + timedelta(hours=24),
        )
        self.assertEqual(replay, first)
        with self.assertRaisesRegex(OrchestratorError, "idempotency_conflict"):
            self.store.execute_control_mutation(
                key,
                "DELETE",
                path,
                {"unexpected": True},
                revision,
                request,
                client_connected=False,
                now=NOW + timedelta(hours=48),
            )
        with self.assertRaisesRegex(OrchestratorError, "idempotency_conflict"):
            self.store.execute_control_mutation(
                key,
                "DELETE",
                path,
                None,
                revision + 1,
                request,
                client_connected=False,
                now=NOW + timedelta(hours=72),
            )
        with self.assertRaisesRegex(OrchestratorError, "idempotency_conflict"):
            self.store.execute_control_mutation(
                key,
                "POST",
                "/v1/conversations",
                {},
                0,
                ControlMutationRequest(
                    ControlMutationKind.CREATE_CONVERSATION,
                ),
                client_connected=False,
                now=NOW + timedelta(hours=96),
            )
        with self.store._lock:
            receipt = self.store._connection.execute(
                "SELECT method,canonical_path FROM idempotency_commands "
                "WHERE idempotency_key=?",
                (key,),
            ).fetchone()
            self.assertEqual(tuple(receipt), ("DELETE", path))

    def test_crash_before_receipt_rolls_back_entire_delete(self) -> None:
        record_id = self._remember(self.store, "crash rollback target")
        revision = self.store.control_memory_record_page(
            limit=1,
            now=NOW,
        )["memory_revision"]
        with (
            patch.object(
                OrchestratorStore,
                "_store_command_tx",
                side_effect=RuntimeError("crash_after_memory_delete"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "crash_after_memory_delete",
            ),
        ):
            self.application.memory_mutate(
                "DELETE",
                f"/v1/memory/records/{record_id}",
                None,
                str(uuid.uuid4()),
                revision,
            )
        self.assertIsNotNone(self.store.memory_record(record_id))
        snapshot = self.store.control_memory_record_page(limit=1, now=NOW)
        self.assertEqual(snapshot["memory_revision"], revision)
        with self.store._lock:
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM memory_record_tombstones WHERE record_id=?",
                    (record_id,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM memory_record_audits "
                    "WHERE record_id=? AND action='CONTROL_DELETED'",
                    (record_id,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM idempotency_commands"
                ).fetchone()[0],
                0,
            )

    def test_two_connections_same_key_commit_one_effect_without_provider_calls(
        self,
    ) -> None:
        record_id = self._remember(self.store, "concurrent delete target")
        revision = self.store.control_memory_record_page(
            limit=1,
            now=NOW,
        )["memory_revision"]
        second_store = self._open_store(prepared=True)
        second_application = self._application(second_store)
        key = str(uuid.uuid4())
        path = f"/v1/memory/records/{record_id}"
        barrier = threading.Barrier(2)
        provider_call_count = len(self.provider.calls)

        def delete(application: ControlApplication):
            barrier.wait(timeout=5)
            return application.memory_mutate(
                "DELETE",
                path,
                None,
                key,
                revision,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(delete, self.application)
            second_future = executor.submit(delete, second_application)
            first = first_future.result(timeout=5)
            second = second_future.result(timeout=5)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 200)
        self.assertEqual(len(self.provider.calls), provider_call_count)
        self.assertEqual(self.provider.transaction_violations, [])
        with self.store._lock:
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM memory_record_tombstones WHERE record_id=?",
                    (record_id,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM memory_record_audits "
                    "WHERE record_id=? AND action='CONTROL_DELETED'",
                    (record_id,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM idempotency_commands WHERE idempotency_key=?",
                    (key,),
                ).fetchone()[0],
                1,
            )

    def test_two_connections_same_revision_allow_only_one_distinct_delete(
        self,
    ) -> None:
        first_id = self._remember(self.store, "first concurrent CAS target")
        second_id = self._remember(
            self.store,
            "second concurrent CAS target",
            now=NOW + timedelta(seconds=1),
        )
        revision = self.store.control_memory_record_page(
            limit=2,
            now=NOW,
        )["memory_revision"]
        second_store = self._open_store(prepared=True)
        second_application = self._application(second_store)
        barrier = threading.Barrier(2)

        def delete(
            application: ControlApplication,
            record_id: str,
        ) -> tuple[int, object | None]:
            barrier.wait(timeout=5)
            return application.memory_mutate(
                "DELETE",
                f"/v1/memory/records/{record_id}",
                None,
                str(uuid.uuid4()),
                revision,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(delete, self.application, first_id)
            second_future = executor.submit(
                delete,
                second_application,
                second_id,
            )
            results = [
                first_future.result(timeout=5),
                second_future.result(timeout=5),
            ]
        self.assertEqual(sorted(result[0] for result in results), [200, 409])
        conflict = next(result for result in results if result[0] == 409)
        self.assertEqual(
            conflict[1],
            {
                "error": {
                    "code": "memory_revision_conflict",
                    "retryable": False,
                }
            },
        )
        with self.store._lock:
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM memory_record_tombstones "
                    "WHERE record_id IN (?,?)",
                    (first_id, second_id),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                self.store._connection.execute(
                    "SELECT COUNT(*) FROM memory_records WHERE record_id IN (?,?)",
                    (first_id, second_id),
                ).fetchone()[0],
                1,
            )

    def test_unavailable_is_stable_and_never_falls_through(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "unprepared.sqlite3"
        provider = TransactionSpyProvider()
        runtime = InjectedMemoryCryptoRuntime(
            provider.install_id,
            MemoryEnvelopeCipherV1(provider),
        )
        store = OrchestratorStore(path, memory_crypto=runtime)
        provider.stores.append(store)
        self.addCleanup(store.close)
        application = self._application(store)
        with self.assertRaisesRegex(
            OrchestratorError,
            "memory_storage_unavailable",
        ) as caught:
            application.get("/v1/memory/records")
        self.assertEqual(caught.exception.status, 503)
        self.assertTrue(caught.exception.retryable)

        key = str(uuid.uuid4())
        path_value = f"/v1/memory/records/{uuid.uuid4()}"
        first = application.memory_mutate(
            "DELETE",
            path_value,
            None,
            key,
            0,
        )
        self.assertEqual(
            first,
            (
                503,
                {
                    "error": {
                        "code": "memory_storage_unavailable",
                        "retryable": True,
                    }
                },
            ),
        )
        self.assertEqual(
            application.memory_mutate(
                "DELETE",
                path_value,
                None,
                key,
                0,
            ),
            first,
        )
        self.assertEqual(provider.calls, [])

    def test_handler_headers_zero_body_auth_and_no_store(self) -> None:
        key = str(uuid.uuid4())
        valid_headers = [
            ("Host", "127.0.0.1:8767"),
            ("Authorization", "Bearer control-private"),
            ("Idempotency-Key", key),
            ("If-Match-Memory-Revision", "7"),
            ("Content-Length", "0"),
        ]
        valid = self._handler(valid_headers)
        valid._admit(authenticate=True)
        self.assertEqual(valid._memory_mutation_headers(), (key, 7))
        self.assertIsNone(valid._memory_delete_body())

        max_revision = self._handler(
            [
                *valid_headers[:3],
                ("If-Match-Memory-Revision", str((1 << 63) - 1)),
                ("Content-Length", "0"),
            ]
        )
        self.assertEqual(
            max_revision._memory_mutation_headers(),
            (key, (1 << 63) - 1),
        )
        oversized_revision = self._handler(
            [
                *valid_headers[:3],
                ("If-Match-Memory-Revision", str(1 << 63)),
                ("Content-Length", "0"),
            ]
        )
        with self.assertRaisesRegex(OrchestratorError, "invalid_request"):
            oversized_revision._memory_mutation_headers()
        oversized_key = str(uuid.uuid4())
        with self.assertRaisesRegex(OrchestratorError, "invalid_request"):
            self.application.memory_mutate(
                "DELETE",
                f"/v1/memory/records/{uuid.uuid4()}",
                None,
                oversized_key,
                1 << 63,
            )
        with self.store._lock:
            self.assertIsNone(
                self.store._connection.execute(
                    "SELECT 1 FROM idempotency_commands WHERE idempotency_key=?",
                    (oversized_key,),
                ).fetchone()
            )

        forbidden_epoch = self._handler(valid_headers + [("If-Match-Epoch", "7")])
        with self.assertRaisesRegex(OrchestratorError, "invalid_request"):
            forbidden_epoch._memory_mutation_headers()
        missing = self._handler(valid_headers[:2])
        with self.assertRaisesRegex(
            OrchestratorError,
            "precondition_required",
        ):
            missing._memory_mutation_headers()
        unauthenticated = self._handler([("Host", "127.0.0.1:8767")])
        with self.assertRaisesRegex(OrchestratorError, "unauthenticated"):
            unauthenticated._admit(authenticate=True)

        nonempty = self._handler(
            [
                *valid_headers[:-1],
                ("Content-Length", "2"),
            ]
        )
        nonempty.rfile = io.BytesIO(b"{}")
        descriptor = nonempty._memory_delete_body()
        self.assertEqual(descriptor["unexpected_body_bytes"], 2)
        self.assertNotIn("{}", str(descriptor))

        sent_headers: list[tuple[str, str]] = []
        valid.send_response = lambda _status: None
        valid.send_header = lambda name, value: sent_headers.append((name, value))
        valid.end_headers = lambda: None
        valid.wfile = io.BytesIO()
        valid._json(200, {"ok": True})
        self.assertIn(("Cache-Control", "no-store"), sent_headers)
        sent_headers.clear()
        valid._error(OrchestratorError("not_found", status=404))
        self.assertIn(("Cache-Control", "no-store"), sent_headers)

        record_id = self._remember(self.store, "handler delete target")
        revision = self.store.control_memory_record_page(
            limit=1,
            now=NOW,
        )["memory_revision"]
        request_key = str(uuid.uuid4())
        routed = self._handler(
            [
                ("Host", "127.0.0.1:8767"),
                ("Authorization", "Bearer control-private"),
                ("Idempotency-Key", request_key),
                ("If-Match-Memory-Revision", str(revision)),
                ("Content-Length", "0"),
            ]
        )
        routed.path = f"/v1/memory/records/{record_id}"
        routed.rfile = io.BytesIO()
        routed.wfile = io.BytesIO()
        routed_statuses: list[int] = []
        routed_headers: list[tuple[str, str]] = []
        routed.send_response = routed_statuses.append
        routed.send_header = lambda name, value: routed_headers.append((name, value))
        routed.end_headers = lambda: None
        self.application._business_admission_provider = lambda: True
        routed.do_DELETE()
        self.assertEqual(routed_statuses, [200])
        self.assertIn(("Cache-Control", "no-store"), routed_headers)
        routed_result = json.loads(routed.wfile.getvalue())
        self.assertFalse(routed_result["chat_history_changed"])
        self.assertFalse(routed_result["physical_erasure_claimed"])
        self.assertIsNone(self.store.memory_record(record_id))


if __name__ == "__main__":
    unittest.main()
