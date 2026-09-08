from __future__ import annotations

import asyncio
import hashlib
import socket
from unittest.mock import patch
import os
import sqlite3
import tempfile
import threading
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401
from test_action_host_protocol_v2 import large_calendar_payload

from jarvis_orchestrator.action_host_protocol import ActionHostProtocolError
from jarvis_orchestrator.private_state import PrivateStateError
from jarvis_orchestrator.action_host_v2_client import ActionHostV2Client
from jarvis_orchestrator.action_host_v2_protocol import ResultVerificationV2
from jarvis_orchestrator.action_host_v2_server import (
    ActionHostResultJournalV2,
    ActionHostV2Server,
    AdapterResultV2,
)
from jarvis_orchestrator.jcs import jcs_bytes
from jarvis_orchestrator.tool_contracts import ToolRegistryV1, ToolStatus


class Keys:
    def __init__(self) -> None:
        self.key = bytes(range(32))

    def active_version(self) -> int:
        return 1

    def load(self, version: int) -> bytes:
        if version != 1:
            raise KeyError(version)
        return self.key


class Adapter:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, *, tool_id, arguments, execution_id, deadline_utc):
        self.calls += 1
        self.last = (tool_id, dict(arguments), execution_id, deadline_utc)
        return AdapterResultV2(
            ToolStatus.VERIFIED,
            ResultVerificationV2.EXACT_READBACK,
            "ok",
            {
                "app_id": "finder",
                "bundle_id": "com.apple.finder",
                "request_observed_at_utc": "2026-08-11T18:00:00.000Z",
            },
            "f" * 64,
        )


class ActionHostV2ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        self.socket_path = self.root / "host.sock"
        self.token_path = self.root / "token"
        self.token_path.write_bytes(bytes(reversed(range(32))))
        os.chmod(self.token_path, 0o600)
        self.keys = Keys()
        self.registry = ToolRegistryV1.build(
            root_ids=("documents",),
            app_ids=("calendar", "finder", "mail", "preview", "reminders", "textedit"),
        )
        self.adapter = Adapter()
        self.journal = ActionHostResultJournalV2(self.root / "journal.sqlite3", self.keys)
        self.server = ActionHostV2Server(
            socket_path=self.socket_path,
            token_path=self.token_path,
            auth_keys=self.keys,
            registry=self.registry,
            journal=self.journal,
            adapter=self.adapter,
        )
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self.server.serve_forever, args=(self.stop,), daemon=True
        )
        self.thread.start()
        for _ in range(100):
            if self.socket_path.exists():
                break
            await __import__("asyncio").sleep(0.01)
        self.client = ActionHostV2Client(
            socket_path=self.socket_path,
            token_path=self.token_path,
            auth_keys=self.keys,
            timeout_seconds=2,
        )

    async def asyncTearDown(self) -> None:
        self.stop.set()
        try:
            wake = __import__("socket").socket(
                __import__("socket").AF_UNIX, __import__("socket").SOCK_STREAM
            )
            wake.connect(os.fspath(self.socket_path))
            wake.close()
        except OSError:
            pass
        self.thread.join(2)
        self.journal.close()
        self.temp.cleanup()

    def payload(self):
        arguments = {"app_id": "finder"}
        definition = self.registry.resolve("app.open@1")
        return {
            "request_id": str(uuid.uuid4()),
            "execution_id": str(uuid.uuid4()),
            "tool_call_id": str(uuid.uuid4()),
            "permit_id": str(uuid.uuid4()),
            "idempotency_key": str(uuid.uuid4()),
            "tool_id": "app.open@1",
            "tool_version": 1,
            "tool_schema_sha256": definition.schema_sha256,
            "registry_sha256": self.registry.digest,
            "policy_sha256": "c" * 64,
            "scheduler_contract_sha256": "d" * 64,
            "user_intent_binding_sha256": "e" * 64,
            "arguments_sha256": hashlib.sha256(jcs_bytes(arguments)).hexdigest(),
            "payload": arguments,
            "deadline_utc": "2030-08-11T18:01:00.000Z",
        }

    async def test_real_server_result_replay_ack_and_at_most_once_adapter(self) -> None:
        execute = self.payload()
        result = await self.client.execute(execute)
        self.assertEqual(result.result_payload["bundle_id"], "com.apple.finder")
        replay = await self.client.query_result(
            execute["execution_id"], execute["idempotency_key"]
        )
        self.assertEqual(replay.result_event_id, result.result_event_id)
        self.assertEqual(self.adapter.calls, 1)
        await self.client.acknowledge(result)
        with self.journal._lock:
            row = self.journal._connection.execute(
                "SELECT acknowledged_at,ciphertext_and_tag FROM executions WHERE execution_id=?",
                (execute["execution_id"],),
            ).fetchone()
        self.assertIsNotNone(row["acknowledged_at"])
        self.assertIsNone(row["ciphertext_and_tag"])

    def test_output_schema_digest_is_additive(self) -> None:
        definition = self.registry.resolve("app.open@1")
        self.assertEqual(len(definition.output_schema_sha256), 64)
        self.assertEqual(
            definition.schema_sha256,
            "1cd3619149bb5e174ca0aef3dbba436fb2789ff15504e15f90f04e19c83ce011",
        )



    async def test_identical_execute_replay_returns_durable_result_without_dispatch(self):
        execute = self.payload()
        first = await self.client.execute(execute)
        second = await self.client.execute(execute)
        self.assertEqual(second.result_event_id, first.result_event_id)
        self.assertEqual(self.adapter.calls, 1)

    def test_disconnect_after_native_return_does_not_lose_result(self):
        execute = self.payload()
        class Sender:
            def encrypt(self, value):
                return value
        class BrokenConnection:
            def sendall(self, value):
                if value["payload"].get("stage") == "OS_CALL_RETURNED":
                    raise BrokenPipeError("client disconnected after native return")
        with self.assertRaises(BrokenPipeError):
            self.server._execute(Sender(), BrokenConnection(), execute)
        result = self.journal.result(execute["execution_id"], execute["idempotency_key"])
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(self.journal.receipts(execute["execution_id"], execute["idempotency_key"])[-1]["stage"],
                         "VERIFICATION_COMPLETE")
        self.assertEqual(self.adapter.calls, 1)

    def test_terminal_result_and_receipt_commit_atomically(self):
        execute = self.payload()
        class Sender:
            def encrypt(self, value):
                return value
        class Connection:
            def sendall(self, value):
                pass
        with self.journal._transaction() as connection:
            connection.execute("""CREATE TRIGGER reject_terminal_receipt BEFORE INSERT ON receipts
                WHEN NEW.stage='VERIFICATION_COMPLETE'
                BEGIN SELECT RAISE(ABORT, 'injected receipt failure'); END""")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected receipt failure"):
            self.server._execute(Sender(), Connection(), execute)
        self.assertIsNone(self.journal.result(execute["execution_id"], execute["idempotency_key"]))
        self.assertEqual(self.adapter.calls, 1)

    def test_failed_transaction_entry_releases_journal_lock(self):
        with self.journal._lock:
            self.journal._connection.execute("BEGIN")
        try:
            with self.assertRaises(sqlite3.OperationalError):
                with self.journal._transaction():
                    pass
            acquired = threading.Event()
            def other_thread():
                if self.journal._lock.acquire(timeout=0.2):
                    acquired.set()
                    self.journal._lock.release()
            worker = threading.Thread(target=other_thread)
            worker.start()
            worker.join(1)
            if not acquired.is_set():
                self.journal._lock.release()  # Clean up the pre-fix leaked acquisition.
            self.assertTrue(acquired.is_set())
        finally:
            self.journal._connection.execute("ROLLBACK")

    def test_journal_rejects_symlink_without_initializing_target(self):
        target = self.root / "target.sqlite3"
        target.write_bytes(b"")
        os.chmod(target, 0o600)
        link = self.root / "alias.sqlite3"
        link.symlink_to(target)
        with self.assertRaises(PrivateStateError):
            journal = ActionHostResultJournalV2(link, self.keys)
            journal.close()
        self.assertEqual(target.read_bytes(), b"")


    async def test_stalled_handshake_expires_without_blocking_next_action(self):
        with patch("jarvis_orchestrator.action_host_v2_server.SESSION_TIMEOUT_SECONDS", 0.08, create=True):
            stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stalled.setblocking(False)
            try:
                await asyncio.get_running_loop().sock_connect(stalled, str(self.socket_path))
                await asyncio.sleep(0.15)
                result = await self.client.execute(self.payload())
                self.assertEqual(result.outcome.status, ToolStatus.VERIFIED)
            finally:
                stalled.close()

    async def test_invalid_tool_request_does_not_terminate_server(self):
        with self.assertRaises(ActionHostProtocolError):
            await self.client.execute(self.payload() | {"tool_id": "unknown@1"})
        self.assertTrue(self.thread.is_alive())
        result = await self.client.execute(self.payload())
        self.assertEqual(result.outcome.status, ToolStatus.VERIFIED)
        self.assertEqual(self.adapter.calls, 1)


    async def test_full_size_calendar_result_replays_and_acknowledges_over_socket(self):
        payload = large_calendar_payload()
        arguments = dict(start_local="2026-09-06T00:00:00.000", end_local="2026-09-07T00:00:00.000",
                         timezone="Europe/Athens", limit=50, cursor="")
        execute = self.payload() | dict(tool_id="calendar.read@1", payload=arguments,
            tool_schema_sha256=self.registry.resolve("calendar.read@1").schema_sha256,
            arguments_sha256=hashlib.sha256(jcs_bytes(arguments)).hexdigest())
        native = AdapterResultV2(ToolStatus.VERIFIED, ResultVerificationV2.EXACT_READBACK,
                                 "ok", payload, "f" * 64)
        with patch.object(self.adapter, "execute", return_value=native) as adapter:
            first = await self.client.execute(execute)
            replay = await self.client.query_result(execute["execution_id"], execute["idempotency_key"])
            self.assertEqual(first.result_payload, payload)
            self.assertEqual(replay.result_payload, payload)
            self.assertEqual(replay.result_event_id, first.result_event_id)
            await self.client.acknowledge(replay)
            self.assertIsNone(self.journal.result(execute["execution_id"], execute["idempotency_key"]))
            self.assertEqual(adapter.call_count, 1)

if __name__ == "__main__":
    unittest.main()

