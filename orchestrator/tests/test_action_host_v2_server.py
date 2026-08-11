from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

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


if __name__ == "__main__":
    unittest.main()

