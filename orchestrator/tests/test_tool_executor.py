from __future__ import annotations

import asyncio
import unittest
import uuid

import _bootstrap  # noqa: F401

from jarvis_orchestrator.action_host_v2_client import NativeExecutionResultV2
from jarvis_orchestrator.tool_contracts import (
    ToolOutcomeV1,
    ToolRegistryV1,
    ToolStatus,
    VerificationStatus,
)
from jarvis_orchestrator.tool_executor import ActionHostV2ToolExecutor


class FakeActionHostV2Client:
    def __init__(self, result: NativeExecutionResultV2) -> None:
        self.result = result
        self.execute_payloads: list[dict[str, object]] = []
        self.queries: list[tuple[str, str]] = []
        self.acks: list[NativeExecutionResultV2] = []

    async def execute(self, payload):
        self.execute_payloads.append(dict(payload))
        return self.result

    async def query_result(self, execution_id: str, idempotency_key: str):
        self.queries.append((execution_id, idempotency_key))
        return self.result

    async def acknowledge(self, result: NativeExecutionResultV2) -> None:
        self.acks.append(result)


class ActionHostV2ToolExecutorTests(unittest.TestCase):
    def test_exact_permit_binding_and_ack_after_result(self) -> None:
        execution_id = str(uuid.uuid4())
        idempotency_key = str(uuid.uuid4())
        permit_id = str(uuid.uuid4())
        tool_call_id = str(uuid.uuid4())
        registry = ToolRegistryV1.build(
            root_ids=("documents",), app_ids=("textedit",)
        )
        definition = registry.resolve("note.capture@1")
        outcome = ToolOutcomeV1(
            ToolStatus.VERIFIED, VerificationStatus.MATCHED, "ok"
        )
        result = NativeExecutionResultV2(
            result_event_id=str(uuid.uuid4()),
            request_id=str(uuid.uuid4()),
            execution_id=execution_id,
            tool_call_id=tool_call_id,
            permit_id=permit_id,
            idempotency_key=idempotency_key,
            tool_id=definition.tool_id,
            tool_version=1,
            result_schema_sha256=definition.output_schema_sha256,
            registry_sha256=registry.digest,
            result_sha256="d" * 64,
            outcome=outcome,
            result_payload={"ok": True},
            terminal_receipt={"stage": "VERIFICATION_COMPLETE"},
        )
        client = FakeActionHostV2Client(result)
        executor = ActionHostV2ToolExecutor(client)

        returned = asyncio.run(
            executor.execute_typed(
                execution_id=execution_id,
                permit_id=permit_id,
                tool_call_id=tool_call_id,
                definition=definition,
                arguments={"title": "Idea", "body": "Coffee"},
                arguments_sha256="a" * 64,
                idempotency_key=idempotency_key,
                registry_sha256=registry.digest,
                policy_sha256="c" * 64,
                scheduler_contract_sha256="e" * 64,
                user_intent_binding_sha256="f" * 64,
                deadline_utc="2030-01-10T12:00:01.000Z",
            )
        )
        self.assertEqual(returned, (outcome, {"ok": True}))
        self.assertEqual(client.acks, [])
        payload = client.execute_payloads[0]
        self.assertEqual(payload["execution_id"], execution_id)
        self.assertEqual(payload["idempotency_key"], idempotency_key)
        self.assertEqual(payload["tool_id"], "note.capture@1")
        self.assertEqual(payload["tool_schema_sha256"], definition.schema_sha256)
        self.assertEqual(payload["payload"], {"title": "Idea", "body": "Coffee"})

        asyncio.run(executor.acknowledge(execution_id, idempotency_key))
        self.assertEqual(client.queries, [])
        self.assertEqual(client.acks, [result])

    def test_restart_ack_queries_without_redispatch(self) -> None:
        result = NativeExecutionResultV2(
            result_event_id=str(uuid.uuid4()),
            request_id=str(uuid.uuid4()),
            execution_id=str(uuid.uuid4()),
            tool_call_id=str(uuid.uuid4()),
            permit_id=str(uuid.uuid4()),
            idempotency_key=str(uuid.uuid4()),
            tool_id="app.open@1",
            tool_version=1,
            result_schema_sha256="a" * 64,
            registry_sha256="b" * 64,
            result_sha256="d" * 64,
            outcome=ToolOutcomeV1(
                ToolStatus.ACCEPTED, VerificationStatus.UNAVAILABLE, "ok"
            ),
            result_payload={"ok": True},
            terminal_receipt={"stage": "VERIFICATION_COMPLETE"},
        )
        client = FakeActionHostV2Client(result)
        executor = ActionHostV2ToolExecutor(client)

        asyncio.run(executor.acknowledge(result.execution_id, result.idempotency_key))

        self.assertEqual(client.execute_payloads, [])
        self.assertEqual(
            client.queries, [(result.execution_id, result.idempotency_key)]
        )
        self.assertEqual(client.acks, [result])


if __name__ == "__main__":
    unittest.main()
