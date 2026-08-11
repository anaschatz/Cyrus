"""Qualified ToolExecutorPort backed by the authenticated Action Host v2."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from typing import Any

from .action_host_protocol import ActionHostProtocolError
from .action_host_v2_client import ActionHostV2Client, NativeExecutionResultV2
from .tool_contracts import ToolDefinitionV1, ToolOutcomeV1


class ActionHostV2ToolExecutor:
    """Translate one persisted Orchestrator permit into one native dispatch."""

    def __init__(self, client: ActionHostV2Client) -> None:
        self.client = client
        self._pending: dict[str, NativeExecutionResultV2] = {}
        self._lock = asyncio.Lock()

    async def execute_typed(
        self,
        *,
        execution_id: str,
        permit_id: str,
        tool_call_id: str,
        definition: ToolDefinitionV1,
        arguments: Mapping[str, Any],
        arguments_sha256: str,
        idempotency_key: str,
        registry_sha256: str,
        policy_sha256: str,
        scheduler_contract_sha256: str,
        user_intent_binding_sha256: str,
        deadline_utc: str,
    ) -> tuple[ToolOutcomeV1, Mapping[str, Any]]:
        result = await self.client.execute(
            {
                "request_id": str(uuid.uuid4()),
                "execution_id": execution_id,
                "tool_call_id": tool_call_id,
                "permit_id": permit_id,
                "idempotency_key": idempotency_key,
                "tool_id": definition.tool_id,
                "tool_version": 1,
                "tool_schema_sha256": definition.schema_sha256,
                "registry_sha256": registry_sha256,
                "policy_sha256": policy_sha256,
                "scheduler_contract_sha256": scheduler_contract_sha256,
                "user_intent_binding_sha256": user_intent_binding_sha256,
                "arguments_sha256": arguments_sha256,
                "payload": dict(arguments),
                "deadline_utc": deadline_utc,
            }
        )
        if any(
            (
                result.execution_id != execution_id,
                result.tool_call_id != tool_call_id,
                result.permit_id != permit_id,
                result.idempotency_key != idempotency_key,
                result.tool_id != definition.tool_id,
                result.tool_version != 1,
                result.result_schema_sha256 != definition.output_schema_sha256,
                result.registry_sha256 != registry_sha256,
            )
        ):
            raise ActionHostProtocolError("host.result_conflict")
        async with self._lock:
            existing = self._pending.get(execution_id)
            if existing is not None and existing.result_sha256 != result.result_sha256:
                raise ActionHostProtocolError("host.result_conflict")
            self._pending[execution_id] = result
        return result.outcome, result.result_payload

    async def acknowledge(self, execution_id: str, idempotency_key: str) -> None:
        async with self._lock:
            result = self._pending.get(execution_id)
        if result is None:
            result = await self.client.query_result(execution_id, idempotency_key)
        if result.execution_id != execution_id or result.idempotency_key != idempotency_key:
            raise ActionHostProtocolError("host.result_conflict")
        await self.client.acknowledge(result)
        async with self._lock:
            if self._pending.get(execution_id) == result:
                self._pending.pop(execution_id, None)
