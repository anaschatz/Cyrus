"""Source-run server and authenticated control CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import urllib.error
import urllib.request
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from .anamnesis_adapter import AnamnesisMemoryAdapter
from .anamnesis_shadow import (
    AnamnesisShadowAdapter,
    LocalLifecycleCompletionClient,
    LocalStructuredLifecycleWriter,
)
from .config import OrchestratorConfig
from .control import ControlApplication, ControlServer
from .gateway import VoiceGateway
from .inference import InferenceControlClient
from .pydantic_adapter import build_local_pydantic_port
from .service import OrchestratorService
from .store import OrchestratorStore


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _request(config: OrchestratorConfig, method: str, path: str, *, body: Any = None, epoch: int | None = None, key: str | None = None) -> tuple[int, Any | None]:
    url = f"http://{config.control_host}:{config.control_port}{path}"
    encoded = None if body is None else json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    headers = {"Authorization": f"Bearer {config.control_bearer_token}"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    if epoch is not None:
        headers["If-Match-Epoch"] = str(epoch)
        headers["Idempotency-Key"] = key or str(uuid.uuid4())
    request = urllib.request.Request(url, data=encoded, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, json.loads(raw) if raw else None


async def _serve(config: OrchestratorConfig) -> None:
    store = OrchestratorStore(
        config.database_path,
        model_request_limit=config.model_request_limit,
        output_token_limit=config.output_token_limit,
        active_budget_seconds=config.active_budget_seconds,
        approval_ttl=timedelta(seconds=config.approval_ttl_seconds),
        disconnect_ttl=timedelta(seconds=config.reconnect_ttl_seconds),
        delivery_ttl=timedelta(seconds=config.delivery_ttl_seconds),
    )
    inference_control = InferenceControlClient(
        config.inference_base_url,
        config.inference_bearer_token,
        timeout_seconds=min(2.0, config.model_timeout_seconds),
    )
    capabilities = await asyncio.to_thread(inference_control.capabilities)
    model = build_local_pydantic_port(
        store,
        base_url=config.inference_base_url,
        bearer_token=config.inference_bearer_token,
        timeout_seconds=config.model_timeout_seconds,
        tools=capabilities.tools,
        json_schema=capabilities.json_schema,
    )
    active_memory = AnamnesisMemoryAdapter(store) if config.memory_enabled else None
    shadow_memory = None
    if config.memory_shadow_enabled:
        if not capabilities.json_schema:
            raise RuntimeError("memory shadow mode requires JSON Schema capability")
        shadow_client = LocalLifecycleCompletionClient(
            base_url=config.inference_base_url,
            bearer_token=config.inference_bearer_token,
            timeout_seconds=config.memory_shadow_timeout_seconds,
        )
        shadow_memory = AnamnesisShadowAdapter(
            store,
            LocalStructuredLifecycleWriter(
                shadow_client,
                model_alias=config.model_alias,
            ),
        )
    service = OrchestratorService(
        config,
        store,
        model,
        inference_control,
        memory=shadow_memory,
        active_memory=active_memory,
    )
    service.startup_recovery()
    control = ControlServer(ControlApplication(config, service))
    control_thread = threading.Thread(target=control.serve_forever, name="jarvis-control", daemon=True)
    control_thread.start()
    gateway = VoiceGateway(service)
    try:
        await gateway.start()
        await asyncio.Future()
    finally:
        await gateway.stop()
        control.shutdown()
        control_thread.join(timeout=2)
        store.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="jarvis-orchestrator")
    result.add_argument("--config", required=True)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("serve")
    commands.add_parser("ready")
    approvals = commands.add_parser("approvals")
    approval_commands = approvals.add_subparsers(dest="approval_command", required=True)
    approval_commands.add_parser("list")
    decide = approval_commands.add_parser("decide")
    decide.add_argument("approval_id")
    decide.add_argument("--epoch", type=int, required=True)
    decision = decide.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--deny", action="store_true")
    tasks = commands.add_parser("tasks")
    task_commands = tasks.add_subparsers(dest="task_command", required=True)
    cancel = task_commands.add_parser("cancel")
    cancel.add_argument("task_id")
    cancel.add_argument("--epoch", type=int, required=True)
    conversations = commands.add_parser("conversations")
    conversation_commands = conversations.add_subparsers(dest="conversation_command", required=True)
    conversation_commands.add_parser("active")
    create = conversation_commands.add_parser("create")
    create.add_argument("--epoch", type=int, required=True)
    delete = conversation_commands.add_parser("delete")
    delete.add_argument("conversation_id")
    delete.add_argument("--epoch", type=int, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parsed = parser().parse_args(argv)
    config = OrchestratorConfig.load(Path(parsed.config))
    if parsed.command == "serve":
        try:
            asyncio.run(_serve(config))
        except KeyboardInterrupt:
            return 0
        return 0
    if parsed.command == "ready":
        status, value = _request(config, "GET", "/readyz")
    elif parsed.command == "approvals" and parsed.approval_command == "list":
        status, value = _request(config, "GET", "/v1/approvals")
    elif parsed.command == "approvals":
        decision = "approve" if parsed.approve else "deny"
        status, value = _request(config, "POST", f"/v1/approvals/{parsed.approval_id}/decision", body={"decision": decision}, epoch=parsed.epoch)
    elif parsed.command == "tasks":
        status, value = _request(config, "POST", f"/v1/tasks/{parsed.task_id}/cancel", body={"reason_code": "user_cancelled"}, epoch=parsed.epoch)
    elif parsed.command == "conversations" and parsed.conversation_command == "active":
        status, value = _request(config, "GET", "/v1/conversations/active")
    elif parsed.command == "conversations" and parsed.conversation_command == "create":
        status, value = _request(config, "POST", "/v1/conversations", body={}, epoch=parsed.epoch)
    elif parsed.command == "conversations" and parsed.conversation_command == "delete":
        confirmation = input(f"Type the full conversation ID to delete {parsed.conversation_id}: ").strip()
        if confirmation != parsed.conversation_id:
            _print({"error": {"code": "confirmation_failed", "retryable": False}})
            return 2
        status, value = _request(config, "DELETE", f"/v1/conversations/{parsed.conversation_id}", epoch=parsed.epoch)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    _print({"status": status, "body": value})
    return 0 if 200 <= status < 300 else 1
