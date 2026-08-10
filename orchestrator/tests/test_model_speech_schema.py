from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import _bootstrap  # noqa: F401
from helpers import StoreCase

from jarvis_orchestrator.common_schema import CommonSchemaError, route_decision_schema, transform_common_schema
from jarvis_orchestrator.domain import ModelChunk, ModelRequest, ModelUsage, OrchestratorError, TaskState, utc_now
from jarvis_orchestrator.model import BudgetedModelPort, LocalInferenceModelPort, active_dispatch_reservation
from jarvis_orchestrator.pydantic_adapter import build_local_pydantic_port
from jarvis_orchestrator.speech import SpeechSegmenterV1, resolve_locale


SNAPSHOTS = Path(__file__).with_name("snapshots")


def snapshot(name: str) -> dict[str, object]:
    return json.loads((SNAPSHOTS / name).read_text(encoding="utf-8"))


class RecordingModel:
    def __init__(self) -> None:
        self.reservation_seen: str | None = None
        self.allowance = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.reservation_seen = active_dispatch_reservation()
        self.allowance = request.output_allowance
        yield ModelChunk("ok", ModelUsage(output_tokens=3))


class ModelBudgetTests(StoreCase, unittest.TestCase):
    def live_executing_task(self, store):
        selected = utc_now()
        admitted = self.admitted(
            store,
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            now=selected,
        )
        return self.executing(store, admitted, now=selected)

    def test_reservation_precedes_dispatch_and_reconciles_usage(self) -> None:
        store = self.make_store()
        task = self.live_executing_task(store)
        delegate = RecordingModel()
        wrapped = BudgetedModelPort(store, delegate)
        request = ModelRequest(task.task_id, task.epoch, "final", "system", (), 128)

        async def collect() -> list[ModelChunk]:
            return [chunk async for chunk in wrapped.stream(request)]

        self.assertEqual(asyncio.run(collect())[0].text, "ok")
        self.assertIsNotNone(delegate.reservation_seen)
        self.assertEqual(delegate.allowance, 128)
        self.assertEqual(store.remaining_model_budget(task.task_id, task.epoch), {"requests": 3, "output_tokens": 2045})

    def test_raw_http_adapter_fails_without_guard(self) -> None:
        adapter = LocalInferenceModelPort(base_url="http://127.0.0.1:9/v1", bearer_token="private")
        request = ModelRequest(str(uuid.uuid4()), 1, "final", "system", (), 16, timeout_seconds=0.01)
        with self.assertRaisesRegex(OrchestratorError, "model_dispatch_without_reservation"):
            adapter._request(request)

    def test_exact_payload_uses_reserved_max_completion_tokens(self) -> None:
        adapter = LocalInferenceModelPort(base_url="http://127.0.0.1:9/v1", bearer_token="private")
        request = ModelRequest("task", 1, "final", "system", ({"role": "user", "content": "hi"},), 257)
        payload = adapter._payload(request)
        self.assertEqual(payload["max_completion_tokens"], 257)
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["messages"][0]["role"], "system")

    def test_pinned_pydantic_serialized_stream_payload_snapshot(self) -> None:
        try:
            import httpx
            import pydantic_ai
        except ImportError:
            self.skipTest("pinned Pydantic AI integration environment required")
        self.assertEqual(pydantic_ai.__version__, "2.27.0")
        captured: list[dict[str, object]] = []

        async def handler(request):
            captured.append(json.loads(request.content))
            chunks = [
                {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "jarvis-local", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}], "usage": None},
                {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "jarvis-local", "choices": [{"index": 0, "delta": {"content": "ok"}, "logprobs": None, "finish_reason": None}], "usage": None},
                {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "jarvis-local", "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}], "usage": None},
                {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "jarvis-local", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
            ]
            body = "".join(f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

        store = self.make_store()
        task = self.live_executing_task(store)
        port = build_local_pydantic_port(
            store,
            base_url="http://127.0.0.1:9/v1",
            bearer_token="private",
            timeout_seconds=2,
            http_transport=httpx.MockTransport(handler),
        )
        request = ModelRequest(
            task.task_id,
            task.epoch,
            "final",
            "system-only",
            ({"role": "user", "content": "hi"},),
            128,
            timeout_seconds=2,
        )

        async def collect():
            return [chunk async for chunk in port.stream(request)]

        self.assertEqual(asyncio.run(collect())[-1].usage.output_tokens, 2)
        self.assertEqual(captured, [snapshot("pydantic_text_stream.json")])

    def test_pinned_pydantic_route_payload_snapshots_and_appliance_oracle(self) -> None:
        try:
            import httpx
            import pydantic_ai
        except ImportError:
            self.skipTest("pinned Pydantic AI integration environment required")
        from jarvis_inference.contracts import Capabilities, normalize_request

        self.assertEqual(pydantic_ai.__version__, "2.27.0")
        decision = {"route": "local", "memory_query": "", "research_query": "", "tool_name": "none", "response_language": "en"}

        async def exercise(native: bool) -> dict[str, object]:
            captured: list[dict[str, object]] = []

            async def handler(request):
                captured.append(json.loads(request.content))
                arguments = json.dumps(decision, separators=(",", ":"))
                if native:
                    message = {"role": "assistant", "content": arguments}
                    finish = "stop"
                else:
                    message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_x", "type": "function", "function": {"name": "route_decision", "arguments": arguments}}]}
                    finish = "tool_calls"
                return httpx.Response(200, json={
                    "id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": "jarvis-local",
                    "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                })

            store = self.make_store()
            task = self.live_executing_task(store)
            port = build_local_pydantic_port(
                store,
                base_url="http://127.0.0.1:9/v1",
                bearer_token="private",
                timeout_seconds=2,
                tools=not native,
                json_schema=native,
                http_transport=httpx.MockTransport(handler),
            )
            result = await port.plan(ModelRequest(
                task.task_id, task.epoch, "route", "plan", ({"role": "user", "content": "hi"},), 256, timeout_seconds=2
            ))
            self.assertEqual(result.route, "local")
            return captured[0]

        native = asyncio.run(exercise(True))
        tool = asyncio.run(exercise(False))
        self.assertEqual(native, snapshot("pydantic_route_native.json"))
        self.assertEqual(tool, snapshot("pydantic_route_tool.json"))
        normalize_request(native, alias="jarvis-local", capabilities=Capabilities(json_schema=True), context_window=8192)
        normalize_request(tool, alias="jarvis-local", capabilities=Capabilities(tools=True), context_window=8192)

    def test_validation_correction_is_a_second_reserved_run(self) -> None:
        try:
            import httpx
        except ImportError:
            self.skipTest("pinned Pydantic AI integration environment required")
        calls: list[dict[str, object]] = []
        valid = {"route": "local", "memory_query": "", "research_query": "", "tool_name": "none", "response_language": "en"}

        async def handler(request):
            calls.append(json.loads(request.content))
            content = "{}" if len(calls) == 1 else json.dumps(valid, separators=(",", ":"))
            return httpx.Response(200, json={
                "id": f"chatcmpl-{len(calls)}", "object": "chat.completion", "created": 1, "model": "jarvis-local",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            })

        store = self.make_store()
        task = self.live_executing_task(store)
        port = build_local_pydantic_port(
            store,
            base_url="http://127.0.0.1:9/v1",
            bearer_token="private",
            timeout_seconds=2,
            json_schema=True,
            http_transport=httpx.MockTransport(handler),
        )
        request = ModelRequest(
            task.task_id, task.epoch, "route", "plan", ({"role": "user", "content": "hi"},), 256, timeout_seconds=2
        )
        result = asyncio.run(port.plan_with_correction(request))
        self.assertEqual(result.route, "local")
        self.assertEqual(len(calls), 2)
        self.assertEqual([call["max_completion_tokens"] for call in calls], [256, 256])
        with store._lock:
            row = store._connection.execute(
                "SELECT model_requests_used,corrections_used FROM tasks WHERE task_id=?", (task.task_id,)
            ).fetchone()
        self.assertEqual(tuple(row), (2, 1))


class SpeechAndSchemaTests(unittest.TestCase):
    def test_speech_is_chunk_independent_nfc_contiguous_and_bounded(self) -> None:
        split = SpeechSegmenterV1(maximum_characters=12)
        result = split.feed("Το α\u0301λφα 1.") + split.feed("25 είναι εδώ! Next") + split.finish()
        self.assertEqual(result[0], "Το άλφα 1.25")
        self.assertTrue(all(result))
        self.assertTrue(all(len(item) <= 12 for item in result))
        self.assertEqual(resolve_locale("el"), "el-GR")
        self.assertEqual(resolve_locale("en-GB"), "en-US")
        self.assertEqual(resolve_locale("fr", "en-US"), "en-US")

    def test_common_schema_inlines_refs_requires_all_and_rejects_unions(self) -> None:
        schema = transform_common_schema(
            {
                "type": "object",
                "$defs": {"Name": {"type": "string", "default": "x", "format": "name"}},
                "properties": {"name": {"$ref": "#/$defs/Name"}},
            }
        )
        self.assertEqual(schema["required"], ["name"])
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("default", schema["properties"]["name"])
        with self.assertRaises(CommonSchemaError):
            transform_common_schema({"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}]}}})

    def test_route_decision_is_closed_and_enum_typed(self) -> None:
        schema = route_decision_schema(("lamp",))
        self.assertEqual(schema["properties"]["tool_name"]["enum"], ["none", "lamp"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))


if __name__ == "__main__":
    unittest.main()
