from __future__ import annotations

import asyncio
import json
import unittest
import uuid

import _bootstrap  # noqa: F401
from helpers import StoreCase
from test_research_contract import corpus_fixture
from test_research_integration import admit, research_config

from jarvis_orchestrator.domain import TaskState, utc_now
from jarvis_orchestrator.pydantic_adapter import build_local_pydantic_port
from jarvis_orchestrator.research_adapter import FakeResearchAdapter
from jarvis_orchestrator.service import OrchestratorService

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


class ResearchModelTests(StoreCase, unittest.TestCase):
    def test_successful_service_turn_dispatches_exactly_two_real_model_requests(self) -> None:
        if httpx is None:
            self.skipTest("pinned Pydantic AI environment required")
        plan, corpus, selection, _ = corpus_fixture()
        captured = []

        async def handler(request):
            captured.append(json.loads(request.content))
            if len(captured) == 1:
                payload = {
                    "route": "research_local",
                    "canonical_query": plan.canonical_query,
                    "facets": [{"facet_id": item.facet_id, "label": item.label} for item in plan.facets],
                    "subqueries": list(plan.subqueries),
                    "freshness": plan.freshness,
                }
            else:
                payload = {
                    "selections": [{
                        "facet_id": selection.facet_id,
                        "passage_id": selection.passage_id,
                        "start_char": selection.start_char,
                        "end_char": selection.end_char,
                        "relation": selection.relation.value,
                    }]
                }
            content = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            return httpx.Response(200, json={
                "id": f"chatcmpl-{len(captured)}", "object": "chat.completion", "created": 1, "model": "jarvis-local",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
            })

        store = self.make_store()
        port = build_local_pydantic_port(
            store,
            base_url="http://127.0.0.1:9/v1",
            bearer_token="private",
            timeout_seconds=2,
            json_schema=True,
            http_transport=httpx.MockTransport(handler),
        )
        service = OrchestratorService(
            research_config(store.path),
            store,
            port,
            research_planner=port,
            research=FakeResearchAdapter({plan.plan_hash: corpus}),
        )
        service.attach_client("model-test")
        task = admit(store)
        waiting = asyncio.run(service.execute_task(task.task_id))
        approval = store.list_pending_approvals()[0]
        service.decide_approval(approval["approval_id"], waiting.epoch, "approve")
        completed = asyncio.run(service.execute_task(task.task_id))
        self.assertEqual(completed.state, TaskState.COMPLETED)
        self.assertEqual(len(captured), 2)
        self.assertEqual([item["max_completion_tokens"] for item in captured], [384, 768])
        self.assertEqual(store.remaining_model_budget(task.task_id, task.epoch)["requests"], 2)

    def test_span_selector_is_one_retry_free_768_token_request_without_prose_fields(self) -> None:
        if httpx is None:
            self.skipTest("pinned Pydantic AI environment required")
        captured = []
        plan, corpus, selection, _ = corpus_fixture()

        async def handler(request):
            captured.append(json.loads(request.content))
            content = json.dumps(
                {
                    "selections": [
                        {
                            "facet_id": selection.facet_id,
                            "passage_id": selection.passage_id,
                            "start_char": selection.start_char,
                            "end_char": selection.end_char,
                            "relation": selection.relation.value,
                        }
                    ]
                },
                separators=(",", ":"),
            )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-spans",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "jarvis-local",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "logprobs": None, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
                },
            )

        store = self.make_store()
        task = self.executing(
            store,
            self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4()), now=utc_now()),
            now=utc_now(),
        )
        port = build_local_pydantic_port(
            store,
            base_url="http://127.0.0.1:9/v1",
            bearer_token="private",
            timeout_seconds=2,
            json_schema=True,
            http_transport=httpx.MockTransport(handler),
        )
        selected = asyncio.run(
            port.select_research_spans(
                task_id=task.task_id,
                epoch=task.epoch,
                plan=plan,
                passages=corpus.passages,
                documents=corpus.documents,
                output_allowance=768,
                timeout_seconds=2,
            )
        )
        self.assertEqual(selected, (selection,))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["max_completion_tokens"], 768)
        schema_text = json.dumps(captured[0]["response_format"], sort_keys=True)
        for forbidden in ("claim_text", "excerpt", "url", "statement"):
            self.assertNotIn(forbidden, schema_text)

    def test_route_and_search_plan_are_one_retry_free_384_token_request(self) -> None:
        if httpx is None:
            self.skipTest("pinned Pydantic AI environment required")
        captured = []

        async def handler(request):
            captured.append(json.loads(request.content))
            content = json.dumps(
                {
                    "route": "research_local",
                    "canonical_query": "official current Example Product price",
                    "facets": [{"facet_id": "price", "label": "Current official price"}],
                    "subqueries": ["site:example.com Example Product price"],
                    "freshness": "current",
                },
                separators=(",", ":"),
            )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-research",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "jarvis-local",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
                },
            )

        store = self.make_store()
        task = self.admitted(store, event_id=str(uuid.uuid4()), turn_id=str(uuid.uuid4()), now=utc_now())
        task = store.transition(task.task_id, task.epoch, TaskState.ADMITTED, TaskState.ROUTING)
        port = build_local_pydantic_port(
            store,
            base_url="http://127.0.0.1:9/v1",
            bearer_token="private",
            timeout_seconds=2,
            json_schema=True,
            http_transport=httpx.MockTransport(handler),
        )
        plan = asyncio.run(
            port.plan_research(
                task_id=task.task_id,
                epoch=task.epoch,
                query="Find the current price",
                language="en",
                timezone="Europe/Athens",
                browser_fallback_allowed=False,
                output_allowance=384,
            )
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["max_completion_tokens"], 384)
        self.assertEqual(captured[0]["stream"], False)
        self.assertNotIn("seed", captured[0])
        self.assertEqual(plan.facets[0].facet_id, "price")
        budget = store.remaining_model_budget(task.task_id, task.epoch)
        self.assertEqual(budget["requests"], 3)
        self.assertEqual(budget["output_tokens"], 2048 - 30)


if __name__ == "__main__":
    unittest.main()
