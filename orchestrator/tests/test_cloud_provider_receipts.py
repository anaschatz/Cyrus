from __future__ import annotations

import unittest
from unittest.mock import Mock

import httpx
import _bootstrap  # noqa: F401

from test_cloud_release_qualification import DEEPSEEK_MODEL, endpoint
from jarvis_orchestrator.cloud_contract import build_dispatch_body
from jarvis_orchestrator.cloud_provider import OpenRouterCloudAdapter
from jarvis_orchestrator.domain import OrchestratorError
from jarvis_orchestrator.jcs import domain_sha256


class CloudProviderReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def dispatch(self, *, receipt_changes=None, usage=None, metadata=None, remove=()):
        selected = endpoint(DEEPSEEK_MODEL)
        body = build_dispatch_body(selected, messages=({'role': 'user', 'content': 'public fixture'},), output_token_limit=256)
        receipt = dict(id='gen-fixture', model=selected.model_canonical_id,
            provider_name=selected.runtime_provider_name, native_tokens_prompt=100,
            native_tokens_completion=10, native_tokens_reasoning=0, total_cost='0.0001')
        receipt.update(receipt_changes or {})
        for key in remove:
            receipt.pop(key)
        post_count = 0

        async def post(request):
            nonlocal post_count
            post_count += 1
            return httpx.Response(200, json=dict(id='gen-fixture', choices=[dict(finish_reason='stop',
                message=dict(content='synthetic answer'))],
                usage=usage if usage is not None else dict(prompt_tokens=100, completion_tokens=10),
                openrouter_metadata=metadata if metadata is not None else dict(attempt=1, strategy='direct')))

        async def control(request):
            self.assertEqual(request.url.params['id'], 'gen-fixture')
            return httpx.Response(200, json={'data': receipt})

        store = Mock()
        store.reconcile_cloud_dispatch.return_value = 100
        self.store = store
        adapter = OpenRouterCloudAdapter(store, bearer_token='synthetic-key',
            transport=httpx.MockTransport(post), control_transport=httpx.MockTransport(control))
        try:
            return await adapter.dispatch(task_id='synthetic-task', epoch=1, endpoint=selected,
                permit_id='synthetic-permit', reservation_id='synthetic-reservation', dispatch_body=body,
                dispatch_body_sha256=domain_sha256('cloud-dispatch-body/1', body),
                input_token_limit=1024, output_token_limit=256, max_reserved_microusd=1000)
        finally:
            await adapter.aclose()
            self.assertEqual(post_count, 1)
            store.consume_cloud_dispatch_permit.assert_called_once()

    async def test_valid_usage_and_receipt_are_published_once(self):
        result = await self.dispatch()
        self.assertEqual((result.input_tokens, result.output_tokens, result.reasoning_tokens), (100, 10, 0))
        self.assertTrue(self.store.store_cloud_provider_receipt.call_args.kwargs['verified'])
        self.assertEqual(result.actual_microusd, 100)

    async def test_native_token_counts_must_be_nonnegative_integers(self):
        for field in ('native_tokens_prompt', 'native_tokens_completion', 'native_tokens_reasoning'):
            for value in (-1, True, 0.5, None, '12'):
                with self.subTest(field=field, value=value), self.assertRaises(OrchestratorError):
                    await self.dispatch(receipt_changes={field: value})
                self.store.reconcile_cloud_dispatch.assert_called_with('synthetic-reservation', actual_microusd=None, ambiguous=True)
        for missing in ('native_tokens_prompt', 'native_tokens_completion'):
            with self.subTest(missing=missing), self.assertRaises(OrchestratorError):
                await self.dispatch(remove=(missing,))

    async def test_usage_requires_prompt_and_completion_counts(self):
        for usage in ({}, {'prompt_tokens': 100}, {'prompt_tokens': True, 'completion_tokens': 10},
                      {'prompt_tokens': 100, 'completion_tokens': -1}):
            with self.subTest(usage=usage), self.assertRaises(OrchestratorError):
                await self.dispatch(usage=usage)

    async def test_receipt_for_another_generation_is_never_verified(self):
        with self.assertRaises(OrchestratorError):
            await self.dispatch(receipt_changes={'id': 'gen-other'})
        self.assertFalse(self.store.store_cloud_provider_receipt.call_args.kwargs['verified'])
        self.store.quarantine_endpoint.assert_called_once()
        self.store.reconcile_cloud_dispatch.assert_any_call('synthetic-reservation', actual_microusd=1000)

    async def test_router_metadata_rejects_invalid_types(self):
        for metadata in ([], {'attempt': True, 'strategy': 'direct'}, {'attempt': 1, 'strategy': []}):
            with self.subTest(metadata=metadata), self.assertRaises(OrchestratorError):
                await self.dispatch(metadata=metadata)


if __name__ == '__main__':
    unittest.main()
