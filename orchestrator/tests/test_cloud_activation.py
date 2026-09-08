from __future__ import annotations

import asyncio
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.cloud_activation import cloud_qualification_status
from jarvis_orchestrator.domain import OrchestratorError


class CloudActivationTests(unittest.IsolatedAsyncioTestCase):
    async def check_gate(self, mode):
        config = SimpleNamespace(cloud_enabled=True, cloud_secret_file=Path('/unused/secret'),
            cloud_qualification_file=Path('/unused/registry'), cloud_release_qualification_file=Path('/unused/release'),
            cloud_release_qualification_sha256='a' * 64, cloud_terra_enabled=True, cloud_timeout_seconds=1)
        endpoints = (object(), object())
        store, adapter = Mock(), Mock()
        finished, closed_after_finish = [], []
        started = asyncio.Event()

        async def revalidate(endpoint):
            if mode == 'failure' and endpoint is endpoints[0]:
                raise OrchestratorError('endpoint_unhealthy', status=409)
            started.set()
            try:
                await asyncio.sleep(0.02 if mode != 'cancel' else 10)
                return SimpleNamespace(key_policy_verified=True)
            finally:
                finished.append(endpoint)

        async def close():
            closed_after_finish.append(len(finished))
            if mode == 'close_failure':
                raise RuntimeError('injected close failure')

        adapter.revalidate = revalidate
        adapter.aclose = AsyncMock(side_effect=close)
        with ExitStack() as patches:
            patches.enter_context(patch('jarvis_orchestrator.cloud_activation.OrchestratorStore', return_value=store))
            patches.enter_context(patch('jarvis_orchestrator.cloud_provider.OpenRouterCloudAdapter', return_value=adapter))
            patches.enter_context(patch('jarvis_orchestrator.cloud_provider.load_openrouter_secret', return_value='fake-key'))
            patches.enter_context(patch('jarvis_orchestrator.cloud_provider.load_endpoint_registry', return_value=endpoints))
            patches.enter_context(patch('jarvis_orchestrator.cloud_qualification.load_cloud_release_qualification',
                                        return_value=SimpleNamespace(active_endpoints=endpoints)))
            task = asyncio.create_task(cloud_qualification_status(config))
            if mode == 'cancel':
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                self.assertEqual(result['cloud_ready'], mode == 'success')
                if mode == 'failure':
                    self.assertEqual(result['code'], 'endpoint_unhealthy')
            # Drain pre-fix orphaned work so a failing regression leaves no tasks behind.
            await asyncio.sleep(0.03)
        store.close.assert_called_once()
        adapter.aclose.assert_awaited_once()
        self.assertEqual(closed_after_finish, [1 if mode == 'failure' else 2])

    async def test_failed_endpoint_finishes_siblings_before_resource_close(self):
        await self.check_gate('failure')

    async def test_close_failure_still_closes_store(self):
        await self.check_gate('close_failure')

    async def test_cancellation_drains_children_before_close(self):
        await self.check_gate('cancel')

    async def test_adapter_closes_control_client_after_dispatch_close_failure(self):
        from jarvis_orchestrator.cloud_provider import OpenRouterCloudAdapter
        adapter = object.__new__(OpenRouterCloudAdapter)
        adapter._dispatch_client = SimpleNamespace(aclose=AsyncMock(side_effect=RuntimeError("injected")))
        adapter._control_client = SimpleNamespace(aclose=AsyncMock())
        with self.assertRaisesRegex(RuntimeError, "injected"):
            await adapter.aclose()
        adapter._control_client.aclose.assert_awaited_once()

    async def test_successful_gate_keeps_both_qualified_endpoints(self):
        await self.check_gate('success')


if __name__ == '__main__':
    unittest.main()
