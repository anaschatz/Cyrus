from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import _bootstrap  # noqa: F401
from jarvis_orchestrator import cli
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.pydantic_adapter import build_local_pydantic_port
from jarvis_orchestrator.runtime_supervisor import RuntimeSupervisor, RuntimeFailure
from test_model_stream_ownership import Store


class ModelClientLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_port_closes_owned_http_client_without_model_dispatch(self):
        clients = []

        class Client(httpx.AsyncClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                clients.append(self)

        with patch.object(httpx, "AsyncClient", Client):
            port = build_local_pydantic_port(
                Store(),
                base_url="http://127.0.0.1:9/v1",
                bearer_token="synthetic",
                timeout_seconds=1,
                http_transport=httpx.MockTransport(
                    lambda request: self.fail("unexpected HTTP request")
                ),
            )
        self.assertEqual(len(clients), 1)
        self.addAsyncCleanup(clients[0].aclose)
        await port.aclose()
        await port.aclose()
        self.assertTrue(clients[0].is_closed)

    async def run_components(
        self, *, setup_error=None, serve_error=None, close_error=None
    ):
        config = OrchestratorConfig.from_test_mapping(
            {
                "database_path": "/tmp/unused-model-client-review.sqlite3",
                "control_bearer_token": "control-test",
                "inference_bearer_token": "inference-test",
            }
        )
        store = Mock()
        if setup_error:
            store.disable_all_endpoint_qualifications.side_effect = setup_error
        model = SimpleNamespace(aclose=AsyncMock(side_effect=close_error))
        supervisor = RuntimeSupervisor(shutdown_timeout_seconds=1)
        deadline = Mock()
        deadline.remaining.return_value = 1
        with (
            patch.object(
                cli, "_apply_native_action_qualification", return_value=(config, None)
            ),
            patch.object(
                cli.InferenceControlClient,
                "capabilities",
                return_value=SimpleNamespace(
                    text=True, stream=True, tools=False, json_schema=False
                ),
            ),
            patch(
                "jarvis_orchestrator.pydantic_adapter.build_local_pydantic_port",
                return_value=model,
            ),
            patch.object(
                cli, "_run_servers", new=AsyncMock(side_effect=serve_error)
            ) as servers,
        ):
            try:
                await cli._run_components(
                    config,
                    store,
                    voice_auth_key=None,
                    runtime_supervisor=supervisor,
                    shutdown_deadline=deadline,
                )
            finally:
                model.aclose.assert_awaited_once()
                deadline.begin.assert_called()
        return servers

    async def test_normal_shutdown_closes_model(self):
        servers = await self.run_components()
        servers.assert_awaited_once()

    async def test_setup_failure_after_model_creation_closes_model(self):
        with self.assertRaisesRegex(ValueError, "setup failed"):
            await self.run_components(setup_error=ValueError("setup failed"))

    async def test_cancelled_runtime_closes_model(self):
        with self.assertRaises(asyncio.CancelledError):
            await self.run_components(serve_error=asyncio.CancelledError())

    async def test_model_close_failure_uses_runtime_failure_contract(self):
        with self.assertRaisesRegex(RuntimeFailure, "runtime_model_shutdown_failed"):
            await self.run_components(close_error=OSError("synthetic close failure"))

    async def test_other_adapter_cleanup_failure_still_closes_model(self):
        config = replace(
            OrchestratorConfig.from_test_mapping(
                {
                    "database_path": "/tmp/unused-model-client-review.sqlite3",
                    "control_bearer_token": "control-test",
                    "inference_bearer_token": "inference-test",
                }
            ),
            cloud_enabled=True,
            cloud_secret_file=Path("/synthetic/secret"),
            cloud_qualification_file=Path("/synthetic/qualification"),
            cloud_release_qualification_file=Path("/synthetic/release"),
        )
        model = SimpleNamespace(aclose=AsyncMock())
        cloud = SimpleNamespace(
            aclose=AsyncMock(side_effect=OSError("synthetic failure"))
        )
        deadline = Mock()
        deadline.remaining.return_value = 1
        with (
            patch.object(
                cli, "_apply_native_action_qualification", return_value=(config, None)
            ),
            patch.object(
                cli.InferenceControlClient,
                "capabilities",
                return_value=SimpleNamespace(tools=False, json_schema=True),
            ),
            patch(
                "jarvis_orchestrator.pydantic_adapter.build_local_pydantic_port",
                return_value=model,
            ),
            patch.object(cli, "_run_servers", new=AsyncMock()),
            patch(
                "jarvis_orchestrator.cloud_provider.load_openrouter_secret",
                return_value="synthetic",
            ),
            patch(
                "jarvis_orchestrator.cloud_provider.load_endpoint_registry",
                return_value=Mock(),
            ),
            patch(
                "jarvis_orchestrator.cloud_qualification.load_cloud_release_qualification",
                return_value=SimpleNamespace(active_endpoints=()),
            ),
            patch(
                "jarvis_orchestrator.cloud_provider.OpenRouterCloudAdapter",
                return_value=cloud,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeFailure, "runtime_cloud_shutdown_failed"
            ):
                await cli._run_components(
                    config,
                    Mock(),
                    voice_auth_key=None,
                    runtime_supervisor=RuntimeSupervisor(shutdown_timeout_seconds=1),
                    shutdown_deadline=deadline,
                )
        cloud.aclose.assert_awaited_once()
        model.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
