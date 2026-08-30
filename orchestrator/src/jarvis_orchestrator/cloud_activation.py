"""Content-free cloud activation readiness gate."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .store import OrchestratorStore


async def cloud_qualification_status(
    config: OrchestratorConfig,
) -> dict[str, Any]:
    """Run static release and live key/ZDR gates without a model dispatch."""

    if not config.cloud_enabled:
        return {
            "active_endpoint_count": 0,
            "cloud_ready": False,
            "code": "cloud_not_requested",
            "terra_route_enabled": False,
            "version": "cloud-activation-ready-gate/1",
        }
    from .cloud_provider import (
        OpenRouterCloudAdapter,
        load_endpoint_registry,
        load_openrouter_secret,
    )
    from .cloud_qualification import load_cloud_release_qualification

    try:
        assert config.cloud_secret_file is not None
        assert config.cloud_qualification_file is not None
        assert config.cloud_release_qualification_file is not None
        bearer_token = load_openrouter_secret(config.cloud_secret_file)
        registry = load_endpoint_registry(config.cloud_qualification_file)
        release = load_cloud_release_qualification(
            config.cloud_release_qualification_file,
            expected_sha256=config.cloud_release_qualification_sha256,
            registry=registry,
            bearer_token=bearer_token,
            terra_enabled=config.cloud_terra_enabled,
        )
        with tempfile.TemporaryDirectory(prefix="jarvis-cloud-gate-") as directory:
            store = OrchestratorStore(Path(directory) / "qualification.sqlite3")
            adapter: OpenRouterCloudAdapter | None = None
            try:
                for endpoint in release.active_endpoints:
                    store.register_endpoint_qualification(endpoint, enabled=True)
                adapter = OpenRouterCloudAdapter(
                    store,
                    bearer_token=bearer_token,
                    timeout_seconds=config.cloud_timeout_seconds,
                )
                results = await asyncio.gather(
                    *(
                        adapter.revalidate(endpoint)
                        for endpoint in release.active_endpoints
                    )
                )
            finally:
                if adapter is not None:
                    await adapter.aclose()
                store.close()
        if not results or any(not item.key_policy_verified for item in results):
            raise RuntimeError("cloud_live_gate_invalid")
        return {
            "active_endpoint_count": len(results),
            "cloud_ready": True,
            "code": "cloud_activation_ready",
            "deepseek_endpoint_ready": True,
            "dedicated_monthly_key_ready": True,
            "terra_route_enabled": config.cloud_terra_enabled,
            "version": "cloud-activation-ready-gate/1",
        }
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
            raise
        return {
            "active_endpoint_count": 0,
            "cloud_ready": False,
            "code": getattr(exc, "code", "cloud_activation_blocked"),
            "terra_route_enabled": False,
            "version": "cloud-activation-ready-gate/1",
        }


__all__ = ["cloud_qualification_status"]
