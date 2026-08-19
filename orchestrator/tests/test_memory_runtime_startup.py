from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.cli import (
    _memory_crypto_identity_state,
    _serve_components,
)
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.memory_crypto import (
    InMemoryMemoryEpochKeyProvider,
    MemoryEnvelopeCipherV1,
)
from jarvis_orchestrator.memory_runtime import ProviderBoundMemoryCryptoRuntime
from jarvis_orchestrator.operations import OperationalError
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore


class _NoBootstrapProvider:
    def __init__(self, delegate: InMemoryMemoryEpochKeyProvider) -> None:
        self.delegate = delegate
        self.bootstrap_calls = 0

    @property
    def install_id(self) -> str:
        return self.delegate.install_id

    def bootstrap(self):
        self.bootstrap_calls += 1
        raise AssertionError("existing encrypted storage must not bootstrap")


def _config(root: Path, *, memory_enabled: bool) -> OrchestratorConfig:
    return OrchestratorConfig.from_test_mapping(
        {
            "database_path": str(root / "state.sqlite3"),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "memory_enabled": memory_enabled,
        }
    )


class MemoryRuntimeStartupTests(unittest.TestCase):
    def test_readiness_probe_never_creates_fresh_database_or_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root, memory_enabled=True)
            with (
                patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider"
                ) as provider_constructor,
                self.assertRaises(OperationalError) as raised,
            ):
                _memory_crypto_identity_state(config)

            self.assertEqual(
                str(raised.exception), "readiness_memory_crypto_unavailable"
            )
            provider_constructor.assert_not_called()
            self.assertFalse(config.database_path.exists())
            self.assertFalse((root / "operations").exists())

    def test_enabled_runtime_is_prepared_before_component_composition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), memory_enabled=True)
            events: list[str] = []
            provider = object()
            cipher = object()
            runtime = object()
            store = MagicMock()
            store.prepare_memory_storage.side_effect = lambda: (
                events.append("prepare")
                or {"storage_state": "ENCRYPTED", "ready": True}
            )
            store.close.side_effect = lambda: events.append("close")
            supervisor = MagicMock()
            deadline = MagicMock()

            def provider_factory(**_kwargs: object) -> object:
                events.append("provider")
                return provider

            def cipher_factory(selected: object) -> object:
                self.assertIs(selected, provider)
                events.append("cipher")
                return cipher

            def runtime_factory(selected: object) -> object:
                self.assertIs(selected, cipher)
                events.append("runtime")
                return runtime

            def store_factory(*_args: object, **_kwargs: object) -> object:
                events.append("store")
                return store

            async def run_components(*args: object, **_kwargs: object) -> None:
                self.assertIs(args[1], store)
                events.append("run")

            with (
                patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider",
                    side_effect=provider_factory,
                ) as provider_constructor,
                patch(
                    "jarvis_orchestrator.cli.MemoryEnvelopeCipherV1",
                    side_effect=cipher_factory,
                ),
                patch(
                    "jarvis_orchestrator.cli.ProviderBoundMemoryCryptoRuntime",
                    side_effect=runtime_factory,
                ),
                patch(
                    "jarvis_orchestrator.cli.OrchestratorStore",
                    side_effect=store_factory,
                ) as store_constructor,
                patch("jarvis_orchestrator.cli._run_components", new=run_components),
            ):
                asyncio.run(
                    _serve_components(
                        config,
                        voice_auth_key=object(),
                        runtime_supervisor=supervisor,
                        shutdown_deadline=deadline,
                    )
                )

            self.assertEqual(
                events,
                ["provider", "cipher", "runtime", "store", "prepare", "run", "close"],
            )
            provider_constructor.assert_called_once_with(
                state_root=Path(directory) / "operations" / "memory-keys"
            )
            self.assertIs(store_constructor.call_args.kwargs["memory_crypto"], runtime)
            self.assertIs(
                store_constructor.call_args.kwargs["memory_storage_enabled"], True
            )
            deadline.begin.assert_called_once_with()

    def test_disabled_runtime_makes_zero_provider_or_prepare_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), memory_enabled=False)
            store = MagicMock()
            supervisor = MagicMock()
            deadline = MagicMock()

            with (
                patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider"
                ) as provider_constructor,
                patch(
                    "jarvis_orchestrator.cli.MemoryEnvelopeCipherV1"
                ) as cipher_constructor,
                patch(
                    "jarvis_orchestrator.cli.ProviderBoundMemoryCryptoRuntime"
                ) as runtime_constructor,
                patch(
                    "jarvis_orchestrator.cli.OrchestratorStore",
                    return_value=store,
                ) as store_constructor,
                patch(
                    "jarvis_orchestrator.cli._run_components",
                    new=AsyncMock(),
                ) as run_components,
            ):
                asyncio.run(
                    _serve_components(
                        config,
                        voice_auth_key=object(),
                        runtime_supervisor=supervisor,
                        shutdown_deadline=deadline,
                    )
                )

            provider_constructor.assert_not_called()
            cipher_constructor.assert_not_called()
            runtime_constructor.assert_not_called()
            store.prepare_memory_storage.assert_not_called()
            self.assertIsNone(store_constructor.call_args.kwargs["memory_crypto"])
            self.assertIs(
                store_constructor.call_args.kwargs["memory_storage_enabled"], False
            )
            self.assertEqual(run_components.await_count, 1)
            self.assertFalse((Path(directory) / "operations").exists())

    def test_disabled_store_failure_keeps_original_non_memory_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), memory_enabled=False)
            failure = RuntimeError("ordinary store failure")
            with (
                patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider"
                ) as provider_constructor,
                patch(
                    "jarvis_orchestrator.cli.OrchestratorStore",
                    side_effect=failure,
                ),
                patch(
                    "jarvis_orchestrator.cli._run_components",
                    new=AsyncMock(),
                ) as run_components,
                self.assertRaises(RuntimeError) as raised,
            ):
                asyncio.run(
                    _serve_components(
                        config,
                        voice_auth_key=object(),
                        runtime_supervisor=MagicMock(),
                        shutdown_deadline=MagicMock(),
                    )
                )
            self.assertIs(raised.exception, failure)
            provider_constructor.assert_not_called()
            run_components.assert_not_awaited()

    def test_enabled_provider_and_prepare_failures_are_content_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), memory_enabled=True)
            with (
                patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider",
                    side_effect=RuntimeError("native path TOP_SECRET"),
                ),
                patch(
                    "jarvis_orchestrator.cli._run_components",
                    new=AsyncMock(),
                ) as run_components,
                self.assertRaises(OperationalError) as raised,
            ):
                asyncio.run(
                    _serve_components(
                        config,
                        voice_auth_key=object(),
                        runtime_supervisor=MagicMock(),
                        shutdown_deadline=MagicMock(),
                    )
                )
            self.assertEqual(str(raised.exception), "memory_storage_unavailable")
            self.assertNotIn("TOP_SECRET", str(raised.exception))
            run_components.assert_not_awaited()

            store = MagicMock()
            store.prepare_memory_storage.side_effect = RuntimeError(
                "provider binding TOP_SECRET"
            )
            with (
                patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider",
                    return_value=object(),
                ),
                patch(
                    "jarvis_orchestrator.cli.MemoryEnvelopeCipherV1",
                    return_value=object(),
                ),
                patch(
                    "jarvis_orchestrator.cli.ProviderBoundMemoryCryptoRuntime",
                    return_value=object(),
                ),
                patch(
                    "jarvis_orchestrator.cli.OrchestratorStore",
                    return_value=store,
                ),
                patch(
                    "jarvis_orchestrator.cli._run_components",
                    new=AsyncMock(),
                ) as run_components,
                self.assertRaises(OperationalError) as raised,
            ):
                asyncio.run(
                    _serve_components(
                        config,
                        voice_auth_key=object(),
                        runtime_supervisor=MagicMock(),
                        shutdown_deadline=MagicMock(),
                    )
                )
            self.assertEqual(str(raised.exception), "memory_storage_unavailable")
            self.assertNotIn("TOP_SECRET", str(raised.exception))
            run_components.assert_not_awaited()
            store.close.assert_called_once_with()

    def test_encrypted_database_rejects_mismatched_provider_without_bootstrap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root, memory_enabled=True)
            original = InMemoryMemoryEpochKeyProvider()
            store = OrchestratorStore(
                config.database_path,
                memory_crypto=ProviderBoundMemoryCryptoRuntime(
                    MemoryEnvelopeCipherV1(original)
                ),
                memory_storage_enabled=True,
            )
            store.prepare_memory_storage()
            store.close()
            replacement = _NoBootstrapProvider(InMemoryMemoryEpochKeyProvider())

            with (
                patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider",
                    return_value=replacement,
                ) as provider_constructor,
                patch(
                    "jarvis_orchestrator.cli._run_components",
                    new=AsyncMock(),
                ) as run_components,
                self.assertRaises(OperationalError) as raised,
            ):
                asyncio.run(
                    _serve_components(
                        config,
                        voice_auth_key=object(),
                        runtime_supervisor=MagicMock(),
                        shutdown_deadline=MagicMock(),
                    )
                )

            self.assertEqual(str(raised.exception), "memory_storage_unavailable")
            self.assertEqual(replacement.bootstrap_calls, 0)
            provider_constructor.assert_called_once_with(
                state_root=root / "operations" / "memory-keys"
            )
            run_components.assert_not_awaited()


class MemoryRuntimeServiceReadinessTests(unittest.TestCase):
    def test_readiness_snapshot_uses_cached_memory_state_and_degrades(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), memory_enabled=True)
            store = MagicMock()
            store.memory_storage_status.side_effect = [
                {
                    "storage_state": "ENCRYPTED",
                    "ready": True,
                    "pending_operation_count": 0,
                },
                {
                    "storage_state": "ENCRYPTED",
                    "ready": False,
                    "pending_operation_count": 0,
                },
            ]
            service = OrchestratorService(
                config,
                store,
                model=object(),
                active_memory=object(),
            )

            first = service.readiness_snapshot()
            self.assertTrue(first["core_ready"])
            self.assertTrue(first["memory_ready"])
            self.assertEqual(store.memory_storage_status.call_count, 1)

            second = service.readiness_snapshot()
            self.assertFalse(second["core_ready"])
            self.assertFalse(second["memory_ready"])
            self.assertTrue(second["inference_ready"])
            self.assertEqual(store.memory_storage_status.call_count, 2)
            store.memory_crypto_readiness.assert_not_called()

    def test_ready_is_a_wrapper_over_one_cached_memory_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), memory_enabled=True)
            store = MagicMock()
            store.memory_storage_status.return_value = {
                "storage_state": "ENCRYPTED",
                "ready": True,
                "pending_operation_count": 0,
            }
            service = OrchestratorService(
                config,
                store,
                model=object(),
                active_memory=object(),
            )

            self.assertTrue(service.ready())
            store.memory_storage_status.assert_called_once_with()
            store.memory_crypto_readiness.assert_not_called()

    def test_pending_memory_operation_keeps_readiness_closed_without_provider_call(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), memory_enabled=True)
            store = MagicMock()
            store.memory_storage_status.return_value = {
                "storage_state": "ENCRYPTED",
                "ready": True,
                "pending_operation_count": 1,
            }
            service = OrchestratorService(
                config,
                store,
                model=object(),
                active_memory=object(),
            )

            self.assertFalse(service.ready())
            store.memory_storage_status.assert_called_once_with()
            store.memory_crypto_readiness.assert_not_called()

    def test_disabled_ready_keeps_non_memory_behavior_and_skips_memory_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), memory_enabled=False)
            store = MagicMock()
            store.memory_storage_status.side_effect = AssertionError(
                "disabled readiness must not inspect memory storage"
            )
            store.memory_crypto_readiness.side_effect = AssertionError(
                "disabled readiness must not call the provider"
            )
            service = OrchestratorService(config, store, model=object())

            snapshot = service.readiness_snapshot()
            self.assertTrue(snapshot["core_ready"])
            self.assertFalse(snapshot["memory_ready"])
            self.assertTrue(snapshot["memory_requirement_satisfied"])
            self.assertEqual(snapshot["memory_state"], "DISABLED")
            self.assertFalse(service.memory_retrospective_ready())
            store.memory_storage_status.assert_not_called()
            store.memory_crypto_readiness.assert_not_called()


if __name__ == "__main__":
    unittest.main()
