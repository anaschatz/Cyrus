from __future__ import annotations

import asyncio
import hashlib
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.cli import (
    _memory_crypto_identity_state,
    _run_components,
    _run_servers,
    _serve,
    _serve_components,
    main,
)
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.memory_crypto import (
    InMemoryMemoryEpochKeyProvider,
    MemoryEnvelopeCipherV1,
)
from jarvis_orchestrator.memory_runtime import ProviderBoundMemoryCryptoRuntime
from jarvis_orchestrator.operations import OperationalError, operational_paths
from jarvis_orchestrator.private_state import (
    create_or_replace_private_file,
    ensure_private_directory,
)
from jarvis_orchestrator.runtime_supervisor import RuntimeFailure
from jarvis_orchestrator.store import OrchestratorStore

from jarvis_voice.voice_auth import VoiceAuthKey, create_voice_auth_key_file


class _CompositionObserved(RuntimeError):
    pass


class _ReadOnlyMemoryProvider:
    def __init__(self, delegate: InMemoryMemoryEpochKeyProvider) -> None:
        self.delegate = delegate
        self.bootstrap_calls = 0

    @property
    def install_id(self) -> str:
        return self.delegate.install_id

    def attest(self):
        return self.delegate.attest()

    def active_version(self) -> int:
        return self.delegate.active_version()

    def load(self, version: int) -> bytes:
        return self.delegate.load(version)

    def bootstrap(self):
        self.bootstrap_calls += 1
        raise AssertionError("readiness must not bootstrap memory keys")


class CliStartupTests(unittest.TestCase):
    def test_scheduler_startup_requires_live_identity_without_blocking_async_composition(self):
        import httpx
        from test_schedule_compiler import qualification_document
        from jarvis_orchestrator.schedule_compiler import CompilerQualification

        document = qualification_document()
        config = OrchestratorConfig.from_test_mapping({
            "database_path": "/unused/state.sqlite3",
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "scheduler_compiler_qualification_file": "/unused/qualification.json",
            "scheduler_compiler_qualification_sha256": "a" * 64,
        })
        live = {
            "schema_version": "jarvis-inference-compiler-identity/1",
            **{name: document["qualification_inputs"][name] for name in (
                "model_artifact_sha256", "model_profile_sha256", "runtime_sha256", "tokenizer_sha256"
            )},
        }
        client_type = httpx.AsyncClient
        for matches in (True, False):
            captured = {}

            def observe(*args, **kwargs):
                captured.update(kwargs)
                raise _CompositionObserved

            with (
                patch("jarvis_orchestrator.cli.platform_key_provider"),
                patch("jarvis_orchestrator.cli.SchedulerStore"),
                patch("jarvis_orchestrator.cli.load_compiler_qualification", return_value=CompilerQualification(
                    document["qualification_sha256"], document,
                )),
                patch("jarvis_orchestrator.cli.OrchestratorService", side_effect=observe),
                patch("httpx.AsyncClient", side_effect=lambda **kwargs: client_type(
                    transport=httpx.MockTransport(lambda request: httpx.Response(
                        200, json=live if matches else {**live, "tokenizer_sha256": "b" * 64},
                    )), **kwargs,
                )),
                self.subTest(matches=matches),
                self.assertRaises(_CompositionObserved),
            ):
                asyncio.run(_run_servers(
                    config, object(), model=object(), inference_control=object(),
                    shadow_memory=None, active_memory=None, research_adapter=None,
                    cloud_adapter=None, cloud_endpoints=(), voice_auth_key=VoiceAuthKey(1, b"k" * 32),
                ))
            self.assertEqual(captured["schedule_compiler"] is not None, matches)

    def test_operations_diagnostics_is_explicitly_database_only(self) -> None:
        config = OrchestratorConfig.from_test_mapping(
            {
                "database_path": "/unused/state.sqlite3",
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
                "memory_enabled": True,
            }
        )
        output = io.StringIO()
        with (
            patch(
                "jarvis_orchestrator.cli.OrchestratorConfig.load",
                return_value=config,
            ),
            patch(
                "jarvis_orchestrator.cli.database_diagnostics",
                return_value=SimpleNamespace(
                    to_dict=lambda: {
                        "ready": True,
                        "version": "operational-diagnostics/1",
                    }
                ),
            ),
            patch(
                "jarvis_orchestrator.cli.platform_memory_key_provider"
            ) as provider_constructor,
            redirect_stdout(output),
        ):
            status = main(
                ["--config", "/unused/config.json", "operations", "diagnostics"]
            )

        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["scope"], "database_only")
        self.assertFalse(report["live_memory_provider_checked"])
        provider_constructor.assert_not_called()

    def test_encrypted_backup_and_restore_receive_the_configured_attestor(
        self,
    ) -> None:
        database = Path("/unused/state.sqlite3")
        config = OrchestratorConfig.from_test_mapping(
            {
                "database_path": str(database),
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
                "memory_enabled": True,
            }
        )
        provider = SimpleNamespace(
            attest=lambda: "attested",
            exists=lambda version: version == 1,
            bootstrap=lambda: (_ for _ in ()).throw(
                AssertionError("backup CLI must never expose bootstrap")
            ),
        )
        with (
            patch(
                "jarvis_orchestrator.cli.OrchestratorConfig.load",
                return_value=config,
            ),
            patch(
                "jarvis_orchestrator.cli.platform_memory_key_provider",
                return_value=provider,
            ) as provider_constructor,
            patch(
                "jarvis_orchestrator.cli._database_has_encrypted_memory",
                return_value=True,
            ) as encrypted_probe,
            patch(
                "jarvis_orchestrator.cli.backup_database",
                return_value={"ready": True},
            ) as backup,
            redirect_stdout(io.StringIO()),
        ):
            status = main(
                [
                    "--config",
                    "/unused/config.json",
                    "operations",
                    "backup",
                    "/unused/backup.sqlite3",
                ]
        )
        self.assertEqual(status, 0)
        encrypted_probe.assert_called_once_with(database, immutable=False)
        backup.assert_called_once()
        self.assertEqual(
            backup.call_args.args,
            (database, Path("/unused/backup.sqlite3")),
        )
        wired_attestor = backup.call_args.kwargs["memory_provider_attestor"]
        self.assertEqual(wired_attestor.attest(), "attested")
        self.assertTrue(wired_attestor.exists(1))
        self.assertFalse(hasattr(wired_attestor, "bootstrap"))
        provider_constructor.assert_called_once_with(
            state_root=Path("/unused/operations/memory-keys")
        )

        with (
            patch(
                "jarvis_orchestrator.cli.OrchestratorConfig.load",
                return_value=config,
            ),
            patch(
                "jarvis_orchestrator.cli.platform_memory_key_provider",
                return_value=provider,
            ) as provider_constructor,
            patch(
                "jarvis_orchestrator.cli._database_has_encrypted_memory",
                return_value=True,
            ) as encrypted_probe,
            patch(
                "jarvis_orchestrator.cli.restore_database",
                return_value={"ready": True},
            ) as restore,
            redirect_stdout(io.StringIO()),
        ):
            status = main(
                [
                    "--config",
                    "/unused/config.json",
                    "operations",
                    "restore",
                    "/unused/backup.sqlite3",
                    "--confirm-database",
                    str(database),
                ]
        )
        self.assertEqual(status, 0)
        encrypted_probe.assert_called_once_with(
            Path("/unused/backup.sqlite3"),
            immutable=True,
        )
        restore.assert_called_once()
        self.assertEqual(
            restore.call_args.args,
            (Path("/unused/backup.sqlite3"), database),
        )
        self.assertEqual(
            restore.call_args.kwargs["service_lock"],
            Path("/unused/operations/orchestrator.lock"),
        )
        wired_attestor = restore.call_args.kwargs["memory_provider_attestor"]
        self.assertEqual(wired_attestor.attest(), "attested")
        self.assertTrue(wired_attestor.exists(1))
        self.assertFalse(hasattr(wired_attestor, "bootstrap"))
        provider_constructor.assert_called_once_with(
            state_root=Path("/unused/operations/memory-keys")
        )

    def test_outer_process_deadline_covers_synchronous_store_close(self) -> None:
        script = """
import asyncio
import time
from types import SimpleNamespace
import jarvis_orchestrator.cli as cli

cli.RUNTIME_TOTAL_SHUTDOWN_SECONDS = 0.10

class BlockingStore:
    def __init__(self, *_args, **_kwargs):
        pass

    def close(self):
        time.sleep(5)

async def run_components(*_args, **_kwargs):
    return None

cli.OrchestratorStore = BlockingStore
cli._run_components = run_components
config = SimpleNamespace(
    database_path='/unused/runtime.sqlite3',
    model_request_limit=1,
    output_token_limit=1,
    active_budget_seconds=1,
    approval_ttl_seconds=1,
    reconnect_ttl_seconds=1,
    delivery_ttl_seconds=1,
    tools_enabled=False,
    memory_enabled=False,
)
asyncio.run(cli._serve_components(config, voice_auth_key=None))
"""
        started = time.monotonic()
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            timeout=2.0,
        )
        self.assertEqual(completed.returncode, 70)
        self.assertLess(time.monotonic() - started, 1.5)

    def test_stop_request_created_before_serve_is_not_lost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auth_key = root / "voice-auth.json"
            create_voice_auth_key_file(auth_key, key=b"k" * 32)
            config = OrchestratorConfig.from_test_mapping(
                {
                    "database_path": str(root / "state.sqlite3"),
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                    "voice_auth_key_file": str(auth_key),
                }
            )
            paths = operational_paths(config.database_path)
            ensure_private_directory(paths["state"], create=True)
            create_or_replace_private_file(paths["stop"], b"stop\n")
            observed: list[bool] = []

            async def observe_startup(*_args: object, **_kwargs: object) -> None:
                observed.append(paths["stop"].is_file())

            with patch(
                "jarvis_orchestrator.cli._serve_components",
                side_effect=observe_startup,
            ):
                asyncio.run(_serve(config))
            self.assertEqual(observed, [True])
            self.assertFalse(paths["stop"].exists())

    def test_failed_serve_preserves_durable_stop_intent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auth_key = root / "voice-auth.json"
            create_voice_auth_key_file(auth_key, key=b"k" * 32)
            config = OrchestratorConfig.from_test_mapping(
                {
                    "database_path": str(root / "state.sqlite3"),
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                    "voice_auth_key_file": str(auth_key),
                }
            )
            paths = operational_paths(config.database_path)
            ensure_private_directory(paths["state"], create=True)
            create_or_replace_private_file(paths["stop"], b"stop\n")

            with (
                patch(
                    "jarvis_orchestrator.cli._serve_components",
                    new=AsyncMock(
                        side_effect=RuntimeFailure(
                            "runtime_gateway_shutdown_timeout",
                            component="gateway",
                        )
                    ),
                ),
                self.assertRaises(RuntimeFailure),
            ):
                asyncio.run(_serve(config))
            self.assertTrue(paths["stop"].is_file())

    def test_invalid_config_is_one_stable_content_free_error(self) -> None:
        output = io.StringIO()
        with patch(
            "jarvis_orchestrator.cli.OrchestratorConfig.load",
            side_effect=ValueError("unknown configuration field: TOP_SECRET"),
        ), patch("jarvis_orchestrator.cli._serve") as serve, redirect_stdout(output):
            status = main(["--config", "/private/TOP_SECRET.json", "serve"])

        self.assertEqual(status, 1)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "error": {
                    "code": "configuration_invalid",
                    "retryable": False,
                }
            },
        )
        self.assertNotIn("TOP_SECRET", output.getvalue())
        serve.assert_not_called()

    def test_runtime_failure_is_reported_exactly_and_exits_nonzero(self) -> None:
        config = OrchestratorConfig.from_test_mapping(
            {
                "database_path": "/tmp/jarvis-runtime-supervision-test.sqlite3",
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
            }
        )
        output = io.StringIO()
        with patch(
            "jarvis_orchestrator.cli.OrchestratorConfig.load",
            return_value=config,
        ), patch(
            "jarvis_orchestrator.cli._serve",
            new=AsyncMock(
                side_effect=RuntimeFailure(
                    "runtime_scheduler_failed",
                    component="scheduler_worker",
                )
            ),
        ), redirect_stdout(output):
            status = main(["--config", "/unused/config.json", "serve"])
        self.assertEqual(status, 1)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "error": {
                    "code": "runtime_scheduler_failed",
                    "retryable": False,
                }
            },
        )

    def test_missing_voice_auth_key_fails_before_operational_state_is_touched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = OrchestratorConfig.from_test_mapping(
                {
                    "database_path": str(Path(directory) / "state.sqlite3"),
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                }
            )
            with patch("jarvis_orchestrator.cli.operational_paths") as paths:
                with self.assertRaisesRegex(
                    OperationalError, "voice_auth_key_required"
                ):
                    asyncio.run(_serve(config))
            paths.assert_not_called()

    def test_local_inference_discovery_failure_fails_startup_after_bounded_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = OrchestratorConfig.from_test_mapping(
                {
                    "database_path": str(Path(directory) / "state.sqlite3"),
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                }
            )
            store = OrchestratorStore(config.database_path)
            self.addCleanup(store.close)
            with patch(
                "jarvis_orchestrator.cli.InferenceControlClient.capabilities",
                side_effect=RuntimeError("unavailable"),
            ) as capability, patch(
                "jarvis_orchestrator.cli.asyncio.sleep", new=AsyncMock()
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "local inference capability discovery failed"
                ):
                    asyncio.run(
                        _run_components(
                            config,
                            store,
                            voice_auth_key=VoiceAuthKey(1, b"k" * 32),
                        )
                    )
            self.assertEqual(capability.call_count, 3)

    def test_memory_enabled_wires_one_provider_and_prepares_before_serving(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            config = OrchestratorConfig.from_test_mapping(
                {
                    "database_path": str(path),
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                    "memory_enabled": True,
                }
            )
            provider = InMemoryMemoryEpochKeyProvider()
            with (
                patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider",
                    return_value=provider,
                ) as provider_factory,
                patch(
                    "jarvis_orchestrator.cli._run_components",
                    new=AsyncMock(),
                ) as run_components,
            ):
                asyncio.run(
                    _serve_components(
                        config,
                        voice_auth_key=VoiceAuthKey(1, b"k" * 32),
                    )
                )
                before = _memory_crypto_identity_state(config)
                after = _memory_crypto_identity_state(config)
            self.assertEqual(before, after)
            self.assertEqual(provider_factory.call_count, 3)
            self.assertTrue(
                all(
                    call.kwargs
                    == {"state_root": path.parent / "operations" / "memory-keys"}
                    for call in provider_factory.call_args_list
                )
            )
            self.assertEqual(provider.active_version(), 1)
            self.assertEqual(run_components.await_count, 1)
            with sqlite3.connect(path) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT storage_state FROM memory_crypto_state"
                    ).fetchone()[0],
                    "ENCRYPTED",
                )
                self.assertIsNone(
                    connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE name='memory_records_v9_legacy'"
                    ).fetchone()
                )

    def test_memory_readiness_identity_is_database_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            config = OrchestratorConfig.from_test_mapping(
                {
                    "database_path": str(path),
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                    "memory_enabled": True,
                }
            )
            provider = InMemoryMemoryEpochKeyProvider()
            runtime = ProviderBoundMemoryCryptoRuntime(
                MemoryEnvelopeCipherV1(provider)
            )
            store = OrchestratorStore(
                path,
                memory_crypto=runtime,
                memory_storage_enabled=True,
            )
            store.prepare_memory_storage()
            expected = store.memory_crypto_readiness()["state_sha256"]
            store.close()
            before_bytes = path.read_bytes()
            before_hash = hashlib.sha256(before_bytes).hexdigest()
            before_mtime = path.stat().st_mtime_ns
            read_only_provider = _ReadOnlyMemoryProvider(provider)

            with (
                patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider",
                    return_value=read_only_provider,
                ) as provider_factory,
                patch(
                    "jarvis_orchestrator.cli.OrchestratorStore",
                    side_effect=AssertionError("readiness opened a writable store"),
                ),
            ):
                observed = _memory_crypto_identity_state(config)

            self.assertEqual(observed, expected)
            self.assertEqual(read_only_provider.bootstrap_calls, 0)
            provider_factory.assert_called_once_with(
                state_root=path.parent / "operations" / "memory-keys"
            )
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before_hash)
            self.assertEqual(path.read_bytes(), before_bytes)
            self.assertEqual(path.stat().st_mtime_ns, before_mtime)

    def test_run_servers_uses_its_config_parameter_for_cloud_release_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            auth_key = Path(directory) / "voice-auth.json"
            create_voice_auth_key_file(auth_key, key=b"k" * 32)
            config = OrchestratorConfig.from_test_mapping(
                {
                    "database_path": str(Path(directory) / "state.sqlite3"),
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                    "voice_auth_key_file": str(auth_key),
                }
            )
            captured: dict[str, object] = {}

            def observe_composition(*args: object, **kwargs: object) -> None:
                captured["config"] = args[0]
                captured["cloud_release_verified"] = kwargs[
                    "cloud_release_verified"
                ]
                raise _CompositionObserved

            with patch(
                "jarvis_orchestrator.cli.OrchestratorService",
                side_effect=observe_composition,
            ):
                with self.assertRaises(_CompositionObserved):
                    asyncio.run(
                        _run_servers(
                            config,
                            object(),  # type: ignore[arg-type]
                            model=object(),
                            inference_control=object(),  # type: ignore[arg-type]
                            shadow_memory=None,
                            active_memory=None,
                            research_adapter=None,
                            cloud_adapter=None,
                            cloud_endpoints=(),
                            voice_auth_key=VoiceAuthKey(1, b"k" * 32),
                        )
                    )

            self.assertIs(captured["config"], config)
            self.assertIs(captured["cloud_release_verified"], False)


if __name__ == "__main__":
    unittest.main()
