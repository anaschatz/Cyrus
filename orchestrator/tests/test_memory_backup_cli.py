from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.cli import main
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.memory_crypto import (
    InMemoryMemoryEpochKeyProvider,
    MemoryEnvelopeCipherV1,
)
from jarvis_orchestrator.memory_runtime import ProviderBoundMemoryCryptoRuntime
from jarvis_orchestrator.operations import backup_database
from jarvis_orchestrator.store import OrchestratorStore


class _NoBootstrapAttestorProvider:
    def __init__(self, delegate: InMemoryMemoryEpochKeyProvider) -> None:
        self.delegate = delegate
        self.bootstrap_calls = 0

    def attest(self):
        return self.delegate.attest()

    def exists(self, version: int) -> bool:
        return self.delegate.exists(version)

    def bootstrap(self):
        self.bootstrap_calls += 1
        raise AssertionError("backup/restore must never bootstrap memory keys")


def _config(database: Path, *, memory_enabled: bool) -> OrchestratorConfig:
    return OrchestratorConfig.from_test_mapping(
        {
            "database_path": str(database),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
            "memory_enabled": memory_enabled,
        }
    )


def _encrypted_database(path: Path) -> InMemoryMemoryEpochKeyProvider:
    provider = InMemoryMemoryEpochKeyProvider()
    store = OrchestratorStore(
        path,
        memory_crypto=ProviderBoundMemoryCryptoRuntime(
            MemoryEnvelopeCipherV1(provider)
        ),
        memory_storage_enabled=True,
    )
    status = store.prepare_memory_storage()
    if status.get("ready") is not True:
        raise AssertionError("encrypted test database was not prepared")
    store.checkpoint()
    store.close()
    return provider


def _run(config: OrchestratorConfig, arguments: list[str]) -> tuple[int, str]:
    output = io.StringIO()
    with (
        patch(
            "jarvis_orchestrator.cli.OrchestratorConfig.load",
            return_value=config,
        ),
        redirect_stdout(output),
    ):
        status = main(["--config", "/unused/config.json", *arguments])
    return status, output.getvalue()


class MemoryBackupCliTests(unittest.TestCase):
    def test_encrypted_backup_and_restore_use_narrow_attestor_without_bootstrap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            restored = root / "restored.sqlite3"
            provider = _encrypted_database(database)
            guarded = _NoBootstrapAttestorProvider(provider)
            config = _config(database, memory_enabled=False)

            with patch(
                "jarvis_orchestrator.cli.platform_memory_key_provider",
                return_value=guarded,
            ) as provider_constructor:
                backup_status, backup_output = _run(
                    config,
                    ["operations", "backup", str(backup)],
                )
            self.assertEqual(backup_status, 0)
            self.assertTrue(json.loads(backup_output)["memory_manifest"])
            provider_constructor.assert_called_once_with(
                state_root=root / "operations" / "memory-keys"
            )
            self.assertEqual(guarded.bootstrap_calls, 0)

            with patch(
                "jarvis_orchestrator.cli.platform_memory_key_provider",
                return_value=guarded,
            ) as provider_constructor:
                restore_status, restore_output = _run(
                    _config(restored, memory_enabled=False),
                    [
                        "operations",
                        "restore",
                        str(backup),
                        "--confirm-database",
                        str(restored),
                    ],
                )
            self.assertEqual(restore_status, 0)
            self.assertTrue(json.loads(restore_output)["memory_manifest"])
            provider_constructor.assert_called_once_with(
                state_root=root / "operations" / "memory-keys"
            )
            self.assertEqual(guarded.bootstrap_calls, 0)

    def test_missing_and_wrong_provider_refuse_encrypted_operations_content_free(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            restored = root / "restored.sqlite3"
            provider = _encrypted_database(database)
            config = _config(database, memory_enabled=True)

            with patch(
                "jarvis_orchestrator.cli.platform_memory_key_provider",
                side_effect=RuntimeError("TOP_SECRET native failure"),
            ):
                status, output = _run(
                    config,
                    ["operations", "backup", str(backup)],
                )
            self.assertEqual(status, 1)
            self.assertEqual(
                json.loads(output),
                {
                    "error": {
                        "code": "memory_backup_provider_unavailable",
                        "retryable": False,
                    }
                },
            )
            self.assertNotIn("TOP_SECRET", output)

            wrong = _NoBootstrapAttestorProvider(
                InMemoryMemoryEpochKeyProvider.initialized()
            )
            with patch(
                "jarvis_orchestrator.cli.platform_memory_key_provider",
                return_value=wrong,
            ):
                status, output = _run(
                    config,
                    ["operations", "backup", str(root / "wrong.sqlite3")],
                )
            self.assertEqual(status, 1)
            self.assertEqual(
                json.loads(output)["error"]["code"],
                "memory_backup_provider_install_mismatch",
            )
            self.assertEqual(wrong.bootstrap_calls, 0)

            backup_database(database, backup, memory_provider_attestor=provider)
            with patch(
                "jarvis_orchestrator.cli.platform_memory_key_provider",
                return_value=wrong,
            ):
                status, output = _run(
                    _config(restored, memory_enabled=True),
                    [
                        "operations",
                        "restore",
                        str(backup),
                        "--confirm-database",
                        str(restored),
                    ],
                )
            self.assertEqual(status, 1)
            self.assertEqual(
                json.loads(output)["error"]["code"],
                "memory_backup_provider_install_mismatch",
            )
            self.assertFalse(restored.exists())
            self.assertEqual(wrong.bootstrap_calls, 0)

    def test_disabled_and_non_encrypted_paths_make_zero_provider_calls(self) -> None:
        for memory_enabled in (False, True):
            with (
                self.subTest(memory_enabled=memory_enabled),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = root / "source.sqlite3"
                backup = root / "backup.sqlite3"
                destination = root / "destination.sqlite3"
                store = OrchestratorStore(source)
                store.checkpoint()
                store.close()

                with patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider"
                ) as provider_constructor:
                    status, output = _run(
                        _config(source, memory_enabled=memory_enabled),
                        ["operations", "backup", str(backup)],
                    )
                self.assertEqual(status, 0)
                self.assertFalse(json.loads(output)["memory_manifest"])
                provider_constructor.assert_not_called()

                with patch(
                    "jarvis_orchestrator.cli.platform_memory_key_provider"
                ) as provider_constructor:
                    status, output = _run(
                        _config(destination, memory_enabled=memory_enabled),
                        [
                            "operations",
                            "restore",
                            str(backup),
                            "--confirm-database",
                            str(destination),
                        ],
                    )
                self.assertEqual(status, 0)
                self.assertFalse(json.loads(output)["memory_manifest"])
                provider_constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
