from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.action_host_protocol import ActionHostProtocolError
from jarvis_orchestrator.action_host_security import load_instance_token
from jarvis_orchestrator.action_host_daemon import ActionHostDaemonConfig, initialize_token, serve
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.cli import _apply_native_action_qualification


class ActionHostDaemonConfigTests(unittest.TestCase):
    def value(self, root: Path) -> dict[str, object]:
        return {
            "enabled": False,
            "socket_path": str(root / "host.sock"),
            "token_path": str(root / "token"),
            "journal_path": str(root / "journal.sqlite3"),
            "reference_store_path": str(root / "references.sqlite3"),
            "notes_path": str(root / "notes"),
            "native_helper_path": str(root / "helper"),
            "native_helper_sha256": "a" * 64,
            "native_helper_code_identity_sha256": "b" * 64,
            "qualification_file": None,
            "qualification_sha256": "",
            "tool_roots": {"documents": str(root / "documents")},
            "tool_app_ids": ["finder"],
        }

    def load(self, root: Path, value: dict[str, object]) -> ActionHostDaemonConfig:
        path = root / "host.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return ActionHostDaemonConfig.load(path)

    def test_action_host_is_initially_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = self.load(root, self.value(root))
            self.assertFalse(selected.enabled)
            self.assertIsNone(selected.qualification_file)

    def test_configuration_rejects_duplicate_keys_and_non_finite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "host.json"
            path.write_text('{"enabled":false,"enabled":false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                ActionHostDaemonConfig.load(path)
            path.write_text('{"enabled":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                ActionHostDaemonConfig.load(path)

    def test_activation_requires_exact_qualification_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = self.value(root)
            value["enabled"] = True
            with self.assertRaisesRegex(ValueError, "qualification"):
                self.load(root, value)
            value["qualification_file"] = str(root / "qualification.json")
            value["qualification_sha256"] = "c" * 64
            selected = self.load(root, value)
            self.assertTrue(selected.enabled)

    def test_disabled_host_rejects_latent_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = self.value(root)
            value["qualification_file"] = str(root / "qualification.json")
            value["qualification_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "disabled action host"):
                self.load(root, value)

    def test_orchestrator_actions_are_off_by_default_and_require_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base: dict[str, object] = {
                "database_path": str(root / "orchestrator.sqlite3"),
                "control_bearer_token": "control-private",
                "inference_bearer_token": "inference-private",
            }
            self.assertFalse(OrchestratorConfig.from_test_mapping(base).tools_enabled)
            base.update(
                {
                    "tools_enabled": True,
                    "model_request_limit": 6,
                    "tool_roots": {"documents": str(root / "documents")},
                    "tool_app_ids": ["finder"],
                    "tool_policy_sha256": "a" * 64,
                    "scheduler_contract_sha256": "b" * 64,
                    "action_host_socket_path": str(root / "host.sock"),
                    "action_host_token_file": str(root / "token"),
                }
            )
            with self.assertRaisesRegex(ValueError, "qualification"):
                OrchestratorConfig.from_test_mapping(base)
            base["action_host_qualification_file"] = str(root / "qualification.json")
            base["action_host_qualification_sha256"] = "c" * 64
            self.assertTrue(OrchestratorConfig.from_test_mapping(base).tools_enabled)

    def test_bad_native_qualification_disables_actions_not_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = OrchestratorConfig.from_test_mapping(
                {
                    "database_path": str(root / "orchestrator.sqlite3"),
                    "control_bearer_token": "control-private",
                    "inference_bearer_token": "inference-private",
                    "tools_enabled": True,
                    "model_request_limit": 6,
                    "tool_roots": {"documents": str(root / "documents")},
                    "tool_app_ids": ["finder"],
                    "tool_policy_sha256": "a" * 64,
                    "scheduler_contract_sha256": "b" * 64,
                    "action_host_socket_path": str(root / "host.sock"),
                    "action_host_token_file": str(root / "token"),
                    "action_host_qualification_file": str(root / "missing.json"),
                    "action_host_qualification_sha256": "c" * 64,
                }
            )
            with patch(
                "jarvis_orchestrator.cli.load_native_qualification",
                side_effect=RuntimeError("not qualified"),
            ):
                degraded, registry = _apply_native_action_qualification(config)
            self.assertFalse(degraded.tools_enabled)
            self.assertIsNone(registry)
            self.assertEqual(degraded.inference_base_url, config.inference_base_url)



    def test_token_loader_rejects_readable_file_and_fifo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_bytes(bytes(32))
            os.chmod(path, 0o600)
            self.assertEqual(load_instance_token(path), bytes(32))
            os.chmod(path, 0o644)
            with self.assertRaises(ActionHostProtocolError):
                load_instance_token(path)
            path.unlink()
            os.mkfifo(path, 0o600)
            script = """import sys
from jarvis_orchestrator.action_host_security import load_instance_token
from jarvis_orchestrator.action_host_protocol import ActionHostProtocolError
try:
    load_instance_token(sys.argv[1])
except ActionHostProtocolError:
    sys.exit(0)
sys.exit(1)
"""
            result = subprocess.run([sys.executable, "-c", script, str(path)],
                                    capture_output=True, timeout=2)
            self.assertEqual(result.returncode, 0, result.stderr.decode())


    def test_configuration_rejects_invalid_scalar_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [(name, invalid) for name in (
                "socket_path", "token_path", "journal_path", "reference_store_path",
                "notes_path", "native_helper_path", "qualification_file")
                for invalid in (42, [], {})]
            cases += [("native_helper_sha256", int("1" * 64)),
                      ("native_helper_code_identity_sha256", int("2" * 64)),
                      ("tool_app_ids", ["finder", 123]), ("tool_roots", {"documents": []})]
            for field, value in cases:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.load(root, self.value(root) | {field: value})

    def test_configuration_rejects_deep_json_and_fifo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host.json"
            path.write_text('{"x":' + '[' * 1500 + '0' + ']' * 1500 + '}')
            with self.subTest(kind="deep"), self.assertRaises(ValueError):
                ActionHostDaemonConfig.load(path)
            path.unlink()
            os.mkfifo(path, 0o600)
            script = """import sys
from pathlib import Path
from jarvis_orchestrator.action_host_daemon import ActionHostDaemonConfig
try:
    ActionHostDaemonConfig.load(Path(sys.argv[1]))
except ValueError:
    sys.exit(0)
sys.exit(1)
"""
            result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, timeout=2)
            self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_serve_unwinds_all_acquired_resources_on_failures(self):
        from jarvis_orchestrator import action_host_daemon as daemon
        for failure in ("qualification", "journal", "server", "signal", "run", "restore", "adapter_close"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
                root = Path(directory)
                config = self.load(root, self.value(root) | {"enabled": True,
                    "qualification_file": str(root / "qualification.json"), "qualification_sha256": "c" * 64})
                adapter, journal, server = Mock(), Mock(), Mock()
                registry = SimpleNamespace(names=("app.open@1",), digest="a" * 64)
                qualification = SimpleNamespace(qualified_tools=registry.names, readiness_sha256="b" * 64)
                handlers = {daemon.signal.SIGINT: object(), daemon.signal.SIGTERM: object()}
                original = handlers.copy()
                def set_signal(number, handler):
                    if failure == "signal" and number == daemon.signal.SIGTERM and callable(handler):
                        raise RuntimeError("injected")
                    if failure == "restore" and number == daemon.signal.SIGTERM and handler is original[number]:
                        raise RuntimeError("injected")
                    prior = handlers[number]
                    handlers[number] = handler
                    return prior
                patches.enter_context(patch.object(daemon.sys, "platform", "darwin"))
                patches.enter_context(patch.object(daemon, "load_instance_token"))
                patches.enter_context(patch.object(daemon, "platform_key_provider"))
                patches.enter_context(patch.object(daemon.ToolRegistryV1, "build", return_value=registry))
                patches.enter_context(patch.object(daemon, "MacOSNativeAdapterV1", return_value=adapter))
                qualified = patches.enter_context(patch.object(daemon, "load_native_qualification", return_value=qualification))
                patches.enter_context(patch.object(daemon, "current_macos_build", return_value="24G90"))
                patches.enter_context(patch.object(daemon, "readiness_digest", return_value="b" * 64))
                journal_factory = patches.enter_context(patch.object(daemon, "ActionHostResultJournalV2", return_value=journal))
                server_factory = patches.enter_context(patch.object(daemon, "ActionHostV2Server", return_value=server))
                patches.enter_context(patch.object(daemon.signal, "signal", side_effect=set_signal))
                if failure == "qualification": qualified.side_effect = RuntimeError("injected")
                if failure == "journal": journal_factory.side_effect = RuntimeError("injected")
                if failure == "server": server_factory.side_effect = RuntimeError("injected")
                if failure == "run": server.serve_forever.side_effect = RuntimeError("injected")
                if failure == "adapter_close": adapter.close.side_effect = RuntimeError("injected")
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    serve(config)
                adapter.close.assert_called_once()
                if failure not in ("qualification", "journal"):
                    journal.close.assert_called_once()
                for number in original:
                    if failure == "restore" and number == daemon.signal.SIGTERM:
                        continue
                    self.assertIs(handlers[number], original[number])

    def test_token_initialization_preserves_existing_and_rejects_symlink_parent(self):
        from jarvis_orchestrator.private_state import PrivateStateError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = root / "private" / "token"
            initialize_token(token)
            first = load_instance_token(token)
            initialize_token(token)
            self.assertEqual(load_instance_token(token), first)
            outside = root / "outside"
            outside.mkdir(mode=0o755)
            os.chmod(outside, 0o755)
            linked = root / "linked"
            linked.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(PrivateStateError):
                initialize_token(linked / "token")
            self.assertEqual(outside.stat().st_mode & 0o777, 0o755)
            self.assertFalse((outside / "token").exists())

    def test_failed_token_initialization_leaves_no_partial_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = root / "token"
            with patch("jarvis_orchestrator.action_host_daemon.os.fsync", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    initialize_token(token)
            self.assertEqual(list(root.iterdir()), [])
            initialize_token(token)
            self.assertEqual(len(load_instance_token(token)), 32)

if __name__ == "__main__":
    unittest.main()
