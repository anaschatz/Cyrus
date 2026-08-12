from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.action_host_daemon import ActionHostDaemonConfig
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
            self.assertFalse(OrchestratorConfig.from_mapping(base).tools_enabled)
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
                OrchestratorConfig.from_mapping(base)
            base["action_host_qualification_file"] = str(root / "qualification.json")
            base["action_host_qualification_sha256"] = "c" * 64
            self.assertTrue(OrchestratorConfig.from_mapping(base).tools_enabled)

    def test_bad_native_qualification_disables_actions_not_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = OrchestratorConfig.from_mapping(
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


if __name__ == "__main__":
    unittest.main()
