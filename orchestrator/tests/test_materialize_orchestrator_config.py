from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import _bootstrap
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.windows_private_file import harden_windows_private_path

from jarvis_voice.voice_auth import create_voice_auth_key_file


def _load_materializer() -> ModuleType:
    path = _bootstrap.ROOT / "scripts" / "materialize_orchestrator_config.py"
    spec = importlib.util.spec_from_file_location(
        "test_materialize_orchestrator_config_script", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("materializer_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATERIALIZER = _load_materializer()


class MaterializeOrchestratorConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        if os.name == "nt":  # pragma: no cover - native Windows CI
            harden_windows_private_path(self.root)
        else:
            self.root.chmod(0o700)
        self.database_parent = self.root / "state"
        self.database_parent.mkdir(mode=0o700)
        if os.name == "nt":  # pragma: no cover - native Windows CI
            harden_windows_private_path(self.database_parent)
        self.database = self.database_parent / "orchestrator.sqlite3"
        self.auth_key = self.root / "voice-auth.json"
        create_voice_auth_key_file(self.auth_key, key=b"k" * 32)
        self.output = self.root / "orchestrator.json"

    def arguments(self) -> list[str]:
        return [
            "materialize-orchestrator-config",
            "--output",
            str(self.output),
            "--database",
            str(self.database),
            "--voice-auth-key",
            str(self.auth_key),
        ]

    def test_generated_config_round_trips_without_disclosing_tokens(self) -> None:
        rendered = io.StringIO()
        real_load = OrchestratorConfig.load
        with (
            patch.object(sys, "argv", self.arguments()),
            patch.object(
                MATERIALIZER.OrchestratorConfig,
                "load",
                wraps=real_load,
            ) as strict_load,
            redirect_stdout(rendered),
        ):
            status = MATERIALIZER.main()

        self.assertEqual(status, 0)
        self.assertEqual(strict_load.call_count, 2)
        report = json.loads(rendered.getvalue())
        self.assertEqual(
            report,
            {
                "code": "materialization.private_orchestrator_config_written",
                "memory_enabled": False,
                "path_disclosure": False,
                "ready": True,
                "secrets_disclosed": False,
            },
        )
        self.assertNotIn(str(self.root), rendered.getvalue())
        config = real_load(self.output)
        for token in (config.control_bearer_token, config.inference_bearer_token):
            decoded = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
            self.assertEqual(len(decoded), 48)
            self.assertEqual(
                base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii"),
                token,
            )
            self.assertNotIn(token, rendered.getvalue())
        if os.name != "nt":
            self.assertEqual(self.output.stat().st_mode & 0o777, 0o600)

    def test_failed_validation_preserves_a_replacement_file(self) -> None:
        rendered = io.StringIO()
        real_load = OrchestratorConfig.load
        replaced = self.root / "published-original.json"

        def replace_before_validation(path):
            if Path(path) == self.output:
                self.output.rename(replaced)
                self.output.write_bytes(b"competing writer")
                raise ValueError("synthetic validation failure")
            return real_load(path)

        with (
            patch.object(sys, "argv", self.arguments()),
            patch.object(
                MATERIALIZER.OrchestratorConfig,
                "load",
                side_effect=replace_before_validation,
            ),
            redirect_stdout(rendered),
        ):
            self.assertEqual(MATERIALIZER.main(), 2)
        self.assertEqual(self.output.read_bytes(), b"competing writer")
        self.assertTrue(replaced.exists())
        self.assertEqual(list(self.root.glob(".orchestrator-*")), [])
        self.assertNotIn(str(self.root), rendered.getvalue())

    def test_failed_validation_removes_only_its_published_file(self) -> None:
        rendered = io.StringIO()
        real_load = OrchestratorConfig.load

        def fail_after_publication(path):
            if Path(path) == self.output:
                raise ValueError("synthetic validation failure")
            return real_load(path)

        with (
            patch.object(sys, "argv", self.arguments()),
            patch.object(
                MATERIALIZER.OrchestratorConfig,
                "load",
                side_effect=fail_after_publication,
            ),
            redirect_stdout(rendered),
        ):
            self.assertEqual(MATERIALIZER.main(), 2)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".orchestrator-*")), [])

    def test_noncanonical_generated_token_fails_closed_and_is_removed(self) -> None:
        rendered = io.StringIO()
        with (
            patch.object(sys, "argv", self.arguments()),
            patch.object(MATERIALIZER.secrets, "token_urlsafe", return_value="weak"),
            redirect_stdout(rendered),
        ):
            status = MATERIALIZER.main()

        self.assertEqual(status, 2)
        self.assertEqual(
            json.loads(rendered.getvalue()),
            {
                "code": "materialization.private_config_validation_failed",
                "ready": False,
            },
        )
        self.assertNotIn("weak", rendered.getvalue())
        self.assertNotIn(str(self.root), rendered.getvalue())
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".orchestrator-*")), [])


if __name__ == "__main__":
    unittest.main()
