from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.errors import BackendError


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_macos_native_collection.py"
SPEC = importlib.util.spec_from_file_location("run_macos_native_collection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _arguments(**overrides):
    values = {
        "f1_worker_python": None,
        "f1_worker_python_size": None,
        "f1_worker_python_sha256": None,
        "f1_model_root": None,
        "f1_approval_manifest": None,
        "f1_approval_manifest_sha256": None,
        "f1_repository_root": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class MacosNativeCollectionScriptTests(unittest.TestCase):
    def test_no_sidecar_arguments_preserve_config(self) -> None:
        config = voice_config_from_mapping({})
        self.assertIs(MODULE.apply_f1_sidecar_profile(config, _arguments()), config)

    def test_complete_sidecar_profile_selects_f1_without_mutating_source(self) -> None:
        config = voice_config_from_mapping({})
        selected = MODULE.apply_f1_sidecar_profile(
            config,
            _arguments(
                f1_worker_python=Path("/private/venv/bin/python"),
                f1_worker_python_size=49_968,
                f1_worker_python_sha256="a" * 64,
                f1_model_root=Path("/private/model"),
                f1_approval_manifest=Path("/private/approved.json"),
                f1_approval_manifest_sha256="b" * 64,
                f1_repository_root=Path("/private/repository"),
            ),
        )
        self.assertEqual(config.macos.tts_backend, "system_say")
        self.assertEqual(selected.macos.tts_backend, "greektts15_f1")
        self.assertEqual(selected.macos.greektts15.worker_python.sha256, "a" * 64)
        self.assertTrue(selected.macos.greektts15.warm_on_start)

    def test_partial_sidecar_profile_fails_closed(self) -> None:
        with self.assertRaisesRegex(BackendError, "qualification.f1_sidecar_profile_incomplete"):
            MODULE.apply_f1_sidecar_profile(
                voice_config_from_mapping({}),
                _arguments(f1_worker_python=Path("/private/python")),
            )


if __name__ == "__main__":
    unittest.main()
