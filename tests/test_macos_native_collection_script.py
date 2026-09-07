from __future__ import annotations

import argparse
import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    def test_unwritable_output_returns_safe_failure_for_all_commands(self) -> None:
        for command, function in (("calibrate", "calibrate_macos_acoustic_threshold"),
                                  ("preflight", "macos_native_collection_preflight"),
                                  ("run", "run_macos_native_collection")):
            for operation_fails in (False, True):
                with self.subTest(command=command, operation_fails=operation_fails), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "missing" / "report.json"
                    args = [command, "--config", "unused.json", "--audio-path", "wired", "--output", str(output)]
                    if command != "calibrate":
                        args += ["--silence-threshold-dbfs", "-40"]

                    def operation(*args, **kwargs):
                        if operation_fails:
                            raise BackendError("qualification.test_failure", "private failure details", subsystem="qualification")
                        if command == "run":
                            MODULE.write_report(kwargs["output"], {"ready": True})
                        return {"ready": True}

                    stdout = io.StringIO()
                    with (mock.patch.object(MODULE, "load_voice_config", return_value=voice_config_from_mapping({})),
                          mock.patch.object(MODULE, function, side_effect=operation), contextlib.redirect_stdout(stdout)):
                        status = MODULE.main(args)
                    report = json.loads(stdout.getvalue()[stdout.getvalue().index("{"):])
                    self.assertEqual(status, 2)
                    self.assertFalse(report["ready"])
                    self.assertEqual(report["code"], "qualification.test_failure" if operation_fails else "qualification.output_directory_missing")
                    self.assertEqual(report["output_error_code"], "qualification.output_directory_missing")
                    self.assertNotIn("private failure details", stdout.getvalue())
                    self.assertFalse(output.exists())

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

    def test_malformed_sidecar_digest_fails_before_runtime(self) -> None:
        with self.assertRaisesRegex(BackendError, "qualification.f1_sidecar_profile_invalid"):
            MODULE.apply_f1_sidecar_profile(
                voice_config_from_mapping({}),
                _arguments(
                    f1_worker_python=Path("/private/python"),
                    f1_worker_python_size=49_968,
                    f1_worker_python_sha256="short",
                    f1_model_root=Path("/private/model"),
                    f1_approval_manifest=Path("/private/approved.json"),
                    f1_approval_manifest_sha256="b" * 64,
                    f1_repository_root=Path("/private/repository"),
                ),
            )

    def test_loopback_port_override_is_bounded_and_non_mutating(self) -> None:
        config = voice_config_from_mapping({"orchestrator_url": "ws://127.0.0.1:8765"})
        selected = MODULE.apply_loopback_port(config, 18_765)
        self.assertEqual(config.orchestrator_url, "ws://127.0.0.1:8765")
        self.assertEqual(selected.orchestrator_url, "ws://127.0.0.1:18765")
        with self.assertRaisesRegex(BackendError, "qualification.loopback_port_invalid"):
            MODULE.apply_loopback_port(config, 80)

    def test_ptt_override_is_process_local(self) -> None:
        config = voice_config_from_mapping({"ptt_key": "F9"})
        selected = MODULE.apply_ptt_key(config, "CONTROL_OPTION")
        self.assertEqual(config.ptt_key, "F9")
        self.assertEqual(selected.ptt_key, "CONTROL_OPTION")


if __name__ == "__main__":
    unittest.main()
