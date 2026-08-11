from __future__ import annotations

import contextlib
import io
import importlib
import json
import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_main_module_is_spawn_import_safe(self) -> None:
        self.assertIsNotNone(importlib.import_module("jarvis_voice.__main__"))

    def test_active_cuda_probe_is_explicit_and_config_only(self) -> None:
        parsed = build_parser().parse_args(["doctor", "--config", "local.json", "--active-cuda-probe"])
        self.assertTrue(parsed.active_cuda_probe)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["doctor", "--model-dir", "local-model", "--active-cuda-probe"])
        self.assertEqual(code, 2)
        report = json.loads(output.getvalue())
        self.assertEqual(report["error"]["code"], "cuda.active_probe_requires_config")
        self.assertFalse(report["network_used"])

    def test_milestone_three_commands_are_explicit(self) -> None:
        self.assertEqual(build_parser().parse_args(["tray"]).command, "tray")
        self.assertEqual(build_parser().parse_args(["dist-manifest", "--directory", "dist"]).directory, "dist")
        self.assertEqual(build_parser().parse_args(["verify-dist", "--directory", "dist"]).directory, "dist")

    def test_milestone_four_commands_are_explicit(self) -> None:
        self.assertEqual(build_parser().parse_args(["qualification-template", "--output", "samples.json"]).command, "qualification-template")
        parsed = build_parser().parse_args(["qualification-record", "--samples", "samples.json", "--case", "no_speech", "--outcome", "pass"])
        self.assertEqual(parsed.metric, [])
        parsed = build_parser().parse_args(["qualify-release", "--config", "voice.json", "--distribution", "dist", "--samples", "samples.json", "--frozen-doctor", "doctor.json", "--native-evidence", "native.json", "--output", "report.json"])
        self.assertEqual(parsed.command, "qualify-release")


if __name__ == "__main__":
    unittest.main()
