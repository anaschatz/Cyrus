from __future__ import annotations

import contextlib
import io
import importlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        template = build_parser().parse_args([
            "qualification-template",
            "--distribution",
            "dist",
            "--output",
            "samples.json",
        ])
        self.assertEqual(template.command, "qualification-template")
        self.assertEqual(template.distribution, "dist")
        parsed = build_parser().parse_args(["qualification-record", "--samples", "samples.json", "--case", "no_speech", "--outcome", "pass"])
        self.assertEqual(parsed.metric, [])
        parsed = build_parser().parse_args(["qualify-release", "--config", "voice.json", "--distribution", "dist", "--samples", "samples.json", "--frozen-doctor", "doctor.json", "--native-evidence", "native.json", "--output", "report.json"])
        self.assertEqual(parsed.command, "qualify-release")

    def test_create_auth_key_is_exclusive_private_and_content_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "voice-auth.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "create-auth-key",
                        "--output",
                        str(output_path),
                        "--key-version",
                        "7",
                    ]
                )
            report = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(report["code"], "voice_auth.key_created")
            self.assertEqual(report["key_version"], 7)
            self.assertFalse(report["secret_disclosed"])
            self.assertNotIn(str(output_path), stdout.getvalue())
            if os.name != "nt":
                self.assertEqual(output_path.stat().st_mode & 0o777, 0o600)
            original_key_file = output_path.read_bytes()

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                second_code = main(
                    ["create-auth-key", "--output", str(output_path)]
                )
            self.assertEqual(second_code, 2)
            self.assertEqual(
                json.loads(stdout.getvalue())["error"]["code"],
                "voice_auth.key_create_failed",
            )
            self.assertNotIn(str(output_path), stdout.getvalue())
            self.assertEqual(output_path.read_bytes(), original_key_file)

    @unittest.skipUnless(os.name == "posix", "POSIX owner/mode contract")
    def test_harden_private_state_migrates_all_journal_families_content_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "voice" / "state"
            state.mkdir(parents=True)
            state.chmod(0o755)
            journals = tuple(
                state / name
                for name in (
                    "delivery.sqlite3",
                    "cloud.sqlite3",
                    "action.sqlite3",
                )
            )
            for journal in journals:
                journal.write_bytes(b"")
                journal.chmod(0o644)
            wal = Path(f"{journals[0]}-wal")
            wal.write_bytes(b"")
            wal.chmod(0o644)

            stdout = io.StringIO()
            with (
                patch(
                    "jarvis_voice.private_state._default_journal_paths",
                    return_value=journals,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                code = main(["harden-private-state"])

            report = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(report["ready"])
            self.assertFalse(report["content_captured"])
            self.assertFalse(report["network_used"])
            self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o700)
            for path in (*journals, wal):
                self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o600)
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertNotIn("delivery.sqlite3", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
