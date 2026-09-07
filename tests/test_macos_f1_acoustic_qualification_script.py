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

from jarvis_voice.errors import BackendError


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_macos_f1_acoustic_qualification.py"
SPEC = importlib.util.spec_from_file_location("run_macos_f1_acoustic_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MacosF1AcousticQualificationScriptTests(unittest.TestCase):
    def test_unwritable_output_returns_safe_failure_for_both_commands(self) -> None:
        for command, function in (("calibrate", "calibrate_macos_acoustic_threshold"),
                                  ("batch", "run_macos_acoustic_barge_in_sample")):
            for operation_fails in (False, True):
                with self.subTest(command=command, operation_fails=operation_fails), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "missing" / "report.json"
                    args = ["--config", "unused.json", "--model-root", "unused-model",
                            "--approval-manifest", "unused-approval.json", "--approval-manifest-sha256", "a" * 64,
                            command, "--output", str(output)]
                    if command == "batch":
                        args += ["--silence-threshold-dbfs", "-40"]
                    stdout = io.StringIO()
                    with (mock.patch.object(MODULE, "qualification_config", return_value=object()),
                          mock.patch.object(MODULE, function, return_value={"ready": True},
                                            side_effect=BackendError("qualification.test_failure", "private failure details", subsystem="qualification") if operation_fails else None),
                          contextlib.redirect_stdout(stdout)):
                        status = MODULE.main(args)
                    report = json.loads(stdout.getvalue()[stdout.getvalue().index("{"):])
                    self.assertEqual(status, 2)
                    self.assertFalse(report["ready"])
                    self.assertEqual(report["code"], "qualification.output_directory_missing")
                    self.assertEqual(report["maximum_samples_per_process"], 2)
                    self.assertNotIn("private failure details", stdout.getvalue())
                    self.assertFalse(output.exists())

    def test_batch_is_hard_bounded_to_two_samples_per_process(self) -> None:
        self.assertEqual(MODULE._bounded_sample_count("1"), 1)
        self.assertEqual(MODULE._bounded_sample_count("2"), 2)
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE._bounded_sample_count("3")

    def test_parser_defaults_to_two_samples_and_five_second_calibration(self) -> None:
        common = [
            "--config", "/tmp/config.json",
            "--model-root", "/tmp/model",
            "--approval-manifest", "/tmp/approval.json",
            "--approval-manifest-sha256", "a" * 64,
        ]
        batch = MODULE.build_parser().parse_args([*common, "batch", "--output", "/tmp/batch.json", "--silence-threshold-dbfs", "-44"])
        calibration = MODULE.build_parser().parse_args([*common, "calibrate", "--output", "/tmp/calibration.json"])
        self.assertEqual(batch.sample_count, 2)
        self.assertEqual(calibration.ambient_seconds, 5.0)


if __name__ == "__main__":
    unittest.main()
