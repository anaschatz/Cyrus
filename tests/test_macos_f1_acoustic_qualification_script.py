from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_macos_f1_acoustic_qualification.py"
SPEC = importlib.util.spec_from_file_location("run_macos_f1_acoustic_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MacosF1AcousticQualificationScriptTests(unittest.TestCase):
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
