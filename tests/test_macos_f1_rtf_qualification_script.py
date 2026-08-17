from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.macos.greektts15_runtime import PcmChunk


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_macos_f1_rtf_qualification.py"
SPEC = importlib.util.spec_from_file_location("run_macos_f1_rtf_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeEngine:
    def __init__(self) -> None:
        self.cancelled = False

    def warm(self) -> float:
        return 12.5

    def stream(self, unit, cancelled):
        del unit, cancelled
        yield PcmChunk(b"\x00\x00" * 24_000)

    def cancel(self) -> None:
        self.cancelled = True


class MacosF1RtfQualificationScriptTests(unittest.TestCase):
    def test_measurement_uses_emitted_pcm_duration_and_excludes_warm_load(self) -> None:
        times = iter((10.0, 10.2, 10.5))
        engine = FakeEngine()
        report = MODULE.measure_rtf_batch(engine, (("one", "Κείμενο."),), clock=lambda: next(times))
        self.assertTrue(report["ready"])
        self.assertEqual(report["warm_load_ms"], 12.5)
        self.assertEqual(report["mean_realtime_factor"], 0.5)
        self.assertEqual(report["samples"][0]["time_to_first_pcm_ms"], 200.0)
        self.assertTrue(engine.cancelled)

    def test_batch_is_hard_bounded_to_two_fixtures(self) -> None:
        self.assertEqual(MODULE._bounded_sample_count("2"), 2)
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE._bounded_sample_count("3")


if __name__ == "__main__":
    unittest.main()
