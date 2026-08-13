from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.greektts15_acoustics import analyze_acoustic_file, compare_acoustic_sets


class GreekTts15AcousticTests(unittest.TestCase):
    def _tone(self, path: Path, *, jump: bool = False) -> None:
        import numpy as np
        import soundfile as sf

        sample_rate = 24_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        frequency = np.where(time < 0.5, 180.0, 360.0 if jump else 180.0)
        phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
        sf.write(path, 0.3 * np.sin(phase), sample_rate)

    @unittest.skipUnless(find_spec("numpy") and find_spec("soundfile"), "optional acoustic dependencies")
    def test_pitch_jump_metric_detects_octave_discontinuity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stable = Path(temp) / "stable.wav"
            jump = Path(temp) / "jump.wav"
            self._tone(stable)
            self._tone(jump, jump=True)
            stable_report = analyze_acoustic_file(stable)
            jump_report = analyze_acoustic_file(jump)
        self.assertGreater(jump_report["pitch_jump_rate"], stable_report["pitch_jump_rate"])

    def test_comparison_requires_identical_four_way_fixture_sets(self) -> None:
        with self.assertRaises(BackendError) as raised:
            compare_acoustic_sets({"native": {}})
        self.assertEqual(raised.exception.code, "tts.greektts15.acoustic_set_incomplete")

    def test_comparison_rejects_unknown_candidate_label(self) -> None:
        with self.assertRaises(BackendError):
            compare_acoustic_sets({"native": {"one": "/x"}, "baseline": {"one": "/x"}, "unknown": {"one": "/x"}})


if __name__ == "__main__":
    unittest.main()
