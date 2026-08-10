from __future__ import annotations

import hashlib
import tempfile
import unittest
from array import array
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.audio import BoundedCaptureBuffer, resample_audio
from jarvis_voice.backends import AudioChunk
from jarvis_voice.errors import BackendError
from jarvis_voice.vad import FakeVadBackend, SileroOnnxVad, validate_artifact


class AudioVadTests(unittest.TestCase):
    def test_frames_before_ptt_are_discarded_without_preroll(self) -> None:
        capture = BoundedCaptureBuffer(capacity=2, sample_rate=16_000)
        self.assertFalse(capture.publish_from_callback(array("f", [9.0]).tobytes()))
        capture.begin_turn()
        self.assertTrue(capture.publish_from_callback(array("f", [1.0, 2.0]).tobytes()))
        audio = capture.end_turn()
        self.assertEqual(list(audio.samples), [1.0, 2.0])
        self.assertEqual(capture.stats.discarded_frames, 1)

    def test_audio_overflow_discards_partial_turn(self) -> None:
        capture = BoundedCaptureBuffer(capacity=1, sample_rate=16_000)
        capture.begin_turn(); capture.publish_from_callback(array("f", [1]).tobytes())
        self.assertFalse(capture.publish_from_callback(array("f", [2]).tobytes()))
        with self.assertRaisesRegex(BackendError, "audio.capture_overflow"):
            capture.end_turn()

    def test_maximum_duration_is_fail_safe(self) -> None:
        capture = BoundedCaptureBuffer(capacity=10, sample_rate=2, maximum_samples=2)
        capture.begin_turn()
        capture.publish_from_callback(array("f", [1, 2]).tobytes())
        self.assertFalse(capture.publish_from_callback(array("f", [3]).tobytes()))
        with self.assertRaises(BackendError):
            capture.end_turn()

    def test_resampling_runs_through_injected_worker_function(self) -> None:
        calls: list[tuple[int, int]] = []
        def fake(samples: array, source: int, target: int) -> array:
            calls.append((source, target)); return array("f", samples[::2])
        converted = resample_audio(AudioChunk(array("f", [1, 2, 3, 4]), 32_000), resampler=fake)
        self.assertEqual(converted.sample_rate, 16_000)
        self.assertEqual(list(converted.samples), [1, 3])
        self.assertEqual(calls, [(32_000, 16_000)])

    def test_fake_vad_trims_and_reports_no_speech(self) -> None:
        audio = AudioChunk(array("f", [0, 1, 2, 0]), 16_000)
        result = FakeVadBackend(start_sample=1, end_sample=3).trim(audio)
        self.assertTrue(result.speech_detected)
        self.assertEqual(list(result.audio.samples), [1, 2])
        self.assertFalse(FakeVadBackend(start_sample=None).trim(audio).speech_detected)

    def test_silero_requires_exact_artifact_and_trims_injected_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "silero.onnx"; content = b"onnx-fixture"; path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            ok, code = validate_artifact(str(path), len(content), digest)
            self.assertTrue(ok); self.assertEqual(code, "vad.artifact_valid")
            vad = SileroOnnxVad(
                artifact_path=str(path), size=len(content), sha256=digest,
                threshold=0.5, minimum_speech_ms=1, padding_ms=1,
                detector=lambda audio: [(20, 60, 0.9)],
            )
            result = vad.trim(AudioChunk(array("f", range(100)), 1_000))
            self.assertEqual((result.speech_start_sample, result.speech_end_sample), (19, 61))
            self.assertEqual(len(result.audio.samples), 42)
            with self.assertRaisesRegex(BackendError, "vad.artifact_hash_mismatch"):
                SileroOnnxVad(artifact_path=str(path), size=len(content), sha256="0" * 64, threshold=0.5, minimum_speech_ms=1, padding_ms=1)

    def test_silero_state_contract_prepends_rolling_context(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy runtime is exercised in the sealed macOS environment")

        class Port:
            def __init__(self, name: str) -> None:
                self.name = name

        class Session:
            def __init__(self) -> None:
                self.inputs: list[object] = []

            def get_inputs(self):  # type: ignore[no-untyped-def]
                return [Port("input"), Port("state"), Port("sr")]

            def run(self, outputs, feeds):  # type: ignore[no-untyped-def]
                del outputs
                self.inputs.append(feeds["input"].copy())
                return [numpy.array([[0.9]], dtype=numpy.float32), numpy.zeros((2, 1, 128), dtype=numpy.float32)]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "silero.onnx"
            content = b"model"
            path.write_bytes(content)
            vad = SileroOnnxVad(
                artifact_path=str(path),
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                threshold=0.5,
                minimum_speech_ms=1,
                padding_ms=0,
            )
            session = Session()
            vad._numpy = numpy
            vad._session = session
            vad.trim(AudioChunk(array("f", [0.25] * 1024), 16_000))
        self.assertEqual([item.shape for item in session.inputs], [(1, 576), (1, 576)])
        self.assertTrue(numpy.all(session.inputs[0][0, :64] == 0.0))
        self.assertTrue(numpy.all(session.inputs[1][0, :64] == 0.25))


if __name__ == "__main__":
    unittest.main()
