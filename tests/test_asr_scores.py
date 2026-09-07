from __future__ import annotations

import math
import unittest
from array import array
from types import SimpleNamespace

import _bootstrap  # noqa: F401
import test_asr
import test_cuda_worker
from jarvis_voice.asr import FasterWhisperCpuBackend, decode_bilingual_transcript, weighted_logprob
from jarvis_voice.backends import AudioChunk
from jarvis_voice.cuda_worker import run_cuda_worker
from jarvis_voice.errors import BackendError


class AsrScoreTests(unittest.TestCase):
    def test_invalid_segment_scores_never_look_perfect(self):
        for value in (float("nan"), float("inf"), float("-inf"), 0.1, 1000.0, True, False, None, "invalid", 10 ** 400):
            with self.subTest(value=repr(value)[:24]):
                with self.assertRaisesRegex(BackendError, "asr.invalid_score"):
                    weighted_logprob([SimpleNamespace(avg_logprob=value, tokens=[1, 2])])

    def test_weighted_average_does_not_overflow_finite_logprobs(self):
        segments = [SimpleNamespace(avg_logprob=-1e308, tokens=[1, 2]),
                    SimpleNamespace(avg_logprob=-1e307, tokens=[3])]
        raw, score = weighted_logprob(iter(segments))
        self.assertTrue(math.isfinite(raw))
        self.assertAlmostEqual(raw / -7e307, 1.0)
        self.assertEqual(score, 0.0)
        self.assertEqual(weighted_logprob([]), (float("-inf"), 0.0))
        self.assertEqual(weighted_logprob([SimpleNamespace(avg_logprob=0.0, tokens=[])]), (0.0, 1.0))

    def test_cpu_and_child_reject_invalid_language_scores(self):
        fixture = test_asr.AsrTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        class Model(test_asr.FakeModel):
            def transcribe(self, samples, **kwargs):
                segments, info = super().transcribe(samples, **kwargs)
                info.language_probability = value
                return segments, info
        audio = AudioChunk(array("f", [0.1]), 16000)
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, None, "invalid", 10 ** 400):
            with self.subTest(value=repr(value)[:24]):
                backend = FasterWhisperCpuBackend(str(fixture.directory), manifest=fixture.manifest,
                                                 model_factory=lambda *args, **kwargs: Model())
                with self.assertRaisesRegex(BackendError, "asr.invalid_score"):
                    backend.transcribe(audio)
                backend.close()
                connection = test_cuda_worker.MemoryConnection([
                    {"type": "transcribe", "request_id": "r", "sample_rate": 16000, "audio": audio.samples.tobytes()},
                    {"type": "shutdown"}])
                config = test_cuda_worker.CudaWorkerConfig(str(fixture.directory), 0, "int8_float16", 1, 1, 1024, fixture.manifest)
                run_cuda_worker(connection, config, model_factory=lambda *args, **kwargs: Model())
                self.assertEqual(connection.sent[1]["type"], "error")
                self.assertEqual(connection.sent[1]["code"], "asr.invalid_score")
                self.assertTrue(connection.closed)

    def test_invalid_candidate_is_skipped_and_valid_ties_stay_deterministic(self):
        for greek, english, expected in ((float("nan"), -0.2, "en"), (None, -0.2, "en"),
                                         (-0.2, -0.2, "el"), (-0.7, -0.2, "en")):
            with self.subTest(greek=greek, english=english):
                class Model:
                    def transcribe(self, samples, **kwargs):
                        language = kwargs.get("language", "auto")
                        raw = greek if language == "el" else english if language == "en" else -0.8
                        return iter([SimpleNamespace(text=language, avg_logprob=raw, tokens=[1])]), SimpleNamespace(language="el", language_probability=0.5)
                text, language, score, _ = decode_bilingual_transcript(Model(), ())
                self.assertEqual((text, language), (expected, expected))
                self.assertTrue(0.5 <= score <= 1.0)
                if greek == english:
                    self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
