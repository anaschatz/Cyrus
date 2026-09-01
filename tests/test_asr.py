from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from jarvis_voice.asr import FasterWhisperCpuBackend, choose_cpu_compute_type, weighted_logprob
from jarvis_voice.backends import AudioChunk
from jarvis_voice.errors import BackendError
from jarvis_voice.manifest import ModelFileSpec, ModelManifest


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def transcribe(self, samples: object, **kwargs: object):
        self.calls.append((samples, kwargs))
        segments = [
            SimpleNamespace(text="α\u0301νοιξε ", avg_logprob=-0.2, tokens=[1, 2]),
            SimpleNamespace(text="το αρχείο", avg_logprob=-0.4, tokens=[3]),
        ]
        return iter(segments), SimpleNamespace(language="el", language_probability=0.96)


class AsrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        content = b"model"
        (self.directory / "model.bin").write_bytes(content)
        self.manifest = ModelManifest("test", "revision", "fake", (ModelFileSpec("model.bin", len(content), hashlib.sha256(content).hexdigest()),))

    def test_manifest_is_checked_before_model_factory(self) -> None:
        calls: list[object] = []
        (self.directory / "model.bin").write_bytes(b"bad-size")
        with self.assertRaisesRegex(BackendError, "model.size_mismatch"):
            FasterWhisperCpuBackend(str(self.directory), model_factory=lambda *a, **k: calls.append(a), manifest=self.manifest)
        self.assertEqual(calls, [])

    def test_explicit_local_cpu_load_and_metadata(self) -> None:
        created: list[tuple[tuple[object, ...], dict[str, object]]] = []
        model = FakeModel()
        def factory(*args: object, **kwargs: object) -> FakeModel:
            created.append((args, kwargs)); return model
        backend = FasterWhisperCpuBackend(str(self.directory), model_factory=factory, manifest=self.manifest)
        result = backend.transcribe(AudioChunk(array("f", [0.1, 0.2]), 16_000))
        args, kwargs = created[0]
        self.assertEqual(args, (str(self.directory),))
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["compute_type"], "int8")
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual(result.text, "άνοιξε το αρχείο")
        self.assertEqual(result.language.code, "el")
        self.assertEqual(result.language.score, 0.96)
        self.assertFalse(result.asr_score.calibrated)
        self.assertNotIn("confidence", result.to_payload())
        self.assertEqual(
            model.calls[0][1],
            {
                "vad_filter": False,
                "task": "transcribe",
                "beam_size": 5,
                "condition_on_previous_text": False,
            },
        )

    def test_weighted_logprob_is_not_labeled_probability(self) -> None:
        raw, score = weighted_logprob([SimpleNamespace(avg_logprob=-0.2, tokens=[1, 2]), SimpleNamespace(avg_logprob=-0.5, tokens=[3])])
        self.assertAlmostEqual(raw, -0.3)
        self.assertAlmostEqual(score, math.exp(-0.3))

    def test_ambiguous_auto_language_is_resolved_by_bilingual_logprob(self) -> None:
        class AmbiguousModel:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def transcribe(self, samples: object, **kwargs: object):
                del samples
                self.calls.append(dict(kwargs))
                language = kwargs.get("language")
                if language == "en":
                    segments = [
                        SimpleNamespace(
                            text="clear English sentence",
                            avg_logprob=-0.2,
                            tokens=[1, 2, 3],
                        )
                    ]
                elif language == "el":
                    segments = [
                        SimpleNamespace(
                            text="λανθασμένη εκδοχή",
                            avg_logprob=-0.7,
                            tokens=[1, 2, 3],
                        )
                    ]
                else:
                    segments = [
                        SimpleNamespace(
                            text="uncertain",
                            avg_logprob=-0.8,
                            tokens=[1, 2],
                        )
                    ]
                return iter(segments), SimpleNamespace(
                    language="el",
                    language_probability=0.6,
                )

        model = AmbiguousModel()
        backend = FasterWhisperCpuBackend(
            str(self.directory),
            model_factory=lambda *args, **kwargs: model,
            manifest=self.manifest,
        )
        result = backend.transcribe(AudioChunk(array("f", [0.1, 0.2]), 16_000))
        self.assertEqual(result.language.code, "en")
        self.assertGreater(result.language.score, 0.5)
        self.assertEqual(result.text, "clear English sentence")
        self.assertEqual(
            [call.get("language") for call in model.calls],
            [None, "el", "en"],
        )
        self.assertTrue(
            all(
                call["beam_size"] == 5
                and call["condition_on_previous_text"] is False
                for call in model.calls
            )
        )

    def test_requires_16_khz_audio(self) -> None:
        backend = FasterWhisperCpuBackend(str(self.directory), model_factory=lambda *a, **k: FakeModel(), manifest=self.manifest)
        with self.assertRaisesRegex(BackendError, "asr.invalid_audio"):
            backend.transcribe(AudioChunk(array("f"), 48_000))

    def test_close_releases_the_persistent_cpu_model(self) -> None:
        backend = FasterWhisperCpuBackend(
            str(self.directory),
            model_factory=lambda *args, **kwargs: FakeModel(),
            manifest=self.manifest,
        )
        backend.load()
        self.assertIsNotNone(backend._model)

        backend.close()

        self.assertIsNone(backend._model)

    def test_compute_type_fallback_stays_on_cpu_types(self) -> None:
        self.assertEqual(choose_cpu_compute_type("float16", {"int8", "float32"}), "int8")
        with self.assertRaisesRegex(BackendError, "asr.cpu_compute_type_unsupported"):
            choose_cpu_compute_type("float16", {"cuda_only"})


if __name__ == "__main__":
    unittest.main()
