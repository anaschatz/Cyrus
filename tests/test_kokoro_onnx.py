from __future__ import annotations

import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.config import ArtifactConfig, voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.kokoro_onnx import (
    KokoroAudio,
    KokoroOnnxTtsBackend,
    validate_kokoro_artifact,
)


class FakeEngine:
    def __init__(self, *, fail_warm: bool = False) -> None:
        self.fail_warm = fail_warm
        self.cancelled = False
        self.texts: list[str] = []

    def warm(self) -> float:
        if self.fail_warm:
            raise RuntimeError("unavailable")
        return 12.5

    def synthesize(self, text: str, cancelled: threading.Event) -> KokoroAudio:
        self.texts.append(text)
        if cancelled.is_set():
            return KokoroAudio(b"")
        return KokoroAudio(b"\x01\x00" * 960)

    def cancel(self) -> None:
        self.cancelled = True


class FakeFallback:
    def __init__(self) -> None:
        self.units: list[SpeechUnit] = []
        self.cancelled: list[tuple[int, str]] = []
        self.closed = False

    def enqueue(self, unit: SpeechUnit) -> None:
        self.units.append(unit)

    def cancel(self, new_epoch: int, reason: str) -> None:
        self.cancelled.append((new_epoch, reason))

    @property
    def is_idle(self) -> bool:
        return True

    @property
    def last_error(self) -> BackendError | None:
        return None

    def close(self) -> None:
        self.closed = True


class FakePlayback:
    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.started = threading.Event()
        self.aborted = threading.Event()
        self.finished = False
        self.blocks: list[bytes] = []

    def start(self, sample_rate: int) -> None:
        if sample_rate != 24_000:
            raise AssertionError("unexpected sample rate")

    def write(self, pcm16: bytes) -> None:
        self.started.set()
        self.blocks.append(pcm16)
        if self.block:
            self.aborted.wait(1.0)

    def finish(self) -> None:
        self.finished = True

    def abort(self) -> None:
        self.aborted.set()


def unit(*, epoch: int = 1, locale: str = "en-US") -> SpeechUnit:
    return SpeechUnit("response", epoch, 1, locale, "All systems are operational.")


class KokoroConfigTests(unittest.TestCase):
    def test_locked_candidate_requires_digest_bound_artifacts(self) -> None:
        digest = "a" * 64
        config = voice_config_from_mapping(
            {
                "macos": {
                    "tts_backend": "kokoro_onnx",
                    "kokoro_onnx": {
                        "model": {"path": "/models/kokoro.int8.onnx", "size": 88_000_000, "sha256": digest},
                        "voices": {"path": "/models/voices.bin", "size": 1_000, "sha256": digest},
                        "language": "en-gb",
                        "speed": 0.96,
                        "voice_mix": {"bm_george": 0.7, "bm_fable": 0.3},
                        "warm_on_start": True,
                    },
                }
            }
        )
        self.assertEqual(config.macos.tts_backend, "kokoro_onnx")
        self.assertEqual(config.macos.kokoro_onnx.voice_mix["bm_george"], 0.7)

        with self.assertRaisesRegex(BackendError, "config.kokoro_candidate_locked"):
            voice_config_from_mapping(
                {
                    "macos": {
                        "tts_backend": "kokoro_onnx",
                        "kokoro_onnx": {
                            "model": {"path": "/m", "size": 1, "sha256": digest},
                            "voices": {"path": "/v", "size": 1, "sha256": digest},
                            "speed": 1.0,
                        },
                    }
                }
            )

    def test_artifact_integrity_rejects_changed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.onnx"
            path.write_bytes(b"model")
            artifact = ArtifactConfig(str(path), 5, hashlib.sha256(b"model").hexdigest())
            self.assertEqual(validate_kokoro_artifact(artifact, "model"), path)
            path.write_bytes(b"other")
            with self.assertRaisesRegex(BackendError, "tts.kokoro.artifact_changed"):
                validate_kokoro_artifact(artifact, "model")


class KokoroBackendTests(unittest.TestCase):
    def test_english_unit_plays_in_bounded_pcm_blocks_and_completes(self) -> None:
        engine = FakeEngine()
        fallback = FakeFallback()
        playback = FakePlayback()
        started: list[int] = []
        completed: list[int] = []
        backend = KokoroOnnxTtsBackend(
            engine=engine,
            fallback_backend=fallback,
            playback_factory=lambda: playback,
            on_started=lambda item: started.append(item.unit_index),
            on_completed=lambda item: completed.append(item.unit_index),
            warm_on_start=False,
        )
        self.addCleanup(backend.close)

        backend.enqueue(unit())

        self.assertTrue(backend.wait_idle())
        self.assertEqual(engine.texts, ["All systems are operational."])
        self.assertEqual((started, completed), ([1], [1]))
        self.assertTrue(playback.finished)
        self.assertTrue(all(len(block) <= 960 for block in playback.blocks))
        self.assertEqual(fallback.units, [])

    def test_ptt_cancel_aborts_playback_and_rejects_completion(self) -> None:
        engine = FakeEngine()
        fallback = FakeFallback()
        playback = FakePlayback(block=True)
        completed: list[int] = []
        backend = KokoroOnnxTtsBackend(
            engine=engine,
            fallback_backend=fallback,
            playback_factory=lambda: playback,
            on_completed=lambda item: completed.append(item.unit_index),
            warm_on_start=False,
        )
        self.addCleanup(backend.close)
        backend.enqueue(unit())
        self.assertTrue(playback.started.wait(1.0))

        started = time.perf_counter()
        backend.cancel(1, "ptt_down")

        self.assertTrue(playback.aborted.wait(0.2))
        self.assertLess(time.perf_counter() - started, 0.2)
        self.assertTrue(backend.wait_idle())
        self.assertTrue(engine.cancelled)
        self.assertEqual(completed, [])

    def test_warm_failure_degrades_to_system_fallback(self) -> None:
        fallback = FakeFallback()
        backend = KokoroOnnxTtsBackend(
            engine=FakeEngine(fail_warm=True),
            fallback_backend=fallback,
            playback_factory=FakePlayback,
            warm_on_start=True,
        )
        self.addCleanup(backend.close)
        backend.enqueue(unit())

        self.assertTrue(backend.wait_idle())
        self.assertTrue(backend.degraded_to_fallback)
        self.assertEqual(fallback.units, [unit()])

    def test_non_english_output_is_rejected_synchronously(self) -> None:
        backend = KokoroOnnxTtsBackend(
            engine=FakeEngine(),
            fallback_backend=FakeFallback(),
            playback_factory=FakePlayback,
            warm_on_start=False,
        )
        self.addCleanup(backend.close)
        with self.assertRaisesRegex(BackendError, "tts.unit_invalid"):
            backend.enqueue(unit(locale="el-GR"))


if __name__ == "__main__":
    unittest.main()
