from __future__ import annotations

import threading
import unittest
from array import array
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import _bootstrap  # noqa: F401
from jarvis_voice.asr_factory import FallbackAsrBackend
from jarvis_voice.backends import AudioChunk
from jarvis_voice.errors import BackendError
from jarvis_voice.transcription import AsyncTranscriptionCoordinator
from test_asr_factory import RESULT


class FallbackCancellationTests(unittest.TestCase):
    def test_cancel_and_close_do_not_fallback_for_the_old_turn(self):
        for operation in ("cancel_current", "close"):
            for fails in (False, True):
                with self.subTest(operation=operation, fails=fails):
                    started, released = threading.Event(), threading.Event()
                    cuda, cpu = Mock(), Mock()
                    def transcribe(audio):
                        started.set()
                        if not released.wait(2):
                            raise RuntimeError("test CUDA did not release")
                        raise BackendError("cuda.worker_crashed", "cancelled", subsystem="cuda")
                    def cancel():
                        released.set()
                        if fails:
                            raise OSError("test cleanup failed")
                    cuda.transcribe.side_effect = transcribe
                    getattr(cuda, operation).side_effect = cancel
                    backend = FallbackAsrBackend(cuda, cpu, enabled=True,
                                                fallbackable_codes=frozenset({"cuda.worker_crashed"}))
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(backend.transcribe, object())
                        try:
                            self.assertTrue(started.wait(2))
                            if fails:
                                with self.assertRaises(OSError):
                                    getattr(backend, operation)()
                            else:
                                getattr(backend, operation)()
                            with self.assertRaisesRegex(BackendError, "cuda.worker_crashed"):
                                future.result(timeout=2)
                        finally:
                            released.set()
                    cpu.transcribe.assert_not_called()
                    getattr(cpu, operation).assert_called_once_with()
                    self.assertEqual(backend.fallbacks, 0)
                    self.assertEqual(backend.metrics, [])

    def test_retired_async_turn_skips_cpu_and_the_next_turn_can_fallback(self):
        started, released = threading.Event(), threading.Event()
        cuda, cpu = Mock(), Mock()
        cpu.transcribe.return_value = RESULT
        def transcribe(audio):
            started.set()
            if not released.wait(2):
                raise RuntimeError("test CUDA did not release")
            raise BackendError("cuda.worker_crashed", "cancelled", subsystem="cuda")
        cuda.transcribe.side_effect = transcribe
        cuda.cancel_current.side_effect = released.set
        backend = FallbackAsrBackend(cuda, cpu, enabled=True,
                                    fallbackable_codes=frozenset({"cuda.worker_crashed"}))
        coordinator = AsyncTranscriptionCoordinator(backend)
        audio = AudioChunk(array("f", [0.1]), 16000)
        try:
            coordinator.submit("old", audio)
            old = coordinator._future
            self.assertTrue(started.wait(2))
            coordinator.cancel_current()
            with self.assertRaises(BackendError):
                old.result(timeout=2)
            self.assertIsNone(coordinator.poll())
            cpu.transcribe.assert_not_called()
            coordinator.submit("new", audio)
            self.assertEqual(coordinator._future.result(timeout=2), RESULT)
            outcome = coordinator.poll()
            self.assertEqual((outcome.turn_id, outcome.result), ("new", RESULT))
            cpu.transcribe.assert_called_once_with(audio)
            self.assertEqual(backend.fallbacks, 1)
        finally:
            released.set()
            coordinator.close()
            coordinator.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
