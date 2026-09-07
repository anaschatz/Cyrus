from __future__ import annotations

import unittest
from array import array
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import test_cuda_worker as fixture
from jarvis_voice.asr_factory import FallbackAsrBackend
from jarvis_voice.backends import AudioChunk
from jarvis_voice.cuda_worker import FALLBACKABLE_CUDA_ERRORS, CudaFasterWhisperBackend, ProcessCudaWorkerSession
from jarvis_voice.errors import BackendError


class CudaIpcFailureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.CudaWorkerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.audio = AudioChunk(array("f", [0.1]), 16000)

    def session(self):
        process = fixture.FakeProcess()
        parent = fixture.PollConnection([{"type": "ready", "load_duration_seconds": 0.1}], process)
        session = ProcessCudaWorkerSession(self.fixture.config(), context=fixture.FakeContext(parent, process))
        self.addCleanup(session.cancel)
        return session, parent, process

    def test_broken_channel_is_typed_and_releases_session(self):
        for phase, method in (("start", "poll"), ("start", "recv"),
                              ("transcribe", "send"), ("transcribe", "poll"), ("transcribe", "recv")):
            for error in (EOFError, OSError, BrokenPipeError):
                with self.subTest(phase=phase, method=method, error=error):
                    session, parent, process = self.session()
                    if phase == "transcribe":
                        session.start()
                        parent.frames.append({})
                    with patch.object(parent, method, side_effect=error("private channel detail")):
                        with self.assertRaisesRegex(BackendError, "cuda.worker_crashed") as caught:
                            session.start() if phase == "start" else session.transcribe(self.audio)
                    self.assertTrue(caught.exception.retryable)
                    self.assertNotIn("private channel detail", str(caught.exception))
                    self.assertTrue(parent.closed)
                    self.assertTrue(process.terminated)
                    self.assertFalse(session.alive)

    def test_channel_failure_reaches_same_audio_cpu_fallback(self):
        for residency in ("warm", "turn_scoped"):
            for method in ("send", "recv"):
                with self.subTest(residency=residency, method=method):
                    session, parent, process = self.session()
                    session.start()
                    parent.frames.append({})
                    cuda = CudaFasterWhisperBackend(self.fixture.config(), inventory=fixture.StaticInventory(),
                        residency=residency, minimum_free_vram_mb=0, reserve_for_other_workloads_mb=0,
                        warm_restart_limit=1, session_factory=lambda config: session)
                    self.addCleanup(cuda.close)
                    cpu = Mock()
                    cpu.transcribe.return_value = fixture.RESULT
                    backend = FallbackAsrBackend(cuda, cpu, enabled=True, fallbackable_codes=FALLBACKABLE_CUDA_ERRORS)
                    with patch.object(parent, method, side_effect=BrokenPipeError):
                        self.assertEqual(backend.transcribe(self.audio), fixture.RESULT)
                    cpu.transcribe.assert_called_once_with(self.audio)
                    self.assertEqual(backend.fallbacks, 1)
                    self.assertTrue(parent.closed)
                    self.assertTrue(process.terminated)
                    self.assertIsNone(cuda._warm_session)
                    self.assertIsNone(cuda._active_session)


if __name__ == "__main__":
    unittest.main()
