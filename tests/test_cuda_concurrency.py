from __future__ import annotations

import threading
import unittest
from array import array
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import test_cuda_worker as fixture
from jarvis_voice.backends import AudioChunk
from jarvis_voice.cuda_worker import CudaFasterWhisperBackend, ProcessCudaWorkerSession
from jarvis_voice.errors import BackendError


class CudaConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.CudaWorkerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.audio = AudioChunk(array("f", [0.1]), 16000)

    def test_cancel_during_poll_is_a_typed_worker_error(self):
        for readable in (True, False):
            with self.subTest(readable=readable):
                process = fixture.FakeProcess()
                parent = fixture.PollConnection([{"type": "ready", "load_duration_seconds": 0.1}], process)
                session = ProcessCudaWorkerSession(self.fixture.config(), context=fixture.FakeContext(parent, process))
                session.start()
                def poll(timeout):
                    session.cancel()
                    return readable
                with patch.object(parent, "poll", side_effect=poll), patch.object(parent, "recv", side_effect=OSError("closed")):
                    with self.assertRaisesRegex(BackendError, "cuda.worker_crashed"):
                        session.transcribe(self.audio)
                self.assertFalse(session.alive)
                self.assertTrue(parent.closed)

    def test_cancelled_or_closed_session_never_spawns_again(self):
        for operation in ("cancel", "close"):
            with self.subTest(operation=operation):
                context = Mock()
                context.Pipe.side_effect = AssertionError("cancelled session must not create a pipe")
                session = ProcessCudaWorkerSession(self.fixture.config(), context=context)
                getattr(session, operation)()
                with self.assertRaisesRegex(BackendError, "cuda.worker_cancelled"):
                    session.transcribe(self.audio)
                context.Pipe.assert_not_called()

    def test_concurrent_cancellation_serializes_process_cleanup(self):
        entered, release, duplicate = threading.Event(), threading.Event(), threading.Event()
        second_started = threading.Event()
        process = fixture.FakeProcess()
        original_terminate = process.terminate
        def terminate():
            if entered.is_set():
                duplicate.set()
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test cleanup not released")
            original_terminate()
        process.terminate = terminate
        session = ProcessCudaWorkerSession(self.fixture.config())
        session._process = process
        session._connection = fixture.PollConnection([], process)
        def second_cancel():
            second_started.set()
            session.cancel()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(session.cancel)
            try:
                self.assertTrue(entered.wait(2))
                second = executor.submit(second_cancel)
                self.assertTrue(second_started.wait(2))
                self.assertFalse(duplicate.wait(0.1), "two threads entered process termination")
            finally:
                release.set()
            first.result(timeout=2)
            second.result(timeout=2)
        self.assertFalse(session.alive)

    def test_cancel_and_close_cover_admission_and_model_loading(self):
        for residency in ("warm", "turn_scoped"):
            for phase in ("admission", "load"):
                for operation in ("cancel_current", "close"):
                    with self.subTest(residency=residency, phase=phase, operation=operation):
                        started, released = threading.Event(), threading.Event()
                        session = fixture.FakeSession([])
                        def wait():
                            started.set()
                            if not released.wait(2):
                                raise RuntimeError("test loading not released")
                        class Inventory(fixture.StaticInventory):
                            def devices(self):
                                if phase == "admission":
                                    wait()
                                return super().devices()
                        def start():
                            wait()
                            if session.cancelled or session.closed:
                                raise BackendError("cuda.worker_crashed", "cancelled", subsystem="cuda")
                            session.started = True
                        if phase == "load":
                            session.start = Mock(side_effect=start)
                        else:
                            session.start = Mock()
                        old_cancel, old_close = session.cancel, session.close
                        session.cancel = lambda: (old_cancel(), released.set())
                        session.close = lambda: (old_close(), released.set())
                        backend = CudaFasterWhisperBackend(self.fixture.config(), inventory=Inventory(),
                            residency=residency, minimum_free_vram_mb=0, reserve_for_other_workloads_mb=0,
                            warm_restart_limit=0, session_factory=lambda config: session)
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            future = executor.submit(backend.transcribe, self.audio)
                            try:
                                self.assertTrue(started.wait(2))
                                cleanup = executor.submit(getattr(backend, operation))
                                cleanup.result(timeout=0.5)
                                released.set()
                                with self.assertRaises(BackendError):
                                    future.result(timeout=2)
                                self.assertEqual(session.calls, 0)
                                if phase == "admission":
                                    session.start.assert_not_called()
                                self.assertIsNone(backend._active_session)
                                self.assertIsNone(backend._warm_session)
                            finally:
                                released.set()
                        if operation == "cancel_current":
                            next_session = fixture.FakeSession([])
                            backend.session_factory = lambda config: next_session
                            self.assertEqual(backend.transcribe(self.audio), fixture.RESULT)
                            self.assertEqual(next_session.calls, 1)
                        backend.close()


if __name__ == "__main__":
    unittest.main()
