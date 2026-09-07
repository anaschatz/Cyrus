from __future__ import annotations

import multiprocessing
import os
import signal
import unittest
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import test_cuda_worker as fixture
from jarvis_voice.cuda_worker import CudaFasterWhisperBackend, ProcessCudaWorkerSession
from jarvis_voice.errors import BackendError


def _ignore_termination(connection):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    connection.send({"type": "ready", "load_duration_seconds": 0.0})
    while True:
        connection.recv()


class OwnedProcess(fixture.FakeProcess):
    def __init__(self):
        super().__init__()
        self.closed = False
        self.killed = False

    def kill(self):
        self.killed = True
        self._alive = False

    def close(self):
        if self._alive:
            raise ValueError("process still alive")
        self.closed = True


class CudaProcessOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.CudaWorkerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_creation_and_start_failures_close_every_owned_handle(self):
        for phase in ("construction", "start"):
            for error in (OSError, KeyboardInterrupt):
                with self.subTest(phase=phase, error=error):
                    process = OwnedProcess()
                    process._alive = False
                    parent = fixture.PollConnection([], process)
                    child = fixture.MemoryConnection([])
                    context = fixture.FakeContext(parent, process)
                    context.Pipe = Mock(return_value=(parent, child))
                    target = context if phase == "construction" else process
                    method = "Process" if phase == "construction" else "start"
                    session = ProcessCudaWorkerSession(self.fixture.config(), context=context)
                    with patch.object(target, method, side_effect=error):
                        with self.assertRaises(error):
                            session.start()
                    self.assertTrue(child.closed)
                    self.assertTrue(parent.closed)
                    self.assertIsNone(session._connection)
                    self.assertIsNone(session._process)
                    if phase == "start":
                        self.assertTrue(process.closed)

    def test_restart_closes_dead_session_before_replacing_it(self):
        old = OwnedProcess()
        old._alive = False
        old_pipe = fixture.PollConnection([], old)
        process = OwnedProcess()
        parent = fixture.PollConnection([{"type": "ready", "load_duration_seconds": 0.1}], process)
        session = ProcessCudaWorkerSession(self.fixture.config(), context=fixture.FakeContext(parent, process))
        session._process, session._connection = old, old_pipe
        self.addCleanup(session.cancel)
        session.start()
        self.assertTrue(old_pipe.closed)
        self.assertTrue(old.closed)

    def test_shutdown_escalates_and_retains_a_process_that_cannot_exit(self):
        for operation in ("cancel", "close"):
            for exits in (True, False):
                with self.subTest(operation=operation, exits=exits):
                    process = OwnedProcess()
                    process.terminate = Mock()
                    if not exits:
                        process.kill = Mock()
                    parent = fixture.PollConnection([{"type": "shutdown.complete"}], process)
                    session = ProcessCudaWorkerSession(self.fixture.config())
                    session._process, session._connection = process, parent
                    if exits:
                        getattr(session, operation)()
                        self.assertTrue(process.killed)
                        self.assertTrue(process.closed)
                        self.assertTrue(parent.closed)
                        self.assertFalse(session.alive)
                    else:
                        with self.assertRaisesRegex(BackendError, "cuda.worker_shutdown_timeout"):
                            getattr(session, operation)()
                        self.assertIs(session._process, process)
                        self.assertTrue(session.alive)
                        self.assertFalse(process.closed)
                        process._alive = False
                        session.cancel()
                        self.assertTrue(process.closed)

    def test_pipe_close_failure_still_closes_dead_process(self):
        process = OwnedProcess()
        process._alive = False
        parent = fixture.PollConnection([], process)
        parent.close = Mock(side_effect=OSError("close failed"))
        session = ProcessCudaWorkerSession(self.fixture.config())
        session._process, session._connection = process, parent
        with self.assertRaises(OSError):
            session.cancel()
        self.assertTrue(process.closed)
        self.assertIsNone(session._process)
        self.assertIsNone(session._connection)

    def test_backend_retains_failed_cleanup_for_retry_and_closes_other_session(self):
        for operation in ("close", "cancel_current"):
            with self.subTest(operation=operation):
                backend = CudaFasterWhisperBackend(self.fixture.config(), inventory=fixture.StaticInventory(),
                    residency="warm", minimum_free_vram_mb=0, reserve_for_other_workloads_mb=0, warm_restart_limit=0)
                active, warm = fixture.FakeSession([]), fixture.FakeSession([])
                backend._active_session = active
                backend._warm_session = warm if operation == "close" else active
                target = "close" if operation == "close" else "cancel"
                with patch.object(active, target, side_effect=BackendError("cuda.worker_shutdown_timeout", "still running", subsystem="cuda")):
                    with self.assertRaises(BackendError):
                        getattr(backend, operation)()
                self.assertIs(backend._active_session, active)
                if operation == "close":
                    self.assertTrue(warm.closed)
                    self.assertIsNone(backend._warm_session)
                else:
                    self.assertIs(backend._warm_session, active)
                getattr(backend, operation)()
                self.assertIsNone(backend._active_session)
                self.assertIsNone(backend._warm_session)

    @unittest.skipIf(os.name == "nt", "POSIX SIGTERM-ignore regression")
    def test_real_owned_child_ignoring_terminate_is_killed_and_reaped(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_ignore_termination, args=(child,))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(5), "test child did not become ready")
            self.assertEqual(parent.recv()["type"], "ready")
            session = ProcessCudaWorkerSession(self.fixture.config())
            session._process, session._connection = process, parent
            original_join = process.join
            with patch.object(process, "join", side_effect=lambda timeout: original_join(min(timeout, 0.5))):
                session.cancel()
            self.assertTrue(process._closed)
            self.assertIsNone(session._process)
            self.assertTrue(parent.closed)
        finally:
            if not process._closed:
                if process.is_alive():
                    process.kill()
                process.join(5)
                process.close()
            parent.close()


if __name__ == "__main__":
    unittest.main()
