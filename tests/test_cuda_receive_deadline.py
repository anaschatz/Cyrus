from __future__ import annotations

import multiprocessing
import os
import pickle
import struct
import threading
import unittest
from array import array
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import _bootstrap  # noqa: F401
import test_cuda_worker as fixture
from jarvis_voice.backends import AudioChunk
from jarvis_voice.cuda_worker import ProcessCudaWorkerSession
from jarvis_voice.errors import BackendError


def _partial_worker(connection, phase, ready):
    if phase != "start":
        connection.send({"type": "ready", "load_duration_seconds": 0.0})
        ready.set()
        connection.recv()
    os.write(connection.fileno(), struct.pack("!i", 10))
    ready.set()
    threading.Event().wait(10)


@unittest.skipIf(os.name == "nt", "POSIX raw pipe framing regression")
class CudaReceiveDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.CudaWorkerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_partial_frame_times_out_during_start_transcription_and_shutdown(self):
        for phase in ("start", "transcribe", "close"):
            with self.subTest(phase=phase):
                context = multiprocessing.get_context("spawn")
                parent, child = context.Pipe()
                ready = context.Event()
                process = context.Process(target=_partial_worker, args=(child, phase, ready))
                class StartedProcess:
                    def start(self):
                        process.start()
                        if not ready.wait(5):
                            raise RuntimeError("test child did not become ready")
                    def __getattr__(self, name):
                        return getattr(process, name)
                class Context:
                    def Pipe(self, duplex=True):
                        return parent, child
                    def Process(self, **kwargs):
                        return StartedProcess()
                config = replace(self.fixture.config(), worker_start_timeout_seconds=0.1, transcription_timeout_seconds=0.1)
                session = ProcessCudaWorkerSession(config, context=Context())
                def operation():
                    if phase == "start":
                        session.start()
                    elif phase == "transcribe":
                        session.transcribe(AudioChunk(array("f", [0.1]), 16000))
                    else:
                        session.start()
                        session.close()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(operation)
                    try:
                        if phase == "close":
                            future.result(timeout=2)
                        else:
                            code = "cuda.worker_start_timeout" if phase == "start" else "cuda.transcription_timeout"
                            with self.assertRaisesRegex(BackendError, code):
                                future.result(timeout=2)
                        self.assertTrue(process._closed)
                        self.assertTrue(parent.closed)
                        self.assertIsNone(session._process)
                    finally:
                        if not process._closed and process.is_alive():
                            process.kill()
                        try:
                            future.result(timeout=3)
                        except BackendError:
                            pass
                        if not process._closed:
                            process.join(3)
                            process.close()
                        parent.close()
                        child.close()
                self.assertFalse(any(item.name == "jarvis-cuda-receive" for item in threading.enumerate()))

    def test_fragmented_frame_completes_within_the_deadline(self):
        reader, writer = multiprocessing.Pipe(duplex=False)
        session = ProcessCudaWorkerSession(self.fixture.config())
        session._connection, session._process = reader, fixture.FakeProcess()
        frame = {"type": "ready", "load_duration_seconds": 0.0}
        payload = pickle.dumps(frame)
        os.write(writer.fileno(), struct.pack("!i", len(payload)) + payload[:2])
        timer = threading.Timer(0.02, os.write, args=(writer.fileno(), payload[2:]))
        timer.start()
        try:
            self.assertEqual(session._receive(1, "cuda.worker_start_timeout"), frame)
        finally:
            timer.join(2)
            writer.close()
            session.cancel()


if __name__ == "__main__":
    unittest.main()
