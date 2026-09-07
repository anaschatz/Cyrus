from __future__ import annotations

import unittest
from array import array
from unittest.mock import patch

import _bootstrap  # noqa: F401
import test_cuda_worker as fixture
from jarvis_voice.backends import AudioChunk
from jarvis_voice.cuda_worker import ProcessCudaWorkerSession, run_cuda_worker
from jarvis_voice.errors import BackendError


class CudaFrameTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.CudaWorkerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.audio = AudioChunk(array("f", [0.1]), 16000)

    def session(self, frames):
        process = fixture.FakeProcess()
        parent = fixture.PollConnection(frames, process)
        session = ProcessCudaWorkerSession(self.fixture.config(), context=fixture.FakeContext(parent, process))
        self.addCleanup(session.cancel)
        return session, parent

    def test_ready_frame_is_exact_with_a_finite_nonnegative_duration(self):
        frames = [{"type": "ready", "load_duration_seconds": value} for value in
                  (True, float("nan"), float("inf"), -0.1, 10 ** 400)]
        frames.append({"type": "ready", "load_duration_seconds": 0.1, "unexpected": 1})
        for frame in frames:
            with self.subTest(frame=repr(frame)[:100]):
                session, parent = self.session([frame])
                with self.assertRaisesRegex(BackendError, "cuda.worker_protocol_error"):
                    session.start()
                self.assertTrue(parent.closed)
                self.assertEqual(session.metrics, [])

    def test_result_metrics_type_and_language_are_validated_before_publication(self):
        result = {"type": "result", "request_id": "r", "text": "Γεια σου.", "language_code": "el",
                  "language_score": 0.9, "asr_score": 0.8, "raw_avg_logprob": -0.2,
                  "inference_duration_seconds": 0.1}
        mutations = [(name, value) for name in ("language_score", "asr_score", "raw_avg_logprob", "inference_duration_seconds")
                     for value in (float("nan"), float("inf"), float("-inf"), True, 10 ** 400)]
        mutations += [("type", "ready"), ("language_code", None), ("language_code", ["el"]),
                      ("language_code", ""), ("language_score", -0.1), ("language_score", 1.1),
                      ("asr_score", -0.1), ("asr_score", 1.1), ("raw_avg_logprob", 0.1),
                      ("inference_duration_seconds", -0.1)]
        for name, value in mutations:
            with self.subTest(name=name, value=repr(value)[:40]):
                session, _ = self.session([{"type": "ready", "load_duration_seconds": 0.0}, {**result, name: value}])
                with patch("jarvis_voice.cuda_worker.uuid.uuid4", return_value="r"):
                    with self.assertRaisesRegex(BackendError, "cuda.worker_protocol_error"):
                        session.transcribe(self.audio)
                self.assertNotIn("cuda_inference", [metric["event"] for metric in session.metrics])
        session, _ = self.session([{"type": "ready", "load_duration_seconds": 0}, result])
        with patch("jarvis_voice.cuda_worker.uuid.uuid4", return_value="r"):
            self.assertEqual(session.transcribe(self.audio), fixture.RESULT)

    def test_error_frames_bind_request_identity_and_validate_retryability(self):
        valid = {"type": "error", "code": "cuda.oom", "retryable": True, "request_id": "r"}
        frames = [{**valid, "request_id": "old"}, {**valid, "retryable": "false"},
                  {**valid, "retryable": 0}, {**valid, "unexpected": 1}, {**valid, "code": ""}]
        frames.append({key: value for key, value in valid.items() if key != "request_id"})
        for frame in frames:
            with self.subTest(frame=frame):
                session, _ = self.session([{"type": "ready", "load_duration_seconds": 0}, frame])
                with patch("jarvis_voice.cuda_worker.uuid.uuid4", return_value="r"):
                    with self.assertRaisesRegex(BackendError, "cuda.worker_protocol_error") as caught:
                        session.transcribe(self.audio)
                self.assertFalse(caught.exception.retryable)
        session, _ = self.session([{"type": "ready", "load_duration_seconds": 0}, valid])
        with patch("jarvis_voice.cuda_worker.uuid.uuid4", return_value="r"):
            with self.assertRaisesRegex(BackendError, "cuda.oom") as caught:
                session.transcribe(self.audio)
        self.assertTrue(caught.exception.retryable)

    def test_invalid_audio_error_keeps_its_request_identity(self):
        connection = fixture.MemoryConnection([
            {"type": "transcribe", "request_id": "r", "sample_rate": 16000, "audio": b"x"},
            {"type": "shutdown"}])
        run_cuda_worker(connection, self.fixture.config(), model_factory=lambda *a, **k: fixture.FakeModel())
        self.assertEqual(connection.sent[1].get("request_id"), "r")
        session, _ = self.session(connection.sent[:2])
        with patch("jarvis_voice.cuda_worker.uuid.uuid4", return_value="r"):
            with self.assertRaisesRegex(BackendError, "cuda.worker_invalid_audio"):
                session.transcribe(self.audio)

    def test_child_rejects_unhashable_message_types_with_a_protocol_error(self):
        for kind in ([], {}):
            with self.subTest(kind=kind):
                connection = fixture.MemoryConnection([{"type": kind}])
                run_cuda_worker(connection, self.fixture.config(), model_factory=lambda *a, **k: fixture.FakeModel())
                self.assertEqual(connection.sent[-1], {"type": "error", "code": "cuda.worker_protocol_error", "retryable": False})
                self.assertTrue(connection.closed)

    def test_child_closes_channel_when_ready_or_load_error_cannot_be_sent(self):
        for load_fails in (False, True):
            with self.subTest(load_fails=load_fails):
                connection = fixture.MemoryConnection([])
                def model(*args, **kwargs):
                    if load_fails:
                        raise RuntimeError("model load failed")
                    return fixture.FakeModel()
                with patch.object(connection, "send", side_effect=BrokenPipeError):
                    run_cuda_worker(connection, self.fixture.config(), model_factory=model)
                self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
