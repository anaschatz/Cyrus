from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos import piper_f1_qualification as piper_qualification
from jarvis_voice.macos.compact_greek_tts_pilot import (
    MAX_EXPORTED_MODEL_BYTES,
    MAX_RUNTIME_RSS_BYTES,
    PIPER_TARGET_SAMPLE_RATE,
)
from jarvis_voice.macos.piper_f1_qualification import (
    CONFIG_NAME,
    FIXTURES,
    MODEL_NAME,
    PcmChunk,
    inspect_piper_f1_candidate,
    piper_f1_qualification_preflight,
    run_piper_f1_qualification,
)


def _config() -> dict[str, object]:
    return {
        "dataset": "stoma_f1",
        "audio": {"sample_rate": PIPER_TARGET_SAMPLE_RATE, "quality": "medium"},
        "espeak": {"voice": "el"},
        "language": {"code": "el"},
        "phoneme_type": "espeak",
        "phoneme_id_map": {"_": [0], "^": [1], " ": [3]},
        "num_speakers": 1,
        "num_symbols": 256,
    }


def _candidate(root: Path) -> None:
    (root / MODEL_NAME).write_bytes(b"fake-onnx")
    (root / CONFIG_NAME).write_text(json.dumps(_config()), encoding="utf-8")


class _FakeSession:
    def __init__(self, model: Path, config: Path) -> None:
        self.model = model
        self.config = config
        self.aborted = False
        self.closed = False
        self.calls: list[str] = []

    @property
    def peak_rss_bytes(self) -> int:
        return MAX_RUNTIME_RSS_BYTES // 2

    def stream(self, text: str):
        self.calls.append(text)
        yield PcmChunk(b"\x00\x00" * PIPER_TARGET_SAMPLE_RATE, PIPER_TARGET_SAMPLE_RATE)

    def abort(self) -> float:
        self.aborted = True
        return 2.0

    def close(self) -> None:
        self.closed = True


class _FakeSink:
    def __init__(self) -> None:
        self.opened = False
        self.aborted = False

    def open(self, chunk: PcmChunk) -> None:
        self.opened = chunk.sample_rate == PIPER_TARGET_SAMPLE_RATE

    def write(self, pcm16: bytes) -> None:
        if not self.opened or not pcm16:
            raise AssertionError("sink was not opened")

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        pass


class ProcessPiperSessionTests(unittest.TestCase):
    def test_failed_startup_closes_both_pipe_ends_and_process(self) -> None:
        for failure in ("constructor", "start", "started", "timeout", "error", "invalid", "interrupt"):
            with self.subTest(failure=failure):
                parent, child = mock.Mock(), mock.Mock()
                parent.poll.return_value = failure != "timeout"
                parent.recv.return_value = {"type": "error" if failure == "error" else "done"}
                process = mock.Mock(pid=None)
                process.is_alive.return_value = False
                error = KeyboardInterrupt() if failure == "interrupt" else OSError("private startup failure")

                def start():
                    if failure == "start":
                        raise error
                    process.pid = 123
                    process.is_alive.return_value = True
                    if failure == "started":
                        raise error

                process.start.side_effect = start
                process.terminate.side_effect = lambda: setattr(process.is_alive, "return_value", False)
                if failure == "interrupt":
                    parent.poll.side_effect = error
                context = mock.Mock()
                context.Pipe.return_value = (parent, child)
                context.Process.return_value = process
                if failure == "constructor":
                    context.Process.side_effect = error
                expected = KeyboardInterrupt if failure == "interrupt" else OSError if failure in {"constructor", "start", "started"} else BackendError
                with mock.patch.object(piper_qualification.multiprocessing, "get_context", return_value=context):
                    with self.assertRaises(expected):
                        piper_qualification.ProcessPiperSession(Path("unused.onnx"), Path("unused.json"))
                child.close.assert_called_once_with()
                parent.close.assert_called_once_with()
                self.assertFalse(process.is_alive())
                if failure != "constructor":
                    process.close.assert_called_once_with()

    def test_close_kills_stubborn_worker_and_retries_failed_stop(self) -> None:
        for failure in (None, "kill_error", "still_alive", "send_error"):
            with self.subTest(failure=failure):
                parent, child = mock.Mock(), mock.Mock()
                parent.poll.return_value = True
                parent.recv.return_value = {"type": "loaded", "rss_bytes": 17}
                process = mock.Mock(pid=123)
                process.is_alive.return_value = True

                def kill():
                    if process.kill.call_count == 1:
                        if failure == "kill_error":
                            raise OSError("private stop failure")
                        if failure == "still_alive":
                            return
                    process.is_alive.return_value = False

                process.kill.side_effect = kill
                context = mock.Mock()
                context.Pipe.return_value = (parent, child)
                context.Process.return_value = process
                if failure == "send_error":
                    parent.send.side_effect = BrokenPipeError()
                with mock.patch.object(piper_qualification.multiprocessing, "get_context", return_value=context):
                    session = piper_qualification.ProcessPiperSession(Path("unused.onnx"), Path("unused.json"))
                if failure in {"kill_error", "still_alive"}:
                    with self.assertRaisesRegex(BackendError, "tts.piper_f1.worker_stop_failed"):
                        session.close()
                    parent.close.assert_called_once_with()
                    process.close.assert_not_called()
                    with self.assertRaisesRegex(BackendError, "tts.piper_f1.worker_stopped"):
                        next(session.stream("unused"))
                session.close()
                session.close()
                self.assertFalse(process.is_alive())
                self.assertEqual(process.kill.call_count, 2 if failure in {"kill_error", "still_alive"} else 1)
                process.close.assert_called_once_with()
                child.close.assert_called_once_with()
                self.assertEqual(session.peak_rss_bytes, 17)


class PiperF1QualificationTests(unittest.TestCase):
    def test_cancel_gate_rejects_failed_or_unstopped_writer(self) -> None:
        for behavior in ("write_failed", "writer_stuck", "abort_releases_writer"):
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                candidate = root / "candidate"
                candidate.mkdir()
                _candidate(candidate)
                release = threading.Event()

                class Sink(_FakeSink):
                    writer = None

                    def write(self, pcm16):
                        Sink.writer = threading.current_thread()
                        if behavior == "write_failed":
                            raise OSError("private playback failure")
                        release.wait(timeout=5)
                        if behavior == "abort_releases_writer":
                            raise OSError("expected PortAudio abort")

                    def abort(self):
                        super().abort()
                        if behavior == "abort_releases_writer":
                            release.set()

                    def close(self):
                        release.set()

                session = _FakeSession(candidate / MODEL_NAME, candidate / CONFIG_NAME)
                try:
                    report = run_piper_f1_qualification(
                        candidate, root / "output", session_factory=lambda *args: session,
                        playback_sink_factory=Sink,
                    )
                    self.assertEqual(report["gates"]["cancellation"], behavior == "abort_releases_writer")
                    self.assertEqual(report["automated_pass"], behavior == "abort_releases_writer")
                    self.assertFalse(report["ready_for_promotion"])
                    self.assertTrue(session.closed)
                finally:
                    release.set()
                    if Sink.writer is not None:
                        Sink.writer.join(timeout=1)
                        self.assertFalse(Sink.writer.is_alive())

    def test_missing_candidate_is_a_read_only_not_ready_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = list(root.iterdir())
            report = inspect_piper_f1_candidate(root)
            preflight = piper_f1_qualification_preflight(root, runtime_probe=lambda: False)
            after = list(root.iterdir())
        self.assertFalse(report["ready"])
        self.assertEqual(report["code"], "tts.piper_f1.model_missing")
        self.assertFalse(preflight["installation_performed"])
        self.assertFalse(preflight["network_used"])
        self.assertFalse(preflight["production_backend_changed"])
        self.assertEqual(before, after)

    def test_candidate_validation_pins_names_hashes_and_greek_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _candidate(root)
            report = inspect_piper_f1_candidate(root)
        self.assertTrue(report["ready"])
        self.assertEqual(report["model"]["name"], MODEL_NAME)
        self.assertEqual(len(report["model"]["sha256"]), 64)
        self.assertEqual(len(report["config"]["sha256"]), 64)

    def test_candidate_rejects_incompatible_config_and_oversize_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _candidate(root)
            incompatible = _config()
            incompatible["language"] = {"code": "en"}
            (root / CONFIG_NAME).write_text(json.dumps(incompatible), encoding="utf-8")
            self.assertEqual(
                inspect_piper_f1_candidate(root)["code"],
                "tts.piper_f1.config_incompatible",
            )
            (root / CONFIG_NAME).write_text(json.dumps(_config()), encoding="utf-8")
            with (root / MODEL_NAME).open("r+b") as stream:
                stream.truncate(MAX_EXPORTED_MODEL_BYTES + 1)
            self.assertEqual(
                inspect_piper_f1_candidate(root)["code"],
                "tts.piper_f1.model_size_invalid",
            )

    def test_fake_runtime_exercises_all_gates_without_piper_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "candidate"
            candidate.mkdir()
            _candidate(candidate)
            output = root / "output"
            sessions: list[_FakeSession] = []

            def factory(model: Path, config: Path) -> _FakeSession:
                session = _FakeSession(model, config)
                sessions.append(session)
                return session

            report = run_piper_f1_qualification(
                candidate,
                output,
                session_factory=factory,
                playback_sink_factory=_FakeSink,
            )

            wavs = sorted(output.glob("*.wav"))
        self.assertEqual(len(wavs), len(FIXTURES))
        self.assertTrue(report["automated_pass"])
        self.assertFalse(report["ready_for_promotion"])
        self.assertEqual(report["code"], "tts.piper_f1.human_approval_required")
        self.assertTrue(report["metrics"]["cancellation"]["measured"])
        self.assertTrue(sessions[0].aborted)
        self.assertTrue(sessions[0].closed)
        self.assertFalse(report["installation_performed"])
        self.assertFalse(report["production_backend_changed"])

    def test_no_playback_measurement_cannot_pass_cancellation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "candidate"
            candidate.mkdir()
            _candidate(candidate)
            report = run_piper_f1_qualification(
                candidate,
                root / "output",
                session_factory=_FakeSession,
            )
        self.assertFalse(report["automated_pass"])
        self.assertFalse(report["gates"]["cancellation"])


if __name__ == "__main__":
    unittest.main()
