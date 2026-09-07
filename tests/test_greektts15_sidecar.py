from __future__ import annotations

import hashlib
import queue
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.config import ArtifactConfig, GreekTts15Config, voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.greektts15_sidecar import GreekTtsF1SidecarBackend, validate_worker_python


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fake_greektts15_sidecar_worker.py"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _config() -> GreekTts15Config:
    python = Path(sys.executable)
    return GreekTts15Config(
        model_root="/private/model",
        approval_manifest_path="/private/approved.json",
        approval_manifest_sha256="a" * 64,
        repository_root="/private/repository",
        warm_on_start=True,
        worker_python=ArtifactConfig(str(python), python.stat().st_size, _sha256(python)),
    )


class GreekTtsF1SidecarTests(unittest.TestCase):
    def test_real_subprocess_forwards_callbacks_and_closes_cleanly(self) -> None:
        callbacks: list[str] = []
        completed = threading.Event()
        backend = GreekTtsF1SidecarBackend(
            config=_config(),
            worker_script=FIXTURE,
            on_started=lambda _unit: callbacks.append("started"),
            on_completed=lambda _unit: (callbacks.append("completed"), completed.set()),
        )
        try:
            self.assertEqual(backend.warm_load_ms, 12.5)
            backend.enqueue(SpeechUnit("response-1", 1, 1, "el-GR", "δοκιμή"))
            self.assertTrue(completed.wait(timeout=2.0))
            self.assertEqual(callbacks, ["started", "completed"])
            self.assertTrue(backend.is_idle)
            self.assertIsNone(backend.last_error)
        finally:
            backend.close()

    def test_cancelled_unit_rejects_late_sidecar_callbacks(self) -> None:
        callbacks: list[str] = []
        backend = GreekTtsF1SidecarBackend(
            config=_config(),
            worker_script=FIXTURE,
            on_started=lambda _unit: callbacks.append("started"),
            on_completed=lambda _unit: callbacks.append("completed"),
        )
        try:
            backend.enqueue(SpeechUnit("response-2", 1, 1, "el-GR", "defer"))
            backend.cancel(2, "ptt_down")
            time.sleep(0.1)
            self.assertEqual(callbacks, [])
            self.assertTrue(backend.is_idle)
        finally:
            backend.close()

    def test_worker_error_is_content_free_and_clears_pending(self) -> None:
        backend = GreekTtsF1SidecarBackend(config=_config(), worker_script=FIXTURE)
        try:
            backend.enqueue(SpeechUnit("response-3", 1, 1, "el-GR", "fail"))
            deadline = time.monotonic() + 2.0
            while backend.last_error is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(backend.last_error)
            self.assertEqual(backend.last_error.code, "tts.greektts15.fake_failure")
            self.assertTrue(backend.is_idle)
        finally:
            backend.close()

    @staticmethod
    def _reap_owned_worker(backend) -> None:
        # Also clean up broken baseline constructors so regressions cannot leak.
        backend._stop.set()
        try:
            backend._commands.put_nowait(None)
        except queue.Full:
            pass
        process = backend._process
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=2.0)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
        for name in ("_writer", "_reader"):
            thread = getattr(backend, name, None)
            if thread is not None and thread.ident is not None:
                thread.join(timeout=2.0)

    def test_partial_startup_releases_worker_and_pipes(self) -> None:
        original_start = threading.Thread.start
        for phase in ("writer", "reader", "missing_stdout"):
            with self.subTest(phase=phase):
                backend = GreekTtsF1SidecarBackend.__new__(GreekTtsF1SidecarBackend)
                def start(thread):
                    if thread.name == f"jarvis-greektts-sidecar-{phase}":
                        raise OSError("thread start failed")
                    original_start(thread)
                def factory(*args, **kwargs):
                    process = subprocess.Popen(*args, **kwargs)
                    if phase == "missing_stdout":
                        process.stdout.close()
                        process.stdout = None
                    return process
                try:
                    with mock.patch.object(threading.Thread, "start", new=start):
                        with self.assertRaises((OSError, BackendError)):
                            backend.__init__(config=_config(), worker_script=FIXTURE,
                                             process_factory=factory, shutdown_timeout_seconds=0.01)
                    self.assertIsNotNone(backend._process.poll())
                    self.assertTrue(backend._process.stdin.closed)
                    if backend._process.stdout is not None:
                        self.assertTrue(backend._process.stdout.closed)
                    for name in ("_writer", "_reader"):
                        thread = getattr(backend, name, None)
                        self.assertTrue(thread is None or not thread.is_alive())
                finally:
                    self._reap_owned_worker(backend)

    def test_close_rejects_new_units_and_clears_pending(self) -> None:
        for pending in (False, True):
            with self.subTest(pending=pending):
                backend = GreekTtsF1SidecarBackend(config=_config(), worker_script=FIXTURE)
                try:
                    if pending:
                        backend.enqueue(SpeechUnit("deferred", 1, 1, "el-GR", "defer"))
                    backend.close()
                    self.assertTrue(backend.is_idle)
                    with self.assertRaises(BackendError) as raised:
                        backend.enqueue(SpeechUnit("after-close", 2, 1, "el-GR", "δοκιμή"))
                    self.assertEqual(raised.exception.code, "tts.greektts15.worker_closed")
                    self.assertTrue(backend.is_idle)
                    backend.close()
                finally:
                    self._reap_owned_worker(backend)

    def test_close_stops_admission_before_waiting_for_worker(self) -> None:
        backend = GreekTtsF1SidecarBackend(config=_config(), worker_script=FIXTURE)
        def waiting(*args, **kwargs):
            with self.assertRaises(BackendError) as raised:
                backend.enqueue(SpeechUnit("during-close", 1, 1, "el-GR", "δοκιμή"))
            self.assertEqual(raised.exception.code, "tts.greektts15.worker_closed")
            return True
        try:
            with mock.patch.object(backend._closed, "wait", side_effect=waiting):
                backend.close()
            self.assertTrue(backend.is_idle)
        finally:
            self._reap_owned_worker(backend)

    def test_termination_error_does_not_skip_pipe_cleanup(self) -> None:
        backend = GreekTtsF1SidecarBackend(config=_config(), worker_script=FIXTURE)
        try:
            with mock.patch.object(backend, "_terminate_process", side_effect=ProcessLookupError("already exited")):
                with self.assertRaises(ProcessLookupError):
                    backend.close()
            self.assertTrue(backend._stdin.closed)
            self.assertTrue(backend._stdout.closed)
            self.assertFalse(backend._writer.is_alive())
            self.assertFalse(backend._reader.is_alive())
        finally:
            self._reap_owned_worker(backend)

    def test_worker_interpreter_is_digest_bound(self) -> None:
        config = _config()
        self.assertEqual(validate_worker_python(config.worker_python), Path(sys.executable))
        changed = ArtifactConfig(
            config.worker_python.path,
            (config.worker_python.size or 0) + 1,
            config.worker_python.sha256,
        )
        with self.assertRaisesRegex(BackendError, "tts.greektts15.worker_changed"):
            validate_worker_python(changed)

    def test_invalid_callback_order_fails_closed(self) -> None:
        for text in ("completed_only", "duplicate_started"):
            with self.subTest(text=text):
                backend = GreekTtsF1SidecarBackend(config=_config(), worker_script=FIXTURE)
                try:
                    backend.enqueue(SpeechUnit("response-invalid", 1, 1, "el-GR", text))
                    deadline = time.monotonic() + 2.0
                    while backend.last_error is None and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertIsNotNone(backend.last_error)
                    self.assertEqual(backend.last_error.code, "tts.greektts15.worker_protocol_invalid")
                finally:
                    backend.close()

    def test_config_requires_complete_worker_manifest(self) -> None:
        base = {
            "tts_backend": "greektts15_f1",
            "greektts15": {
                "model_root": "/private/model",
                "approval_manifest_path": "/private/approved.json",
                "approval_manifest_sha256": "a" * 64,
                "repository_root": "/private/repository",
                "worker_python": {"path": "/private/python"},
            },
        }
        with self.assertRaisesRegex(BackendError, "config.greektts15_worker_required"):
            voice_config_from_mapping({"macos": base})


if __name__ == "__main__":
    unittest.main()
