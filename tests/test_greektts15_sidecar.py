from __future__ import annotations

import hashlib
import sys
import threading
import time
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.config import ArtifactConfig, GreekTts15Config, voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.greektts15_sidecar import GreekTtsF1SidecarBackend, validate_worker_python


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fake_greektts15_sidecar_worker.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
