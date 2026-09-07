from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import test_cuda_probe as fixture
from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.cuda_probe import run_active_cuda_probe


class ProbeCleanupTests(unittest.TestCase):
    def test_sampler_and_backend_cleanup_cover_start_clock_and_close_failures(self):
        config = voice_config_from_mapping({"whisper_model_directory": "fake-model",
            "asr": {"execution_profile": "cuda", "cuda": {"device_index": 0, "vram_poll_interval_ms": 10}}})
        for phase in ("close", "clock", "start"):
            with self.subTest(phase=phase):
                stop = threading.Event()
                created = []
                backend = Mock(metrics=[])
                if phase == "close":
                    backend.close.side_effect = OSError("test close failed")
                def make_thread(**kwargs):
                    thread = threading.Thread(**kwargs)
                    created.append(thread)
                    if phase == "start":
                        thread.start = Mock(side_effect=RuntimeError("test start failed"))
                    return thread
                def clock():
                    if phase == "clock":
                        raise ValueError("test clock failed")
                    return 0.0
                api = SimpleNamespace(Event=lambda: stop, Thread=make_thread)
                error = {"close": OSError, "clock": ValueError, "start": RuntimeError}[phase]
                try:
                    with patch("jarvis_voice.cuda_probe.threading", api):
                        with self.assertRaises(error):
                            run_active_cuda_probe(config, inventory=fixture.Inventory(),
                                backend_factory=lambda *args, **kwargs: backend, clock=clock)
                    self.assertTrue(stop.is_set())
                    self.assertFalse(any(thread.is_alive() for thread in created))
                    backend.close.assert_called_once_with()
                    if phase != "close":
                        backend.transcribe.assert_not_called()
                finally:
                    stop.set()
                    for thread in created:
                        if thread.ident is not None:
                            thread.join(2)


if __name__ == "__main__":
    unittest.main()
