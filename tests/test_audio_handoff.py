from __future__ import annotations

import threading
import unittest
from array import array
from concurrent.futures import ThreadPoolExecutor, wait
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from jarvis_voice.audio import BoundedCaptureBuffer
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.audio_capture import CoreAudioCapture
from jarvis_voice.windows.audio_capture import WasapiAudioCapture


class AudioHandoffTests(unittest.TestCase):
    def test_old_callback_cannot_append_into_the_next_turn(self):
        for maximum in (None, 8):
            with self.subTest(maximum=maximum):
                capture = BoundedCaptureBuffer(capacity=2, sample_rate=16000, maximum_samples=maximum)
                capture.begin_turn()
                entered, release = threading.Event(), threading.Event()
                transition_entered = threading.Event()
                def copy(*args):
                    entered.set()
                    if not release.wait(2):
                        raise RuntimeError("test callback not released")
                    return bytearray(*args)
                def transition():
                    transition_entered.set()
                    capture.abort_turn()
                    capture.begin_turn()
                with patch("jarvis_voice.audio.bytearray", side_effect=copy, create=True), ThreadPoolExecutor(max_workers=2) as executor:
                    callback = executor.submit(capture.publish_from_callback, array("f", [7.0]).tobytes())
                    try:
                        self.assertTrue(entered.wait(2))
                        next_turn = executor.submit(transition)
                        self.assertTrue(transition_entered.wait(2))
                        wait([next_turn], timeout=0.1)
                    finally:
                        release.set()
                    callback.result(timeout=2)
                    next_turn.result(timeout=2)
                self.assertEqual(list(capture.end_turn().samples), [])

    def test_competing_callback_fails_immediately_and_discards_partial_capture(self):
        capture = BoundedCaptureBuffer(capacity=2, sample_rate=16000, maximum_samples=8)
        capture.begin_turn()
        entered, release = threading.Event(), threading.Event()
        def copy(*args):
            if not entered.is_set():
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test callback not released")
            return bytearray(*args)
        with patch("jarvis_voice.audio.bytearray", side_effect=copy, create=True), ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(capture.publish_from_callback, array("f", [1.0]).tobytes())
            try:
                self.assertTrue(entered.wait(2))
                second = executor.submit(capture.publish_from_callback, array("f", [2.0]).tobytes())
                self.assertFalse(second.result(timeout=0.3))
            finally:
                release.set()
            first.result(timeout=2)
        with self.assertRaisesRegex(BackendError, "audio.capture_overflow"):
            capture.end_turn()

    def test_conversion_owns_old_frames_without_blocking_new_capture(self):
        capture = BoundedCaptureBuffer(capacity=2, sample_rate=16000, maximum_samples=8)
        capture.begin_turn()
        capture.publish_from_callback(array("f", [1.0]).tobytes())
        entered, release = threading.Event(), threading.Event()
        class SlowArray(array):
            def frombytes(self, data):
                super().frombytes(data)
                if not entered.is_set():
                    entered.set()
                    if not release.wait(2):
                        raise RuntimeError("test conversion not released")
        def next_turn():
            capture.begin_turn()
            return capture.publish_from_callback(array("f", [2.0]).tobytes())
        with patch("jarvis_voice.audio.array", SlowArray), ThreadPoolExecutor(max_workers=2) as executor:
            old = executor.submit(capture.end_turn)
            try:
                self.assertTrue(entered.wait(2))
                new = executor.submit(next_turn)
                self.assertTrue(new.result(timeout=0.3))
            finally:
                release.set()
            self.assertEqual(list(old.result(timeout=2).samples), [1.0])
        self.assertEqual(list(capture.end_turn().samples), [2.0])

    def test_native_callbacks_keep_the_buffer_reference_during_stop(self):
        for capture_type in (WasapiAudioCapture, CoreAudioCapture):
            with self.subTest(capture=capture_type.__name__):
                class Capture(capture_type):
                    def __getattribute__(self, name):
                        value = super().__getattribute__(name)
                        if name == "_buffer" and value is not None:
                            self._buffer = None
                        return value
                capture = Capture(queue_capacity=2, maximum_utterance_seconds=1)
                buffer = Mock()
                capture._buffer = buffer
                data = array("f", [1.0]).tobytes()
                capture._callback(data, 1, None, None)
                buffer.publish_from_callback.assert_called_once_with(data)


if __name__ == "__main__":
    unittest.main()
