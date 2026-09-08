from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.errors import BackendError
from jarvis_voice.protocol import new_uuid4
from jarvis_voice.windows.sapi import SapiTtsBackend


def unit(epoch: int, index: int, locale: str = "el-GR") -> SpeechUnit:
    return SpeechUnit(new_uuid4(), epoch, index, locale, f"unit-{index}")


class FakeSapiAdapter:
    def __init__(self, *, initially_done: bool = True) -> None:
        self.done = initially_done
        self.opened = False
        self.closed = False
        self.spoken: list[tuple[str, str]] = []
        self.purges = 0
        self.started = threading.Event()

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def speak_async(self, text: str, voice_description: str) -> None:
        self.spoken.append((text, voice_description)); self.started.set()

    def wait_until_done(self, timeout_ms: int) -> bool:
        del timeout_ms
        return self.done

    def purge(self) -> None:
        self.purges += 1; self.done = True

    def voices(self) -> list[str]:
        return ["Stefanos", "David"]


class SapiTests(unittest.TestCase):
    def test_terminal_failures_are_reported_and_discard_queued_speech(self) -> None:
        for stage in ("factory", "open", "speak_async", "close"):
            with self.subTest(stage=stage):
                release = threading.Event()
                adapter = FakeSapiAdapter()
                adapter.close = Mock()
                if stage != "factory":
                    setattr(adapter, stage, Mock(side_effect=OSError("private failure")))

                def factory():
                    release.wait(1)
                    if stage == "factory":
                        raise OSError("private failure")
                    return adapter

                backend = SapiTtsBackend(voice_mapping={}, adapter_factory=factory)
                try:
                    backend.enqueue(unit(1, 1))
                    backend.enqueue(unit(1, 2))
                    release.set()
                    if stage == "close":
                        backend.close()
                    backend._thread.join(1)
                    self.assertFalse(backend._thread.is_alive())
                    self.assertIsInstance(backend.last_error, BackendError)
                    self.assertTrue(backend._queue.empty())
                    self.assertTrue(backend.is_idle)
                    with self.assertRaises(BackendError):
                        backend.enqueue(unit(2, 1))
                    self.assertEqual(adapter.close.call_count, 0 if stage == "factory" else 1)
                finally:
                    release.set()
                    backend.close()

    def test_older_cancellation_does_not_purge_newer_active_speech(self) -> None:
        adapter = FakeSapiAdapter(initially_done=False)
        backend = SapiTtsBackend(voice_mapping={}, adapter_factory=lambda: adapter)
        self.addCleanup(backend.close)
        backend.enqueue(unit(5, 1))
        self.assertTrue(adapter.started.wait(1))
        backend.cancel(3, "late_cancel")
        self.assertFalse(backend._cancel_event.is_set())
        adapter.done = True
        self.assertTrue(backend.wait_idle())
        self.assertEqual(adapter.purges, 0)

    def test_closed_backend_rejects_new_speech(self) -> None:
        backend = SapiTtsBackend(voice_mapping={}, adapter_factory=FakeSapiAdapter)
        backend.close()
        with self.assertRaises(BackendError):
            backend.enqueue(unit(1, 1))

    def test_close_from_started_callback_does_not_emit_completion(self) -> None:
        completed = []
        backend = SapiTtsBackend(
            voice_mapping={},
            adapter_factory=FakeSapiAdapter,
            on_started=lambda _unit: backend.close(),
            on_completed=completed.append,
        )
        self.addCleanup(backend.close)
        backend.enqueue(unit(1, 1))
        backend._thread.join(1)
        self.assertFalse(backend._thread.is_alive())
        self.assertIsNone(backend.last_error)
        self.assertEqual(completed, [])

    def test_ordered_units_and_locale_voice_mapping(self) -> None:
        adapter = FakeSapiAdapter()
        started: list[int] = []; completed: list[int] = []
        backend = SapiTtsBackend(
            voice_mapping={"el-GR": "Stefanos", "en-US": "David"},
            adapter_factory=lambda: adapter,
            on_started=lambda item: started.append(item.unit_index),
            on_completed=lambda item: completed.append(item.unit_index),
        )
        self.addCleanup(backend.close)
        response_id = new_uuid4()
        backend.enqueue(SpeechUnit(response_id, 1, 1, "el-GR", "ένα"))
        backend.enqueue(SpeechUnit(response_id, 1, 2, "en-US", "two"))
        self.assertTrue(backend.wait_idle())
        self.assertEqual(started, [1, 2]); self.assertEqual(completed, [1, 2])
        self.assertEqual(adapter.spoken, [("ένα", "Stefanos"), ("two", "David")])

    def test_cancel_purges_active_stream_and_ignores_late_completion(self) -> None:
        adapter = FakeSapiAdapter(initially_done=False)
        completed: list[int] = []
        backend = SapiTtsBackend(voice_mapping={}, adapter_factory=lambda: adapter, on_completed=lambda item: completed.append(item.unit_index))
        self.addCleanup(backend.close)
        backend.enqueue(unit(4, 1))
        self.assertTrue(adapter.started.wait(1))
        backend.cancel(5, "ptt_down")
        self.assertTrue(backend.wait_idle())
        self.assertEqual(adapter.purges, 1)
        self.assertEqual(completed, [])
        self.assertTrue(any(timing.event == "purge_dispatch" for timing in backend.timings))

    def test_cancel_clears_pending_units_of_old_epoch(self) -> None:
        adapter = FakeSapiAdapter(initially_done=False)
        backend = SapiTtsBackend(voice_mapping={}, adapter_factory=lambda: adapter)
        self.addCleanup(backend.close)
        backend.enqueue(unit(2, 1)); backend.enqueue(unit(2, 2))
        self.assertTrue(adapter.started.wait(1))
        backend.cancel(3, "ptt_down")
        self.assertTrue(backend.wait_idle())
        self.assertEqual(len(adapter.spoken), 1)


if __name__ == "__main__":
    unittest.main()
