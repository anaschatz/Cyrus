from __future__ import annotations

import threading
import time
import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.audio_queue_playback import MacAudioQueueStreamingPlayback


class FakeAudioQueueApi:
    def __init__(self) -> None:
        self.callback = None
        self.queue = object()
        self.buffers: list[object] = []
        self.pending: list[object] = []
        self.payloads: list[bytes] = []
        self.calls: list[tuple[str, bool | None]] = []

    def create(self, sample_rate: int, callback: object) -> object:
        if sample_rate != 24_000:
            raise AssertionError("unexpected sample rate")
        self.callback = callback
        self.calls.append(("create", None))
        return self.queue

    def allocate(self, queue: object, size: int) -> object:
        if queue is not self.queue or size != 512:
            raise AssertionError("unexpected allocation")
        buffer = object()
        self.buffers.append(buffer)
        return buffer

    def enqueue(self, queue: object, buffer: object, pcm16: bytes) -> None:
        if queue is not self.queue or buffer in self.pending:
            raise AssertionError("invalid enqueue")
        self.pending.append(buffer)
        self.payloads.append(pcm16)

    def prime(self, queue: object) -> None:
        self.calls.append(("prime", None))

    def start(self, queue: object) -> None:
        self.calls.append(("start", None))

    def stop(self, queue: object, *, immediate: bool) -> None:
        self.calls.append(("stop", immediate))

    def dispose(self, queue: object, *, immediate: bool) -> None:
        self.calls.append(("dispose", immediate))

    def complete_next(self) -> None:
        buffer = self.pending.pop(0)
        assert callable(self.callback)
        self.callback(buffer)


class MacAudioQueueStreamingPlaybackTests(unittest.TestCase):
    def create_playback(self, api: FakeAudioQueueApi) -> MacAudioQueueStreamingPlayback:
        return MacAudioQueueStreamingPlayback(
            api=api,
            buffer_count=4,
            buffer_bytes=512,
            maximum_staging_bytes=4_096,
            write_timeout_seconds=0.2,
            finish_timeout_seconds=1.0,
        )

    def test_prebuffers_native_queue_before_starting(self) -> None:
        api = FakeAudioQueueApi()
        playback = self.create_playback(api)
        playback.start(24_000)

        playback.write(b"\x01\x00" * 1_024)

        self.assertEqual(len(api.pending), 4)
        self.assertEqual(api.calls[-2:], [("prime", None), ("start", None)])
        self.assertEqual(b"".join(api.payloads), b"\x01\x00" * 1_024)
        playback.abort()

    def test_finish_flushes_partial_buffer_and_waits_for_playback(self) -> None:
        api = FakeAudioQueueApi()
        playback = self.create_playback(api)
        playback.start(24_000)
        payload = b"\x02\x00" * 300
        playback.write(payload)
        finished = threading.Event()

        def finish() -> None:
            playback.finish()
            finished.set()

        worker = threading.Thread(target=finish)
        worker.start()
        deadline = time.monotonic() + 0.5
        while len(api.pending) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(b"".join(api.payloads), payload)
        self.assertIn(("start", None), api.calls)
        while api.pending:
            api.complete_next()
        worker.join(1.0)

        self.assertTrue(finished.is_set())
        self.assertEqual(api.calls[-2:], [("stop", True), ("dispose", True)])

    def test_abort_is_immediate_idempotent_and_discards_staged_audio(self) -> None:
        api = FakeAudioQueueApi()
        playback = self.create_playback(api)
        playback.start(24_000)
        playback.write(b"\x03\x00" * 100)

        playback.abort()
        playback.abort()
        playback.write(b"\x04\x00" * 300)

        self.assertEqual(api.payloads, [])
        self.assertEqual(api.calls.count(("stop", True)), 1)
        self.assertEqual(api.calls.count(("dispose", True)), 1)

    def test_rejects_malformed_pcm_and_write_before_start(self) -> None:
        playback = self.create_playback(FakeAudioQueueApi())
        with self.assertRaisesRegex(BackendError, "playback_not_started"):
            playback.write(b"\x00\x00")
        playback.start(24_000)
        with self.assertRaisesRegex(BackendError, "playback_audio_invalid"):
            playback.write(b"\x00")
        playback.abort()


if __name__ == "__main__":
    unittest.main()
