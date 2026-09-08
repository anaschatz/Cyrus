from __future__ import annotations

import unittest
from unittest.mock import Mock

import _bootstrap  # noqa: F401

from jarvis_voice.core import VoiceCore
from jarvis_voice.fakes import FakeAsrBackend, FakeClock, FakeTtsBackend
from jarvis_voice.protocol import EventFactory
from jarvis_voice.reliable import ReliableSubmissionManager
from jarvis_voice.runtime import WindowsVoiceRuntime
from jarvis_voice.macos.runtime import MacVoiceRuntime


class ReliableSubmissionTests(unittest.TestCase):
    def test_new_capture_retires_old_submission_in_both_runtimes(self) -> None:
        for runtime_type in (WindowsVoiceRuntime, MacVoiceRuntime):
            with self.subTest(runtime=runtime_type.__name__):
                clock = FakeClock()
                core = VoiceCore(asr=FakeAsrBackend(), tts=FakeTtsBackend())
                core.ptt_down()
                old = core.ptt_up_and_transcribe(object())
                manager = ReliableSubmissionManager(clock)
                manager.begin(old)
                runtime = runtime_type.__new__(runtime_type)
                runtime.core, runtime.reliable = core, manager
                runtime.transport = Mock()
                runtime.timings = []
                runtime._record_timing = Mock()
                runtime._turn_cycles, runtime._ptt_cycle = {}, 1
                runtime._send_all(core.ptt_down())
                clock.advance(4)
                self.assertEqual(manager.due_retransmissions(core), [])
                self.assertFalse(manager.consume_connected_timeout())
                self.assertFalse(manager.can_reconnect)
                self.assertIsNone(core.pending_submission_event_id)
                fresh = core.ptt_up_and_transcribe(object())
                runtime._send_all([fresh])
                clock.advance(0.5)
                self.assertEqual([event.event_id for event in manager.due_retransmissions(core)], [fresh.event_id])

    def test_delayed_initialization_keeps_elapsed_retries_available(self) -> None:
        core = VoiceCore(asr=FakeAsrBackend(), tts=FakeTtsBackend())
        core.ptt_down()
        original = core.ptt_up_and_transcribe(object())
        times = iter((0.0, 1.0, 1.1))
        manager = ReliableSubmissionManager(lambda: next(times))
        manager.begin(original)
        retries = manager.due_retransmissions(core)
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0].event_id, original.event_id)
        self.assertEqual(manager.due_retransmissions(core), [])

    def setUp(self) -> None:
        self.clock = FakeClock()
        self.core = VoiceCore(asr=FakeAsrBackend(), tts=FakeTtsBackend())
        self.core.ptt_down(); self.original = self.core.ptt_up_and_transcribe(object())
        self.manager = ReliableSubmissionManager(self.clock); self.manager.begin(self.original)

    def test_retries_get_new_seq_and_same_event_id(self) -> None:
        self.clock.advance(0.5)
        retry = self.manager.due_retransmissions(self.core)[0]
        self.assertEqual(retry.event_id, self.original.event_id)
        self.assertGreater(retry.seq, self.original.seq)

    def test_connected_timeout_is_consumed_once_and_ttl_is_original(self) -> None:
        self.clock.advance(4)
        self.assertTrue(self.manager.consume_connected_timeout())
        self.assertFalse(self.manager.consume_connected_timeout())
        self.assertTrue(self.manager.can_reconnect)
        self.clock.advance(26)
        self.assertFalse(self.manager.can_reconnect)

    def test_reconnect_resets_seq_but_preserves_event_identity(self) -> None:
        self.core.begin_new_connection()
        hello = self.core.client_hello()
        retry = self.manager.reconnect_event(self.core)
        self.assertEqual(hello.seq, 1)
        self.assertEqual(retry.seq, 2)
        self.assertEqual(retry.event_id, self.original.event_id)

    def test_ack_stops_processing(self) -> None:
        server = EventFactory(session_id=self.core.session_id)
        ack = server.create("server.ack", {"acked_event_id": self.original.event_id.upper(), "status": "accepted"})
        self.assertTrue(self.manager.acknowledge(ack))
        self.core.handle_server_event(ack)
        self.assertIsNone(self.core.pending_submission_event_id)
        self.clock.advance(10)
        self.assertEqual(self.manager.due_retransmissions(self.core), [])


if __name__ == "__main__":
    unittest.main()
