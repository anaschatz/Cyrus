from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.core import VoiceCore
from jarvis_voice.fakes import FakeAsrBackend, FakeClock, FakeTtsBackend
from jarvis_voice.protocol import EventFactory
from jarvis_voice.reliable import ReliableSubmissionManager


class ReliableSubmissionTests(unittest.TestCase):
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
        ack = server.create("server.ack", {"acked_event_id": self.original.event_id, "status": "accepted"})
        self.assertTrue(self.manager.acknowledge(ack))
        self.clock.advance(10)
        self.assertEqual(self.manager.due_retransmissions(self.core), [])


if __name__ == "__main__":
    unittest.main()
