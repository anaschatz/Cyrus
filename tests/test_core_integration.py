from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.core import VoiceCore
from jarvis_voice.fakes import FakeAsrBackend, FakeOrchestrator, FakeTtsBackend
from jarvis_voice.protocol import new_uuid4


class CoreIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tts = FakeTtsBackend()
        self.core = VoiceCore(asr=FakeAsrBackend(), tts=self.tts)
        self.server = FakeOrchestrator(session_id=self.core.session_id)
        self.core.handle_server_events(self.server.accept(self.core.client_hello()))

    def submit_turn(self) -> str:
        turn_id = new_uuid4()
        self.core.ptt_down(turn_id)
        transcript = self.core.ptt_up_and_transcribe(object())
        self.core.handle_server_events(self.server.accept(transcript))
        return turn_id

    def test_streamed_text_does_not_control_speech_completion(self) -> None:
        turn_id = self.submit_turn()
        scripted = self.server.script_response(
            ["Βεβαίως.", "Το ανοίγω τώρα."],
            locale="el-GR",
            turn_id=turn_id,
            text_chunks=["Βε", "βαίως. Το ", "ανοίγω τώρα."],
        )
        text_only = scripted.events[:5]  # started, 3 deltas, text.done
        self.assertEqual(self.core.handle_server_events(text_only), [])
        self.assertTrue(self.tts.is_idle)
        speech_events = scripted.events[5:]
        self.assertEqual(self.core.handle_server_events(speech_events), [])
        self.assertEqual(len(self.tts.pending), 2)

        first = self.tts.start_next()
        started = self.core.on_tts_unit_started(first)
        self.assertEqual([event.type for event in started], ["playback.started"])
        self.assertEqual(self.core.on_tts_unit_completed(self.tts.complete_active()), [])

        second = self.tts.start_next()
        self.assertEqual(self.core.on_tts_unit_started(second), [])
        completed = self.core.on_tts_unit_completed(self.tts.complete_active())
        self.assertEqual([event.type for event in completed], ["playback.completed"])

    def test_empty_speech_response_completes_on_speech_done(self) -> None:
        turn_id = self.submit_turn()
        scripted = self.server.script_response([], locale="el-GR", turn_id=turn_id)
        outbound = self.core.handle_server_events(scripted.events)
        self.assertEqual([event.type for event in outbound], ["playback.completed"])

    def test_ptt_hard_barge_in_tombstones_and_cancels_tts(self) -> None:
        turn_id = self.submit_turn()
        scripted = self.server.script_response(
            ["A long answer."], locale="en-US", turn_id=turn_id
        )
        self.core.handle_server_events(scripted.events)
        active = self.tts.start_next()
        self.core.on_tts_unit_started(active)
        events = self.core.ptt_down()
        self.assertEqual(
            [event.type for event in events],
            ["response.cancel", "playback.interrupted", "input.capture.started"],
        )
        self.assertTrue(self.tts.is_idle)
        self.assertEqual(self.core.on_tts_unit_completed(active), [])
        self.assertFalse(any(event.type == "playback.completed" for event in events))

    def test_duplicate_server_frame_has_no_second_effect(self) -> None:
        turn_id = self.submit_turn()
        scripted = self.server.script_response(
            ["Μία."], locale="el-GR", turn_id=turn_id
        )
        for event in scripted.events:
            self.core.handle_server_event(event)
            self.assertEqual(
                self.core.handle_server_event(self.server.exact_duplicate(event)), []
            )
        self.assertEqual(len(self.tts.pending), 1)

    def test_reconnect_ack_deduplicates_same_event_id(self) -> None:
        self.core.ptt_down()
        transcript = self.core.ptt_up_and_transcribe(object())
        first_ack = self.server.accept(transcript)[0]
        self.assertEqual(first_ack.payload["status"], "accepted")

        reconnect_factory = type(self.core.client_events)(
            session_id=self.core.session_id
        )
        retransmission = reconnect_factory.create(
            transcript.type,
            transcript.payload,
            event_id=transcript.event_id,
            turn_id=transcript.turn_id,
        )
        duplicate_ack = self.server.accept(retransmission)[0]
        self.assertEqual(retransmission.seq, 1)
        self.assertEqual(duplicate_ack.payload["status"], "duplicate")

    def test_fake_orchestrator_delays_events_deterministically(self) -> None:
        event = self.server.script_response([], locale="en-US").events[0]
        self.server.queue_event(event, delay_steps=2)
        self.assertIsNone(self.server.receive())
        self.server.advance()
        self.assertIsNone(self.server.receive())
        self.server.advance()
        self.assertEqual(self.server.receive(), event)


if __name__ == "__main__":
    unittest.main()
