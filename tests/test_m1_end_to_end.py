from __future__ import annotations

import unittest
from array import array

import _bootstrap  # noqa: F401

from jarvis_voice.backends import AsrScore, AudioChunk, LanguageMetadata, TranscriptResult
from jarvis_voice.core import VoiceCore
from jarvis_voice.fakes import FakeAsrBackend, FakeOrchestrator, FakeTtsBackend
from jarvis_voice.protocol import new_uuid4
from jarvis_voice.vad import FakeVadBackend


class MilestoneOneEndToEndTests(unittest.TestCase):
    def run_turn(self, *, language: str, locale: str, transcript: str, sentences: list[str]) -> list[str]:
        tts = FakeTtsBackend()
        asr = FakeAsrBackend(TranscriptResult(transcript, LanguageMetadata(language, 0.95), AsrScore(0.8, -0.2)))
        core = VoiceCore(asr=asr, tts=tts)
        server = FakeOrchestrator(session_id=core.session_id)
        core.handle_server_events(server.accept(core.client_hello()))
        turn_id = new_uuid4(); outbound_types: list[str] = []
        outbound_types.extend(event.type for event in core.ptt_down(turn_id))
        final = core.ptt_up_and_process(AudioChunk(array("f", [1]), 16_000), FakeVadBackend())
        core.handle_server_events(server.accept(final))
        scripted = server.script_response(sentences, locale=locale, turn_id=turn_id, text_chunks=[sentences[0], sentences[-1]], interleave_speech=True)
        types = [event.type for event in scripted.events]
        self.assertLess(types.index("response.speech.commit"), types.index("response.text.done"))
        for event in scripted.events:
            outbound_types.extend(item.type for item in core.handle_server_event(event))
            if event.type == "response.speech.commit":
                active = tts.start_next()
                outbound_types.extend(item.type for item in core.on_tts_unit_started(active))
                outbound_types.extend(item.type for item in core.on_tts_unit_completed(tts.complete_active()))
        self.assertIn("playback.started", outbound_types)
        self.assertEqual(outbound_types.count("playback.completed"), 1)
        return outbound_types

    def test_full_fake_greek_turn(self) -> None:
        self.run_turn(language="el", locale="el-GR", transcript="Άνοιξέ το.", sentences=["Βεβαίως.", "Το ανοίγω τώρα."])

    def test_full_fake_english_turn(self) -> None:
        self.run_turn(language="en", locale="en-US", transcript="Open it.", sentences=["Certainly.", "I will open it now."])

    def test_interruption_during_last_unit_never_completes_old_response(self) -> None:
        tts = FakeTtsBackend(); core = VoiceCore(asr=FakeAsrBackend(), tts=tts); server = FakeOrchestrator(session_id=core.session_id)
        core.handle_server_events(server.accept(core.client_hello()))
        scripted = server.script_response(["First.", "Last."], locale="en-US")
        core.handle_server_events(scripted.events)
        first = tts.start_next(); core.on_tts_unit_started(first); core.on_tts_unit_completed(tts.complete_active())
        last = tts.start_next(); core.on_tts_unit_started(last)
        events = core.ptt_down()
        self.assertEqual([event.type for event in events[:2]], ["response.cancel", "playback.interrupted"])
        self.assertEqual(core.on_tts_unit_completed(last), [])
        self.assertFalse(core.state.response.playback_completed_emitted)

    def test_interruption_during_first_unit_purges_remaining_response(self) -> None:
        tts = FakeTtsBackend(); core = VoiceCore(asr=FakeAsrBackend(), tts=tts); server = FakeOrchestrator(session_id=core.session_id)
        core.handle_server_events(server.accept(core.client_hello()))
        scripted = server.script_response(["First.", "Last."], locale="en-US")
        core.handle_server_events(scripted.events)
        first = tts.start_next(); core.on_tts_unit_started(first)
        events = core.ptt_down()
        self.assertEqual([event.type for event in events[:2]], ["response.cancel", "playback.interrupted"])
        self.assertTrue(tts.is_idle)
        self.assertEqual(core.on_tts_unit_completed(first), [])
        self.assertFalse(core.state.response.playback_completed_emitted)


if __name__ == "__main__":
    unittest.main()
