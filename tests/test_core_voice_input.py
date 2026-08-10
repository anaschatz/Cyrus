from __future__ import annotations

import unittest
from array import array

import _bootstrap  # noqa: F401

from jarvis_voice.backends import AsrScore, AudioChunk, LanguageMetadata, TranscriptResult
from jarvis_voice.core import VoiceCore
from jarvis_voice.errors import BackendError
from jarvis_voice.fakes import FakeAsrBackend, FakeTtsBackend
from jarvis_voice.state import InputState
from jarvis_voice.vad import FakeVadBackend


class EmptyAsr:
    def transcribe(self, audio: object) -> TranscriptResult:
        raise BackendError("asr.empty_transcript", "empty", subsystem="asr")


class CoreVoiceInputTests(unittest.TestCase):
    def test_vad_no_speech_emits_no_speech_and_returns_idle(self) -> None:
        core = VoiceCore(asr=FakeAsrBackend(), tts=FakeTtsBackend())
        core.ptt_down()
        event = core.ptt_up_and_process(AudioChunk(array("f", [0]), 16_000), FakeVadBackend(start_sample=None))
        self.assertEqual(event.type, "input.no_speech")
        self.assertEqual(core.state.input_state, InputState.IDLE)

    def test_asr_empty_transcript_becomes_no_speech(self) -> None:
        core = VoiceCore(asr=EmptyAsr(), tts=FakeTtsBackend())
        core.ptt_down()
        event = core.ptt_up_and_process(AudioChunk(array("f", [1]), 16_000), FakeVadBackend())
        self.assertEqual(event.type, "input.no_speech")
        self.assertEqual(event.payload["reason"], "asr_empty_transcript")

    def test_english_fake_turn_keeps_raw_language_metadata(self) -> None:
        result = TranscriptResult("Open it.", LanguageMetadata("en", 0.93), AsrScore(0.8, -0.22))
        core = VoiceCore(asr=FakeAsrBackend(result), tts=FakeTtsBackend())
        core.ptt_down()
        event = core.ptt_up_and_process(AudioChunk(array("f", [1]), 16_000), FakeVadBackend())
        self.assertEqual(event.type, "input.transcript.final")
        self.assertEqual(event.payload["language"]["code"], "en")
        self.assertNotIn("confidence", event.payload)

    def test_capture_failure_is_redacted_client_error(self) -> None:
        core = VoiceCore(asr=FakeAsrBackend(), tts=FakeTtsBackend())
        core.ptt_down()
        event = core.input_failed("audio.capture_overflow")
        self.assertEqual(event.type, "client.error")
        self.assertEqual(event.payload, {"code": "audio.capture_overflow", "retryable": False})
        self.assertEqual(core.state.input_state, InputState.IDLE)


if __name__ == "__main__":
    unittest.main()
