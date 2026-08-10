from __future__ import annotations

import unittest
from array import array

import _bootstrap  # noqa: F401

from jarvis_voice.asr_factory import FallbackAsrBackend
from jarvis_voice.backends import AsrScore, AudioChunk, LanguageMetadata, TranscriptResult
from jarvis_voice.core import PreparedTranscription, VoiceCore
from jarvis_voice.errors import BackendError
from jarvis_voice.fakes import FakeOrchestrator, FakeTtsBackend
from jarvis_voice.vad import FakeVadBackend


class Backend:
    def __init__(self, result: TranscriptResult | None = None, error: BackendError | None = None) -> None:
        self.result = result; self.error = error; self.calls = 0; self.cancelled = False
    def transcribe(self, audio):
        del audio; self.calls += 1
        if self.error: raise self.error
        assert self.result is not None
        return self.result
    def cancel_current(self): self.cancelled = True


class MilestoneTwoEndToEndTests(unittest.TestCase):
    def fake_cuda_turn(self, text: str, language: str) -> None:
        result = TranscriptResult(text, LanguageMetadata(language, 0.95), AsrScore(0.8, -0.2))
        cuda = Backend(result); cpu = Backend(result)
        asr = FallbackAsrBackend(cuda, cpu, enabled=True, fallbackable_codes=frozenset({"cuda.oom"}))
        core = VoiceCore(asr=asr, tts=FakeTtsBackend()); server = FakeOrchestrator(session_id=core.session_id)
        core.handle_server_events(server.accept(core.client_hello()))
        core.ptt_down(); prepared = core.prepare_transcription(AudioChunk(array("f", [1]), 16_000), FakeVadBackend())
        self.assertIsInstance(prepared, PreparedTranscription)
        final = core.complete_transcription(prepared.turn_id, asr.transcribe(prepared.audio))  # type: ignore[union-attr]
        self.assertIsNotNone(final); self.assertEqual(final.type, "input.transcript.final")  # type: ignore[union-attr]
        self.assertEqual(final.payload["language"]["code"], language)  # type: ignore[union-attr]
        self.assertEqual(len(server.accept(final)), 1)  # type: ignore[arg-type]
        self.assertEqual((cuda.calls, cpu.calls), (1, 0))

    def test_fake_full_greek_cuda_turn(self) -> None:
        self.fake_cuda_turn("Άνοιξέ το.", "el")

    def test_fake_full_english_cuda_turn(self) -> None:
        self.fake_cuda_turn("Open it.", "en")

    def test_fake_cuda_to_cpu_fallback_produces_one_final(self) -> None:
        result = TranscriptResult("Fallback.", LanguageMetadata("en", 0.9), AsrScore(0.7, -0.3))
        cuda = Backend(error=BackendError("cuda.oom", "failed", subsystem="cuda")); cpu = Backend(result)
        asr = FallbackAsrBackend(cuda, cpu, enabled=True, fallbackable_codes=frozenset({"cuda.oom"}))
        core = VoiceCore(asr=asr, tts=FakeTtsBackend()); core.ptt_down()
        final = core.ptt_up_and_process(AudioChunk(array("f", [1]), 16_000), FakeVadBackend())
        self.assertEqual(final.type, "input.transcript.final")
        self.assertEqual((cuda.calls, cpu.calls), (1, 1))

    def test_late_cuda_result_after_new_ptt_is_stale(self) -> None:
        result = TranscriptResult("Late.", LanguageMetadata("en", 0.9), AsrScore(0.7, -0.3))
        core = VoiceCore(asr=Backend(result), tts=FakeTtsBackend()); core.ptt_down()
        prepared = core.prepare_transcription(AudioChunk(array("f", [1]), 16_000), FakeVadBackend())
        assert isinstance(prepared, PreparedTranscription)
        core.ptt_down()
        self.assertIsNone(core.complete_transcription(prepared.turn_id, result))

    def test_barge_in_tombstones_output_while_cuda_turn_is_pending(self) -> None:
        tts = FakeTtsBackend(); result = TranscriptResult("Pending.", LanguageMetadata("en", 0.9), AsrScore(0.7, -0.3))
        core = VoiceCore(asr=Backend(result), tts=tts); server = FakeOrchestrator(session_id=core.session_id)
        core.handle_server_events(server.script_response(["Long LLM output."], locale="en-US").events)
        active = tts.start_next(); core.on_tts_unit_started(active)
        events = core.ptt_down()
        self.assertEqual([item.type for item in events[:2]], ["response.cancel", "playback.interrupted"])
        self.assertTrue(tts.is_idle)
        self.assertEqual(core.on_tts_unit_completed(active), [])

    def test_dual_backend_failure_can_emit_only_one_redacted_client_error(self) -> None:
        result = TranscriptResult("unused", LanguageMetadata("en", 0.1), AsrScore(0.1, -2.0))
        asr = FallbackAsrBackend(
            Backend(error=BackendError("cuda.oom", "secret", subsystem="cuda")),
            Backend(result, BackendError("asr.cpu_failed", "secret", subsystem="asr")),
            enabled=True,
            fallbackable_codes=frozenset({"cuda.oom"}),
        )
        core = VoiceCore(asr=asr, tts=FakeTtsBackend()); core.ptt_down()
        prepared = core.prepare_transcription(AudioChunk(array("f", [1]), 16_000), FakeVadBackend())
        assert isinstance(prepared, PreparedTranscription)
        with self.assertRaises(BackendError) as captured:
            asr.transcribe(prepared.audio)
        error = core.transcription_failed(prepared.turn_id, captured.exception.code)
        self.assertIsNotNone(error); self.assertEqual(error.type, "client.error")  # type: ignore[union-attr]
        self.assertEqual(error.payload, {"code": "asr.cpu_failed", "retryable": False})  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
