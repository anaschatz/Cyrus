from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.fakes import FakeOrchestrator
from jarvis_voice.macos.interactive_smoke import GuidedSmokeController, MacSmokeLoopbackServer
from jarvis_voice.protocol import EventFactory, new_uuid4
from jarvis_voice.voice_auth import VoiceAuthKey


def event(name: str, timestamp: float, **details: object) -> dict[str, object]:
    return {"event": name, "timestamp": timestamp, **details}


def complete_voice(controller: GuidedSmokeController, *, start: float, language: str) -> None:
    controller.feed(event("ptt_down_received", start, output_active=False))
    controller.feed(event("ptt_up_received", start + 1.0))
    controller.feed(event("capture_turn_closed", start + 1.01))
    controller.feed(event("audio_normalized", start + 1.02))
    controller.feed(event("vad_completed", start + 1.04, speech_detected=True))
    controller.feed(event("asr_submitted", start + 1.05))
    controller.feed(event("asr_result_ready", start + 1.95))
    controller.feed(event("transcript_final_created", start + 2.0, language=language))
    controller.feed(event("speech_commit_received", start + 2.1, unit_index=1, locale=f"{language}-test"))
    controller.feed(event("speech_done_received", start + 2.2, last_unit_index=2))
    controller.feed(event("tts_unit_started", start + 2.3, unit_index=1, locale=f"{language}-test"))
    controller.feed(event("tts_unit_completed", start + 2.8, unit_index=1, locale=f"{language}-test"))
    controller.feed(event("playback_completed_created", start + 3.0))


class GuidedSmokeControllerTests(unittest.TestCase):
    def test_four_native_cases_produce_content_free_ready_evidence(self) -> None:
        controller = GuidedSmokeController(ptt_key="F9")
        self.assertEqual(controller.take_prompts()[0].code, "smoke.prompt.greek")
        controller.feed(event("audio_stream_started", 0.1))
        controller.feed(event("ptt_source_started", 0.2))

        complete_voice(controller, start=1.0, language="el")
        self.assertEqual(controller.stage, "english_turn")
        self.assertEqual(controller.take_prompts()[0].code, "smoke.prompt.english")
        complete_voice(controller, start=10.0, language="en")
        self.assertEqual(controller.stage, "no_speech")

        controller.take_prompts()
        controller.feed(event("ptt_down_received", 20.0, output_active=False))
        controller.feed(event("ptt_up_received", 20.1))
        controller.feed(event("input_no_speech_created", 20.25))
        self.assertEqual(controller.stage, "barge_in")

        controller.take_prompts()
        controller.feed(event("ptt_down_received", 30.0, output_active=False))
        controller.feed(event("ptt_up_received", 30.5))
        controller.feed(event("capture_turn_closed", 30.51))
        controller.feed(event("audio_normalized", 30.52))
        controller.feed(event("vad_completed", 30.54, speech_detected=True))
        controller.feed(event("asr_submitted", 30.55))
        controller.feed(event("asr_result_ready", 30.95))
        controller.feed(event("transcript_final_created", 31.0, language="el"))
        controller.feed(event("speech_commit_received", 31.1, unit_index=1, locale="el-GR"))
        controller.feed(event("tts_unit_started", 31.2, unit_index=1, locale="el-GR"))
        self.assertEqual(controller.take_prompts()[0].code, "smoke.prompt.barge_now")
        controller.feed(event("ptt_down_received", 31.3, output_active=True))
        controller.feed(event("tts_cancel_dispatched", 31.31))
        controller.feed(event("response_cancel_created", 31.32))
        controller.feed(event("playback_interrupted_created", 31.33))
        controller.feed(event("ptt_up_received", 31.4))
        controller.tick(backend_idle=True, now=31.45)

        report = controller.report()
        self.assertTrue(report["ready"])
        self.assertEqual(report["code"], "smoke.macos_interactive_ready")
        self.assertEqual(report["cases"]["greek_turn"]["language"], "el")  # type: ignore[index]
        self.assertEqual(report["cases"]["english_turn"]["language"], "en")  # type: ignore[index]
        self.assertEqual(report["cases"]["greek_turn"]["metrics"]["ptt_up_to_transcript_final_ms"], 1000.0)  # type: ignore[index]
        self.assertEqual(report["cases"]["greek_turn"]["metrics"]["asr_submitted_to_result_ms"], 900.0)  # type: ignore[index]
        self.assertEqual(report["cases"]["greek_turn"]["metrics"]["transcript_final_to_speech_commit_ms"], 100.0)  # type: ignore[index]
        self.assertEqual(report["cases"]["greek_turn"]["metrics"]["speech_commit_to_tts_start_ms"], 200.0)  # type: ignore[index]
        self.assertEqual(report["cases"]["greek_turn"]["metrics"]["ptt_up_to_tts_start_ms"], 1300.0)  # type: ignore[index]
        self.assertEqual(report["cases"]["no_speech"]["metrics"]["ptt_up_to_no_speech_ms"], 150.0)  # type: ignore[index]
        self.assertAlmostEqual(report["cases"]["barge_in"]["metrics"]["ptt_down_to_backend_idle_ms"], 150.0)  # type: ignore[index]
        self.assertEqual(
            report["native"],
            {
                "microphone_opened": True,
                "passive_ptt_started": True,
                "system_tts_exercised": True,
                "acoustic_silence_measured": False,
                "satisfies_native_ready_gate": False,
            },
        )
        encoded = json.dumps(report)
        for forbidden in ("event_id", "turn_id", "response_id", "/Users/", "user phrase", "audio_samples"):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(report["privacy"]["raw_audio_persisted"])  # type: ignore[index]
        self.assertFalse(report["privacy"]["transcript_text_persisted"])  # type: ignore[index]

    def test_language_mismatch_and_timeout_are_typed_blockers(self) -> None:
        mismatch = GuidedSmokeController(ptt_key="F9")
        mismatch.take_prompts()
        mismatch.feed(event("ptt_up_received", 1.0))
        mismatch.feed(event("transcript_final_created", 2.0, language="en", language_score=0.75))
        self.assertEqual(mismatch.report()["code"], "smoke.language_mismatch")
        self.assertEqual(mismatch.report()["cases"]["greek_turn"]["detected_language"], "en")
        self.assertEqual(mismatch.report()["cases"]["greek_turn"]["language_score"], 0.75)
        self.assertFalse(mismatch.report()["ready"])

        timed_out = GuidedSmokeController(ptt_key="F12")
        timed_out.timeout()
        self.assertEqual(timed_out.report()["code"], "smoke.interaction_timeout")

    def test_tombstoned_response_cannot_complete_after_barge_in(self) -> None:
        controller = GuidedSmokeController(ptt_key="F9")
        controller.stage = "barge_in"
        controller.take_prompts()
        controller.feed(event("transcript_final_created", 1.0, language="el"))
        controller.feed(event("tts_unit_started", 1.1, unit_index=1, locale="el-GR"))
        controller.take_prompts()
        controller.feed(event("ptt_down_received", 1.2, output_active=True))
        controller.feed(event("playback_completed_created", 1.3))
        self.assertEqual(controller.report()["code"], "smoke.tombstoned_response_completed")

    def test_client_error_fails_fast_with_content_free_runtime_code(self) -> None:
        controller = GuidedSmokeController(ptt_key="CONTROL_OPTION")
        controller.take_prompts()
        controller.feed(event("client_error_created", 1.0, code="audio.capture_overflow"))
        report = controller.report()
        self.assertEqual(report["code"], "smoke.client_error")
        self.assertEqual(report["cases"]["greek_turn"]["runtime_code"], "audio.capture_overflow")

    def test_server_rejects_non_loopback_url(self) -> None:
        with self.assertRaisesRegex(BackendError, "smoke.non_loopback_server"):
            MacSmokeLoopbackServer(
                "ws://192.0.2.1:8765",
                VoiceAuthKey(1, b"k" * 32),
            )

    def test_server_emits_frozen_response_cancelled_payload(self) -> None:
        response_id = new_uuid4()
        turn_id = new_uuid4()
        request = EventFactory().create(
            "response.cancel",
            {},
            turn_id=turn_id,
            response_id=response_id,
        )
        reply = MacSmokeLoopbackServer._cancelled_reply(FakeOrchestrator(), request)

        self.assertEqual(reply.type, "response.cancelled")
        self.assertEqual(
            reply.payload,
            {"reason_code": "user_cancelled", "partial_output": False},
        )
        self.assertEqual(reply.turn_id, turn_id)
        self.assertEqual(reply.response_id, response_id)

    def test_barge_in_sentence_is_bounded_without_artificial_repetition(self) -> None:
        greek = MacSmokeLoopbackServer._response_sentences(greek=True, long_response=True)
        english = MacSmokeLoopbackServer._response_sentences(greek=False, long_response=True)

        self.assertEqual(len(greek), 2)
        self.assertEqual(len(english), 2)
        self.assertLessEqual(len(greek[0].encode("utf-8")), 512)
        self.assertLessEqual(len(english[0].encode("utf-8")), 512)
        self.assertEqual(greek[0].count("Αυτή"), 1)
        self.assertEqual(english[0].count("This"), 1)


class InteractiveSmokeScriptTests(unittest.TestCase):
    def test_script_requires_private_inputs_and_exposes_timeout(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_macos_interactive_smoke.py"
        spec = importlib.util.spec_from_file_location("run_macos_interactive_smoke", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parsed = module.build_parser().parse_args(
            [
                "--config",
                "/private/voice.json",
                "--model-dir",
                "/private/model",
                "--output",
                "/private/evidence.json",
                "--timeout-seconds",
                "90",
                "--ptt-key",
                "CONTROL_OPTION",
                "--trace-stages",
            ]
        )
        self.assertEqual(parsed.timeout_seconds, 90.0)
        self.assertEqual(parsed.model_dir, "/private/model")
        self.assertEqual(parsed.ptt_key, "CONTROL_OPTION")
        self.assertTrue(parsed.trace_stages)


if __name__ == "__main__":
    unittest.main()
