from __future__ import annotations

import importlib.util
import contextlib
import io
import tempfile
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.macos import interactive_smoke
from jarvis_voice.fakes import FakeOrchestrator
from jarvis_voice.macos.interactive_smoke import GuidedSmokeController, MacSmokeLoopbackServer
from jarvis_voice.protocol import EventFactory, new_uuid4
from jarvis_voice.voice_auth import VoiceAuthKey


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_macos_interactive_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_macos_interactive_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT_MODULE)


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
        failed = controller.report(runtime_error="smoke.runtime_failed")
        self.assertFalse(failed["ready"])
        self.assertEqual(failed["code"], "smoke.runtime_failed")
        self.assertFalse(controller.report()["ready"])

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
        for stage in interactive_smoke.CASE_ORDER:
            with self.subTest(stage=stage):
                controller = GuidedSmokeController(ptt_key="CONTROL_OPTION")
                controller.stage = stage
                controller.take_prompts()
                controller.feed(event("client_error_created", 1.0, code="audio.capture_overflow"))
                report = controller.report()
                self.assertTrue(controller.finished)
                self.assertEqual(report["code"], "smoke.client_error")
                self.assertEqual(report["cases"][stage]["runtime_code"], "audio.capture_overflow")

    def test_server_rejects_non_loopback_url(self) -> None:
        with self.assertRaisesRegex(BackendError, "smoke.non_loopback_server"):
            MacSmokeLoopbackServer(
                "ws://192.0.2.1:8765",
                VoiceAuthKey(1, b"k" * 32),
            )

    def test_server_can_lock_english_output_for_bilingual_asr(self) -> None:
        server = MacSmokeLoopbackServer(
            "ws://127.0.0.1:8765",
            VoiceAuthKey(1, b"k" * 32),
            response_locale="en-US",
        )

        self.assertEqual(server._response_profile("el"), ("en-US", False))
        self.assertEqual(server._response_profile("en"), ("en-US", False))

    def test_server_can_use_one_unambiguous_unit_for_bounded_batches(self) -> None:
        server = MacSmokeLoopbackServer(
            "ws://127.0.0.1:8765",
            VoiceAuthKey(1, b"k" * 32),
            response_locale="en-US",
            single_unit_responses=True,
        )

        self.assertEqual(
            server._selected_response_sentences(greek=False, long_response=True),
            ("The voice qualification turn is complete.",),
        )

    def test_server_rejects_unknown_response_locale(self) -> None:
        with self.assertRaisesRegex(BackendError, "smoke.response_locale_invalid"):
            MacSmokeLoopbackServer(
                "ws://127.0.0.1:8765",
                VoiceAuthKey(1, b"k" * 32),
                response_locale="fr-FR",
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


class InteractiveSmokeRuntimeTests(unittest.TestCase):
    def test_completed_cases_and_requested_shutdown_remain_ready(self) -> None:
        timings = [event("audio_stream_started", 0.0), event("ptt_source_started", 0.1)]
        recorder = SimpleNamespace(feed=timings.append)
        complete_voice(recorder, start=1.0, language="el")
        complete_voice(recorder, start=10.0, language="en")
        timings.extend([
            event("ptt_up_received", 20.0), event("input_no_speech_created", 20.1),
            event("transcript_final_created", 30.0, language="el"),
            event("tts_unit_started", 30.1), event("ptt_down_received", 30.2, output_active=True),
            event("tts_cancel_dispatched", 30.3), event("response_cancel_created", 30.4),
            event("playback_interrupted_created", 30.5), event("ptt_up_received", 30.6),
        ])
        stopped = threading.Event()
        runtime = mock.Mock(timing_snapshot=lambda: tuple(timings), tts=SimpleNamespace(is_idle=True))
        runtime.request_stop.side_effect = stopped.set

        def run():
            try:
                stopped.wait(timeout=2.0)
            finally:
                runtime.close()

        runtime.run = run
        server = mock.Mock()
        try:
            with (mock.patch.object(interactive_smoke, "configured_voice_auth_key", return_value=object()),
                  mock.patch.object(interactive_smoke, "write_report")):
                report = interactive_smoke.run_interactive_macos_smoke(
                    voice_config_from_mapping({}), output="unused.json",
                    runtime_factory=lambda *args: runtime, server_factory=lambda *args: server,
                    prompt_sink=lambda prompt: None,
                )
        finally:
            stopped.set()
        self.assertTrue(report["ready"])
        runtime.request_stop.assert_called_once_with()
        runtime.close.assert_called_once_with()
        server.stop.assert_called_once_with()

    def test_invalid_timeout_fails_before_runtime_or_server_construction(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf"), 0.0, -1.0, 3601.0):
            with self.subTest(value=value), mock.patch.object(
                interactive_smoke, "configured_voice_auth_key", side_effect=AssertionError("auth before validation"),
            ), self.assertRaisesRegex(BackendError, "smoke.timeout_invalid"):
                interactive_smoke.run_interactive_macos_smoke(
                    voice_config_from_mapping({}), output="unused.json", timeout_seconds=value,
                )

    def test_runtime_resources_close_across_start_and_stop_failures(self) -> None:
        thread_class = threading.Thread
        for failure in ("server_factory", "runtime_factory", "thread_factory", "thread_start",
                        "server_start", "request_stop", "runtime_close", "runtime_timeout", "server_stop", "prompt", "write"):
            with self.subTest(failure=failure):
                acquired = set()
                workers = []
                stopped = threading.Event()
                entered = threading.Event()

                def fail(stage):
                    if failure == stage:
                        raise BackendError("smoke.test_failure", "private failure details", subsystem="smoke")

                def release(name, stage):
                    acquired.discard(name)
                    fail(stage)

                server = mock.Mock(start=lambda: fail("server_start"), stop=lambda: release("server", "server_stop"))
                runtime = mock.Mock(close=lambda: release("runtime", "runtime_close"), timing_snapshot=lambda: ())

                def run():
                    entered.set()
                    try:
                        stopped.wait(timeout=2.0)
                    finally:
                        runtime.close()

                def stop():
                    if failure != "runtime_timeout":
                        stopped.set()
                    fail("request_stop")

                runtime.run = run
                runtime.request_stop = stop

                def make_server(*args):
                    fail("server_factory")
                    acquired.add("server")
                    return server

                def make_runtime(*args):
                    fail("runtime_factory")
                    acquired.add("runtime")
                    return runtime

                def make_thread(*args, **kwargs):
                    fail("thread_factory")
                    worker = thread_class(*args, **kwargs)
                    if failure == "thread_start":
                        worker.start = lambda: fail("thread_start")
                    workers.append(worker)
                    return worker

                def prompt(_prompt):
                    if not entered.wait(timeout=1.0):
                        raise AssertionError("fake runtime did not enter")
                    fail("prompt")
                    raise KeyboardInterrupt

                report = error = None
                try:
                    with (mock.patch.object(interactive_smoke, "configured_voice_auth_key", return_value=object()),
                          mock.patch.object(interactive_smoke.threading, "Thread", side_effect=make_thread),
                          mock.patch.object(interactive_smoke, "write_report", side_effect=lambda *args: fail("write"))):
                        report = interactive_smoke.run_interactive_macos_smoke(
                            voice_config_from_mapping({"application": {"shutdown_timeout_seconds": 0.01}}), output="unused.json",
                            runtime_factory=make_runtime, server_factory=make_server, prompt_sink=prompt,
                        )
                        if failure == "runtime_timeout":
                            self.assertIn("runtime", acquired)
                except BaseException as exc:
                    error = exc
                finally:
                    # Also stop/join owned workers when exercising the broken baseline.
                    stopped.set()
                    for worker in workers:
                        if worker.ident is not None:
                            worker.join(timeout=1.0)
                            self.assertFalse(worker.is_alive())
                self.assertEqual(acquired, set())
                if failure == "write":
                    self.assertIsInstance(error, BackendError)
                else:
                    self.assertIsNone(error)
                    self.assertFalse(report["ready"])
                    self.assertNotIn("private failure details", str(report))
                    if failure == "runtime_timeout":
                        self.assertEqual(report["code"], "smoke.runtime_shutdown_timeout")

    def test_unexpected_runtime_return_fails_without_waiting_for_interaction_timeout(self) -> None:
        class ExitingThread(threading.Thread):
            def start(self):
                super().start()
                self.join(timeout=1.0)
                if self.is_alive():
                    raise AssertionError("fake runtime did not exit")

        runtime = mock.Mock(timing_snapshot=lambda: ())
        with (mock.patch.object(interactive_smoke, "configured_voice_auth_key", return_value=object()),
              mock.patch.object(interactive_smoke.threading, "Thread", ExitingThread),
              mock.patch.object(interactive_smoke, "write_report"),
              mock.patch.object(interactive_smoke, "time", SimpleNamespace(
                  monotonic=lambda: 0.0, perf_counter=lambda: 0.0,
                  sleep=mock.Mock(side_effect=AssertionError("collector waited after runtime exit")),
              ))):
            report = interactive_smoke.run_interactive_macos_smoke(
                voice_config_from_mapping({}), output="unused.json",
                runtime_factory=lambda *args: runtime, server_factory=lambda *args: mock.Mock(),
                prompt_sink=lambda prompt: None,
            )
        self.assertFalse(report["ready"])
        self.assertEqual(report["code"], "smoke.runtime_exited")
        runtime.request_stop.assert_called_once_with()

    def test_loopback_server_owns_unstarted_and_late_threads(self) -> None:
        server = MacSmokeLoopbackServer("ws://127.0.0.1:8765", VoiceAuthKey(1, b"k" * 32))
        with self.subTest(stage="unstarted"):
            unstarted = mock.Mock(ident=None, is_alive=lambda: False)
            unstarted.start.side_effect = RuntimeError("thread creation failed")
            with mock.patch.object(interactive_smoke.threading, "Thread", return_value=unstarted):
                with self.assertRaisesRegex(RuntimeError, "thread creation failed"):
                    server.start()
            unstarted.join.assert_not_called()
            self.assertIsNone(server._thread)

        with self.subTest(stage="late"):
            late = mock.Mock(ident=123)
            late.is_alive.return_value = True
            server._thread = late
            with self.assertRaisesRegex(BackendError, "smoke.server_stop_timeout"):
                server.stop()
            self.assertIs(server._thread, late)
            late.is_alive.return_value = False
            server.stop()
            self.assertIsNone(server._thread)

        with self.subTest(stage="restart"):
            # A completed stop must not poison a later start with stale flags/error.
            server._thread = None
            server._ready.set()
            server._error = RuntimeError("old startup failure")
            restarted = mock.Mock(ident=124, is_alive=lambda: False)

            def start():
                self.assertFalse(server._stop.is_set())
                self.assertFalse(server._ready.is_set())
                self.assertIsNone(server._error)
                server._ready.set()

            restarted.start.side_effect = start
            with mock.patch.object(interactive_smoke.threading, "Thread", return_value=restarted):
                server.start()
            server.stop()
            self.assertIsNone(server._thread)


class InteractiveSmokeScriptTests(unittest.TestCase):
    def test_blocked_report_write_failure_keeps_safe_json_and_exit_status(self) -> None:
        for code in ("smoke.test_failure", "qualification.output_directory_missing"):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "missing" / "evidence.json"
                stdout = io.StringIO()
                with (mock.patch.object(SCRIPT_MODULE, "load_voice_config", return_value=object()),
                      mock.patch.object(SCRIPT_MODULE, "with_faster_whisper_cpu_smoke", return_value=object()),
                      mock.patch.object(SCRIPT_MODULE, "run_interactive_macos_smoke",
                                        side_effect=BackendError(code, "private failure details", subsystem="smoke")),
                      contextlib.redirect_stdout(stdout)):
                    status = SCRIPT_MODULE.main([
                        "--config", "unused.json", "--model-dir", "unused", "--output", str(output),
                    ])
                report = json.loads(stdout.getvalue())
                self.assertEqual(status, 2)
                self.assertFalse(report["ready"])
                self.assertEqual(report["code"], code)
                self.assertEqual(report["output_error_code"], "qualification.output_directory_missing")
                self.assertNotIn("private failure details", stdout.getvalue())
                self.assertFalse(output.exists())

    def test_script_requires_private_inputs_and_exposes_timeout(self) -> None:
        parsed = SCRIPT_MODULE.build_parser().parse_args(
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
