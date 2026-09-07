from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from array import array
from dataclasses import replace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.backends import AudioChunk, LanguageMetadata, VadResult
from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.errors import BackendError
from jarvis_voice.fakes import FakeAsrBackend, FakeTtsBackend
from jarvis_voice.macos import automated_smoke
from jarvis_voice.macos.automated_smoke import (
    CASE_ORDER,
    MacSpeechFixtureSynthesizer,
    _base_report,
    read_pcm16_wav,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_macos_automated_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_macos_automated_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT_MODULE)


def write_wav(path: Path, *, sample_rate: int = 16_000, channels: int = 1, width: int = 2) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x00\x00" * 1600 * channels)


class FixtureAudioTests(unittest.TestCase):
    def test_pcm16_loader_is_strict_and_normalizes_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            valid = Path(temporary) / "fixture.wav"
            with wave.open(str(valid), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16_000)
                stream.writeframes(b"\x00\x80\x00\x00\xff\x7f")
            audio = read_pcm16_wav(valid)
            self.assertEqual(audio.sample_rate, 16_000)
            self.assertEqual(len(audio.samples), 3)
            self.assertAlmostEqual(audio.samples[0], -1.0)
            self.assertAlmostEqual(audio.samples[2], 32767 / 32768)

            wrong = Path(temporary) / "stereo.wav"
            write_wav(wrong, channels=2)
            with self.assertRaisesRegex(BackendError, "smoke.fixture_format_invalid"):
                read_pcm16_wav(wrong)

    def test_synthesis_uses_stdin_no_shell_and_deletes_temporary_files(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []
        generated: list[Path] = []

        def runner(argv, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((list(argv), dict(kwargs)))
            if argv[0] == "/usr/bin/say":
                output = Path(argv[argv.index("-o") + 1])
                output.write_bytes(b"fixed-aiff")
                generated.append(output)
            else:
                output = Path(argv[2])
                write_wav(output)
                generated.append(output)
            return subprocess.CompletedProcess(argv, 0)

        audio = MacSpeechFixtureSynthesizer(process_runner=runner).synthesize(locale="el-GR", voice="TestVoice")
        self.assertEqual(audio.sample_rate, 16_000)
        self.assertEqual(len(calls), 2)
        fixture_text = calls[0][1]["input"]
        self.assertIsInstance(fixture_text, bytes)
        self.assertNotIn(fixture_text.decode("utf-8"), " ".join(calls[0][0]))
        self.assertFalse(calls[0][1]["shell"])
        self.assertFalse(calls[1][1]["shell"])
        self.assertTrue(all(not path.exists() for path in generated))


class AutomatedEvidenceTests(unittest.TestCase):
    def test_startup_and_teardown_close_all_acquired_backends(self) -> None:
        core_class = automated_smoke.VoiceCore
        orchestrator_class = automated_smoke.FakeOrchestrator
        for failure in ("tts_factory", "asr_factory", "core_factory", "vad_factory", "orchestrator_factory",
                        "fixture", "asr_close", "tts_close", "asr_close_interrupt", "write"):
            with self.subTest(failure=failure):
                acquired = set()

                def fail(stage):
                    if failure == stage:
                        raise BackendError("smoke.test_failure", "private failure details", subsystem="smoke")

                def close(name):
                    acquired.remove(name)
                    if name == "asr" and failure == "asr_close_interrupt":
                        raise KeyboardInterrupt
                    fail(name + "_close")

                tts = mock.Mock(close=lambda: close("tts"))
                asr = mock.Mock(close=lambda: close("asr"))

                def create(name, backend):
                    fail(name + "_factory")
                    acquired.add(name)
                    return backend

                def make_core(**kwargs):
                    fail("core_factory")
                    return core_class(**kwargs)

                def make_orchestrator(**kwargs):
                    fail("orchestrator_factory")
                    return orchestrator_class(**kwargs)

                report = error = None
                try:
                    with (mock.patch.object(automated_smoke, "MacSystemTtsBackend", side_effect=lambda **kwargs: create("tts", tts)),
                          mock.patch.object(automated_smoke, "create_macos_asr_backend", side_effect=lambda config: create("asr", asr)),
                          mock.patch.object(automated_smoke, "VoiceCore", side_effect=make_core),
                          mock.patch.object(automated_smoke, "_vad_from_config", side_effect=lambda config: fail("vad_factory")),
                          mock.patch.object(automated_smoke, "FakeOrchestrator", side_effect=make_orchestrator),
                          mock.patch.object(automated_smoke, "write_report", side_effect=lambda *args: fail("write"))):
                        fixtures = mock.Mock()
                        fixtures.synthesize.side_effect = BackendError("smoke.fixture_unavailable", "private fixture details", subsystem="smoke")
                        report = automated_smoke.run_automated_macos_smoke(
                            voice_config_from_mapping({}), output="unused.json", synthesizer=fixtures,
                        )
                except BaseException as exc:
                    error = exc
                self.assertEqual(acquired, set())
                if failure == "asr_close_interrupt":
                    self.assertIsInstance(error, KeyboardInterrupt)
                elif failure == "write":
                    self.assertIsInstance(error, BackendError)
                else:
                    self.assertIsNone(error)
                    self.assertFalse(report["ready"])
                    self.assertNotIn("private", str(report))

    def test_complete_synthetic_run_closes_asr_and_rejects_cancel_failure(self) -> None:
        for cancel_failure in (False, True):
            with self.subTest(cancel_failure=cancel_failure):
                class Tts(FakeTtsBackend):
                    last_error = None
                    close = mock.Mock()

                    def __init__(self, *, on_started, on_completed, **kwargs):
                        super().__init__()
                        self.started = on_started
                        self.completed_callback = on_completed

                    def enqueue(self, unit):
                        super().enqueue(unit)
                        if self.active is None:
                            self.started(self.start_next())
                            if unit.text not in automated_smoke.BARGE_TEXT:
                                self.completed_callback(self.complete_active())

                    def cancel(self, epoch, reason):
                        active = self.active is not None
                        super().cancel(epoch, reason)
                        if active and cancel_failure:
                            self.last_error = BackendError("tts.say_failed", "private playback failure", subsystem="tts")

                    def wait_idle(self, timeout):
                        return self.is_idle

                greek = FakeAsrBackend().result
                asr = mock.Mock()
                asr.transcribe.side_effect = [greek, replace(greek, language=LanguageMetadata("en", 0.97))]
                vad = mock.Mock(trim=lambda audio: VadResult(any(audio.samples), audio))
                fixtures = mock.Mock(synthesize=lambda **kwargs: AudioChunk(array("f", [0.5] * 1600), 16000))
                with (mock.patch.object(automated_smoke, "MacSystemTtsBackend", side_effect=Tts),
                      mock.patch.object(automated_smoke, "create_macos_asr_backend", return_value=asr),
                      mock.patch.object(automated_smoke, "_vad_from_config", return_value=vad),
                      mock.patch.object(automated_smoke, "write_report")):
                    report = automated_smoke.run_automated_macos_smoke(
                        voice_config_from_mapping({}), output="unused.json", synthesizer=fixtures,
                    )
                self.assertEqual(report["ready"], not cancel_failure)
                self.assertEqual(report["code"], "tts.say_failed" if cancel_failure else "smoke.macos_automated_fixture_ready")
                self.assertEqual(asr.transcribe.call_count, 2)
                asr.close.assert_called_once_with()
                Tts.close.assert_called_once_with()

    def test_cli_returns_safe_failure_when_blocked_report_cannot_be_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "missing" / "evidence.json"
            stdout = io.StringIO()
            with (mock.patch.object(SCRIPT_MODULE, "load_voice_config", return_value=object()),
                  mock.patch.object(SCRIPT_MODULE, "with_faster_whisper_cpu_smoke", return_value=object()),
                  mock.patch.object(SCRIPT_MODULE, "run_automated_macos_smoke",
                                    side_effect=BackendError("smoke.test_failure", "private failure details", subsystem="smoke")),
                  contextlib.redirect_stdout(stdout)):
                status = SCRIPT_MODULE.main(["--config", "unused.json", "--model-dir", "unused", "--output", str(output)])
            report = json.loads(stdout.getvalue())
            self.assertEqual(status, 2)
            self.assertEqual(report["code"], "smoke.test_failure")
            self.assertEqual(report["output_error_code"], "qualification.output_directory_missing")
            self.assertNotIn("private", stdout.getvalue())
            self.assertFalse(output.exists())

    def test_report_is_explicitly_not_interactive_and_content_free(self) -> None:
        cases = {name: {"status": "ready", "metrics": {}} for name in CASE_ORDER}
        report = _base_report(cases, ready=True, code="smoke.ready", tts_exercised=True)
        self.assertTrue(report["native"]["fixture_injected"])  # type: ignore[index]
        self.assertFalse(report["native"]["microphone_opened"])  # type: ignore[index]
        self.assertFalse(report["native"]["physical_ptt_exercised"])  # type: ignore[index]
        self.assertFalse(report["native"]["satisfies_interactive_m5e"])  # type: ignore[index]
        encoded = json.dumps(report)
        for forbidden in ("/Users/", "event_id", "turn_id", "response_id", "sha256", "Καλημέρα", "Good morning"):
            self.assertNotIn(forbidden, encoded)

    def test_cli_requires_private_config_model_and_output(self) -> None:
        parsed = SCRIPT_MODULE.build_parser().parse_args(
            ["--config", "/private/voice.json", "--model-dir", "/private/model", "--output", "/private/evidence.json"]
        )
        self.assertEqual(parsed.model_dir, "/private/model")


if __name__ == "__main__":
    unittest.main()
