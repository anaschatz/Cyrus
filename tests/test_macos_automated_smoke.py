from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.automated_smoke import (
    CASE_ORDER,
    MacSpeechFixtureSynthesizer,
    _base_report,
    read_pcm16_wav,
)


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
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_macos_automated_smoke.py"
        spec = importlib.util.spec_from_file_location("run_macos_automated_smoke", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parsed = module.build_parser().parse_args(
            ["--config", "/private/voice.json", "--model-dir", "/private/model", "--output", "/private/evidence.json"]
        )
        self.assertEqual(parsed.model_dir, "/private/model")


if __name__ == "__main__":
    unittest.main()
