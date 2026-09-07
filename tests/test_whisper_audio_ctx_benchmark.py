"""Run the benchmark report path with real fixture files and fake inference."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_macos_whisper_audio_ctx.py"
)
SPEC = importlib.util.spec_from_file_location("audio_ctx_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WhisperAudioContextBenchmarkTests(unittest.TestCase):
    def test_each_fixture_retains_its_audio_and_baseline_with_duplicate_basenames(self):
        for same_basename in (False, True):
            with (
                self.subTest(same_basename=same_basename),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                audio = root / "audio"
                names = [
                    "left/sample.wav",
                    "right/sample.wav" if same_basename else "right/other.wav",
                ]
                references = dict(zip(names, ("alpha", "beta")))
                for name in names:
                    path = audio / name
                    path.parent.mkdir(parents=True)
                    with wave.open(str(path), "wb") as wav:
                        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        wav.writeframes(b"\x00\x00" * 32000)
                metadata = root / "metadata.csv"
                metadata.write_text(
                    "".join(f"{name}|{text}\n" for name, text in references.items())
                )
                server, model, output = (
                    root / name for name in ("server", "model", "report.json")
                )
                server.touch()
                model.touch()
                argv = [
                    str(SCRIPT),
                    "--server",
                    str(server),
                    "--model",
                    str(model),
                    "--metadata",
                    str(metadata),
                    "--audio-root",
                    str(audio),
                    "--sample-count",
                    "2",
                    "--audio-contexts",
                    "0",
                    "384",
                    "--output",
                    str(output),
                ]
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(
                        MODULE,
                        "_resample",
                        side_effect=lambda path, _ffmpeg: references[
                            path.relative_to(audio).as_posix()
                        ].encode(),
                    ),
                    patch.object(
                        MODULE,
                        "_request",
                        side_effect=lambda _port, _route, wav, _context: (
                            0.1,
                            wav.decode(),
                            "en",
                        ),
                    ),
                    patch.object(MODULE, "_available_port", return_value=8766),
                    patch.object(MODULE, "_wait_ready"),
                    patch.object(MODULE.subprocess, "Popen") as process,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(MODULE.main(), 0)
                process.return_value.terminate.assert_called_once()
                process.return_value.wait.assert_called_once_with(timeout=2.0)
                report = json.loads(output.read_text())
                self.assertEqual(len(report["records"]), 4)
                self.assertEqual(
                    len({row["fixture"] for row in report["records"]}), 2
                )
                for row in report["records"]:
                    self.assertEqual(row["text"], row["reference"])
                    self.assertEqual(row["word_errors"], 0)
                for summary in report["summary"].values():
                    self.assertEqual(summary["count"], 2)
                    self.assertEqual(summary["word_error_rate"], 0)
                    self.assertEqual(summary["changed_from_full_context"], 0)
                    self.assertEqual(summary["degraded_from_full_context"], 0)


if __name__ == "__main__":
    unittest.main()
