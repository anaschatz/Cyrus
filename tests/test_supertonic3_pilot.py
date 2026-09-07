from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos import supertonic3_pilot as supertonic
from jarvis_voice.macos.supertonic3_pilot import (
    DEFAULT_SPEED,
    DEFAULT_TOTAL_STEPS,
    DEFAULT_VOICE,
    MAX_ASR_MICRO_WER_PERCENT,
    MODEL_ARTIFACTS,
    MODEL_REVISION,
    QUALITY_FIXTURES,
    SAMPLE_RATE,
    _deny_network_during_pilot,
    _normalized_words,
    score_supertonic3_transcripts,
    supertonic3_pilot_preflight,
    validate_supertonic3_model_directory,
)


class Supertonic3PilotTests(unittest.TestCase):
    def test_invalid_settings_are_rejected_before_runtime_or_output_creation(self) -> None:
        settings = (
            {"speed": float("nan")}, {"speed": True}, {"speed": "1"},
            {"total_steps": True}, {"total_steps": 1.5},
            {"seed": -1}, {"seed": True}, {"seed": 1.5},
            {"seed": 2**32 - len(QUALITY_FIXTURES)},
        )
        for options in settings:
            with self.subTest(options=options), mock.patch.object(supertonic, "validate_supertonic3_model_directory", return_value=SimpleNamespace(ready=True)):
                with mock.patch.object(supertonic, "_prepare_output_directory", side_effect=AssertionError("invalid settings reached output creation")):
                    with self.assertRaisesRegex(BackendError, "tts.supertonic3.settings_invalid"):
                        supertonic.run_supertonic3_quality_qualification("unused", "unused", **options)

    def test_audio_duration_and_shape_must_be_valid_before_saving(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is an optional runtime dependency")
        valid_wave = np.zeros((1, SAMPLE_RATE), dtype=np.float32)
        cases = (
            (valid_wave, [1.0], True),
            (valid_wave, [-1.0], False), (valid_wave, [0.0], False),
            (valid_wave, [float("nan")], False), (valid_wave, [float("inf")], False),
            (valid_wave, [], False), (valid_wave, ["invalid"], False),
            (np.array(0.0), [1.0], False), (np.zeros(1), [1.0], False),
        )
        for waveform, duration, valid in cases:
            with self.subTest(duration=duration, shape=waveform.shape), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "output"
                tts = mock.Mock(sample_rate=SAMPLE_RATE, voice_style_names=supertonic.VOICE_NAMES)
                tts.synthesize.return_value = (waveform, duration)
                tts.save_audio.side_effect = lambda waveform, path: Path(path).write_bytes(b"synthetic waveform")
                with (mock.patch.object(supertonic, "validate_supertonic3_model_directory", return_value=SimpleNamespace(ready=True)),
                      mock.patch.dict("sys.modules", {"supertonic": SimpleNamespace(TTS=mock.Mock(return_value=tts))}),
                      mock.patch.object(supertonic.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=1)),
                      mock.patch.object(np.random, "seed")):
                    if valid:
                        report = supertonic.run_supertonic3_quality_qualification("unused", output)
                        self.assertTrue(report["ready"])
                        self.assertEqual(tts.save_audio.call_count, len(QUALITY_FIXTURES))
                        self.assertFalse(report["production_promotion_allowed"])
                    else:
                        with self.assertRaisesRegex(BackendError, "tts.supertonic3.audio_invalid"):
                            supertonic.run_supertonic3_quality_qualification("unused", output)
                        tts.save_audio.assert_not_called()
                        self.assertFalse(list(output.iterdir()))

    def test_defaults_are_the_qualified_greek_profile(self) -> None:
        self.assertEqual(DEFAULT_VOICE, "F5")
        self.assertEqual(DEFAULT_TOTAL_STEPS, 12)
        self.assertEqual(DEFAULT_SPEED, 1.0)
        self.assertEqual(SAMPLE_RATE, 44_100)
        self.assertEqual(len(QUALITY_FIXTURES), 20)
        self.assertEqual(len({fixture_id for fixture_id, _ in QUALITY_FIXTURES}), 20)
        self.assertEqual(len(MODEL_REVISION), 40)

    def test_network_guard_fails_closed_and_restores_socket(self) -> None:
        original_socket = socket.socket

        @_deny_network_during_pilot
        def guarded() -> dict[str, object]:
            with socket.socket() as stream:
                with self.assertRaises(BackendError) as raised:
                    stream.connect(("127.0.0.1", 9))
            self.assertEqual(raised.exception.code, "tts.supertonic3.network_forbidden")
            return {"ready": True}

        self.assertTrue(guarded()["ready"])
        self.assertIs(socket.socket, original_socket)

    def test_preflight_requires_native_arm64_python_311(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = supertonic3_pilot_preflight(
                Path(temp) / "model",
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 14, 0),
                pointer_bits=64,
                dependency_probe=lambda name, version: True,
            )
        self.assertFalse(report["ready"])
        self.assertEqual(report["code"], "tts.supertonic3.host_unsupported")
        self.assertFalse(report["network_used"])
        self.assertFalse(report["installation_performed"])

    def test_model_validation_accepts_exact_pinned_artifacts_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes: dict[str, str] = {}
            for artifact in MODEL_ARTIFACTS:
                path = root / artifact.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as stream:
                    stream.truncate(artifact.size)
                hashes[str(path)] = artifact.sha256
            (root / "config.json").write_text(
                json.dumps({"model_name": "Supertonic 3", "model_type": "onnx"}).ljust(
                    MODEL_ARTIFACTS[1].size
                )
            )
            tts_artifact = next(item for item in MODEL_ARTIFACTS if item.relative_path == "onnx/tts.json")
            (root / "onnx" / "tts.json").write_text(
                json.dumps(
                    {
                        "tts_version": "v1.7.3",
                        "split": "opensource-multilingual",
                        "ae": {"sample_rate": SAMPLE_RATE},
                    }
                ).ljust(tts_artifact.size)
            )
            with mock.patch(
                "jarvis_voice.macos.supertonic3_pilot._sha256",
                side_effect=lambda path: hashes[str(path)],
            ):
                result = validate_supertonic3_model_directory(root)
        self.assertTrue(result.ready)
        self.assertEqual(result.files_present, len(MODEL_ARTIFACTS))

    def test_model_validation_rejects_symlink_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.mkdir(exist_ok=True)
            result = validate_supertonic3_model_directory(root)
        self.assertFalse(result.ready)
        self.assertEqual(result.code, "tts.supertonic3.model_artifact_invalid")

    def test_transcript_scoring_requires_complete_fixed_suite(self) -> None:
        with self.assertRaises(BackendError) as raised:
            score_supertonic3_transcripts({})
        self.assertEqual(raised.exception.code, "tts.supertonic3.transcripts_incomplete")

    def test_exact_transcripts_pass_near_perfect_gate(self) -> None:
        transcripts = {fixture_id: text for fixture_id, text in QUALITY_FIXTURES}
        report = score_supertonic3_transcripts(transcripts)
        self.assertEqual(report["micro_wer_percent"], 0.0)
        self.assertTrue(report["near_perfect_pronunciation_gate"])
        self.assertEqual(report["near_perfect_pronunciation_limit_percent"], MAX_ASR_MICRO_WER_PERCENT)
        self.assertFalse(report["production_promotion_allowed"])

    def test_digit_normalization_avoids_false_number_error(self) -> None:
        self.assertEqual(_normalized_words("στις 8"), ["στισ", "οκτώ"])


if __name__ == "__main__":
    unittest.main()
