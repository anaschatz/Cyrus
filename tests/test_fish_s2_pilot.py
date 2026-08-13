from __future__ import annotations

import json
import socket
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos import fish_s2_pilot as pilot


class FishS2PilotTests(unittest.TestCase):
    def test_memory_gate_accounts_for_unified_mlx_allocations(self) -> None:
        self.assertEqual(pilot.MAX_STANDALONE_MLX_BYTES, 7 * 1024**3)

    def test_network_guard_fails_closed_and_restores_socket(self) -> None:
        original_socket = socket.socket

        @pilot._deny_network
        def guarded() -> dict[str, object]:
            with socket.socket() as stream:
                with self.assertRaises(BackendError) as raised:
                    stream.connect(("127.0.0.1", 9))
            self.assertEqual(raised.exception.code, "tts.fish_s2.network_forbidden")
            return {"ready": True}

        self.assertTrue(guarded()["ready"])
        self.assertIs(socket.socket, original_socket)

    def test_model_validation_requires_hashes_and_quantization(self) -> None:
        specs = (
            pilot.ArtifactSpec("config.json", 99, "config-hash"),
            pilot.ArtifactSpec("model.safetensors", 7, "model-hash"),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = json.dumps(
                {
                    "model_type": "fish_qwen3_omni",
                    "quantization": {"bits": 4, "group_size": 64},
                }
            ).ljust(99)
            (root / "config.json").write_text(config)
            (root / "model.safetensors").write_bytes(b"weights")
            hashes = {
                "config.json": "config-hash",
                "model.safetensors": "model-hash",
            }
            with (
                mock.patch.object(pilot, "MODEL_ARTIFACTS", specs),
                mock.patch.object(pilot, "_sha256", side_effect=lambda path: hashes[path.name]),
            ):
                result = pilot.validate_model_directory(root)
        self.assertTrue(result.ready)

    def test_reference_requires_exact_audio_profile_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "f1.wav"
            with wave.open(str(reference), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(44_100)
                stream.writeframes(b"\0\0" * 441_000)
            with mock.patch.object(pilot, "_sha256", return_value=pilot.F1_REFERENCE_SHA256):
                self.assertEqual(pilot.validate_reference_audio(reference), reference)
            with mock.patch.object(pilot, "_sha256", return_value="wrong"):
                with self.assertRaises(BackendError):
                    pilot.validate_reference_audio(reference)

    def test_fixture_selection_is_bounded_and_unique(self) -> None:
        self.assertEqual(pilot._selected_fixtures(["greeting"])[0][0], "greeting")
        with self.assertRaises(BackendError):
            pilot._selected_fixtures(["greeting", "greeting"])
        with self.assertRaises(BackendError):
            pilot._selected_fixtures(["unknown"])

    def test_transcript_score_has_strict_near_perfect_gate(self) -> None:
        report = pilot.score_transcripts(
            {"greeting": "Καλημέρα. Είμαι έτοιμη να σε βοηθήσω."},
            fixture_ids=["greeting"],
        )
        self.assertEqual(report["micro_wer_percent"], 0.0)
        self.assertTrue(report["pronunciation_gate_passed"])
        self.assertFalse(report["production_promotion_allowed"])

    def test_incomplete_transcripts_fail_closed(self) -> None:
        with self.assertRaises(BackendError) as raised:
            pilot.score_transcripts({}, fixture_ids=["greeting"])
        self.assertEqual(raised.exception.code, "tts.fish_s2.transcripts_incomplete")


if __name__ == "__main__":
    unittest.main()
