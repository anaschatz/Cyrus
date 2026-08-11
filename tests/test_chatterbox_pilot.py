from __future__ import annotations

import hashlib
import json
import socket
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.chatterbox_pilot import (
    MINIMUM_ACQUISITION_FREE_BYTES,
    MINIMUM_RUN_FREE_BYTES,
    MODEL_ARTIFACTS,
    MODEL_DOWNLOAD_BYTES,
    TOKENIZER_FILENAME,
    _deny_network_during_pilot,
    _validated_audio_path,
    chatterbox_pilot_preflight,
    validate_chatterbox_model_directory,
)


class ChatterboxPilotTests(unittest.TestCase):
    def test_network_guard_fails_closed_and_restores_socket(self) -> None:
        original_socket = socket.socket

        @_deny_network_during_pilot
        def guarded() -> dict[str, object]:
            self.assertIsNot(socket.socket, original_socket)
            with socket.socket() as stream:
                with self.assertRaises(BackendError) as raised:
                    stream.connect(("127.0.0.1", 9))
            self.assertEqual(raised.exception.code, "tts.chatterbox.network_forbidden")
            return {"ready": True}

        self.assertTrue(guarded()["ready"])
        self.assertIs(socket.socket, original_socket)

    def test_low_disk_blocks_acquisition_without_network_or_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = chatterbox_pilot_preflight(
                temp,
                Path(temp) / "missing-model",
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 11, 14),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=4 * 1024**3),
                module_probe=lambda name: False,
                source_revision_probe=lambda: False,
            )
        self.assertFalse(report["ready"])
        self.assertFalse(report["acquisition_ready"])
        self.assertEqual(report["code"], "tts.chatterbox.disk_space_insufficient")
        self.assertFalse(report["network_used"])
        self.assertFalse(report["installation_performed"])
        self.assertEqual(report["disk"]["acquisition_required_mib"], MINIMUM_ACQUISITION_FREE_BYTES // 1024**2)

    def test_host_gate_requires_darwin_arm64_python_311(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = chatterbox_pilot_preflight(
                temp,
                Path(temp) / "model",
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 14, 0),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=16 * 1024**3),
                module_probe=lambda name: True,
                source_revision_probe=lambda: True,
            )
        self.assertEqual(report["code"], "tts.chatterbox.host_unsupported")
        self.assertFalse(report["acquisition_ready"])

    def test_installed_dependencies_reduce_acquisition_gate_to_model_plus_reserve(self) -> None:
        required = MODEL_DOWNLOAD_BYTES + MINIMUM_RUN_FREE_BYTES
        with tempfile.TemporaryDirectory() as temp:
            report = chatterbox_pilot_preflight(
                temp,
                Path(temp) / "model",
                platform_name="darwin",
                machine="arm64",
                python_version=(3, 11, 14),
                pointer_bits=64,
                disk_usage=lambda path: SimpleNamespace(free=required),
                module_probe=lambda name: True,
                source_revision_probe=lambda: True,
            )
        self.assertTrue(report["acquisition_ready"])
        self.assertEqual(report["disk"]["acquisition_required_mib"], required // 1024**2)

    def test_model_validation_is_strict_and_tokenizer_requires_greek(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for artifact in MODEL_ARTIFACTS:
                path = root / artifact.filename
                if artifact.filename != TOKENIZER_FILENAME:
                    path.write_bytes(b"fixture")
            tokenizer = json.dumps({"added_tokens": [{"content": "[el]"}]})
            (root / TOKENIZER_FILENAME).write_text(tokenizer.ljust(69_989))
            with mock.patch("jarvis_voice.macos.chatterbox_pilot._sha256", return_value="wrong"):
                result = validate_chatterbox_model_directory(root)
        self.assertFalse(result.ready)
        self.assertEqual(result.code, "tts.chatterbox.model_artifact_invalid")

    def test_model_validation_accepts_pinned_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes: dict[str, str] = {}
            for artifact in MODEL_ARTIFACTS:
                path = root / artifact.filename
                if artifact.filename == TOKENIZER_FILENAME:
                    tokenizer = json.dumps({"added_tokens": [{"content": "[el]"}]})
                    path.write_text(tokenizer.ljust(artifact.size))
                else:
                    with path.open("wb") as stream:
                        stream.truncate(artifact.size)
                hashes[str(path)] = artifact.sha256

            def fake_hash(path: Path) -> str:
                return hashes[str(path)]

            with mock.patch("jarvis_voice.macos.chatterbox_pilot._sha256", side_effect=fake_hash):
                result = validate_chatterbox_model_directory(root)
        self.assertTrue(result.ready)
        self.assertEqual(result.files_present, len(MODEL_ARTIFACTS))

    def test_voice_conversion_audio_validation_rejects_unsafe_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            valid = root / "reference.wav"
            with wave.open(str(valid), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(24_000)
                stream.writeframes(b"\0\0")
            self.assertEqual(_validated_audio_path(valid, role="target"), valid)
            invalid = root / "reference.txt"
            invalid.write_text("not audio")
            with self.assertRaises(BackendError) as raised:
                _validated_audio_path(invalid, role="target")
        self.assertEqual(raised.exception.code, "tts.chatterbox.target_audio_invalid")

    def test_voice_conversion_audio_validation_rejects_empty_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "empty.wav"
            with wave.open(str(empty), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(24_000)
            with self.assertRaises(BackendError) as raised:
                _validated_audio_path(empty, role="target")
        self.assertEqual(raised.exception.code, "tts.chatterbox.target_audio_invalid")


if __name__ == "__main__":
    unittest.main()
