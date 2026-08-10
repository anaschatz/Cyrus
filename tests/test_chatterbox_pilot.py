from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.macos.chatterbox_pilot import (
    MINIMUM_ACQUISITION_FREE_BYTES,
    MODEL_ARTIFACTS,
    TOKENIZER_FILENAME,
    chatterbox_pilot_preflight,
    validate_chatterbox_model_directory,
)


class ChatterboxPilotTests(unittest.TestCase):
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
            )
        self.assertEqual(report["code"], "tts.chatterbox.host_unsupported")
        self.assertFalse(report["acquisition_ready"])

    def test_model_validation_is_strict_and_tokenizer_requires_greek(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for artifact in MODEL_ARTIFACTS:
                path = root / artifact.filename
                path.write_bytes(b"fixture")
            (root / TOKENIZER_FILENAME).write_text(json.dumps({"added_tokens": [{"content": "[el]"}]}))
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
                with path.open("wb") as stream:
                    stream.truncate(artifact.size)
                hashes[str(path)] = artifact.sha256
            (root / TOKENIZER_FILENAME).write_text(json.dumps({"added_tokens": [{"content": "[el]"}]}))

            def fake_hash(path: Path) -> str:
                return hashes[str(path)]

            with mock.patch("jarvis_voice.macos.chatterbox_pilot._sha256", side_effect=fake_hash):
                result = validate_chatterbox_model_directory(root)
        self.assertTrue(result.ready)
        self.assertEqual(result.files_present, len(MODEL_ARTIFACTS) + 1)


if __name__ == "__main__":
    unittest.main()
