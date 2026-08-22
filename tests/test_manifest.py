from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.manifest import (
    LARGE_V3_TURBO_MODEL_MANIFEST,
    MODEL_PROFILE_LARGE_V3_TURBO,
    ModelFileSpec,
    ModelManifest,
    model_manifest_for_profile,
    validate_model_directory,
)


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.contents = {"config.json": b"config", "model.bin": b"model"}
        self.manifest = ModelManifest(
            model_id="test/model",
            revision="pinned",
            engine="fake",
            files=tuple(
                ModelFileSpec(name, len(content), hashlib.sha256(content).hexdigest())
                for name, content in self.contents.items()
            ),
        )

    def write_valid_pack(self) -> None:
        for name, content in self.contents.items():
            (self.directory / name).write_bytes(content)

    def test_success_and_symlink_contents(self) -> None:
        blob = self.directory / ".model-blob"
        blob.write_bytes(self.contents["model.bin"])
        (self.directory / "config.json").write_bytes(self.contents["config.json"])
        (self.directory / "model.bin").symlink_to(blob)
        result = validate_model_directory(self.directory, self.manifest)
        self.assertTrue(result.ok)
        self.assertEqual(result.files_checked, 2)

    def test_missing_file(self) -> None:
        (self.directory / "config.json").write_bytes(self.contents["config.json"])
        result = validate_model_directory(self.directory, self.manifest)
        self.assertEqual(result.code, "model.missing_artifact")

    def test_unexpected_artifact(self) -> None:
        self.write_valid_pack()
        (self.directory / "extra.bin").write_bytes(b"extra")
        result = validate_model_directory(self.directory, self.manifest)
        self.assertEqual(result.code, "model.unexpected_artifact")

    def test_wrong_size(self) -> None:
        self.write_valid_pack()
        (self.directory / "model.bin").write_bytes(b"different-size")
        result = validate_model_directory(self.directory, self.manifest)
        self.assertEqual(result.code, "model.size_mismatch")

    def test_wrong_hash_with_same_size(self) -> None:
        self.write_valid_pack()
        (self.directory / "model.bin").write_bytes(b"xxxxx")
        result = validate_model_directory(self.directory, self.manifest)
        self.assertEqual(result.code, "model.hash_mismatch")

    def test_explicit_directory_is_required(self) -> None:
        result = validate_model_directory(None, self.manifest)
        self.assertEqual(result.code, "model.directory_required")

    def test_large_v3_turbo_manifest_is_exact_and_profile_selected(self) -> None:
        manifest = model_manifest_for_profile(MODEL_PROFILE_LARGE_V3_TURBO)
        self.assertIs(manifest, LARGE_V3_TURBO_MODEL_MANIFEST)
        self.assertEqual(
            manifest.revision,
            "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        )
        self.assertEqual(
            tuple(item.name for item in manifest.files),
            (
                "config.json",
                "model.bin",
                "preprocessor_config.json",
                "tokenizer.json",
                "vocabulary.json",
            ),
        )
        self.assertEqual(sum(item.size for item in manifest.files), 1_621_665_983)
        self.assertEqual(
            next(item.sha256 for item in manifest.files if item.name == "model.bin"),
            "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
        )

    def test_unknown_model_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Whisper"):
            model_manifest_for_profile("automatic-download")


if __name__ == "__main__":
    unittest.main()
