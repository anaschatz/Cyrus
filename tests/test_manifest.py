from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.manifest import (
    ModelFileSpec,
    ModelManifest,
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


if __name__ == "__main__":
    unittest.main()
