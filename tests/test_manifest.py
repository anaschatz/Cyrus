from __future__ import annotations

import hashlib
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from jarvis_voice.manifest import (
    LARGE_V3_TURBO_MODEL_MANIFEST,
    MODEL_PROFILE_LARGE_V3_TURBO,
    ModelFileSpec,
    ModelManifest,
    model_manifest_for_profile,
    _is_local_windows_absolute,
    _is_reparse,
    _windows_drive_is_local,
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

    def test_strict_windows_pack_rejects_symlinked_members(self) -> None:
        pack = self.directory / "pack"
        pack.mkdir()
        blob = self.directory / "model-blob"
        blob.write_bytes(self.contents["model.bin"])
        (pack / "config.json").write_bytes(self.contents["config.json"])
        (pack / "model.bin").symlink_to(blob)

        result = validate_model_directory(
            pack,
            self.manifest,
            reject_links=True,
        )

        self.assertEqual(result.code, "model.unsafe_artifact")
        self.assertEqual(result.artifact_name, "model.bin")

    def test_strict_windows_pack_rejects_symlinked_directory(self) -> None:
        actual = self.directory / "actual"
        actual.mkdir()
        for name, content in self.contents.items():
            (actual / name).write_bytes(content)
        linked = self.directory / "linked"
        linked.symlink_to(actual, target_is_directory=True)

        result = validate_model_directory(
            linked,
            self.manifest,
            reject_links=True,
        )

        self.assertEqual(result.code, "model.unsafe_directory")

    def test_strict_windows_pack_rejects_symlinked_parent_component(self) -> None:
        actual_parent = self.directory / "actual-parent"
        actual_parent.mkdir()
        pack = actual_parent / "pack"
        pack.mkdir()
        for name, content in self.contents.items():
            (pack / name).write_bytes(content)
        linked_parent = self.directory / "linked-parent"
        linked_parent.symlink_to(actual_parent, target_is_directory=True)

        result = validate_model_directory(
            linked_parent / "pack",
            self.manifest,
            reject_links=True,
        )

        self.assertEqual(result.code, "model.unsafe_directory")

    def test_windows_reparse_attribute_is_detected_portably(self) -> None:
        metadata = SimpleNamespace(
            st_file_attributes=getattr(
                stat,
                "FILE_ATTRIBUTE_REPARSE_POINT",
                0x400,
            )
        )
        self.assertTrue(_is_reparse(metadata))

    def test_windows_model_paths_reject_unc_device_and_network_drives(self) -> None:
        self.assertTrue(_is_local_windows_absolute(r"C:\models\whisper"))
        self.assertFalse(_is_local_windows_absolute(r"\\server\share\whisper"))
        self.assertFalse(_is_local_windows_absolute(r"\\?\C:\models\whisper"))
        self.assertTrue(
            _windows_drive_is_local(
                r"C:\models\whisper",
                drive_type_reader=lambda _root: 3,
            )
        )
        self.assertFalse(
            _windows_drive_is_local(
                r"Z:\models\whisper",
                drive_type_reader=lambda _root: 4,
            )
        )
        self.assertFalse(
            _windows_drive_is_local(
                r"C:\models\whisper",
                drive_type_reader=lambda _root: 0,
            )
        )

    def test_strict_windows_pack_rejects_hidden_or_directory_members(self) -> None:
        self.write_valid_pack()
        (self.directory / ".cache").mkdir()

        result = validate_model_directory(
            self.directory,
            self.manifest,
            reject_links=True,
        )

        self.assertEqual(result.code, "model.unexpected_artifact")

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
