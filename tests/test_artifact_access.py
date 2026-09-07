from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_voice.manifest import ModelFileSpec, ModelManifest, validate_model_directory
from jarvis_voice.vad import validate_artifact


class ArtifactAccessTests(unittest.TestCase):
    def test_permission_failures_are_validation_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            digest = hashlib.sha256(b"model").hexdigest()
            manifest = ModelManifest("test", "pinned", "fake", (ModelFileSpec("model.bin", 5, digest),))
            original_stat = Path.stat
            for phase, denied, expected in (
                ("vad", artifact, "vad.artifact_unreadable"),
                ("model-member", artifact, "model.directory_unreadable"),
                ("model-directory", root, "model.directory_missing"),
            ):
                with self.subTest(phase=phase):
                    def stat(path, *args, **kwargs):
                        if path == denied and kwargs.get("follow_symlinks", True):
                            raise PermissionError("synthetic access denial")
                        return original_stat(path, *args, **kwargs)
                    with patch.object(Path, "stat", stat):
                        if phase == "vad":
                            self.assertEqual(validate_artifact(str(artifact), 5, digest), (False, expected))
                        else:
                            result = validate_model_directory(root, manifest, reject_links=False)
                            self.assertFalse(result.ok)
                            self.assertEqual(result.code, expected)
            self.assertTrue(validate_artifact(str(artifact), 5, digest)[0])
            self.assertTrue(validate_model_directory(root, manifest, reject_links=False).ok)

    def test_vad_rejects_malformed_json_artifact_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vad.onnx"
            path.write_bytes(b"v")
            digest = hashlib.sha256(b"v").hexdigest()
            for artifact_path, size, sha256 in (
                ([str(path)], 1, digest),
                (str(path), True, digest),
                (str(path), 1.0, digest),
                (str(path), "1", digest),
                (str(path), 1, [digest]),
            ):
                with self.subTest(path=artifact_path, size=size, sha256=sha256):
                    self.assertEqual(
                        validate_artifact(artifact_path, size, sha256),
                        (False, "vad.artifact_manifest_required"),
                    )
            self.assertEqual(validate_artifact(str(path), 1, digest.upper()), (True, "vad.artifact_valid"))


if __name__ == "__main__":
    unittest.main()
