from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jarvis_inference.errors import ApplianceError
from jarvis_inference.manifest import load_manifest, verify_artifacts


def artifact(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class ManifestTests(unittest.TestCase):
    @staticmethod
    def document(root: Path) -> dict:
        exe = root / "llama-server.exe"; exe.write_bytes(b"x")
        model = root / "model.gguf"; model.write_bytes(b"x")
        return {
            "manifest_schema_version": 1, "alias": "jarvis-local", "platform": "windows-x86_64",
            "backend": {"type": "llama-server", "llama_cpp_commit": "a" * 40, "build_id": "x", "executable": artifact(exe, root), "dlls": [], "parameters": {}},
            "model": {"format": "gguf", "context_window": 4096, "artifacts": [artifact(model, root)]},
            "python": {}, "capabilities": {"text": True, "stream": True, "tools": False, "json_object": False, "json_schema": False},
            "qualification_report": "report.json",
        }

    def test_strict_load_and_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "llama-server.exe"
            model = root / "model.gguf"
            executable.write_bytes(b"exe")
            model.write_bytes(b"model")
            document = {
                "manifest_schema_version": 1,
                "alias": "jarvis-local",
                "platform": "windows-x86_64",
                "backend": {
                    "type": "llama-server", "llama_cpp_commit": "a" * 40,
                    "build_id": "test", "executable": artifact(executable, root),
                    "dlls": [], "parameters": {"context_window": 4096},
                },
                "model": {"format": "gguf", "context_window": 4096, "artifacts": [artifact(model, root)]},
                "python": {},
                "capabilities": {"text": True, "stream": True, "tools": False, "json_object": False, "json_schema": False},
                "qualification_report": "qualification-report.json",
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            manifest = load_manifest(path)
            verify_artifacts(manifest)
            model.write_bytes(b"changed")
            with self.assertRaises(ApplianceError) as caught:
                verify_artifacts(manifest)
            self.assertEqual(caught.exception.code, "artifact_identity_mismatch")

    def test_unknown_nested_field_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe = root / "x.exe"; exe.write_bytes(b"x")
            model = root / "x.gguf"; model.write_bytes(b"x")
            document = {
                "manifest_schema_version": 1, "alias": "a", "platform": "windows-x86_64",
                "backend": {"type": "llama-server", "llama_cpp_commit": "a" * 40, "build_id": "x", "executable": artifact(exe, root), "dlls": [], "parameters": {}, "args": []},
                "model": {"format": "gguf", "context_window": 10, "artifacts": [artifact(model, root)]},
                "python": {}, "capabilities": {"text": True, "stream": True, "tools": False, "json_object": False, "json_schema": False},
                "qualification_report": "report.json",
            }
            path = root / "manifest.json"; path.write_text(json.dumps(document))
            with self.assertRaises(ApplianceError) as caught:
                load_manifest(path)
            self.assertEqual(caught.exception.code, "unsupported_parameter")

    def test_traversal_and_symlink_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = self.document(root)
            document["model"]["artifacts"][0]["path"] = "../outside.gguf"
            path = root / "manifest.json"; path.write_text(json.dumps(document))
            with self.assertRaises(ApplianceError):
                load_manifest(path)

            target = root / "real.gguf"; target.write_bytes(b"model")
            link = root / "linked.gguf"; link.symlink_to(target)
            document = self.document(root)
            document["model"]["artifacts"] = [artifact(target, root) | {"path": "linked.gguf"}]
            path.write_text(json.dumps(document))
            with self.assertRaises(ApplianceError) as caught:
                load_manifest(path)
            self.assertEqual(caught.exception.code, "invalid_manifest_path")

    def test_qualification_report_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "real-report.json"; target.write_text("{}")
            (root / "report.json").symlink_to(target)
            path = root / "manifest.json"; path.write_text(json.dumps(self.document(root)))
            with self.assertRaises(ApplianceError):
                load_manifest(path)

    def test_undeclared_bundled_dll_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"; path.write_text(json.dumps(self.document(root)))
            manifest = load_manifest(path)
            (root / "surprise.dll").write_bytes(b"dll")
            with self.assertRaises(ApplianceError) as caught:
                verify_artifacts(manifest)
            self.assertEqual(caught.exception.code, "artifact_identity_mismatch")
