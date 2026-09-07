from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path, PureWindowsPath
from unittest import mock

from jarvis_inference import manifest as manifest_module
from jarvis_inference.errors import ApplianceError
from jarvis_inference.gemma4 import (
    GEMMA4_PROMPT_PROFILE,
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
)
from jarvis_inference.manifest import load_manifest, verify_artifacts


def artifact(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class ManifestTests(unittest.TestCase):
    def test_malformed_manifest_values_return_typed_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            base = self.document(root)
            for field, value in (("platform", []), ("platform", {}), ("manifest_schema_version", True), ("manifest_schema_version", 1.0), ("format", [])):
                with self.subTest(field=field, value=value):
                    document = deepcopy(base)
                    (document["model"] if field == "format" else document)[field] = value
                    path.write_text(json.dumps(document))
                    with self.assertRaises(ApplianceError):
                        load_manifest(path)
            path.write_bytes(b"[" * 2000 + b"0" + b"]" * 2000)
            with self.assertRaises(ApplianceError):
                load_manifest(path)

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

    def test_manifest_leaf_symlink_and_oversize_input_are_rejected_before_parse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real-manifest.json"
            real.write_text(json.dumps(self.document(root)), encoding="utf-8")
            link = root / "manifest.json"
            try:
                link.symlink_to(real)
            except (NotImplementedError, OSError):
                self.skipTest("symlinks are unavailable")
            with self.assertRaises(ApplianceError) as caught:
                load_manifest(link)
            self.assertEqual(caught.exception.code, "invalid_manifest_path")

            link.unlink()
            link.write_bytes(b" " * (262_144 + 1))
            with self.assertRaises(ApplianceError) as caught:
                load_manifest(link)
            self.assertEqual(caught.exception.code, "manifest_too_large")

    def test_windows_reparse_ancestor_is_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            real_root = parent / "real"
            real_root.mkdir()
            manifest = real_root / "manifest.json"
            manifest.write_text(json.dumps(self.document(real_root)), encoding="utf-8")
            linked_root = parent / "linked"
            try:
                linked_root.symlink_to(real_root, target_is_directory=True)
            except (NotImplementedError, OSError):
                self.skipTest("directory links are unavailable")

            with mock.patch.object(
                manifest_module, "_native_windows", return_value=True
            ), mock.patch.object(
                manifest_module, "_require_windows_local_drive_path"
            ), self.assertRaises(ApplianceError) as caught:
                load_manifest(linked_root / "manifest.json")
            self.assertEqual(caught.exception.code, "invalid_manifest_path")

    def test_windows_manifest_path_requires_absolute_drive_letter_syntax(self) -> None:
        self.assertTrue(
            manifest_module._windows_local_drive_path(
                PureWindowsPath(r"C:\CYRUS\inference\manifest.json")
            )
        )
        for value in (
            PureWindowsPath(r"\\server\share\manifest.json"),
            PureWindowsPath(r"\\?\C:\CYRUS\manifest.json"),
            PureWindowsPath(r"\\.\C:\CYRUS\manifest.json"),
            PureWindowsPath(r"C:relative\manifest.json"),
            PureWindowsPath(r"\rooted\manifest.json"),
        ):
            with self.subTest(value=str(value)):
                self.assertFalse(
                    manifest_module._windows_local_drive_path(value)
                )

        with mock.patch.object(
            manifest_module, "_native_windows", return_value=True
        ), self.assertRaises(ApplianceError) as caught:
            manifest_module._require_windows_local_drive_path(
                Path("//server/share/manifest.json")
            )
        self.assertEqual(caught.exception.code, "invalid_manifest_path")

        self.assertTrue(
            manifest_module._windows_drive_is_local(
                PureWindowsPath(r"Z:\CYRUS\inference\manifest.json"),
                drive_type_reader=lambda _root: 3,
            )
        )
        self.assertFalse(
            manifest_module._windows_drive_is_local(
                PureWindowsPath(r"Z:\CYRUS\inference\manifest.json"),
                drive_type_reader=lambda _root: 4,
            )
        )
        with mock.patch.object(
            manifest_module, "_native_windows", return_value=True
        ), mock.patch.object(
            manifest_module, "_native_windows_drive_type", return_value=4
        ), self.assertRaises(ApplianceError) as mapped:
            manifest_module._require_windows_local_drive_path(
                Path(r"Z:\CYRUS\inference\manifest.json")
            )
        self.assertEqual(mapped.exception.code, "invalid_manifest_path")

    def test_windows_runtime_ram_uses_global_memory_status_contract(self) -> None:
        expected = 32 * 1024**3
        self.assertEqual(
            manifest_module._ram_bytes(
                platform_name="nt",
                windows_reader=lambda: expected,
            ),
            expected,
        )
        for invalid in (None, 0, -1, True, 2**64):
            with self.subTest(invalid=invalid):
                self.assertIsNone(
                    manifest_module._ram_bytes(
                        platform_name="nt",
                        windows_reader=lambda value=invalid: value,
                    )
                )

        source = Path(manifest_module.__file__).read_text(encoding="utf-8")
        self.assertIn("GlobalMemoryStatusEx", source)
        self.assertIn("ullTotalPhys", source)

    def test_windows_manifest_guard_receives_original_unnormalized_path(self) -> None:
        supplied = Path(r"C:relative\manifest.json")
        with mock.patch.object(
            manifest_module, "_require_windows_local_drive_path"
        ) as guard:
            guard.side_effect = ApplianceError(
                400, "invalid_manifest_path", "unsafe"
            )
            with self.assertRaises(ApplianceError):
                load_manifest(supplied)
        guard.assert_called_once_with(supplied)

    def test_undeclared_bundled_dll_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"; path.write_text(json.dumps(self.document(root)))
            manifest = load_manifest(path)
            (root / "surprise.dll").write_bytes(b"dll")
            with self.assertRaises(ApplianceError) as caught:
                verify_artifacts(manifest)
            self.assertEqual(caught.exception.code, "artifact_identity_mismatch")

    def test_exact_gemma_prompt_profile_is_explicit_and_model_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = self.document(root)
            model = root / MODEL_FILENAME
            model.write_bytes(b"identity is verified before spawn")
            document["model"]["artifacts"] = [{
                "path": MODEL_FILENAME,
                "size_bytes": MODEL_SIZE_BYTES,
                "sha256": MODEL_SHA256,
            }]
            document["prompt_profile"] = deepcopy(GEMMA4_PROMPT_PROFILE)
            document["profile_qualification_report"] = "qualification-profile-local-primary.json"
            path = root / "manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            manifest = load_manifest(path)
            self.assertEqual(manifest.prompt_profile, GEMMA4_PROMPT_PROFILE)

            document["prompt_profile"]["renderer"]["sha256"] = "0" * 64
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ApplianceError) as caught:
                load_manifest(path)
            self.assertEqual(caught.exception.code, "invalid_manifest")

    def test_gemma_prompt_profile_rejects_a_different_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = self.document(root)
            document["prompt_profile"] = deepcopy(GEMMA4_PROMPT_PROFILE)
            document["profile_qualification_report"] = "qualification-profile-local-primary.json"
            path = root / "manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ApplianceError) as caught:
                load_manifest(path)
            self.assertEqual(caught.exception.code, "invalid_manifest")

    def test_windows_structured_capabilities_require_exact_prompt_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for capability in ("tools", "json_object", "json_schema"):
                document = self.document(root)
                document["capabilities"][capability] = True
                path = root / "manifest.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(capability=capability), self.assertRaises(ApplianceError) as caught:
                    load_manifest(path)
                self.assertEqual(caught.exception.code, "invalid_manifest")
                self.assertEqual(caught.exception.parameter, "prompt_profile")
