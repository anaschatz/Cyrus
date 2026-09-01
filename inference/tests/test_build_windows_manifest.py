from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

from jarvis_inference.gemma4 import GEMMA4_PROMPT_PROFILE


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_windows_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_windows_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class WindowsManifestBuilderTests(unittest.TestCase):
    def layout(self, root: Path) -> tuple[Path, Path, Path]:
        runtime = root / "runtime" / "llama"
        models = root / "models"
        runtime.mkdir(parents=True)
        models.mkdir()
        executable = runtime / "llama-server.exe"
        dll = runtime / "ggml-cuda.dll"
        model = models / builder.GEMMA_FILENAME
        executable.write_bytes(b"exe")
        dll.write_bytes(b"dll")
        model.write_bytes(b"fixture-model")
        return executable, dll, model

    def test_builds_offline_manifest_for_exact_locked_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable, dll, model = self.layout(root)
            original_hash = builder._hash

            def pinned_hash(path: Path) -> tuple[int, str]:
                if path == model:
                    return builder.GEMMA_SIZE_BYTES, builder.GEMMA_SHA256
                return original_hash(path)

            output = root / "manifest.json"
            with mock.patch.object(builder, "_hash", side_effect=pinned_hash):
                result = builder.build(
                    root,
                    output,
                    "a" * 40,
                    "llama-b12345-cuda12.9",
                    enable_tools=True,
                    enable_structured=True,
                )

            self.assertTrue(result["ready"])
            self.assertFalse(result["network_used"])
            self.assertFalse(result["model_loaded"])
            self.assertFalse(result["native_qualification_complete"])
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["platform"], "windows-x86_64")
            self.assertEqual(value["backend"]["executable"]["path"], "runtime/llama/llama-server.exe")
            self.assertEqual(value["backend"]["dlls"][0]["path"], "runtime/llama/ggml-cuda.dll")
            self.assertEqual(value["model"]["artifacts"], [{
                "path": f"models/{builder.GEMMA_FILENAME}",
                "size_bytes": builder.GEMMA_SIZE_BYTES,
                "sha256": builder.GEMMA_SHA256,
            }])
            self.assertEqual(value["prompt_profile"], GEMMA4_PROMPT_PROFILE)
            self.assertEqual(
                value["profile_qualification_report"],
                "qualification-profile-local-primary.json",
            )
            self.assertEqual(builder.GEMMA_PROMPT_PROFILE, GEMMA4_PROMPT_PROFILE)
            self.assertEqual(value["capabilities"], {
                "text": True,
                "stream": True,
                "tools": True,
                "json_object": True,
                "json_schema": True,
            })
            self.assertEqual(original_hash(executable), (
                value["backend"]["executable"]["size_bytes"],
                value["backend"]["executable"]["sha256"],
            ))
            self.assertEqual(original_hash(dll), (
                value["backend"]["dlls"][0]["size_bytes"],
                value["backend"]["dlls"][0]["sha256"],
            ))

    def test_rejects_model_that_does_not_match_frozen_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.layout(root)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                builder.build(root, root / "manifest.json", "b" * 40, "build-1")

    def test_rejects_unsafe_or_incomplete_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, model = self.layout(root)
            (root / "runtime" / "llama" / "notes.txt").write_text("not sealed", encoding="utf-8")
            original_hash = builder._hash

            def pinned_hash(path: Path) -> tuple[int, str]:
                if path == model:
                    return builder.GEMMA_SIZE_BYTES, builder.GEMMA_SHA256
                return original_hash(path)

            with mock.patch.object(builder, "_hash", side_effect=pinned_hash):
                with self.assertRaisesRegex(ValueError, "undeclared"):
                    builder.build(root, root / "manifest.json", "c" * 40, "build-1")

    def test_rejects_non_frozen_profile_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.layout(root)
            with self.assertRaisesRegex(ValueError, "tool capability"):
                builder.build(
                    root,
                    root / "manifest.json",
                    "d" * 40,
                    "build-1",
                    enable_tools=True,
                    enable_structured=False,
                )
            with self.assertRaisesRegex(ValueError, "gpu_layers"):
                builder.build(
                    root,
                    root / "manifest.json",
                    "d" * 40,
                    "build-1",
                    gpu_layers=0,
                )

    def test_windows_reparse_ancestor_is_rejected_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            real_root = parent / "real"
            real_root.mkdir()
            self.layout(real_root)
            linked_root = parent / "linked"
            try:
                linked_root.symlink_to(real_root, target_is_directory=True)
            except (NotImplementedError, OSError):
                self.skipTest("directory links are unavailable")

            with mock.patch.object(
                builder, "_native_windows", return_value=True
            ), mock.patch.object(
                builder, "_require_windows_local_drive_paths"
            ), self.assertRaisesRegex(ValueError, "reparse point"):
                builder.build(
                    linked_root,
                    linked_root / "manifest.json",
                    "e" * 40,
                    "build-1",
                )

    def test_manifest_publication_is_exclusive_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, model = self.layout(root)
            output = root / "manifest.json"
            sentinel = b"pre-existing sealed manifest"
            output.write_bytes(sentinel)
            original_hash = builder._hash

            def pinned_hash(path: Path) -> tuple[int, str]:
                if path == model:
                    return builder.GEMMA_SIZE_BYTES, builder.GEMMA_SHA256
                return original_hash(path)

            with mock.patch.object(builder, "_hash", side_effect=pinned_hash):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    builder.build(
                        root,
                        output,
                        "f" * 40,
                        "build-1",
                    )
            self.assertEqual(output.read_bytes(), sentinel)

    def test_windows_builder_requires_absolute_drive_letter_paths(self) -> None:
        self.assertTrue(
            builder._windows_local_drive_path(
                PureWindowsPath(r"D:\CYRUS\inference")
            )
        )
        for value in (
            PureWindowsPath(r"\\server\share\inference"),
            PureWindowsPath(r"\\?\D:\CYRUS\inference"),
            PureWindowsPath(r"\\.\D:\CYRUS\inference"),
            PureWindowsPath(r"D:relative\inference"),
            PureWindowsPath(r"\rooted\inference"),
        ):
            with self.subTest(value=str(value)):
                self.assertFalse(builder._windows_local_drive_path(value))

        with mock.patch.object(
            builder, "_native_windows", return_value=True
        ), self.assertRaisesRegex(ValueError, "local-drive"):
            builder._require_windows_local_drive_paths(
                Path("//server/share/runtime"),
                Path("//server/share/runtime/manifest.json"),
            )

        self.assertTrue(
            builder._windows_drive_is_local(
                PureWindowsPath(r"Z:\CYRUS\inference"),
                drive_type_reader=lambda _root: 3,
            )
        )
        self.assertFalse(
            builder._windows_drive_is_local(
                PureWindowsPath(r"Z:\CYRUS\inference"),
                drive_type_reader=lambda _root: 4,
            )
        )
        with mock.patch.object(
            builder, "_native_windows", return_value=True
        ), mock.patch.object(
            builder, "_native_windows_drive_type", return_value=4
        ), self.assertRaisesRegex(ValueError, "local-drive"):
            builder._require_windows_local_drive_paths(
                Path(r"Z:\CYRUS\inference"),
                Path(r"Z:\CYRUS\inference\manifest.json"),
            )

    def test_windows_builder_guard_receives_original_unnormalized_paths(self) -> None:
        root = Path(r"D:relative\runtime")
        output = Path(r"D:relative\runtime\manifest.json")
        with mock.patch.object(
            builder, "_require_windows_local_drive_paths"
        ) as guard:
            guard.side_effect = ValueError("unsafe")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                builder.build(root, output, "f" * 40, "build-1")
        guard.assert_called_once_with(root, output)


if __name__ == "__main__":
    unittest.main()
