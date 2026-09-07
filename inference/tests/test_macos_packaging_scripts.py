from __future__ import annotations

import importlib.util
import hashlib
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_qualified_bundle_clone import QualifiedBundleFixture
from jarvis_inference.manifest import load_manifest, _verify_python_record_files
from jarvis_inference.python_environment import hash_environment, hash_file
from jarvis_inference.qualification import _run_fixture_runner
from jarvis_inference.manifest import resolve_local
import os


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_macos_manifest")
records = load_script("check_python_records")


class MacosPackagingScriptTests(unittest.TestCase):
    def test_artifact_hash_handles_windows_executable_mode_without_ignoring_posix_changes(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runner.exe"
            payload = b"runner"
            path.write_bytes(payload)
            metadata = path.stat()
            fields = {
                name: getattr(metadata, name)
                for name in (
                    "st_dev",
                    "st_ino",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            }
            for windows in (True, False):
                attributes = {"st_file_attributes": 0} if windows else {}
                initial = SimpleNamespace(
                    **fields, **attributes, st_mode=metadata.st_mode | 0o111
                )
                opened = SimpleNamespace(
                    **fields, **attributes, st_mode=metadata.st_mode & ~0o111
                )
                with (
                    self.subTest(windows=windows),
                    patch.object(Path, "lstat", return_value=initial),
                    patch.object(os, "fstat", return_value=opened),
                ):
                    if windows:
                        self.assertEqual(
                            hash_file(path),
                            (len(payload), hashlib.sha256(payload).hexdigest()),
                        )
                    else:
                        with self.assertRaisesRegex(
                            OSError, "artifact changed before hashing"
                        ):
                            hash_file(path)

    def test_native_mac_fixture_launch_disables_bytecode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            QualifiedBundleFixture(root)
            manifest = load_manifest(root / "manifest.json")
            runner = root / "fixture.py"
            runner.write_text(
                "import json, os, sys\nfrom pathlib import Path\nPath(os.environ['JARVIS_QUAL_OUTPUT']).write_text(json.dumps({'dont_write_bytecode': sys.dont_write_bytecode}))\n"
            )
            runner.chmod(0o700)

            def local_path(root, path, parameter, **kwargs):
                if parameter == "python.executable":
                    return Path(sys.executable)
                return resolve_local(root, path, parameter, **kwargs)

            with patch(
                "jarvis_inference.qualification.resolve_local", side_effect=local_path
            ):
                evidence, _, _, _ = _run_fixture_runner(
                    runner,
                    manifest,
                    "c" * 64,
                    "d" * 64,
                    "http://127.0.0.1:1",
                    "test-only",
                )
            self.assertEqual(evidence, {"dont_write_bytecode": True})

    def test_builder_binds_the_complete_python_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            QualifiedBundleFixture(root)
            output = root / "generated.json"
            with patch.object(
                builder, "_python_identity", return_value=("CPython", "3.11.9")
            ):
                result = builder.build(root, root / "models/qwen", output, 4096)
            self.assertTrue(result["ready"])
            manifest = load_manifest(output)
            self.assertEqual(
                manifest.python["environment_sha256"], hash_environment(root / "python")
            )
            _verify_python_record_files(manifest)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires FIFO support")
    def test_artifact_hash_rejects_fifo_substitution_without_blocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact"
            path.write_bytes(b"original")
            original_open = os.open
            descriptors = []

            def replace_with_fifo(target, flags):
                self.assertTrue(flags & os.O_NONBLOCK)
                path.unlink()
                os.mkfifo(path)
                descriptor = original_open(target, flags)
                descriptors.append(descriptor)
                return descriptor

            with patch.object(os, "open", side_effect=replace_with_fifo):
                with self.assertRaises(OSError):
                    hash_file(path)
            self.assertEqual(len(descriptors), 1)
            with self.assertRaises(OSError):
                os.fstat(descriptors[0])

    def test_builder_rejects_noninteger_context_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            QualifiedBundleFixture(root)
            output = root / "generated.json"
            with patch.object(
                builder, "_python_identity", return_value=("CPython", "3.11.9")
            ):
                for value in (True, 4096.0, "4096"):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        builder.build(root, root / "models/qwen", output, value)
                    self.assertFalse(output.exists())

    def test_record_preflight_rejects_empty_or_unnamed_distributions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "python/lib/python3.11/site-packages").mkdir(parents=True)
            self.assertFalse(records.check(root)[0])
            fixture = QualifiedBundleFixture(root)
            self.assertTrue(records.check(root)[0])
            metadata = fixture.sdk_dist / "METADATA"
            metadata.write_text("Metadata-Version: 2.1\nVersion: 2.46.0\n")
            self.assertFalse(records.check(root)[0])
            with self.assertRaises(ValueError):
                builder._installed(fixture.site)
