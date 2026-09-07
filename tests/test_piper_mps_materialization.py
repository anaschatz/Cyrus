from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.piper_mps_materialization import materialize_piper_mps_extension
from jarvis_voice.piper_mps_training import PIPER_REVISION, PIPER_VERSION


class FakeRunner:
    def __init__(self, platlib: Path) -> None:
        self.platlib = platlib
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if "Cython.Build.Cythonize" in argv:
            Path(str(kwargs["cwd"]), "core.cpython-311-darwin.so").write_bytes(b"extension")
            output = ""
        elif "sysconfig.get_path" in argv[-1]:
            output = json.dumps(
                {
                    "piper": PIPER_VERSION,
                    "cython": "3.2.9",
                    "numpy": "2.4.6",
                    "platlib": str(self.platlib),
                    "ext_suffix": ".cpython-311-darwin.so",
                }
            )
        else:
            output = ""
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")


class PiperMpsMaterializationTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path]:
        environment = root / "environment"
        python = environment / "bin/python"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")
        platlib = environment / "lib/python3.11/site-packages"
        package = platlib / "piper/train/vits/monotonic_align"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        source = root / "source"
        core = source / "src/piper/train/vits/monotonic_align/core.pyx"
        core.parent.mkdir(parents=True)
        core.write_text("# cython", encoding="utf-8")
        return python, platlib, source

    def test_materializes_and_import_probes_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            python, platlib, source = self.fixture(Path(temporary))
            runner = FakeRunner(platlib)
            report = materialize_piper_mps_extension(
                python,
                source,
                runner=runner,
                revision_probe=lambda _source: PIPER_REVISION,
                compiler_path="/usr/bin/clang",
            )
            target = platlib / "piper/train/vits/monotonic_align/monotonic_align"
            built = tuple(target.glob("core*.so"))
            provenance_ready = (target / "jarvis-provenance.json").is_file()
        self.assertTrue(report["ready"])
        self.assertEqual(report["code"], "tts.f1_mps.extension_materialized")
        self.assertEqual(len(built), 1)
        self.assertTrue(provenance_ready)
        self.assertEqual(len(runner.calls), 3)

    def test_second_run_validates_provenance_without_rebuilding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            python, platlib, source = self.fixture(Path(temporary))
            runner = FakeRunner(platlib)
            first = materialize_piper_mps_extension(
                python,
                source,
                runner=runner,
                revision_probe=lambda _source: PIPER_REVISION,
                compiler_path="/usr/bin/clang",
            )
            runner.calls.clear()
            second = materialize_piper_mps_extension(
                python,
                source,
                runner=runner,
                revision_probe=lambda _source: PIPER_REVISION,
                compiler_path="/usr/bin/clang",
            )
        self.assertEqual(first["extension_sha256"], second["extension_sha256"])
        self.assertEqual(second["code"], "tts.f1_mps.extension_already_ready")
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(runner.calls), 2)

    def test_rejects_unpinned_source_before_compilation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            python, platlib, source = self.fixture(Path(temporary))
            with self.assertRaisesRegex(
                BackendError, "tts.f1_mps.source_revision_mismatch"
            ):
                materialize_piper_mps_extension(
                    python,
                    source,
                    runner=FakeRunner(platlib),
                    revision_probe=lambda _source: "wrong",
                    compiler_path="/usr/bin/clang",
                )

    def test_rejects_platlib_outside_isolated_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python, _platlib, source = self.fixture(root)
            outside = root / "outside"
            outside.mkdir()
            with self.assertRaisesRegex(BackendError, "tts.f1_mps.platlib_invalid"):
                materialize_piper_mps_extension(
                    python,
                    source,
                    runner=FakeRunner(outside),
                    revision_probe=lambda _source: PIPER_REVISION,
                    compiler_path="/usr/bin/clang",
                )

    def test_rejects_resolved_paths_outside_environment_before_build(self):
        routes = ("parent-components",) if os.name == "nt" else ("parent-components", "platlib-link", "package-link")
        for route in routes:
            with self.subTest(route=route), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                python, platlib, source = self.fixture(root)
                outside = root / "outside"
                outside_package = outside / "piper/train/vits/monotonic_align"
                outside_package.mkdir(parents=True)
                outside_target = outside_package / "monotonic_align"
                outside_target.mkdir()
                (outside_package / "__init__.py").write_text("")
                if route == "parent-components":
                    selected = python.parent.parent / ".." / "outside"
                elif route == "platlib-link":
                    selected = python.parent.parent / "linked-packages"
                    selected.symlink_to(outside, target_is_directory=True)
                else:
                    selected = platlib
                    (platlib / "piper/train/vits/monotonic_align/monotonic_align").symlink_to(outside_package / "monotonic_align", target_is_directory=True)
                runner = FakeRunner(selected)
                with self.assertRaisesRegex(BackendError, "tts.f1_mps.platlib_invalid"):
                    materialize_piper_mps_extension(
                        python, source, runner=runner,
                        revision_probe=lambda _: PIPER_REVISION,
                        compiler_path="/usr/bin/clang",
                    )
                self.assertEqual(len(runner.calls), 1)
                self.assertEqual(list(outside_target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
