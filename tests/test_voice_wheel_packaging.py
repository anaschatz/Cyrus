from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class VoiceWheelPackagingTests(unittest.TestCase):
    def test_dirty_source_and_stale_build_outputs_cannot_enter_wheel(self) -> None:
        repository = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = project / "src" / "jarvis_voice"
            distribution = root / "dist"
            project.mkdir()
            shutil.copy2(repository / "pyproject.toml", project / "pyproject.toml")
            shutil.copy2(repository / "setup.py", project / "setup.py")
            shutil.copytree(
                repository / "src" / "jarvis_voice",
                source,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
            license_private = project / "LICENSE-private.pem"
            license_private.write_text(
                "must not be selected as a license\n",
                encoding="utf-8",
            )

            # The regression must not depend on user-owned duplicates remaining
            # in the checkout.  Add one invalid source module to the disposable
            # fixture and both invalid and valid-looking stale build outputs.
            (source / "synthetic 2.py").write_text(
                "raise AssertionError('must not be packaged')\n",
                encoding="utf-8",
            )
            (source / "unmanaged.data").write_text(
                "must not be packaged\n",
                encoding="utf-8",
            )
            rogue_package = project / "src" / "rogue_package"
            rogue_package.mkdir()
            (rogue_package / "__init__.py").write_text(
                "raise AssertionError('must not be packaged')\n",
                encoding="utf-8",
            )
            stale = project / "build" / "lib" / "jarvis_voice"
            stale.mkdir(parents=True)
            (stale / "stale 2.py").write_text("stale = True\n", encoding="utf-8")
            (stale / "obsolete.py").write_text("stale = True\n", encoding="utf-8")
            stale_sibling = project / "build" / "lib" / "stale_sibling"
            stale_sibling.mkdir()
            (stale_sibling / "__init__.py").write_text(
                "stale = True\n",
                encoding="utf-8",
            )
            (stale_sibling / "payload.data").write_text(
                "stale\n",
                encoding="utf-8",
            )
            staging = project / "build" / "wheel-staging"
            staged_sibling = staging / "stale_sibling"
            staged_sibling.mkdir(parents=True)
            (staged_sibling / "__init__.py").write_text(
                "stale = True\n",
                encoding="utf-8",
            )

            command = [
                sys.executable,
                "setup.py",
                "bdist_wheel",
                "--bdist-dir",
                "build/wheel-staging",
                "--dist-dir",
                str(distribution),
            ]
            stale_egg_info = project / "src" / "jarvis_voice.egg-info"
            stale_egg_info.mkdir()
            stale_private = stale_egg_info / "private.pem"
            stale_private.write_text("must be preserved\n", encoding="utf-8")
            rejected_stale_metadata = subprocess.run(
                command,
                cwd=project,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertNotEqual(rejected_stale_metadata.returncode, 0)
            self.assertEqual(list(distribution.glob("*.whl")), [])
            self.assertEqual(
                stale_private.read_text(encoding="utf-8"),
                "must be preserved\n",
            )
            stale_private.unlink()

            completed = subprocess.run(
                command,
                cwd=project,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"wheel build failed:\n{completed.stdout}\n{completed.stderr}",
            )
            wheels = list(distribution.glob("jarvis_voice-*.whl"))
            self.assertEqual(len(wheels), 1)

            with zipfile.ZipFile(wheels[0]) as archive:
                members = set(archive.namelist())

            packaged_modules = {
                name
                for name in members
                if name.startswith("jarvis_voice/") and name.endswith(".py")
            }
            expected_modules = {
                path.relative_to(project / "src").as_posix()
                for path in source.rglob("*.py")
                if path.stem.isidentifier()
                and all(
                    part.isidentifier()
                    for part in path.relative_to(source).parts[:-1]
                )
            }
            self.assertEqual(packaged_modules, expected_modules)
            dist_info_roots = {
                Path(name).parts[0]
                for name in members
                if Path(name).parts[0].endswith(".dist-info")
            }
            self.assertEqual(len(dist_info_roots), 1)
            dist_info = next(iter(dist_info_roots))
            expected_metadata = {
                f"{dist_info}/METADATA",
                f"{dist_info}/RECORD",
                f"{dist_info}/WHEEL",
                f"{dist_info}/entry_points.txt",
                f"{dist_info}/top_level.txt",
            }
            self.assertEqual(members, expected_modules | expected_metadata)
            self.assertNotIn("jarvis_voice/obsolete.py", members)
            self.assertFalse(
                any(name.startswith("stale_sibling/") for name in members)
            )
            self.assertFalse(
                any(name.startswith("rogue_package/") for name in members)
            )
            self.assertFalse(
                any(
                    " " in component
                    for name in members
                    if name.startswith("jarvis_voice/")
                    for component in Path(name).parts
                )
            )
            self.assertTrue(
                all(
                    name.endswith(".py")
                    for name in members
                    if name.startswith("jarvis_voice/")
                )
            )
            self.assertFalse(any(".data/" in name for name in members))
            self.assertTrue(license_private.is_file())

            metadata_root = root / "pep517-metadata"
            metadata_root.mkdir()
            prepare_metadata = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import setuptools.build_meta as backend, sys; "
                        "result = backend.prepare_metadata_for_build_wheel(sys.argv[1]); "
                        "print('PREPARED=' + result)"
                    ),
                    str(metadata_root),
                ],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(
                prepare_metadata.returncode,
                0,
                msg=(
                    "PEP 517 metadata preparation failed:\n"
                    f"{prepare_metadata.stdout}\n{prepare_metadata.stderr}"
                ),
            )
            prepared_names = [
                line.removeprefix("PREPARED=")
                for line in prepare_metadata.stdout.splitlines()
                if line.startswith("PREPARED=")
            ]
            self.assertEqual(len(prepared_names), 1)
            prepared_directory = metadata_root / prepared_names[0]
            injected_license_directory = prepared_directory / "licenses"
            injected_license_directory.mkdir()
            injected_metadata = injected_license_directory / "private.pem"
            injected_metadata.write_text(
                "must not cross the PEP 517 boundary\n",
                encoding="utf-8",
            )
            pep517_distribution = root / "pep517-dist"
            build_with_metadata = (
                "import setuptools.build_meta as backend, sys; "
                "backend.build_wheel(sys.argv[2], metadata_directory=sys.argv[1])"
            )
            rejected_metadata_injection = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    build_with_metadata,
                    str(prepared_directory),
                    str(pep517_distribution),
                ],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertNotEqual(rejected_metadata_injection.returncode, 0)
            self.assertEqual(list(pep517_distribution.glob("*.whl")), [])
            self.assertEqual(
                injected_metadata.read_text(encoding="utf-8"),
                "must not cross the PEP 517 boundary\n",
            )

            injected_metadata.unlink()
            injected_license_directory.rmdir()
            clean_pep517_build = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    build_with_metadata,
                    str(prepared_directory),
                    str(pep517_distribution),
                ],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(
                clean_pep517_build.returncode,
                0,
                msg=(
                    "PEP 517 wheel build failed:\n"
                    f"{clean_pep517_build.stdout}\n{clean_pep517_build.stderr}"
                ),
            )
            pep517_wheels = list(pep517_distribution.glob("*.whl"))
            self.assertEqual(len(pep517_wheels), 1)
            with zipfile.ZipFile(pep517_wheels[0]) as archive:
                self.assertEqual(set(archive.namelist()), members)

            skip_distribution = root / "skip-dist"
            skipped = subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "bdist_wheel",
                    "--skip-build",
                    "--dist-dir",
                    str(skip_distribution),
                ],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(skipped.returncode, 0)
            self.assertEqual(list(skip_distribution.glob("*.whl")), [])

            external_staging = root / "external-wheel-staging"
            external_distribution = root / "external-dist"
            external_wheel = subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "bdist_wheel",
                    "--bdist-dir",
                    str(external_staging),
                    "--dist-dir",
                    str(external_distribution),
                ],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(external_wheel.returncode, 0)
            self.assertFalse(external_staging.exists())
            self.assertEqual(list(external_distribution.glob("*.whl")), [])

            unsafe = subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "build_py",
                    "--build-lib",
                    "src",
                ],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(unsafe.returncode, 0)
            self.assertTrue((source / "synthetic 2.py").is_file())

            unowned = root / "unowned-build" / "jarvis_voice"
            unowned.mkdir(parents=True)
            sentinel = unowned / "sentinel.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            external = subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "build_py",
                    "--build-lib",
                    str(unowned.parent),
                ],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(external.returncode, 0)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

            shutil.rmtree(project / "build" / "lib")
            symlink_target = root / "symlink-target"
            symlink_target.mkdir()
            symlink_sentinel = symlink_target / "sentinel.txt"
            symlink_sentinel.write_text("keep\n", encoding="utf-8")
            try:
                (project / "build" / "lib").symlink_to(
                    symlink_target,
                    target_is_directory=True,
                )
            except (NotImplementedError, OSError):
                # Windows developer environments may not grant symlink rights.
                pass
            else:
                symlinked = subprocess.run(
                    [sys.executable, "setup.py", "build_py"],
                    cwd=project,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertNotEqual(symlinked.returncode, 0)
                self.assertEqual(
                    symlink_sentinel.read_text(encoding="utf-8"),
                    "keep\n",
                )
                self.assertFalse((symlink_target / "jarvis_voice").exists())

            invalid_package = source / "synthetic package"
            invalid_package.mkdir()
            (invalid_package / "__init__.py").write_text(
                "raise AssertionError('must not be packaged')\n",
                encoding="utf-8",
            )
            invalid_discovery = subprocess.run(
                [sys.executable, "setup.py", "egg_info"],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(invalid_discovery.returncode, 0)
            shutil.rmtree(invalid_package)

            linked_file_target = root / "linked-source.py"
            linked_file_target.write_text("linked = True\n", encoding="utf-8")
            linked_file = source / "linked_source.py"
            linked_directory_target = root / "linked-package"
            linked_directory_target.mkdir()
            (linked_directory_target / "__init__.py").write_text(
                "linked = True\n",
                encoding="utf-8",
            )
            linked_directory = source / "linked_package"
            try:
                linked_file.symlink_to(linked_file_target)
                linked_file_result = subprocess.run(
                    [sys.executable, "setup.py", "egg_info"],
                    cwd=project,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertNotEqual(linked_file_result.returncode, 0)
                linked_file.unlink()

                linked_directory.symlink_to(
                    linked_directory_target,
                    target_is_directory=True,
                )
                linked_directory_result = subprocess.run(
                    [sys.executable, "setup.py", "egg_info"],
                    cwd=project,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertNotEqual(linked_directory_result.returncode, 0)
            except (NotImplementedError, OSError):
                pass

    @unittest.skipUnless(
        sys.platform == "win32",
        "native Windows junction coverage required",
    )
    def test_windows_reparse_points_are_rejected_before_reads_or_writes(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = project / "src" / "jarvis_voice"
            project.mkdir()
            shutil.copy2(repository / "pyproject.toml", project / "pyproject.toml")
            shutil.copy2(repository / "setup.py", project / "setup.py")
            shutil.copytree(
                repository / "src" / "jarvis_voice",
                source,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )

            def create_junction(link: Path, target: Path) -> None:
                result = subprocess.run(
                    [
                        "cmd.exe",
                        "/d",
                        "/c",
                        "mklink",
                        "/J",
                        str(link),
                        str(target),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if result.returncode != 0:
                    self.fail(
                        "native Windows junction creation unavailable; "
                        "reparse-point coverage was not proven"
                    )

            source_target = root / "source-junction-target"
            source_target.mkdir()
            source_sentinel = source_target / "sentinel.py"
            source_sentinel.write_text("keep = True\n", encoding="utf-8")
            source_junction = source / "junction_package"
            create_junction(source_junction, source_target)
            rejected_source = subprocess.run(
                [sys.executable, "setup.py", "egg_info"],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(rejected_source.returncode, 0)
            self.assertEqual(
                source_sentinel.read_text(encoding="utf-8"),
                "keep = True\n",
            )
            source_junction.rmdir()

            project_build = project / "build"
            project_build.mkdir()
            build_target = root / "build-junction-target"
            build_target.mkdir()
            build_sentinel = build_target / "sentinel.txt"
            build_sentinel.write_text("keep\n", encoding="utf-8")
            build_junction = project_build / "lib"
            create_junction(build_junction, build_target)
            rejected_build = subprocess.run(
                [sys.executable, "setup.py", "build_py"],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(rejected_build.returncode, 0)
            self.assertEqual(
                build_sentinel.read_text(encoding="utf-8"),
                "keep\n",
            )
            build_junction.rmdir()

            staging_target = root / "staging-junction-target"
            staging_target.mkdir()
            staging_sentinel = staging_target / "sentinel.txt"
            staging_sentinel.write_text("keep\n", encoding="utf-8")
            staging_junction = project_build / "wheel-staging"
            create_junction(staging_junction, staging_target)
            rejected_staging = subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "bdist_wheel",
                    "--bdist-dir",
                    "build/wheel-staging",
                    "--dist-dir",
                    str(root / "dist"),
                ],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(rejected_staging.returncode, 0)
            self.assertEqual(
                staging_sentinel.read_text(encoding="utf-8"),
                "keep\n",
            )
            self.assertEqual(list((root / "dist").glob("*.whl")), [])


if __name__ == "__main__":
    unittest.main()
