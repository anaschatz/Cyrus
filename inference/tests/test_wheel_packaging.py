from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class InferenceWheelPackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        repository = Path(__file__).resolve().parents[1]
        self.project = self.root / "project"
        self.source = self.project / "src" / "jarvis_inference"
        self.project.mkdir()
        shutil.copy2(repository / "pyproject.toml", self.project / "pyproject.toml")
        shutil.copy2(repository / "setup.py", self.project / "setup.py")
        shutil.copytree(
            repository / "src" / "jarvis_inference",
            self.source,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
        self.expected_modules = {
            path.relative_to(self.project / "src").as_posix()
            for path in self.source.glob("*.py")
        }

    def run_command(
        self,
        arguments: list[str],
        *,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            arguments,
            cwd=self.project,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def build_command(self, distribution: Path) -> list[str]:
        return [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--bdist-dir",
            "build/wheel-staging",
            "--dist-dir",
            str(distribution),
        ]

    def assert_exact_wheel(self, wheel: Path) -> set[str]:
        with zipfile.ZipFile(wheel) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            payloads = {name: archive.read(name) for name in names}
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(
            all(
                not name.startswith("/")
                and ".." not in Path(name).parts
                and not stat.S_ISLNK(item.external_attr >> 16)
                for name, item in zip(names, infos, strict=True)
            )
        )
        dist_info_roots = {
            Path(name).parts[0]
            for name in names
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
        self.assertEqual(set(names), self.expected_modules | expected_metadata)
        record_name = f"{dist_info}/RECORD"
        rows = list(
            csv.reader(
                io.StringIO(payloads[record_name].decode("utf-8"), newline="")
            )
        )
        self.assertTrue(all(len(row) == 3 for row in rows))
        self.assertEqual(len(rows), len(names))
        records = {row[0]: (row[1], row[2]) for row in rows}
        self.assertEqual(len(records), len(rows))
        self.assertEqual(set(records), set(names))
        for name in names:
            digest, size = records[name]
            if name == record_name:
                self.assertEqual((digest, size), ("", ""))
                continue
            encoded_digest = base64.urlsafe_b64encode(
                hashlib.sha256(payloads[name]).digest()
            ).rstrip(b"=").decode("ascii")
            self.assertEqual(digest, f"sha256={encoded_digest}")
            self.assertEqual(size, str(len(payloads[name])))
        for name in self.expected_modules:
            self.assertEqual(
                payloads[name],
                (self.project / "src" / name).read_bytes(),
            )
        return set(names)

    def test_closed_payload_survives_dirty_source_build_and_pep517_paths(
        self,
    ) -> None:
        distribution = self.root / "dist"

        # Invalid recovery copies and non-code files remain untouched but are
        # outside the reviewed Python-module allowlist.
        recovery_copy = self.source / "gateway 2.py"
        recovery_copy.write_text("raise AssertionError('not packaged')\n")
        unmanaged = self.source / "private.data"
        unmanaged.write_text("not packaged\n")
        license_candidate = self.project / "LICENSE-private.pem"
        license_candidate.write_text("not a wheel license\n")

        # A second importable source module is different: fail closed until it
        # receives explicit review and an allowlist update.
        rogue_module = self.source / "rogue_secret.py"
        rogue_module.write_text("secret = True\n")
        rejected_rogue = self.run_command(self.build_command(distribution))
        self.assertNotEqual(rejected_rogue.returncode, 0)
        self.assertEqual(list(distribution.glob("*.whl")), [])
        self.assertTrue(rogue_module.is_file())
        rogue_module.unlink()

        rogue_package = self.project / "src" / "rogue_package"
        rogue_package.mkdir()
        (rogue_package / "__init__.py").write_text("secret = True\n")
        stale_package = self.project / "build" / "lib" / "jarvis_inference"
        stale_package.mkdir(parents=True)
        (stale_package / "obsolete.py").write_text("stale = True\n")
        stale_sibling = self.project / "build" / "lib" / "stale_sibling"
        stale_sibling.mkdir()
        (stale_sibling / "private.pem").write_text("stale\n")
        stale_staging = (
            self.project / "build" / "wheel-staging" / "stale_package"
        )
        stale_staging.mkdir(parents=True)
        (stale_staging / "__init__.py").write_text("stale = True\n")

        egg_info = self.project / "src" / "jarvis_inference.egg-info"
        egg_info.mkdir()
        unknown_metadata = egg_info / "PKG-INFO 2"
        unknown_metadata.write_text("must be preserved\n")
        rejected_metadata = self.run_command(self.build_command(distribution))
        self.assertNotEqual(rejected_metadata.returncode, 0)
        self.assertEqual(list(distribution.glob("*.whl")), [])
        self.assertEqual(unknown_metadata.read_text(), "must be preserved\n")
        unknown_metadata.unlink()
        linked_metadata_target = self.root / "linked-metadata-target"
        linked_metadata_target.write_text("must be preserved\n")
        linked_metadata = egg_info / "SOURCES.txt"
        try:
            linked_metadata.symlink_to(linked_metadata_target)
        except (NotImplementedError, OSError):
            linked_metadata = None
        if linked_metadata is not None:
            rejected_linked_metadata = self.run_command(
                self.build_command(distribution)
            )
            self.assertNotEqual(rejected_linked_metadata.returncode, 0)
            self.assertEqual(list(distribution.glob("*.whl")), [])
            self.assertEqual(
                linked_metadata_target.read_text(),
                "must be preserved\n",
            )
            linked_metadata.unlink()
        stale_recognized_metadata = egg_info / "PKG-INFO"
        stale_recognized_metadata.write_text("stale private metadata\n")

        completed = self.run_command(self.build_command(distribution))
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"wheel build failed:\n{completed.stdout}\n{completed.stderr}",
        )
        wheels = list(distribution.glob("jarvis_inference-*.whl"))
        self.assertEqual(len(wheels), 1)
        direct_members = self.assert_exact_wheel(wheels[0])
        with zipfile.ZipFile(wheels[0]) as archive:
            self.assertNotIn(
                b"stale private metadata",
                b"".join(archive.read(name) for name in archive.namelist()),
            )
        self.assertTrue(recovery_copy.is_file())
        self.assertTrue(unmanaged.is_file())
        self.assertTrue(license_candidate.is_file())
        self.assertFalse(any(name.startswith("rogue_package/") for name in direct_members))
        self.assertFalse(any(" " in Path(name).parts[-1] for name in direct_members))

        metadata_root = self.root / "pep517-metadata"
        metadata_root.mkdir()
        prepared = self.run_command(
            [
                sys.executable,
                "-c",
                (
                    "import setuptools.build_meta as b,sys; "
                    "print('PREPARED='+b.prepare_metadata_for_build_wheel(sys.argv[1]))"
                ),
                str(metadata_root),
            ]
        )
        self.assertEqual(
            prepared.returncode,
            0,
            msg=f"metadata preparation failed:\n{prepared.stdout}\n{prepared.stderr}",
        )
        names = [
            line.removeprefix("PREPARED=")
            for line in prepared.stdout.splitlines()
            if line.startswith("PREPARED=")
        ]
        self.assertEqual(len(names), 1)
        prepared_directory = metadata_root / names[0]
        injection_directory = prepared_directory / "licenses"
        injection_directory.mkdir()
        injection = injection_directory / "private.pem"
        injection.write_text("must be preserved\n")
        pep517_distribution = self.root / "pep517-dist"
        build_pep517 = (
            "import setuptools.build_meta as b,sys; "
            "b.build_wheel(sys.argv[2],metadata_directory=sys.argv[1])"
        )
        rejected_injection = self.run_command(
            [
                sys.executable,
                "-c",
                build_pep517,
                str(prepared_directory),
                str(pep517_distribution),
            ]
        )
        self.assertNotEqual(rejected_injection.returncode, 0)
        self.assertEqual(list(pep517_distribution.glob("*.whl")), [])
        self.assertEqual(injection.read_text(), "must be preserved\n")

        injection.unlink()
        injection_directory.rmdir()
        clean_pep517 = self.run_command(
            [
                sys.executable,
                "-c",
                build_pep517,
                str(prepared_directory),
                str(pep517_distribution),
            ]
        )
        self.assertEqual(
            clean_pep517.returncode,
            0,
            msg=f"PEP 517 build failed:\n{clean_pep517.stdout}\n{clean_pep517.stderr}",
        )
        pep517_wheels = list(pep517_distribution.glob("*.whl"))
        self.assertEqual(len(pep517_wheels), 1)
        self.assertEqual(self.assert_exact_wheel(pep517_wheels[0]), direct_members)

        skipped = self.run_command(
            [
                sys.executable,
                "setup.py",
                "bdist_wheel",
                "--skip-build",
                "--dist-dir",
                str(self.root / "skip-dist"),
            ]
        )
        self.assertNotEqual(skipped.returncode, 0)
        self.assertEqual(list((self.root / "skip-dist").glob("*.whl")), [])

        editable_distribution = self.root / "editable-dist"
        editable = self.run_command(
            [
                sys.executable,
                "-c",
                (
                    "import setuptools.build_meta as b,sys; "
                    "b.build_editable(sys.argv[1])"
                ),
                str(editable_distribution),
            ]
        )
        self.assertNotEqual(editable.returncode, 0)
        self.assertEqual(list(editable_distribution.glob("*.whl")), [])

    def test_external_build_paths_and_source_links_fail_without_mutation(self) -> None:
        outside = self.root / "outside-build"
        outside.mkdir()
        sentinel = outside / "sentinel.txt"
        sentinel.write_text("keep\n")
        external_build = self.run_command(
            [
                sys.executable,
                "setup.py",
                "build_py",
                "--build-lib",
                str(outside),
            ]
        )
        self.assertNotEqual(external_build.returncode, 0)
        self.assertEqual(sentinel.read_text(), "keep\n")

        external_staging = self.root / "external-staging"
        external_wheel = self.run_command(
            [
                sys.executable,
                "setup.py",
                "bdist_wheel",
                "--bdist-dir",
                str(external_staging),
                "--dist-dir",
                str(self.root / "external-dist"),
            ]
        )
        self.assertNotEqual(external_wheel.returncode, 0)
        self.assertFalse(external_staging.exists())
        self.assertEqual(list((self.root / "external-dist").glob("*.whl")), [])

        egg_target = self.root / "linked-egg-info-target"
        egg_target.mkdir()
        egg_sentinel = egg_target / "sentinel.txt"
        egg_sentinel.write_text("keep\n")
        egg_link = self.project / "src" / "jarvis_inference.egg-info"
        try:
            egg_link.symlink_to(egg_target, target_is_directory=True)
        except (NotImplementedError, OSError):
            return
        linked_egg_info = self.run_command(
            [sys.executable, "setup.py", "--name"]
        )
        self.assertNotEqual(linked_egg_info.returncode, 0)
        self.assertEqual(egg_sentinel.read_text(), "keep\n")
        egg_link.unlink()

        target = self.root / "outside.py"
        target.write_text("keep = True\n")
        linked = self.source / "linked_source.py"
        try:
            linked.symlink_to(target)
        except (NotImplementedError, OSError):
            return
        linked_source = self.run_command([sys.executable, "setup.py", "egg_info"])
        self.assertNotEqual(linked_source.returncode, 0)
        self.assertEqual(target.read_text(), "keep = True\n")
        linked.unlink()

        project_build = self.project / "build"
        project_build.mkdir(exist_ok=True)
        build_target = self.root / "linked-build-target"
        build_target.mkdir()
        build_sentinel = build_target / "sentinel.txt"
        build_sentinel.write_text("keep\n")
        build_link = project_build / "lib"
        build_link.symlink_to(build_target, target_is_directory=True)
        linked_build = self.run_command([sys.executable, "setup.py", "build_py"])
        self.assertNotEqual(linked_build.returncode, 0)
        self.assertEqual(build_sentinel.read_text(), "keep\n")
        self.assertFalse((build_target / "jarvis_inference").exists())
        build_link.unlink()

        staging_target = self.root / "linked-staging-target"
        staging_target.mkdir()
        staging_sentinel = staging_target / "sentinel.txt"
        staging_sentinel.write_text("keep\n")
        staging_link = project_build / "wheel-staging"
        staging_link.symlink_to(staging_target, target_is_directory=True)
        linked_staging = self.run_command(
            self.build_command(self.root / "linked-staging-dist")
        )
        self.assertNotEqual(linked_staging.returncode, 0)
        self.assertEqual(staging_sentinel.read_text(), "keep\n")
        self.assertEqual(
            list((self.root / "linked-staging-dist").glob("*.whl")),
            [],
        )

    @unittest.skipUnless(sys.platform == "win32", "native Windows reparse test")
    def test_windows_source_and_build_junctions_fail_closed(self) -> None:
        def junction(link: Path, target: Path) -> None:
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                self.fail("Windows junction coverage could not be exercised")

        source_target = self.root / "source-target"
        source_target.mkdir()
        source_sentinel = source_target / "sentinel.py"
        source_sentinel.write_text("keep = True\n")
        source_link = self.source / "linked_package"
        junction(source_link, source_target)
        rejected_source = self.run_command([sys.executable, "setup.py", "egg_info"])
        self.assertNotEqual(rejected_source.returncode, 0)
        self.assertEqual(source_sentinel.read_text(), "keep = True\n")
        source_link.rmdir()

        egg_target = self.root / "egg-info-junction-target"
        egg_target.mkdir()
        egg_sentinel = egg_target / "sentinel.txt"
        egg_sentinel.write_text("keep\n")
        egg_link = self.project / "src" / "jarvis_inference.egg-info"
        junction(egg_link, egg_target)
        rejected_egg_info = self.run_command(
            [sys.executable, "setup.py", "--name"]
        )
        self.assertNotEqual(rejected_egg_info.returncode, 0)
        self.assertEqual(egg_sentinel.read_text(), "keep\n")
        egg_link.rmdir()

        build_root = self.project / "build"
        build_root.mkdir(exist_ok=True)
        build_target = self.root / "build-target"
        build_target.mkdir()
        build_sentinel = build_target / "sentinel.txt"
        build_sentinel.write_text("keep\n")
        build_link = build_root / "lib"
        junction(build_link, build_target)
        rejected_build = self.run_command([sys.executable, "setup.py", "build_py"])
        self.assertNotEqual(rejected_build.returncode, 0)
        self.assertEqual(build_sentinel.read_text(), "keep\n")


if __name__ == "__main__":
    unittest.main()
