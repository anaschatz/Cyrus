from __future__ import annotations

import csv
import io
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_EVAL_DISTRIBUTIONS = {
    "fastembed",
    "inspect-ai",
    "numpy",
    "openai",
}


def _distribution_name(requirement: str) -> str:
    return (
        requirement.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0]
    )


def test_base_distribution_has_only_the_runtime_dependency() -> None:
    document = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = document["project"]
    assert project["dependencies"] == ["pydantic>=2.10,<3"]

    extras = project["optional-dependencies"]
    assert {
        _distribution_name(requirement) for requirement in extras["eval"]
    } == OPTIONAL_EVAL_DISTRIBUTIONS
    assert {
        _distribution_name(requirement) for requirement in extras["dev"]
    } >= OPTIONAL_EVAL_DISTRIBUTIONS


def test_orchestrator_runtime_modules_do_not_import_eval_dependencies() -> None:
    script = r"""
import importlib.abc
import sys

blocked = {"fastembed", "inspect_ai", "numpy", "openai"}
sys.path.insert(0, sys.argv[1])

class BlockEvalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError(f"blocked optional eval dependency: {fullname}")
        return None

sys.meta_path.insert(0, BlockEvalImports())
from anamnesis.openmemory_recall import OpenMemoryRecallIndex
from anamnesis.runtime_contract import anamnesis_runtime_contract

assert OpenMemoryRecallIndex is not None
assert anamnesis_runtime_contract()["memory_schema_version"]
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(PROJECT_ROOT / "src")],
        check=True,
        cwd=PROJECT_ROOT,
    )


def test_wheel_payload_is_closed_with_dirty_recovery_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        project = root / "project"
        distribution = root / "dist"
        shutil.copytree(
            PROJECT_ROOT,
            project,
            ignore=shutil.ignore_patterns(
                ".venv",
                "__pycache__",
                "*.pyc",
                "dist",
                ".pytest_cache",
            ),
        )
        package = project / "src" / "anamnesis"
        recovery = package / "synthetic recovery 2.py"
        recovery.write_text("raise AssertionError('not packaged')\n", encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "hatchling",
                "build",
                "-t",
                "wheel",
                "-d",
                str(distribution),
            ],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        wheels = list(distribution.glob("*.whl"))
        assert len(wheels) == 1
        with zipfile.ZipFile(wheels[0]) as archive:
            names = set(archive.namelist())

        expected_modules = {
            path.relative_to(project / "src").as_posix()
            for path in package.rglob("*.py")
            if path.stem.isidentifier()
            and all(
                component.isidentifier()
                for component in path.relative_to(package).parts[:-1]
            )
        }
        dist_info_roots = {
            Path(name).parts[0]
            for name in names
            if Path(name).parts[0].endswith(".dist-info")
        }
        assert len(dist_info_roots) == 1
        dist_info = next(iter(dist_info_roots))
        expected_metadata = {
            f"{dist_info}/METADATA",
            f"{dist_info}/RECORD",
            f"{dist_info}/WHEEL",
            f"{dist_info}/entry_points.txt",
        }
        assert names == expected_modules | expected_metadata
        assert recovery.is_file()


def test_wheel_build_rejects_unknown_package_payload() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        project = root / "project"
        distribution = root / "dist"
        shutil.copytree(
            PROJECT_ROOT,
            project,
            ignore=shutil.ignore_patterns(
                ".venv",
                "__pycache__",
                "*.pyc",
                "dist",
                ".pytest_cache",
            ),
        )
        package = project / "src" / "anamnesis"
        data = package / "private.pem"
        data.write_text("must remain private\n", encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "hatchling",
                "build",
                "-t",
                "wheel",
                "-d",
                str(distribution),
            ],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode != 0
        assert list(distribution.glob("*.whl")) == []
        assert data.read_text(encoding="utf-8") == "must remain private\n"


def test_wheel_build_rejects_source_links_and_unknown_package_payload() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        project = root / "project"
        distribution = root / "dist"
        shutil.copytree(
            PROJECT_ROOT,
            project,
            ignore=shutil.ignore_patterns(
                ".venv",
                "__pycache__",
                "*.pyc",
                "dist",
                ".pytest_cache",
            ),
        )
        package = project / "src" / "anamnesis"
        outside = root / "outside.py"
        outside.write_text("secret = True\n", encoding="utf-8")
        linked = package / "linked.py"
        try:
            linked.symlink_to(outside)
        except (NotImplementedError, OSError):
            return

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "hatchling",
                "build",
                "-t",
                "wheel",
                "-d",
                str(distribution),
            ],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode != 0
        assert list(distribution.glob("*.whl")) == []
        assert outside.read_text(encoding="utf-8") == "secret = True\n"


def test_wheel_build_rejects_force_include_module_replacement() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        project = root / "project"
        distribution = root / "dist"
        distribution.mkdir()
        unrelated_wheel = distribution / "unrelated-1.0-py3-none-any.whl"
        unrelated_wheel.write_text("must remain untouched\n", encoding="utf-8")
        shutil.copytree(
            PROJECT_ROOT,
            project,
            ignore=shutil.ignore_patterns(
                ".venv",
                "__pycache__",
                "*.pyc",
                "dist",
                ".pytest_cache",
            ),
        )
        source = project / "src" / "anamnesis" / "runtime_contract.py"
        original = source.read_bytes()
        injected = root / "injected.py"
        injected.write_text("raise AssertionError('injected')\n", encoding="utf-8")
        pyproject = project / "pyproject.toml"
        pyproject.write_text(
            pyproject.read_text(encoding="utf-8")
            + "\n[tool.hatch.build.targets.wheel.force-include]\n"
            + f'"{injected.as_posix()}" = "anamnesis/runtime_contract.py"\n',
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "hatchling",
                "build",
                "-t",
                "wheel",
                "-d",
                str(distribution),
            ],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode != 0
        assert set(distribution.glob("*.whl")) == {unrelated_wheel}
        assert unrelated_wheel.read_text(encoding="utf-8") == (
            "must remain untouched\n"
        )
        assert source.read_bytes() == original
        assert injected.read_text(encoding="utf-8") == (
            "raise AssertionError('injected')\n"
        )


def test_wheel_verifier_rejects_record_digest_tampering() -> None:
    from hatch_build import _expected_modules, _verify_wheel

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        distribution = root / "dist"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "hatchling",
                "build",
                "-t",
                "wheel",
                "-d",
                str(distribution),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        wheel = next(distribution.glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            entries = {
                item.filename: (item, archive.read(item.filename))
                for item in archive.infolist()
            }
        record_name = next(name for name in entries if name.endswith("/RECORD"))
        rows = list(
            csv.reader(io.StringIO(entries[record_name][1].decode("utf-8"), newline=""))
        )
        target = next(row for row in rows if row[0].startswith("anamnesis/"))
        target[1] = "sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        entries[record_name] = (
            entries[record_name][0],
            buffer.getvalue().encode("utf-8"),
        )
        with zipfile.ZipFile(wheel, "w") as archive:
            for info, payload in entries.values():
                archive.writestr(info, payload)

        try:
            _verify_wheel(wheel, _expected_modules())
        except RuntimeError as exc:
            assert "RECORD" in str(exc)
        else:
            raise AssertionError("tampered RECORD was accepted")


def test_pep517_prepared_metadata_cannot_inject_wheel_payload() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        project = root / "project"
        metadata_root = root / "metadata"
        distribution = root / "dist"
        shutil.copytree(
            PROJECT_ROOT,
            project,
            ignore=shutil.ignore_patterns(
                ".venv",
                "__pycache__",
                "*.pyc",
                "dist",
                ".pytest_cache",
            ),
        )
        metadata_root.mkdir()
        prepared = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import hatchling.build as backend,sys; "
                    "print('PREPARED=' + "
                    "backend.prepare_metadata_for_build_wheel(sys.argv[1]))"
                ),
                str(metadata_root),
            ],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert prepared.returncode == 0, prepared.stdout + prepared.stderr
        names = [
            line.removeprefix("PREPARED=")
            for line in prepared.stdout.splitlines()
            if line.startswith("PREPARED=")
        ]
        assert len(names) == 1
        prepared_directory = metadata_root / names[0]
        metadata = prepared_directory / "METADATA"
        marker = b"X-Private-Build-Injection: rejected\n"
        metadata.write_bytes(metadata.read_bytes() + marker)
        injected = prepared_directory / "licenses" / "private.pem"
        injected.parent.mkdir()
        injected.write_text("must remain outside the wheel\n", encoding="utf-8")

        built = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import hatchling.build as backend,sys; "
                    "print('WHEEL=' + backend.build_wheel("
                    "sys.argv[2], metadata_directory=sys.argv[1]))"
                ),
                str(prepared_directory),
                str(distribution),
            ],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert built.returncode == 0, built.stdout + built.stderr
        wheels = list(distribution.glob("*.whl"))
        assert len(wheels) == 1
        with zipfile.ZipFile(wheels[0]) as archive:
            payloads = [archive.read(name) for name in archive.namelist()]
            assert not any("licenses" in name for name in archive.namelist())
        assert marker not in payloads
        assert injected.read_text(encoding="utf-8") == (
            "must remain outside the wheel\n"
        )
