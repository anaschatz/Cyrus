"""Closed-payload wheel builder for the Inference appliance package.

The checkout can contain user-owned recovery copies and stale setuptools
metadata.  A release wheel must contain only the explicitly reviewed Python
modules and the minimal wheel metadata generated during this invocation.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.command.build_py import build_py as _build_py
from setuptools.errors import SetupError

PACKAGE = "jarvis_inference"
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_BUILD_ROOT = PROJECT_ROOT / "build"
SOURCE_ROOT = PROJECT_ROOT / "src" / PACKAGE
EGG_INFO_ROOT = PROJECT_ROOT / "src" / f"{PACKAGE}.egg-info"

# This is deliberately a closed, reviewed module set. Adding a runtime module
# requires updating the allowlist and the packaging regression together.
MODULES = frozenset(
    {
        "__init__.py",
        "__main__.py",
        "circuit.py",
        "cli.py",
        "contracts.py",
        "durable.py",
        "engine.py",
        "errors.py",
        "gateway.py",
        "gemma4.py",
        "http_backend.py",
        "limits.py",
        "llama_engine.py",
        "macos_fixture.py",
        "manifest.py",
        "mlx_engine.py",
        "mlx_worker_runtime.py",
        "private_temp.py",
        "primary_profile.py",
        "primary_quality_client.py",
        "primary_qualification.py",
        "profile_preflight.py",
        "profile_performance.py",
        "profile_quality.py",
        "profile_report.py",
        "qualification.py",
        "qwen35_sampling_candidate.py",
        "rendering.py",
        "schema_subset.py",
        "strict_json.py",
        "windows_primary_fixture.py",
        "windows_primary_metrics.py",
        "windows_primary_report.py",
        "windows_primary_runner.py",
        "windows_asr_corpus.py",
        "windows_acl.py",
        "windows_fixture.py",
        "windows_process.py",
    }
)
EGG_INFO_MEMBERS = frozenset(
    {
        "PKG-INFO",
        "SOURCES.txt",
        "dependency_links.txt",
        "entry_points.txt",
        "requires.txt",
        "top_level.txt",
    }
)
DIST_INFO_MEMBERS = frozenset(
    {
        "METADATA",
        "entry_points.txt",
        "top_level.txt",
    }
)


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _owned_build_path(value: str | os.PathLike[str], purpose: str) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else PROJECT_ROOT / raw
    candidate = Path(os.path.abspath(candidate))
    build_root = Path(os.path.abspath(PROJECT_BUILD_ROOT))
    if not _is_within(candidate, build_root):
        raise SetupError(f"{purpose} must stay in the project build tree")
    current = build_root
    if _is_link_or_reparse(current):
        raise SetupError("project build root must not be a link or reparse point")
    for component in candidate.relative_to(build_root).parts:
        current /= component
        if _is_link_or_reparse(current):
            raise SetupError(f"{purpose} must not traverse a link or reparse point")
    resolved_root = build_root.resolve()
    resolved_candidate = candidate.resolve()
    if not _is_within(resolved_candidate, resolved_root):
        raise SetupError(f"{purpose} resolved outside the project build tree")
    return candidate


def _flat_regular_members(root: Path, purpose: str) -> set[str]:
    if _is_link_or_reparse(root) or not root.is_dir():
        raise SetupError(f"{purpose} must be a real directory")
    members: set[str] = set()
    with os.scandir(root) as entries:
        for entry in entries:
            path = Path(entry.path)
            if _is_link_or_reparse(path) or not entry.is_file(
                follow_symlinks=False
            ):
                raise SetupError(f"{purpose} contains an unsupported entry")
            members.add(entry.name)
    return members


def _verify_exact_flat_tree(
    root: Path,
    expected: frozenset[str] | set[str],
    purpose: str,
) -> None:
    if _flat_regular_members(root, purpose) != set(expected):
        raise SetupError(f"{purpose} does not match its exact allowlist")


def _validate_source_tree() -> None:
    source_parent = SOURCE_ROOT.parent
    if (
        _is_link_or_reparse(source_parent)
        or _is_link_or_reparse(SOURCE_ROOT)
        or not SOURCE_ROOT.is_dir()
    ):
        raise SetupError("Inference source root must be a real directory")
    observed: set[str] = set()
    for current_raw, directories, filenames in os.walk(
        SOURCE_ROOT,
        followlinks=False,
    ):
        current = Path(current_raw)
        for name in (*directories, *filenames):
            if _is_link_or_reparse(current / name):
                raise SetupError("Inference source tree must not contain links")
        relative_directory = current.relative_to(SOURCE_ROOT)
        if relative_directory.parts and "__init__.py" in filenames:
            raise SetupError("Inference source contains an unexpected subpackage")
        if not relative_directory.parts:
            observed.update(
                name
                for name in filenames
                if Path(name).suffix == ".py" and Path(name).stem.isidentifier()
            )
    if observed != MODULES:
        raise SetupError("Inference source modules do not match their allowlist")


def _validate_existing_egg_info() -> None:
    if not EGG_INFO_ROOT.exists() and not _is_link_or_reparse(EGG_INFO_ROOT):
        return
    members = _flat_regular_members(EGG_INFO_ROOT, "project egg-info")
    if not members.issubset(EGG_INFO_MEMBERS):
        raise SetupError("project egg-info contains an unexpected member")


def _clean_project_egg_info() -> None:
    """Remove only recognized generated metadata; preserve surprises."""

    _validate_existing_egg_info()
    if EGG_INFO_ROOT.exists():
        shutil.rmtree(EGG_INFO_ROOT)


def _verify_exact_package_tree(root: Path) -> None:
    if _is_link_or_reparse(root) or not root.is_dir():
        raise SetupError("Inference package output was not produced safely")
    actual: set[str] = set()
    for candidate in root.rglob("*"):
        if _is_link_or_reparse(candidate):
            raise SetupError("Inference package output contains a link")
        if candidate.is_dir():
            raise SetupError("Inference package output contains a directory")
        if not candidate.is_file():
            raise SetupError("Inference package output contains an unsupported entry")
        relative = candidate.relative_to(root)
        if len(relative.parts) != 1:
            raise SetupError("Inference package output escaped its flat allowlist")
        actual.add(relative.name)
    if actual != MODULES:
        raise SetupError("Inference package output does not match its allowlist")


_validate_source_tree()
_validate_existing_egg_info()


class StrictInferenceBuildPy(_build_py):
    """Build exactly one flat Python package into fresh owned output."""

    def find_package_modules(
        self,
        package: str,
        package_dir: str,
    ) -> list[tuple[str, str, str]]:
        if package != PACKAGE:
            raise SetupError("build_py received an unexpected package")
        modules = super().find_package_modules(package, package_dir)
        selected = [
            (found_package, module, source)
            for found_package, module, source in modules
            if f"{module}.py" in MODULES
        ]
        if {f"{module}.py" for _, module, _ in selected} != MODULES:
            raise SetupError("build_py module discovery did not match its allowlist")
        return selected

    def _validate_packages(self) -> None:
        if tuple(self.packages or ()) != (PACKAGE,):
            raise SetupError("Inference package discovery escaped its allowlist")
        if self.py_modules:
            raise SetupError("top-level Python modules are not allowed")

    def run(self) -> None:
        if self.editable_mode:
            raise SetupError("editable Inference builds are not permitted")
        self._validate_packages()
        build_root = _owned_build_path(self.build_lib, "build_lib")
        if build_root.exists():
            if not build_root.is_dir():
                raise SetupError("build_lib is not a directory")
            shutil.rmtree(build_root)
        super().run()
        if not build_root.is_dir():
            raise SetupError("build_lib was not produced")
        children = tuple(build_root.iterdir())
        if (
            len(children) != 1
            or children[0].name != PACKAGE
            or _is_link_or_reparse(children[0])
            or not children[0].is_dir()
        ):
            raise SetupError("build_lib contains an unexpected payload")
        _verify_exact_package_tree(children[0])


class StrictInferenceBdistWheel(_bdist_wheel):
    """Build only from fresh project-owned staging and exact metadata."""

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.skip_build:
            raise SetupError("bdist_wheel --skip-build is not permitted")
        _owned_build_path(self.bdist_dir, "bdist_wheel staging")
        if self.license_paths:
            raise SetupError("wheel license payload is not permitted")

    def run(self) -> None:
        if self.skip_build:
            raise SetupError("bdist_wheel --skip-build is not permitted")
        _clean_project_egg_info()
        staging = _owned_build_path(self.bdist_dir, "bdist_wheel staging")
        if staging.exists():
            if not staging.is_dir():
                raise SetupError("bdist_wheel staging is not a directory")
            shutil.rmtree(staging)
        super().run()

    def write_wheelfile(
        self,
        wheelfile_base: str,
        generator: str = "setuptools",
    ) -> None:
        staging = _owned_build_path(self.bdist_dir, "bdist_wheel staging")
        dist_info = _owned_build_path(wheelfile_base, "wheel dist-info")
        expected_dist_info = staging / f"{self.wheel_dist_name}.dist-info"
        if dist_info != expected_dist_info:
            raise SetupError("wheel dist-info path escaped its exact location")
        allowed = {PACKAGE, expected_dist_info.name}
        if _is_link_or_reparse(staging) or not staging.is_dir():
            raise SetupError("bdist_wheel staging must be a real directory")
        children = tuple(staging.iterdir())
        if {candidate.name for candidate in children} != allowed or any(
            _is_link_or_reparse(candidate) or not candidate.is_dir()
            for candidate in children
        ):
            raise SetupError("bdist_wheel staging payload is not exact")
        _verify_exact_package_tree(staging / PACKAGE)
        _verify_exact_flat_tree(
            EGG_INFO_ROOT,
            EGG_INFO_MEMBERS,
            "project egg-info",
        )
        _verify_exact_flat_tree(
            dist_info,
            DIST_INFO_MEMBERS,
            "wheel dist-info",
        )
        super().write_wheelfile(wheelfile_base, generator)
        _verify_exact_flat_tree(
            dist_info,
            DIST_INFO_MEMBERS | {"WHEEL"},
            "wheel dist-info",
        )


setup(
    cmdclass={
        "bdist_wheel": StrictInferenceBdistWheel,
        "build_py": StrictInferenceBuildPy,
    }
)
