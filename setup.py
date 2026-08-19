"""Build-time hardening for the Voice wheel.

The source tree can contain user-owned recovery copies such as ``module 2.py``.
They are not importable Python modules and must never enter a distribution.
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


VOICE_PACKAGE = "jarvis_voice"
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_BUILD_ROOT = PROJECT_ROOT / "build"
VOICE_SOURCE_ROOT = PROJECT_ROOT / "src" / VOICE_PACKAGE
VOICE_EGG_INFO_ROOT = PROJECT_ROOT / "src" / f"{VOICE_PACKAGE}.egg-info"
EGG_INFO_MEMBERS = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
}
DIST_INFO_MEMBERS = {
    "METADATA",
    "entry_points.txt",
    "top_level.txt",
}


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _is_link_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(status, "st_file_attributes", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
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
        raise SetupError("project build root must not be a symlink")
    for component in candidate.relative_to(build_root).parts:
        current /= component
        if _is_link_or_reparse(current):
            raise SetupError(f"{purpose} must not traverse a link or reparse point")
    resolved_root = build_root.resolve()
    resolved_candidate = candidate.resolve()
    if not _is_within(resolved_candidate, resolved_root):
        raise SetupError(f"{purpose} resolved outside the project build tree")
    return candidate


def _validate_voice_source_tree() -> None:
    source_parent = VOICE_SOURCE_ROOT.parent
    if (
        _is_link_or_reparse(source_parent)
        or _is_link_or_reparse(VOICE_SOURCE_ROOT)
        or not VOICE_SOURCE_ROOT.is_dir()
    ):
        raise SetupError("Voice source root must be a real directory")
    for current_raw, directories, filenames in os.walk(
        VOICE_SOURCE_ROOT,
        followlinks=False,
    ):
        current = Path(current_raw)
        for name in (*directories, *filenames):
            if _is_link_or_reparse(current / name):
                raise SetupError("Voice source tree must not contain links")
        if "__init__.py" in filenames:
            relative = current.relative_to(source_parent)
            if any(not component.isidentifier() for component in relative.parts):
                raise SetupError("Voice source contains an invalid package name")


def _expected_voice_modules() -> set[Path]:
    expected: set[Path] = set()
    source_parent = VOICE_SOURCE_ROOT.parent
    for current_raw, _directories, filenames in os.walk(
        VOICE_SOURCE_ROOT,
        followlinks=False,
    ):
        current = Path(current_raw)
        if "__init__.py" not in filenames:
            continue
        package = current.relative_to(source_parent)
        for filename in filenames:
            candidate = Path(filename)
            if candidate.suffix == ".py" and candidate.stem.isidentifier():
                expected.add(package / candidate.name)
    return expected


def _verify_exact_voice_tree(root: Path) -> None:
    if _is_link_or_reparse(root) or not root.is_dir():
        raise SetupError("Voice package output was not produced safely")
    expected = _expected_voice_modules()
    actual: set[Path] = set()
    for candidate in root.rglob("*"):
        if _is_link_or_reparse(candidate):
            raise SetupError("Voice package output contains a link")
        if candidate.is_dir():
            relative = candidate.relative_to(root)
            if any(not component.isidentifier() for component in relative.parts):
                raise SetupError("Voice package output contains an invalid directory")
            continue
        if not candidate.is_file():
            raise SetupError("Voice package output contains an unsupported entry")
        relative = Path(VOICE_PACKAGE) / candidate.relative_to(root)
        actual.add(relative)
    if actual != expected:
        raise SetupError("Voice package output does not match the source module set")


def _flat_regular_members(root: Path, purpose: str) -> set[str]:
    if _is_link_or_reparse(root) or not root.is_dir():
        raise SetupError(f"{purpose} must be a real directory")
    members: set[str] = set()
    with os.scandir(root) as entries:
        for entry in entries:
            if _is_link_or_reparse(Path(entry.path)) or not entry.is_file(
                follow_symlinks=False
            ):
                raise SetupError(f"{purpose} contains an unsupported entry")
            members.add(entry.name)
    return members


def _clean_project_egg_info() -> None:
    """Discard only recognized generated metadata; preserve and reject surprises."""

    if not VOICE_EGG_INFO_ROOT.exists() and not _is_link_or_reparse(
        VOICE_EGG_INFO_ROOT
    ):
        return
    members = _flat_regular_members(VOICE_EGG_INFO_ROOT, "project egg-info")
    if not members.issubset(EGG_INFO_MEMBERS):
        raise SetupError("project egg-info contains an unexpected member")
    shutil.rmtree(VOICE_EGG_INFO_ROOT)


def _verify_exact_flat_tree(
    root: Path,
    expected: set[str],
    purpose: str,
) -> None:
    if _flat_regular_members(root, purpose) != expected:
        raise SetupError(f"{purpose} does not match its exact allowlist")


_validate_voice_source_tree()


class StrictVoiceBuildPy(_build_py):
    """Build only importable modules into a clean, isolated package tree."""

    def find_package_modules(
        self,
        package: str,
        package_dir: str,
    ) -> list[tuple[str, str, str]]:
        if (
            package != VOICE_PACKAGE
            and not package.startswith(f"{VOICE_PACKAGE}.")
        ) or not all(
            component.isidentifier() for component in package.split(".")
        ):
            raise SetupError("build_py received an unexpected package")
        modules = super().find_package_modules(package, package_dir)
        return [
            (found_package, module, source)
            for found_package, module, source in modules
            if module.isidentifier()
        ]

    def _validated_packages(self) -> tuple[str, ...]:
        packages = tuple(self.packages or ())
        if not packages or any(
            (
                package != VOICE_PACKAGE
                and not package.startswith(f"{VOICE_PACKAGE}.")
            )
            or not all(
                component.isidentifier() for component in package.split(".")
            )
            for package in packages
        ):
            raise SetupError("Voice package discovery escaped its allowlist")
        if self.py_modules:
            raise SetupError("top-level Python modules are not allowed")
        return packages

    def _clean_build_lib(self) -> Path:
        build_root = _owned_build_path(self.build_lib, "build_lib")
        if build_root.exists():
            if not build_root.is_dir():
                raise SetupError("build_lib is not a directory")
            shutil.rmtree(build_root)
        return build_root

    def run(self) -> None:
        if self.editable_mode:
            super().run()
            return
        self._validated_packages()
        build_root = self._clean_build_lib()
        super().run()
        if not build_root.is_dir():
            raise SetupError("build_lib was not produced")
        children = {candidate.name for candidate in build_root.iterdir()}
        if children != {VOICE_PACKAGE}:
            raise SetupError("build_lib contains an unexpected sibling package")
        _verify_exact_voice_tree(build_root / VOICE_PACKAGE)


class StrictVoiceBdistWheel(_bdist_wheel):
    """Create wheels only from fresh project-owned build and staging trees."""

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
        allowed = {
            VOICE_PACKAGE,
            expected_dist_info.name,
        }
        if dist_info != expected_dist_info:
            raise SetupError("wheel dist-info path escaped its exact location")
        if _is_link_or_reparse(staging) or not staging.is_dir():
            raise SetupError("bdist_wheel staging must be a real directory")
        children = tuple(staging.iterdir())
        if {candidate.name for candidate in children} != allowed or any(
            _is_link_or_reparse(candidate) or not candidate.is_dir()
            for candidate in children
        ):
            raise SetupError("bdist_wheel staging payload is not exact")
        _verify_exact_voice_tree(staging / VOICE_PACKAGE)
        _verify_exact_flat_tree(
            VOICE_EGG_INFO_ROOT,
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
        "bdist_wheel": StrictVoiceBdistWheel,
        "build_py": StrictVoiceBuildPy,
    }
)
