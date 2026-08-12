"""Offline native-extension materialization for the pinned Piper MPS trainer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from .errors import BackendError
from .macos.compact_greek_tts_pilot import PIPER_REVISION, PIPER_VERSION


Runner = Callable[..., subprocess.CompletedProcess[str]]
RevisionProbe = Callable[[Path], str | None]


def _error(code: str, message: str) -> BackendError:
    return BackendError(code, message, subsystem="tts")


def _run(
    argv: Sequence[str],
    *,
    runner: Runner,
    cwd: Path | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            tuple(argv),
            cwd=None if cwd is None else str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _error(
            "tts.f1_mps.extension_command_failed",
            "Piper native-extension command failed",
        ) from exc


def _source_revision(source_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(source_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_piper_mps_extension(
    python_executable: str | Path,
    source_root: str | Path,
    *,
    runner: Runner = subprocess.run,
    revision_probe: RevisionProbe = _source_revision,
    compiler_path: str | None = None,
) -> dict[str, object]:
    """Compile and install only Piper's missing monotonic-alignment extension."""

    python = Path(python_executable).expanduser()
    source = Path(source_root).expanduser()
    if not python.is_absolute() or not python.is_file():
        raise _error("tts.f1_mps.python_invalid", "Training Python is invalid")
    if not source.is_absolute() or not source.is_dir() or source.is_symlink():
        raise _error("tts.f1_mps.source_invalid", "Pinned Piper source is invalid")
    if revision_probe(source) != PIPER_REVISION:
        raise _error(
            "tts.f1_mps.source_revision_mismatch",
            "Pinned Piper source revision does not match",
        )
    core = source / "src/piper/train/vits/monotonic_align/core.pyx"
    if not core.is_file() or core.is_symlink():
        raise _error("tts.f1_mps.source_core_missing", "Piper training core is missing")
    clang = shutil.which("clang") if compiler_path is None else compiler_path
    if clang is None:
        raise _error("tts.f1_mps.compiler_missing", "Clang is required")

    environment_probe = _run(
        (
            str(python),
            "-c",
            "import importlib.metadata as m,json,sysconfig;"
            "print(json.dumps({'piper':m.version('piper-tts'),"
            "'cython':m.version('cython'),'numpy':m.version('numpy'),"
            "'platlib':sysconfig.get_path('platlib')}))",
        ),
        runner=runner,
    )
    try:
        environment = json.loads(environment_probe.stdout)
        platlib = Path(environment["platlib"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise _error(
            "tts.f1_mps.environment_probe_invalid",
            "Training environment probe was invalid",
        ) from exc
    if environment.get("piper") != PIPER_VERSION:
        raise _error(
            "tts.f1_mps.piper_version_mismatch",
            "Training environment has the wrong Piper version",
        )
    environment_root = python.parent.parent.absolute()
    try:
        inside_environment = os.path.commonpath((environment_root, platlib)) == str(
            environment_root
        )
    except ValueError:
        inside_environment = False
    if not platlib.is_absolute() or not inside_environment:
        raise _error(
            "tts.f1_mps.platlib_invalid",
            "Training site-packages path is outside the isolated environment",
        )

    target_directory = (
        platlib / "piper/train/vits/monotonic_align/monotonic_align"
    )
    parent_package = target_directory.parent / "__init__.py"
    if not parent_package.is_file() or parent_package.is_symlink():
        raise _error(
            "tts.f1_mps.piper_package_invalid",
            "Installed Piper training package is incomplete",
        )

    with tempfile.TemporaryDirectory(prefix="jarvis-piper-mps-") as temporary:
        build_directory = Path(temporary)
        build_core = build_directory / "core.pyx"
        shutil.copyfile(core, build_core)
        _run(
            (
                str(python),
                "-m",
                "Cython.Build.Cythonize",
                "-i",
                str(build_core),
            ),
            runner=runner,
            cwd=build_directory,
            timeout=300,
        )
        extensions = tuple(build_directory.glob("core*.so"))
        if len(extensions) != 1 or not extensions[0].is_file():
            raise _error(
                "tts.f1_mps.extension_build_invalid",
                "Piper native-extension build output was invalid",
            )
        extension = extensions[0]
        target_directory.mkdir(parents=True, exist_ok=True)
        target = target_directory / extension.name
        extension_hash = _sha256(extension)
        if target.exists():
            if target.is_file() and not target.is_symlink() and _sha256(target) == extension_hash:
                duplicate = True
            else:
                raise _error(
                    "tts.f1_mps.extension_target_exists",
                    "A different Piper native extension already exists",
                )
        else:
            temporary_target = target_directory / f".{target.name}.{os.getpid()}.tmp"
            try:
                shutil.copy2(extension, temporary_target)
                os.link(temporary_target, target)
            finally:
                temporary_target.unlink(missing_ok=True)
            duplicate = False

    _run(
        (
            str(python),
            "-c",
            "from piper.train.vits.monotonic_align import maximum_path;"
            "assert callable(maximum_path)",
        ),
        runner=runner,
    )
    return {
        "ready": True,
        "code": (
            "tts.f1_mps.extension_already_ready"
            if duplicate
            else "tts.f1_mps.extension_materialized"
        ),
        "source_revision": PIPER_REVISION,
        "piper_version": PIPER_VERSION,
        "cython_version": environment.get("cython"),
        "numpy_version": environment.get("numpy"),
        "extension": target.name,
        "extension_sha256": extension_hash,
        "duplicate": duplicate,
        "network_used": False,
        "training_started": False,
    }
