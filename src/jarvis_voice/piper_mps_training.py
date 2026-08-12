"""Fail-closed native MPS readiness and command profile for STOMA F1."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .errors import BackendError
from .macos.compact_greek_tts_pilot import (
    PIPER_CHECKPOINT_DIRECTORY,
    PIPER_REVISION,
    PIPER_RUNTIME_ARTIFACTS,
    PIPER_TARGET_SAMPLE_RATE,
    PIPER_VERSION,
    STOMA_SPEAKER_CLIPS,
    validate_artifact_set,
)
from .piper_training_handoff import build_piper_f1_handoff


GIB = 1024**3
MINIMUM_TOTAL_MEMORY_BYTES = 16 * GIB
MINIMUM_SMOKE_DISK_BYTES = 8 * GIB
MINIMUM_FULL_TRAINING_DISK_BYTES = 20 * GIB
LOCKED_TORCH_VERSION = "2.12.1"
LOCKED_LIGHTNING_VERSION = "2.6.1"
INITIAL_MPS_EPOCHS = 25
MPS_ENVIRONMENT = {
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.90",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.80",
}


def _error(code: str, message: str) -> BackendError:
    return BackendError(code, message, subsystem="tts")


def _directory(value: str | Path, *, code: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise _error(code, "MPS training directory is invalid")
    return path.resolve()


def _total_memory() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError) as exc:
        raise _error("tts.f1_mps.memory_probe_failed", "System memory probe failed") from exc


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _mps_probe() -> tuple[bool, bool]:
    try:
        import torch
    except ImportError:
        return False, False
    return bool(torch.backends.mps.is_built()), bool(torch.backends.mps.is_available())


def _source_revision(source_root: Path) -> str | None:
    if source_root.is_symlink() or not source_root.is_dir():
        return None
    core = source_root / "src/piper/train/vits/monotonic_align/core.pyx"
    if not core.is_file() or core.is_symlink():
        return None
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


def _wheel_has_training_core(wheel: Path) -> bool:
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    return any(
        name.endswith("piper/train/vits/monotonic_align/core.pyx")
        or (
            "piper/train/vits/monotonic_align/core." in name
            and name.endswith((".so", ".dylib"))
        )
        for name in names
    )


def piper_mps_training_readiness(
    data_directory: str | Path,
    artifact_root: str | Path,
    work_directory: str | Path,
    *,
    source_root: str | Path | None = None,
    platform_name: str | None = None,
    machine: str | None = None,
    python_version: tuple[int, int, int] | None = None,
    total_memory_bytes: int | None = None,
    disk_usage: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
    compiler_path: str | None = None,
    distribution_probe: Callable[[str], str | None] = _distribution_version,
    mps_probe: Callable[[], tuple[bool, bool]] = _mps_probe,
    source_revision_probe: Callable[[Path], str | None] = _source_revision,
) -> dict[str, object]:
    """Inspect M3 training readiness without installing or loading a checkpoint."""

    data = _directory(data_directory, code="tts.f1_mps.data_directory_invalid")
    artifacts = _directory(artifact_root, code="tts.f1_mps.artifact_root_invalid")
    work = _directory(work_directory, code="tts.f1_mps.work_directory_invalid")
    selected_platform = sys.platform if platform_name is None else platform_name
    selected_machine = platform.machine() if machine is None else machine
    selected_python = sys.version_info[:3] if python_version is None else python_version
    memory = _total_memory() if total_memory_bytes is None else int(total_memory_bytes)
    free_disk = int(getattr(disk_usage(work), "free"))
    clang = shutil.which("clang") if compiler_path is None else compiler_path

    host_ready = (
        selected_platform == "darwin"
        and selected_machine == "arm64"
        and selected_python[:2] == (3, 11)
        and memory >= MINIMUM_TOTAL_MEMORY_BYTES
        and clang is not None
    )
    handoff = build_piper_f1_handoff(data, artifacts)
    inputs_ready = handoff["totals"]["files"] == STOMA_SPEAKER_CLIPS + 5
    runtime_root = artifacts / "runtime"
    runtime = validate_artifact_set(runtime_root, PIPER_RUNTIME_ARTIFACTS)
    wheel = runtime_root / PIPER_RUNTIME_ARTIFACTS[0].relative_path
    wheel_core_ready = runtime.ready and _wheel_has_training_core(wheel)

    source_revision = None
    if source_root is not None:
        raw_source = Path(source_root).expanduser()
        source_revision = source_revision_probe(raw_source)
    source_ready = source_revision == PIPER_REVISION

    versions = {
        "piper-tts": distribution_probe("piper-tts"),
        "torch": distribution_probe("torch"),
        "lightning": distribution_probe("lightning"),
    }
    dependencies_ready = (
        versions["piper-tts"] == PIPER_VERSION
        and versions["torch"] == LOCKED_TORCH_VERSION
        and versions["lightning"] == LOCKED_LIGHTNING_VERSION
    )
    mps_built, mps_available = mps_probe() if dependencies_ready else (False, False)
    smoke_disk_ready = free_disk >= MINIMUM_SMOKE_DISK_BYTES
    full_disk_ready = free_disk >= MINIMUM_FULL_TRAINING_DISK_BYTES
    materialization_ready = host_ready and inputs_ready and runtime.ready and smoke_disk_ready
    smoke_ready = materialization_ready and source_ready and dependencies_ready and mps_built and mps_available
    full_training_ready = smoke_ready and full_disk_ready

    if selected_platform != "darwin" or selected_machine != "arm64":
        code = "tts.f1_mps.apple_silicon_required"
    elif selected_python[:2] != (3, 11):
        code = "tts.f1_mps.python_311_required"
    elif memory < MINIMUM_TOTAL_MEMORY_BYTES:
        code = "tts.f1_mps.memory_insufficient"
    elif clang is None:
        code = "tts.f1_mps.compiler_missing"
    elif not inputs_ready or not runtime.ready:
        code = "tts.f1_mps.inputs_invalid"
    elif not smoke_disk_ready:
        code = "tts.f1_mps.smoke_disk_insufficient"
    elif not source_ready and not wheel_core_ready:
        code = "tts.f1_mps.source_checkout_required"
    elif not dependencies_ready:
        code = "tts.f1_mps.environment_materialization_required"
    elif not mps_built or not mps_available:
        code = "tts.f1_mps.backend_unavailable"
    elif not full_disk_ready:
        code = "tts.f1_mps.full_training_disk_insufficient"
    else:
        code = "tts.f1_mps.training_ready"

    return {
        "ready": full_training_ready,
        "smoke_ready": smoke_ready,
        "materialization_ready": materialization_ready,
        "code": code,
        "host": {
            "platform": selected_platform,
            "machine": selected_machine,
            "python": ".".join(str(value) for value in selected_python),
            "total_memory_bytes": memory,
            "compiler_ready": clang is not None,
            "ready": host_ready,
        },
        "inputs": {
            "ready": inputs_ready and runtime.ready,
            "files": handoff["totals"]["files"],
            "bytes": handoff["totals"]["bytes"],
            "runtime_wheel_ready": runtime.ready,
            "runtime_wheel_contains_training_core": wheel_core_ready,
        },
        "source": {
            "required_revision": PIPER_REVISION,
            "observed_revision": source_revision,
            "ready": source_ready,
        },
        "environment": {
            "versions": versions,
            "required": {
                "piper-tts": PIPER_VERSION,
                "torch": LOCKED_TORCH_VERSION,
                "lightning": LOCKED_LIGHTNING_VERSION,
            },
            "dependencies_ready": dependencies_ready,
            "mps_built": mps_built,
            "mps_available": mps_available,
        },
        "disk": {
            "free_bytes": free_disk,
            "minimum_smoke_bytes": MINIMUM_SMOKE_DISK_BYTES,
            "minimum_full_training_bytes": MINIMUM_FULL_TRAINING_DISK_BYTES,
            "smoke_ready": smoke_disk_ready,
            "full_training_ready": full_disk_ready,
        },
        "profile": {
            "batch_size": 1,
            "devices": 1,
            "precision": "32-true",
            "num_workers": 0,
            "initial_epochs": INITIAL_MPS_EPOCHS,
            "main_llm_must_be_stopped": True,
            "mps_experimental": True,
            "environment": dict(MPS_ENVIRONMENT),
        },
        "network_used": False,
        "installation_performed": False,
        "checkpoint_loaded": False,
        "training_started": False,
    }


def piper_mps_training_commands(
    python_executable: str | Path,
    training_entrypoint: str | Path,
    data_directory: str | Path,
    artifact_root: str | Path,
    work_directory: str | Path,
) -> dict[str, object]:
    """Return argv arrays for the one-batch gate and bounded initial run."""

    raw_paths = tuple(
        Path(value).expanduser()
        for value in (
            python_executable,
            training_entrypoint,
            data_directory,
            artifact_root,
            work_directory,
        )
    )
    if not all(path.is_absolute() for path in raw_paths):
        raise _error("tts.f1_mps.training_path_invalid", "MPS training paths must be absolute")
    python = raw_paths[0].absolute()
    entrypoint = raw_paths[1].resolve(strict=False)
    if not entrypoint.is_file() or entrypoint.is_symlink():
        raise _error(
            "tts.f1_mps.training_entrypoint_invalid",
            "MPS training entrypoint is invalid",
        )
    data, artifacts, work = (
        path.resolve(strict=False) for path in raw_paths[2:]
    )
    checkpoint = artifacts / "checkpoint" / PIPER_CHECKPOINT_DIRECTORY / "epoch=1999-step=485528.ckpt"
    shared: list[str] = [
        str(python),
        str(entrypoint),
        "fit",
        "--trainer.accelerator",
        "mps",
        "--trainer.devices",
        "1",
        "--trainer.precision",
        "32-true",
        "--data.voice_name",
        "el_GR-jarvis-f1-medium",
        "--data.csv_path",
        str(data / "metadata.csv"),
        "--data.audio_dir",
        str(data / "audio"),
        "--model.sample_rate",
        str(PIPER_TARGET_SAMPLE_RATE),
        "--model.mos_metric",
        "none",
        "--data.espeak_voice",
        "el",
        "--data.batch_size",
        "1",
        "--data.validation_split",
        "0.1",
        "--data.num_test_examples",
        "2",
        "--data.num_workers",
        "0",
        "--model.warmstart_ckpt",
        str(checkpoint),
    ]
    smoke = [
        *shared,
        "--trainer.fast_dev_run",
        "true",
        "--trainer.default_root_dir",
        str(work / "smoke"),
        "--data.cache_dir",
        str(work / "cache"),
        "--data.config_path",
        str(work / "smoke" / "el_GR-jarvis-f1-medium.onnx.json"),
    ]
    initial = [
        *shared,
        "--trainer.max_epochs",
        str(INITIAL_MPS_EPOCHS),
        "--trainer.default_root_dir",
        str(work / "runs"),
        "--trainer.log_every_n_steps",
        "25",
        "--data.cache_dir",
        str(work / "cache"),
        "--data.config_path",
        str(work / "export" / "el_GR-jarvis-f1-medium.onnx.json"),
    ]
    return {
        "ready": False,
        "code": "tts.f1_mps.smoke_required",
        "environment": dict(MPS_ENVIRONMENT),
        "smoke_argv": smoke,
        "initial_training_argv": initial,
        "initial_epochs": INITIAL_MPS_EPOCHS,
        "training_must_not_start_before_smoke": True,
        "network_used": False,
        "installation_performed": False,
        "training_started": False,
    }
