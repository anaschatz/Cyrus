"""Passive readiness gate for STOMA F1 Piper training inside WSL2.

The probe never installs packages, loads a model, enumerates GPU processes, or
allocates VRAM.  It only inspects host, memory, disk, and aggregate GPU fields.
"""

from __future__ import annotations

import os
import platform
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .errors import BackendError


GIB = 1024**3
PIPER_VERSION = "1.6.0"
PIPER_REVISION = "f04d52c5528ac7cf2d73757f57990ff490f75005"
PYTORCH_VERSION = "2.12.1"
PYTORCH_CUDA_PROFILE = "cu126"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"
MINIMUM_WINDOWS_DRIVER = (560, 76)
MINIMUM_TOTAL_VRAM_MB = 8_000
MINIMUM_FREE_VRAM_MB = 6_500
MINIMUM_SYSTEM_RAM_BYTES = 16 * GIB
MINIMUM_WORK_DISK_BYTES = 25 * GIB

NVIDIA_QUERY = (
    "--query-gpu=index,name,driver_version,memory.total,memory.free",
    "--format=csv,noheader,nounits",
)


@dataclass(frozen=True, slots=True)
class TrainingGpu:
    index: int
    name: str
    driver_version: str
    total_vram_mb: int
    free_vram_mb: int


def _error(code: str, message: str) -> BackendError:
    return BackendError(code, message, subsystem="tts")


def _driver_tuple(value: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(part) for part in value.strip().split("."))
    except ValueError as exc:
        raise _error("tts.f1_training.driver_version_invalid", "NVIDIA driver version is invalid") from exc
    if len(parts) < 2:
        raise _error("tts.f1_training.driver_version_invalid", "NVIDIA driver version is invalid")
    return parts


def _driver_ready(value: str) -> bool:
    installed = _driver_tuple(value)
    required = MINIMUM_WINDOWS_DRIVER + (0,) * max(0, len(installed) - len(MINIMUM_WINDOWS_DRIVER))
    installed = installed + (0,) * max(0, len(required) - len(installed))
    return installed >= required


def _parse_inventory(output: str) -> tuple[TrainingGpu, ...]:
    devices: list[TrainingGpu] = []
    try:
        for raw_line in output.splitlines():
            if not raw_line.strip():
                continue
            fields = [field.strip() for field in raw_line.split(",")]
            if len(fields) != 5:
                raise ValueError("field count")
            index, name, driver, total, free = fields
            if not name or not driver:
                raise ValueError("empty field")
            devices.append(TrainingGpu(int(index), name, driver, int(total), int(free)))
    except ValueError as exc:
        raise _error("tts.f1_training.gpu_inventory_malformed", "NVIDIA inventory is malformed") from exc
    if not devices:
        raise _error("tts.f1_training.gpu_unavailable", "No NVIDIA GPU was reported")
    return tuple(devices)


def _read_memory_total() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError) as exc:
        raise _error("tts.f1_training.memory_probe_failed", "System memory probe failed") from exc
    raise _error("tts.f1_training.memory_probe_failed", "System memory probe failed")


def _nvidia_smi_path() -> str:
    candidate = shutil.which("nvidia-smi")
    if candidate:
        return candidate
    wsl_path = Path("/usr/lib/wsl/lib/nvidia-smi")
    if wsl_path.is_file() and not wsl_path.is_symlink():
        return str(wsl_path)
    raise _error("tts.f1_training.nvidia_smi_missing", "nvidia-smi is unavailable inside WSL2")


def _run_inventory(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _error("tts.f1_training.gpu_probe_failed", "Passive NVIDIA probe failed") from exc
    return completed.stdout


def piper_training_readiness(
    work_root: str | Path,
    *,
    device_index: int = 0,
    platform_name: str | None = None,
    machine: str | None = None,
    kernel_release: str | None = None,
    python_version: tuple[int, int, int] | None = None,
    pointer_bits: int | None = None,
    environment: Mapping[str, str] | None = None,
    memory_total_bytes: int | None = None,
    disk_usage: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
    inventory_runner: Callable[[Sequence[str]], str] | None = None,
    nvidia_smi: str | None = None,
) -> dict[str, object]:
    """Return a no-install WSL2/CUDA training gate report."""

    root = Path(work_root).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise _error("tts.f1_training.work_root_invalid", "Training work root is invalid")
    selected_platform = sys.platform if platform_name is None else platform_name
    selected_machine = platform.machine() if machine is None else machine
    selected_release = platform.release() if kernel_release is None else kernel_release
    selected_python = sys.version_info[:3] if python_version is None else python_version
    selected_bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    selected_environment = os.environ if environment is None else environment

    release_lower = selected_release.lower()
    wsl2 = (
        selected_platform == "linux"
        and "microsoft" in release_lower
        and ("wsl2" in release_lower or "microsoft-standard" in release_lower)
        and (
            bool(selected_environment.get("WSL_INTEROP"))
            or bool(selected_environment.get("WSL_DISTRO_NAME"))
            or "wsl2" in release_lower
        )
    )
    host_ready = (
        wsl2
        and selected_machine in {"x86_64", "amd64"}
        and selected_python[:2] == (3, 11)
        and selected_bits == 64
    )
    try:
        free_disk = int(getattr(disk_usage(root), "free"))
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        raise _error("tts.f1_training.disk_probe_failed", "Training disk probe failed") from exc
    if memory_total_bytes is None:
        total_ram = _read_memory_total() if wsl2 else 0
    else:
        total_ram = int(memory_total_bytes)

    gpu: TrainingGpu | None = None
    gpu_error: str | None = None
    if host_ready:
        try:
            runner = _run_inventory if inventory_runner is None else inventory_runner
            executable = _nvidia_smi_path() if nvidia_smi is None else nvidia_smi
            devices = _parse_inventory(runner((executable, *NVIDIA_QUERY)))
            gpu = next((device for device in devices if device.index == device_index), None)
            if gpu is None:
                gpu_error = "tts.f1_training.gpu_index_missing"
        except BackendError as exc:
            gpu_error = exc.code

    try:
        driver_ready = gpu is not None and _driver_ready(gpu.driver_version)
    except BackendError as exc:
        driver_ready = False
        gpu_error = exc.code
    vram_ready = (
        gpu is not None
        and gpu.total_vram_mb >= MINIMUM_TOTAL_VRAM_MB
        and gpu.free_vram_mb >= MINIMUM_FREE_VRAM_MB
    )
    ram_ready = total_ram >= MINIMUM_SYSTEM_RAM_BYTES
    disk_ready = free_disk >= MINIMUM_WORK_DISK_BYTES
    ready = host_ready and gpu_error is None and driver_ready and vram_ready and ram_ready and disk_ready

    if not wsl2:
        code = "tts.f1_training.wsl2_host_required"
    elif selected_machine not in {"x86_64", "amd64"} or selected_bits != 64:
        code = "tts.f1_training.x86_64_host_required"
    elif selected_python[:2] != (3, 11):
        code = "tts.f1_training.python_311_required"
    elif gpu_error is not None:
        code = gpu_error
    elif not driver_ready:
        code = "tts.f1_training.driver_too_old"
    elif not vram_ready:
        code = "tts.f1_training.vram_insufficient"
    elif not ram_ready:
        code = "tts.f1_training.system_ram_insufficient"
    elif not disk_ready:
        code = "tts.f1_training.disk_space_insufficient"
    else:
        code = "tts.f1_training.readiness_passed"

    return {
        "ready": ready,
        "code": code,
        "host": {
            "platform": selected_platform,
            "machine": selected_machine,
            "kernel_release": selected_release,
            "python": ".".join(str(part) for part in selected_python),
            "pointer_bits": selected_bits,
            "wsl2": wsl2,
            "ready": host_ready,
        },
        "gpu": None if gpu is None else {**asdict(gpu), "driver_ready": driver_ready, "vram_ready": vram_ready},
        "gpu_error": gpu_error,
        "system_ram_bytes": total_ram,
        "system_ram_ready": ram_ready,
        "work_disk_free_bytes": free_disk,
        "work_disk_ready": disk_ready,
        "requirements": {
            "minimum_total_vram_mb": MINIMUM_TOTAL_VRAM_MB,
            "minimum_free_vram_mb": MINIMUM_FREE_VRAM_MB,
            "minimum_system_ram_bytes": MINIMUM_SYSTEM_RAM_BYTES,
            "minimum_work_disk_bytes": MINIMUM_WORK_DISK_BYTES,
        },
        "lock_profile": {
            "piper_version": PIPER_VERSION,
            "piper_revision": PIPER_REVISION,
            "torch_version": PYTORCH_VERSION,
            "cuda_wheel_profile": PYTORCH_CUDA_PROFILE,
            "torch_index_url": PYTORCH_INDEX_URL,
            "minimum_windows_driver": ".".join(str(part) for part in MINIMUM_WINDOWS_DRIVER),
            "materialization_allowed": ready,
        },
        "privacy": {
            "gpu_processes_queried": False,
            "gpu_uuid_queried": False,
            "network_used": False,
        },
        "installation_performed": False,
        "driver_changed": False,
        "cuda_toolkit_installed": False,
        "training_started": False,
    }
