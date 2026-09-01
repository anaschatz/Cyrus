"""Read-only GPU inventory and deterministic CUDA admission policy."""

from __future__ import annotations

import csv
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .errors import BackendError


@dataclass(frozen=True, slots=True)
class GpuDevice:
    index: int
    name: str
    total_vram_mb: int
    free_vram_mb: int
    driver_version: str


class GpuInventory(Protocol):
    def devices(self) -> tuple[GpuDevice, ...]: ...


class NvidiaSmiInventory:
    """Query only non-sensitive device fields; never enumerate GPU processes."""

    QUERY_ARGUMENTS = (
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    )

    def __init__(
        self,
        runner: Callable[[Sequence[str]], str] | None = None,
        *,
        executable: Path | None = None,
    ) -> None:
        selected = "nvidia-smi" if executable is None else os.fspath(executable)
        if (
            not selected
            or "\x00" in selected
            or (executable is not None and not executable.is_absolute())
        ):
            raise BackendError(
                "cuda.inventory_path_invalid",
                "NVIDIA inventory path is invalid",
                subsystem="cuda",
            )
        self._query = (selected, *self.QUERY_ARGUMENTS)
        self._runner = runner or self._run

    @staticmethod
    def _run(command: Sequence[str]) -> str:
        try:
            completed = subprocess.run(
                list(command),
                check=True,
                capture_output=True,
                timeout=5,
                shell=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
            raise BackendError("cuda.inventory_unavailable", "NVIDIA inventory is unavailable", subsystem="cuda") from exc
        if not isinstance(completed.stdout, bytes) or len(completed.stdout) > 65_536:
            raise BackendError(
                "cuda.inventory_malformed",
                "NVIDIA inventory response is malformed",
                subsystem="cuda",
            )
        try:
            return completed.stdout.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise BackendError(
                "cuda.inventory_malformed",
                "NVIDIA inventory response is malformed",
                subsystem="cuda",
            ) from exc

    def devices(self) -> tuple[GpuDevice, ...]:
        try:
            rows = csv.reader(
                self._runner(self._query).splitlines(),
                skipinitialspace=True,
            )
            devices = tuple(
                GpuDevice(
                    index=int(row[0].strip()),
                    name=row[1].strip(),
                    total_vram_mb=int(row[2].strip()),
                    free_vram_mb=int(row[3].strip()),
                    driver_version=row[4].strip(),
                )
                for row in rows
                if row
            )
        except (IndexError, TypeError, ValueError, csv.Error) as exc:
            raise BackendError("cuda.inventory_malformed", "NVIDIA inventory response is malformed", subsystem="cuda") from exc
        if not devices:
            raise BackendError("cuda.no_devices", "No NVIDIA GPUs were reported", subsystem="cuda")
        return devices


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    allowed: bool
    code: str
    device_index: int
    total_vram_mb: int | None
    free_vram_mb: int | None
    device_name: str | None
    driver_version: str | None
    reserve_vram_mb: int
    available_for_voice_mb: int | None
    minimum_free_vram_mb: int
    duration_seconds: float


class GpuAdmissionPolicy:
    """A point-in-time headroom heuristic, not a cross-process reservation."""

    def __init__(self, inventory: GpuInventory, clock: Callable[[], float] = time.perf_counter) -> None:
        self.inventory = inventory
        self.clock = clock

    def evaluate(self, *, device_index: int, minimum_free_vram_mb: int, reserve_for_other_workloads_mb: int) -> AdmissionDecision:
        started = self.clock()
        devices = self.inventory.devices()
        device = next((candidate for candidate in devices if candidate.index == device_index), None)
        duration = self.clock() - started
        if device is None:
            return AdmissionDecision(
                False,
                "cuda.invalid_device_index",
                device_index,
                None,
                None,
                None,
                None,
                reserve_for_other_workloads_mb,
                None,
                minimum_free_vram_mb,
                duration,
            )
        available = device.free_vram_mb - reserve_for_other_workloads_mb
        allowed = available >= minimum_free_vram_mb
        return AdmissionDecision(
            allowed,
            "cuda.admission_allowed" if allowed else "cuda.insufficient_free_vram",
            device_index,
            device.total_vram_mb,
            device.free_vram_mb,
            device.name,
            device.driver_version,
            reserve_for_other_workloads_mb,
            available,
            minimum_free_vram_mb,
            duration,
        )


def require_admission(decision: AdmissionDecision) -> None:
    if decision.allowed:
        return
    raise BackendError(
        decision.code,
        "CUDA admission was denied",
        subsystem="cuda",
        retryable=True,
        safe_context={"device_index": decision.device_index},
    )
