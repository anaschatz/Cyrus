"""Bounded native GPU and power probes for Windows ASR qualification."""

from __future__ import annotations

import csv
import math
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .asr_collector_contract import CollectorFailure, PowerContract


@dataclass(frozen=True, slots=True)
class GpuObservation:
    free_vram_mib: int
    driver_version: str
    power_limit_watts: float
    clock_policy: str
    throttle_reasons: int


class GpuProbe(Protocol):
    def observe(self, device_index: int) -> GpuObservation: ...


class PowerProbe(Protocol):
    def assert_matches(
        self,
        expected: PowerContract,
        device_index: int,
    ) -> GpuObservation: ...


def _run_fixed_command(arguments: Sequence[str], *, timeout: float = 10.0) -> bytes:
    try:
        completed = subprocess.run(
            list(arguments),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            shell=False,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CollectorFailure("native_probe_failed") from exc
    if (
        completed.returncode != 0
        or not isinstance(completed.stdout, bytes)
        or len(completed.stdout) > 65_536
    ):
        raise CollectorFailure("native_probe_failed")
    return completed.stdout


def _probe_bytes(value: bytes | str, code: str) -> bytes:
    try:
        raw = value if isinstance(value, bytes) else value.encode("utf-8", errors="strict")
    except (AttributeError, UnicodeError) as exc:
        raise CollectorFailure(code) from exc
    if not raw or len(raw) > 65_536:
        raise CollectorFailure(code)
    return raw


def _probe_ascii(value: bytes | str, code: str) -> str:
    try:
        return _probe_bytes(value, code).decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise CollectorFailure(code) from exc


class NativeGpuProbe:
    """Read only bounded, non-process NVIDIA fields from one explicit index."""

    QUERY_ARGUMENTS = (
        (
            "--query-gpu=index,memory.free,driver_version,power.limit,"
            "clocks.applications.graphics,clocks.applications.memory,"
            "clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,"
            "clocks_event_reasons.hw_thermal_slowdown,"
            "clocks_event_reasons.hw_power_brake_slowdown"
        ),
        "--format=csv,noheader,nounits",
    )

    def __init__(
        self,
        executable: Path,
        runner: Callable[[Sequence[str]], bytes | str] = _run_fixed_command,
    ) -> None:
        if not executable.is_absolute():
            raise CollectorFailure("native_probe_path_invalid")
        self.command = (str(executable), *self.QUERY_ARGUMENTS)
        self.runner = runner

    def observe(self, device_index: int) -> GpuObservation:
        try:
            rows = list(
                csv.reader(
                    _probe_ascii(
                        self.runner(self.command),
                        "gpu_probe_invalid",
                    ).splitlines(),
                    skipinitialspace=True,
                )
            )
            matching = [row for row in rows if row and int(row[0].strip()) == device_index]
            if len(matching) != 1 or len(matching[0]) != 10:
                raise ValueError
            row = matching[0]
            free = int(row[1].strip())
            driver = row[2].strip()
            power = float(row[3].strip())
            application_graphics = row[4].strip()
            application_memory = row[5].strip()
            if not application_graphics or not application_memory:
                raise ValueError
            clock_policy = (
                f"applications_graphics_mhz={application_graphics};"
                f"applications_memory_mhz={application_memory}"
            )
            states = [item.strip().casefold() for item in row[6:]]
            if any(state not in {"active", "not active"} for state in states):
                raise ValueError
            reasons = states.count("active")
        except (TypeError, ValueError, csv.Error) as exc:
            raise CollectorFailure("gpu_probe_invalid") from exc
        if free < 0 or not driver or not math.isfinite(power) or power <= 0:
            raise CollectorFailure("gpu_probe_invalid")
        return GpuObservation(free, driver, power, clock_policy, reasons)


class NativePowerProbe:
    POWER_GUID = re.compile(
        rb"([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
        rb"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})"
    )
    CUDA_VERSION = re.compile(rb"CUDA Version:\s*([0-9]+\.[0-9]+)", re.ASCII)

    def __init__(
        self,
        gpu: GpuProbe,
        powercfg: Path,
        nvidia_smi: Path,
        command_runner: Callable[[Sequence[str]], bytes | str] = _run_fixed_command,
        ac_reader: Callable[[], bool] | None = None,
    ) -> None:
        self.gpu = gpu
        if not powercfg.is_absolute() or not nvidia_smi.is_absolute():
            raise CollectorFailure("native_probe_path_invalid")
        self.powercfg = str(powercfg)
        self.nvidia_smi = str(nvidia_smi)
        self.command_runner = command_runner
        self.ac_reader = ac_reader or self._native_ac_state

    @staticmethod
    def _native_ac_state() -> bool:  # pragma: no cover - native Windows only
        try:
            import ctypes

            class SystemPowerStatus(ctypes.Structure):
                _fields_ = [
                    ("ACLineStatus", ctypes.c_ubyte),
                    ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte),
                    ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong),
                ]

            status = SystemPowerStatus()
            if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                raise OSError
            return status.ACLineStatus == 1
        except Exception as exc:
            raise CollectorFailure("power_probe_failed") from exc

    def assert_matches(
        self,
        expected: PowerContract,
        device_index: int,
    ) -> GpuObservation:
        raw = _probe_bytes(
            self.command_runner((self.powercfg, "/getactivescheme")),
            "power_probe_failed",
        )
        match = self.POWER_GUID.search(raw)
        compute = _probe_bytes(
            self.command_runner((self.nvidia_smi,)),
            "power_probe_failed",
        )
        compute_match = self.CUDA_VERSION.search(compute)
        observation = self.gpu.observe(device_index)
        if (
            match is None
            or compute_match is None
            or match.group(1).decode("ascii").casefold() != expected.scheme_guid
            or self.ac_reader() is not True
            or observation.driver_version != expected.driver_version
            or f"CUDA {compute_match.group(1).decode('ascii')}" != expected.compute_runtime
            or observation.clock_policy != expected.gpu_clock_policy
            or observation.throttle_reasons != 0
            or not math.isclose(
                observation.power_limit_watts,
                expected.power_limit_watts,
                rel_tol=0,
                abs_tol=0.01,
            )
        ):
            raise CollectorFailure("power_identity_changed")
        return observation


__all__ = [
    "GpuObservation",
    "GpuProbe",
    "NativeGpuProbe",
    "NativePowerProbe",
    "PowerProbe",
]
