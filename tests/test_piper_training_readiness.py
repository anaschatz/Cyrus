from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from jarvis_voice.piper_training_readiness import (
    MINIMUM_SYSTEM_RAM_BYTES,
    MINIMUM_WORK_DISK_BYTES,
    NVIDIA_QUERY,
    piper_training_readiness,
)


class PiperTrainingReadinessTests(unittest.TestCase):
    def _probe(self, **overrides):
        defaults = {
            "platform_name": "linux",
            "machine": "x86_64",
            "kernel_release": "6.6.87.2-microsoft-standard-WSL2",
            "python_version": (3, 11, 14),
            "pointer_bits": 64,
            "environment": {"WSL_INTEROP": "/run/WSL/1_interop"},
            "memory_total_bytes": MINIMUM_SYSTEM_RAM_BYTES,
            "disk_usage": lambda path: SimpleNamespace(free=MINIMUM_WORK_DISK_BYTES),
            "inventory_runner": lambda command: "0, NVIDIA Test GPU, 560.76, 8192, 7000\n",
            "nvidia_smi": "/usr/lib/wsl/lib/nvidia-smi",
        }
        defaults.update(overrides)
        with tempfile.TemporaryDirectory() as temp:
            return piper_training_readiness(Path(temp), **defaults)

    def test_ready_profile_is_passive_and_allows_lock_materialization(self) -> None:
        commands: list[tuple[str, ...]] = []
        report = self._probe(
            inventory_runner=lambda command: commands.append(tuple(command))
            or "0, NVIDIA Test GPU, 560.76, 8192, 7000\n"
        )
        self.assertTrue(report["ready"])
        self.assertTrue(report["lock_profile"]["materialization_allowed"])
        self.assertEqual(commands[0][1:], NVIDIA_QUERY)
        self.assertFalse(report["privacy"]["gpu_processes_queried"])
        self.assertFalse(report["installation_performed"])
        self.assertFalse(report["training_started"])

    def test_mac_dry_run_does_not_invoke_gpu_runner(self) -> None:
        calls: list[object] = []
        report = self._probe(
            platform_name="darwin",
            machine="arm64",
            kernel_release="25.0.0",
            environment={},
            inventory_runner=lambda command: calls.append(command) or "",
        )
        self.assertEqual(report["code"], "tts.f1_training.wsl2_host_required")
        self.assertEqual(calls, [])
        self.assertFalse(report["privacy"]["network_used"])

    def test_driver_gate_matches_cuda_126_minimum(self) -> None:
        report = self._probe(inventory_runner=lambda command: "0, GPU, 560.75, 8192, 7000\n")
        self.assertEqual(report["code"], "tts.f1_training.driver_too_old")
        self.assertFalse(report["ready"])

    def test_wrong_python_has_specific_block_code(self) -> None:
        report = self._probe(python_version=(3, 12, 0))
        self.assertEqual(report["code"], "tts.f1_training.python_311_required")
        self.assertFalse(report["ready"])

    def test_free_vram_gate_does_not_enumerate_processes(self) -> None:
        report = self._probe(inventory_runner=lambda command: "0, GPU, 600.1, 8192, 6400\n")
        self.assertEqual(report["code"], "tts.f1_training.vram_insufficient")
        self.assertFalse(report["ready"])

    def test_disk_and_ram_fail_closed(self) -> None:
        low_ram = self._probe(memory_total_bytes=MINIMUM_SYSTEM_RAM_BYTES - 1)
        low_disk = self._probe(disk_usage=lambda path: SimpleNamespace(free=MINIMUM_WORK_DISK_BYTES - 1))
        self.assertEqual(low_ram["code"], "tts.f1_training.system_ram_insufficient")
        self.assertEqual(low_disk["code"], "tts.f1_training.disk_space_insufficient")

    def test_gpu_inventory_is_strict(self) -> None:
        report = self._probe(inventory_runner=lambda command: "unexpected,data\n")
        self.assertEqual(report["code"], "tts.f1_training.gpu_inventory_malformed")
        self.assertFalse(report["ready"])


if __name__ == "__main__":
    unittest.main()
