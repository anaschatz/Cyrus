from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.gpu import GpuAdmissionPolicy, GpuDevice, NvidiaSmiInventory


class StaticInventory:
    def __init__(self, *devices: GpuDevice) -> None:
        self.value = tuple(devices)

    def devices(self) -> tuple[GpuDevice, ...]:
        return self.value


class GpuPolicyTests(unittest.TestCase):
    def test_nvidia_smi_query_excludes_processes_and_sensitive_ids(self) -> None:
        commands: list[tuple[str, ...]] = []
        inventory = NvidiaSmiInventory(lambda command: commands.append(tuple(command)) or "0, Test GPU, 8192, 6000, 555.1\n")
        device = inventory.devices()[0]
        self.assertEqual((device.index, device.total_vram_mb, device.free_vram_mb), (0, 8192, 6000))
        command = " ".join(commands[0])
        self.assertNotIn("process", command)
        self.assertNotIn("uuid", command)

    def test_headroom_subtracts_other_workload_reserve(self) -> None:
        policy = GpuAdmissionPolicy(StaticInventory(GpuDevice(0, "GPU", 12_000, 7_000, "driver")))
        allowed = policy.evaluate(device_index=0, minimum_free_vram_mb=4_000, reserve_for_other_workloads_mb=3_000)
        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.available_for_voice_mb, 4_000)
        denied = policy.evaluate(device_index=0, minimum_free_vram_mb=4_001, reserve_for_other_workloads_mb=3_000)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.code, "cuda.insufficient_free_vram")

    def test_device_index_is_never_changed_implicitly(self) -> None:
        policy = GpuAdmissionPolicy(StaticInventory(GpuDevice(2, "GPU", 8_000, 8_000, "driver")))
        decision = policy.evaluate(device_index=0, minimum_free_vram_mb=0, reserve_for_other_workloads_mb=0)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "cuda.invalid_device_index")
        self.assertEqual(decision.device_index, 0)


if __name__ == "__main__":
    unittest.main()
