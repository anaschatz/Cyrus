from __future__ import annotations

import os
import shutil
import sys
import unittest

import _bootstrap  # noqa: F401


@unittest.skipUnless(sys.platform == "win32", "native Windows 11 verification required")
class NativeWindowsSmokeTests(unittest.TestCase):
    def test_win32_user32_is_available(self) -> None:
        import ctypes
        self.assertIsNotNone(ctypes.windll.user32)

    def test_tray_mutex_and_dll_hardening_apis_are_available(self) -> None:
        import ctypes
        self.assertIsNotNone(ctypes.windll.shell32.Shell_NotifyIconW)
        self.assertIsNotNone(ctypes.windll.kernel32.CreateMutexW)
        self.assertIsNotNone(ctypes.windll.kernel32.SetDefaultDllDirectories)


@unittest.skipUnless(
    sys.platform == "win32"
    and os.environ.get("CYRUS_NATIVE_RELEASE_HOST_TEST") == "1",
    "explicit native Windows 11 release-host qualification required",
)
class NativeQualificationHostTests(unittest.TestCase):
    def test_release_host_is_windows_11_x64_python_311(self) -> None:
        from jarvis_voice.qualification import host_qualification

        self.assertEqual(host_qualification()["status"], "ready")


@unittest.skipUnless(sys.platform == "win32" and shutil.which("signtool"), "native Windows SignTool verification required")
class NativeSigningToolSmokeTests(unittest.TestCase):
    def test_signtool_is_explicitly_available(self) -> None:
        self.assertIsNotNone(shutil.which("signtool"))


@unittest.skipUnless(sys.platform == "win32" and shutil.which("nvidia-smi"), "native Windows NVIDIA CUDA verification required")
class NativeCudaSmokeTests(unittest.TestCase):
    def test_nvidia_inventory_is_available_without_process_enumeration(self) -> None:
        from jarvis_voice.gpu import NvidiaSmiInventory
        self.assertGreaterEqual(len(NvidiaSmiInventory().devices()), 1)


if __name__ == "__main__":
    unittest.main()
