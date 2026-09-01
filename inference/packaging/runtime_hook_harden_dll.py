"""Harden DLL resolution before importing a frozen Inference runner.

This hook is deliberately stdlib-only. Qualification runners must not gain a
dependency on the Voice package merely to establish the Windows loader policy.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path


if sys.platform != "win32":  # pragma: no cover - executed by Windows bootloader
    raise RuntimeError("Inference DLL hardening requires native Windows")

library_directory = Path(
    getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)
).resolve()
if not library_directory.is_dir():  # pragma: no cover - frozen runtime only
    raise RuntimeError("frozen Inference library directory is unavailable")

set_default_directories = ctypes.WinDLL(  # type: ignore[attr-defined]
    "kernel32", use_last_error=True
).SetDefaultDllDirectories
set_default_directories.argtypes = [wintypes.DWORD]
set_default_directories.restype = wintypes.BOOL
if not set_default_directories(0x00001000):  # LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    raise ctypes.WinError()  # pragma: no cover - native failure path

# Keep the handle alive for the lifetime of the process. The selected policy
# excludes the current directory and PATH while retaining application/System32
# and explicitly registered user directories.
_inference_dll_directory_handle = os.add_dll_directory(str(library_directory))
