"""Argument-free entry point for the sealed Windows ASR collector."""

from __future__ import annotations

import ctypes
import multiprocessing
import sys
from pathlib import Path

from jarvis_voice.windows.cuda_bundle import REQUIRED_CUDA_DLLS


def _load_sealed_cuda_runtime() -> tuple[object, ...]:
    """Load every required bundled DLL by exact frozen-root path."""

    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        raise RuntimeError("sealed CUDA collector requires frozen Windows")
    root = Path(getattr(sys, "_MEIPASS", ""))
    if not root.is_dir():
        raise RuntimeError("sealed CUDA collector root is unavailable")
    handles = []
    # Load split cuDNN/CUBLAS libraries before the CTranslate2 extension.  The
    # runtime hook has already removed PATH/current-directory DLL resolution.
    for name in sorted(REQUIRED_CUDA_DLLS):
        candidate = root / name
        if not candidate.is_file():
            raise RuntimeError("sealed CUDA runtime is incomplete")
        handles.append(ctypes.WinDLL(str(candidate)))  # type: ignore[attr-defined]
    import ctranslate2  # type: ignore[import-not-found]

    ctranslate2.get_cuda_device_count()
    return tuple(handles)


try:
    _CUDA_RUNTIME_HANDLES = _load_sealed_cuda_runtime()
    from jarvis_voice.windows.asr_collector import main as _collector_main
except Exception:  # noqa: BLE001 - the runner's failure channel is exit-only
    # This runner is consumed only through a content-free parent protocol.
    # Import/DLL failures happen before the collector's own exception gate, so
    # collapse them here instead of allowing a traceback or native path to
    # reach an operator-controlled console.
    _CUDA_RUNTIME_HANDLES = ()

    def _collector_main() -> int:
        return 2


def _entrypoint() -> int:
    try:
        multiprocessing.freeze_support()
        return _collector_main() if len(sys.argv) == 1 else 2
    except Exception:  # noqa: BLE001 - keep the entrypoint content-free
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
