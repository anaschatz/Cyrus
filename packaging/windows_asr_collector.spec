# Build only on native Windows 11 x64 with the provisioned Voice CPython 3.11
# environment.  This is a single console EXE because the Inference manifest
# hashes one collector file; all native libraries remain inside that EXE.

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


root = Path(SPEC).resolve().parents[1]
voice_source = root / "src"
inference_source = root / "inference" / "src"
sys.path.insert(0, str(voice_source))

from jarvis_voice.windows.cuda_bundle import (  # noqa: E402
    is_cuda_runtime_name,
    load_cuda_bundle,
)


cuda_manifest_value = os.environ.get("CYRUS_WINDOWS_CUDA_BUNDLE_MANIFEST", "")
if not cuda_manifest_value:
    raise SystemExit("CYRUS_WINDOWS_CUDA_BUNDLE_MANIFEST is required")
cuda_bundle = load_cuda_bundle(Path(cuda_manifest_value))

datas = []
binaries = []
hiddenimports = [
    "jarvis_inference.windows_asr_corpus",
    "multiprocessing.popen_spawn_win32",
]
for package in ("faster_whisper", "ctranslate2", "numpy"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# The wheel's partial CUDA payload is not authoritative.  Remove every CUDA or
# cuDNN runtime DLL collected from site-packages, then add only the validated,
# hash-bound private bundle at the onefile extraction root.  ctranslate2.dll
# and non-CUDA support libraries remain package-owned.
binaries = [item for item in binaries if not is_cuda_runtime_name(item[0])]
binaries += [(str(item.path), ".") for item in cuda_bundle.files]

analysis = Analysis(
    [str(root / "packaging" / "windows_asr_collector_entry.py")],
    pathex=[str(voice_source), str(inference_source)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(root / "packaging" / "runtime_hook_harden_dll.py")],
    excludes=["torch", "tensorflow", "gradio", "sounddevice", "onnxruntime"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="cyrus-windows-asr-collector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=True,
    manifest=str(root / "packaging" / "windows-manifest.xml"),
    uac_admin=False,
    uac_uiaccess=False,
)
