# Build only on native macOS 14+ arm64 with the pinned CPython environment.

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

if sys.platform != "darwin":
    raise SystemExit("The macOS spec requires native macOS.")

root = Path(SPEC).resolve().parents[1]
source = root / "src"
orchestrator_source = root / "orchestrator" / "src"
helper = Path(os.environ["JARVIS_MACOS_WHISPER_SERVER"]).resolve(strict=True)
profile = Path(os.environ["JARVIS_MACOS_BUILD_PROFILE"]).resolve(strict=True)
identity = os.environ["JARVIS_MACOS_CODESIGN_IDENTITY"]

datas = [(str(profile), ".")]
binaries = [(str(helper), "helpers")]
hiddenimports = [
    "jarvis_orchestrator.jcs",
    "jarvis_orchestrator.scheduler_crypto",
    "websockets.sync.client",
]

for package in ("sounddevice", "_sounddevice_data", "soxr", "onnxruntime", "websockets", "cryptography"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

analysis = Analysis(
    [str(root / "packaging" / "macos_entry.py")],
    pathex=[str(source), str(orchestrator_source)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "tensorflow", "gradio", "faster_whisper", "ctranslate2", "cuda"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="JARVIS Voice",
    target_arch="arm64",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=True,
    codesign_identity=identity,
    entitlements_file=str(root / "packaging" / "macos-entitlements.plist"),
)
collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="JARVIS Voice",
)
app = BUNDLE(
    collect,
    name="JARVIS Voice.app",
    icon=None,
    bundle_identifier="com.jarvis.voice",
    version="0.7.0",
    codesign_identity=identity,
    entitlements_file=str(root / "packaging" / "macos-entitlements.plist"),
    info_plist={
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": "JARVIS Voice uses the microphone only while push-to-talk is held.",
    },
)
