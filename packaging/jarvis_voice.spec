# Build only on native Windows 11 x64 with the approved CPython 3.11 environment.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

root = Path(SPEC).resolve().parents[1]
source = root / "src"

datas = []
binaries = []
hiddenimports = [
    "multiprocessing.popen_spawn_win32",
    "pythoncom",
    "pywintypes",
    "win32com.client",
    "websockets.asyncio.server",
    "websockets.sync.client",
]

# Runtime packages are collected from the already-approved, already-installed
# Windows environment. No model, VAD file, config, CUDA toolkit or voice pack is
# part of this list.
for package in (
    "faster_whisper",
    "ctranslate2",
    "sounddevice",
    "_sounddevice_data",
    "soxr",
    "onnxruntime",
    "websockets",
    "win32com",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

analysis = Analysis(
    [str(root / "packaging" / "windows_entry.py")],
    pathex=[str(source)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(root / "packaging" / "runtime_hook_harden_dll.py")],
    excludes=["torch", "tensorflow", "gradio"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="JARVIS Voice",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=True,
    manifest=str(root / "packaging" / "windows-manifest.xml"),
    version=str(root / "packaging" / "windows-version-info.txt"),
    uac_admin=False,
    uac_uiaccess=False,
)

collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="JARVIS Voice",
)
