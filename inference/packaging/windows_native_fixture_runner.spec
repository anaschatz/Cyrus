# Build only on native Windows 11 x64 with the locked Inference CPython 3.11
# environment.  This deliberately produces one console executable: the parent
# qualifier redirects its stdout/stderr and accepts only its strict JSON file.

from pathlib import Path


root = Path(SPEC).resolve().parents[2]
source = root / "inference" / "src"

analysis = Analysis(
    [str(root / "inference" / "scripts" / "windows_native_fixture_runner.py")],
    pathex=[str(source)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        str(root / "inference" / "packaging" / "runtime_hook_harden_dll.py")
    ],
    excludes=["torch", "tensorflow", "gradio"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="native-fixture-runner",
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
