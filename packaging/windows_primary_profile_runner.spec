# Build only on native Windows x64 with the provisioned CPython 3.11 toolchain.

from pathlib import Path


root = Path(SPEC).resolve().parents[1]
inference_source = root / "inference" / "src"

analysis = Analysis(
    [str(root / "inference" / "scripts" / "windows_primary_profile_runner.py")],
    pathex=[str(inference_source)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        str(root / "inference" / "packaging" / "runtime_hook_harden_dll.py")
    ],
    excludes=[
        "torch",
        "tensorflow",
        "mlx",
        "mlx_lm",
        "numpy",
        "openai",
        "sounddevice",
        "faster_whisper",
        "ctranslate2",
        "onnxruntime",
        "websockets",
    ],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="cyrus-windows-primary-profile-runner",
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
