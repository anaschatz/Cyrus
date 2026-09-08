"""Sealed, read-only prerequisite staging for the macOS CPU smoke profile."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from .. import PROTOCOL_VERSION, __version__
from ..config import VoiceConfig
from ..manifest import (
    ManifestValidationResult,
    model_manifest_for_profile,
    validate_model_directory,
)
from ..vad import validate_artifact
from .doctor import _default_voice_probe, _macos_version_tuple, _status
from .ptt import CoreGraphicsEventTapAdapter


WHEELHOUSE_SCHEMA_VERSION = 1
REQUIRED_TOP_LEVEL: Mapping[str, str] = {
    "faster-whisper": "1.2.1",
    "sounddevice": "0.5.3",
    "soxr": "1.1.0",
    "onnxruntime": "1.27.0",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CompletedProbe(Protocol):
    returncode: int
    stdout: str


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def probe_python(
    executable: str,
    *,
    runner: Callable[..., CompletedProbe] = subprocess.run,
) -> dict[str, object]:
    path = Path(executable).expanduser()
    if not path.is_absolute():
        return _status("blocked", "staging.python_absolute_path_required")
    if not path.is_file() or not os.access(path, os.X_OK):
        return _status("blocked", "staging.python_executable_missing")
    code = (
        "import json,platform,struct,sys;"
        "print(json.dumps({'platform':sys.platform,'machine':platform.machine(),"
        "'macos':platform.mac_ver()[0],'version':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro],"
        "'bits':struct.calcsize('P')*8},sort_keys=True))"
    )
    try:
        completed = runner(
            [str(path), "-I", "-S", "-c", code],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
            shell=False,
        )
        raw = json.loads(completed.stdout) if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, RecursionError):
        raw = None
    if not isinstance(raw, Mapping) or set(raw) != {"platform", "machine", "macos", "version", "bits"}:
        return _status("blocked", "staging.python_probe_failed")
    version = raw.get("version")
    if (
        not isinstance(version, list)
        or len(version) != 3
        or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in version)
    ):
        return _status("blocked", "staging.python_probe_failed")
    ready = (
        raw.get("platform") == "darwin"
        and raw.get("machine") == "arm64"
        and _macos_version_tuple(str(raw.get("macos"))) >= (14, 0)
        and version[:2] == [3, 11]
        and raw.get("bits") == 64
    )
    return _status(
        "ready" if ready else "blocked",
        "staging.python_311_arm64_ready" if ready else "staging.python_311_arm64_required",
        python=f"{version[0]}.{version[1]}.{version[2]}",
        machine=str(raw.get("machine")),
        macos_major_minor=".".join(str(item) for item in _macos_version_tuple(str(raw.get("macos")))),
        pointer_bits=raw.get("bits"),
        isolated_probe=True,
    )


def _valid_top_level(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {*REQUIRED_TOP_LEVEL, "websockets"}:
        return False
    if any(value.get(name) != pinned for name, pinned in REQUIRED_TOP_LEVEL.items()):
        return False
    websockets = value.get("websockets")
    return isinstance(websockets, str) and websockets.split(".", 1)[0] == "15"


def validate_wheelhouse(wheelhouse: str | None, manifest_path: str | None) -> dict[str, object]:
    if not wheelhouse and not manifest_path:
        return _status("blocked", "staging.wheelhouse_not_supplied", wheels_checked=0)
    if not wheelhouse or not manifest_path:
        return _status("blocked", "staging.wheelhouse_and_manifest_required", wheels_checked=0)
    root = Path(wheelhouse).expanduser()
    manifest_file = Path(manifest_path).expanduser()
    if not root.is_absolute() or not manifest_file.is_absolute():
        return _status("blocked", "staging.wheelhouse_absolute_paths_required", wheels_checked=0)
    if not root.is_dir() or not manifest_file.is_file() or root.is_symlink() or manifest_file.is_symlink():
        return _status("blocked", "staging.wheelhouse_missing", wheels_checked=0)
    try:
        if manifest_file.stat().st_size > 1_048_576:
            return _status("blocked", "staging.wheelhouse_manifest_invalid", wheels_checked=0)
        raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return _status("blocked", "staging.wheelhouse_manifest_unreadable", wheels_checked=0)
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema_version", "target", "top_level", "wheels"}
        or raw.get("schema_version") != WHEELHOUSE_SCHEMA_VERSION
        or raw.get("target") != {"python": "3.11", "platform": "macos_arm64"}
        or not _valid_top_level(raw.get("top_level"))
        or not isinstance(raw.get("wheels"), list)
        or not raw["wheels"]
        or len(raw["wheels"]) > 512
    ):
        return _status("blocked", "staging.wheelhouse_manifest_invalid", wheels_checked=0)
    expected: dict[str, tuple[int, str]] = {}
    for item in raw["wheels"]:
        if not isinstance(item, Mapping) or set(item) != {"filename", "size", "sha256"}:
            return _status("blocked", "staging.wheelhouse_manifest_invalid", wheels_checked=0)
        filename, size, digest = item.get("filename"), item.get("size"), item.get("sha256")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".whl")
            or filename in expected
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
        ):
            return _status("blocked", "staging.wheelhouse_manifest_invalid", wheels_checked=0)
        expected[filename] = (size, digest)
    try:
        actual = {path.name for path in root.iterdir() if path.is_file() and path.suffix == ".whl"}
    except OSError:
        return _status("blocked", "staging.wheelhouse_unreadable", wheels_checked=0)
    if actual != set(expected):
        return _status("blocked", "staging.wheelhouse_file_set_mismatch", wheels_checked=0)
    checked = 0
    for filename in sorted(expected):
        path = root / filename
        size, digest = expected[filename]
        try:
            if path.is_symlink() or path.stat().st_size != size:
                return _status("blocked", "staging.wheelhouse_artifact_mismatch", wheels_checked=checked)
            if _hash_file(path) != digest:
                return _status("blocked", "staging.wheelhouse_artifact_mismatch", wheels_checked=checked)
        except OSError:
            return _status("blocked", "staging.wheelhouse_unreadable", wheels_checked=checked)
        checked += 1
    return _status(
        "ready",
        "staging.wheelhouse_hash_locked",
        wheels_checked=checked,
        top_level_packages=len(REQUIRED_TOP_LEVEL) + 1,
        network_install_forbidden=True,
    )


def macos_staging_report(
    config: VoiceConfig,
    *,
    python_executable: str,
    wheelhouse: str | None = None,
    wheelhouse_manifest: str | None = None,
    python_runner: Callable[..., CompletedProbe] = subprocess.run,
    model_validator: Callable[[str | None], ManifestValidationResult] | None = None,
    permission_probe: Callable[[], bool] | None = None,
    voice_probe: Callable[[], set[str]] | None = None,
) -> dict[str, object]:
    python = probe_python(python_executable, runner=python_runner)
    manifest = model_manifest_for_profile(config.asr.model_profile)
    model_validation = (
        model_validator(config.whisper_model_directory)
        if model_validator is not None
        else validate_model_directory(config.whisper_model_directory, manifest)
    )
    model = _status(
        "ready" if model_validation.ok else "blocked",
        "staging.faster_whisper_model_valid" if model_validation.ok else model_validation.code,
        files_checked=model_validation.files_checked,
        model_profile=config.asr.model_profile,
        model_id=manifest.model_id,
        model_loaded=False,
    )
    artifact = config.vad.artifact
    vad_ok, vad_code = validate_artifact(artifact.path, artifact.size, artifact.sha256)
    vad = _status("ready" if vad_ok else "blocked", vad_code)
    wheelhouse_check = validate_wheelhouse(wheelhouse, wheelhouse_manifest)
    try:
        permitted = bool((permission_probe or CoreGraphicsEventTapAdapter().preflight)())
    except Exception:
        permitted = False
    ptt = _status(
        "ready" if permitted else "blocked",
        "staging.input_monitoring_ready" if permitted else "staging.input_monitoring_required",
        permission_prompted=False,
    )
    if config.macos.tts_backend == "greektts15_f1":
        missing: list[str] = []
        voices = _status("ready", "staging.greektts15_profile_selected", missing_locales=missing)
    else:
        configured = config.macos.system_voices
        try:
            installed = (voice_probe or _default_voice_probe)()
        except Exception:
            installed = set()
        missing = sorted(locale for locale in ("el-GR", "en-US") if not configured.get(locale) or configured.get(locale) not in installed)
        voices = _status(
            "ready" if not missing else "blocked",
            "staging.system_voices_ready" if not missing else "staging.system_voices_required",
            missing_locales=missing,
        )
    checks = {"python": python, "wheelhouse": wheelhouse_check, "model": model, "vad": vad, "ptt": ptt, "voices": voices}
    ready = all(item["status"] == "ready" for item in checks.values())
    if python["status"] != "ready":
        next_action = "staging.approval_python_311_required"
    elif wheelhouse_check["status"] != "ready":
        next_action = "staging.approval_wheelhouse_required"
    elif vad["status"] != "ready":
        next_action = "staging.approval_silero_artifact_required"
    elif voices["status"] != "ready":
        next_action = "staging.explicit_voice_mapping_required"
    elif ptt["status"] != "ready":
        next_action = "staging.manual_input_monitoring_required"
    else:
        next_action = "staging.offline_install_requires_explicit_approval"
    return {
        "ready": ready,
        "milestone": "macos-sealed-prerequisite-staging",
        "product_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "checks": checks,
        "next_action": next_action,
        "approval_required": True,
        "installation_performed": False,
        "permission_prompted": False,
        "runtime_started": False,
        "model_loaded": False,
        "network_used": False,
        "content_captured": False,
    }
