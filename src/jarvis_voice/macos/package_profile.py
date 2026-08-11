"""Strict signed-bundle profile and embedded whisper.cpp helper binding."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .. import PROTOCOL_VERSION, __version__
from ..config import ArtifactConfig, VoiceConfig
from ..errors import BackendError


PROFILE_SCHEMA = "jarvis-macos-runtime-profile/1"
PROFILE_NAME = "jarvis-macos-runtime-profile.json"
HELPER_RELATIVE_PATH = "helpers/whisper-server"
PROFILE_FIELDS = {
    "schema_version",
    "product_version",
    "voice_protocol",
    "minimum_macos",
    "architecture",
    "python_version",
    "pyinstaller_version",
    "helper",
    "models_embedded",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_macos_package_profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PROFILE_FIELDS:
        raise BackendError("packaging.macos_profile_schema", "macOS package profile is invalid", subsystem="packaging")
    helper = value.get("helper")
    if not isinstance(helper, Mapping) or set(helper) != {"relative_path", "size", "sha256"}:
        raise BackendError("packaging.macos_profile_schema", "macOS helper profile is invalid", subsystem="packaging")
    if (
        value["schema_version"] != PROFILE_SCHEMA
        or value["product_version"] != __version__
        or value["voice_protocol"] != PROTOCOL_VERSION
        or value["minimum_macos"] != "14.0"
        or value["architecture"] != "arm64"
        or value["python_version"] != "3.11.14"
        or value["pyinstaller_version"] != "6.21.0"
        or value["models_embedded"] is not False
        or helper["relative_path"] != HELPER_RELATIVE_PATH
        or not isinstance(helper["size"], int)
        or isinstance(helper["size"], bool)
        or helper["size"] < 1
        or not isinstance(helper["sha256"], str)
        or len(helper["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in helper["sha256"])
    ):
        raise BackendError("packaging.macos_profile_contract", "macOS package profile contract is invalid", subsystem="packaging")
    return {**dict(value), "helper": dict(helper)}


def build_macos_package_profile(helper: Path, *, pyinstaller_version: str) -> dict[str, Any]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise BackendError("packaging.macos_native_arm64_required", "macOS package build requires native arm64", subsystem="packaging")
    if sys.version_info[:3] != (3, 11, 14) or pyinstaller_version != "6.21.0":
        raise BackendError("packaging.macos_build_runtime_mismatch", "macOS package build runtime is not pinned", subsystem="packaging")
    if helper.is_symlink():
        raise BackendError("packaging.macos_helper_invalid", "whisper helper must be a regular file", subsystem="packaging")
    selected = helper.resolve(strict=True)
    if not selected.is_file():
        raise BackendError("packaging.macos_helper_invalid", "whisper helper must be a regular file", subsystem="packaging")
    return validate_macos_package_profile(
        {
            "schema_version": PROFILE_SCHEMA,
            "product_version": __version__,
            "voice_protocol": PROTOCOL_VERSION,
            "minimum_macos": "14.0",
            "architecture": "arm64",
            "python_version": "3.11.14",
            "pyinstaller_version": pyinstaller_version,
            "helper": {
                "relative_path": HELPER_RELATIVE_PATH,
                "size": selected.stat().st_size,
                "sha256": _sha256_file(selected),
            },
            "models_embedded": False,
        }
    )


def apply_packaged_macos_profile(
    config: VoiceConfig,
    *,
    resources_root: Path | None = None,
) -> VoiceConfig:
    """Bind a frozen app to its signed helper while models remain external."""

    if resources_root is None:
        if not getattr(sys, "frozen", False):
            return config
        extracted = getattr(sys, "_MEIPASS", None)
        if not extracted:
            raise BackendError("packaging.macos_bundle_root_missing", "macOS bundle resources are unavailable", subsystem="packaging")
        resources_root = Path(extracted)
    root = resources_root.resolve(strict=True)
    try:
        value = json.loads((root / PROFILE_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("packaging.macos_profile_unreadable", "macOS package profile is unavailable", subsystem="packaging") from exc
    profile = validate_macos_package_profile(value)
    helper_candidate = root / profile["helper"]["relative_path"]
    if helper_candidate.is_symlink():
        raise BackendError("packaging.macos_helper_invalid", "Packaged whisper helper is invalid", subsystem="packaging")
    helper = helper_candidate.resolve(strict=True)
    if root not in helper.parents or not helper.is_file():
        raise BackendError("packaging.macos_helper_invalid", "Packaged whisper helper is invalid", subsystem="packaging")
    if helper.stat().st_size != profile["helper"]["size"] or _sha256_file(helper) != profile["helper"]["sha256"]:
        raise BackendError("packaging.macos_helper_integrity", "Packaged whisper helper failed integrity validation", subsystem="packaging")
    whisper = replace(
        config.macos.whisper_cpp,
        executable=ArtifactConfig(str(helper), helper.stat().st_size, profile["helper"]["sha256"]),
    )
    return replace(config, macos=replace(config.macos, whisper_cpp=whisper))
