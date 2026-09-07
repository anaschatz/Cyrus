"""Offline materialization helpers for the approved macOS CPU environment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from array import array
from contextlib import closing, suppress
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from typing import Mapping

from ..app_paths import require_external_mutable_path
from ..asr import FasterWhisperCpuBackend
from ..backends import AudioChunk
from ..config import load_voice_config, voice_config_from_mapping
from ..errors import BackendError
from ..manifest import model_manifest_for_profile, validate_model_directory
from ..qualification import write_report
from ..vad import SileroOnnxVad, validate_artifact
from ..voice_auth import VoiceAuthError, load_voice_auth_key
from .doctor import _default_voice_probe


DIRECT_REQUIREMENTS: Mapping[str, str] = {
    "faster-whisper": "1.2.1",
    "sounddevice": "0.5.3",
    "soxr": "1.1.0",
    "onnxruntime": "1.27.0",
    "kokoro-onnx": "0.6.1",
    "websockets": "15.0.1",
}
SILERO_VERSION = "6.2.1"
SILERO_WHEEL_NAME = "silero_vad-6.2.1-py3-none-any.whl"
SILERO_WHEEL_SHA256 = "09de93c4d874bb19c53e62a47dd38be5f163cedad2b5599583231f2a84ef79cb"
SILERO_ONNX_MEMBER = "silero_vad/data/silero_vad.onnx"
SILERO_MANIFEST_SCHEMA = 1
MINIMUM_FREE_BYTES = 2 * 1024 * 1024 * 1024


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_new_json(path: str, value: object) -> None:
    supplied = Path(path).expanduser()
    output = require_external_mutable_path(supplied.parent) / supplied.name
    try:
        with tempfile.TemporaryDirectory(prefix=".materialization-", dir=output.parent) as directory:
            artifact = write_report(Path(directory) / "artifact.json", value)
            os.link(artifact, output)
    except OSError as exc:
        raise BackendError("materialization.output_write_failed", "Materialization output could not be created", subsystem="materialization") from exc


def disk_space_gate(path: str, *, minimum_free_bytes: int = MINIMUM_FREE_BYTES) -> dict[str, object]:
    target = Path(path).expanduser()
    if not target.is_absolute() or not target.exists():
        raise BackendError("materialization.disk_probe_path_invalid", "Disk probe path is invalid", subsystem="materialization")
    try:
        available = shutil.disk_usage(target).free
    except OSError as exc:
        raise BackendError("materialization.disk_probe_failed", "Disk space probe failed", subsystem="materialization") from exc
    ready = available >= minimum_free_bytes
    return {
        "ready": ready,
        "code": "materialization.disk_space_ready" if ready else "materialization.disk_space_insufficient",
        "available_mib": available // (1024 * 1024),
        "required_mib": minimum_free_bytes // (1024 * 1024),
        "installation_performed": False,
        "network_used": False,
    }


def blocked_materialization_summary(
    *,
    python_version: str,
    wheels_locked: int,
    silero_bytes: int,
    available_mib: int,
    required_mib: int = MINIMUM_FREE_BYTES // (1024 * 1024),
) -> dict[str, object]:
    if (
        not re.fullmatch(r"3\.11\.\d+", python_version)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (wheels_locked, silero_bytes, available_mib, required_mib))
    ):
        raise BackendError("materialization.summary_invalid", "Materialization summary values are invalid", subsystem="materialization")
    return {
        "ready": False,
        "code": "materialization.disk_space_insufficient",
        "python": {"installed": True, "version": python_version, "managed_user_local": True},
        "wheelhouse": {"locked": True, "wheels": wheels_locked},
        "silero": {"extracted": True, "bytes": silero_bytes},
        "private_config_written": True,
        "available_mib": available_mib,
        "required_mib": required_mib,
        "runtime_dependencies_installed": False,
        "offline_install_attempted": False,
        "network_acquisition_performed": True,
        "runtime_started": False,
        "content_captured": False,
    }


def completed_materialization_summary(
    *,
    python_version: str,
    wheels_locked: int,
    silero_bytes: int,
) -> dict[str, object]:
    if (
        not re.fullmatch(r"3\.11\.\d+", python_version)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (wheels_locked, silero_bytes))
    ):
        raise BackendError("materialization.summary_invalid", "Materialization summary values are invalid", subsystem="materialization")
    return {
        "ready": True,
        "code": "materialization.macos_cpu_environment_ready",
        "python": {"installed": True, "version": python_version, "managed_user_local": True},
        "wheelhouse": {"locked": True, "wheels": wheels_locked},
        "silero": {"extracted": True, "bytes": silero_bytes},
        "private_config_written": True,
        "runtime_dependencies_installed": True,
        "offline_install": True,
        "pip_check": True,
        "doctor_ready": True,
        "backend_probe_ready": True,
        "network_acquisition_performed": True,
        "microphone_opened": False,
        "runtime_started": False,
        "content_captured": False,
    }


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _wheel_identity(path: Path) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = [name for name in archive.namelist() if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("metadata count")
            info = archive.getinfo(metadata[0])
            if info.file_size > 1_048_576 or info.flag_bits & 1:
                raise ValueError("metadata bounds")
            message = BytesParser(policy=email_policy).parsebytes(archive.read(info))
    except (OSError, zipfile.BadZipFile, KeyError, ValueError):
        raise BackendError("materialization.wheel_invalid", "Wheel metadata is invalid", subsystem="materialization") from None
    name, version = message.get("Name"), message.get("Version")
    if not isinstance(name, str) or not isinstance(version, str) or not name or not version:
        raise BackendError("materialization.wheel_invalid", "Wheel identity is invalid", subsystem="materialization")
    return _canonical_name(name), version


def _platform_compatible_filename(filename: str) -> bool:
    lowered = filename.lower()
    if any(marker in lowered for marker in ("win32", "win_amd64", "manylinux", "musllinux")):
        return False
    if "macosx" in lowered and "arm64" not in lowered and "universal2" not in lowered:
        return False
    return "x86_64" not in lowered or "universal2" in lowered


def lock_wheelhouse(wheelhouse: str, output: str) -> dict[str, object]:
    root = Path(wheelhouse).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise BackendError("materialization.wheelhouse_invalid", "Wheelhouse must be an absolute regular directory", subsystem="materialization")
    try:
        wheels = sorted((path for path in root.iterdir() if path.is_file() and path.suffix == ".whl"), key=lambda item: item.name)
    except OSError as exc:
        raise BackendError("materialization.wheelhouse_unreadable", "Wheelhouse is unreadable", subsystem="materialization") from exc
    if not 1 <= len(wheels) <= 512 or any(path.is_symlink() for path in wheels):
        raise BackendError("materialization.wheelhouse_invalid", "Wheelhouse file set is invalid", subsystem="materialization")
    identities: dict[str, str] = {}
    items: list[dict[str, object]] = []
    for wheel in wheels:
        if not _platform_compatible_filename(wheel.name):
            raise BackendError("materialization.wheel_platform_mismatch", "Wheelhouse contains an incompatible platform wheel", subsystem="materialization")
        name, version = _wheel_identity(wheel)
        if name in identities:
            raise BackendError("materialization.wheel_duplicate_distribution", "Wheelhouse contains a duplicate distribution", subsystem="materialization")
        identities[name] = version
        try:
            size = wheel.stat().st_size
            digest = _sha256(wheel)
        except OSError as exc:
            raise BackendError("materialization.wheelhouse_unreadable", "Wheelhouse artifact is unreadable", subsystem="materialization") from exc
        items.append({"filename": wheel.name, "size": size, "sha256": digest})
    for name, expected in DIRECT_REQUIREMENTS.items():
        if identities.get(name) != expected:
            raise BackendError(
                "materialization.direct_requirement_mismatch",
                "Wheelhouse does not contain the exact approved direct requirement",
                subsystem="materialization",
                safe_context={"package": name},
            )
    manifest = {
        "schema_version": 1,
        "target": {"python": "3.11", "platform": "macos_arm64"},
        "top_level": dict(DIRECT_REQUIREMENTS),
        "wheels": items,
    }
    _write_new_json(output, manifest)
    return {"ready": True, "code": "materialization.wheelhouse_locked", "wheels": len(items), "network_used": False}


def extract_silero_onnx(wheel_path: str, output_directory: str, manifest_output: str) -> dict[str, object]:
    wheel = Path(wheel_path).expanduser()
    destination = Path(output_directory).expanduser()
    if not wheel.is_absolute() or not destination.is_absolute() or wheel.name != SILERO_WHEEL_NAME:
        raise BackendError("materialization.silero_input_invalid", "Silero input paths are invalid", subsystem="materialization")
    if not wheel.is_file() or wheel.is_symlink() or not destination.is_dir() or destination.is_symlink():
        raise BackendError("materialization.silero_input_missing", "Silero wheel or output directory is missing", subsystem="materialization")
    try:
        if _sha256(wheel) != SILERO_WHEEL_SHA256:
            raise BackendError("materialization.silero_wheel_hash_mismatch", "Silero wheel hash does not match PyPI", subsystem="materialization")
        with zipfile.ZipFile(wheel) as archive:
            info = archive.getinfo(SILERO_ONNX_MEMBER)
            unix_mode = info.external_attr >> 16
            if info.flag_bits & 1 or not 100_000 <= info.file_size <= 16 * 1024 * 1024 or unix_mode & 0o170000 == 0o120000:
                raise BackendError("materialization.silero_member_invalid", "Silero ONNX member is invalid", subsystem="materialization")
            final = destination / f"silero_vad-{SILERO_VERSION}.onnx"
            if os.path.lexists(final):
                raise BackendError("materialization.silero_output_exists", "Silero output already exists", subsystem="materialization")
            temporary: str | None = None
            published = False
            try:
                with archive.open(info) as source, tempfile.NamedTemporaryFile(prefix=".silero-", dir=destination, delete=False) as target:
                    temporary = target.name
                    os.chmod(temporary, 0o600)
                    digest = hashlib.sha256()
                    size = 0
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > info.file_size:
                            raise BackendError("materialization.silero_member_invalid", "Silero ONNX size exceeded metadata", subsystem="materialization")
                        digest.update(chunk)
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                if size != info.file_size:
                    raise BackendError("materialization.silero_member_invalid", "Silero ONNX extraction was incomplete", subsystem="materialization")
                os.link(temporary, final)
                published = True
                manifest = {
                    "schema_version": SILERO_MANIFEST_SCHEMA,
                    "artifact": {"path": str(final), "size": size, "sha256": digest.hexdigest()},
                    "source": {"package": "silero-vad", "version": SILERO_VERSION, "wheel_sha256": SILERO_WHEEL_SHA256},
                }
                _write_new_json(manifest_output, manifest)
            except BaseException:
                if published:
                    with suppress(OSError):
                        if os.path.samestat(os.stat(temporary), final.lstat()):
                            final.unlink()
                raise
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise BackendError("materialization.silero_extract_failed", "Silero ONNX extraction failed", subsystem="materialization") from exc
    return {"ready": True, "code": "materialization.silero_extracted", "bytes": size, "network_used": False}


def materialize_private_config(
    base_config: str,
    model_directory: str,
    vad_manifest: str,
    output: str,
    *,
    greek_voice: str,
    english_voice: str,
    voice_auth_key: str,
) -> dict[str, object]:
    try:
        raw = json.loads(Path(base_config).read_text(encoding="utf-8"))
        vad_raw = json.loads(Path(vad_manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise BackendError("materialization.config_input_unreadable", "Configuration input is unreadable", subsystem="materialization") from exc
    if (
        not isinstance(raw, dict)
        or not isinstance(vad_raw, Mapping)
        or set(vad_raw) != {"schema_version", "artifact", "source"}
        or vad_raw.get("schema_version") != SILERO_MANIFEST_SCHEMA
        or not isinstance(vad_raw.get("artifact"), Mapping)
        or set(vad_raw["artifact"]) != {"path", "size", "sha256"}
    ):
        raise BackendError("materialization.vad_manifest_invalid", "VAD manifest is invalid", subsystem="materialization")
    base_voice_config = voice_config_from_mapping(raw)
    validation = validate_model_directory(
        model_directory,
        model_manifest_for_profile(base_voice_config.asr.model_profile),
    )
    if not validation.ok:
        raise BackendError(validation.code, "Whisper model validation failed", subsystem="materialization")
    artifact = vad_raw["artifact"]
    ok, code = validate_artifact(artifact.get("path"), artifact.get("size"), artifact.get("sha256"))
    if not ok:
        raise BackendError(code, "VAD artifact validation failed", subsystem="materialization")
    installed = _default_voice_probe()
    if greek_voice not in installed or english_voice not in installed:
        raise BackendError("materialization.voice_missing", "Configured system voice is not installed", subsystem="materialization")
    try:
        load_voice_auth_key(voice_auth_key)
    except VoiceAuthError as exc:
        raise BackendError(
            "materialization.voice_auth_key_invalid",
            "Voice authentication key is unavailable or invalid",
            subsystem="materialization",
        ) from exc
    raw["whisper_model_directory"] = str(Path(model_directory).expanduser())
    raw["voice_auth_key_file"] = str(Path(voice_auth_key).expanduser())
    raw.setdefault("asr", {})["execution_profile"] = "cpu"
    raw.setdefault("vad", {})["artifact"] = dict(artifact)
    raw.setdefault("macos", {})["asr_backend"] = "faster_whisper_cpu_smoke"
    raw["macos"]["system_voices"] = {"el-GR": greek_voice, "en-US": english_voice}
    voice_config_from_mapping(raw)
    _write_new_json(output, raw)
    return {"ready": True, "code": "materialization.private_config_written", "network_used": False}


def probe_local_backends(config_path: str, model_directory: str) -> dict[str, object]:
    config = load_voice_config(config_path)
    with closing(FasterWhisperCpuBackend(
        model_directory,
        compute_type=config.asr.cpu_compute_type,
        manifest=model_manifest_for_profile(config.asr.model_profile),
    )) as asr:
        artifact = config.vad.artifact
        if not artifact.path or artifact.size is None or not artifact.sha256:
            raise BackendError("vad.artifact_manifest_required", "VAD manifest is required", subsystem="materialization")
        vad = SileroOnnxVad(
            artifact_path=artifact.path,
            size=artifact.size,
            sha256=artifact.sha256,
            threshold=config.vad.threshold,
            minimum_speech_ms=config.vad.minimum_speech_ms,
            padding_ms=config.vad.padding_ms,
        )
        asr.load()
        result = vad.trim(AudioChunk(array("f", [0.0] * 512), 16_000))
        try:
            import sounddevice as sd  # type: ignore[import-not-found]

            device = sd.query_devices(config.audio_input_device, "input")
            audio_ready = int(device.get("max_input_channels", 0)) >= 1
        except Exception as exc:
            raise BackendError("materialization.audio_probe_failed", "CoreAudio input query failed", subsystem="materialization") from exc
        return {
            "ready": audio_ready and not result.speech_detected,
            "code": "materialization.local_backends_loaded" if audio_ready and not result.speech_detected else "materialization.local_backend_probe_failed",
            "asr_model_loaded": True,
            "vad_session_loaded": True,
            "synthetic_silence_only": True,
            "microphone_opened": False,
            "runtime_started": False,
            "network_used": False,
            "content_captured": False,
        }
