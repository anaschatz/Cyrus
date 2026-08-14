"""Offline, content-free macOS source diagnostics."""

from __future__ import annotations

import os
import platform
import struct
import subprocess
import sys
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .. import PROTOCOL_VERSION, __version__
from ..config import VoiceConfig, is_loopback_websocket_url
from ..manifest import ManifestValidationResult, validate_model_directory
from ..vad import validate_artifact
from .ptt import CoreGraphicsEventTapAdapter, mac_ptt_keycode
from .whisper_cpp import _validate_artifact
from .greektts15_pilot import validate_greektts_model_root
from .greektts15_runtime import load_digest_bound_greektts_candidate


def _status(status: str, code: str, **details: object) -> dict[str, object]:
    return {"status": status, "code": code, **details}


def _dependency(name: str) -> dict[str, object]:
    try:
        installed = version(name)
    except PackageNotFoundError:
        return _status("blocked", "dependency.missing", package=name)
    return _status("ready", "dependency.available", package=name, version=installed)


def _default_audio_probe(device: int | str | None) -> bool:
    import sounddevice as sd  # type: ignore[import-not-found]

    info = sd.query_devices(device, "input")
    return int(info.get("max_input_channels", 0)) >= 1 and float(info.get("default_samplerate", 0)) > 0


def _default_voice_probe() -> set[str]:
    if not os.path.isfile("/usr/bin/say") or not os.access("/usr/bin/say", os.X_OK):
        return set()
    result = subprocess.run(
        ["/usr/bin/say", "-v", "?"],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
        shell=False,
    )
    if result.returncode != 0:
        return set()
    # Only voice identifiers are retained; sample phrases are discarded.
    return {line.split(maxsplit=1)[0] for line in result.stdout.splitlines() if line.strip()}


def _macos_version_tuple(value: str) -> tuple[int, int]:
    try:
        parts = value.split(".")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        return (0, 0)


def macos_doctor_report(
    config: VoiceConfig,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    macos_version: str | None = None,
    python_version: tuple[int, int] | None = None,
    pointer_bits: int | None = None,
    permission_probe: Callable[[], bool] | None = None,
    audio_probe: Callable[[int | str | None], bool] | None = None,
    voice_probe: Callable[[], set[str]] | None = None,
    dependencies: Mapping[str, Mapping[str, object]] | None = None,
    model_validator: Callable[[str | None], ManifestValidationResult] | None = None,
) -> dict[str, object]:
    selected_platform = sys.platform if platform_name is None else platform_name
    selected_machine = platform.machine() if machine is None else machine
    selected_version = platform.mac_ver()[0] if macos_version is None else macos_version
    selected_python = (sys.version_info.major, sys.version_info.minor) if python_version is None else python_version
    selected_bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    host_ready = (
        selected_platform == "darwin"
        and selected_machine == "arm64"
        and _macos_version_tuple(selected_version) >= (14, 0)
        and selected_python == (3, 11)
        and selected_bits == 64
    )
    host = _status(
        "ready" if host_ready else "blocked",
        "host.macos_apple_silicon_ready" if host_ready else "host.macos_14_arm64_python_311_required",
        system="Darwin" if selected_platform == "darwin" else platform.system(),
        machine=selected_machine,
        macos_major_minor=".".join(str(item) for item in _macos_version_tuple(selected_version)),
        python=f"{selected_python[0]}.{selected_python[1]}",
        pointer_bits=selected_bits,
    )
    dependency_names = ["sounddevice", "soxr", "onnxruntime", "websockets"]
    if config.macos.asr_backend == "faster_whisper_cpu_smoke":
        dependency_names.extend(("faster-whisper", "ctranslate2", "numpy"))
    if config.macos.tts_backend == "greektts15_f1":
        dependency_names.extend(("mlx", "mlx-lm", "torch", "snac"))
    selected_dependencies: dict[str, Mapping[str, object]] = (
        dict(dependencies)
        if dependencies is not None
        else {name: _dependency(name) for name in dependency_names}
    )

    whisper = config.macos.whisper_cpp
    artifacts: dict[str, dict[str, object]] = {}
    if config.macos.asr_backend == "whisper_cpp":
        for name, artifact, executable in (
            ("executable", whisper.executable, True),
            ("model", whisper.model, False),
        ):
            try:
                _validate_artifact(artifact, name, executable=executable)
            except Exception as exc:
                code = getattr(exc, "code", f"asr.whisper_cpp_{name}_invalid")
                artifacts[name] = _status("blocked", str(code))
            else:
                artifacts[name] = _status("ready", f"asr.whisper_cpp_{name}_valid")

    whisper_cpp_ready = bool(artifacts) and all(item["status"] == "ready" for item in artifacts.values())
    whisper_cpp_check: dict[str, object] = {
        "status": "ready" if whisper_cpp_ready else "blocked",
        "code": "asr.whisper_cpp_artifacts_valid" if whisper_cpp_ready else "asr.whisper_cpp_artifacts_blocked",
        "artifacts": artifacts,
        "revision": whisper.revision,
        "owned_loopback_sidecar": True,
        "ffmpeg_conversion": False,
        "initial_prompt": False,
    }
    if config.macos.asr_backend == "faster_whisper_cpu_smoke":
        validation = (model_validator or validate_model_directory)(config.whisper_model_directory)
        asr = {
            "status": "ready" if validation.ok else "blocked",
            "code": "asr.faster_whisper_model_valid" if validation.ok else validation.code,
            "backend": "faster_whisper_cpu_smoke",
            "engine": "faster-whisper/ctranslate2",
            "model_id": "Systran/faster-whisper-base",
            "files_checked": validation.files_checked,
            "local_files_only": True,
            "network_fallback": False,
            "model_loaded": False,
            "gpu_used": False,
            "temporary_smoke_profile": True,
        }
        whisper_cpp_check = _status("deferred", "asr.whisper_cpp_not_selected")
    else:
        asr = {
            **whisper_cpp_check,
            "backend": "whisper_cpp",
            "model_loaded": False,
            "gpu_used": False,
            "temporary_smoke_profile": False,
        }

    try:
        mac_ptt_keycode(config.ptt_key)
        permitted = bool((permission_probe or CoreGraphicsEventTapAdapter().preflight)())
        ptt = _status(
            "ready" if permitted else "blocked",
            "input.macos_listen_permission_ready" if permitted else "input.macos_listen_permission_required",
            passive=True,
            suppression=False,
        )
    except Exception as exc:
        ptt = _status("blocked", str(getattr(exc, "code", "input.macos_event_tap_probe_failed")), passive=True, suppression=False)

    if selected_dependencies.get("sounddevice", {}).get("status") != "ready":
        audio = _status("blocked", "audio.sounddevice_missing")
    else:
        try:
            available = bool((audio_probe or _default_audio_probe)(config.audio_input_device))
        except Exception:
            available = False
        audio = _status("ready" if available else "blocked", "audio.coreaudio_input_available" if available else "audio.coreaudio_probe_failed")

    if config.macos.tts_backend == "greektts15_f1":
        greek = config.macos.greektts15
        try:
            candidate = load_digest_bound_greektts_candidate(
                greek.approval_manifest_path or "",
                greek.approval_manifest_sha256 or "",
                greek.repository_root or "",
            )
            model_ready = validate_greektts_model_root(greek.model_root or "").ready
            candidate_ready = candidate.checkpoint_sha256 != "" and model_ready
        except Exception:
            candidate_ready = False
        tts = _status(
            "ready" if candidate_ready else "blocked",
            "tts.greektts15_approved_profile_ready" if candidate_ready else "tts.greektts15_approved_profile_invalid",
            backend="greektts15_f1",
            warm_on_start=greek.warm_on_start,
            local_only=True,
        )
    else:
        configured = config.macos.system_voices
        required_locales = {"el-GR", "en-US"}
        try:
            installed = (voice_probe or _default_voice_probe)()
        except Exception:
            installed = set()
        missing_locales = sorted(locale for locale in required_locales if not configured.get(locale) or configured.get(locale) not in installed)
        tts = _status(
            "ready" if not missing_locales else "blocked",
            "tts.macos_voices_available" if not missing_locales else "tts.macos_configured_voice_missing",
            configured_locales=sorted(configured),
            missing_locales=missing_locales,
            stdin_only=True,
        )

    vad_artifact = config.vad.artifact
    vad_ok, vad_code = validate_artifact(vad_artifact.path, vad_artifact.size, vad_artifact.sha256)
    vad = _status("ready" if vad_ok else "blocked", vad_code)
    transport_ok = is_loopback_websocket_url(config.orchestrator_url)
    transport = _status("ready" if transport_ok else "blocked", "transport.loopback_only" if transport_ok else "transport.non_loopback_url")
    checks: dict[str, Any] = {
        "host": host,
        "dependencies": selected_dependencies,
        "asr": asr,
        "whisper_cpp": whisper_cpp_check,
        "ptt": ptt,
        "audio": audio,
        "tts": tts,
        "vad": vad,
        "transport": transport,
        "packaging": _status("deferred", "packaging.macos_app_not_in_milestone"),
    }
    ready = (
        host["status"] == "ready"
        and all(item.get("status") == "ready" for item in selected_dependencies.values())
        and asr["status"] == "ready"
        and all(item["status"] == "ready" for item in (ptt, audio, tts, vad, transport))
    )
    return {
        "ready": ready,
        "milestone": "macos-cpu-smoke" if config.macos.asr_backend == "faster_whisper_cpu_smoke" else "macos-apple-silicon-parity",
        "product_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "checks": checks,
        "network_used": False,
        "content_captured": False,
    }
