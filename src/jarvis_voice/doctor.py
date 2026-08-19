"""Offline, redacted diagnostics for Milestones 0 through 2."""

from __future__ import annotations

import platform
import os
import struct
import sys
from importlib.metadata import PackageNotFoundError, version
from collections.abc import Callable
from typing import Any

from .asr import FasterWhisperCpuBackend
from . import PROTOCOL_VERSION, __version__
from .app_paths import frozen_bundle_root, frozen_library_directory, require_external_mutable_path
from .config import VoiceConfig, configured_voice_auth_key, is_loopback_websocket_url
from .errors import BackendError
from .gpu import GpuAdmissionPolicy, GpuInventory, NvidiaSmiInventory
from .manifest import BASE_MODEL_MANIFEST, validate_model_directory
from .vad import validate_artifact
from .windows.hardening import dll_policy_applied
from .voice_auth import AUTH_PROTOCOL


def _status(status: str, code: str, **details: object) -> dict[str, object]:
    return {"status": status, "code": code, **details}


def _dependency(name: str) -> dict[str, object]:
    try:
        installed = version(name)
    except PackageNotFoundError:
        return _status("blocked", "dependency.missing", package=name)
    return _status("ready", "dependency.available", package=name, version=installed)


def doctor_report(model_directory: str | None) -> dict[str, object]:
    """Backwards-compatible Milestone 0 report."""

    manifest_result = validate_model_directory(model_directory, BASE_MODEL_MANIFEST)
    return {
        "ready": manifest_result.ok,
        "model": {
            "id": BASE_MODEL_MANIFEST.model_id,
            "revision": BASE_MODEL_MANIFEST.revision,
            "manifest": manifest_result.to_dict(),
            "load_probe": _status("deferred", "asr.load_probe_deferred"),
        },
        "network_used": False,
    }


def _cuda_report(
    config: VoiceConfig,
    *,
    is_windows: bool,
    dependencies: dict[str, dict[str, object]],
    inventory: GpuInventory | None,
    active_cuda_probe: bool,
    active_probe_runner: Callable[..., dict[str, object]] | None,
) -> tuple[dict[str, object], bool]:
    if config.asr.execution_profile == "cpu":
        if active_cuda_probe:
            return _status("blocked", "cuda.active_probe_requires_cuda_profile", selected_profile="cpu"), False
        return _status("ready", "cuda.not_required", selected_profile="cpu"), False
    cuda = config.asr.cuda
    report: dict[str, object] = {
        "status": "deferred",
        "code": "cuda.active_probe_required",
        "selected_profile": "cuda",
        "device_index": cuda.device_index,
        "residency": cuda.residency,
        "compute_type": cuda.compute_type,
        "fallback_to_cpu": cuda.fallback_to_cpu,
        "admission_is_point_in_time_heuristic": True,
        "compatibility": _status("deferred", "cuda.runtime_compatibility_active_probe_required"),
        "worker": _status("ready" if is_windows else "deferred", "cuda.worker_spawn_supported" if is_windows else "cuda.native_windows_worker_probe_required"),
    }
    try:
        selected_inventory = inventory or NvidiaSmiInventory()
        decision = GpuAdmissionPolicy(selected_inventory).evaluate(
            device_index=cuda.device_index if cuda.device_index is not None else -1,
            minimum_free_vram_mb=cuda.minimum_free_vram_mb,
            reserve_for_other_workloads_mb=cuda.reserve_for_other_workloads_mb,
        )
        report["admission"] = {
            "status": "ready" if decision.allowed else ("warning" if cuda.fallback_to_cpu else "blocked"),
            "code": decision.code,
            "total_vram_mb": decision.total_vram_mb,
            "free_vram_mb": decision.free_vram_mb,
            "device_name": decision.device_name,
            "driver_version": decision.driver_version,
            "reserve_vram_mb": decision.reserve_vram_mb,
            "available_for_voice_mb": decision.available_for_voice_mb,
            "minimum_free_vram_mb": decision.minimum_free_vram_mb,
        }
        if not decision.allowed:
            report["status"] = "warning" if cuda.fallback_to_cpu else "blocked"
            report["code"] = "cuda.admission_denied_cpu_fallback" if cuda.fallback_to_cpu else decision.code
    except BackendError as exc:
        report["inventory"] = _status("warning" if cuda.fallback_to_cpu else "blocked", exc.code)
        report["status"] = "warning" if cuda.fallback_to_cpu else "blocked"
        report["code"] = "cuda.unavailable_cpu_fallback" if cuda.fallback_to_cpu else exc.code

    gpu_used = False
    if active_cuda_probe:
        if not is_windows:
            report["active_probe"] = _status("blocked", "cuda.native_windows_active_probe_required")
            report["status"] = "blocked"
            report["code"] = "cuda.native_windows_active_probe_required"
        elif dependencies["faster-whisper"]["status"] != "ready" or dependencies["ctranslate2"]["status"] != "ready":
            report["active_probe"] = _status("blocked", "cuda.runtime_missing")
        else:  # pragma: no cover - native Windows CUDA only
            try:
                if active_probe_runner is None:
                    from .cuda_probe import run_active_cuda_probe
                    active_probe_runner = run_active_cuda_probe
                report["active_probe"] = active_probe_runner(config, inventory=inventory)
                gpu_used = report["active_probe"].get("status") == "ready"  # type: ignore[union-attr]
                if gpu_used:
                    report["status"] = "ready"
                    report["code"] = "cuda.active_probe_succeeded"
            except BackendError as exc:
                report["active_probe"] = _status("blocked", exc.code)
                report["status"] = "warning" if cuda.fallback_to_cpu else "blocked"
                report["code"] = "cuda.active_probe_failed_cpu_fallback" if cuda.fallback_to_cpu else exc.code
    else:
        report["active_probe"] = _status("deferred", "cuda.active_probe_not_requested")
    return report, gpu_used


def doctor_report_config(
    config: VoiceConfig,
    *,
    active_cuda_probe: bool = False,
    gpu_inventory: GpuInventory | None = None,
    active_probe_runner: Callable[..., dict[str, object]] | None = None,
) -> dict[str, object]:
    is_windows = sys.platform == "win32"
    windows_11 = bool(is_windows and sys.getwindowsversion().build >= 22_000)
    is_64_bit = struct.calcsize("P") * 8 == 64
    python_ok = sys.version_info >= (3, 11)
    host = _status(
        "ready" if windows_11 and is_64_bit and python_ok else "blocked",
        "host.windows_ready" if windows_11 and is_64_bit and python_ok else "host.native_windows_11_x64_required",
        system=platform.system(),
        machine=platform.machine(),
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    dependencies = {name: _dependency(name) for name in ("faster-whisper", "ctranslate2", "sounddevice", "soxr", "onnxruntime", "websockets", "pywin32")}
    manifest = validate_model_directory(config.whisper_model_directory, BASE_MODEL_MANIFEST)
    model: dict[str, Any] = {
        "id": BASE_MODEL_MANIFEST.model_id,
        "revision": BASE_MODEL_MANIFEST.revision,
        "manifest": manifest.to_dict(),
    }
    if not manifest.ok:
        model["load_probe"] = _status("blocked", "asr.manifest_required_before_load")
    elif dependencies["faster-whisper"]["status"] != "ready":
        model["load_probe"] = _status("blocked", "asr.runtime_missing")
    elif config.asr.execution_profile == "cuda":
        model["load_probe"] = _status("deferred", "asr.cuda_active_probe_required")
    elif not is_windows:
        model["load_probe"] = _status("deferred", "asr.native_windows_probe_required")
    else:  # pragma: no cover - native Windows only
        try:
            FasterWhisperCpuBackend(config.whisper_model_directory or "", compute_type=config.asr.cpu_compute_type).load()
        except Exception:
            model["load_probe"] = _status("blocked", "asr.offline_cpu_load_failed")
        else:
            model["load_probe"] = _status("ready", "asr.offline_cpu_load_succeeded", device="cpu", compute_type=config.asr.cpu_compute_type)

    artifact = config.vad.artifact
    vad_ok, vad_code = validate_artifact(artifact.path, artifact.size, artifact.sha256)
    vad = _status("ready" if vad_ok else "blocked", vad_code)
    transport_loopback = is_loopback_websocket_url(config.orchestrator_url)
    transport_auth_code = "transport.authentication_key_required"
    try:
        configured_voice_auth_key(config)
        transport_authenticated = True
    except BackendError as exc:
        transport_authenticated = False
        if exc.code != "config.voice_auth_key_required":
            transport_auth_code = "transport.authentication_key_invalid"
    transport = _status(
        "ready" if transport_loopback and transport_authenticated else "blocked",
        (
            "transport.loopback_authenticated"
            if transport_loopback and transport_authenticated
            else (
                transport_auth_code
                if transport_loopback
                else "transport.non_loopback_url"
            )
        ),
    )

    is_frozen = bool(getattr(sys, "frozen", False))
    bundle_root = frozen_bundle_root()
    library_directory = frozen_library_directory()
    if is_frozen:
        onedir = bool(bundle_root is not None and library_directory is not None and (library_directory == bundle_root or library_directory.parent == bundle_root))
        mutable_status = "ready"
        mutable_code = "packaging.mutable_paths_external"
        try:
            for configured_path in (config.whisper_model_directory, config.vad.artifact.path):
                if configured_path:
                    require_external_mutable_path(configured_path, bundle_root)
        except BackendError:
            mutable_status = "blocked"; mutable_code = "packaging.mutable_path_inside_bundle"
        packaging = {
            "status": "ready" if onedir and mutable_status == "ready" and dll_policy_applied() else "blocked",
            "code": "packaging.frozen_onedir_ready" if onedir and mutable_status == "ready" and dll_policy_applied() else "packaging.frozen_preflight_failed",
            "frozen": True,
            "bundle_mode": "onedir" if onedir else "unsupported_or_onefile",
            "mutable_data": _status(mutable_status, mutable_code),
            "dll_search": _status("ready" if dll_policy_applied() else "blocked", "hardening.dll_policy_applied" if dll_policy_applied() else "hardening.dll_policy_not_applied"),
            "authenticode": _status("deferred", "packaging.authenticode_release_gate"),
        }
    else:
        packaging = {
            "status": "deferred",
            "code": "packaging.native_windows_frozen_probe_required",
            "frozen": False,
            "bundle_mode": "source",
            "mutable_data": _status("deferred", "packaging.frozen_boundary_not_active"),
            "dll_search": _status("deferred", "hardening.runtime_hook_not_active"),
            "authenticode": _status("deferred", "packaging.authenticode_release_gate"),
        }
    win32_tray_apis = False
    if is_windows:  # pragma: no cover - native Windows only
        try:
            import ctypes
            win32_tray_apis = all((ctypes.windll.shell32.Shell_NotifyIconW, ctypes.windll.kernel32.CreateMutexW, ctypes.windll.user32.RegisterWindowMessageW))
        except Exception:
            win32_tray_apis = False
    local_app_data_configured = bool(os.environ.get("LOCALAPPDATA")) if is_windows else False
    tray_ready = is_windows and win32_tray_apis and local_app_data_configured
    tray = _status(
        "ready" if tray_ready else "blocked",
        "tray.win32_apis_available" if tray_ready else ("tray.local_app_data_missing" if is_windows and not local_app_data_configured else "tray.native_windows_probe_required"),
        shell_notify_icon=win32_tray_apis,
        taskbar_created_recovery=True,
        single_instance="local_named_mutex",
        local_app_data_configured=local_app_data_configured,
    )

    if not is_windows:
        raw_input = _status("blocked", "input.native_windows_probe_required", passive=True)
        audio = _status("blocked", "audio.native_wasapi_probe_required")
        sapi = _status("blocked", "tts.native_sapi_probe_required", configured_locales=sorted(config.sapi_voices))
    else:  # pragma: no cover - native Windows only
        raw_input = _status("ready", "input.raw_input_api_available", passive=True)
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
            device = sd.query_devices(config.audio_input_device, "input")
            audio = _status("ready", "audio.wasapi_input_available", default_sample_rate=int(device["default_samplerate"]))
        except Exception:
            audio = _status("blocked", "audio.wasapi_probe_failed")
        try:
            from .windows.sapi import Win32SapiAdapter
            adapter = Win32SapiAdapter(); adapter.open(); voices = adapter.voices(); adapter.close()
            missing = sorted(locale for locale, name in config.sapi_voices.items() if not name or name not in voices)
            sapi = _status("ready" if not missing else "blocked", "tts.sapi_voices_available" if not missing else "tts.configured_voice_missing", installed_voice_count=len(voices), missing_locales=missing)
        except Exception:
            sapi = _status("blocked", "tts.sapi_probe_failed")

    cuda_report, gpu_used = _cuda_report(
        config,
        is_windows=is_windows,
        dependencies=dependencies,
        inventory=gpu_inventory,
        active_cuda_probe=active_cuda_probe,
        active_probe_runner=active_probe_runner,
    )
    cpu_fallback = _status(
        "ready" if manifest.ok and dependencies["faster-whisper"]["status"] == "ready" else "blocked",
        "asr.cpu_fallback_ready" if manifest.ok and dependencies["faster-whisper"]["status"] == "ready" else "asr.cpu_fallback_unavailable",
        enabled=config.asr.cuda.fallback_to_cpu,
    )
    checks = {"host": host, "dependencies": dependencies, "model": model, "vad": vad, "raw_input": raw_input, "audio": audio, "sapi": sapi, "transport": transport, "asr_profile": _status("ready", "asr.profile_selected", profile=config.asr.execution_profile), "cpu_fallback": cpu_fallback, "cuda": cuda_report, "tray": tray, "packaging": packaging}
    blocked = (
        host["status"] == "blocked"
        or any(item["status"] == "blocked" for item in dependencies.values())
        or not manifest.ok
        or model["load_probe"]["status"] == "blocked"
        or any(item["status"] == "blocked" for item in (vad, raw_input, audio, sapi, transport))
        or tray["status"] == "blocked"
        or packaging["status"] == "blocked"
        or cuda_report["status"] == "blocked"
        or (config.asr.execution_profile == "cuda" and cuda_report["status"] != "ready")
    )
    return {"ready": not blocked, "milestone": "windows-packaging-tray-hardening", "product_version": __version__, "protocol_version": PROTOCOL_VERSION, "voice_auth_protocol": AUTH_PROTOCOL, "checks": checks, "network_used": False, "gpu_used": gpu_used}
