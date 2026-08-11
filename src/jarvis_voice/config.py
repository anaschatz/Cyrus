"""Dependency-free JSON configuration for the source-run runtime."""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .errors import BackendError


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    path: str | None = None
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class VadConfig:
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    threshold: float = 0.5
    minimum_speech_ms: int = 80
    padding_ms: int = 80


@dataclass(frozen=True, slots=True)
class CudaAsrConfig:
    device_index: int | None = None
    compute_type: str = "int8_float16"
    residency: str = "turn_scoped"
    minimum_free_vram_mb: int = 0
    reserve_for_other_workloads_mb: int = 0
    fallback_to_cpu: bool = True
    worker_start_timeout_seconds: float = 30.0
    transcription_timeout_seconds: float = 120.0
    warm_restart_limit: int = 1
    vram_poll_interval_ms: int = 100


@dataclass(frozen=True, slots=True)
class AsrConfig:
    execution_profile: str = "cpu"
    cpu_compute_type: str = "int8"
    cuda: CudaAsrConfig = field(default_factory=CudaAsrConfig)


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    shutdown_timeout_seconds: float = 5.0
    log_maximum_bytes: int = 1_048_576
    log_backup_count: int = 3


@dataclass(frozen=True, slots=True)
class SchedulerClientConfig:
    enabled: bool = False
    compatibility_manifest_path: str | None = None
    compatibility_manifest_sha256: str | None = None
    native_qualification_path: str | None = None
    native_qualification_sha256: str | None = None
    full_body_when_locked: bool = False


@dataclass(frozen=True, slots=True)
class WhisperCppConfig:
    revision: str = "v1.9.2"
    executable: ArtifactConfig = field(default_factory=ArtifactConfig)
    model: ArtifactConfig = field(default_factory=ArtifactConfig)
    language: str = "auto"
    threads: int = 4
    startup_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 120.0
    maximum_response_bytes: int = 1_048_576


@dataclass(frozen=True, slots=True)
class MacosConfig:
    asr_backend: str = "whisper_cpp"
    whisper_cpp: WhisperCppConfig = field(default_factory=WhisperCppConfig)
    system_voices: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    ptt_key: str = "F9"
    ptt_queue_capacity: int = 64
    audio_queue_capacity: int = 256
    maximum_hold_seconds: float = 30.0
    maximum_utterance_seconds: float = 30.0
    audio_input_device: int | str | None = None
    whisper_model_directory: str | None = None
    asr: AsrConfig = field(default_factory=AsrConfig)
    application: ApplicationConfig = field(default_factory=ApplicationConfig)
    scheduler: SchedulerClientConfig = field(default_factory=SchedulerClientConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    orchestrator_url: str = "ws://127.0.0.1:8765"
    sapi_voices: Mapping[str, str] = field(default_factory=dict)
    macos: MacosConfig = field(default_factory=MacosConfig)
    logging_level: str = "INFO"


def is_loopback_websocket_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        if parsed.scheme not in {"ws", "wss"} or parsed.hostname is None:
            return False
        if parsed.hostname.lower() == "localhost":
            return True
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except (TypeError, ValueError):
        return False


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise BackendError("config.invalid_field", "Configuration value must be a positive integer", subsystem="config", safe_context={"field": field_name})
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BackendError("config.invalid_field", "Configuration value must be a non-negative integer", subsystem="config", safe_context={"field": field_name})
    return value


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise BackendError("config.invalid_field", "Configuration value must be boolean", subsystem="config", safe_context={"field": field_name})
    return value


def voice_config_from_mapping(data: Mapping[str, Any]) -> VoiceConfig:
    if not isinstance(data, Mapping):
        raise BackendError("config.invalid", "Configuration must be an object", subsystem="config")
    vad_data = data.get("vad", {})
    asr_data = data.get("asr", {})
    application_data = data.get("application", {})
    scheduler_data = data.get("scheduler", {})
    cuda_data = asr_data.get("cuda", {}) if isinstance(asr_data, Mapping) else None
    artifact_data = vad_data.get("artifact", {}) if isinstance(vad_data, Mapping) else None
    macos_data = data.get("macos", {})
    whisper_cpp_data = macos_data.get("whisper_cpp", {}) if isinstance(macos_data, Mapping) else None
    executable_data = whisper_cpp_data.get("executable", {}) if isinstance(whisper_cpp_data, Mapping) else None
    mac_model_data = whisper_cpp_data.get("model", {}) if isinstance(whisper_cpp_data, Mapping) else None
    if not isinstance(vad_data, Mapping) or not isinstance(artifact_data, Mapping):
        raise BackendError("config.invalid_field", "VAD configuration must be an object", subsystem="config", safe_context={"field": "vad"})
    if not isinstance(asr_data, Mapping) or not isinstance(cuda_data, Mapping):
        raise BackendError("config.invalid_field", "ASR configuration must be an object", subsystem="config", safe_context={"field": "asr"})
    if not isinstance(application_data, Mapping):
        raise BackendError("config.invalid_field", "Application configuration must be an object", subsystem="config", safe_context={"field": "application"})
    if not isinstance(scheduler_data, Mapping) or set(scheduler_data) - {
        "enabled", "compatibility_manifest_path", "compatibility_manifest_sha256",
        "native_qualification_path", "native_qualification_sha256", "full_body_when_locked",
    }:
        raise BackendError("config.invalid_field", "Scheduler configuration is invalid", subsystem="config", safe_context={"field": "scheduler"})
    if not isinstance(macos_data, Mapping) or set(macos_data) - {"asr_backend", "whisper_cpp", "system_voices"}:
        raise BackendError("config.invalid_field", "macOS configuration has unknown fields", subsystem="config", safe_context={"field": "macos"})
    if not isinstance(whisper_cpp_data, Mapping) or set(whisper_cpp_data) - {
        "revision", "executable", "model", "language", "threads", "startup_timeout_seconds", "request_timeout_seconds", "maximum_response_bytes"
    }:
        raise BackendError("config.invalid_field", "whisper.cpp configuration has unknown fields", subsystem="config", safe_context={"field": "macos.whisper_cpp"})
    if not isinstance(executable_data, Mapping) or set(executable_data) - {"path", "size", "sha256"}:
        raise BackendError("config.invalid_field", "whisper.cpp executable manifest is invalid", subsystem="config", safe_context={"field": "macos.whisper_cpp.executable"})
    if not isinstance(mac_model_data, Mapping) or set(mac_model_data) - {"path", "size", "sha256"}:
        raise BackendError("config.invalid_field", "whisper.cpp model manifest is invalid", subsystem="config", safe_context={"field": "macos.whisper_cpp.model"})
    url = str(data.get("orchestrator_url", "ws://127.0.0.1:8765"))
    if not is_loopback_websocket_url(url):
        raise BackendError("config.non_loopback_orchestrator", "Only loopback WebSocket URLs are permitted", subsystem="config", safe_context={"field": "orchestrator_url"})
    voices = data.get("sapi_voices", {})
    if not isinstance(voices, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in voices.items()):
        raise BackendError("config.invalid_field", "SAPI voice mappings must be strings", subsystem="config", safe_context={"field": "sapi_voices"})
    macos_voices = macos_data.get("system_voices", {})
    if not isinstance(macos_voices, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in macos_voices.items()):
        raise BackendError("config.invalid_field", "macOS voice mappings must be strings", subsystem="config", safe_context={"field": "macos.system_voices"})
    config = VoiceConfig(
        ptt_key=str(data.get("ptt_key", "F9")).upper(),
        ptt_queue_capacity=_positive_int(data.get("ptt_queue_capacity", 64), "ptt_queue_capacity"),
        audio_queue_capacity=_positive_int(data.get("audio_queue_capacity", 256), "audio_queue_capacity"),
        maximum_hold_seconds=float(data.get("maximum_hold_seconds", 30.0)),
        maximum_utterance_seconds=float(data.get("maximum_utterance_seconds", 30.0)),
        audio_input_device=data.get("audio_input_device"),
        whisper_model_directory=data.get("whisper_model_directory"),
        asr=AsrConfig(
            execution_profile=str(asr_data.get("execution_profile", "cpu")).lower(),
            cpu_compute_type=str(asr_data.get("cpu_compute_type", "int8")),
            cuda=CudaAsrConfig(
                device_index=cuda_data.get("device_index"),
                compute_type=str(cuda_data.get("compute_type", "int8_float16")),
                residency=str(cuda_data.get("residency", "turn_scoped")).lower(),
                minimum_free_vram_mb=_non_negative_int(cuda_data.get("minimum_free_vram_mb", 0), "asr.cuda.minimum_free_vram_mb"),
                reserve_for_other_workloads_mb=_non_negative_int(cuda_data.get("reserve_for_other_workloads_mb", 0), "asr.cuda.reserve_for_other_workloads_mb"),
                fallback_to_cpu=_boolean(cuda_data.get("fallback_to_cpu", True), "asr.cuda.fallback_to_cpu"),
                worker_start_timeout_seconds=float(cuda_data.get("worker_start_timeout_seconds", 30.0)),
                transcription_timeout_seconds=float(cuda_data.get("transcription_timeout_seconds", 120.0)),
                warm_restart_limit=_non_negative_int(cuda_data.get("warm_restart_limit", 1), "asr.cuda.warm_restart_limit"),
                vram_poll_interval_ms=_positive_int(cuda_data.get("vram_poll_interval_ms", 100), "asr.cuda.vram_poll_interval_ms"),
            ),
        ),
        application=ApplicationConfig(
            shutdown_timeout_seconds=float(application_data.get("shutdown_timeout_seconds", 5.0)),
            log_maximum_bytes=_positive_int(application_data.get("log_maximum_bytes", 1_048_576), "application.log_maximum_bytes"),
            log_backup_count=_non_negative_int(application_data.get("log_backup_count", 3), "application.log_backup_count"),
        ),
        scheduler=SchedulerClientConfig(
            enabled=_boolean(scheduler_data.get("enabled", False), "scheduler.enabled"),
            compatibility_manifest_path=scheduler_data.get("compatibility_manifest_path"),
            compatibility_manifest_sha256=scheduler_data.get("compatibility_manifest_sha256"),
            native_qualification_path=scheduler_data.get("native_qualification_path"),
            native_qualification_sha256=scheduler_data.get("native_qualification_sha256"),
            full_body_when_locked=_boolean(
                scheduler_data.get("full_body_when_locked", False),
                "scheduler.full_body_when_locked",
            ),
        ),
        vad=VadConfig(
            artifact=ArtifactConfig(artifact_data.get("path"), artifact_data.get("size"), artifact_data.get("sha256")),
            threshold=float(vad_data.get("threshold", 0.5)),
            minimum_speech_ms=int(vad_data.get("minimum_speech_ms", 80)),
            padding_ms=int(vad_data.get("padding_ms", 80)),
        ),
        orchestrator_url=url,
        sapi_voices=dict(voices),
        macos=MacosConfig(
            asr_backend=str(macos_data.get("asr_backend", "whisper_cpp")).lower(),
            whisper_cpp=WhisperCppConfig(
                revision=str(whisper_cpp_data.get("revision", "v1.9.2")),
                executable=ArtifactConfig(executable_data.get("path"), executable_data.get("size"), executable_data.get("sha256")),
                model=ArtifactConfig(mac_model_data.get("path"), mac_model_data.get("size"), mac_model_data.get("sha256")),
                language=str(whisper_cpp_data.get("language", "auto")).lower(),
                threads=_positive_int(whisper_cpp_data.get("threads", 4), "macos.whisper_cpp.threads"),
                startup_timeout_seconds=float(whisper_cpp_data.get("startup_timeout_seconds", 30.0)),
                request_timeout_seconds=float(whisper_cpp_data.get("request_timeout_seconds", 120.0)),
                maximum_response_bytes=_positive_int(whisper_cpp_data.get("maximum_response_bytes", 1_048_576), "macos.whisper_cpp.maximum_response_bytes"),
            ),
            system_voices=dict(macos_voices),
        ),
        logging_level=str(data.get("logging_level", "INFO")).upper(),
    )
    function_key = (
        config.ptt_key.startswith("F")
        and config.ptt_key[1:].isdigit()
        and 1 <= int(config.ptt_key[1:]) <= 24
    )
    if not function_key and config.ptt_key != "CONTROL_OPTION":
        raise BackendError(
            "config.unsupported_ptt_key",
            "PTT key must be F1 through F24 or the macOS CONTROL_OPTION chord",
            subsystem="config",
            safe_context={"field": "ptt_key"},
        )
    if config.ptt_key in {"F10", "F11"}:
        raise BackendError(
            "config.reserved_approval_key",
            "F10 and F11 are reserved for negotiated cloud approval",
            subsystem="config",
            safe_context={"field": "ptt_key"},
        )
    if config.maximum_hold_seconds <= 0 or config.maximum_utterance_seconds <= 0:
        raise BackendError("config.invalid_field", "Durations must be positive", subsystem="config", safe_context={"field": "duration"})
    if config.asr.execution_profile not in {"cpu", "cuda"}:
        raise BackendError("config.invalid_field", "ASR execution profile must be cpu or cuda", subsystem="config", safe_context={"field": "asr.execution_profile"})
    if config.asr.cuda.residency not in {"turn_scoped", "warm"}:
        raise BackendError("config.invalid_field", "CUDA residency must be turn_scoped or warm", subsystem="config", safe_context={"field": "asr.cuda.residency"})
    if config.asr.execution_profile == "cuda" and (
        not isinstance(config.asr.cuda.device_index, int)
        or isinstance(config.asr.cuda.device_index, bool)
        or config.asr.cuda.device_index < 0
    ):
        raise BackendError("config.cuda_device_required", "CUDA profile requires an explicit non-negative device index", subsystem="config", safe_context={"field": "asr.cuda.device_index"})
    if config.asr.cuda.worker_start_timeout_seconds <= 0 or config.asr.cuda.transcription_timeout_seconds <= 0:
        raise BackendError("config.invalid_field", "CUDA timeouts must be positive", subsystem="config", safe_context={"field": "asr.cuda.timeout"})
    if config.application.shutdown_timeout_seconds <= 0:
        raise BackendError("config.invalid_field", "Application shutdown timeout must be positive", subsystem="config", safe_context={"field": "application.shutdown_timeout_seconds"})
    scheduler_values = (
        config.scheduler.compatibility_manifest_path,
        config.scheduler.compatibility_manifest_sha256,
        config.scheduler.native_qualification_path,
        config.scheduler.native_qualification_sha256,
    )
    if config.scheduler.enabled:
        if any(value is None for value in scheduler_values):
            raise BackendError("config.scheduler_qualification_required", "Scheduler qualification files and digests are required", subsystem="config")
        for name in ("compatibility_manifest_path", "native_qualification_path"):
            path = Path(str(getattr(config.scheduler, name))).expanduser()
            if not path.is_absolute():
                raise BackendError("config.scheduler_path_not_absolute", "Scheduler qualification paths must be absolute", subsystem="config", safe_context={"field": f"scheduler.{name}"})
        for name in ("compatibility_manifest_sha256", "native_qualification_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(config.scheduler, name))):
                raise BackendError("config.scheduler_digest_invalid", "Scheduler qualification digests must be lowercase SHA-256", subsystem="config", safe_context={"field": f"scheduler.{name}"})
    elif any(value is not None for value in scheduler_values) or config.scheduler.full_body_when_locked:
        raise BackendError("config.scheduler_disabled_has_authority", "Disabled scheduler configuration cannot retain authority fields", subsystem="config")
    if config.macos.whisper_cpp.revision != "v1.9.2":
        raise BackendError("config.invalid_field", "whisper.cpp revision must match the approved runtime", subsystem="config", safe_context={"field": "macos.whisper_cpp.revision"})
    if config.macos.asr_backend not in {"whisper_cpp", "faster_whisper_cpu_smoke"}:
        raise BackendError(
            "config.invalid_field",
            "macOS ASR backend must be whisper_cpp or faster_whisper_cpu_smoke",
            subsystem="config",
            safe_context={"field": "macos.asr_backend"},
        )
    if config.macos.whisper_cpp.language != "auto" and (
        len(config.macos.whisper_cpp.language) not in {2, 3}
        or not config.macos.whisper_cpp.language.isalpha()
    ):
        raise BackendError("config.invalid_field", "whisper.cpp language must be auto or an ISO language code", subsystem="config", safe_context={"field": "macos.whisper_cpp.language"})
    if config.macos.whisper_cpp.startup_timeout_seconds <= 0 or config.macos.whisper_cpp.request_timeout_seconds <= 0:
        raise BackendError("config.invalid_field", "whisper.cpp timeouts must be positive", subsystem="config", safe_context={"field": "macos.whisper_cpp.timeout"})
    if not 0.0 <= config.vad.threshold <= 1.0:
        raise BackendError("config.invalid_field", "VAD threshold must be between zero and one", subsystem="config", safe_context={"field": "vad.threshold"})
    return config


def load_voice_config(path: str | Path) -> VoiceConfig:
    try:
        return voice_config_from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("config.read_failed", "Configuration could not be read", subsystem="config") from exc
