"""Dependency-free JSON configuration for the source-run runtime."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .errors import BackendError
from .manifest import MODEL_MANIFESTS, MODEL_PROFILE_LARGE_V3_TURBO
from .ptt import FUNCTION_KEY_VKEYS
from .voice_auth import VoiceAuthError, VoiceAuthKey, load_voice_auth_key


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
    model_profile: str = "base"
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
    tts_qualification_path: str | None = None
    tts_qualification_sha256: str | None = None
    full_body_when_locked: bool = False
    native_helper_path: str | None = None
    native_helper_sha256: str | None = None
    native_helper_code_identity_sha256: str | None = None
    callback_qualification_path: str | None = None
    callback_qualification_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CoreMlEncoderConfig:
    revision: str | None = None
    directory: str | None = None
    files: tuple[ArtifactConfig, ...] = ()


@dataclass(frozen=True, slots=True)
class WhisperCppConfig:
    revision: str = "v1.9.2"
    executable: ArtifactConfig = field(default_factory=ArtifactConfig)
    model: ArtifactConfig = field(default_factory=ArtifactConfig)
    coreml_encoder: CoreMlEncoderConfig = field(default_factory=CoreMlEncoderConfig)
    language: str = "auto"
    threads: int = 4
    qualified_audio_context: int = 0
    qualified_audio_context_minimum_seconds: float = 3.0
    qualified_audio_context_maximum_seconds: float = 12.0
    startup_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 120.0
    maximum_response_bytes: int = 1_048_576


@dataclass(frozen=True, slots=True)
class GreekTts15Config:
    model_root: str | None = None
    approval_manifest_path: str | None = None
    approval_manifest_sha256: str | None = None
    repository_root: str | None = None
    warm_on_start: bool = True
    worker_python: ArtifactConfig = field(default_factory=ArtifactConfig)


@dataclass(frozen=True, slots=True)
class KokoroOnnxConfig:
    model: ArtifactConfig = field(default_factory=ArtifactConfig)
    voices: ArtifactConfig = field(default_factory=ArtifactConfig)
    language: str = "en-gb"
    speed: float = 0.96
    voice_mix: Mapping[str, float] = field(
        default_factory=lambda: {"bm_george": 0.70, "bm_fable": 0.30}
    )
    warm_on_start: bool = True


@dataclass(frozen=True, slots=True)
class PocketTtsConfig:
    model: ArtifactConfig = field(default_factory=ArtifactConfig)
    tokenizer: ArtifactConfig = field(default_factory=ArtifactConfig)
    voice_state: ArtifactConfig = field(default_factory=ArtifactConfig)
    temperature: float = 0.7
    max_tokens: int = 50
    warm_on_start: bool = True


@dataclass(frozen=True, slots=True)
class MacosConfig:
    asr_backend: str = "whisper_cpp"
    tts_backend: str = "system_say"
    whisper_cpp: WhisperCppConfig = field(default_factory=WhisperCppConfig)
    greektts15: GreekTts15Config = field(default_factory=GreekTts15Config)
    kokoro_onnx: KokoroOnnxConfig = field(default_factory=KokoroOnnxConfig)
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
    voice_auth_key_file: str | None = None
    windows_tts_backend: str = "sapi5"
    sapi_voices: Mapping[str, str] = field(default_factory=dict)
    pocket_tts: PocketTtsConfig = field(default_factory=PocketTtsConfig)
    macos: MacosConfig = field(default_factory=MacosConfig)
    logging_level: str = "INFO"


def is_loopback_websocket_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return (
            parsed.scheme == "ws"
            and parsed.hostname == "127.0.0.1"
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.params
            and not parsed.query
            and not parsed.fragment
            and parsed.port is not None
            and 1 <= parsed.port <= 65_535
        )
    except (TypeError, ValueError):
        return False


def configured_voice_auth_key(config: VoiceConfig) -> VoiceAuthKey:
    """Load the external per-install key or fail without disclosing its path."""

    if config.voice_auth_key_file is None:
        raise BackendError(
            "config.voice_auth_key_required",
            "A per-install Voice authentication key is required",
            subsystem="config",
        )
    try:
        return load_voice_auth_key(config.voice_auth_key_file)
    except VoiceAuthError as exc:
        raise BackendError(
            "config.voice_auth_key_invalid",
            "The Voice authentication key is unavailable or invalid",
            subsystem="config",
        ) from exc


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


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
        if isinstance(value, bool) or not math.isfinite(result):
            raise ValueError
    except (ValueError, TypeError, OverflowError) as exc:
        raise BackendError("config.invalid_field", "Configuration value must be a finite number", subsystem="config", safe_context={"field": field_name}) from exc
    return result


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
    greektts15_data = macos_data.get("greektts15", {}) if isinstance(macos_data, Mapping) else None
    greektts15_worker_data = greektts15_data.get("worker_python", {}) if isinstance(greektts15_data, Mapping) else None
    kokoro_data = macos_data.get("kokoro_onnx", {}) if isinstance(macos_data, Mapping) else None
    kokoro_model_data = kokoro_data.get("model", {}) if isinstance(kokoro_data, Mapping) else None
    kokoro_voices_data = kokoro_data.get("voices", {}) if isinstance(kokoro_data, Mapping) else None
    kokoro_mix_data = kokoro_data.get("voice_mix", {"bm_george": 0.70, "bm_fable": 0.30}) if isinstance(kokoro_data, Mapping) else None
    pocket_data = data.get("pocket_tts", {})
    pocket_model_data = pocket_data.get("model", {}) if isinstance(pocket_data, Mapping) else None
    pocket_tokenizer_data = pocket_data.get("tokenizer", {}) if isinstance(pocket_data, Mapping) else None
    pocket_voice_data = pocket_data.get("voice_state", {}) if isinstance(pocket_data, Mapping) else None
    executable_data = whisper_cpp_data.get("executable", {}) if isinstance(whisper_cpp_data, Mapping) else None
    mac_model_data = whisper_cpp_data.get("model", {}) if isinstance(whisper_cpp_data, Mapping) else None
    coreml_encoder_data = whisper_cpp_data.get("coreml_encoder", {}) if isinstance(whisper_cpp_data, Mapping) else None
    coreml_encoder_files_data = coreml_encoder_data.get("files", []) if isinstance(coreml_encoder_data, Mapping) else None
    if not isinstance(vad_data, Mapping) or not isinstance(artifact_data, Mapping):
        raise BackendError("config.invalid_field", "VAD configuration must be an object", subsystem="config", safe_context={"field": "vad"})
    if not isinstance(asr_data, Mapping) or not isinstance(cuda_data, Mapping):
        raise BackendError("config.invalid_field", "ASR configuration must be an object", subsystem="config", safe_context={"field": "asr"})
    if not isinstance(application_data, Mapping):
        raise BackendError("config.invalid_field", "Application configuration must be an object", subsystem="config", safe_context={"field": "application"})
    if not isinstance(scheduler_data, Mapping) or set(scheduler_data) - {
        "enabled", "compatibility_manifest_path", "compatibility_manifest_sha256",
        "native_qualification_path", "native_qualification_sha256",
        "tts_qualification_path", "tts_qualification_sha256", "full_body_when_locked",
        "native_helper_path", "native_helper_sha256", "native_helper_code_identity_sha256",
        "callback_qualification_path", "callback_qualification_sha256",
    }:
        raise BackendError("config.invalid_field", "Scheduler configuration is invalid", subsystem="config", safe_context={"field": "scheduler"})
    if not isinstance(macos_data, Mapping) or set(macos_data) - {
        "asr_backend", "tts_backend", "whisper_cpp", "greektts15", "kokoro_onnx", "system_voices"
    }:
        raise BackendError("config.invalid_field", "macOS configuration has unknown fields", subsystem="config", safe_context={"field": "macos"})
    if not isinstance(whisper_cpp_data, Mapping) or set(whisper_cpp_data) - {
        "revision", "executable", "model", "coreml_encoder", "language", "threads",
        "qualified_audio_context", "qualified_audio_context_minimum_seconds",
        "qualified_audio_context_maximum_seconds", "startup_timeout_seconds",
        "request_timeout_seconds", "maximum_response_bytes"
    }:
        raise BackendError("config.invalid_field", "whisper.cpp configuration has unknown fields", subsystem="config", safe_context={"field": "macos.whisper_cpp"})
    if not isinstance(executable_data, Mapping) or set(executable_data) - {"path", "size", "sha256"}:
        raise BackendError("config.invalid_field", "whisper.cpp executable manifest is invalid", subsystem="config", safe_context={"field": "macos.whisper_cpp.executable"})
    if not isinstance(mac_model_data, Mapping) or set(mac_model_data) - {"path", "size", "sha256"}:
        raise BackendError("config.invalid_field", "whisper.cpp model manifest is invalid", subsystem="config", safe_context={"field": "macos.whisper_cpp.model"})
    if (
        not isinstance(coreml_encoder_data, Mapping)
        or set(coreml_encoder_data) - {"revision", "directory", "files"}
        or not isinstance(coreml_encoder_files_data, list)
        or any(
            not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}
            for item in coreml_encoder_files_data
        )
    ):
        raise BackendError("config.invalid_field", "Core ML encoder manifest is invalid", subsystem="config", safe_context={"field": "macos.whisper_cpp.coreml_encoder"})
    if not isinstance(greektts15_data, Mapping) or set(greektts15_data) - {
        "model_root", "approval_manifest_path", "approval_manifest_sha256", "repository_root", "warm_on_start", "worker_python"
    }:
        raise BackendError("config.invalid_field", "GreekTTS runtime configuration is invalid", subsystem="config", safe_context={"field": "macos.greektts15"})
    if not isinstance(greektts15_worker_data, Mapping) or set(greektts15_worker_data) - {"path", "size", "sha256"}:
        raise BackendError("config.invalid_field", "GreekTTS worker manifest is invalid", subsystem="config", safe_context={"field": "macos.greektts15.worker_python"})
    if not isinstance(kokoro_data, Mapping) or set(kokoro_data) - {
        "model", "voices", "language", "speed", "voice_mix", "warm_on_start"
    }:
        raise BackendError("config.invalid_field", "Kokoro ONNX configuration is invalid", subsystem="config", safe_context={"field": "macos.kokoro_onnx"})
    if (
        not isinstance(kokoro_model_data, Mapping)
        or set(kokoro_model_data) - {"path", "size", "sha256"}
        or not isinstance(kokoro_voices_data, Mapping)
        or set(kokoro_voices_data) - {"path", "size", "sha256"}
        or not isinstance(kokoro_mix_data, Mapping)
    ):
        raise BackendError("config.invalid_field", "Kokoro ONNX artifact manifest is invalid", subsystem="config", safe_context={"field": "macos.kokoro_onnx"})
    if not isinstance(pocket_data, Mapping) or set(pocket_data) - {
        "model", "tokenizer", "voice_state", "temperature", "max_tokens", "warm_on_start"
    }:
        raise BackendError(
            "config.invalid_field",
            "Pocket TTS configuration is invalid",
            subsystem="config",
            safe_context={"field": "pocket_tts"},
        )
    for field_name, artifact_data_value in (
        ("model", pocket_model_data),
        ("tokenizer", pocket_tokenizer_data),
        ("voice_state", pocket_voice_data),
    ):
        if not isinstance(artifact_data_value, Mapping) or set(artifact_data_value) - {
            "path", "size", "sha256"
        }:
            raise BackendError(
                "config.invalid_field",
                "Pocket TTS artifact manifest is invalid",
                subsystem="config",
                safe_context={"field": f"pocket_tts.{field_name}"},
            )
    url = str(data.get("orchestrator_url", "ws://127.0.0.1:8765"))
    if not is_loopback_websocket_url(url):
        raise BackendError("config.non_loopback_orchestrator", "Only loopback WebSocket URLs are permitted", subsystem="config", safe_context={"field": "orchestrator_url"})
    voice_auth_key_file = data.get("voice_auth_key_file")
    if voice_auth_key_file is not None:
        if not isinstance(voice_auth_key_file, str) or not voice_auth_key_file:
            raise BackendError(
                "config.voice_auth_key_path_invalid",
                "Voice authentication key path must be absolute",
                subsystem="config",
                safe_context={"field": "voice_auth_key_file"},
            )
        selected_auth_key = Path(voice_auth_key_file).expanduser()
        if not selected_auth_key.is_absolute():
            raise BackendError(
                "config.voice_auth_key_path_invalid",
                "Voice authentication key path must be absolute",
                subsystem="config",
                safe_context={"field": "voice_auth_key_file"},
            )
        voice_auth_key_file = str(selected_auth_key)
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
        maximum_hold_seconds=_finite_float(data.get("maximum_hold_seconds", 30.0), "maximum_hold_seconds"),
        maximum_utterance_seconds=_finite_float(data.get("maximum_utterance_seconds", 30.0), "maximum_utterance_seconds"),
        audio_input_device=data.get("audio_input_device"),
        whisper_model_directory=data.get("whisper_model_directory"),
        asr=AsrConfig(
            model_profile=str(asr_data.get("model_profile", "base")).lower(),
            execution_profile=str(asr_data.get("execution_profile", "cpu")).lower(),
            cpu_compute_type=str(asr_data.get("cpu_compute_type", "int8")),
            cuda=CudaAsrConfig(
                device_index=cuda_data.get("device_index"),
                compute_type=str(cuda_data.get("compute_type", "int8_float16")),
                residency=str(cuda_data.get("residency", "turn_scoped")).lower(),
                minimum_free_vram_mb=_non_negative_int(cuda_data.get("minimum_free_vram_mb", 0), "asr.cuda.minimum_free_vram_mb"),
                reserve_for_other_workloads_mb=_non_negative_int(cuda_data.get("reserve_for_other_workloads_mb", 0), "asr.cuda.reserve_for_other_workloads_mb"),
                fallback_to_cpu=_boolean(cuda_data.get("fallback_to_cpu", True), "asr.cuda.fallback_to_cpu"),
                worker_start_timeout_seconds=_finite_float(cuda_data.get("worker_start_timeout_seconds", 30.0), "asr.cuda.worker_start_timeout_seconds"),
                transcription_timeout_seconds=_finite_float(cuda_data.get("transcription_timeout_seconds", 120.0), "asr.cuda.transcription_timeout_seconds"),
                warm_restart_limit=_non_negative_int(cuda_data.get("warm_restart_limit", 1), "asr.cuda.warm_restart_limit"),
                vram_poll_interval_ms=_positive_int(cuda_data.get("vram_poll_interval_ms", 100), "asr.cuda.vram_poll_interval_ms"),
            ),
        ),
        application=ApplicationConfig(
            shutdown_timeout_seconds=_finite_float(application_data.get("shutdown_timeout_seconds", 5.0), "application.shutdown_timeout_seconds"),
            log_maximum_bytes=_positive_int(application_data.get("log_maximum_bytes", 1_048_576), "application.log_maximum_bytes"),
            log_backup_count=_non_negative_int(application_data.get("log_backup_count", 3), "application.log_backup_count"),
        ),
        scheduler=SchedulerClientConfig(
            enabled=_boolean(scheduler_data.get("enabled", False), "scheduler.enabled"),
            compatibility_manifest_path=scheduler_data.get("compatibility_manifest_path"),
            compatibility_manifest_sha256=scheduler_data.get("compatibility_manifest_sha256"),
            native_qualification_path=scheduler_data.get("native_qualification_path"),
            native_qualification_sha256=scheduler_data.get("native_qualification_sha256"),
            tts_qualification_path=scheduler_data.get("tts_qualification_path"),
            tts_qualification_sha256=scheduler_data.get("tts_qualification_sha256"),
            full_body_when_locked=_boolean(
                scheduler_data.get("full_body_when_locked", False),
                "scheduler.full_body_when_locked",
            ),
            native_helper_path=scheduler_data.get("native_helper_path"),
            native_helper_sha256=scheduler_data.get("native_helper_sha256"),
            native_helper_code_identity_sha256=scheduler_data.get(
                "native_helper_code_identity_sha256"
            ),
            callback_qualification_path=scheduler_data.get("callback_qualification_path"),
            callback_qualification_sha256=scheduler_data.get("callback_qualification_sha256"),
        ),
        vad=VadConfig(
            artifact=ArtifactConfig(artifact_data.get("path"), artifact_data.get("size"), artifact_data.get("sha256")),
            threshold=_finite_float(vad_data.get("threshold", 0.5), "vad.threshold"),
            minimum_speech_ms=_non_negative_int(vad_data.get("minimum_speech_ms", 80), "vad.minimum_speech_ms"),
            padding_ms=_non_negative_int(vad_data.get("padding_ms", 80), "vad.padding_ms"),
        ),
        orchestrator_url=url,
        voice_auth_key_file=voice_auth_key_file,
        windows_tts_backend=str(data.get("windows_tts_backend", "sapi5")).lower(),
        sapi_voices=dict(voices),
        pocket_tts=PocketTtsConfig(
            model=ArtifactConfig(
                pocket_model_data.get("path"),
                pocket_model_data.get("size"),
                pocket_model_data.get("sha256"),
            ),
            tokenizer=ArtifactConfig(
                pocket_tokenizer_data.get("path"),
                pocket_tokenizer_data.get("size"),
                pocket_tokenizer_data.get("sha256"),
            ),
            voice_state=ArtifactConfig(
                pocket_voice_data.get("path"),
                pocket_voice_data.get("size"),
                pocket_voice_data.get("sha256"),
            ),
            temperature=_finite_float(pocket_data.get("temperature", 0.7), "pocket_tts.temperature"),
            max_tokens=_positive_int(pocket_data.get("max_tokens", 50), "pocket_tts.max_tokens"),
            warm_on_start=_boolean(pocket_data.get("warm_on_start", True), "pocket_tts.warm_on_start"),
        ),
        macos=MacosConfig(
            asr_backend=str(macos_data.get("asr_backend", "whisper_cpp")).lower(),
            tts_backend=str(macos_data.get("tts_backend", "system_say")).lower(),
            whisper_cpp=WhisperCppConfig(
                revision=str(whisper_cpp_data.get("revision", "v1.9.2")),
                executable=ArtifactConfig(executable_data.get("path"), executable_data.get("size"), executable_data.get("sha256")),
                model=ArtifactConfig(mac_model_data.get("path"), mac_model_data.get("size"), mac_model_data.get("sha256")),
                coreml_encoder=CoreMlEncoderConfig(
                    revision=coreml_encoder_data.get("revision"),
                    directory=coreml_encoder_data.get("directory"),
                    files=tuple(
                        ArtifactConfig(item.get("path"), item.get("size"), item.get("sha256"))
                        for item in coreml_encoder_files_data
                    ),
                ),
                language=str(whisper_cpp_data.get("language", "auto")).lower(),
                threads=_positive_int(whisper_cpp_data.get("threads", 4), "macos.whisper_cpp.threads"),
                qualified_audio_context=_non_negative_int(
                    whisper_cpp_data.get("qualified_audio_context", 0),
                    "macos.whisper_cpp.qualified_audio_context",
                ),
                qualified_audio_context_minimum_seconds=_finite_float(
                    whisper_cpp_data.get("qualified_audio_context_minimum_seconds", 3.0),
                    "macos.whisper_cpp.qualified_audio_context_minimum_seconds",
                ),
                qualified_audio_context_maximum_seconds=_finite_float(
                    whisper_cpp_data.get("qualified_audio_context_maximum_seconds", 12.0),
                    "macos.whisper_cpp.qualified_audio_context_maximum_seconds",
                ),
                startup_timeout_seconds=_finite_float(whisper_cpp_data.get("startup_timeout_seconds", 30.0), "macos.whisper_cpp.startup_timeout_seconds"),
                request_timeout_seconds=_finite_float(whisper_cpp_data.get("request_timeout_seconds", 120.0), "macos.whisper_cpp.request_timeout_seconds"),
                maximum_response_bytes=_positive_int(whisper_cpp_data.get("maximum_response_bytes", 1_048_576), "macos.whisper_cpp.maximum_response_bytes"),
            ),
            greektts15=GreekTts15Config(
                model_root=greektts15_data.get("model_root"),
                approval_manifest_path=greektts15_data.get("approval_manifest_path"),
                approval_manifest_sha256=greektts15_data.get("approval_manifest_sha256"),
                repository_root=greektts15_data.get("repository_root"),
                warm_on_start=_boolean(greektts15_data.get("warm_on_start", True), "macos.greektts15.warm_on_start"),
                worker_python=ArtifactConfig(
                    greektts15_worker_data.get("path"),
                    greektts15_worker_data.get("size"),
                    greektts15_worker_data.get("sha256"),
                ),
            ),
            kokoro_onnx=KokoroOnnxConfig(
                model=ArtifactConfig(kokoro_model_data.get("path"), kokoro_model_data.get("size"), kokoro_model_data.get("sha256")),
                voices=ArtifactConfig(kokoro_voices_data.get("path"), kokoro_voices_data.get("size"), kokoro_voices_data.get("sha256")),
                language=str(kokoro_data.get("language", "en-gb")).lower(),
                speed=_finite_float(kokoro_data.get("speed", 0.96), "macos.kokoro_onnx.speed"),
                voice_mix={str(name): _finite_float(weight, "macos.kokoro_onnx.voice_mix") for name, weight in kokoro_mix_data.items()},
                warm_on_start=_boolean(kokoro_data.get("warm_on_start", True), "macos.kokoro_onnx.warm_on_start"),
            ),
            system_voices=dict(macos_voices),
        ),
        logging_level=str(data.get("logging_level", "INFO")).upper(),
    )
    if config.ptt_key not in FUNCTION_KEY_VKEYS and config.ptt_key not in {"CONTROL_OPTION", "RIGHT_OPTION"}:
        raise BackendError(
            "config.unsupported_ptt_key",
            "PTT key must be F1 through F24 or an approved macOS modifier input",
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
    if config.asr.model_profile not in MODEL_MANIFESTS:
        raise BackendError(
            "config.invalid_field",
            "ASR model profile is not supported",
            subsystem="config",
            safe_context={"field": "asr.model_profile"},
        )
    if config.asr.cuda.residency not in {"turn_scoped", "warm"}:
        raise BackendError("config.invalid_field", "CUDA residency must be turn_scoped or warm", subsystem="config", safe_context={"field": "asr.cuda.residency"})
    if config.windows_tts_backend not in {"sapi5", "pocket_tts"}:
        raise BackendError(
            "config.invalid_field",
            "Windows TTS backend is invalid",
            subsystem="config",
            safe_context={"field": "windows_tts_backend"},
        )
    if config.macos.tts_backend not in {"system_say", "greektts15_f1", "kokoro_onnx", "pocket_tts"}:
        raise BackendError("config.invalid_field", "macOS TTS backend is invalid", subsystem="config", safe_context={"field": "macos.tts_backend"})
    if "pocket_tts" in {config.windows_tts_backend, config.macos.tts_backend}:
        pocket = config.pocket_tts
        if abs(pocket.temperature - 0.7) > 1e-9 or pocket.max_tokens != 50:
            raise BackendError(
                "config.pocket_tts_candidate_locked",
                "Pocket TTS settings do not match the approved candidate profile",
                subsystem="config",
            )
        for artifact in (pocket.model, pocket.tokenizer, pocket.voice_state):
            if (
                not isinstance(artifact.path, str)
                or not Path(artifact.path).is_absolute()
                or not isinstance(artifact.size, int)
                or isinstance(artifact.size, bool)
                or artifact.size < 1
                or not isinstance(artifact.sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", artifact.sha256) is None
            ):
                raise BackendError(
                    "config.pocket_tts_profile_required",
                    "Pocket TTS requires absolute, digest-bound artifacts",
                    subsystem="config",
                )
    if config.macos.tts_backend == "kokoro_onnx":
        kokoro = config.macos.kokoro_onnx
        if (
            kokoro.language != "en-gb"
            or abs(kokoro.speed - 0.96) > 1e-9
            or dict(kokoro.voice_mix) != {"bm_george": 0.70, "bm_fable": 0.30}
        ):
            raise BackendError("config.kokoro_candidate_locked", "Kokoro candidate settings do not match the approved profile", subsystem="config")
        for artifact in (kokoro.model, kokoro.voices):
            if (
                not isinstance(artifact.path, str)
                or not Path(artifact.path).is_absolute()
                or not isinstance(artifact.size, int)
                or isinstance(artifact.size, bool)
                or artifact.size < 1
                or not isinstance(artifact.sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", artifact.sha256) is None
            ):
                raise BackendError("config.kokoro_profile_required", "Kokoro requires absolute, digest-bound artifacts", subsystem="config")
    if config.macos.tts_backend == "greektts15_f1":
        greek = config.macos.greektts15
        paths = (greek.model_root, greek.approval_manifest_path, greek.repository_root)
        if (
            any(not isinstance(value, str) or not Path(value).is_absolute() for value in paths)
            or not isinstance(greek.approval_manifest_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", greek.approval_manifest_sha256) is None
        ):
            raise BackendError("config.greektts15_profile_required", "GreekTTS requires absolute, digest-bound runtime paths", subsystem="config")
        worker = greek.worker_python
        worker_values = (worker.path, worker.size, worker.sha256)
        if any(value is not None for value in worker_values) and (
            not isinstance(worker.path, str)
            or not Path(worker.path).is_absolute()
            or not isinstance(worker.size, int)
            or isinstance(worker.size, bool)
            or worker.size < 1
            or not isinstance(worker.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", worker.sha256) is None
        ):
            raise BackendError("config.greektts15_worker_required", "GreekTTS worker requires an absolute digest-bound interpreter", subsystem="config")
    if config.asr.execution_profile == "cuda" and (
        not isinstance(config.asr.cuda.device_index, int)
        or isinstance(config.asr.cuda.device_index, bool)
        or config.asr.cuda.device_index < 0
    ):
        raise BackendError("config.cuda_device_required", "CUDA profile requires an explicit non-negative device index", subsystem="config", safe_context={"field": "asr.cuda.device_index"})
    if (
        config.asr.model_profile == MODEL_PROFILE_LARGE_V3_TURBO
        and config.asr.execution_profile == "cuda"
        and (
            config.asr.cuda.compute_type != "int8_float16"
            or config.asr.cuda.minimum_free_vram_mb < 2_048
            or config.asr.cuda.reserve_for_other_workloads_mb < 2_048
            or not config.asr.cuda.fallback_to_cpu
        )
    ):
        raise BackendError(
            "config.large_v3_turbo_cuda_profile_required",
            "Large-v3-turbo CUDA requires the qualified coexistence safety profile",
            subsystem="config",
            safe_context={"field": "asr.cuda"},
        )
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
    scheduler_tts_values = (
        config.scheduler.tts_qualification_path,
        config.scheduler.tts_qualification_sha256,
    )
    scheduler_helper_values = (
        config.scheduler.native_helper_path,
        config.scheduler.native_helper_sha256,
        config.scheduler.native_helper_code_identity_sha256,
    )
    scheduler_callback_values = (
        config.scheduler.callback_qualification_path,
        config.scheduler.callback_qualification_sha256,
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
        if any(value is None for value in scheduler_tts_values) and any(
            value is not None for value in scheduler_tts_values
        ):
            raise BackendError("config.scheduler_tts_qualification_incomplete", "Scheduler TTS qualification path and digest must be paired", subsystem="config")
        if all(value is not None for value in scheduler_tts_values):
            tts_path = Path(str(config.scheduler.tts_qualification_path)).expanduser()
            if not tts_path.is_absolute():
                raise BackendError("config.scheduler_path_not_absolute", "Scheduler qualification paths must be absolute", subsystem="config", safe_context={"field": "scheduler.tts_qualification_path"})
            if not re.fullmatch(r"[0-9a-f]{64}", str(config.scheduler.tts_qualification_sha256)):
                raise BackendError("config.scheduler_digest_invalid", "Scheduler qualification digests must be lowercase SHA-256", subsystem="config", safe_context={"field": "scheduler.tts_qualification_sha256"})
        if any(value is None for value in scheduler_helper_values) and any(
            value is not None for value in scheduler_helper_values
        ):
            raise BackendError(
                "config.scheduler_native_helper_incomplete",
                "Scheduler native helper path and digests must be complete",
                subsystem="config",
            )
        if all(value is not None for value in scheduler_helper_values):
            helper_path = Path(str(config.scheduler.native_helper_path)).expanduser()
            if not helper_path.is_absolute():
                raise BackendError(
                    "config.scheduler_path_not_absolute",
                    "Scheduler native helper path must be absolute",
                    subsystem="config",
                    safe_context={"field": "scheduler.native_helper_path"},
                )
            for name in ("native_helper_sha256", "native_helper_code_identity_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(config.scheduler, name))):
                    raise BackendError(
                        "config.scheduler_digest_invalid",
                        "Scheduler native helper digests must be lowercase SHA-256",
                        subsystem="config",
                        safe_context={"field": f"scheduler.{name}"},
                    )
        if any(value is None for value in scheduler_callback_values) and any(
            value is not None for value in scheduler_callback_values
        ):
            raise BackendError(
                "config.scheduler_callback_qualification_incomplete",
                "Scheduler callback qualification path and digest must be paired",
                subsystem="config",
            )
        if all(value is not None for value in scheduler_callback_values):
            callback_path = Path(str(config.scheduler.callback_qualification_path)).expanduser()
            if not callback_path.is_absolute():
                raise BackendError(
                    "config.scheduler_path_not_absolute",
                    "Scheduler callback qualification path must be absolute",
                    subsystem="config",
                    safe_context={"field": "scheduler.callback_qualification_path"},
                )
            if not re.fullmatch(r"[0-9a-f]{64}", str(config.scheduler.callback_qualification_sha256)):
                raise BackendError(
                    "config.scheduler_digest_invalid",
                    "Scheduler callback qualification digest must be lowercase SHA-256",
                    subsystem="config",
                    safe_context={"field": "scheduler.callback_qualification_sha256"},
                )
        if all(value is not None for value in scheduler_helper_values) != all(
            value is not None for value in scheduler_callback_values
        ):
            raise BackendError(
                "config.scheduler_callback_authority_incomplete",
                "Native callback helper and qualification must be enabled together",
                subsystem="config",
            )
    elif (
        any(value is not None for value in scheduler_values)
        or any(value is not None for value in scheduler_tts_values)
        or any(value is not None for value in scheduler_helper_values)
        or any(value is not None for value in scheduler_callback_values)
        or config.scheduler.full_body_when_locked
    ):
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
    coreml = config.macos.whisper_cpp.coreml_encoder
    coreml_configured = any(
        value is not None for value in (coreml.revision, coreml.directory)
    ) or bool(coreml.files)
    if coreml_configured:
        relative_paths = [artifact.path for artifact in coreml.files]
        if (
            not isinstance(coreml.revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", coreml.revision) is None
            or not isinstance(coreml.directory, str)
            or not Path(coreml.directory).is_absolute()
            or not coreml.files
            or any(
                not isinstance(artifact.path, str)
                or not artifact.path
                or "\\" in artifact.path
                or Path(artifact.path).is_absolute()
                or not Path(artifact.path).parts
                or Path(artifact.path).as_posix() != artifact.path
                or ".." in Path(artifact.path).parts
                or not isinstance(artifact.size, int)
                or isinstance(artifact.size, bool)
                or artifact.size < 1
                or not isinstance(artifact.sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", artifact.sha256) is None
                for artifact in coreml.files
            )
            or len(set(relative_paths)) != len(relative_paths)
        ):
            raise BackendError(
                "config.whisper_cpp_coreml_profile_required",
                "Core ML requires an absolute directory and a complete digest-bound file manifest",
                subsystem="config",
            )
        if config.macos.whisper_cpp.qualified_audio_context != 0:
            raise BackendError(
                "config.whisper_cpp_coreml_full_context_required",
                "The qualified Core ML profile requires full decoder context",
                subsystem="config",
            )
    if not 0 <= config.macos.whisper_cpp.qualified_audio_context <= 1500:
        raise BackendError(
            "config.invalid_field",
            "whisper.cpp qualified audio context must be between zero and 1500",
            subsystem="config",
            safe_context={"field": "macos.whisper_cpp.qualified_audio_context"},
        )
    if (
        config.macos.whisper_cpp.qualified_audio_context_minimum_seconds < 0
        or config.macos.whisper_cpp.qualified_audio_context_maximum_seconds
        < config.macos.whisper_cpp.qualified_audio_context_minimum_seconds
    ):
        raise BackendError(
            "config.invalid_field",
            "whisper.cpp qualified audio-context duration window is invalid",
            subsystem="config",
            safe_context={"field": "macos.whisper_cpp.qualified_audio_context_duration"},
        )
    if not 0.0 <= config.vad.threshold <= 1.0:
        raise BackendError("config.invalid_field", "VAD threshold must be between zero and one", subsystem="config", safe_context={"field": "vad.threshold"})
    return config


def load_voice_config(path: str | Path) -> VoiceConfig:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise BackendError("config.read_failed", "Configuration could not be read", subsystem="config") from exc
    return voice_config_from_mapping(document)
