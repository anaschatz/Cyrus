"""Native Windows qualification runner, packaged as a sealed local executable.

The runner owns generic gateway protocol checks and then launches the separate
hash-pinned ASR collector.  The collector receives paths and a boot-scoped
loopback bearer only through its child environment; it returns metrics only.
Neither process is allowed to write prompts, outputs, audio, transcripts or
secrets into the evidence document, diagnostic, or a command line.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from .macos_fixture import (
    FixtureFailure,
    QualificationClient,
    _phase_cancel,
    _read_identity,
)
from .private_temp import PrivateLaunchDirectory, private_temp_environment
from .qualification import SUITE_VERSION
from .strict_json import canonical_bytes, parse_body
from .windows_asr_corpus import WindowsAsrCorpusError, validate_corpus_manifest
from .windows_process import WindowsProcessFailure, run_windows_job

_STAGES = {
    "bootstrap",
    "streaming",
    "tools",
    "json_schema",
    "cancellation",
    "crash_restart",
    "isolation",
    "asr_coexistence",
    "evidence_write",
}
_current_stage = "bootstrap"
_HEX = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_CONTENT_KEYS = {
    "audio",
    "content",
    "message",
    "messages",
    "output",
    "prompt",
    "raw_output",
    "response",
    "secret",
    "transcript",
}
_ASSET_NAMES = ("fixture", "collector", "audio_corpus", "power_profile", "asr_config")
_ASSET_LIMITS = {
    "fixture": 65_536,
    "collector": 1_073_741_824,
    "audio_corpus": 262_144,
    "power_profile": 65_536,
    "asr_config": 65_536,
}
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _stage(value: str) -> None:
    global _current_stage
    if value not in _STAGES:
        raise FixtureFailure("invalid qualification stage")
    _current_stage = value


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise FixtureFailure(f"invalid {name} shape")
    return value


def _finite_metric(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, Decimal))
        and math.isfinite(float(value))
    )


def _is_local_windows_absolute(value: str) -> bool:
    path = PureWindowsPath(value)
    return bool(
        path.is_absolute()
        and len(path.drive) == 2
        and path.drive[0].isalpha()
        and path.drive[1] == ":"
    )


def _native_drive_type(root: str) -> int:  # pragma: no cover - native Windows only
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetDriveTypeW(root))
    except BaseException:
        return 0


def _windows_drive_is_local(
    value: str,
    *,
    drive_type_reader: Callable[[str], int] | None = None,
) -> bool:
    path = PureWindowsPath(value)
    if not _is_local_windows_absolute(value):
        return False
    reader = drive_type_reader or _native_drive_type
    try:
        return reader(path.anchor) in {2, 3, 5, 6}
    except BaseException:
        return False


def _safe_existing_chain(path: Path, *, final_directory: bool, name: str) -> None:
    if (
        not path.is_absolute()
        or any(component in {".", ".."} for component in path.parts)
        or (os.name == "nt" and not _windows_drive_is_local(str(path)))
    ):
        raise FixtureFailure(f"unsafe {name} path")
    current = Path(path.anchor)
    components = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(components):
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise FixtureFailure(f"unsafe {name} path") from exc
        if current.is_symlink() or (
            getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT
        ):
            raise FixtureFailure(f"unsafe {name} path")
        is_final = index == len(components) - 1
        if not is_final and not stat.S_ISDIR(metadata.st_mode):
            raise FixtureFailure(f"unsafe {name} path")
        if is_final:
            expected = stat.S_ISDIR if final_directory else stat.S_ISREG
            if not expected(metadata.st_mode):
                raise FixtureFailure(f"unsafe {name} path")


def _safe_file(path: Path, *, maximum: int, name: str) -> bytes:
    _safe_existing_chain(path, final_directory=False, name=name)
    try:
        info = path.lstat()
    except OSError as exc:
        raise FixtureFailure(f"unsafe {name} path") from exc
    reparse = bool(
        os.name == "nt"
        and getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or reparse
        or info.st_size <= 0
        or info.st_size > maximum
    ):
        raise FixtureFailure(f"unsafe {name} path")
    try:
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise FixtureFailure(f"unreadable {name}") from exc
    if (
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or path.is_symlink()
        or getattr(after, "st_file_attributes", 0) & _REPARSE_POINT
        or len(raw) != info.st_size
    ):
        raise FixtureFailure(f"unstable {name}")
    return raw


def _contains_content(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            not isinstance(key, str)
            or key.casefold() in _CONTENT_KEYS
            or _contains_content(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_content(child) for child in value)
    if isinstance(value, str):
        return len(value.encode("utf-8")) > 128
    return False


def _asset(path: Path, *, maximum: int, name: str, expected_sha256: str) -> tuple[Path, dict[str, Any], bytes]:
    if not path.is_absolute() or _HEX.fullmatch(expected_sha256) is None:
        raise FixtureFailure("qualification asset identity is invalid")
    raw = _safe_file(path, maximum=maximum, name=name)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise FixtureFailure("qualification asset identity changed")
    return path, {"path": None, "size_bytes": len(raw), "sha256": digest}, raw


def _load_power(raw: bytes) -> dict[str, Any]:
    value = _exact(
        parse_body(raw),
        {
            "windows_power_scheme_guid",
            "ac_state",
            "gpu_power_limit_watts",
            "gpu_clock_policy",
            "driver_version",
            "compute_runtime",
            "require_zero_thermal_power_throttle_reasons",
        },
        "power profile",
    )
    if (
        not isinstance(value["windows_power_scheme_guid"], str)
        or re.fullmatch(
            r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
            value["windows_power_scheme_guid"],
        ) is None
        or value["ac_state"] != "ac"
        or isinstance(value["gpu_power_limit_watts"], bool)
        or not isinstance(value["gpu_power_limit_watts"], (int, float, Decimal))
        or not math.isfinite(float(value["gpu_power_limit_watts"]))
        or value["gpu_power_limit_watts"] <= 0
        or any(not isinstance(value[key], str) or not value[key] for key in ("gpu_clock_policy", "driver_version", "compute_runtime"))
        or re.fullmatch(r"CUDA [0-9]+\.[0-9]+", value["compute_runtime"]) is None
        or re.fullmatch(
            r"applications_graphics_mhz=[^;\r\n]{1,32};"
            r"applications_memory_mhz=[^;\r\n]{1,32}",
            value["gpu_clock_policy"],
        )
        is None
        or value["require_zero_thermal_power_throttle_reasons"] is not True
    ):
        raise FixtureFailure("power profile contract failed")
    return value


def _load_asr_config(raw: bytes) -> dict[str, Any]:
    value = _exact(
        parse_body(raw),
        {
            "schema_version",
            "backend",
            "model_pack",
            "model_directory",
            "model_profile",
            "execution_profile",
            "cpu_compute_type",
            "cuda",
            "sample_rate_hz",
            "channels",
            "sample_width_bytes",
            "language_mode",
            "maximum_utterance_seconds",
        },
        "ASR configuration",
    )
    pack = _exact(value["model_pack"], {"id", "revision", "artifacts_sha256"}, "ASR model pack")
    cuda = _exact(
        value["cuda"],
        {
            "device_index",
            "compute_type",
            "minimum_free_vram_mb",
            "reserve_for_other_workloads_mb",
            "fallback_to_cpu",
            "worker_start_timeout_seconds",
            "transcription_timeout_seconds",
            "warm_restart_limit",
            "vram_poll_interval_ms",
        },
        "ASR CUDA profile",
    )
    if (
        value["schema_version"] != 1
        or value["backend"] != "faster-whisper"
        or not isinstance(value["model_directory"], str)
        or not Path(value["model_directory"]).is_absolute()
        or any(part in {".", ".."} for part in Path(value["model_directory"]).parts)
        or (
            os.name == "nt"
            and not _windows_drive_is_local(value["model_directory"])
        )
        or value["model_profile"] not in {"base", "large-v3-turbo"}
        or value["execution_profile"] != "cuda_turn_scoped"
        or value["cpu_compute_type"] != "int8"
        or value["sample_rate_hz"] != 16000
        or value["channels"] != 1
        or value["sample_width_bytes"] != 2
        or value["language_mode"] != "multilingual-auto"
        or value["maximum_utterance_seconds"] != 30
        or any(not isinstance(pack[key], str) or not pack[key] for key in ("id", "revision"))
        or _HEX.fullmatch(pack["artifacts_sha256"]) is None
        or type(cuda["device_index"]) is not int
        or cuda["device_index"] < 0
        or cuda["compute_type"] != "int8_float16"
        or any(
            type(cuda[key]) is not int or cuda[key] < 2048
            for key in ("minimum_free_vram_mb", "reserve_for_other_workloads_mb")
        )
        or cuda["fallback_to_cpu"] is not True
        or any(
            type(cuda[key]) is not int or not 0 < cuda[key] <= 600
            for key in ("worker_start_timeout_seconds", "transcription_timeout_seconds")
        )
        or cuda["warm_restart_limit"] != 0
        or cuda["vram_poll_interval_ms"] != 100
    ):
        raise FixtureFailure("ASR configuration contract failed")
    return value


def _load_fixture(raw: bytes, power: dict[str, Any]) -> dict[str, Any]:
    fixture = _exact(
        parse_body(raw),
        {
            "fixture_schema_version",
            "fixture_id",
            "platform",
            "prompt_workload_units",
            "max_completion_tokens",
            "request_count",
            "concurrency",
            "schedule",
            "minimum_overlapping_utterances",
            "sampling",
            "audio_corpus",
        },
        "Windows coexistence fixture",
    )
    corpus = _exact(
        fixture["audio_corpus"],
        {
            "schema",
            "utterance_count",
            "language_counts",
            "sample_rate_hz",
            "channels",
            "sample_width_bytes",
            "minimum_duration_ms",
            "maximum_duration_ms",
            "clean_language_accuracy_minimum_correct_per_language",
            "clean_aggregate_wer_max_percent_per_language",
            "coexistence_selection",
            "reference_sidecars",
        },
        "fixture audio corpus",
    )
    if (
        fixture["fixture_schema_version"] != 1
        or fixture["fixture_id"] != "cyrus-windows-asr-coexistence-v1"
        or fixture["platform"] != "windows-x86_64"
        or fixture["prompt_workload_units"] != [512, 2048, 4096]
        or fixture["max_completion_tokens"] != 512
        or fixture["request_count"] != 100
        or fixture["concurrency"] != 1
        or fixture["schedule"] != "serial-continuous-from-audio-start"
        or fixture["minimum_overlapping_utterances"] != 80
        or fixture["sampling"] != {"temperature": 0, "top_p": 1}
        or corpus
        != {
            "schema": "cyrus.windows-asr-audio-corpus/1",
            "utterance_count": 200,
            "language_counts": {"el": 100, "en": 100},
            "sample_rate_hz": 16000,
            "channels": 1,
            "sample_width_bytes": 2,
            "minimum_duration_ms": 1500,
            "maximum_duration_ms": 30000,
            "clean_language_accuracy_minimum_correct_per_language": 95,
            "clean_aggregate_wer_max_percent_per_language": 20,
            "coexistence_selection": "first-50-per-locale",
            "reference_sidecars": {
                "encoding": "utf-8",
                "normalization": "NFC",
                "controls_allowed": False,
                "word_normalization": "unicode-lmn-casefold-v1",
            },
        }
    ):
        raise FixtureFailure("Windows coexistence fixture contract failed")
    if _contains_content(fixture):
        raise FixtureFailure("Windows fixture contains content")
    # `power` is not embedded in the source fixture: it is captured on the
    # native PC and hashed separately.  The generic qualifier verifies it
    # against its boot identity after this runner returns.
    return fixture


@dataclass(frozen=True, slots=True)
class WindowsQualificationAssets:
    fixture: dict[str, Any]
    identities: dict[str, dict[str, Any]]
    collector: Path
    corpus: Path
    power: dict[str, Any]
    asr_config: dict[str, Any]
    asr_config_digest: str
    corpus_digest: str


def load_windows_assets(environ: dict[str, str] | None = None) -> WindowsQualificationAssets:
    """Validate every child-supplied Windows asset before any native work."""

    env = os.environ if environ is None else environ
    paths: dict[str, Path] = {}
    identities: dict[str, dict[str, Any]] = {}
    raws: dict[str, bytes] = {}
    for name in _ASSET_NAMES:
        prefix = "JARVIS_QUAL_WINDOWS_" + name.upper()
        value = env.get(prefix, "")
        path, identity, raw = _asset(
            Path(value),
            maximum=_ASSET_LIMITS[name],
            name=name,
            expected_sha256=env.get(prefix + "_SHA256", ""),
        )
        paths[name] = path
        identities[name] = identity
        raws[name] = raw
    power = _load_power(raws["power_profile"])
    fixture = _load_fixture(raws["fixture"], power)
    asr_config = _load_asr_config(raws["asr_config"])
    _safe_existing_chain(
        Path(asr_config["model_directory"]),
        final_directory=True,
        name="ASR model directory",
    )
    try:
        corpus = validate_corpus_manifest(
            paths["audio_corpus"], corpus_root=paths["audio_corpus"].parent
        )
    except WindowsAsrCorpusError as exc:
        raise FixtureFailure("audio corpus contract failed") from exc
    if len(corpus["entries"]) != 200:
        raise FixtureFailure("audio corpus count failed")
    # The qualifier passes manifest-relative asset identities.  The runner
    # cannot know those relative paths from the child environment, so it binds
    # the exact values at report construction after safely resolving the
    # manifest root below.
    return WindowsQualificationAssets(
        fixture=fixture,
        identities=identities,
        collector=paths["collector"],
        corpus=paths["audio_corpus"],
        power=power,
        asr_config=asr_config,
        asr_config_digest=hashlib.sha256(canonical_bytes(asr_config)).hexdigest(),
        corpus_digest=hashlib.sha256(raws["audio_corpus"]).hexdigest(),
    )


def _qualification_asset_paths(
    assets: WindowsQualificationAssets,
    env: dict[str, str],
) -> dict[str, Path]:
    return {
        "fixture": Path(env["JARVIS_QUAL_WINDOWS_FIXTURE"]),
        "collector": assets.collector,
        "audio_corpus": assets.corpus,
        "power_profile": Path(env["JARVIS_QUAL_WINDOWS_POWER_PROFILE"]),
        "asr_config": Path(env["JARVIS_QUAL_WINDOWS_ASR_CONFIG"]),
    }


def _revalidate_asset_identities(
    assets: WindowsQualificationAssets,
    env: dict[str, str],
) -> None:
    for name, path in _qualification_asset_paths(assets, env).items():
        raw = _safe_file(path, maximum=_ASSET_LIMITS[name], name=name)
        expected = assets.identities[name]
        if (
            len(raw) != expected["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]
        ):
            raise FixtureFailure("Windows qualification asset identity changed")


def _manifest_relative_identities(
    manifest_path: Path,
    assets: WindowsQualificationAssets,
    env: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if not manifest_path.is_absolute():
        raise FixtureFailure("Windows manifest path is unsafe")
    root = manifest_path.parent
    _safe_existing_chain(root, final_directory=True, name="manifest root")
    result: dict[str, dict[str, Any]] = {}
    paths = _qualification_asset_paths(assets, env)
    for name, path in paths.items():
        try:
            if not path.is_absolute() or any(
                component in {".", ".."} for component in path.parts
            ):
                raise ValueError
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise FixtureFailure("Windows qualification asset escapes manifest root") from exc
        _safe_existing_chain(path, final_directory=False, name=name)
        result[name] = {**assets.identities[name], "path": relative}
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise FixtureFailure("qualification output path failed")
    _safe_existing_chain(
        path.parent,
        final_directory=True,
        name="qualification output parent",
    )
    if path.exists() or path.is_symlink():
        raise FixtureFailure("qualification output path failed")
    raw = canonical_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".qualification-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, FixtureFailure):
            raise
        raise FixtureFailure("qualification output write failed") from exc


def _native_windows_directory() -> Path:  # pragma: no cover - native Windows only
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(32_768)
        length = ctypes.windll.kernel32.GetWindowsDirectoryW(buffer, len(buffer))
        if length == 0 or length >= len(buffer):
            raise OSError
        return Path(buffer.value)
    except BaseException as exc:
        raise FixtureFailure("trusted native tool path failed") from exc


def _native_program_files_x64() -> Path:  # pragma: no cover - native Windows only
    """Read FOLDERID_ProgramFilesX64 without trusting ProgramW6432."""

    try:
        import ctypes
        from ctypes import wintypes

        class Guid(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        folder = Guid(
            0x6D809377,
            0x6AF0,
            0x444B,
            (ctypes.c_ubyte * 8)(0x89, 0x57, 0xA3, 0x77, 0x3F, 0x02, 0x20, 0x0E),
        )
        pointer = ctypes.c_wchar_p()
        known_folder = ctypes.windll.shell32.SHGetKnownFolderPath
        known_folder.argtypes = [
            ctypes.POINTER(Guid),
            wintypes.DWORD,
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_wchar_p),
        ]
        known_folder.restype = ctypes.c_long
        result = known_folder(ctypes.byref(folder), 0, None, ctypes.byref(pointer))
        if result != 0 or not pointer.value:
            raise OSError
        try:
            return Path(pointer.value)
        finally:
            free = ctypes.windll.ole32.CoTaskMemFree
            free.argtypes = [ctypes.c_void_p]
            free.restype = None
            free(ctypes.cast(pointer, ctypes.c_void_p))
    except BaseException as exc:
        raise FixtureFailure("trusted native tool path failed") from exc


def _trusted_nvidia_smi(
    system32: Path,
    *,
    program_files_reader: Any = None,
) -> Path:
    system_candidate = system32 / "nvidia-smi.exe"
    if os.path.lexists(system_candidate):
        _safe_existing_chain(
            system_candidate,
            final_directory=False,
            name="nvidia-smi",
        )
        return system_candidate

    program_files = (program_files_reader or _native_program_files_x64)()
    if not program_files.is_absolute() or any(
        component in {".", ".."} for component in program_files.parts
    ):
        raise FixtureFailure("trusted native tool path failed")
    _safe_existing_chain(
        program_files,
        final_directory=True,
        name="ProgramFilesX64",
    )
    candidate = program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
    _safe_existing_chain(candidate, final_directory=False, name="nvidia-smi")
    return candidate


def _trusted_native_tools(
    windows_directory_reader: Any = None,
    program_files_reader: Any = None,
) -> tuple[Path, Path, Path]:
    """Return OS-derived probes without consulting PATH or environment roots."""

    reader = windows_directory_reader or _native_windows_directory
    system_root = reader()
    if not system_root.is_absolute() or any(
        component in {".", ".."} for component in system_root.parts
    ):
        raise FixtureFailure("trusted native tool path failed")
    _safe_existing_chain(system_root, final_directory=True, name="SystemRoot")
    system32 = system_root / "System32"
    _safe_existing_chain(system32, final_directory=True, name="System32")
    powercfg = system32 / "powercfg.exe"
    _safe_existing_chain(powercfg, final_directory=False, name="powercfg")
    nvidia_smi = _trusted_nvidia_smi(
        system32,
        program_files_reader=program_files_reader,
    )
    return system_root, nvidia_smi, powercfg


def _collector_environment(
    output: Path,
    assets: WindowsQualificationAssets,
    env: dict[str, str],
    *,
    private_temp: Path,
    windows_directory_reader: Any = None,
    program_files_reader: Any = None,
) -> dict[str, str]:
    system_root, nvidia_smi, powercfg = _trusted_native_tools(
        windows_directory_reader,
        program_files_reader,
    )
    result = private_temp_environment(private_temp)
    result["SYSTEMROOT"] = str(system_root)
    result["WINDIR"] = str(system_root)
    result.update(
        {
            "CYRUS_ASR_QUALIFICATION_OUTPUT": str(output),
            "CYRUS_ASR_QUALIFICATION_CORPUS": str(assets.corpus),
            "CYRUS_ASR_QUALIFICATION_FIXTURE": env["JARVIS_QUAL_WINDOWS_FIXTURE"],
            "CYRUS_ASR_QUALIFICATION_POWER_PROFILE": env["JARVIS_QUAL_WINDOWS_POWER_PROFILE"],
            "CYRUS_ASR_QUALIFICATION_ASR_CONFIG": env["JARVIS_QUAL_WINDOWS_ASR_CONFIG"],
            "CYRUS_ASR_QUALIFICATION_NVIDIA_SMI": str(nvidia_smi),
            "CYRUS_ASR_QUALIFICATION_POWERCFG": str(powercfg),
            "CYRUS_ASR_QUALIFICATION_BASE_URL": env["JARVIS_QUAL_BASE_URL"],
            "CYRUS_ASR_QUALIFICATION_API_KEY": env["JARVIS_QUAL_API_KEY"],
            "CYRUS_ASR_QUALIFICATION_FIXTURE_DIGEST": env["JARVIS_QUAL_FIXTURE_DIGEST"],
        }
    )
    return result


def _collector_gate(
    assets: WindowsQualificationAssets,
    manifest_path: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    with PrivateLaunchDirectory(
        manifest_path.parent, prefix="cyrus-asr-collector-"
    ) as launch:
        if launch.path is None:
            raise FixtureFailure("private ASR collector directory is unavailable")
        directory = launch.path
        output = directory / "collector.json"
        try:
            environment = _collector_environment(
                output,
                assets,
                env,
                private_temp=directory,
            )
            launch.mark_launch_started()
            if os.name == "nt":
                result = run_windows_job(
                    [str(assets.collector)],
                    environment=environment,
                    cwd=manifest_path.parent,
                    timeout=4 * 60 * 60,
                )
            else:
                result = subprocess.run(
                    [str(assets.collector)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                    cwd=manifest_path.parent,
                    timeout=4 * 60 * 60,
                    check=False,
                )
            launch.mark_tree_drained()
        except (OSError, subprocess.TimeoutExpired, WindowsProcessFailure) as exc:
            raise FixtureFailure("ASR collector failed") from exc
        if result.returncode != 0:
            raise FixtureFailure("ASR collector failed")
        _revalidate_asset_identities(assets, env)
        raw = _safe_file(output, maximum=65_536, name="ASR collector output")
    gate = _exact(
        parse_body(raw),
        {
            "asr_runtime_identity_digest",
            "audio_corpus_digest",
            "greek_utterances",
            "english_utterances",
            "greek_language_correct",
            "english_language_correct",
            "greek_reference_words",
            "greek_word_errors",
            "english_reference_words",
            "english_word_errors",
            "forced_cpu_fallbacks",
            "overlapping_utterances",
            "baseline_p95",
            "coexistence_p95",
            "baseline_llm_p95",
            "coexistence_llm_p95",
            "minimum_free_vram_mib",
            "lost_final_transcripts",
            "duplicate_final_transcripts",
            "stale_results",
            "oom",
            "crashes",
            "throttle_reasons",
        },
        "ASR collector output",
    )
    if _contains_content(gate):
        raise FixtureFailure("ASR collector output contains content")
    if (
        gate["asr_runtime_identity_digest"] != assets.asr_config_digest
        or gate["audio_corpus_digest"] != assets.corpus_digest
        or gate["greek_utterances"] != 100
        or gate["english_utterances"] != 100
        or any(
            type(gate[key]) is not int or gate[key] < 0
            for key in (
                "greek_language_correct",
                "english_language_correct",
                "greek_reference_words",
                "greek_word_errors",
                "english_reference_words",
                "english_word_errors",
                "overlapping_utterances",
                "minimum_free_vram_mib",
                "lost_final_transcripts",
                "duplicate_final_transcripts",
                "stale_results",
                "oom",
                "crashes",
                "throttle_reasons",
            )
        )
        or gate["greek_language_correct"] < 95
        or gate["english_language_correct"] < 95
        or gate["greek_reference_words"] <= 0
        or gate["english_reference_words"] <= 0
        or gate["greek_word_errors"] * 100 > gate["greek_reference_words"] * 20
        or gate["english_word_errors"] * 100 > gate["english_reference_words"] * 20
        or type(gate["forced_cpu_fallbacks"]) is not int
        or gate["forced_cpu_fallbacks"] != 5
        or gate["overlapping_utterances"] < 80
        or gate["minimum_free_vram_mib"] < 2048
        or any(
            gate[key] != 0
            for key in (
                "lost_final_transcripts",
                "duplicate_final_transcripts",
                "stale_results",
                "oom",
                "crashes",
                "throttle_reasons",
            )
        )
        or any(
            not _finite_metric(gate[key]) or gate[key] <= 0
            for key in (
                "baseline_p95",
                "coexistence_p95",
                "baseline_llm_p95",
                "coexistence_llm_p95",
            )
        )
        or gate["coexistence_p95"] > gate["baseline_p95"] * Decimal("1.15")
        or gate["coexistence_llm_p95"]
        > gate["baseline_llm_p95"] * Decimal("1.15")
    ):
        raise FixtureFailure("ASR collector report contract failed")
    power_digest = hashlib.sha256(canonical_bytes(assets.power)).hexdigest()
    return {
        "kind": "windows_asr_coexistence",
        "fixture": {**assets.fixture, "power_profile": assets.power},
        "assets": _manifest_relative_identities(manifest_path, assets, env),
        **gate,
        "baseline_power_identity_digest": power_digest,
        "coexistence_power_identity_digest": power_digest,
    }


def _generic_protocol_gates(client: QualificationClient, capabilities: dict[str, bool]) -> dict[str, Any]:
    _stage("streaming")
    greek_success = 0
    english_success = 0
    for _ in range(20):
        greek = client.completion("Απάντησε μόνο με τη λέξη ναι. /no_think", max_tokens=24, stream=True)
        english = client.completion("Reply using only the word yes. /no_think", max_tokens=24, stream=True)
        greek_success += int(any("\u0370" <= char <= "\u03ff" for char in greek.lower()))
        english_success += int(any("a" <= char.lower() <= "z" for char in english))
    if greek_success != 20 or english_success != 20:
        raise FixtureFailure("language streaming gate failed")

    _stage("tools")
    named_success = 0
    required_success = 0
    if capabilities["tools"]:
        for index in range(10):
            client.tool_call(f"Call record_index with index {index}. Return no prose.", index=index, named=True)
            named_success += 1
        for index in range(10, 20):
            client.tool_call(f"You must call record_index with index {index}. Return no prose.", index=index, named=False)
            required_success += 1

    _stage("json_schema")
    schema_success = 0
    if capabilities["json_schema"]:
        for index in range(20):
            language = "el" if index % 2 == 0 else "en"
            schema = {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "enum": [index]},
                    "language": {"type": "string", "enum": [language]},
                    "accepted": {"type": "boolean", "const": True},
                },
                "required": ["index", "language", "accepted"],
                "additionalProperties": False,
            }
            result = client.structured("Return the required qualification object exactly. Do not add prose.", schema)
            if result != {"index": index, "language": language, "accepted": True}:
                raise FixtureFailure("structured semantic fixture failed")
            schema_success += 1

    _stage("cancellation")
    for phase in ("prefill", "stream"):
        for _ in range(5):
            _phase_cancel(client, phase)
    _stage("crash_restart")
    for _ in range(5):
        response = client.inject("crash", {})
        if response.get("exit_ok") is not True:
            raise FixtureFailure("crash termination contract failed")
        client.wait_ready()

    _stage("isolation")
    previous_marker = ""
    pairs = 0
    for index in range(50):
        first = f"JARVIS_ISOLATION_A_{index:03d}"
        second = f"JARVIS_ISOLATION_B_{index:03d}"
        first_text = client.completion(f"Reply with OK only. Marker: {first}. /no_think", max_tokens=16)
        second_text = client.completion(f"Reply with OK only. Marker: {second}. /no_think", max_tokens=16)
        if previous_marker and previous_marker in first_text or first in second_text or previous_marker and previous_marker in second_text:
            raise FixtureFailure("semantic isolation leak")
        previous_marker = second
        pairs += 1
    return {
        "streaming": {"greek_success": greek_success, "english_success": english_success},
        "tools": {"named_success": named_success, "required_success": required_success},
        "json_schema": {"strict_success": schema_success},
        "cancellation": {"prefill_success": 5, "mid_stream_success": 5},
        "crash_restart": {"isolated_success": 5},
        "isolation": {"pairs_success": pairs, "leaks": 0},
    }


def run(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    started_wall = time.time_ns()
    started_mono = time.monotonic_ns()
    challenge = env.get("JARVIS_QUAL_CHALLENGE", "")
    identity_digest = env.get("JARVIS_QUAL_IDENTITY_DIGEST", "")
    fixture_digest = env.get("JARVIS_QUAL_FIXTURE_DIGEST", "")
    manifest_path = Path(env.get("JARVIS_QUAL_MANIFEST", ""))
    output = Path(env.get("JARVIS_QUAL_OUTPUT", ""))
    if (
        _HEX.fullmatch(challenge) is None
        or _HEX.fullmatch(identity_digest) is None
        or _HEX.fullmatch(fixture_digest) is None
        or not manifest_path.is_absolute()
        or not output.is_absolute()
        or not env.get("JARVIS_QUAL_API_KEY")
    ):
        raise FixtureFailure("qualification environment failed")
    assets = load_windows_assets(dict(env))
    model, identity, capabilities = _read_identity(manifest_path, challenge)
    client = QualificationClient(env.get("JARVIS_QUAL_BASE_URL", ""), env["JARVIS_QUAL_API_KEY"], model)
    client.wait_ready()
    generic = _generic_protocol_gates(client, capabilities)
    _stage("asr_coexistence")
    platform_gate = _collector_gate(assets, manifest_path, dict(env))
    completed_wall = time.time_ns()
    return {
        "suite_version": SUITE_VERSION,
        "fixture_digest": fixture_digest,
        "deterministic_seeds": [1],
        "identity": identity,
        "provenance": {
            "challenge": challenge,
            "identity_digest": identity_digest,
            "fixture_digest": fixture_digest,
            "started_unix_ns": str(started_wall),
            "completed_unix_ns": str(completed_wall),
            "monotonic_elapsed_ns": time.monotonic_ns() - started_mono,
        },
        **generic,
        "protocol_conformance": 1,
        "privacy_leaks": 0,
        "unexpected_oom": 0,
        "unexpected_crashes": 0,
        "unexpected_hangs": 0,
        "platform_gate": platform_gate,
    }


def main() -> int:
    try:
        evidence = run()
        _stage("evidence_write")
        _write_json(Path(os.environ["JARVIS_QUAL_OUTPUT"]), evidence)
        return 0
    except BaseException as exc:
        diagnostic = Path(os.environ.get("JARVIS_QUAL_DIAGNOSTIC", ""))
        if diagnostic.is_absolute() and not diagnostic.exists():
            try:
                _write_json(
                    diagnostic,
                    {
                        "stage": _current_stage,
                        "kind": "fixture_failure" if isinstance(exc, FixtureFailure) else "unexpected",
                    },
                )
            except BaseException:
                pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
