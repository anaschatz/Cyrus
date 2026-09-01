"""Strict, content-free inputs for the native Windows ASR collector.

The qualification executable receives no command-line arguments.  This module
validates its small child environment, every private asset path, the frozen
coexistence schedule, the production Voice ASR profile and all 200 WAV/reference
pairs before model or gateway work begins.  Path checks are lexical and
component by component so a junction cannot escape the sealed qualification
directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import wave
from array import array
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from ..backends import AudioChunk
from ..config import VoiceConfig, voice_config_from_mapping
from ..manifest import ModelManifest, model_manifest_for_profile


ENVIRONMENT_PREFIX = "CYRUS_ASR_QUALIFICATION_"
ENVIRONMENT_FIELDS = frozenset(
    {
        "OUTPUT",
        "CORPUS",
        "FIXTURE",
        "POWER_PROFILE",
        "ASR_CONFIG",
        "BASE_URL",
        "API_KEY",
        "FIXTURE_DIGEST",
        "NVIDIA_SMI",
        "POWERCFG",
    }
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
MAX_JSON_BYTES = 262_144
MAX_WAV_BYTES = 16 * 1024 * 1024
MAX_REFERENCE_BYTES = 4_096
EXPECTED_FIXTURE_ID = "cyrus-windows-asr-coexistence-v1"
EXPECTED_CORPUS_ID = "cyrus-windows-asr-audio-corpus-v1"
EXPECTED_CLEAN_CASE_COUNT = 200
EXPECTED_COEXISTENCE_CASE_COUNT = 100
EXPECTED_LEAD_IN_MS = 2_000
EXPECTED_POLL_INTERVAL_MS = 100
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class CollectorFailure(RuntimeError):
    """One fixed-token collector failure; paths and content are never exposed."""

    def __init__(self, code: str) -> None:
        if not re.fullmatch(r"[a-z0-9_]+", code):
            code = "collector_failure"
        super().__init__(code)
        self.code = code


class _DuplicateKey(ValueError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKey
        value[key] = item
    return value


def _reject_constant(_value: str) -> None:
    raise ValueError


def _parse_json(raw: bytes) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CollectorFailure("invalid_json")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_float=Decimal,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, _DuplicateKey) as exc:
        raise CollectorFailure("invalid_json") from exc
    if not isinstance(value, dict):
        raise CollectorFailure("invalid_json")
    return value


def _exact(value: Any, fields: set[str], code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CollectorFailure(code)
    return value


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _absolute_without_dot_segments(value: str, code: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CollectorFailure(code)
    path = Path(value)
    if (
        not path.is_absolute()
        or any(part in {".", ".."} for part in path.parts)
        or (os.name == "nt" and not _windows_drive_is_local(value))
    ):
        raise CollectorFailure(code)
    return path


def _is_local_windows_absolute(value: str) -> bool:
    """Reject UNC and device namespaces; qualification is local-drive only."""

    path = PureWindowsPath(value)
    return bool(
        path.is_absolute()
        and re.fullmatch(r"[A-Za-z]:", path.drive, re.ASCII)
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


def _validate_existing_chain(path: Path, *, final_directory: bool, code: str) -> None:
    """Inspect every existing component without resolving or following links."""

    if not path.is_absolute() or (
        os.name == "nt" and not _windows_drive_is_local(str(path))
    ):
        raise CollectorFailure(code)
    anchor = Path(path.anchor)
    current = anchor
    components = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(components):
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise CollectorFailure(code) from exc
        if current.is_symlink() or _is_reparse(metadata):
            raise CollectorFailure(code)
        is_final = index == len(components) - 1
        if not is_final and not stat.S_ISDIR(metadata.st_mode):
            raise CollectorFailure(code)
        if is_final:
            expected = stat.S_ISDIR if final_directory else stat.S_ISREG
            if not expected(metadata.st_mode):
                raise CollectorFailure(code)


def _native_system32() -> Path:  # pragma: no cover - native Windows only
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(32_768)
        length = ctypes.windll.kernel32.GetWindowsDirectoryW(buffer, len(buffer))
        if length == 0 or length >= len(buffer):
            raise OSError
        return Path(buffer.value) / "System32"
    except BaseException as exc:
        raise CollectorFailure("native_probe_path_invalid") from exc


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
        raise CollectorFailure("native_probe_path_invalid") from exc


def _nvidia_smi_candidates(system32: Path, program_files_x64: Path) -> tuple[Path, Path]:
    return (
        system32 / "nvidia-smi.exe",
        program_files_x64 / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
    )


def _safe_regular(path: Path, *, maximum: int, code: str) -> bytes:
    _validate_existing_chain(path, final_directory=False, code=code)
    try:
        before = path.stat(follow_symlinks=False)
        if before.st_size <= 0 or before.st_size > maximum:
            raise CollectorFailure(code)
        raw = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except CollectorFailure:
        raise
    except OSError as exc:
        raise CollectorFailure(code) from exc
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(raw) != before.st_size:
        raise CollectorFailure(code)
    return raw


def _canonical_json_bytes(value: Any) -> bytes:
    """The restricted JCS subset used by the integer-only ASR config."""

    def validate(item: Any) -> None:
        if item is None or isinstance(item, (bool, int, str)):
            if isinstance(item, int) and not isinstance(item, bool) and abs(item) > 2**53 - 1:
                raise CollectorFailure("asr_config_invalid")
            if isinstance(item, str):
                item.encode("utf-8")
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise CollectorFailure("asr_config_invalid")
            for child in item.values():
                validate(child)
            return
        raise CollectorFailure("asr_config_invalid")

    validate(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def model_pack_identity_digest(manifest: ModelManifest) -> str:
    identity = {
        "id": manifest.model_id,
        "revision": manifest.revision,
        "files": [
            {"name": item.name, "size": item.size, "sha256": item.sha256}
            for item in manifest.files
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()


@dataclass(frozen=True, slots=True)
class CollectorEnvironment:
    output: Path
    corpus: Path
    fixture: Path
    power_profile: Path
    asr_config: Path
    nvidia_smi: Path
    powercfg: Path
    base_url: str
    api_key: str
    fixture_digest: str


@dataclass(frozen=True, slots=True)
class FixtureContract:
    request_count: int
    clean_case_count: int
    maximum_completion_tokens: int
    prompt_workload_units: tuple[int, ...]
    minimum_overlaps: int
    inference_lead_in_ms: int
    minimum_language_correct_per_locale: int
    maximum_wer_percent_per_locale: int


@dataclass(frozen=True, slots=True)
class PowerContract:
    scheme_guid: str
    power_limit_watts: float
    driver_version: str
    compute_runtime: str
    gpu_clock_policy: str


@dataclass(frozen=True, slots=True)
class AsrContract:
    document: dict[str, Any]
    digest: str
    voice_config: VoiceConfig
    model_manifest: ModelManifest
    model_directory: Path
    device_index: int
    poll_interval_ms: int


@dataclass(frozen=True, slots=True)
class CollectorInputs:
    environment: CollectorEnvironment
    fixture: FixtureContract
    power: PowerContract
    asr: AsrContract
    corpus: dict[str, Any]
    corpus_digest: str
    raw_identities: tuple[tuple[Path, int, str], ...]


def load_environment(environ: Mapping[str, str]) -> CollectorEnvironment:
    observed = {
        key.removeprefix(ENVIRONMENT_PREFIX)
        for key in environ
        if key.startswith(ENVIRONMENT_PREFIX)
    }
    if observed != ENVIRONMENT_FIELDS:
        raise CollectorFailure("environment_invalid")
    values = {
        name: environ[ENVIRONMENT_PREFIX + name]
        for name in ENVIRONMENT_FIELDS
    }
    for name in (
        "CORPUS",
        "FIXTURE",
        "POWER_PROFILE",
        "ASR_CONFIG",
        "NVIDIA_SMI",
        "POWERCFG",
    ):
        path = _absolute_without_dot_segments(values[name], "asset_path_invalid")
        _validate_existing_chain(path, final_directory=False, code="asset_path_invalid")
    output = _absolute_without_dot_segments(values["OUTPUT"], "output_path_invalid")
    _validate_existing_chain(output.parent, final_directory=True, code="output_path_invalid")
    if output.exists() or output.is_symlink():
        raise CollectorFailure("output_path_invalid")
    parsed = urlparse(values["BASE_URL"])
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise CollectorFailure("gateway_invalid")
    key_bytes = values["API_KEY"].encode("utf-8")
    if not 32 <= len(key_bytes) <= 1_024:
        raise CollectorFailure("gateway_invalid")
    if HEX_SHA256.fullmatch(values["FIXTURE_DIGEST"]) is None:
        raise CollectorFailure("environment_invalid")
    paths = [
        Path(values[name])
        for name in (
            "CORPUS",
            "FIXTURE",
            "POWER_PROFILE",
            "ASR_CONFIG",
            "NVIDIA_SMI",
            "POWERCFG",
        )
    ]
    if len(set(paths + [output])) != 7:
        raise CollectorFailure("asset_path_invalid")
    if sys.platform == "win32" and os.name == "nt":
        system32 = _native_system32()
        supplied_nvidia_smi = Path(values["NVIDIA_SMI"])
        allowed_nvidia_smi = (system32 / "nvidia-smi.exe",)
        if supplied_nvidia_smi != allowed_nvidia_smi[0]:
            allowed_nvidia_smi = _nvidia_smi_candidates(
                system32,
                _native_program_files_x64(),
            )
        if (
            supplied_nvidia_smi not in allowed_nvidia_smi
            or Path(values["POWERCFG"]) != system32 / "powercfg.exe"
        ):
            raise CollectorFailure("native_probe_path_invalid")
    return CollectorEnvironment(
        output=output,
        corpus=Path(values["CORPUS"]),
        fixture=Path(values["FIXTURE"]),
        power_profile=Path(values["POWER_PROFILE"]),
        asr_config=Path(values["ASR_CONFIG"]),
        nvidia_smi=Path(values["NVIDIA_SMI"]),
        powercfg=Path(values["POWERCFG"]),
        base_url=values["BASE_URL"],
        api_key=values["API_KEY"],
        fixture_digest=values["FIXTURE_DIGEST"],
    )


def _load_fixture(document: dict[str, Any]) -> FixtureContract:
    value = _exact(
        document,
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
        "fixture_invalid",
    )
    corpus = _exact(
        value["audio_corpus"],
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
        "fixture_invalid",
    )
    if (
        value["fixture_schema_version"] != 1
        or value["fixture_id"] != EXPECTED_FIXTURE_ID
        or value["platform"] != "windows-x86_64"
        or value["prompt_workload_units"] != [512, 2048, 4096]
        or value["max_completion_tokens"] != 512
        or value["request_count"] != EXPECTED_COEXISTENCE_CASE_COUNT
        or value["concurrency"] != 1
        or value["schedule"] != "serial-continuous-from-audio-start"
        or value["minimum_overlapping_utterances"] != 80
        or value["sampling"] != {"temperature": 0, "top_p": 1}
        or corpus
        != {
            "schema": "cyrus.windows-asr-audio-corpus/1",
            "utterance_count": EXPECTED_CLEAN_CASE_COUNT,
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
        raise CollectorFailure("fixture_invalid")
    return FixtureContract(
        EXPECTED_COEXISTENCE_CASE_COUNT,
        EXPECTED_CLEAN_CASE_COUNT,
        512,
        (512, 2048, 4096),
        80,
        EXPECTED_LEAD_IN_MS,
        95,
        20,
    )


def _load_power(document: dict[str, Any]) -> PowerContract:
    value = _exact(
        document,
        {
            "windows_power_scheme_guid",
            "ac_state",
            "gpu_power_limit_watts",
            "gpu_clock_policy",
            "driver_version",
            "compute_runtime",
            "require_zero_thermal_power_throttle_reasons",
        },
        "power_profile_invalid",
    )
    watts = value["gpu_power_limit_watts"]
    if (
        not isinstance(value["windows_power_scheme_guid"], str)
        or re.fullmatch(
            r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
            value["windows_power_scheme_guid"],
        )
        is None
        or value["ac_state"] != "ac"
        or isinstance(watts, bool)
        or not isinstance(watts, (int, Decimal))
        or not math.isfinite(float(watts))
        or float(watts) <= 0
        or any(
            not isinstance(value[name], str) or not value[name]
            for name in ("gpu_clock_policy", "driver_version", "compute_runtime")
        )
        or re.fullmatch(r"CUDA [0-9]+\.[0-9]+", value["compute_runtime"]) is None
        or re.fullmatch(
            r"applications_graphics_mhz=[^;\r\n]{1,32};"
            r"applications_memory_mhz=[^;\r\n]{1,32}",
            value["gpu_clock_policy"],
        )
        is None
        or value["require_zero_thermal_power_throttle_reasons"] is not True
    ):
        raise CollectorFailure("power_profile_invalid")
    return PowerContract(
        scheme_guid=value["windows_power_scheme_guid"].casefold(),
        power_limit_watts=float(watts),
        driver_version=value["driver_version"],
        compute_runtime=value["compute_runtime"],
        gpu_clock_policy=value["gpu_clock_policy"],
    )


def _positive_int(value: Any, *, maximum: int | None = None) -> bool:
    return type(value) is int and value > 0 and (maximum is None or value <= maximum)


def _load_asr(document: dict[str, Any]) -> AsrContract:
    value = _exact(
        document,
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
        "asr_config_invalid",
    )
    pack = _exact(
        value["model_pack"],
        {"id", "revision", "artifacts_sha256"},
        "asr_config_invalid",
    )
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
        "asr_config_invalid",
    )
    if (
        value["schema_version"] != 1
        or value["backend"] != "faster-whisper"
        or value["model_profile"] not in {"base", "large-v3-turbo"}
        or value["execution_profile"] != "cuda_turn_scoped"
        or value["cpu_compute_type"] != "int8"
        or value["sample_rate_hz"] != 16000
        or value["channels"] != 1
        or value["sample_width_bytes"] != 2
        or value["language_mode"] != "multilingual-auto"
        or value["maximum_utterance_seconds"] != 30
        or type(cuda["device_index"]) is not int
        or cuda["device_index"] < 0
        or cuda["compute_type"] != "int8_float16"
        or not _positive_int(cuda["minimum_free_vram_mb"])
        or cuda["minimum_free_vram_mb"] < 2048
        or not _positive_int(cuda["reserve_for_other_workloads_mb"])
        or cuda["reserve_for_other_workloads_mb"] < 2048
        or cuda["fallback_to_cpu"] is not True
        or not _positive_int(cuda["worker_start_timeout_seconds"], maximum=600)
        or not _positive_int(cuda["transcription_timeout_seconds"], maximum=600)
        or cuda["warm_restart_limit"] != 0
        or cuda["vram_poll_interval_ms"] != EXPECTED_POLL_INTERVAL_MS
    ):
        raise CollectorFailure("asr_config_invalid")
    try:
        manifest = model_manifest_for_profile(value["model_profile"])
    except ValueError as exc:
        raise CollectorFailure("asr_config_invalid") from exc
    if (
        pack["id"] != manifest.model_id
        or pack["revision"] != manifest.revision
        or pack["artifacts_sha256"] != model_pack_identity_digest(manifest)
    ):
        raise CollectorFailure("asr_model_identity_invalid")
    model_directory = _absolute_without_dot_segments(
        value["model_directory"], "asr_model_path_invalid"
    )
    _validate_existing_chain(
        model_directory,
        final_directory=True,
        code="asr_model_path_invalid",
    )
    mapping = {
        "maximum_utterance_seconds": value["maximum_utterance_seconds"],
        "whisper_model_directory": str(model_directory),
        "asr": {
            "model_profile": value["model_profile"],
            "execution_profile": "cuda",
            "cpu_compute_type": value["cpu_compute_type"],
            "cuda": dict(cuda),
        },
    }
    try:
        voice_config = voice_config_from_mapping(mapping)
    except BaseException as exc:
        raise CollectorFailure("asr_config_invalid") from exc
    return AsrContract(
        document=value,
        digest=hashlib.sha256(_canonical_json_bytes(value)).hexdigest(),
        voice_config=voice_config,
        model_manifest=manifest,
        model_directory=model_directory,
        device_index=cuda["device_index"],
        poll_interval_ms=cuda["vram_poll_interval_ms"],
    )


def load_inputs(environ: Mapping[str, str]) -> CollectorInputs:
    environment = load_environment(environ)
    raws: dict[str, bytes] = {}
    for name, path, maximum in (
        ("fixture", environment.fixture, 65_536),
        ("power", environment.power_profile, 65_536),
        ("asr", environment.asr_config, 65_536),
        ("corpus", environment.corpus, MAX_JSON_BYTES),
    ):
        raws[name] = _safe_regular(path, maximum=maximum, code=f"{name}_asset_invalid")
    fixture = _load_fixture(_parse_json(raws["fixture"]))
    power = _load_power(_parse_json(raws["power"]))
    asr = _load_asr(_parse_json(raws["asr"]))
    try:
        from jarvis_inference.windows_asr_corpus import (
            WindowsAsrCorpusError,
            coexistence_entries,
            validate_corpus_manifest,
        )
    except ImportError as exc:
        raise CollectorFailure("corpus_asset_invalid") from exc
    try:
        corpus = validate_corpus_manifest(
            environment.corpus,
            corpus_root=environment.corpus.parent,
        )
        selected = coexistence_entries(corpus)
    except WindowsAsrCorpusError as exc:
        raise CollectorFailure("corpus_asset_invalid") from exc
    if (
        corpus.get("corpus_id") != EXPECTED_CORPUS_ID
        or corpus.get("fixture_id") != EXPECTED_FIXTURE_ID
        or len(corpus.get("entries", ())) != fixture.clean_case_count
        or corpus.get("schedule", {}).get("inference_lead_in_ms")
        != fixture.inference_lead_in_ms
        or len(selected) != fixture.request_count
    ):
        raise CollectorFailure("corpus_asset_invalid")
    identities = tuple(
        (path, len(raws[name]), hashlib.sha256(raws[name]).hexdigest())
        for name, path in (
            ("fixture", environment.fixture),
            ("power", environment.power_profile),
            ("asr", environment.asr_config),
            ("corpus", environment.corpus),
        )
    )
    return CollectorInputs(
        environment=environment,
        fixture=fixture,
        power=power,
        asr=asr,
        corpus=corpus,
        corpus_digest=hashlib.sha256(raws["corpus"]).hexdigest(),
        raw_identities=identities,
    )


def revalidate_inputs(inputs: CollectorInputs) -> None:
    for path, size, digest in inputs.raw_identities:
        raw = _safe_regular(path, maximum=MAX_JSON_BYTES, code="asset_identity_changed")
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise CollectorFailure("asset_identity_changed")
    try:
        from jarvis_inference.windows_asr_corpus import (
            WindowsAsrCorpusError,
            validate_corpus_manifest,
        )
    except ImportError as exc:
        raise CollectorFailure("asset_identity_changed") from exc
    try:
        validate_corpus_manifest(
            inputs.environment.corpus,
            corpus_root=inputs.environment.corpus.parent,
        )
    except WindowsAsrCorpusError as exc:
        raise CollectorFailure("asset_identity_changed") from exc


def load_audio(inputs: CollectorInputs, entry: Mapping[str, Any]) -> AudioChunk:
    relative = entry.get("relative_path")
    if not isinstance(relative, str):
        raise CollectorFailure("audio_invalid")
    # The corpus validator already performed the component-by-component path
    # check and exact hash.  Repeat the final identity immediately before use.
    path = inputs.environment.corpus.parent / Path(relative)
    raw = _safe_regular(path, maximum=MAX_WAV_BYTES, code="audio_invalid")
    if (
        len(raw) != entry.get("size_bytes")
        or hashlib.sha256(raw).hexdigest() != entry.get("sha256")
    ):
        raise CollectorFailure("audio_invalid")
    try:
        with wave.open(io.BytesIO(raw), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getframerate() != 16000
                or reader.getcomptype() != "NONE"
            ):
                raise CollectorFailure("audio_invalid")
            frames = reader.readframes(reader.getnframes())
    except (OSError, wave.Error) as exc:
        raise CollectorFailure("audio_invalid") from exc
    pcm = array("h")
    pcm.frombytes(frames)
    if sys.byteorder != "little":
        pcm.byteswap()
    samples = array("f", (sample / 32768.0 for sample in pcm))
    return AudioChunk(samples, 16000)


def load_reference_words(
    inputs: CollectorInputs,
    entry: Mapping[str, Any],
) -> tuple[str, ...]:
    """Load one private reference, verify identity, and return transient words."""

    relative = entry.get("reference_relative_path")
    if not isinstance(relative, str):
        raise CollectorFailure("reference_invalid")
    path = inputs.environment.corpus.parent / Path(relative)
    raw = _safe_regular(
        path,
        maximum=MAX_REFERENCE_BYTES,
        code="reference_invalid",
    )
    if (
        len(raw) != entry.get("reference_size_bytes")
        or hashlib.sha256(raw).hexdigest() != entry.get("reference_sha256")
        or raw.startswith(b"\xef\xbb\xbf")
    ):
        raise CollectorFailure("reference_invalid")
    try:
        from jarvis_inference.windows_asr_corpus import (
            WindowsAsrCorpusError,
            normalize_words,
        )
    except ImportError as exc:
        raise CollectorFailure("reference_invalid") from exc
    try:
        text = raw.decode("utf-8", errors="strict")
        words = normalize_words(text)
    except (UnicodeError, WindowsAsrCorpusError) as exc:
        raise CollectorFailure("reference_invalid") from exc
    if len(words) != entry.get("reference_word_count"):
        raise CollectorFailure("reference_invalid")
    return words


def normalize_hypothesis_words(text: str) -> tuple[str, ...]:
    """Normalize one transient ASR hypothesis without retaining its text."""

    try:
        from jarvis_inference.windows_asr_corpus import (
            WindowsAsrCorpusError,
            normalize_words,
        )
    except ImportError as exc:
        raise CollectorFailure("transcript_invalid") from exc
    try:
        words = normalize_words(text)
    except WindowsAsrCorpusError as exc:
        raise CollectorFailure("transcript_invalid") from exc
    if not 1 <= len(words) <= 1_024:
        raise CollectorFailure("transcript_invalid")
    return words


def select_coexistence_entries(inputs: CollectorInputs) -> list[dict[str, Any]]:
    """Return the centrally frozen exact 001..050-per-locale subset."""

    try:
        from jarvis_inference.windows_asr_corpus import (
            WindowsAsrCorpusError,
            coexistence_entries,
        )
    except ImportError as exc:
        raise CollectorFailure("corpus_asset_invalid") from exc
    try:
        return coexistence_entries(inputs.corpus)
    except WindowsAsrCorpusError as exc:
        raise CollectorFailure("corpus_asset_invalid") from exc


__all__ = [
    "AsrContract",
    "CollectorEnvironment",
    "CollectorFailure",
    "CollectorInputs",
    "FixtureContract",
    "PowerContract",
    "load_audio",
    "load_environment",
    "load_inputs",
    "load_reference_words",
    "normalize_hypothesis_words",
    "model_pack_identity_digest",
    "revalidate_inputs",
    "select_coexistence_entries",
]
