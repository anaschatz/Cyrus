"""Content-free native Windows release qualification evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from . import PROTOCOL_VERSION, __version__
from .app_paths import require_external_mutable_path
from .config import VoiceConfig
from .distribution import MANIFEST_NAME, verify_distribution
from .doctor import doctor_report_config
from .errors import BackendError

SAMPLE_SCHEMA_VERSION = 1
QUALIFICATION_SCHEMA_VERSION = 1

CASE_REQUIREMENTS: Mapping[str, int] = {
    "greek_turn": 10,
    "english_turn": 10,
    "no_speech": 3,
    "forced_release": 3,
    "barge_in_first_unit": 5,
    "barge_in_last_unit": 5,
    "keyboard_removal": 2,
    "suspend_resume": 2,
    "reconnect_dedup": 3,
}

TRANSCRIPT_LATENCY = "ptt_up_to_transcript_final_ms"
PURGE_LATENCY = "ptt_down_to_purge_dispatch_ms"
DISPATCH_LATENCY = "speech_commit_to_sapi_dispatch_ms"
PLAYBACK_LATENCY = "stream_complete_to_playback_completed_ms"
ALLOWED_METRICS = {TRANSCRIPT_LATENCY, PURGE_LATENCY, DISPATCH_LATENCY, PLAYBACK_LATENCY}
CASE_REQUIRED_METRICS: Mapping[str, frozenset[str]] = {
    "greek_turn": frozenset({TRANSCRIPT_LATENCY, DISPATCH_LATENCY, PLAYBACK_LATENCY}),
    "english_turn": frozenset({TRANSCRIPT_LATENCY, DISPATCH_LATENCY, PLAYBACK_LATENCY}),
    "barge_in_first_unit": frozenset({PURGE_LATENCY}),
    "barge_in_last_unit": frozenset({PURGE_LATENCY}),
}


class CompletedCommand(Protocol):
    returncode: int


def _blocked(code: str, **details: object) -> dict[str, object]:
    return {"status": "blocked", "code": code, **details}


def _ready(code: str, **details: object) -> dict[str, object]:
    return {"status": "ready", "code": code, **details}


def _atomic_json(path: str | Path, value: object) -> Path:
    output = require_external_mutable_path(path)
    if output.is_symlink():
        raise BackendError("qualification.output_symlink_forbidden", "Evidence output cannot be a symlink", subsystem="qualification")
    if not output.parent.is_dir():
        raise BackendError("qualification.output_directory_missing", "Evidence output directory does not exist", subsystem="qualification")
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".jarvis-evidence-", dir=output.parent, delete=False) as stream:
            temporary = stream.name
            os.chmod(temporary, 0o600)
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        Path(temporary).replace(output)
        return output
    except (OSError, ValueError) as exc:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise BackendError("qualification.output_write_failed", "Evidence output could not be written", subsystem="qualification") from exc


def write_report(path: str | Path, value: object) -> Path:
    return _atomic_json(path, value)


def sample_template() -> dict[str, object]:
    return {"schema_version": SAMPLE_SCHEMA_VERSION, "samples": []}


def write_sample_template(path: str | Path) -> Path:
    return _atomic_json(path, sample_template())


def _load_json(path: str | Path, code: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError(code, "Qualification JSON is missing or invalid", subsystem="qualification") from exc


def _canonical_uuid4(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return None
    return value if parsed.version == 4 and str(parsed) == value else None


def _validate_metrics(raw: Any, *, required: frozenset[str]) -> dict[str, float]:
    if not isinstance(raw, Mapping) or set(raw) - ALLOWED_METRICS:
        raise BackendError("qualification.samples_schema_invalid", "Sample metrics contain unknown fields", subsystem="qualification")
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise BackendError("qualification.samples_schema_invalid", "Sample metric must be numeric", subsystem="qualification")
        number = float(value)
        if not math.isfinite(number) or number < 0 or number > 600_000:
            raise BackendError("qualification.samples_schema_invalid", "Sample metric is outside the permitted range", subsystem="qualification")
        metrics[str(name)] = number
    if not required.issubset(metrics):
        raise BackendError("qualification.required_metric_missing", "A passing sample is missing a required metric", subsystem="qualification")
    return metrics


def validate_samples(raw: Any) -> list[dict[str, object]]:
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "samples"} or raw.get("schema_version") != SAMPLE_SCHEMA_VERSION or not isinstance(raw.get("samples"), list):
        raise BackendError("qualification.samples_schema_invalid", "Qualification samples have an invalid schema", subsystem="qualification")
    validated: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for item in raw["samples"]:
        if not isinstance(item, Mapping) or set(item) != {"sample_id", "case", "outcome", "metrics"}:
            raise BackendError("qualification.samples_schema_invalid", "Qualification sample has unknown or missing fields", subsystem="qualification")
        sample_id = _canonical_uuid4(item.get("sample_id"))
        case, outcome = item.get("case"), item.get("outcome")
        if sample_id is None or sample_id in identifiers:
            raise BackendError("qualification.sample_id_invalid", "Qualification sample ID is invalid or replayed", subsystem="qualification")
        if case not in CASE_REQUIREMENTS or outcome not in {"pass", "fail"}:
            raise BackendError("qualification.samples_schema_invalid", "Qualification case or outcome is invalid", subsystem="qualification")
        required = CASE_REQUIRED_METRICS.get(str(case), frozenset()) if outcome == "pass" else frozenset()
        metrics = _validate_metrics(item.get("metrics"), required=required)
        identifiers.add(sample_id)
        validated.append({"sample_id": sample_id, "case": case, "outcome": outcome, "metrics": metrics})
    return validated


def load_samples(path: str | Path) -> list[dict[str, object]]:
    return validate_samples(_load_json(path, "qualification.samples_unreadable"))


def parse_metrics(values: Sequence[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw in values:
        name, separator, value = raw.partition("=")
        if not separator or name in metrics or name not in ALLOWED_METRICS:
            raise BackendError("qualification.metric_argument_invalid", "Metric argument must be a unique allowed name=value pair", subsystem="qualification")
        try:
            number = float(value)
        except ValueError as exc:
            raise BackendError("qualification.metric_argument_invalid", "Metric argument must be numeric", subsystem="qualification") from exc
        metrics[name] = number
    return _validate_metrics(metrics, required=frozenset())


def record_sample(path: str | Path, *, case: str, outcome: str, metrics: Mapping[str, float], id_factory: Callable[[], uuid.UUID] = uuid.uuid4) -> str:
    raw = _load_json(path, "qualification.samples_unreadable")
    samples = validate_samples(raw)
    sample_id = str(id_factory())
    candidate = {"sample_id": sample_id, "case": case, "outcome": outcome, "metrics": dict(metrics)}
    validate_samples({"schema_version": SAMPLE_SCHEMA_VERSION, "samples": [*samples, candidate]})
    _atomic_json(path, {"schema_version": SAMPLE_SCHEMA_VERSION, "samples": [*samples, candidate]})
    return sample_id


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position); upper = math.ceil(position)
    result = ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(result, 3)


def evaluate_samples(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cases: dict[str, dict[str, int]] = {name: {"required": required, "passed": 0, "failed": 0} for name, required in CASE_REQUIREMENTS.items()}
    metric_values: dict[str, list[float]] = {name: [] for name in sorted(ALLOWED_METRICS)}
    for sample in samples:
        case = str(sample["case"]); outcome = str(sample["outcome"])
        cases[case]["passed" if outcome == "pass" else "failed"] += 1
        for name, value in dict(sample["metrics"]).items():
            metric_values[name].append(float(value))
    failures = sum(item["failed"] for item in cases.values())
    enough = all(item["passed"] >= item["required"] for item in cases.values())
    metrics = {
        name: {"count": len(values), "p50_ms": _percentile(values, 0.50), "p95_ms": _percentile(values, 0.95)}
        for name, values in metric_values.items() if values
    }
    ready = failures == 0 and enough
    return {
        "status": "ready" if ready else "blocked",
        "code": "qualification.samples_sufficient" if ready else ("qualification.failed_sample_present" if failures else "qualification.samples_insufficient"),
        "samples_total": len(samples),
        "failed_total": failures,
        "cases": cases,
        "metrics": metrics,
        "content_captured": False,
    }


def host_qualification(*, platform_name: str | None = None, build: int | None = None, pointer_bits: int | None = None, python_version: tuple[int, int] | None = None) -> dict[str, object]:
    selected_platform = sys.platform if platform_name is None else platform_name
    selected_bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    selected_python = (sys.version_info.major, sys.version_info.minor) if python_version is None else python_version
    if build is None and selected_platform == "win32":  # pragma: no cover - native Windows only
        build = sys.getwindowsversion().build
    ready = selected_platform == "win32" and (build or 0) >= 22_000 and selected_bits == 64 and selected_python == (3, 11)
    return {
        "status": "ready" if ready else "blocked",
        "code": "qualification.windows_11_x64_python_311" if ready else "qualification.native_windows_11_x64_python_311_required",
        "windows_build": build if selected_platform == "win32" else None,
        "pointer_bits": selected_bits,
        "python": f"{selected_python[0]}.{selected_python[1]}",
    }


def verify_authenticode(
    directory: str | Path,
    *,
    platform_name: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., CompletedCommand] = subprocess.run,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    selected_platform = sys.platform if platform_name is None else platform_name
    if selected_platform != "win32":
        return _blocked("qualification.authenticode_native_windows_required", files_checked=0, failed_files=0)
    tool = which("signtool")
    if tool is None:
        return _blocked("qualification.signtool_missing", files_checked=0, failed_files=0)
    root = Path(directory)
    artifacts = sorted((path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".exe", ".dll", ".pyd"}), key=lambda path: path.relative_to(root).as_posix())
    if not artifacts:
        return _blocked("qualification.no_pe_artifacts", files_checked=0, failed_files=0)
    checked = 0; failed = 0; warnings = 0
    for artifact in artifacts:
        try:
            result = runner([tool, "verify", "/pa", "/all", "/tw", str(artifact)], capture_output=True, text=False, timeout=timeout_seconds, check=False, shell=False)
        except subprocess.TimeoutExpired:
            return _blocked("qualification.signtool_timeout", files_checked=checked, failed_files=failed + 1)
        except OSError:
            return _blocked("qualification.signtool_failed", files_checked=checked, failed_files=failed + 1)
        checked += 1
        if result.returncode == 2:
            warnings += 1; failed += 1
        elif result.returncode != 0:
            failed += 1
    if failed:
        return _blocked("qualification.authenticode_invalid", files_checked=checked, failed_files=failed, warning_files=warnings)
    return _ready("qualification.authenticode_valid", files_checked=checked, failed_files=0, warning_files=0)


def evaluate_frozen_doctor(raw: Any) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        return _blocked("qualification.frozen_doctor_invalid")
    checks = raw.get("checks")
    if not isinstance(checks, Mapping):
        return _blocked("qualification.frozen_doctor_invalid")
    packaging = checks.get("packaging"); tray = checks.get("tray"); host = checks.get("host")
    valid = (
        raw.get("ready") is True
        and raw.get("network_used") is False
        and raw.get("product_version") == __version__
        and raw.get("protocol_version") == PROTOCOL_VERSION
        and isinstance(packaging, Mapping)
        and packaging.get("status") == "ready"
        and packaging.get("frozen") is True
        and packaging.get("bundle_mode") == "onedir"
        and isinstance(packaging.get("dll_search"), Mapping)
        and packaging["dll_search"].get("status") == "ready"
        and isinstance(packaging.get("mutable_data"), Mapping)
        and packaging["mutable_data"].get("status") == "ready"
        and isinstance(tray, Mapping) and tray.get("status") == "ready"
        and isinstance(host, Mapping) and host.get("status") == "ready"
    )
    return _ready("qualification.frozen_doctor_ready", product_version=__version__, protocol_version=PROTOCOL_VERSION) if valid else _blocked("qualification.frozen_doctor_not_ready")


def _manifest_gate(directory: str | Path) -> dict[str, object]:
    try:
        result = verify_distribution(directory)
        if not result.ok:
            return _blocked(result.code, files_checked=result.files_checked)
        digest = hashlib.sha256((Path(directory) / MANIFEST_NAME).read_bytes()).hexdigest()
        return _ready("qualification.distribution_valid", files_checked=result.files_checked, manifest_sha256=digest)
    except (BackendError, OSError) as exc:
        code = exc.code if isinstance(exc, BackendError) else "qualification.distribution_unreadable"
        return _blocked(code, files_checked=0)


def build_release_report(
    config: VoiceConfig,
    *,
    distribution: str | Path,
    samples_path: str | Path,
    frozen_doctor_path: str | Path,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    host_probe: Callable[[], dict[str, object]] = host_qualification,
    doctor_runner: Callable[[VoiceConfig], dict[str, object]] = doctor_report_config,
    signature_verifier: Callable[[str | Path], dict[str, object]] = verify_authenticode,
) -> dict[str, object]:
    host = host_probe()
    try:
        source_doctor_raw = doctor_runner(config)
        source_doctor = _ready("qualification.source_doctor_ready") if source_doctor_raw.get("ready") is True and source_doctor_raw.get("network_used") is False else _blocked("qualification.source_doctor_not_ready")
    except Exception:
        source_doctor = _blocked("qualification.source_doctor_failed")
    distribution_gate = _manifest_gate(distribution)
    signatures = signature_verifier(distribution) if distribution_gate["status"] == "ready" else _blocked("qualification.distribution_prerequisite_failed", files_checked=0, failed_files=0)
    try:
        frozen = evaluate_frozen_doctor(_load_json(frozen_doctor_path, "qualification.frozen_doctor_unreadable"))
    except BackendError as exc:
        frozen = _blocked(exc.code)
    try:
        samples = evaluate_samples(load_samples(samples_path))
    except BackendError as exc:
        samples = _blocked(exc.code, samples_total=0, failed_total=0, cases={}, metrics={}, content_captured=False)
    checks = {"host": host, "source_doctor": source_doctor, "distribution": distribution_gate, "authenticode": signatures, "frozen_doctor": frozen, "samples": samples}
    ready = all(check.get("status") == "ready" for check in checks.values())
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "product": "jarvis-voice",
        "product_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": now().astimezone(UTC).isoformat(timespec="seconds"),
        "ready": ready,
        "code": "qualification.release_ready" if ready else "qualification.release_blocked",
        "checks": checks,
        "network_used": False,
        "content_captured": False,
    }
