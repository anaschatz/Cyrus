"""Offline-first qualification harness for a future Chatterbox V3 backend.

This module is deliberately separate from the production ``TtsBackend``.  It
never installs packages or downloads model files.  Acquisition must happen in
an isolated environment after :func:`chatterbox_pilot_preflight` reports that
the host has enough disk headroom.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import resource
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

from ..errors import BackendError


CHATTERBOX_PACKAGE_VERSION = "0.1.7"
MODEL_REPO_ID = "ResembleAI/chatterbox"
MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
T3_MODEL = "v3"

GIB = 1024 * 1024 * 1024
MINIMUM_ACQUISITION_FREE_BYTES = 8 * GIB
MINIMUM_RUN_FREE_BYTES = 2 * GIB


class ArtifactSpec(NamedTuple):
    filename: str
    size: int
    sha256: str


# Pinned binary artifacts used by ChatterboxMultilingualTTS.from_local().
# The small tokenizer JSON is revision-pinned and structurally checked below.
MODEL_ARTIFACTS = (
    ArtifactSpec(
        "t3_mtl23ls_v3.safetensors",
        2_143_989_928,
        "5abca8321ede76f8e61f1cc0d19aea6c946b28871017ce8726f8a69203f05953",
    ),
    ArtifactSpec(
        "s3gen.pt",
        1_057_165_844,
        "9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3",
    ),
    ArtifactSpec(
        "ve.pt",
        5_698_626,
        "4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1",
    ),
    ArtifactSpec(
        "conds.pt",
        107_374,
        "6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e",
    ),
)
TOKENIZER_FILENAME = "grapheme_mtl_merged_expanded_v1.json"
MODEL_DOWNLOAD_BYTES = sum(item.size for item in MODEL_ARTIFACTS)

PILOT_FIXTURES = (
    ("greek_greeting", "Καλημέρα. Είμαι ο Τζάρβις και είμαι έτοιμος να βοηθήσω."),
    ("greek_status", "Η σύνδεση ολοκληρώθηκε με επιτυχία."),
    ("greek_barge_in", "Θα σταματήσω αμέσως μόλις αρχίσεις να μιλάς."),
)


@dataclass(frozen=True, slots=True)
class ModelValidation:
    ready: bool
    code: str
    files_present: int
    files_required: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _validate_tokenizer(path: Path) -> bool:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    added = raw.get("added_tokens") if isinstance(raw, dict) else None
    if not isinstance(added, list):
        return False
    return any(isinstance(item, dict) and item.get("content") == "[el]" for item in added)


def validate_chatterbox_model_directory(model_directory: str | Path) -> ModelValidation:
    directory = Path(model_directory).expanduser()
    required = len(MODEL_ARTIFACTS) + 1
    if not directory.is_absolute() or not directory.is_dir() or directory.is_symlink():
        return ModelValidation(False, "tts.chatterbox.model_directory_missing", 0, required)
    present = 0
    for artifact in MODEL_ARTIFACTS:
        path = directory / artifact.filename
        try:
            if path.is_symlink() or path.stat().st_size != artifact.size or _sha256(path) != artifact.sha256:
                return ModelValidation(False, "tts.chatterbox.model_artifact_invalid", present, required)
        except OSError:
            return ModelValidation(False, "tts.chatterbox.model_artifact_missing", present, required)
        present += 1
    if not _validate_tokenizer(directory / TOKENIZER_FILENAME):
        return ModelValidation(False, "tts.chatterbox.tokenizer_invalid", present, required)
    return ModelValidation(True, "tts.chatterbox.model_ready", required, required)


def chatterbox_pilot_preflight(
    storage_root: str | Path,
    model_directory: str | Path,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    python_version: tuple[int, int, int] | None = None,
    pointer_bits: int | None = None,
    disk_usage: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
    module_probe: Callable[[str], bool] = _module_available,
) -> dict[str, object]:
    root = Path(storage_root).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise BackendError(
            "tts.chatterbox.storage_root_invalid",
            "Chatterbox pilot storage root is invalid",
            subsystem="tts",
        )
    selected_platform = sys.platform if platform_name is None else platform_name
    selected_machine = platform.machine() if machine is None else machine
    selected_version = sys.version_info[:3] if python_version is None else python_version
    selected_bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    try:
        available = int(getattr(disk_usage(root), "free"))
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        raise BackendError(
            "tts.chatterbox.disk_probe_failed",
            "Chatterbox pilot disk probe failed",
            subsystem="tts",
        ) from exc

    host_ready = (
        selected_platform == "darwin"
        and selected_machine == "arm64"
        and selected_version[:2] == (3, 11)
        and selected_bits == 64
    )
    dependencies = {
        "torch": bool(module_probe("torch")),
        "torchaudio": bool(module_probe("torchaudio")),
        "chatterbox": bool(module_probe("chatterbox")),
    }
    model = validate_chatterbox_model_directory(model_directory)
    acquisition_ready = host_ready and available >= MINIMUM_ACQUISITION_FREE_BYTES
    run_ready = host_ready and available >= MINIMUM_RUN_FREE_BYTES and all(dependencies.values()) and model.ready

    if not host_ready:
        code = "tts.chatterbox.host_unsupported"
    elif available < MINIMUM_RUN_FREE_BYTES:
        code = "tts.chatterbox.disk_space_critical"
    elif model.ready and all(dependencies.values()):
        code = "tts.chatterbox.pilot_ready"
    elif available < MINIMUM_ACQUISITION_FREE_BYTES:
        code = "tts.chatterbox.disk_space_insufficient"
    elif not all(dependencies.values()):
        code = "tts.chatterbox.dependencies_missing"
    else:
        code = model.code

    return {
        "schema_version": 1,
        "ready": run_ready,
        "code": code,
        "acquisition_ready": acquisition_ready,
        "host": {
            "platform": selected_platform,
            "machine": selected_machine,
            "python": ".".join(str(item) for item in selected_version),
            "bits": selected_bits,
        },
        "disk": {
            "available_mib": available // (1024 * 1024),
            "acquisition_required_mib": MINIMUM_ACQUISITION_FREE_BYTES // (1024 * 1024),
            "run_required_mib": MINIMUM_RUN_FREE_BYTES // (1024 * 1024),
            "model_download_mib": MODEL_DOWNLOAD_BYTES // (1024 * 1024),
        },
        "dependencies": dependencies,
        "model": {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "t3_model": T3_MODEL,
            "ready": model.ready,
            "code": model.code,
            "files_present": model.files_present,
            "files_required": model.files_required,
        },
        "network_used": False,
        "installation_performed": False,
        "runtime_started": False,
        "content_captured": False,
    }


def _synchronize_mps(torch_module: object) -> None:
    mps = getattr(torch_module, "mps", None)
    synchronize = getattr(mps, "synchronize", None)
    if callable(synchronize):
        synchronize()


def run_chatterbox_pilot(model_directory: str | Path, output_directory: str | Path) -> dict[str, object]:
    model_path = Path(model_directory).expanduser()
    validation = validate_chatterbox_model_directory(model_path)
    if not validation.ready:
        raise BackendError(validation.code, "Chatterbox model directory is not ready", subsystem="tts")
    output = Path(output_directory).expanduser()
    if not output.is_absolute() or output.is_symlink():
        raise BackendError("tts.chatterbox.output_invalid", "Chatterbox pilot output is invalid", subsystem="tts")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)

    try:
        import torch
        import torchaudio
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as exc:  # pragma: no cover - exercised only in isolated pilot env
        raise BackendError("tts.chatterbox.dependencies_missing", "Chatterbox dependencies are unavailable", subsystem="tts") from exc
    if not torch.backends.mps.is_available():
        raise BackendError("tts.chatterbox.mps_unavailable", "Chatterbox MPS device is unavailable", subsystem="tts")

    torch.manual_seed(0)
    load_started = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_local(model_path, device="mps", t3_model=T3_MODEL)
    _synchronize_mps(torch)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    samples: list[dict[str, object]] = []
    for index, (fixture_id, text) in enumerate(PILOT_FIXTURES, start=1):
        torch.manual_seed(10_000 + index)
        _synchronize_mps(torch)
        started = time.perf_counter()
        waveform = model.generate(text, language_id="el")
        _synchronize_mps(torch)
        generation_ms = (time.perf_counter() - started) * 1000.0
        sample_count = int(waveform.shape[-1])
        duration_ms = sample_count * 1000.0 / float(model.sr)
        filename = f"{index:02d}-{fixture_id}.wav"
        destination = output / filename
        torchaudio.save(str(destination), waveform.detach().cpu(), model.sr)
        os.chmod(destination, 0o600)
        samples.append(
            {
                "fixture_id": fixture_id,
                "file": filename,
                "generation_ms": round(generation_ms, 3),
                "audio_duration_ms": round(duration_ms, 3),
                "realtime_factor": round(generation_ms / duration_ms, 4) if duration_ms else None,
            }
        )

    return {
        "schema_version": 1,
        "ready": True,
        "code": "tts.chatterbox.pilot_completed",
        "backend": "chatterbox_multilingual_v3",
        "device": "mps",
        "language": "el",
        "model": {"repo_id": MODEL_REPO_ID, "revision": MODEL_REVISION, "t3_model": T3_MODEL},
        "model_load_ms": round(load_ms, 3),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "samples": samples,
        "network_used": False,
        "content_captured": False,
    }
