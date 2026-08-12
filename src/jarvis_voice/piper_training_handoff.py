"""Deterministic, offline handoff for STOMA F1 Piper training inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .errors import BackendError
from .macos.compact_greek_tts_pilot import (
    MIB,
    PIPER_CHECKPOINT_ARTIFACTS,
    PIPER_CHECKPOINT_DIRECTORY,
    PIPER_REVISION,
    PIPER_TARGET_SAMPLE_RATE,
    PIPER_VERSION,
    STOMA_GITHUB_TREE,
    STOMA_REVISION,
    STOMA_SPEAKER,
    STOMA_SPEAKER_CLIPS,
)


HANDOFF_SCHEMA = "jarvis.voice.piper-f1-wsl-handoff/1"
BUNDLE_SCHEMA = "jarvis.voice.stoma-f1-training-bundle/1"
MAX_BUNDLE_MANIFEST_BYTES = 512 * 1024
MAX_HANDOFF_MANIFEST_BYTES = 1024 * 1024


def _error(code: str, message: str) -> BackendError:
    return BackendError(code, message, subsystem="tts")


def _directory(value: str | Path, *, code: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise _error(code, "Piper F1 handoff directory is invalid")
    return path.resolve()


def _regular_file(path: Path, *, code: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise _error(code, "Piper F1 handoff file is invalid")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_json(path: Path, *, maximum_bytes: int, code: str) -> Mapping[str, Any]:
    _regular_file(path, code=code)
    try:
        size = path.stat().st_size
        if size < 2 or size > maximum_bytes:
            raise _error(code, "Piper F1 manifest size is invalid")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _error(code, "Piper F1 manifest is invalid") from exc
    if not isinstance(value, Mapping):
        raise _error(code, "Piper F1 manifest must be an object")
    return value


def _safe_relative(value: object) -> str:
    if not isinstance(value, str):
        raise _error("tts.f1_handoff.path_invalid", "Handoff path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise _error("tts.f1_handoff.path_invalid", "Handoff path is unsafe")
    return path.as_posix()


def _record(path: Path, relative_path: str) -> dict[str, object]:
    _regular_file(path, code="tts.f1_handoff.file_missing")
    return {
        "path": _safe_relative(relative_path),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validated_bundle(data: Path) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    bundle_path = data / "bundle-manifest.json"
    bundle = _read_json(
        bundle_path,
        maximum_bytes=MAX_BUNDLE_MANIFEST_BYTES,
        code="tts.f1_handoff.bundle_invalid",
    )
    required = {
        "schema": BUNDLE_SCHEMA,
        "dataset_revision": STOMA_REVISION,
        "recipe_tree": STOMA_GITHUB_TREE,
        "speaker": STOMA_SPEAKER,
        "rows": STOMA_SPEAKER_CLIPS,
        "piper_sample_rate": PIPER_TARGET_SAMPLE_RATE,
        "network_used": False,
    }
    if any(bundle.get(name) != expected for name, expected in required.items()):
        raise _error("tts.f1_handoff.bundle_incompatible", "F1 training bundle is incompatible")
    metadata_digest = bundle.get("metadata_sha256")
    audio_digests = bundle.get("audio_sha256")
    if (
        not _digest(metadata_digest)
        or not isinstance(audio_digests, Mapping)
        or len(audio_digests) != STOMA_SPEAKER_CLIPS
    ):
        raise _error("tts.f1_handoff.bundle_incompatible", "F1 training bundle hashes are invalid")

    names: list[str] = []
    for raw_name, digest in audio_digests.items():
        if (
            not isinstance(raw_name, str)
            or PurePosixPath(raw_name).name != raw_name
            or not raw_name.endswith(".wav")
            or not _digest(digest)
        ):
            raise _error("tts.f1_handoff.bundle_incompatible", "F1 audio manifest is invalid")
        names.append(raw_name)
    if len(set(names)) != len(names):
        raise _error("tts.f1_handoff.bundle_incompatible", "F1 audio names are not unique")
    return bundle, tuple(sorted(names))


def build_piper_f1_handoff(
    data_directory: str | Path,
    artifact_root: str | Path,
) -> dict[str, object]:
    """Hash every transfer input and return a deterministic handoff manifest."""

    data = _directory(data_directory, code="tts.f1_handoff.data_directory_invalid")
    artifacts = _directory(artifact_root, code="tts.f1_handoff.artifact_root_invalid")
    bundle, audio_names = _validated_bundle(data)
    metadata = _regular_file(data / "metadata.csv", code="tts.f1_handoff.metadata_missing")
    if _sha256(metadata) != bundle["metadata_sha256"]:
        raise _error("tts.f1_handoff.metadata_hash_mismatch", "F1 metadata hash mismatch")

    audio_root = _directory(data / "audio", code="tts.f1_handoff.audio_directory_invalid")
    present = {entry.name for entry in audio_root.iterdir()}
    if present != set(audio_names):
        raise _error("tts.f1_handoff.audio_set_mismatch", "F1 audio file set is incomplete")

    files = [
        _record(data / "bundle-manifest.json", "f1-training-data/bundle-manifest.json"),
        _record(metadata, "f1-training-data/metadata.csv"),
    ]
    audio_digests = bundle["audio_sha256"]
    for name in audio_names:
        item = _record(audio_root / name, f"f1-training-data/audio/{name}")
        if item["sha256"] != audio_digests[name]:
            raise _error("tts.f1_handoff.audio_hash_mismatch", "F1 audio hash mismatch")
        files.append(item)

    checkpoint_root = artifacts / "checkpoint" / PIPER_CHECKPOINT_DIRECTORY
    _directory(checkpoint_root, code="tts.f1_handoff.checkpoint_directory_invalid")
    for spec in sorted(PIPER_CHECKPOINT_ARTIFACTS, key=lambda value: value.relative_path):
        path = artifacts / "checkpoint" / spec.relative_path
        item = _record(path, f"checkpoint/{spec.relative_path}")
        if item["size"] != spec.size or item["sha256"] != spec.sha256:
            raise _error("tts.f1_handoff.checkpoint_hash_mismatch", "Piper warm-start hash mismatch")
        files.append(item)

    files.sort(key=lambda item: str(item["path"]))
    return {
        "schema": HANDOFF_SCHEMA,
        "candidate": "stoma-f1-piper-medium",
        "dataset": {
            "revision": STOMA_REVISION,
            "recipe_tree": STOMA_GITHUB_TREE,
            "speaker": STOMA_SPEAKER,
            "clips": STOMA_SPEAKER_CLIPS,
            "sample_rate": PIPER_TARGET_SAMPLE_RATE,
        },
        "piper": {"version": PIPER_VERSION, "revision": PIPER_REVISION},
        "files": files,
        "totals": {
            "files": len(files),
            "bytes": sum(int(item["size"]) for item in files),
        },
        "network_used": False,
        "installation_performed": False,
        "training_started": False,
    }


def write_piper_f1_handoff(path: str | Path, manifest: Mapping[str, object]) -> None:
    destination = Path(path).expanduser()
    if not destination.is_absolute() or not destination.parent.is_dir() or destination.exists():
        raise _error("tts.f1_handoff.output_invalid", "Handoff output must be a new absolute file")
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(raw) > MAX_HANDOFF_MANIFEST_BYTES:
        raise _error("tts.f1_handoff.output_too_large", "Handoff manifest is too large")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".piper-f1-handoff-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_piper_f1_handoff(transfer_root: str | Path, manifest_path: str | Path) -> dict[str, object]:
    """Verify a copied handoff on WSL without installing or starting training."""

    root = _directory(transfer_root, code="tts.f1_handoff.transfer_root_invalid")
    manifest = _read_json(
        Path(manifest_path).expanduser(),
        maximum_bytes=MAX_HANDOFF_MANIFEST_BYTES,
        code="tts.f1_handoff.manifest_invalid",
    )
    expected_top_level = {
        "schema",
        "candidate",
        "dataset",
        "piper",
        "files",
        "totals",
        "network_used",
        "installation_performed",
        "training_started",
    }
    if (
        set(manifest) != expected_top_level
        or manifest.get("schema") != HANDOFF_SCHEMA
        or manifest.get("candidate") != "stoma-f1-piper-medium"
        or manifest.get("installation_performed") is not False
    ):
        raise _error("tts.f1_handoff.manifest_incompatible", "Handoff manifest is incompatible")
    if manifest.get("network_used") is not False or manifest.get("training_started") is not False:
        raise _error("tts.f1_handoff.manifest_incompatible", "Handoff provenance is invalid")
    piper = manifest.get("piper")
    dataset = manifest.get("dataset")
    if (
        not isinstance(piper, Mapping)
        or piper.get("version") != PIPER_VERSION
        or piper.get("revision") != PIPER_REVISION
        or not isinstance(dataset, Mapping)
        or dataset.get("revision") != STOMA_REVISION
        or dataset.get("recipe_tree") != STOMA_GITHUB_TREE
        or dataset.get("speaker") != STOMA_SPEAKER
        or dataset.get("clips") != STOMA_SPEAKER_CLIPS
        or dataset.get("sample_rate") != PIPER_TARGET_SAMPLE_RATE
    ):
        raise _error("tts.f1_handoff.manifest_incompatible", "Handoff pins are incompatible")

    raw_files = manifest.get("files")
    totals = manifest.get("totals")
    if not isinstance(raw_files, list) or not isinstance(totals, Mapping):
        raise _error("tts.f1_handoff.manifest_incompatible", "Handoff file table is invalid")
    seen: set[str] = set()
    total_bytes = 0
    for raw_item in raw_files:
        if not isinstance(raw_item, Mapping) or set(raw_item) != {"path", "size", "sha256"}:
            raise _error("tts.f1_handoff.manifest_incompatible", "Handoff file entry is invalid")
        relative = _safe_relative(raw_item.get("path"))
        if relative in seen:
            raise _error("tts.f1_handoff.manifest_incompatible", "Handoff file entry is duplicated")
        seen.add(relative)
        expected_size = raw_item.get("size")
        expected_digest = raw_item.get("sha256")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 1
            or not _digest(expected_digest)
        ):
            raise _error("tts.f1_handoff.manifest_incompatible", "Handoff file metadata is invalid")
        path = (root / relative).resolve(strict=False)
        if root not in path.parents:
            raise _error("tts.f1_handoff.path_invalid", "Handoff path escapes transfer root")
        _regular_file(path, code="tts.f1_handoff.file_missing")
        if path.stat().st_size != expected_size or _sha256(path) != expected_digest:
            raise _error("tts.f1_handoff.file_hash_mismatch", "Transferred F1 input hash mismatch")
        total_bytes += expected_size
    expected_paths = {
        "f1-training-data/bundle-manifest.json",
        "f1-training-data/metadata.csv",
        *{
            f"checkpoint/{spec.relative_path}"
            for spec in PIPER_CHECKPOINT_ARTIFACTS
        },
    }
    audio_paths = {
        value
        for value in seen
        if value.startswith("f1-training-data/audio/") and value.endswith(".wav")
    }
    if len(audio_paths) != STOMA_SPEAKER_CLIPS or seen != expected_paths | audio_paths:
        raise _error("tts.f1_handoff.manifest_incompatible", "Handoff file set is invalid")
    if totals.get("files") != len(raw_files) or totals.get("bytes") != total_bytes:
        raise _error("tts.f1_handoff.manifest_incompatible", "Handoff totals are invalid")
    return {
        "ready": True,
        "code": "tts.f1_handoff.verified",
        "files": len(raw_files),
        "bytes": total_bytes,
        "network_used": False,
        "installation_performed": False,
        "training_started": False,
    }
