"""Pinned acquisition and F1 extraction for the compact Greek TTS pilot.

Network access is possible only through the explicit ``acquire`` operation.
Planning, extraction, and training-plan generation are offline operations.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..errors import BackendError
from .compact_greek_tts_pilot import (
    GIB,
    MIB,
    ArtifactSpec,
    PIPER_CHECKPOINT_ARTIFACTS,
    PIPER_CHECKPOINT_DIRECTORY,
    PIPER_REVISION,
    PIPER_RUNTIME_ARTIFACTS,
    PIPER_TARGET_SAMPLE_RATE,
    PIPER_VERSION,
    STOMA_DOWNLOAD_ROWS,
    STOMA_GITHUB_TREE,
    STOMA_NON_F1_ROWS_TO_REJECT,
    STOMA_REVISION,
    STOMA_SHARDS,
    STOMA_SOURCE_CHANNELS,
    STOMA_SOURCE_SAMPLE_RATE,
    STOMA_SOURCE_SAMPLE_WIDTH_BITS,
    STOMA_SPEAKER,
    STOMA_SPEAKER_CLIPS,
    ValidationResult,
    validate_artifact_set,
    validate_f1_rows,
)


ACQUISITION_GROUPS: dict[str, tuple[ArtifactSpec, ...]] = {
    "stoma": STOMA_SHARDS,
    "checkpoint": PIPER_CHECKPOINT_ARTIFACTS,
    "runtime": PIPER_RUNTIME_ARTIFACTS,
}
ACQUISITION_HEADROOM_BYTES = 2 * GIB
DOWNLOAD_CHUNK_BYTES = MIB
VOICE_NAME = "el_GR-jarvis-f1-medium"

_ALLOWED_DOWNLOAD_HOSTS = {
    "github.com",
    "huggingface.co",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
    "us.aws.cdn.hf.co",
}
_ALLOWED_DOWNLOAD_SUFFIXES = (".huggingface.co", ".xethub.hf.co")


def _error(code: str, message: str, *, retryable: bool = False) -> BackendError:
    return BackendError(code, message, subsystem="tts", retryable=retryable)


def _validated_root(value: str | Path, *, code: str) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise _error(code, "Compact Greek TTS root is invalid")
    return root.resolve()


def _selected_groups(group: str) -> tuple[str, ...]:
    if group == "all":
        return tuple(ACQUISITION_GROUPS)
    if group not in ACQUISITION_GROUPS:
        raise _error("tts.f1_compact.group_invalid", "Acquisition group is invalid")
    return (group,)


def _safe_destination(root: Path, group: str, relative_path: str) -> Path:
    destination = (root / group / relative_path).resolve(strict=False)
    if root != destination and root not in destination.parents:
        raise _error("tts.f1_compact.destination_unsafe", "Artifact destination is unsafe")
    return destination


def _download_url_allowed(value: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(value)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    return parsed.scheme == "https" and (
        host in _ALLOWED_DOWNLOAD_HOSTS or any(host.endswith(suffix) for suffix in _ALLOWED_DOWNLOAD_SUFFIXES)
    )


class _ArtifactRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _download_url_allowed(newurl):
            fp.close()
            raise _error("tts.f1_compact.redirect_forbidden", "Artifact redirect is forbidden")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def compact_acquisition_plan(storage_root: str | Path, *, group: str = "all") -> dict[str, object]:
    """Return an exact download plan without creating files or using network."""

    root = _validated_root(storage_root, code="tts.f1_compact.storage_root_invalid")
    groups = _selected_groups(group)
    artifacts: list[dict[str, object]] = []
    total = 0
    for group_name in groups:
        for spec in ACQUISITION_GROUPS[group_name]:
            total += spec.size
            artifacts.append(
                {
                    "group": group_name,
                    "destination": str(_safe_destination(root, group_name, spec.relative_path)),
                    **asdict(spec),
                }
            )
    try:
        free_bytes = shutil.disk_usage(root).free
    except OSError as exc:
        raise _error("tts.f1_compact.disk_probe_failed", "Disk probe failed") from exc
    return {
        "ready": free_bytes >= total + ACQUISITION_HEADROOM_BYTES,
        "code": (
            "tts.f1_compact.acquisition_plan_ready"
            if free_bytes >= total + ACQUISITION_HEADROOM_BYTES
            else "tts.f1_compact.disk_space_insufficient"
        ),
        "group": group,
        "download_bytes": total,
        "download_mib": total // MIB,
        "required_free_bytes": total + ACQUISITION_HEADROOM_BYTES,
        "free_bytes": free_bytes,
        "artifacts": artifacts,
        "network_used": False,
        "installation_performed": False,
    }


def _hash_existing(path: Path) -> tuple[Any, int]:
    digest = hashlib.sha256()
    size = 0
    if path.is_symlink():
        raise _error("tts.f1_compact.partial_unsafe", "Partial download is unsafe")
    if path.exists():
        if not path.is_file():
            raise _error("tts.f1_compact.partial_unsafe", "Partial download is unsafe")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK_BYTES), b""):
                size += len(chunk)
                digest.update(chunk)
    return digest, size


def _download_artifact(
    spec: ArtifactSpec,
    destination: Path,
    *,
    urlopen: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Download one exact artifact with resumable ``.part`` and atomic publish."""

    if not _download_url_allowed(spec.source_url):
        raise _error("tts.f1_compact.source_url_forbidden", "Artifact URL is forbidden")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        result = validate_artifact_set(destination.parent, (ArtifactSpec(destination.name, spec.size, spec.sha256, spec.source_url),))
        if result.ready:
            return {"relative_path": spec.relative_path, "status": "already_present", "bytes": spec.size}
        raise _error("tts.f1_compact.existing_artifact_invalid", "Existing artifact is invalid")

    partial = destination.with_name(f"{destination.name}.part")
    digest, offset = _hash_existing(partial)
    if offset > spec.size:
        raise _error("tts.f1_compact.partial_too_large", "Partial artifact is too large")
    total = offset
    if offset < spec.size:
        headers = {"Accept-Encoding": "identity", "User-Agent": "JARVIS-Voice-F1-Pilot/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(spec.source_url, headers=headers, method="GET")

        try:
            opener = urllib.request.build_opener(_ArtifactRedirectHandler()).open if urlopen is None else urlopen
            response_manager = opener(request, timeout=60)
            with response_manager as response:
                final_url = str(response.geturl())
                if not _download_url_allowed(final_url):
                    raise _error("tts.f1_compact.redirect_forbidden", "Artifact redirect is forbidden")
                status = int(getattr(response, "status", response.getcode()))
                content_encoding = str(response.headers.get("Content-Encoding", "identity")).lower()
                if content_encoding not in {"", "identity"}:
                    raise _error("tts.f1_compact.content_encoding_invalid", "Artifact response is encoded")
                append = offset > 0 and status == 206
                if append:
                    content_range = str(response.headers.get("Content-Range", ""))
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise _error("tts.f1_compact.range_invalid", "Artifact range response is invalid")
                elif status in {200, 206}:
                    digest = hashlib.sha256()
                    offset = 0
                else:
                    raise _error("tts.f1_compact.http_status_invalid", "Artifact HTTP status is invalid", retryable=True)

                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    total = offset
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > spec.size:
                            raise _error("tts.f1_compact.download_too_large", "Artifact exceeded pinned size")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
        except OSError as exc:
            raise _error("tts.f1_compact.download_failed", "Artifact download failed", retryable=True) from exc

    if total != spec.size or digest.hexdigest() != spec.sha256:
        raise _error("tts.f1_compact.download_hash_mismatch", "Artifact hash validation failed")
    os.link(partial, destination)
    partial.unlink()
    return {"relative_path": spec.relative_path, "status": "downloaded", "bytes": total}


def acquire_pinned_artifacts(
    storage_root: str | Path,
    *,
    group: str = "all",
    confirm_download_bytes: int,
    downloader: Callable[[ArtifactSpec, Path], Mapping[str, object]] = _download_artifact,
) -> dict[str, object]:
    """Acquire an allow-listed group only after exact byte-count confirmation."""

    plan = compact_acquisition_plan(storage_root, group=group)
    expected = int(plan["download_bytes"])
    if confirm_download_bytes != expected:
        raise _error("tts.f1_compact.download_confirmation_invalid", "Exact download size was not confirmed")
    if not plan["ready"]:
        raise _error("tts.f1_compact.disk_space_insufficient", "Not enough disk for acquisition")
    results: list[Mapping[str, object]] = []
    for artifact in plan["artifacts"]:
        spec = ArtifactSpec(
            str(artifact["relative_path"]),
            int(artifact["size"]),
            str(artifact["sha256"]),
            str(artifact["source_url"]),
        )
        results.append(downloader(spec, Path(str(artifact["destination"]))))
    return {
        "ready": True,
        "code": "tts.f1_compact.acquisition_complete",
        "group": group,
        "download_bytes": expected,
        "artifacts": list(results),
        "network_used": True,
        "installation_performed": False,
        "production_backend_changed": False,
    }


def _validated_wav_bytes(value: object) -> bytes:
    if not isinstance(value, Mapping):
        raise _error("tts.f1_compact.audio_schema_invalid", "Parquet audio schema is invalid")
    audio_bytes = value.get("bytes")
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise _error("tts.f1_compact.audio_embedded_bytes_missing", "Embedded WAV bytes are missing")
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as stream:
            valid = (
                stream.getnchannels() == STOMA_SOURCE_CHANNELS
                and stream.getsampwidth() * 8 == STOMA_SOURCE_SAMPLE_WIDTH_BITS
                and stream.getframerate() == STOMA_SOURCE_SAMPLE_RATE
                and stream.getnframes() > 0
                and stream.getcomptype() == "NONE"
            )
            if valid and len(stream.readframes(stream.getnframes())) != (
                stream.getnframes() * stream.getnchannels() * stream.getsampwidth()
            ):
                raise _error("tts.f1_compact.audio_wav_invalid", "Embedded WAV is truncated")
    except (EOFError, wave.Error) as exc:
        raise _error("tts.f1_compact.audio_wav_invalid", "Embedded WAV is invalid") from exc
    if not valid:
        raise _error("tts.f1_compact.audio_format_invalid", "Embedded WAV format is invalid")
    return audio_bytes


def _load_parquet_module() -> object:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise _error("tts.f1_compact.pyarrow_missing", "PyArrow is required only for F1 extraction") from exc
    return parquet


def extract_stoma_f1(
    artifact_root: str | Path,
    output_directory: str | Path,
    *,
    parquet_module: object | None = None,
    artifact_validator: Callable[[str | Path, tuple[ArtifactSpec, ...]], ValidationResult] = validate_artifact_set,
) -> dict[str, object]:
    """Extract exactly F1 into an atomic Piper CSV/WAV training directory."""

    artifacts = _validated_root(artifact_root, code="tts.f1_compact.artifact_root_invalid")
    stoma_root = artifacts / "stoma"
    validated = artifact_validator(stoma_root, STOMA_SHARDS)
    if not validated.ready:
        raise _error(validated.code, "Pinned STOMA shards are unavailable")
    destination = Path(output_directory).expanduser()
    parent = _validated_root(destination.parent, code="tts.f1_compact.output_parent_invalid")
    destination = destination.resolve(strict=False)
    if parent not in destination.parents or destination.exists():
        raise _error("tts.f1_compact.output_directory_unsafe", "F1 output directory is unsafe")

    parquet = _load_parquet_module() if parquet_module is None else parquet_module
    temporary = Path(tempfile.mkdtemp(prefix=".jarvis-f1-extract-", dir=parent))
    audio_directory = temporary / "audio"
    rows_for_validation: list[dict[str, object]] = []
    metadata_lines: list[str] = []
    audio_hashes: dict[str, str] = {}
    total_rows = 0
    rejected_f2 = 0
    try:
        audio_directory.mkdir(mode=0o700)
        for shard in STOMA_SHARDS:
            parquet_file = parquet.ParquetFile(stoma_root / shard.relative_path)
            batches: Iterable[object] = parquet_file.iter_batches(
                batch_size=16,
                columns=("speaker_id", "section", "text", "audio", "file_name"),
            )
            for batch in batches:
                for row in batch.to_pylist():
                    total_rows += 1
                    speaker = row.get("speaker_id")
                    if speaker == "F2":
                        rejected_f2 += 1
                        continue
                    if speaker != STOMA_SPEAKER:
                        raise _error("tts.f1_compact.dataset_speaker_unexpected", "Unexpected speaker in pinned shards")
                    filename = row.get("file_name")
                    text = row.get("text")
                    if not isinstance(filename, str) or not isinstance(text, str):
                        raise _error("tts.f1_compact.dataset_row_invalid", "STOMA row is invalid")
                    clean_text = text.strip()
                    if not clean_text or "|" in clean_text or "\n" in clean_text or "\r" in clean_text:
                        raise _error("tts.f1_compact.transcript_unsafe", "STOMA transcript is unsafe for Piper CSV")
                    wav_bytes = _validated_wav_bytes(row.get("audio"))
                    wav_path = audio_directory / filename
                    if wav_path.exists() or wav_path.parent != audio_directory:
                        raise _error("tts.f1_compact.audio_filename_unsafe", "STOMA audio filename is unsafe")
                    wav_path.write_bytes(wav_bytes)
                    audio_hashes[filename] = hashlib.sha256(wav_bytes).hexdigest()
                    row_for_validation = {
                        "speaker_id": speaker,
                        "section": row.get("section"),
                        "file_name": filename,
                        "text": clean_text,
                    }
                    rows_for_validation.append(row_for_validation)
                    metadata_lines.append(f"{filename}|{clean_text}")

        dataset_result = validate_f1_rows(rows_for_validation)
        if not dataset_result.ready:
            raise _error(dataset_result.code, "Extracted STOMA F1 dataset failed validation")
        if total_rows != STOMA_DOWNLOAD_ROWS or rejected_f2 != STOMA_NON_F1_ROWS_TO_REJECT:
            raise _error("tts.f1_compact.shard_row_count_invalid", "Pinned shard row counts changed")

        metadata_path = temporary / "metadata.csv"
        metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
        bundle_manifest = {
            "schema": "jarvis.voice.stoma-f1-training-bundle/1",
            "dataset_revision": STOMA_REVISION,
            "recipe_tree": STOMA_GITHUB_TREE,
            "speaker": STOMA_SPEAKER,
            "rows": len(rows_for_validation),
            "rejected_f2_rows": rejected_f2,
            "source_sample_rate": STOMA_SOURCE_SAMPLE_RATE,
            "piper_sample_rate": PIPER_TARGET_SAMPLE_RATE,
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "audio_sha256": audio_hashes,
            "network_used": False,
        }
        (temporary / "bundle-manifest.json").write_text(
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "ready": True,
        "code": "tts.f1_compact.extraction_complete",
        "output_directory": str(destination),
        "f1_rows": STOMA_SPEAKER_CLIPS,
        "rejected_f2_rows": rejected_f2,
        "network_used": False,
        "installation_performed": False,
    }


def compact_f1_training_plan(
    data_directory: str | Path,
    artifact_root: str | Path,
    work_directory: str | Path,
) -> dict[str, object]:
    """Build the pinned WSL2/CUDA command plan without executing it."""

    raw_paths = tuple(Path(value).expanduser() for value in (data_directory, artifact_root, work_directory))
    if not all(path.is_absolute() for path in raw_paths):
        raise _error("tts.f1_compact.training_path_invalid", "Training plan paths must be absolute")
    data, artifacts, work = (path.resolve(strict=False) for path in raw_paths)
    checkpoint = artifacts / "checkpoint" / PIPER_CHECKPOINT_DIRECTORY / "epoch=1999-step=485528.ckpt"
    config = work / "export" / f"{VOICE_NAME}.onnx.json"
    training_argv = [
        "python3",
        "-m",
        "piper.train",
        "fit",
        "--trainer.accelerator",
        "gpu",
        "--trainer.devices",
        "1",
        "--trainer.max_epochs",
        "1000",
        "--trainer.default_root_dir",
        str(work / "runs"),
        "--trainer.log_every_n_steps",
        "25",
        "--data.voice_name",
        VOICE_NAME,
        "--data.csv_path",
        str(data / "metadata.csv"),
        "--data.audio_dir",
        str(data / "audio"),
        "--model.sample_rate",
        str(PIPER_TARGET_SAMPLE_RATE),
        "--data.espeak_voice",
        "el",
        "--data.cache_dir",
        str(work / "cache"),
        "--data.config_path",
        str(config),
        "--data.batch_size",
        "4",
        "--data.validation_split",
        "0.1",
        "--data.num_test_examples",
        "8",
        "--data.num_workers",
        "2",
        "--ckpt_path",
        str(checkpoint),
    ]
    export_argv = [
        "python3",
        "scripts/export_piper_mps_checkpoint.py",
        "--checkpoint",
        "<approved-checkpoint.ckpt>",
        "--output-file",
        str(work / "export" / f"{VOICE_NAME}.onnx"),
    ]
    return {
        "ready": False,
        "code": "tts.f1_compact.training_environment_pending",
        "platform": "WSL2 Ubuntu with NVIDIA CUDA",
        "python": "3.11",
        "piper_version": PIPER_VERSION,
        "piper_revision": PIPER_REVISION,
        "gpu_probe_required": True,
        "checkpoint_load_probe_required": True,
        "dependency_lock_required": True,
        "recommended_minimum_vram_gib": 8,
        "training_argv": training_argv,
        "export_argv": export_argv,
        "max_epochs_is_ceiling": True,
        "listen_before_continuing_epochs": [100, 250, 500, 1000],
        "network_used": False,
        "installation_performed": False,
        "training_started": False,
    }
