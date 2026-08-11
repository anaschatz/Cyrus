"""Read-only preflight for the compact STOMA F1 Greek TTS pilot.

The module deliberately performs no downloads, installation, training, or
production-backend changes.  It records the exact upstream inputs and the
qualification gates that must pass before the selected voice can be promoted.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import shutil
import struct
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, NamedTuple

from ..errors import BackendError


MIB = 1024 * 1024
GIB = 1024 * MIB

STOMA_REPO_ID = "aangelakis/STOMA"
STOMA_REVISION = "ea413ab51a4fc3b702f93afe26a4f4a2eb7d6a18"
STOMA_GITHUB_REPOSITORY = "https://github.com/aangelakis/STOMA"
STOMA_GITHUB_TREE = "92b1967ad5ab59c400ec7ee27b5bc07920ec7bc2"
STOMA_LICENSE = "CC-BY-4.0"
STOMA_SPEAKER = "F1"
STOMA_SPEAKER_CLIPS = 1_301
STOMA_SPEAKER_DURATION_SECONDS = 6_418
STOMA_SOURCE_SAMPLE_RATE = 44_100
STOMA_SOURCE_SAMPLE_WIDTH_BITS = 16
STOMA_SOURCE_CHANNELS = 1
STOMA_SECTION_COUNTS = {"B2": 190, "C1": 194, "C2": 197, "Harvard": 720}
STOMA_DOWNLOAD_ROWS = 1_942
STOMA_NON_F1_ROWS_TO_REJECT = STOMA_DOWNLOAD_ROWS - STOMA_SPEAKER_CLIPS

PIPER_REPOSITORY = "https://github.com/OHF-Voice/piper1-gpl"
PIPER_VERSION = "1.6.0"
PIPER_REVISION = "f04d52c5528ac7cf2d73757f57990ff490f75005"
PIPER_LICENSE = "GPL-3.0-or-later"
PIPER_WHEEL_NAME = "piper_tts-1.6.0-cp39-abi3-macosx_11_0_arm64.whl"

PIPER_CHECKPOINT_REPO_ID = "rhasspy/piper-checkpoints"
PIPER_CHECKPOINT_REVISION = "78b42203befdcf7c5c35fbe00aef5796e03aab44"
PIPER_CHECKPOINT_DIRECTORY = "el/el_GR/rapunzelina/medium"
PIPER_CHECKPOINT_DATASET_LICENSE = "CC0"
PIPER_TARGET_SAMPLE_RATE = 22_050

# The new F1 voice must remain small enough to coexist with the local LLM.
MAX_EXPORTED_MODEL_BYTES = 100 * MIB
MAX_RUNTIME_RSS_BYTES = 1 * GIB
MAX_WARM_FIRST_AUDIO_MS = 600
MAX_CANCEL_TO_SILENCE_MS = 100
MINIMUM_POST_ACQUISITION_FREE_BYTES = 2 * GIB


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    relative_path: str
    size: int
    sha256: str
    source_url: str


STOMA_SHARDS = (
    ArtifactSpec(
        "data/train-00000-of-00015.parquet",
        483_108_265,
        "827f910b4e53ac51c29fdab97bb68707ed752b2819b5749870c3dce9173f1d18",
        f"https://huggingface.co/datasets/{STOMA_REPO_ID}/resolve/{STOMA_REVISION}/"
        "data/train-00000-of-00015.parquet",
    ),
    ArtifactSpec(
        "data/train-00001-of-00015.parquet",
        430_719_339,
        "9e2734d147f11196b6ac42c4d6331f40a638e612a468a3aefcd60e885fcda7a7",
        f"https://huggingface.co/datasets/{STOMA_REPO_ID}/resolve/{STOMA_REVISION}/"
        "data/train-00001-of-00015.parquet",
    ),
)

PIPER_CHECKPOINT_ARTIFACTS = (
    ArtifactSpec(
        f"{PIPER_CHECKPOINT_DIRECTORY}/MODEL_CARD",
        276,
        "dc69f15382dabed061fa8d31b65c14e5c4b57ba2fe6906012c756478da156b50",
        f"https://huggingface.co/datasets/{PIPER_CHECKPOINT_REPO_ID}/resolve/"
        f"{PIPER_CHECKPOINT_REVISION}/{PIPER_CHECKPOINT_DIRECTORY}/MODEL_CARD",
    ),
    ArtifactSpec(
        f"{PIPER_CHECKPOINT_DIRECTORY}/config.json",
        7_088,
        "eed3d2596088c0d864b2232c5d0c38102b14d351cc6936da63b8dee9ad976201",
        f"https://huggingface.co/datasets/{PIPER_CHECKPOINT_REPO_ID}/resolve/"
        f"{PIPER_CHECKPOINT_REVISION}/{PIPER_CHECKPOINT_DIRECTORY}/config.json",
    ),
    ArtifactSpec(
        f"{PIPER_CHECKPOINT_DIRECTORY}/epoch=1999-step=485528.ckpt",
        845_898_776,
        "acf06940309a1accc591228e85b4ab96445393ca2f174a7cc9fff176495288d5",
        f"https://huggingface.co/datasets/{PIPER_CHECKPOINT_REPO_ID}/resolve/"
        f"{PIPER_CHECKPOINT_REVISION}/{PIPER_CHECKPOINT_DIRECTORY}/"
        "epoch%3D1999-step%3D485528.ckpt",
    ),
)

PIPER_RUNTIME_ARTIFACTS = (
    ArtifactSpec(
        PIPER_WHEEL_NAME,
        34_098_290,
        "b37ddd191b31995fbe981d5900df9c15888cccf427bff8e0b368cc95489fd60a",
        f"https://github.com/OHF-Voice/piper1-gpl/releases/download/v{PIPER_VERSION}/"
        f"{PIPER_WHEEL_NAME}",
    ),
)

PINNED_DOWNLOAD_BYTES = sum(
    item.size for item in STOMA_SHARDS + PIPER_CHECKPOINT_ARTIFACTS + PIPER_RUNTIME_ARTIFACTS
)
MINIMUM_ACQUISITION_FREE_BYTES = PINNED_DOWNLOAD_BYTES + MINIMUM_POST_ACQUISITION_FREE_BYTES

_F1_FILENAME = re.compile(r"^Greek-F1-(B2|C1|C2|Harvard)-\d{2}-\d{2}\.wav$")


class ValidationResult(NamedTuple):
    ready: bool
    code: str
    files_present: int
    files_required: int


class DatasetValidation(NamedTuple):
    ready: bool
    code: str
    rows_seen: int
    unique_files: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_distribution_version(name: str, expected: str) -> bool:
    try:
        return importlib.metadata.version(name) == expected
    except importlib.metadata.PackageNotFoundError:
        return False


def validate_artifact_set(root: str | Path, artifacts: tuple[ArtifactSpec, ...]) -> ValidationResult:
    """Validate an already materialized allow-list without following symlinks."""

    directory = Path(root).expanduser()
    required = len(artifacts)
    if not directory.is_absolute() or not directory.is_dir() or directory.is_symlink():
        return ValidationResult(False, "tts.f1_compact.artifact_root_missing", 0, required)
    present = 0
    for artifact in artifacts:
        path = directory / artifact.relative_path
        try:
            valid = (
                not path.is_symlink()
                and path.is_file()
                and path.stat().st_size == artifact.size
                and _sha256(path) == artifact.sha256
            )
        except OSError:
            valid = False
        if not valid:
            return ValidationResult(False, "tts.f1_compact.artifact_invalid", present, required)
        present += 1
    return ValidationResult(True, "tts.f1_compact.artifacts_ready", required, required)


def validate_f1_rows(rows: Iterable[Mapping[str, object]]) -> DatasetValidation:
    """Fail closed if shard extraction includes F2 or loses any F1 material."""

    seen: set[str] = set()
    sections: Counter[str] = Counter()
    count = 0
    for row in rows:
        count += 1
        speaker = row.get("speaker_id")
        filename = row.get("file_name")
        section = row.get("section")
        text = row.get("text")
        if (
            speaker != STOMA_SPEAKER
            or not isinstance(filename, str)
            or _F1_FILENAME.fullmatch(filename) is None
            or not isinstance(section, str)
            or not isinstance(text, str)
            or not text.strip()
            or filename in seen
        ):
            return DatasetValidation(False, "tts.f1_compact.dataset_contaminated", count, len(seen))
        seen.add(filename)
        sections[section] += 1
    if count != STOMA_SPEAKER_CLIPS or dict(sections) != STOMA_SECTION_COUNTS:
        return DatasetValidation(False, "tts.f1_compact.dataset_incomplete", count, len(seen))
    return DatasetValidation(True, "tts.f1_compact.dataset_ready", count, len(seen))


def pinned_f1_manifest() -> dict[str, object]:
    """Return a JSON-safe, immutable-in-source specification for the pilot."""

    return {
        "schema": "jarvis.voice.compact-greek-tts-pilot/1",
        "candidate": "stoma-f1-piper-medium",
        "dataset": {
            "repo_id": STOMA_REPO_ID,
            "revision": STOMA_REVISION,
            "github_tree": STOMA_GITHUB_TREE,
            "license": STOMA_LICENSE,
            "speaker": STOMA_SPEAKER,
            "clips": STOMA_SPEAKER_CLIPS,
            "duration_seconds": STOMA_SPEAKER_DURATION_SECONDS,
            "source_audio": {
                "sample_rate": STOMA_SOURCE_SAMPLE_RATE,
                "sample_width_bits": STOMA_SOURCE_SAMPLE_WIDTH_BITS,
                "channels": STOMA_SOURCE_CHANNELS,
            },
            "download_rows": STOMA_DOWNLOAD_ROWS,
            "non_f1_rows_to_reject": STOMA_NON_F1_ROWS_TO_REJECT,
            "section_counts": dict(STOMA_SECTION_COUNTS),
            "shards": [asdict(item) for item in STOMA_SHARDS],
        },
        "runtime": {
            "project": PIPER_REPOSITORY,
            "version": PIPER_VERSION,
            "revision": PIPER_REVISION,
            "license": PIPER_LICENSE,
            "artifacts": [asdict(item) for item in PIPER_RUNTIME_ARTIFACTS],
        },
        "warm_start": {
            "repo_id": PIPER_CHECKPOINT_REPO_ID,
            "revision": PIPER_CHECKPOINT_REVISION,
            "directory": PIPER_CHECKPOINT_DIRECTORY,
            "dataset_license": PIPER_CHECKPOINT_DATASET_LICENSE,
            "target_sample_rate": PIPER_TARGET_SAMPLE_RATE,
            "artifacts": [asdict(item) for item in PIPER_CHECKPOINT_ARTIFACTS],
        },
        "qualification": {
            "max_exported_model_bytes": MAX_EXPORTED_MODEL_BYTES,
            "max_runtime_rss_bytes": MAX_RUNTIME_RSS_BYTES,
            "max_warm_first_audio_ms": MAX_WARM_FIRST_AUDIO_MS,
            "max_cancel_to_silence_ms": MAX_CANCEL_TO_SILENCE_MS,
            "network_allowed": False,
            "production_backend_unchanged_until_pass": True,
            "human_f1_voice_approval_required": True,
        },
    }


def compact_greek_tts_preflight(
    storage_root: str | Path,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    python_version: tuple[int, int, int] | None = None,
    pointer_bits: int | None = None,
    disk_usage: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
    dependency_probe: Callable[[str, str], bool] = _exact_distribution_version,
) -> dict[str, object]:
    """Inspect host readiness while guaranteeing no network or filesystem writes."""

    root = Path(storage_root).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise BackendError(
            "tts.f1_compact.storage_root_invalid",
            "Compact Greek TTS storage root is invalid",
            subsystem="tts",
        )
    selected_platform = sys.platform if platform_name is None else platform_name
    selected_machine = platform.machine() if machine is None else machine
    selected_version = sys.version_info[:3] if python_version is None else python_version
    selected_bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    try:
        free_bytes = int(getattr(disk_usage(root), "free"))
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        raise BackendError(
            "tts.f1_compact.disk_probe_failed",
            "Compact Greek TTS disk probe failed",
            subsystem="tts",
        ) from exc

    host_ready = (
        selected_platform == "darwin"
        and selected_machine == "arm64"
        and selected_version[:2] == (3, 11)
        and selected_bits == 64
    )
    acquisition_ready = host_ready and free_bytes >= MINIMUM_ACQUISITION_FREE_BYTES
    piper_ready = bool(dependency_probe("piper-tts", PIPER_VERSION))
    if not host_ready:
        code = "tts.f1_compact.host_unsupported"
    elif free_bytes < MINIMUM_ACQUISITION_FREE_BYTES:
        code = "tts.f1_compact.disk_space_insufficient"
    else:
        code = "tts.f1_compact.preflight_ready"

    return {
        "ready": acquisition_ready,
        "code": code,
        "candidate": "stoma-f1-piper-medium",
        "host": {
            "platform": selected_platform,
            "machine": selected_machine,
            "python": ".".join(str(part) for part in selected_version),
            "pointer_bits": selected_bits,
            "ready": host_ready,
        },
        "disk": {
            "free_mib": free_bytes // MIB,
            "pinned_download_mib": PINNED_DOWNLOAD_BYTES // MIB,
            "acquisition_required_mib": MINIMUM_ACQUISITION_FREE_BYTES // MIB,
            "ready": free_bytes >= MINIMUM_ACQUISITION_FREE_BYTES,
        },
        "dependencies": {"piper-tts": {"expected": PIPER_VERSION, "ready": piper_ready}},
        "acquisition_ready": acquisition_ready,
        "inference_runtime_ready": piper_ready,
        "training_target": "separate CUDA host; 8 GiB VRAM minimum reported upstream",
        "official_stoma_weights_available": False,
        "official_stoma_weights_access": "contact STOMA authors",
        "license_review_required": PIPER_LICENSE,
        "distribution_ready": False,
        "production_ready": False,
        "network_used": False,
        "installation_performed": False,
        "model_downloaded": False,
        "production_backend_changed": False,
        "manifest": pinned_f1_manifest(),
    }
