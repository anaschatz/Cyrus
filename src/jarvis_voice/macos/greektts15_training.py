"""Deterministic preflight and dataset planning for GreekTTS-1.5 F1 tuning.

This module deliberately has no MLX, Torch, or audio-model imports.  It proves
that the local STOMA F1 bundle is intact and creates a stable train/validation/
control split before any heavyweight dependency is installed or model loaded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
import wave
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from jarvis_voice.errors import BackendError


DATASET_SCHEMA = "jarvis.voice.stoma-f1-training-bundle/1"
PLAN_SCHEMA = "jarvis.voice.greektts15-f1-plan/1"
EXPECTED_ROWS = 1_301
EXPECTED_SAMPLE_RATE = 44_100
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2
SPLIT_SEED = "jarvis-greektts15-f1-v1"
MLX_TUNE_REVISION = "9690fe1fed19c72f0810902b9c188d9a2625eb5b"
BASE_REVISION = "b30aed4ecfa03c0177a5419b51919d14a105f183"
GREEK_ADAPTER_REVISION = "0da4d9fff8b974fb1127f3c829f793dc0439dfd1"
SNAC_REVISION = "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"

_LEADING_CONFUSABLES = str.maketrans(
    {
        "A": "Α",
        "B": "Β",
        "E": "Ε",
        "H": "Η",
        "I": "Ι",
        "K": "Κ",
        "M": "Μ",
        "N": "Ν",
        "O": "Ο",
        "P": "Ρ",
        "T": "Τ",
        "X": "Χ",
        "Y": "Υ",
        "Z": "Ζ",
        "o": "ο",
    }
)
_LEADING_ASCII_WORD = re.compile(r"^(?P<word>[A-Za-z]{1,2})(?=\s)")


@dataclass(frozen=True)
class PlannedSample:
    filename: str
    text: str
    split: str
    corpus: str
    duration_seconds: float
    audio_sha256: str
    transcript_normalized: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_transcript(text: str) -> tuple[str, bool]:
    """Normalize layout and repair only a leading Greek-looking ASCII token.

    STOMA contains OCR-like sentence initials such as ``Tην`` and ``To``.
    The replacement is intentionally limited to the first one- or two-letter
    token so English names and abbreviations elsewhere remain untouched.
    """

    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    match = _LEADING_ASCII_WORD.match(normalized)
    if match:
        word = match.group("word")
        replacement = word.translate(_LEADING_CONFUSABLES)
        if replacement != word and all(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" for character in replacement):
            normalized = replacement + normalized[len(word) :]
    elif normalized:
        first = normalized[0].translate(_LEADING_CONFUSABLES)
        if first != normalized[0]:
            normalized = first + normalized[1:]
    return normalized, normalized != text


def _corpus(filename: str) -> str:
    parts = filename.removesuffix(".wav").split("-")
    if len(parts) < 4 or parts[:2] != ["Greek", "F1"]:
        raise BackendError("tts.greektts15.dataset_filename_invalid", "Unexpected F1 filename", subsystem="tts")
    return parts[2]


def _split_rows(filenames: Iterable[str]) -> dict[str, str]:
    """Assign stable 90/5/5 splits independently inside each corpus."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for filename in filenames:
        grouped[_corpus(filename)].append(filename)
    result: dict[str, str] = {}
    for corpus, members in sorted(grouped.items()):
        ordered = sorted(
            members,
            key=lambda name: hashlib.sha256(f"{SPLIT_SEED}:{corpus}:{name}".encode()).digest(),
        )
        validation_count = max(1, round(len(ordered) * 0.05))
        control_count = max(1, round(len(ordered) * 0.05))
        for index, filename in enumerate(ordered):
            if index < validation_count:
                split = "validation"
            elif index < validation_count + control_count:
                split = "control"
            else:
                split = "train"
            result[filename] = split
    return result


def build_training_plan(dataset_root: str | Path, *, verify_audio_hashes: bool = True) -> dict[str, object]:
    root = Path(dataset_root).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise BackendError("tts.greektts15.dataset_root_invalid", "F1 dataset root is invalid", subsystem="tts")
    metadata_path = root / "metadata.csv"
    manifest_path = root / "bundle-manifest.json"
    audio_root = root / "audio"
    if any(path.is_symlink() for path in (metadata_path, manifest_path, audio_root)):
        raise BackendError("tts.greektts15.dataset_symlink_forbidden", "F1 bundle contains a symlink", subsystem="tts")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.dataset_manifest_invalid", "F1 manifest is unreadable", subsystem="tts") from exc
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("rows") != EXPECTED_ROWS:
        raise BackendError("tts.greektts15.dataset_manifest_invalid", "F1 manifest does not match the pinned bundle", subsystem="tts")
    if _sha256(metadata_path) != manifest.get("metadata_sha256"):
        raise BackendError("tts.greektts15.dataset_metadata_hash_invalid", "F1 metadata hash mismatch", subsystem="tts")

    rows: list[tuple[str, str]] = []
    with metadata_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="|"):
            if len(row) != 2 or not row[0] or not row[1]:
                raise BackendError("tts.greektts15.dataset_row_invalid", "F1 metadata row is invalid", subsystem="tts")
            rows.append((row[0], row[1]))
    filenames = [filename for filename, _ in rows]
    if len(rows) != EXPECTED_ROWS or len(set(filenames)) != EXPECTED_ROWS:
        raise BackendError("tts.greektts15.dataset_row_count_invalid", "F1 rows are missing or duplicated", subsystem="tts")
    expected_hashes = manifest.get("audio_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(filenames):
        raise BackendError("tts.greektts15.dataset_audio_manifest_invalid", "F1 audio manifest is incomplete", subsystem="tts")

    splits = _split_rows(filenames)
    samples: list[PlannedSample] = []
    for filename, source_text in rows:
        audio_path = audio_root / filename
        if audio_path.is_symlink() or not audio_path.is_file() or audio_path.parent != audio_root:
            raise BackendError("tts.greektts15.dataset_audio_invalid", "F1 audio path is invalid", subsystem="tts")
        digest = _sha256(audio_path) if verify_audio_hashes else str(expected_hashes[filename])
        if digest != expected_hashes[filename]:
            raise BackendError("tts.greektts15.dataset_audio_hash_invalid", "F1 audio hash mismatch", subsystem="tts")
        try:
            with wave.open(str(audio_path), "rb") as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
                    EXPECTED_SAMPLE_RATE,
                    EXPECTED_CHANNELS,
                    EXPECTED_SAMPLE_WIDTH,
                ):
                    raise BackendError("tts.greektts15.dataset_audio_format_invalid", "F1 WAV format is invalid", subsystem="tts")
                duration = wav.getnframes() / wav.getframerate()
        except (OSError, wave.Error) as exc:
            raise BackendError("tts.greektts15.dataset_audio_invalid", "F1 WAV is unreadable", subsystem="tts") from exc
        if not 1.5 <= duration <= 16.0:
            raise BackendError("tts.greektts15.dataset_duration_invalid", "F1 sample duration is outside the safe bound", subsystem="tts")
        text, changed = normalize_transcript(source_text)
        samples.append(
            PlannedSample(filename, text, splits[filename], _corpus(filename), round(duration, 6), digest, changed)
        )

    counts = {name: sum(sample.split == name for sample in samples) for name in ("train", "validation", "control")}
    return {
        "schema": PLAN_SCHEMA,
        "ready": True,
        "dataset": {
            "schema": DATASET_SCHEMA,
            "revision": manifest.get("dataset_revision"),
            "speaker": "F1",
            "license": "CC-BY-4.0",
            "rows": len(samples),
            "duration_seconds": round(sum(sample.duration_seconds for sample in samples), 6),
            "normalized_transcripts": sum(sample.transcript_normalized for sample in samples),
        },
        "split": {"seed": SPLIT_SEED, "counts": counts},
        "pinned_runtime": {
            "mlx_tune_revision": MLX_TUNE_REVISION,
            "base_revision": BASE_REVISION,
            "greek_adapter_revision": GREEK_ADAPTER_REVISION,
            "snac_revision": SNAC_REVISION,
        },
        "training_policy": {
            "continue_existing_rank_64_adapter": True,
            "merge_into_quantized_base": False,
            "initial_learning_rates": [5e-6, 1e-5],
            "checkpoint_steps": [25, 50, 100],
            "requires_human_listening_gate": True,
        },
        "samples": [asdict(sample) for sample in samples],
    }


def write_training_plan(dataset_root: str | Path, output_path: str | Path) -> dict[str, object]:
    plan = build_training_plan(dataset_root)
    destination = Path(output_path).expanduser()
    if not destination.is_absolute() or destination.is_symlink():
        raise BackendError("tts.greektts15.plan_path_invalid", "Training plan path is invalid", subsystem="tts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return plan
