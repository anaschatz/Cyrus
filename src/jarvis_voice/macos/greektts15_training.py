"""Deterministic preflight and dataset planning for GreekTTS-1.5 F1 tuning.

This module deliberately has no MLX, Torch, or audio-model imports.  It proves
that the local STOMA F1 bundle is intact and creates a stable train/validation/
control split before any heavyweight dependency is installed or model loaded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import unicodedata
import wave
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.greektts15_pilot import (
    AUDIO_TOKEN_OFFSET,
    CODEBOOK_SIZE,
    END_OF_HUMAN,
    END_OF_SPEECH,
    END_OF_TEXT,
    START_OF_HUMAN,
    START_OF_SPEECH,
)


DATASET_SCHEMA = "jarvis.voice.stoma-f1-training-bundle/1"
PLAN_SCHEMA = "jarvis.voice.greektts15-f1-plan/1"
CACHE_SCHEMA = "jarvis.voice.greektts15-snac-cache/1"
EXPECTED_ROWS = 1_301
EXPECTED_SAMPLE_RATE = 44_100
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2
EDGE_TRIM_MILLISECONDS = 50
SPLIT_SEED = "jarvis-greektts15-f1-v1"
MLX_TUNE_REVISION = "9690fe1fed19c72f0810902b9c188d9a2625eb5b"
BASE_REVISION = "b30aed4ecfa03c0177a5419b51919d14a105f183"
GREEK_ADAPTER_REVISION = "0da4d9fff8b974fb1127f3c829f793dc0439dfd1"
SNAC_REVISION = "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"
MAX_TRAINING_SEQUENCE = 1_536
EXPECTED_LORA_GROUPS = 392

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


class SnacEncoder(Protocol):
    signature: dict[str, object]

    def encode(self, audio_path: Path) -> list[int]: ...


class TorchSnacEncoder:
    """Offline CPU encoder for the pinned 24 kHz SNAC checkpoint."""

    def __init__(self, snac_directory: str | Path) -> None:
        directory = Path(snac_directory).expanduser()
        if not directory.is_absolute() or not directory.is_dir() or directory.is_symlink():
            raise BackendError("tts.greektts15.snac_root_invalid", "SNAC model root is invalid", subsystem="tts")
        try:
            import numpy as np
            import soundfile as sf
            import soxr
            import torch
            from snac import SNAC
        except ImportError as exc:
            raise BackendError("tts.greektts15.dependencies_missing", "SNAC cache dependencies are missing", subsystem="tts") from exc
        self._np = np
        self._sf = sf
        self._soxr = soxr
        self._torch = torch
        previous = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        try:
            self._model = SNAC.from_pretrained(str(directory)).to("cpu").eval()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.signature = {
            "backend": "torch-snac",
            "model_revision": SNAC_REVISION,
            "model_sha256": _sha256(directory / "pytorch_model.bin"),
            "sample_rate": 24_000,
            "resampler": "soxr-vhq",
            "edge_trim_ms": EDGE_TRIM_MILLISECONDS,
            "device": "cpu",
        }

    def encode(self, audio_path: Path) -> list[int]:
        waveform, sample_rate = self._sf.read(str(audio_path), dtype="float32", always_2d=False)
        if waveform.ndim != 1 or waveform.size == 0 or not self._np.isfinite(waveform).all():
            raise BackendError("tts.greektts15.dataset_audio_invalid", "F1 waveform is invalid", subsystem="tts")
        trim = round(sample_rate * EDGE_TRIM_MILLISECONDS / 1_000)
        if waveform.size <= trim * 2:
            raise BackendError("tts.greektts15.dataset_audio_invalid", "F1 waveform is too short to trim", subsystem="tts")
        waveform = waveform[trim:-trim]
        if sample_rate != 24_000:
            waveform = self._soxr.resample(waveform, sample_rate, 24_000, quality="VHQ")
        tensor = self._torch.from_numpy(self._np.asarray(waveform, dtype=self._np.float32)).reshape(1, 1, -1)
        with self._torch.inference_mode():
            layers = self._model.encode(tensor)
        values = [layer.detach().to("cpu").reshape(-1).tolist() for layer in layers]
        if len(values) != 3:
            raise BackendError("tts.greektts15.snac_shape_invalid", "SNAC returned the wrong layer count", subsystem="tts")
        return interleave_snac_codes(values[0], values[1], values[2])


def interleave_snac_codes(
    layer_1: Iterable[int],
    layer_2: Iterable[int],
    layer_3: Iterable[int],
) -> list[int]:
    """Encode the exact seven-token Orpheus frame layout used at runtime."""

    first, second, third = list(layer_1), list(layer_2), list(layer_3)
    if not first or len(second) != len(first) * 2 or len(third) != len(first) * 4:
        raise BackendError("tts.greektts15.snac_shape_invalid", "SNAC layers do not form complete frames", subsystem="tts")
    if any(code < 0 or code >= CODEBOOK_SIZE for layer in (first, second, third) for code in layer):
        raise BackendError("tts.greektts15.snac_code_invalid", "SNAC code is outside its codebook", subsystem="tts")
    tokens: list[int] = []
    for frame in range(len(first)):
        raw = (
            first[frame],
            second[frame * 2],
            third[frame * 4],
            third[frame * 4 + 1],
            second[frame * 2 + 1],
            third[frame * 4 + 2],
            third[frame * 4 + 3],
        )
        tokens.extend(AUDIO_TOKEN_OFFSET + position * CODEBOOK_SIZE + code for position, code in enumerate(raw))
    return tokens


def build_orpheus_training_sequence(text_ids: Iterable[int], audio_tokens: Iterable[int]) -> tuple[list[int], list[int]]:
    """Build GreekTTS teacher-forcing IDs and completion-only labels.

    ``text_ids`` contains only the tokenizer's text payload (normally BOS plus
    text).  Control tokens are added explicitly so training and inference share
    the same protocol. The start-of-speech token is part of the learned
    completion; all human prompt positions are masked.
    """

    text = list(text_ids)
    audio = list(audio_tokens)
    if not text or not audio or len(audio) % 7:
        raise BackendError("tts.greektts15.training_tokens_invalid", "Training tokens are incomplete", subsystem="tts")
    if text[-1] == END_OF_TEXT:
        text = text[:-1]
    prefix = [START_OF_HUMAN, *text, END_OF_TEXT, END_OF_HUMAN]
    completion = [START_OF_SPEECH, *audio, END_OF_SPEECH]
    return prefix + completion, [-100] * len(prefix) + completion


def load_cached_split(
    plan_path: str | Path,
    cache_root: str | Path,
    tokenizer: object,
    split: str,
) -> list[tuple[list[int], int]]:
    """Materialize an MLX-LM dataset as ``(tokens, completion_offset)``."""

    if split not in {"train", "validation", "control"}:
        raise BackendError("tts.greektts15.split_invalid", "Training split is invalid", subsystem="tts")
    plan_file, cache = Path(plan_path).expanduser(), Path(cache_root).expanduser()
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise BackendError("tts.greektts15.tokenizer_invalid", "Tokenizer is invalid", subsystem="tts")
    result: list[tuple[list[int], int]] = []
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("split") != split:
            continue
        filename, text = sample.get("filename"), sample.get("text")
        if not isinstance(filename, str) or not isinstance(text, str):
            raise BackendError("tts.greektts15.plan_invalid", "Training sample is invalid", subsystem="tts")
        entry = cache / f"{Path(filename).stem}.json"
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError("tts.greektts15.cache_incomplete", "SNAC cache entry is unavailable", subsystem="tts") from exc
        if not _cache_payload_is_valid(payload, sample, payload.get("encoder") if isinstance(payload, dict) else {}):
            raise BackendError("tts.greektts15.cache_tokens_invalid", "SNAC cache entry is invalid", subsystem="tts")
        text_ids = list(encode(text, add_special_tokens=True))
        sequence, labels = build_orpheus_training_sequence(text_ids, payload["tokens"])
        if len(sequence) > MAX_TRAINING_SEQUENCE:
            raise BackendError("tts.greektts15.sequence_too_long", "Training sequence exceeds the safe bound", subsystem="tts")
        offset = next((index for index, label in enumerate(labels) if label != -100), len(labels))
        result.append((sequence, offset))
    if not result:
        raise BackendError("tts.greektts15.split_empty", "Training split is empty", subsystem="tts")
    return result


def run_gradient_memory_probe(
    model_root: str | Path,
    plan_path: str | Path,
    cache_root: str | Path,
) -> dict[str, object]:
    """Run one real forward/backward pass without applying an optimizer update."""

    import resource
    import time

    root = Path(model_root).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise BackendError("tts.greektts15.model_root_missing", "GreekTTS model root is unavailable", subsystem="tts")
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.tuner.trainer import default_loss
        from mlx_lm.tuner.utils import load_adapters
    except ImportError as exc:
        raise BackendError("tts.greektts15.dependencies_missing", "MLX training dependencies are missing", subsystem="tts") from exc

    previous = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")}
    os.environ.update({name: "1" for name in previous})
    try:
        started = time.perf_counter()
        model, tokenizer = load(str(root / "base"), lazy=False)
        model.freeze()
        model = load_adapters(model, str(root / "mlx-adapter"))
        mx.synchronize()
        load_seconds = time.perf_counter() - started
        trainable = tree_flatten(model.trainable_parameters())
        if len(trainable) != EXPECTED_LORA_GROUPS or any("lora_" not in name for name, _ in trainable):
            raise BackendError("tts.greektts15.trainable_surface_invalid", "Only the Greek LoRA adapter may be trainable", subsystem="tts")
        dataset = load_cached_split(plan_path, cache_root, tokenizer, "train")
        sequence, offset = min(dataset, key=lambda item: len(item[0]))
        batch = mx.array([sequence], dtype=mx.int32)
        lengths = mx.array([[offset, len(sequence)]], dtype=mx.int32)
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        probe_started = time.perf_counter()
        value_and_grad = nn.value_and_grad(model, default_loss)
        (loss, trained_tokens), gradients = value_and_grad(model, batch, lengths)
        mx.eval(loss, trained_tokens, gradients)
        probe_seconds = time.perf_counter() - probe_started
        if not float(loss.item()) > 0 or int(trained_tokens.item()) <= 0:
            raise BackendError("tts.greektts15.gradient_probe_invalid", "Gradient probe returned invalid metrics", subsystem="tts")
        gradient_groups = tree_flatten(gradients)
        if len(gradient_groups) != EXPECTED_LORA_GROUPS:
            raise BackendError("tts.greektts15.gradient_surface_invalid", "Gradient tree does not match the adapter", subsystem="tts")
        return {
            "schema": "jarvis.voice.greektts15-gradient-probe/1",
            "ready": True,
            "optimizer_update_applied": False,
            "sequence_tokens": len(sequence),
            "trained_tokens": int(trained_tokens.item()),
            "loss": round(float(loss.item()), 6),
            "trainable_groups": len(trainable),
            "trainable_parameters": sum(int(value.size) for _, value in trainable),
            "gradient_groups": len(gradient_groups),
            "model_load_seconds": round(load_seconds, 3),
            "probe_seconds": round(probe_seconds, 3),
            "peak_mlx_bytes": int(mx.get_peak_memory()),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_payload_is_valid(payload: object, sample: dict[str, object], signature: dict[str, object]) -> bool:
    if not isinstance(payload, dict):
        return False
    tokens = payload.get("tokens")
    return (
        payload.get("schema") == CACHE_SCHEMA
        and payload.get("filename") == sample.get("filename")
        and payload.get("audio_sha256") == sample.get("audio_sha256")
        and payload.get("encoder") == signature
        and isinstance(tokens, list)
        and bool(tokens)
        and len(tokens) % 7 == 0
        and _audio_tokens_are_valid(tokens)
    )


def _audio_tokens_are_valid(tokens: list[object]) -> bool:
    if not tokens or len(tokens) % 7:
        return False
    return all(
        isinstance(token, int)
        and AUDIO_TOKEN_OFFSET + position * CODEBOOK_SIZE <= token < AUDIO_TOKEN_OFFSET + (position + 1) * CODEBOOK_SIZE
        for index, token in enumerate(tokens)
        for position in (index % 7,)
    )


def cache_snac_samples(
    plan_path: str | Path,
    dataset_root: str | Path,
    cache_root: str | Path,
    encoder: SnacEncoder,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """Create a resumable, content-addressed SNAC cache from a verified plan."""

    plan_file = Path(plan_path).expanduser()
    dataset = Path(dataset_root).expanduser()
    cache = Path(cache_root).expanduser()
    if not plan_file.is_absolute() or not dataset.is_absolute() or not cache.is_absolute() or cache.is_symlink():
        raise BackendError("tts.greektts15.cache_path_invalid", "SNAC cache path is invalid", subsystem="tts")
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list) or not samples:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    selected = samples if limit is None else samples[:limit]
    if not selected or (limit is not None and limit < 1):
        raise BackendError("tts.greektts15.cache_limit_invalid", "SNAC cache limit is invalid", subsystem="tts")
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = reused = 0
    token_count = 0
    for sample in selected:
        if not isinstance(sample, dict) or not isinstance(sample.get("filename"), str):
            raise BackendError("tts.greektts15.plan_invalid", "Training sample is invalid", subsystem="tts")
        filename = str(sample["filename"])
        source = dataset / "audio" / filename
        if source.is_symlink() or not source.is_file() or source.parent != dataset / "audio":
            raise BackendError("tts.greektts15.dataset_audio_invalid", "F1 audio path is invalid", subsystem="tts")
        if _sha256(source) != sample.get("audio_sha256"):
            raise BackendError("tts.greektts15.dataset_audio_hash_invalid", "F1 audio changed after planning", subsystem="tts")
        destination = cache / f"{Path(filename).stem}.json"
        if destination.is_symlink():
            raise BackendError("tts.greektts15.cache_path_invalid", "SNAC cache contains a symlink", subsystem="tts")
        try:
            existing = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else None
        except (OSError, json.JSONDecodeError):
            existing = None
        if _cache_payload_is_valid(existing, sample, encoder.signature):
            tokens = existing["tokens"]
            reused += 1
        else:
            tokens = encoder.encode(source)
            payload = {
                "schema": CACHE_SCHEMA,
                "filename": filename,
                "audio_sha256": sample["audio_sha256"],
                "encoder": encoder.signature,
                "tokens": tokens,
            }
            if not _cache_payload_is_valid(payload, sample, encoder.signature):
                raise BackendError("tts.greektts15.cache_tokens_invalid", "SNAC encoder returned invalid tokens", subsystem="tts")
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
            temporary.replace(destination)
            encoded += 1
        token_count += len(tokens)
    return {
        "schema": CACHE_SCHEMA,
        "ready": True,
        "requested": len(selected),
        "encoded": encoded,
        "reused": reused,
        "audio_tokens": token_count,
        "encoder": encoder.signature,
    }


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
            "owned_greektts_token_collator": True,
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
