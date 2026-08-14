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
from array import array
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
WORD_FOCUS_SCHEMA = "jarvis.voice.greektts15-word-focus/1"
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
TARGETED_LORA_GROUPS = 364
ATTENTION_ONLY_LORA_GROUPS = 224
MLP_ONLY_LORA_GROUPS = 168
EOS_LOSS_WEIGHTS = {1, 16}
MAX_GRADIENT_PROBE_MLX_BYTES = 6 * 1024**3
PILOT_STEPS = 25
EXACT_CORE_STEPS = 100
GENERAL_STAGE_STEPS = 100
PILOT_VALIDATION_SAMPLES = 8
TRAINING_SEED = 10_001
PILOT_LEARNING_RATES = {2.5e-7, 5e-7, 1e-6, 5e-6, 1e-5}
TRAINING_TARGET_PROFILES = {"all", "freeze-output-layers", "attention-only", "mlp-only"}
TRAINING_SAMPLE_PROFILES = {
    "uniform",
    "jarvis-pronunciation",
    "jarvis-lexical",
    "jarvis-exact-core",
    "jarvis-general-stage",
    "jarvis-utterance-focus",
    "jarvis-operation-morphology",
    "jarvis-operation-morphology-mlp",
    "jarvis-word-focus",
    "jarvis-operation-focus",
    "jarvis-state-focus",
    "jarvis-importance-focus",
}
PRONUNCIATION_CURRICULUM_ROOTS = (
    ("system_state", "καταστα"),
    ("operation", "λειτουργ"),
    ("reminder", "θυμ"),
    ("importance", "σημαντικ"),
    ("work", "εργασ"),
)
PRONUNCIATION_TARGETED_STEPS = 15
LEXICAL_CURRICULUM_TARGETS = (
    ("system_state", "κατασταση", True),
    ("operation", "λειτουργουν", True),
    ("reminder", "θυμ", False),
    ("importance", "σημαντικεσ", True),
    ("work", "εργασιεσ", True),
)
EXACT_CORE_TARGETS = tuple(target for target in LEXICAL_CURRICULUM_TARGETS if target[0] != "reminder")
EXACT_CORE_TARGETED_STEPS = 60
UTTERANCE_FOCUS_TARGETS = ("λειτουργουν", "σημαντικεσ")
UTTERANCE_FOCUS_TARGETED_STEPS = 20
OPERATION_MORPHOLOGY_EXPOSURES = {
    "λειτουργει": 8,
    "λειτουργουν": 4,
    "other_operation_forms": 6,
    "importance_retention": 2,
}
WORD_FOCUS_CROPS = (
    ("katastasi", "Greek-F1-B2-51-22.wav", "κατάσταση", 3.92, 4.60, 0.972),
    ("leitourgoun", "Greek-F1-B2-51-06.wav", "λειτουργούν", 2.52, 3.24, 0.848),
    ("simantikes-a", "Greek-F1-B2-09-05.wav", "σημαντικές", 1.24, 1.86, 0.907),
    ("simantikes-b", "Greek-F1-C1-25-07.wav", "σημαντικές", 1.98, 2.62, 0.940),
    ("ergasies", "Greek-F1-C1-02-03.wav", "εργασίες", 0.34, 0.92, 0.845),
)
WORD_FOCUS_PADDING_SECONDS = 0.10
PHRASE_EDGE_PADDING_SECONDS = 0.03
PHRASE_GAP_SECONDS = 0.05
PHRASE_FADE_SECONDS = 0.01
WORD_FOCUS_TARGETED_STEPS = 15
OPERATION_FOCUS_TARGETED_STEPS = 10
STATE_FOCUS_TARGETED_STEPS = 15
IMPORTANCE_FOCUS_TARGETED_STEPS = 10
ADAPTER_BLEND_FACTORS = {0.25, 0.375, 0.5}
ASR_CANDIDATE_LABELS = {"baseline", "lr-5e-7", "lr-1e-6"}
PRONUNCIATION_ASR_CANDIDATE_LABELS = {
    "blend-25pct",
    "blend-37_5pct",
    "lr-2_5e-7",
    "target-freeze-output-layers",
    "attention-only",
}
QUALITY_ASR_CANDIDATE_LABELS = {"attention-only"}

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


def build_word_focus_bundle(
    plan_path: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Crop pinned real-F1 word examples with content-bound provenance."""

    plan_file = Path(plan_path).expanduser()
    dataset = Path(dataset_root).expanduser()
    output = Path(output_root).expanduser()
    if (
        not plan_file.is_absolute()
        or not dataset.is_absolute()
        or not output.is_absolute()
        or output.exists()
        or output.is_symlink()
    ):
        raise BackendError("tts.greektts15.word_focus_path_invalid", "Word-focus paths are invalid", subsystem="tts")
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    by_filename = {
        sample.get("filename"): sample
        for sample in samples
        if isinstance(sample, dict) and isinstance(sample.get("filename"), str)
    }

    audio_output = output / "audio"
    audio_output.mkdir(mode=0o700, parents=True)
    manifest_samples: list[dict[str, object]] = []
    for label, source_name, text, word_start, word_end, confidence in WORD_FOCUS_CROPS:
        source_sample = by_filename.get(source_name)
        source = dataset / "audio" / source_name
        if (
            not isinstance(source_sample, dict)
            or source_sample.get("split") != "train"
            or source.is_symlink()
            or not source.is_file()
            or source.parent != dataset / "audio"
            or _sha256(source) != source_sample.get("audio_sha256")
        ):
            raise BackendError("tts.greektts15.word_focus_source_invalid", "Word-focus source is invalid", subsystem="tts")
        with wave.open(str(source), "rb") as reader:
            if (
                reader.getnchannels() != EXPECTED_CHANNELS
                or reader.getsampwidth() != EXPECTED_SAMPLE_WIDTH
                or reader.getframerate() != EXPECTED_SAMPLE_RATE
            ):
                raise BackendError("tts.greektts15.dataset_audio_invalid", "F1 waveform format is invalid", subsystem="tts")
            total_frames = reader.getnframes()
            crop_start = max(0.0, word_start - WORD_FOCUS_PADDING_SECONDS)
            crop_end = min(total_frames / EXPECTED_SAMPLE_RATE, word_end + WORD_FOCUS_PADDING_SECONDS)
            start_frame = round(crop_start * EXPECTED_SAMPLE_RATE)
            end_frame = round(crop_end * EXPECTED_SAMPLE_RATE)
            if end_frame <= start_frame:
                raise BackendError("tts.greektts15.word_focus_bounds_invalid", "Word-focus bounds are invalid", subsystem="tts")
            reader.setpos(start_frame)
            frames = reader.readframes(end_frame - start_frame)
        filename = f"word-focus-{label}.wav"
        destination = audio_output / filename
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(EXPECTED_CHANNELS)
            writer.setsampwidth(EXPECTED_SAMPLE_WIDTH)
            writer.setframerate(EXPECTED_SAMPLE_RATE)
            writer.writeframes(frames)
        os.chmod(destination, 0o600)
        manifest_samples.append(
            {
                "filename": filename,
                "text": text,
                "split": "train",
                "corpus": "STOMA-F1-word-focus",
                "duration_seconds": round((end_frame - start_frame) / EXPECTED_SAMPLE_RATE, 6),
                "audio_sha256": _sha256(destination),
                "transcript_normalized": False,
                "source_filename": source_name,
                "source_audio_sha256": source_sample["audio_sha256"],
                "word_start_seconds": word_start,
                "word_end_seconds": word_end,
                "crop_start_seconds": round(start_frame / EXPECTED_SAMPLE_RATE, 6),
                "crop_end_seconds": round(end_frame / EXPECTED_SAMPLE_RATE, 6),
                "alignment_backend": "faster-whisper-base-word-timestamps",
                "alignment_confidence": confidence,
            }
        )
    bundle = {
        "schema": WORD_FOCUS_SCHEMA,
        "ready": True,
        "padding_seconds": WORD_FOCUS_PADDING_SECONDS,
        "samples": manifest_samples,
    }
    (output / "word-focus-manifest.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cache_plan = {"schema": PLAN_SCHEMA, "samples": manifest_samples}
    (output / "word-focus-plan.json").write_text(
        json.dumps(cache_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def build_word_focus_training_plan(
    base_plan_path: str | Path,
    word_focus_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Append approved real-F1 crops to the original plan without changing validation."""

    base_path = Path(base_plan_path).expanduser()
    focus_path = Path(word_focus_manifest_path).expanduser()
    output = Path(output_path).expanduser()
    if not base_path.is_absolute() or not focus_path.is_absolute() or not output.is_absolute() or output.exists():
        raise BackendError("tts.greektts15.word_focus_path_invalid", "Word-focus plan paths are invalid", subsystem="tts")
    try:
        base = json.loads(base_path.read_text(encoding="utf-8"))
        focus = json.loads(focus_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Word-focus plan input is unreadable", subsystem="tts") from exc
    base_samples = base.get("samples") if isinstance(base, dict) else None
    focus_samples = focus.get("samples") if isinstance(focus, dict) else None
    if (
        not isinstance(base, dict)
        or base.get("schema") != PLAN_SCHEMA
        or not isinstance(base_samples, list)
        or not isinstance(focus, dict)
        or focus.get("schema") != WORD_FOCUS_SCHEMA
        or focus.get("ready") is not True
        or not isinstance(focus_samples, list)
        or len(focus_samples) != len(WORD_FOCUS_CROPS)
        or any(sample.get("split") != "train" for sample in focus_samples if isinstance(sample, dict))
    ):
        raise BackendError("tts.greektts15.plan_invalid", "Word-focus plan input is invalid", subsystem="tts")
    filenames = [sample.get("filename") for sample in [*base_samples, *focus_samples] if isinstance(sample, dict)]
    if len(filenames) != len(set(filenames)):
        raise BackendError("tts.greektts15.plan_invalid", "Word-focus filenames collide", subsystem="tts")
    combined = dict(base)
    combined["samples"] = [*base_samples, *focus_samples]
    combined["word_focus"] = {
        "schema": WORD_FOCUS_SCHEMA,
        "samples": len(focus_samples),
        "manifest_sha256": _sha256(focus_path),
    }
    output.write_text(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return combined


def build_f1_phrase_candidates(
    word_focus_root: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Join approved real-F1 word crops into review-only phrase candidates."""

    focus = Path(word_focus_root).expanduser()
    output = Path(output_root).expanduser()
    if not focus.is_absolute() or not focus.is_dir() or focus.is_symlink() or not output.is_absolute() or output.exists():
        raise BackendError("tts.greektts15.phrase_path_invalid", "F1 phrase paths are invalid", subsystem="tts")
    try:
        manifest = json.loads((focus / "word-focus-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.phrase_source_invalid", "Word-focus evidence is unreadable", subsystem="tts") from exc
    samples = manifest.get("samples") if isinstance(manifest, dict) else None
    if manifest.get("schema") != WORD_FOCUS_SCHEMA or manifest.get("ready") is not True or not isinstance(samples, list):
        raise BackendError("tts.greektts15.phrase_source_invalid", "Word-focus evidence is invalid", subsystem="tts")
    by_name = {sample.get("filename"): sample for sample in samples if isinstance(sample, dict)}
    importance_names = ("word-focus-simantikes-a.wav", "word-focus-simantikes-b.wav")
    work_name = "word-focus-ergasies.wav"

    def read_verified(name: str) -> array[int]:
        evidence = by_name.get(name)
        path = focus / "audio" / name
        if (
            not isinstance(evidence, dict)
            or path.is_symlink()
            or not path.is_file()
            or _sha256(path) != evidence.get("audio_sha256")
        ):
            raise BackendError("tts.greektts15.phrase_source_invalid", "Approved word crop changed", subsystem="tts")
        with wave.open(str(path), "rb") as reader:
            if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, EXPECTED_SAMPLE_RATE):
                raise BackendError("tts.greektts15.phrase_source_invalid", "Approved word crop format changed", subsystem="tts")
            values = array("h")
            values.frombytes(reader.readframes(reader.getnframes()))
        trim = round((WORD_FOCUS_PADDING_SECONDS - PHRASE_EDGE_PADDING_SECONDS) * EXPECTED_SAMPLE_RATE)
        if len(values) <= trim * 2:
            raise BackendError("tts.greektts15.phrase_source_invalid", "Approved word crop is too short", subsystem="tts")
        return values[trim:-trim]

    def faded(values: array[int]) -> array[int]:
        result = array("h", values)
        fade = min(round(PHRASE_FADE_SECONDS * EXPECTED_SAMPLE_RATE), len(result) // 2)
        for index in range(fade):
            result[index] = round(result[index] * index / fade)
            result[-1 - index] = round(result[-1 - index] * index / fade)
        return result

    work = faded(read_verified(work_name))
    silence = array("h", [0]) * round(PHRASE_GAP_SECONDS * EXPECTED_SAMPLE_RATE)
    output.mkdir(mode=0o700, parents=True)
    results: list[dict[str, object]] = []
    for label, importance_name in zip(("a", "b"), importance_names, strict=True):
        importance = faded(read_verified(importance_name))
        waveform = importance + silence + work
        filename = f"f1-phrase-simantikes-ergasies-{label}.wav"
        destination = output / filename
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(EXPECTED_SAMPLE_RATE)
            writer.writeframes(waveform.tobytes())
        os.chmod(destination, 0o600)
        results.append(
            {
                "filename": filename,
                "text": "σημαντικές εργασίες",
                "audio_sha256": _sha256(destination),
                "duration_seconds": round(len(waveform) / EXPECTED_SAMPLE_RATE, 6),
                "importance_source": importance_name,
                "work_source": work_name,
            }
        )
    report = {
        "schema": "jarvis.voice.greektts15-f1-phrase-candidates/1",
        "ready": True,
        "review_only": True,
        "training_allowed": False,
        "edge_padding_seconds": PHRASE_EDGE_PADDING_SECONDS,
        "gap_seconds": PHRASE_GAP_SECONDS,
        "fade_seconds": PHRASE_FADE_SECONDS,
        "samples": results,
    }
    (output / "phrase-candidates-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


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


def build_pronunciation_curriculum_order(
    plan_path: str | Path,
    *,
    steps: int = PILOT_STEPS,
) -> tuple[list[int], dict[str, object]]:
    """Select a bounded mix of real F1 pronunciation examples and general speech.

    Indices refer to the train-only order returned by :func:`load_cached_split`.
    No transcript is synthesized or relabelled: the curriculum only reorders
    existing, immutable F1 samples from the training split.
    """

    import random

    try:
        plan = json.loads(Path(plan_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    train_samples = [sample for sample in samples if isinstance(sample, dict) and sample.get("split") == "train"]
    if len(train_samples) < steps or any(not isinstance(sample.get("text"), str) for sample in train_samples):
        raise BackendError("tts.greektts15.plan_invalid", "Training samples are incomplete", subsystem="tts")

    normalized = [" ".join(_normalized_words(sample["text"])) for sample in train_samples]
    candidates = {
        label: [index for index, text in enumerate(normalized) if root in text]
        for label, root in PRONUNCIATION_CURRICULUM_ROOTS
    }
    randomizer = random.Random(TRAINING_SEED)
    for indices in candidates.values():
        randomizer.shuffle(indices)

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_counts = {label: 0 for label, _ in PRONUNCIATION_CURRICULUM_ROOTS}
    while len(selected) < min(PRONUNCIATION_TARGETED_STEPS, steps):
        progressed = False
        for label, _ in PRONUNCIATION_CURRICULUM_ROOTS:
            while candidates[label] and candidates[label][0] in selected_set:
                candidates[label].pop(0)
            if candidates[label] and len(selected) < min(PRONUNCIATION_TARGETED_STEPS, steps):
                index = candidates[label].pop(0)
                selected.append(index)
                selected_set.add(index)
                selected_counts[label] += 1
                progressed = True
        if not progressed:
            break

    general = [index for index in range(len(train_samples)) if index not in selected_set]
    randomizer.shuffle(general)
    selected.extend(general[: steps - len(selected)])
    if len(selected) != steps or len(set(selected)) != steps:
        raise BackendError("tts.greektts15.curriculum_invalid", "Pronunciation curriculum is incomplete", subsystem="tts")
    filenames = [train_samples[index].get("filename") for index in selected]
    return selected, {
        "profile": "jarvis-pronunciation",
        "targeted_steps": sum(selected_counts.values()),
        "general_steps": steps - sum(selected_counts.values()),
        "target_counts": selected_counts,
        "sample_filenames": filenames,
    }


def build_lexical_curriculum_order(
    plan_path: str | Path,
    *,
    steps: int = PILOT_STEPS,
) -> tuple[list[int], dict[str, object]]:
    """Repeat exact real F1 lexical examples in a bounded MLP-only curriculum."""

    import random

    try:
        plan = json.loads(Path(plan_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    train_samples = [sample for sample in samples if isinstance(sample, dict) and sample.get("split") == "train"]
    if len(train_samples) < steps or any(not isinstance(sample.get("text"), str) for sample in train_samples):
        raise BackendError("tts.greektts15.plan_invalid", "Training samples are incomplete", subsystem="tts")

    words = [set(_normalized_words(sample["text"])) for sample in train_samples]
    normalized = [" ".join(_normalized_words(sample["text"])) for sample in train_samples]
    candidates: dict[str, list[int]] = {}
    for label, needle, exact_word in LEXICAL_CURRICULUM_TARGETS:
        matches = [
            index
            for index in range(len(train_samples))
            if (needle in words[index] if exact_word else needle in normalized[index])
        ]
        if not matches:
            raise BackendError("tts.greektts15.curriculum_invalid", "Lexical F1 coverage is incomplete", subsystem="tts")
        candidates[label] = matches

    randomizer = random.Random(TRAINING_SEED)
    for indices in candidates.values():
        randomizer.shuffle(indices)
    selected: list[int] = []
    selected_counts = {label: 0 for label, _, _ in LEXICAL_CURRICULUM_TARGETS}
    rounds = PRONUNCIATION_TARGETED_STEPS // len(LEXICAL_CURRICULUM_TARGETS)
    for round_index in range(rounds):
        for label, _, _ in LEXICAL_CURRICULUM_TARGETS:
            indices = candidates[label]
            selected.append(indices[round_index % len(indices)])
            selected_counts[label] += 1

    targeted_set = {index for indices in candidates.values() for index in indices}
    general = [index for index in range(len(train_samples)) if index not in targeted_set]
    randomizer.shuffle(general)
    selected.extend(general[: steps - len(selected)])
    if len(selected) != steps:
        raise BackendError("tts.greektts15.curriculum_invalid", "Lexical curriculum is incomplete", subsystem="tts")
    return selected, {
        "profile": "jarvis-lexical",
        "targeted_steps": sum(selected_counts.values()),
        "general_steps": steps - sum(selected_counts.values()),
        "target_counts": selected_counts,
        "sample_filenames": [train_samples[index].get("filename") for index in selected],
        "repeated_exact_examples": True,
    }


def build_exact_core_curriculum_order(
    plan_path: str | Path,
    *,
    steps: int = EXACT_CORE_STEPS,
) -> tuple[list[int], dict[str, object]]:
    """Build a longer exact-word curriculum with no synthetic reminder proxy."""

    import random

    if steps != EXACT_CORE_STEPS:
        raise BackendError("tts.greektts15.curriculum_invalid", "Exact-core step count is invalid", subsystem="tts")
    try:
        plan = json.loads(Path(plan_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    train_samples = [sample for sample in samples if isinstance(sample, dict) and sample.get("split") == "train"]
    if len(train_samples) < steps or any(not isinstance(sample.get("text"), str) for sample in train_samples):
        raise BackendError("tts.greektts15.plan_invalid", "Training samples are incomplete", subsystem="tts")

    words = [set(_normalized_words(sample["text"])) for sample in train_samples]
    candidates: dict[str, list[int]] = {}
    for label, needle, exact_word in EXACT_CORE_TARGETS:
        if not exact_word:
            raise BackendError("tts.greektts15.curriculum_invalid", "Exact-core target is not exact", subsystem="tts")
        matches = [index for index, sample_words in enumerate(words) if needle in sample_words]
        if not matches:
            raise BackendError("tts.greektts15.curriculum_invalid", "Exact F1 coverage is incomplete", subsystem="tts")
        candidates[label] = matches

    randomizer = random.Random(TRAINING_SEED)
    for indices in candidates.values():
        randomizer.shuffle(indices)
    selected: list[int] = []
    selected_counts = {label: 0 for label, _, _ in EXACT_CORE_TARGETS}
    rounds = EXACT_CORE_TARGETED_STEPS // len(EXACT_CORE_TARGETS)
    for round_index in range(rounds):
        for label, _, _ in EXACT_CORE_TARGETS:
            indices = candidates[label]
            selected.append(indices[round_index % len(indices)])
            selected_counts[label] += 1

    targeted_set = {index for indices in candidates.values() for index in indices}
    general = [index for index in range(len(train_samples)) if index not in targeted_set]
    randomizer.shuffle(general)
    selected.extend(general[: steps - len(selected)])
    if len(selected) != steps:
        raise BackendError("tts.greektts15.curriculum_invalid", "Exact-core curriculum is incomplete", subsystem="tts")
    return selected, {
        "profile": "jarvis-exact-core",
        "targeted_steps": EXACT_CORE_TARGETED_STEPS,
        "general_steps": steps - EXACT_CORE_TARGETED_STEPS,
        "target_counts": selected_counts,
        "sample_filenames": [train_samples[index].get("filename") for index in selected],
        "repeated_exact_examples": True,
        "reminder_proxy_excluded": True,
    }


def build_general_stage_curriculum_order(
    plan_path: str | Path,
    *,
    stage_index: int = 0,
    steps: int = GENERAL_STAGE_STEPS,
) -> tuple[list[int], dict[str, object]]:
    """Select one deterministic, non-overlapping stage of full F1 utterances."""

    import random

    if steps != GENERAL_STAGE_STEPS or stage_index < 0:
        raise BackendError("tts.greektts15.curriculum_invalid", "General stage configuration is invalid", subsystem="tts")
    try:
        plan = json.loads(Path(plan_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    train_samples = [sample for sample in samples if isinstance(sample, dict) and sample.get("split") == "train"]
    general = [
        index
        for index, sample in enumerate(train_samples)
        if sample.get("corpus") != "STOMA-F1-word-focus"
        and isinstance(sample.get("text"), str)
        and sample["text"].strip()
    ]
    randomizer = random.Random(TRAINING_SEED)
    randomizer.shuffle(general)
    start = stage_index * steps
    selected = general[start : start + steps]
    if len(selected) != steps:
        raise BackendError("tts.greektts15.curriculum_invalid", "General F1 stage is incomplete", subsystem="tts")
    filenames = [train_samples[index].get("filename") for index in selected]
    if len(set(selected)) != steps or len(set(filenames)) != steps:
        raise BackendError("tts.greektts15.curriculum_invalid", "General F1 stage contains duplicates", subsystem="tts")
    return selected, {
        "profile": "jarvis-general-stage",
        "stage_index": stage_index,
        "general_steps": steps,
        "unique_full_utterances": True,
        "word_focus_crops_excluded": True,
        "sample_filenames": filenames,
    }


def build_utterance_focus_curriculum_order(
    plan_path: str | Path,
    *,
    steps: int = PILOT_STEPS,
) -> tuple[list[int], dict[str, object]]:
    """Repeat exact target words only inside their original complete F1 utterances."""

    import random

    if steps != PILOT_STEPS:
        raise BackendError("tts.greektts15.curriculum_invalid", "Utterance-focus step count is invalid", subsystem="tts")
    try:
        plan = json.loads(Path(plan_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    train_samples = [sample for sample in samples if isinstance(sample, dict) and sample.get("split") == "train"]
    words = [set(_normalized_words(str(sample.get("text", "")))) for sample in train_samples]
    matches = {
        target: [
            index
            for index, sample_words in enumerate(words)
            if target in sample_words and train_samples[index].get("corpus") != "STOMA-F1-word-focus"
        ]
        for target in UTTERANCE_FOCUS_TARGETS
    }
    if len(matches["λειτουργουν"]) != 1 or len(matches["σημαντικεσ"]) != 2:
        raise BackendError("tts.greektts15.curriculum_invalid", "Exact word coverage changed", subsystem="tts")
    selected: list[int] = []
    for round_index in range(UTTERANCE_FOCUS_TARGETED_STEPS // 2):
        selected.append(matches["λειτουργουν"][0])
        selected.append(matches["σημαντικεσ"][round_index % 2])
    targeted = {index for indices in matches.values() for index in indices}
    general = [
        index
        for index, sample in enumerate(train_samples)
        if index not in targeted and sample.get("corpus") != "STOMA-F1-word-focus"
    ]
    randomizer = random.Random(TRAINING_SEED)
    randomizer.shuffle(general)
    selected.extend(general[: steps - len(selected)])
    if len(selected) != steps:
        raise BackendError("tts.greektts15.curriculum_invalid", "Utterance-focus curriculum is incomplete", subsystem="tts")
    return selected, {
        "profile": "jarvis-utterance-focus",
        "targeted_steps": UTTERANCE_FOCUS_TARGETED_STEPS,
        "general_steps": steps - UTTERANCE_FOCUS_TARGETED_STEPS,
        "target_counts": {"λειτουργούν": 10, "σημαντικές": 10},
        "complete_original_utterances_only": True,
        "word_focus_crops_excluded": True,
        "sample_filenames": [train_samples[index].get("filename") for index in selected],
    }


def build_operation_morphology_curriculum_order(
    plan_path: str | Path,
    *,
    profile: str = "jarvis-operation-morphology",
    steps: int = PILOT_STEPS,
) -> tuple[list[int], dict[str, object]]:
    """Cover the operation root across real inflections while retaining F1 quality."""

    import random

    if profile not in {"jarvis-operation-morphology", "jarvis-operation-morphology-mlp"} or steps != PILOT_STEPS:
        raise BackendError("tts.greektts15.curriculum_invalid", "Operation morphology step count is invalid", subsystem="tts")
    try:
        plan = json.loads(Path(plan_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    train_samples = [sample for sample in samples if isinstance(sample, dict) and sample.get("split") == "train"]
    words = [set(_normalized_words(str(sample.get("text", "")))) for sample in train_samples]
    eligible = [
        index for index, sample in enumerate(train_samples) if sample.get("corpus") != "STOMA-F1-word-focus"
    ]
    exact_singular = [index for index in eligible if "λειτουργει" in words[index]]
    exact_plural = [index for index in eligible if "λειτουργουν" in words[index]]
    importance = [index for index in eligible if "σημαντικεσ" in words[index]]
    exact_set = set(exact_singular + exact_plural)
    variants = [
        index
        for index in eligible
        if index not in exact_set and any(word.startswith("λειτουργ") or word.startswith("πρωτολειτουργ") for word in words[index])
    ]
    if len(exact_singular) != 1 or len(exact_plural) != 1 or len(importance) != 2 or len(variants) < 6:
        raise BackendError("tts.greektts15.curriculum_invalid", "Operation morphology coverage changed", subsystem="tts")
    randomizer = random.Random(TRAINING_SEED)
    randomizer.shuffle(variants)
    variants = variants[: OPERATION_MORPHOLOGY_EXPOSURES["other_operation_forms"]]
    selected: list[int] = []
    for round_index in range(OPERATION_MORPHOLOGY_EXPOSURES["λειτουργει"]):
        selected.append(exact_singular[0])
        if round_index < OPERATION_MORPHOLOGY_EXPOSURES["λειτουργουν"]:
            selected.append(exact_plural[0])
        if round_index < len(variants):
            selected.append(variants[round_index])
        if round_index < OPERATION_MORPHOLOGY_EXPOSURES["importance_retention"]:
            selected.append(importance[round_index])
    targeted = set(exact_singular + exact_plural + importance + variants)
    general = [index for index in eligible if index not in targeted]
    randomizer.shuffle(general)
    selected.extend(general[: steps - len(selected)])
    if len(selected) != steps:
        raise BackendError("tts.greektts15.curriculum_invalid", "Operation morphology curriculum is incomplete", subsystem="tts")
    return selected, {
        "profile": profile,
        "targeted_steps": sum(OPERATION_MORPHOLOGY_EXPOSURES.values()),
        "general_steps": steps - sum(OPERATION_MORPHOLOGY_EXPOSURES.values()),
        "target_counts": dict(OPERATION_MORPHOLOGY_EXPOSURES),
        "complete_original_utterances_only": True,
        "word_focus_crops_excluded": True,
        "sample_filenames": [train_samples[index].get("filename") for index in selected],
    }


def build_word_focus_curriculum_order(
    plan_path: str | Path,
    *,
    steps: int = PILOT_STEPS,
) -> tuple[list[int], dict[str, object]]:
    """Repeat human-approved real word crops and retain ten general F1 samples."""

    import random

    if steps != PILOT_STEPS:
        raise BackendError("tts.greektts15.curriculum_invalid", "Word-focus step count is invalid", subsystem="tts")
    try:
        plan = json.loads(Path(plan_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    train_samples = [sample for sample in samples if isinstance(sample, dict) and sample.get("split") == "train"]
    focus_indices = [
        index for index, sample in enumerate(train_samples) if sample.get("corpus") == "STOMA-F1-word-focus"
    ]
    if len(focus_indices) != len(WORD_FOCUS_CROPS):
        raise BackendError("tts.greektts15.curriculum_invalid", "Approved word-focus samples are incomplete", subsystem="tts")
    selected = [focus_indices[round_index % len(focus_indices)] for round_index in range(WORD_FOCUS_TARGETED_STEPS)]
    general = [index for index, sample in enumerate(train_samples) if sample.get("corpus") != "STOMA-F1-word-focus"]
    randomizer = random.Random(TRAINING_SEED)
    randomizer.shuffle(general)
    selected.extend(general[: steps - len(selected)])
    return selected, {
        "profile": "jarvis-word-focus",
        "targeted_steps": WORD_FOCUS_TARGETED_STEPS,
        "general_steps": steps - WORD_FOCUS_TARGETED_STEPS,
        "focus_samples": len(focus_indices),
        "sample_filenames": [train_samples[index].get("filename") for index in selected],
        "human_crop_approval_required": True,
    }


def build_single_word_focus_curriculum_order(
    plan_path: str | Path,
    profile: str,
    *,
    steps: int = PILOT_STEPS,
) -> tuple[list[int], dict[str, object]]:
    """Isolate one approved word crop so unrelated phrases cannot regress."""

    import random

    settings = {
        "jarvis-operation-focus": ("word-focus-leitourgoun.wav", OPERATION_FOCUS_TARGETED_STEPS),
        "jarvis-state-focus": ("word-focus-katastasi.wav", STATE_FOCUS_TARGETED_STEPS),
    }
    if profile not in settings or steps != PILOT_STEPS:
        raise BackendError("tts.greektts15.curriculum_invalid", "Single-word focus profile is invalid", subsystem="tts")
    try:
        plan = json.loads(Path(plan_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    train_samples = [sample for sample in samples if isinstance(sample, dict) and sample.get("split") == "train"]
    filename, targeted_steps = settings[profile]
    matches = [index for index, sample in enumerate(train_samples) if sample.get("filename") == filename]
    if len(matches) != 1 or train_samples[matches[0]].get("corpus") != "STOMA-F1-word-focus":
        raise BackendError("tts.greektts15.curriculum_invalid", "Single-word F1 crop is unavailable", subsystem="tts")
    focus_index = matches[0]
    selected = [focus_index] * targeted_steps
    general = [
        index for index, sample in enumerate(train_samples) if sample.get("corpus") != "STOMA-F1-word-focus"
    ]
    randomizer = random.Random(TRAINING_SEED)
    randomizer.shuffle(general)
    selected.extend(general[: steps - targeted_steps])
    return selected, {
        "profile": profile,
        "targeted_steps": targeted_steps,
        "general_steps": steps - targeted_steps,
        "focus_filename": filename,
        "unrelated_word_crops_excluded": True,
        "sample_filenames": [train_samples[index].get("filename") for index in selected],
    }


def build_importance_focus_curriculum_order(
    plan_path: str | Path,
    *,
    steps: int = PILOT_STEPS,
) -> tuple[list[int], dict[str, object]]:
    """Use only the two approved real pronunciations of ``σημαντικές``."""

    import random

    if steps != PILOT_STEPS:
        raise BackendError("tts.greektts15.curriculum_invalid", "Importance-focus step count is invalid", subsystem="tts")
    try:
        plan = json.loads(Path(plan_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is unreadable", subsystem="tts") from exc
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or not isinstance(samples, list):
        raise BackendError("tts.greektts15.plan_invalid", "Training plan is invalid", subsystem="tts")
    train_samples = [sample for sample in samples if isinstance(sample, dict) and sample.get("split") == "train"]
    filenames = ("word-focus-simantikes-a.wav", "word-focus-simantikes-b.wav")
    focus_indices = [index for index, sample in enumerate(train_samples) if sample.get("filename") in filenames]
    if len(focus_indices) != len(filenames) or any(
        train_samples[index].get("corpus") != "STOMA-F1-word-focus" for index in focus_indices
    ):
        raise BackendError("tts.greektts15.curriculum_invalid", "Importance F1 crops are unavailable", subsystem="tts")
    selected = [focus_indices[index % len(focus_indices)] for index in range(IMPORTANCE_FOCUS_TARGETED_STEPS)]
    general = [
        index for index, sample in enumerate(train_samples) if sample.get("corpus") != "STOMA-F1-word-focus"
    ]
    randomizer = random.Random(TRAINING_SEED)
    randomizer.shuffle(general)
    selected.extend(general[: steps - len(selected)])
    return selected, {
        "profile": "jarvis-importance-focus",
        "targeted_steps": IMPORTANCE_FOCUS_TARGETED_STEPS,
        "general_steps": steps - IMPORTANCE_FOCUS_TARGETED_STEPS,
        "focus_filenames": list(filenames),
        "unrelated_word_crops_excluded": True,
        "sample_filenames": [train_samples[index].get("filename") for index in selected],
    }


def run_gradient_memory_probe(
    model_root: str | Path,
    plan_path: str | Path,
    cache_root: str | Path,
    *,
    gradient_checkpointing: bool = True,
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
        from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
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
        if gradient_checkpointing:
            grad_checkpoint(model.layers[0])
        mx.synchronize()
        load_seconds = time.perf_counter() - started
        trainable = tree_flatten(model.trainable_parameters())
        if len(trainable) != EXPECTED_LORA_GROUPS or any("lora_" not in name for name, _ in trainable):
            raise BackendError("tts.greektts15.trainable_surface_invalid", "Only the Greek LoRA adapter may be trainable", subsystem="tts")
        dataset = load_cached_split(plan_path, cache_root, tokenizer, "train")
        # Cover the worst activation shape allowed by the bounded plan.
        sequence, offset = max(dataset, key=lambda item: len(item[0]))
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
            "sequence_selection": "longest_train_sample",
            "gradient_checkpointing": gradient_checkpointing,
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


def _validated_gradient_probe(report_path: str | Path) -> dict[str, object]:
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.gradient_gate_missing", "Gradient probe report is unavailable", subsystem="tts") from exc
    if not (
        isinstance(report, dict)
        and report.get("schema") == "jarvis.voice.greektts15-gradient-probe/1"
        and report.get("ready") is True
        and report.get("optimizer_update_applied") is False
        and report.get("trainable_groups") == EXPECTED_LORA_GROUPS
        and report.get("gradient_groups") == EXPECTED_LORA_GROUPS
        and report.get("gradient_checkpointing") is True
        and report.get("sequence_selection") == "longest_train_sample"
        and isinstance(report.get("peak_mlx_bytes"), int)
        and 0 < report["peak_mlx_bytes"] <= MAX_GRADIENT_PROBE_MLX_BYTES
    ):
        raise BackendError("tts.greektts15.gradient_gate_failed", "Gradient probe did not pass the memory gate", subsystem="tts")
    return report


def run_optimizer_smoke(
    model_root: str | Path,
    plan_path: str | Path,
    cache_root: str | Path,
    gradient_report: str | Path,
    output_directory: str | Path,
    *,
    learning_rate: float = 5e-6,
) -> dict[str, object]:
    """Apply exactly one AdamW update and save a standalone adapter checkpoint."""

    import resource
    import shutil
    import time

    if learning_rate != 5e-6:
        raise BackendError("tts.greektts15.smoke_lr_invalid", "Optimizer smoke must use the qualified learning rate", subsystem="tts")
    probe = _validated_gradient_probe(gradient_report)
    root, output = Path(model_root).expanduser(), Path(output_directory).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise BackendError("tts.greektts15.model_root_missing", "GreekTTS model root is unavailable", subsystem="tts")
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise BackendError("tts.greektts15.smoke_output_invalid", "Optimizer smoke output must be a new absolute path", subsystem="tts")
    output.mkdir(mode=0o700, parents=True)
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.tuner.trainer import default_loss
        from mlx_lm.tuner.utils import load_adapters
    except ImportError as exc:
        raise BackendError("tts.greektts15.dependencies_missing", "MLX training dependencies are missing", subsystem="tts") from exc

    previous = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")}
    os.environ.update({name: "1" for name in previous})
    try:
        model, tokenizer = load(str(root / "base"), lazy=False)
        model.freeze()
        model = load_adapters(model, str(root / "mlx-adapter"))
        trainable = tree_flatten(model.trainable_parameters())
        if len(trainable) != EXPECTED_LORA_GROUPS or any("lora_" not in name for name, _ in trainable):
            raise BackendError("tts.greektts15.trainable_surface_invalid", "Only the Greek LoRA adapter may be trainable", subsystem="tts")
        dataset = load_cached_split(plan_path, cache_root, tokenizer, "train")
        sequence, offset = min(dataset, key=lambda item: len(item[0]))
        batch = mx.array([sequence], dtype=mx.int32)
        lengths = mx.array([[offset, len(sequence)]], dtype=mx.int32)
        optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
        value_and_grad = nn.value_and_grad(model, default_loss)
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        started = time.perf_counter()
        (loss_before, trained_tokens), gradients = value_and_grad(model, batch, lengths)
        optimizer.update(model, gradients)
        mx.eval(model.state, optimizer.state, loss_before, trained_tokens)
        loss_after, after_tokens = default_loss(model, batch, lengths)
        mx.eval(loss_after, after_tokens)
        elapsed = time.perf_counter() - started
        adapter_file = output / "adapters.safetensors"
        mx.save_safetensors(str(adapter_file), dict(tree_flatten(model.trainable_parameters())), metadata={"format": "mlx"})
        shutil.copyfile(root / "mlx-adapter" / "adapter_config.json", output / "adapter_config.json")
        source_hash = _sha256(root / "mlx-adapter" / "adapters.safetensors")
        output_hash = _sha256(adapter_file)
        if source_hash == output_hash:
            raise BackendError("tts.greektts15.optimizer_update_missing", "Optimizer did not change the adapter checkpoint", subsystem="tts")
        report = {
            "schema": "jarvis.voice.greektts15-optimizer-smoke/1",
            "ready": True,
            "optimizer": "AdamW",
            "optimizer_updates": 1,
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "sequence_tokens": len(sequence),
            "trained_tokens": int(trained_tokens.item()),
            "loss_before": round(float(loss_before.item()), 6),
            "loss_after": round(float(loss_after.item()), 6),
            "trainable_groups": len(trainable),
            "trainable_parameters": sum(int(value.size) for _, value in trainable),
            "elapsed_seconds": round(elapsed, 3),
            "peak_mlx_bytes": int(mx.get_peak_memory()),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "source_adapter_sha256": source_hash,
            "output_adapter_sha256": output_hash,
            "gradient_gate_peak_mlx_bytes": probe["peak_mlx_bytes"],
            "base_frozen": True,
            "adapter_merged": False,
        }
        (output / "training_manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_training_pilot(
    model_root: str | Path,
    plan_path: str | Path,
    cache_root: str | Path,
    gradient_report: str | Path,
    output_directory: str | Path,
    *,
    learning_rate: float,
    steps: int = PILOT_STEPS,
    target_profile: str = "all",
    eos_loss_weight: int = 1,
    sample_profile: str = "uniform",
    source_adapter_directory: str | Path | None = None,
    general_stage_index: int = 0,
) -> dict[str, object]:
    """Train one deterministic pilot branch from the immutable Greek adapter."""

    import random
    import resource
    import shutil
    import time

    if (
        learning_rate not in PILOT_LEARNING_RATES
        or (
            steps != PILOT_STEPS
            and not (
                steps == EXACT_CORE_STEPS
                and sample_profile in {"jarvis-exact-core", "jarvis-general-stage"}
            )
        )
        or target_profile not in TRAINING_TARGET_PROFILES
        or sample_profile not in TRAINING_SAMPLE_PROFILES
        or (target_profile in {"freeze-output-layers", "attention-only"} and learning_rate != 1e-6)
        or (target_profile == "mlp-only" and learning_rate != 2.5e-7)
        or eos_loss_weight not in EOS_LOSS_WEIGHTS
        or (eos_loss_weight != 1 and target_profile != "attention-only")
        or (sample_profile == "jarvis-pronunciation" and target_profile != "attention-only")
        or (sample_profile == "jarvis-lexical" and target_profile != "mlp-only")
        or (sample_profile == "jarvis-exact-core" and target_profile != "mlp-only")
        or (
            sample_profile == "jarvis-general-stage"
            and (target_profile != "all" or learning_rate != 2.5e-7 or source_adapter_directory is None)
        )
        or (
            sample_profile == "jarvis-utterance-focus"
            and (target_profile != "all" or learning_rate != 2.5e-7 or source_adapter_directory is None)
        )
        or (
            sample_profile == "jarvis-operation-morphology"
            and (target_profile != "all" or learning_rate != 2.5e-7 or source_adapter_directory is None)
        )
        or (
            sample_profile == "jarvis-operation-morphology-mlp"
            and (target_profile != "mlp-only" or learning_rate != 2.5e-7 or source_adapter_directory is None)
        )
        or (sample_profile != "jarvis-general-stage" and general_stage_index != 0)
        or general_stage_index < 0
        or (sample_profile == "jarvis-word-focus" and target_profile != "mlp-only")
        or (sample_profile == "jarvis-operation-focus" and target_profile != "mlp-only")
        or (sample_profile == "jarvis-state-focus" and target_profile != "attention-only")
        or (sample_profile == "jarvis-importance-focus" and target_profile != "mlp-only")
        or (
            source_adapter_directory is not None
            and sample_profile
            not in {
                "jarvis-general-stage",
                "jarvis-utterance-focus",
                "jarvis-operation-morphology",
                "jarvis-operation-morphology-mlp",
                "jarvis-word-focus",
                "jarvis-operation-focus",
                "jarvis-state-focus",
                "jarvis-importance-focus",
            }
        )
    ):
        raise BackendError("tts.greektts15.pilot_config_invalid", "Training pilot configuration is not approved", subsystem="tts")
    gate = _validated_gradient_probe(gradient_report)
    root, output = Path(model_root).expanduser(), Path(output_directory).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise BackendError("tts.greektts15.model_root_missing", "GreekTTS model root is unavailable", subsystem="tts")
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise BackendError("tts.greektts15.pilot_output_invalid", "Training pilot output must be a new absolute path", subsystem="tts")
    baseline_adapter = root / "mlx-adapter"
    source_adapter = baseline_adapter if source_adapter_directory is None else Path(source_adapter_directory).expanduser()
    if (
        not source_adapter.is_absolute()
        or not source_adapter.is_dir()
        or source_adapter.is_symlink()
        or not (source_adapter / "adapters.safetensors").is_file()
        or not (source_adapter / "adapter_config.json").is_file()
    ):
        raise BackendError("tts.greektts15.training_source_invalid", "Training source adapter is invalid", subsystem="tts")
    if source_adapter != baseline_adapter:
        try:
            source_config = json.loads((source_adapter / "adapter_config.json").read_text(encoding="utf-8"))
            baseline_config = json.loads((baseline_adapter / "adapter_config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError("tts.greektts15.training_source_invalid", "Training source evidence is unreadable", subsystem="tts") from exc
        source_hash = _sha256(source_adapter / "adapters.safetensors")
        baseline_hash = _sha256(baseline_adapter / "adapters.safetensors")
        training_manifest_path = source_adapter / "training_manifest.json"
        composition_manifest_path = source_adapter / "composition_manifest.json"
        try:
            if training_manifest_path.is_file():
                source_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
                lineage_valid = (
                    source_manifest.get("schema") == "jarvis.voice.greektts15-training-pilot/1"
                    and source_manifest.get("ready") is True
                    and source_manifest.get("output_adapter_sha256") == source_hash
                    and (
                        source_manifest.get("immutable_adapter_sha256") == baseline_hash
                        or source_manifest.get("source_adapter_sha256") == baseline_hash
                    )
                )
            else:
                source_manifest = json.loads(composition_manifest_path.read_text(encoding="utf-8"))
                lineage_valid = (
                    source_manifest.get("schema") == "jarvis.voice.greektts15-disjoint-composition/1"
                    and source_manifest.get("ready") is True
                    and source_manifest.get("output_adapter_sha256") == source_hash
                    and source_manifest.get("immutable_adapter_sha256") == baseline_hash
                    and source_manifest.get("averaging_used") is False
                )
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError("tts.greektts15.training_source_invalid", "Training source evidence is unreadable", subsystem="tts") from exc
        if not lineage_valid or source_config != baseline_config:
            raise BackendError("tts.greektts15.training_source_invalid", "Training source lineage is invalid", subsystem="tts")
    output.mkdir(mode=0o700, parents=True)
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
        from mlx_lm.tuner.utils import load_adapters
    except ImportError as exc:
        raise BackendError("tts.greektts15.dependencies_missing", "MLX training dependencies are missing", subsystem="tts") from exc

    def evaluate(model: object, samples: list[tuple[list[int], int]]) -> float:
        total_loss = 0.0
        total_tokens = 0
        model.eval()
        for sequence, offset in samples:
            batch = mx.array([sequence], dtype=mx.int32)
            lengths = mx.array([[offset, len(sequence)]], dtype=mx.int32)
            loss, tokens = default_loss(model, batch, lengths)
            mx.eval(loss, tokens)
            count = int(tokens.item())
            total_loss += float(loss.item()) * count
            total_tokens += count
        model.train()
        return total_loss / total_tokens

    previous = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")}
    os.environ.update({name: "1" for name in previous})
    try:
        mx.random.seed(TRAINING_SEED)
        model, tokenizer = load(str(root / "base"), lazy=False)
        model.freeze()
        model = load_adapters(model, str(source_adapter))
        frozen_layers: list[int] = []
        if target_profile == "freeze-output-layers":
            frozen_layers = [26, 27]
            for layer_index in frozen_layers:
                model.layers[layer_index].freeze()
        elif target_profile == "attention-only":
            for layer in model.layers:
                layer.mlp.freeze()
        elif target_profile == "mlp-only":
            for layer in model.layers:
                layer.self_attn.freeze()
        trainable = tree_flatten(model.trainable_parameters())
        expected_trainable_groups = {
            "all": EXPECTED_LORA_GROUPS,
            "freeze-output-layers": TARGETED_LORA_GROUPS,
            "attention-only": ATTENTION_ONLY_LORA_GROUPS,
            "mlp-only": MLP_ONLY_LORA_GROUPS,
        }[target_profile]
        if len(trainable) != expected_trainable_groups or any("lora_" not in name for name, _ in trainable):
            raise BackendError("tts.greektts15.trainable_surface_invalid", "Only the Greek LoRA adapter may be trainable", subsystem="tts")
        grad_checkpoint(model.layers[0])
        train = load_cached_split(plan_path, cache_root, tokenizer, "train")
        validation = load_cached_split(plan_path, cache_root, tokenizer, "validation")
        randomizer = random.Random(TRAINING_SEED)
        if sample_profile == "jarvis-pronunciation":
            order, curriculum = build_pronunciation_curriculum_order(plan_path, steps=steps)
        elif sample_profile == "jarvis-lexical":
            order, curriculum = build_lexical_curriculum_order(plan_path, steps=steps)
        elif sample_profile == "jarvis-exact-core":
            order, curriculum = build_exact_core_curriculum_order(plan_path, steps=steps)
        elif sample_profile == "jarvis-general-stage":
            order, curriculum = build_general_stage_curriculum_order(
                plan_path,
                stage_index=general_stage_index,
                steps=steps,
            )
        elif sample_profile == "jarvis-utterance-focus":
            order, curriculum = build_utterance_focus_curriculum_order(plan_path, steps=steps)
        elif sample_profile in {"jarvis-operation-morphology", "jarvis-operation-morphology-mlp"}:
            order, curriculum = build_operation_morphology_curriculum_order(
                plan_path,
                profile=sample_profile,
                steps=steps,
            )
        elif sample_profile == "jarvis-word-focus":
            order, curriculum = build_word_focus_curriculum_order(plan_path, steps=steps)
        elif sample_profile in {"jarvis-operation-focus", "jarvis-state-focus"}:
            order, curriculum = build_single_word_focus_curriculum_order(plan_path, sample_profile, steps=steps)
        elif sample_profile == "jarvis-importance-focus":
            order, curriculum = build_importance_focus_curriculum_order(plan_path, steps=steps)
        else:
            order = list(range(len(train)))
            randomizer.shuffle(order)
            curriculum = {
                "profile": "uniform",
                "targeted_steps": 0,
                "general_steps": steps,
                "target_counts": {},
                "sample_filenames": [],
            }
        validation_order = sorted(
            range(len(validation)),
            key=lambda index: hashlib.sha256(f"{TRAINING_SEED}:validation:{index}".encode()).digest(),
        )[:PILOT_VALIDATION_SAMPLES]
        validation_samples = [validation[index] for index in validation_order]
        optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
        def training_loss(model: object, batch: object, lengths: object) -> tuple[object, object]:
            if eos_loss_weight == 1:
                return default_loss(model, batch, lengths)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            token_weights = mask + (eos_loss_weight - 1) * mx.logical_and(mask, targets == END_OF_SPEECH)
            cross_entropy = nn.losses.cross_entropy(logits, targets)
            token_count = mask.sum()
            return (cross_entropy * token_weights).astype(mx.float32).sum() / token_count, token_count

        value_and_grad = nn.value_and_grad(model, training_loss)
        validation_before = evaluate(model, validation_samples)
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        started = time.perf_counter()
        losses: list[float] = []
        trained_tokens = 0
        sample_lengths: list[int] = []
        for step, index in enumerate(order[:steps], start=1):
            sequence, offset = train[index]
            batch = mx.array([sequence], dtype=mx.int32)
            lengths = mx.array([[offset, len(sequence)]], dtype=mx.int32)
            (loss, tokens), gradients = value_and_grad(model, batch, lengths)
            optimizer.update(model, gradients)
            mx.eval(model.state, optimizer.state, loss, tokens)
            losses.append(float(loss.item()))
            trained_tokens += int(tokens.item())
            sample_lengths.append(len(sequence))
        elapsed = time.perf_counter() - started
        validation_after = evaluate(model, validation_samples)
        adapter_file = output / "adapters.safetensors"
        complete_adapter = {
            name: value
            for name, value in tree_flatten(model.parameters())
            if "lora_" in name
        }
        if len(complete_adapter) != EXPECTED_LORA_GROUPS:
            raise BackendError(
                "tts.greektts15.adapter_save_invalid",
                "Targeted training did not retain the complete Greek adapter",
                subsystem="tts",
            )
        mx.save_safetensors(str(adapter_file), complete_adapter, metadata={"format": "mlx"})
        shutil.copyfile(source_adapter / "adapter_config.json", output / "adapter_config.json")
        source_hash = _sha256(source_adapter / "adapters.safetensors")
        output_hash = _sha256(adapter_file)
        if source_hash == output_hash:
            raise BackendError("tts.greektts15.optimizer_update_missing", "Training pilot did not change the adapter", subsystem="tts")
        report = {
            "schema": "jarvis.voice.greektts15-training-pilot/1",
            "ready": True,
            "steps": steps,
            "seed": TRAINING_SEED,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "gradient_checkpointing": True,
            "eos_loss_weight": eos_loss_weight,
            "target_profile": target_profile,
            "sample_profile": sample_profile,
            "curriculum": curriculum,
            "frozen_adapter_layers": frozen_layers,
            "frozen_adapter_sections": (
                ["mlp"] if target_profile == "attention-only" else ["self_attn"] if target_profile == "mlp-only" else []
            ),
            "trained_tokens": trained_tokens,
            "mean_train_loss": round(sum(losses) / len(losses), 6),
            "first_train_loss": round(losses[0], 6),
            "last_train_loss": round(losses[-1], 6),
            "validation_samples": len(validation_samples),
            "validation_loss_before": round(validation_before, 6),
            "validation_loss_after": round(validation_after, 6),
            "validation_delta": round(validation_after - validation_before, 6),
            "min_sequence_tokens": min(sample_lengths),
            "max_sequence_tokens": max(sample_lengths),
            "elapsed_seconds": round(elapsed, 3),
            "peak_mlx_bytes": int(mx.get_peak_memory()),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "trainable_groups": len(trainable),
            "saved_adapter_groups": len(complete_adapter),
            "trainable_parameters": sum(int(value.size) for _, value in trainable),
            "source_adapter_sha256": source_hash,
            "immutable_adapter_sha256": _sha256(baseline_adapter / "adapters.safetensors"),
            "source_adapter_kind": "immutable_baseline" if source_adapter == baseline_adapter else "training_checkpoint",
            "output_adapter_sha256": output_hash,
            "gradient_gate_peak_mlx_bytes": gate["peak_mlx_bytes"],
            "base_frozen": True,
            "adapter_merged": False,
        }
        (output / "training_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def blend_adapter_checkpoint(
    model_root: str | Path,
    candidate_directory: str | Path,
    output_directory: str | Path,
    *,
    candidate_fraction: float,
) -> dict[str, object]:
    """Interpolate a trained adapter back toward the immutable Greek baseline."""

    import shutil

    if candidate_fraction not in ADAPTER_BLEND_FACTORS:
        raise BackendError(
            "tts.greektts15.blend_factor_invalid",
            "GreekTTS adapter blend factor is not approved",
            subsystem="tts",
        )
    root = Path(model_root).expanduser()
    baseline = root / "mlx-adapter"
    candidate = Path(candidate_directory).expanduser()
    output = Path(output_directory).expanduser()
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or not candidate.is_absolute()
        or not candidate.is_dir()
        or candidate.is_symlink()
        or not output.is_absolute()
        or output.exists()
        or output.is_symlink()
    ):
        raise BackendError(
            "tts.greektts15.blend_path_invalid",
            "GreekTTS adapter blend paths are invalid",
            subsystem="tts",
        )
    baseline_weights = baseline / "adapters.safetensors"
    candidate_weights = candidate / "adapters.safetensors"
    baseline_config = baseline / "adapter_config.json"
    candidate_config = candidate / "adapter_config.json"
    manifest_path = candidate / "training_manifest.json"
    try:
        baseline_config_value = json.loads(baseline_config.read_text(encoding="utf-8"))
        candidate_config_value = json.loads(candidate_config.read_text(encoding="utf-8"))
        candidate_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError(
            "tts.greektts15.blend_source_invalid",
            "GreekTTS adapter blend source is unreadable",
            subsystem="tts",
        ) from exc
    baseline_hash = _sha256(baseline_weights)
    candidate_hash = _sha256(candidate_weights)
    if not (
        baseline_config_value == candidate_config_value
        and isinstance(candidate_manifest, dict)
        and candidate_manifest.get("schema") == "jarvis.voice.greektts15-training-pilot/1"
        and candidate_manifest.get("ready") is True
        and candidate_manifest.get("source_adapter_sha256") == baseline_hash
        and candidate_manifest.get("output_adapter_sha256") == candidate_hash
    ):
        raise BackendError(
            "tts.greektts15.blend_source_invalid",
            "GreekTTS candidate does not descend from the immutable adapter",
            subsystem="tts",
        )
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise BackendError(
            "tts.greektts15.dependencies_missing",
            "MLX training dependencies are missing",
            subsystem="tts",
        ) from exc
    baseline_tensors = mx.load(str(baseline_weights))
    candidate_tensors = mx.load(str(candidate_weights))
    if (
        not isinstance(baseline_tensors, dict)
        or not isinstance(candidate_tensors, dict)
        or set(baseline_tensors) != set(candidate_tensors)
        or len(baseline_tensors) != EXPECTED_LORA_GROUPS
    ):
        raise BackendError(
            "tts.greektts15.blend_weights_invalid",
            "GreekTTS adapter tensors do not match",
            subsystem="tts",
        )
    blended = {}
    for key in sorted(baseline_tensors):
        baseline_tensor = baseline_tensors[key]
        candidate_tensor = candidate_tensors[key]
        if baseline_tensor.shape != candidate_tensor.shape or baseline_tensor.dtype != candidate_tensor.dtype:
            raise BackendError(
                "tts.greektts15.blend_weights_invalid",
                "GreekTTS adapter tensor shapes do not match",
                subsystem="tts",
            )
        blended[key] = baseline_tensor + candidate_fraction * (candidate_tensor - baseline_tensor)
    mx.eval(blended)
    output.mkdir(mode=0o700, parents=True)
    adapter_file = output / "adapters.safetensors"
    mx.save_safetensors(str(adapter_file), blended, metadata={"format": "mlx"})
    shutil.copyfile(baseline_config, output / "adapter_config.json")
    output_hash = _sha256(adapter_file)
    if output_hash in {baseline_hash, candidate_hash}:
        raise BackendError(
            "tts.greektts15.blend_output_invalid",
            "GreekTTS adapter interpolation did not create a distinct checkpoint",
            subsystem="tts",
        )
    report = {
        "schema": "jarvis.voice.greektts15-adapter-blend/1",
        "ready": True,
        "method": "linear_parameter_interpolation",
        "formula": "baseline + candidate_fraction * (candidate - baseline)",
        "candidate_fraction": candidate_fraction,
        "baseline_fraction": 1.0 - candidate_fraction,
        "tensor_groups": len(blended),
        "baseline_adapter_sha256": baseline_hash,
        "candidate_adapter_sha256": candidate_hash,
        "output_adapter_sha256": output_hash,
        "candidate_training_schema": candidate_manifest["schema"],
        "production_promotion_allowed": False,
        "human_listening_required": True,
    }
    (output / "blend_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def compose_disjoint_adapter_branches(
    model_root: str | Path,
    operation_directory: str | Path,
    state_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Compose approved disjoint MLP and attention updates without averaging."""

    import shutil

    root = Path(model_root).expanduser()
    operation = Path(operation_directory).expanduser()
    state = Path(state_directory).expanduser()
    output = Path(output_directory).expanduser()
    if any(not path.is_absolute() or not path.is_dir() or path.is_symlink() for path in (root, operation, state)) or (
        not output.is_absolute() or output.exists() or output.is_symlink()
    ):
        raise BackendError("tts.greektts15.compose_path_invalid", "Adapter composition paths are invalid", subsystem="tts")
    baseline = root / "mlx-adapter"
    try:
        operation_manifest = json.loads((operation / "training_manifest.json").read_text(encoding="utf-8"))
        state_manifest = json.loads((state / "training_manifest.json").read_text(encoding="utf-8"))
        operation_config = json.loads((operation / "adapter_config.json").read_text(encoding="utf-8"))
        state_config = json.loads((state / "adapter_config.json").read_text(encoding="utf-8"))
        baseline_config = json.loads((baseline / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("tts.greektts15.compose_source_invalid", "Adapter composition evidence is unreadable", subsystem="tts") from exc
    operation_hash = _sha256(operation / "adapters.safetensors")
    state_hash = _sha256(state / "adapters.safetensors")
    parent_hash = operation_manifest.get("source_adapter_sha256")
    if not (
        operation_manifest.get("ready") is True
        and operation_manifest.get("target_profile") == "mlp-only"
        and operation_manifest.get("sample_profile") == "jarvis-operation-focus"
        and operation_manifest.get("output_adapter_sha256") == operation_hash
        and state_manifest.get("ready") is True
        and state_manifest.get("target_profile") == "attention-only"
        and state_manifest.get("sample_profile") == "jarvis-state-focus"
        and state_manifest.get("output_adapter_sha256") == state_hash
        and state_manifest.get("source_adapter_sha256") == parent_hash
        and operation_config == state_config == baseline_config
    ):
        raise BackendError("tts.greektts15.compose_source_invalid", "Adapter branches are not composable", subsystem="tts")
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise BackendError("tts.greektts15.dependencies_missing", "Safetensors is unavailable", subsystem="tts") from exc
    operation_tensors = load_file(str(operation / "adapters.safetensors"), device="cpu")
    state_tensors = load_file(str(state / "adapters.safetensors"), device="cpu")
    if set(operation_tensors) != set(state_tensors) or len(operation_tensors) != EXPECTED_LORA_GROUPS:
        raise BackendError("tts.greektts15.compose_source_invalid", "Adapter tensor surfaces differ", subsystem="tts")
    composed = {
        name: operation_tensors[name] if ".mlp." in name else state_tensors[name]
        for name in operation_tensors
    }
    if sum(".mlp." in name for name in composed) != MLP_ONLY_LORA_GROUPS or sum(
        ".self_attn." in name for name in composed
    ) != ATTENTION_ONLY_LORA_GROUPS:
        raise BackendError("tts.greektts15.compose_surface_invalid", "Adapter composition surface is invalid", subsystem="tts")
    output.mkdir(mode=0o700, parents=True)
    save_file(composed, str(output / "adapters.safetensors"), metadata={"format": "mlx"})
    shutil.copyfile(operation / "adapter_config.json", output / "adapter_config.json")
    output_hash = _sha256(output / "adapters.safetensors")
    report = {
        "schema": "jarvis.voice.greektts15-disjoint-composition/1",
        "ready": True,
        "operation_adapter_sha256": operation_hash,
        "state_adapter_sha256": state_hash,
        "parent_adapter_sha256": parent_hash,
        "immutable_adapter_sha256": _sha256(baseline / "adapters.safetensors"),
        "output_adapter_sha256": output_hash,
        "mlp_source": "jarvis-operation-focus",
        "attention_source": "jarvis-state-focus",
        "averaging_used": False,
        "production_promotion_allowed": False,
    }
    (output / "composition_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


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


def _normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.findall(r"[α-ωϊϋΐΰ]+", without_marks)


def _word_error_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1]


def _score_transcript_sets(
    references: dict[str, str],
    transcripts: dict[str, dict[str, str]],
    *,
    required_labels: set[str],
    schema: str,
) -> dict[str, object]:
    if set(transcripts) != required_labels or not references or any(set(items) != set(references) for items in transcripts.values()):
        raise BackendError("tts.greektts15.asr_transcripts_incomplete", "Candidate transcripts are incomplete", subsystem="tts")
    candidates: list[dict[str, object]] = []
    for label in sorted(transcripts):
        samples: list[dict[str, object]] = []
        total_words = total_errors = 0
        for fixture_id, reference_text in references.items():
            hypothesis = transcripts[label][fixture_id]
            if not isinstance(hypothesis, str):
                raise BackendError("tts.greektts15.asr_transcripts_incomplete", "Candidate transcript is invalid", subsystem="tts")
            reference_words = _normalized_words(reference_text)
            errors = _word_error_distance(reference_words, _normalized_words(hypothesis))
            total_words += len(reference_words)
            total_errors += errors
            samples.append(
                {
                    "fixture_id": fixture_id,
                    "reference": reference_text,
                    "transcript": hypothesis,
                    "reference_words": len(reference_words),
                    "word_errors": errors,
                }
            )
        candidates.append(
            {
                "label": label,
                "total_reference_words": total_words,
                "total_word_errors": total_errors,
                "micro_wer_percent": round(100.0 * total_errors / total_words, 3),
                "samples": samples,
            }
        )
    return {
        "schema": schema,
        "ready": True,
        "candidates": candidates,
        "human_listening_required": True,
        "promotion_allowed": False,
    }


def score_candidate_transcripts(
    references: dict[str, str],
    transcripts: dict[str, dict[str, str]],
) -> dict[str, object]:
    return _score_transcript_sets(
        references,
        transcripts,
        required_labels=ASR_CANDIDATE_LABELS,
        schema="jarvis.voice.greektts15-asr-comparison/1",
    )


def score_pronunciation_transcripts(
    references: dict[str, str],
    transcripts: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Score native, baseline, and conservative-blend pronunciation audio."""

    candidate_labels = set(transcripts) - {"native", "baseline"}
    if len(candidate_labels) != 1 or not candidate_labels <= PRONUNCIATION_ASR_CANDIDATE_LABELS:
        raise BackendError(
            "tts.greektts15.asr_transcripts_incomplete",
            "Pronunciation candidate label is invalid",
            subsystem="tts",
        )
    candidate_label = next(iter(candidate_labels))
    report = _score_transcript_sets(
        references,
        transcripts,
        required_labels={"native", "baseline", candidate_label},
        schema="jarvis.voice.greektts15-pronunciation-asr/1",
    )
    by_label = {item["label"]: item for item in report["candidates"]}
    report["native_calibration_wer_percent"] = by_label["native"]["micro_wer_percent"]
    report["candidate_label"] = candidate_label
    report["candidate_not_worse_than_baseline"] = (
        by_label[candidate_label]["micro_wer_percent"] <= by_label["baseline"]["micro_wer_percent"]
    )
    return report


def score_quality_candidate_transcripts(
    references: dict[str, str],
    transcripts: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Compare one approved JARVIS-text candidate with the immutable baseline."""

    candidate_labels = set(transcripts) - {"baseline"}
    if len(candidate_labels) != 1 or not candidate_labels <= QUALITY_ASR_CANDIDATE_LABELS:
        raise BackendError(
            "tts.greektts15.asr_transcripts_incomplete",
            "Quality candidate label is invalid",
            subsystem="tts",
        )
    candidate_label = next(iter(candidate_labels))
    report = _score_transcript_sets(
        references,
        transcripts,
        required_labels={"baseline", candidate_label},
        schema="jarvis.voice.greektts15-quality-asr/1",
    )
    by_label = {item["label"]: item for item in report["candidates"]}
    report["candidate_label"] = candidate_label
    report["candidate_not_worse_than_baseline"] = (
        by_label[candidate_label]["micro_wer_percent"] <= by_label["baseline"]["micro_wer_percent"]
    )
    return report


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
