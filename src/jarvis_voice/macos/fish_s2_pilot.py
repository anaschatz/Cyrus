"""Fail-closed local qualification for Fish Audio S2 Pro on Apple Silicon.

The pilot is intentionally isolated from the production ``TtsBackend``. It
never downloads or installs anything, rejects unpinned files, blocks network
access during model use, and records that the current MLX adapter has no
incremental audio streaming or generation-time cancellation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

from ..errors import BackendError


MODEL_REPO_ID = "majentik/fishaudio-s2-pro-MLX-4bit"
MODEL_REVISION = "dda28aa6e33fe946826792e3feb643e9c463b75f"
CODEC_REPO_ID = "mlx-community/fish-audio-s2-pro-bf16"
CODEC_REVISION = "eccd57bf5c1ebc13cb2f993df867f4e49931a36a"
MLX_AUDIO_REVISION = "76f8363b28e82656d537ea17a38dc97e9ed9a0f8"
FISH_LICENSE = "Fish Audio Research License"

GIB = 1024**3
MAX_STANDALONE_PEAK_RSS_BYTES = 7 * GIB
MAX_STANDALONE_MLX_BYTES = 7 * GIB
MAX_MEAN_REALTIME_FACTOR = 1.0
MAX_ASR_MICRO_WER_PERCENT = 5.0


class ArtifactSpec(NamedTuple):
    filename: str
    size: int
    sha256: str


MODEL_ARTIFACTS = (
    ArtifactSpec("LICENSE.md", 10_360, "aa7d9206e9d710590987a3636934f643529c00cd490323594e6206aaa0c32d80"),
    ArtifactSpec("NOTICE", 462, "aa5985ca4b48a4006efa0cfa4e04a5185513194e858bda735214da9bdde85191"),
    ArtifactSpec("README.md", 3_136, "74bea38fa4b68b4ca2f2df4994079bbd9b64e1a4b6c4d548503747ce6d93d6ae"),
    ArtifactSpec("chat_template.jinja", 4_116, "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5"),
    ArtifactSpec("codec.safetensors", 1_870_921_167, "c2096e27a1bb69a05151ccb502649a88744712270d9b029a68a3c9305f2b1edf"),
    ArtifactSpec("config.json", 2_315, "04a6afe6a550a46a3fd616cbad33f3382f737628434896eef1fa4eb6eba13376"),
    ArtifactSpec("model.safetensors", 2_566_443_153, "1cbb30334f1b265a601dd46a8429c7889d240095cf2a39e55e82d4a2c0e934c2"),
    ArtifactSpec("model.safetensors.index.json", 53_694, "0a3a55c4051c60b83f9e4e77be6b1ce33d2a975f22cb3ead0b22dfdecd16baac"),
    ArtifactSpec("special_tokens_map.json", 101_864, "c2ff18fde6e43b7408435bc8ed079af74531befba549358a97cbc59ce606bc6b"),
    ArtifactSpec("tokenizer.json", 12_217_872, "f24e08099d45a8adf3f52f5f0b03276e433bb9d689bb15fcbcc48ce58744588b"),
    ArtifactSpec("tokenizer_config.json", 860_832, "b8d149343ae425b0da67e6708686aceb51be7815d9792f265fc12ff04d5e9856"),
)

F1_REFERENCE_SHA256 = "a22adf7c5650d6c2029c24229a7a1a216927a71baa578f60aae6f43df736fcbe"
F1_REFERENCE_TEXT = (
    "Η πρώτη καταγραφή των Ολυμπιακών αγώνων στην αρχαιότητα ήταν στην "
    "Ολυμπία, το εφτακόσια εβδομήντα έξι προ Χριστού με το σημερινό "
    "ημερολόγιο. Ελπίζουμε ότι η επόμενη δουλειά του θα είναι "
    "αντιπροσωπευτική για αυτόν."
)

QUALIFICATION_FIXTURES = (
    ("greeting", "Καλημέρα. Είμαι έτοιμη να σε βοηθήσω."),
    (
        "articulation",
        "Η τεχνητή νοημοσύνη επεξεργάζεται πληροφορίες και προφέρει κάθε "
        "ελληνική λέξη με σταθερή και φυσική άρθρωση.",
    ),
    ("source_control", "Ελπίζουμε ότι η επόμενη δουλειά του θα είναι αντιπροσωπευτική για αυτόν."),
    ("consonants", "Ο ξεναγός εξήγησε προσεκτικά την πολύπλοκη αρχιτεκτονική του κτιρίου."),
    ("clusters", "Η ψύχραιμη αξιολόγηση απέτρεψε μια βιαστική και λανθασμένη απόφαση."),
    ("names", "Η Αλεξάνδρα και ο Χρήστος θα συμμετάσχουν στη συνάντηση."),
    ("numbers", "Υπάρχουν δεκατρία νέα μηνύματα και είκοσι επτά εκκρεμείς ειδοποιήσεις."),
    (
        "long_sentence",
        "Μόλις ολοκληρωθεί η επεξεργασία, θα συγκρίνω τα αποτελέσματα και θα "
        "σου παρουσιάσω μόνο τις σημαντικότερες διαφορές.",
    ),
)


@dataclass(frozen=True, slots=True)
class Validation:
    ready: bool
    code: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_directory(value: str | Path, *, code: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise BackendError(code, "Fish S2 pilot directory is invalid", subsystem="tts")
    return path


def validate_model_directory(model_directory: str | Path) -> Validation:
    try:
        root = _safe_directory(model_directory, code="tts.fish_s2.model_directory_invalid")
    except BackendError as exc:
        return Validation(False, exc.code)
    for artifact in MODEL_ARTIFACTS:
        path = root / artifact.filename
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size != artifact.size:
                return Validation(False, "tts.fish_s2.model_artifact_invalid")
            if _sha256(path) != artifact.sha256:
                return Validation(False, "tts.fish_s2.model_artifact_invalid")
        except OSError:
            return Validation(False, "tts.fish_s2.model_artifact_missing")
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return Validation(False, "tts.fish_s2.config_invalid")
    quantization = config.get("quantization") if isinstance(config, dict) else None
    if (
        config.get("model_type") != "fish_qwen3_omni"
        or not isinstance(quantization, dict)
        or quantization.get("bits") != 4
        or quantization.get("group_size") != 64
    ):
        return Validation(False, "tts.fish_s2.config_invalid")
    return Validation(True, "tts.fish_s2.model_ready")


def validate_reference_audio(reference_audio: str | Path) -> Path:
    path = Path(reference_audio).expanduser()
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            rate = stream.getframerate()
            frames = stream.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise BackendError(
            "tts.fish_s2.reference_invalid",
            "Fish S2 F1 reference is invalid",
            subsystem="tts",
        ) from exc
    duration = frames / rate if rate else 0.0
    invalid = (
        not path.is_absolute()
        or path.is_symlink()
        or channels != 1
        or width != 2
        or rate != 44_100
        or not 10.0 <= duration <= 30.0
        or _sha256(path) != F1_REFERENCE_SHA256
    )
    if invalid:
        raise BackendError(
            "tts.fish_s2.reference_invalid",
            "Fish S2 F1 reference is invalid",
            subsystem="tts",
        )
    return path


def validate_runtime_source(source_directory: str | Path) -> Path:
    root = _safe_directory(source_directory, code="tts.fish_s2.runtime_source_invalid")
    module = root / "mlx_audio" / "tts" / "models" / "fish_qwen3_omni" / "fish_speech.py"
    if not module.is_file() or module.is_symlink():
        raise BackendError(
            "tts.fish_s2.runtime_source_invalid",
            "Pinned MLX Audio source is unavailable",
            subsystem="tts",
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackendError(
            "tts.fish_s2.runtime_revision_unavailable",
            "Pinned MLX Audio revision cannot be verified",
            subsystem="tts",
        ) from exc
    if revision != MLX_AUDIO_REVISION:
        raise BackendError(
            "tts.fish_s2.runtime_revision_mismatch",
            "MLX Audio revision does not match the qualification pin",
            subsystem="tts",
        )
    return root


def _deny_network(function: Callable[..., dict[str, object]]) -> Callable[..., dict[str, object]]:
    @wraps(function)
    def guarded(*args, **kwargs):
        original_socket = socket.socket
        original_create_connection = socket.create_connection

        class DeniedSocket(original_socket):
            def connect(self, *connect_args, **connect_kwargs):  # type: ignore[no-untyped-def]
                raise BackendError(
                    "tts.fish_s2.network_forbidden",
                    "Fish S2 qualification is local-only",
                    subsystem="tts",
                )

            def connect_ex(self, *connect_args, **connect_kwargs):  # type: ignore[no-untyped-def]
                self.connect(*connect_args, **connect_kwargs)
                return 1

        def denied_connection(*connect_args, **connect_kwargs):  # type: ignore[no-untyped-def]
            raise BackendError(
                "tts.fish_s2.network_forbidden",
                "Fish S2 qualification is local-only",
                subsystem="tts",
            )

        socket.socket = DeniedSocket
        socket.create_connection = denied_connection
        try:
            return function(*args, **kwargs)
        finally:
            socket.socket = original_socket
            socket.create_connection = original_create_connection

    return guarded


def _selected_fixtures(fixture_ids: Sequence[str] | None) -> tuple[tuple[str, str], ...]:
    fixtures = dict(QUALIFICATION_FIXTURES)
    selected = tuple(fixtures) if fixture_ids is None else tuple(fixture_ids)
    if not selected or len(set(selected)) != len(selected) or any(item not in fixtures for item in selected):
        raise BackendError(
            "tts.fish_s2.fixture_selection_invalid",
            "Fish S2 fixture selection is invalid",
            subsystem="tts",
        )
    return tuple((fixture_id, fixtures[fixture_id]) for fixture_id in selected)


def pilot_preflight(
    model_directory: str | Path,
    reference_audio: str | Path,
    runtime_source: str | Path,
) -> dict[str, object]:
    validation = validate_model_directory(model_directory)
    if not validation.ready:
        return {"ready": False, "code": validation.code, "network_used": False}
    validate_reference_audio(reference_audio)
    validate_runtime_source(runtime_source)
    host_ready = sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info[:2] == (3, 11)
    return {
        "schema_version": 1,
        "ready": host_ready,
        "code": "tts.fish_s2.preflight_ready" if host_ready else "tts.fish_s2.host_unsupported",
        "model": {"repo_id": MODEL_REPO_ID, "revision": MODEL_REVISION},
        "codec": {"repo_id": CODEC_REPO_ID, "revision": CODEC_REVISION, "safe_serialization": True},
        "runtime": {"revision": MLX_AUDIO_REVISION},
        "voice": {"reference_sha256": F1_REFERENCE_SHA256, "duration_range_seconds": [10, 30]},
        "license": {"name": FISH_LICENSE, "commercial_use_requires_separate_license": True},
        "network_used": False,
        "installation_performed": False,
        "production_backend_changed": False,
    }


@_deny_network
def run_pilot(
    model_directory: str | Path,
    output_directory: str | Path,
    reference_audio: str | Path,
    runtime_source: str | Path,
    *,
    fixture_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    preflight = pilot_preflight(model_directory, reference_audio, runtime_source)
    if not preflight["ready"]:
        raise BackendError(str(preflight["code"]), "Fish S2 preflight failed", subsystem="tts")
    model_path = Path(model_directory).expanduser()
    reference_path = validate_reference_audio(reference_audio)
    selected = _selected_fixtures(fixture_ids)
    output = Path(output_directory).expanduser()
    if not output.is_absolute() or output.is_symlink():
        raise BackendError("tts.fish_s2.output_invalid", "Fish S2 output directory is invalid", subsystem="tts")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)

    try:
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from mlx_audio.tts.utils import load_model
    except ImportError as exc:
        raise BackendError(
            "tts.fish_s2.dependencies_missing",
            "Fish S2 local dependencies are unavailable",
            subsystem="tts",
        ) from exc

    load_started = time.perf_counter()
    model = load_model(model_path, lazy=False, strict=True)
    load_seconds = time.perf_counter() - load_started
    reference_samples, reference_rate = sf.read(
        reference_path,
        dtype="float32",
        always_2d=False,
    )
    if reference_rate != model.sample_rate or reference_samples.ndim != 1:
        raise BackendError(
            "tts.fish_s2.reference_runtime_mismatch",
            "Fish S2 reference does not match the model audio profile",
            subsystem="tts",
        )
    reference = mx.array(np.asarray(reference_samples, dtype=np.float32))

    samples: list[dict[str, object]] = []
    for index, (fixture_id, text) in enumerate(selected, start=1):
        mx.random.seed(20_260_813 + index)
        started = time.perf_counter()
        results = list(
            model.generate(
                text=text,
                ref_audio=reference,
                ref_text=F1_REFERENCE_TEXT,
                max_tokens=512,
                temperature=0.45,
                top_p=0.7,
                top_k=30,
                stream=False,
                speed=1.0,
                verbose=False,
            )
        )
        generation_seconds = time.perf_counter() - started
        if len(results) != 1:
            raise BackendError("tts.fish_s2.output_invalid", "Fish S2 produced an unexpected segment count", subsystem="tts")
        result = results[0]
        audio = np.asarray(result.audio, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise BackendError("tts.fish_s2.output_invalid", "Fish S2 produced invalid audio", subsystem="tts")
        destination = output / f"{index:02d}-{fixture_id}.wav"
        sf.write(destination, audio, result.sample_rate, subtype="PCM_16", format="WAV")
        os.chmod(destination, 0o600)
        duration_seconds = audio.size / float(result.sample_rate)
        samples.append(
            {
                "fixture_id": fixture_id,
                "text": text,
                "file": destination.name,
                "sha256": _sha256(destination),
                "sample_rate": result.sample_rate,
                "duration_seconds": round(duration_seconds, 4),
                "generation_seconds": round(generation_seconds, 4),
                "realtime_factor": round(generation_seconds / duration_seconds, 4),
            }
        )

    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    mlx_peak = int(mx.get_peak_memory())
    mean_rtf = sum(float(item["realtime_factor"]) for item in samples) / len(samples)
    automated_gates = {
        "pinned_safe_model_pack": True,
        "pinned_f1_reference": True,
        "all_selected_fixtures_generated": len(samples) == len(selected),
        "standalone_rss_within_limit": peak_rss <= MAX_STANDALONE_PEAK_RSS_BYTES,
        "standalone_mlx_memory_within_limit": mlx_peak <= MAX_STANDALONE_MLX_BYTES,
        "mean_realtime_factor_within_limit": mean_rtf <= MAX_MEAN_REALTIME_FACTOR,
        "incremental_audio_streaming": False,
        "interrupt_during_generation": False,
    }
    return {
        "schema_version": 1,
        "ready": True,
        "code": "tts.fish_s2.pilot_completed",
        "backend": "fish_s2_pro_mlx_4bit",
        "device": "metal",
        "language": "el",
        "model": preflight["model"],
        "codec": preflight["codec"],
        "runtime": preflight["runtime"],
        "voice": {"reference_sha256": F1_REFERENCE_SHA256, "reference_text_pinned": True},
        "sampling": {"temperature": 0.45, "top_p": 0.7, "top_k": 30, "speed": 1.0},
        "model_load_seconds": round(load_seconds, 4),
        "mlx_peak_bytes": mlx_peak,
        "peak_rss_bytes": peak_rss,
        "mean_realtime_factor": round(mean_rtf, 4),
        "samples": samples,
        "qualification": {
            "automated_gates": automated_gates,
            "automated_gates_passed": all(automated_gates.values()),
            "asr_round_trip_measured": False,
            "llm_coexistence_measured": False,
            "human_voice_approved": False,
            "production_promotion_allowed": False,
        },
        "license": preflight["license"],
        "network_used": False,
        "network_guard": "python_socket_fail_closed",
        "installation_performed": False,
        "production_backend_changed": False,
    }


def _normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


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


def score_transcripts(
    transcripts: dict[str, str],
    *,
    fixture_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    selected = _selected_fixtures(fixture_ids)
    expected = {fixture_id for fixture_id, _ in selected}
    if set(transcripts) != expected or any(not isinstance(value, str) for value in transcripts.values()):
        raise BackendError(
            "tts.fish_s2.transcripts_incomplete",
            "Fish S2 transcripts are incomplete",
            subsystem="tts",
        )
    samples: list[dict[str, object]] = []
    total_errors = 0
    total_words = 0
    for fixture_id, reference_text in selected:
        reference_words = _normalized_words(reference_text)
        hypothesis = transcripts[fixture_id]
        errors = _word_error_distance(reference_words, _normalized_words(hypothesis))
        total_errors += errors
        total_words += len(reference_words)
        samples.append(
            {
                "fixture_id": fixture_id,
                "reference_words": len(reference_words),
                "word_errors": errors,
                "wer_percent": round(100.0 * errors / len(reference_words), 3),
                "transcript": hypothesis,
            }
        )
    micro_wer = 100.0 * total_errors / total_words
    return {
        "schema_version": 1,
        "ready": True,
        "code": "tts.fish_s2.asr_scoring_completed",
        "fixture_count": len(samples),
        "total_reference_words": total_words,
        "total_word_errors": total_errors,
        "micro_wer_percent": round(micro_wer, 3),
        "pronunciation_gate_percent": MAX_ASR_MICRO_WER_PERCENT,
        "pronunciation_gate_passed": micro_wer <= MAX_ASR_MICRO_WER_PERCENT,
        "samples": samples,
        "human_voice_approved": False,
        "production_promotion_allowed": False,
    }


@_deny_network
def score_outputs(
    output_directory: str | Path,
    whisper_model_directory: str | Path,
    *,
    fixture_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    output = _safe_directory(output_directory, code="tts.fish_s2.output_invalid")
    whisper = _safe_directory(whisper_model_directory, code="tts.fish_s2.whisper_model_invalid")
    selected = _selected_fixtures(fixture_ids)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise BackendError(
            "tts.fish_s2.whisper_dependency_missing",
            "Local faster-whisper dependency is unavailable",
            subsystem="tts",
        ) from exc
    model = WhisperModel(str(whisper), device="cpu", compute_type="int8", local_files_only=True)
    transcripts: dict[str, str] = {}
    for index, (fixture_id, _) in enumerate(selected, start=1):
        path = output / f"{index:02d}-{fixture_id}.wav"
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 44:
            raise BackendError("tts.fish_s2.output_missing", "Fish S2 output is missing", subsystem="tts")
        segments, _ = model.transcribe(
            str(path),
            language="el",
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        transcripts[fixture_id] = " ".join(segment.text.strip() for segment in segments).strip()
    report = score_transcripts(transcripts, fixture_ids=[item[0] for item in selected])
    report["asr"] = {
        "backend": "faster_whisper",
        "model_path_explicit": True,
        "compute_type": "int8",
        "network_used": False,
    }
    return report
