"""Pinned, offline Supertonic 3 qualification for native Greek speech."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import socket
import struct
import sys
import time
import unicodedata
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, NamedTuple

from .. import manifest
from ..errors import BackendError


SUPERTONIC_VERSION = "1.3.1"
ONNXRUNTIME_VERSION = "1.23.2"
NUMPY_VERSION = "2.4.6"
SOUNDFILE_VERSION = "0.14.0"

MODEL_REPO_ID = "Supertone/supertonic-3"
# This is the exact revision pinned by supertonic==1.3.1.
MODEL_REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"
MODEL_LICENSE = "OpenRAIL-M"
SAMPLE_RATE = 44_100
VOICE_NAMES = ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5")
FEMALE_VOICE_NAMES = VOICE_NAMES[:5]
DEFAULT_VOICE = "F5"
DEFAULT_TOTAL_STEPS = 12
DEFAULT_SPEED = 1.0
DEFAULT_SEED = 10_000

MAX_LOAD_SECONDS = 2.0
MAX_MEAN_REALTIME_FACTOR = 0.6
MAX_PEAK_RSS_BYTES = 1024 * 1024 * 1024
MAX_ASR_MICRO_WER_PERCENT = 5.0

QUALITY_FIXTURES = (
    ("greeting", "Καλημέρα. Είμαι έτοιμη να σε βοηθήσω."),
    ("system_status", "Έλεγξα την κατάσταση του συστήματος και όλα λειτουργούν κανονικά."),
    ("calendar", "Η επόμενη συνάντησή σου είναι στις οκτώ και μισή."),
    ("reminder", "Αν θέλεις, μπορώ να σου υπενθυμίσω τις σημαντικές εργασίες."),
    (
        "articulation",
        "Η τεχνητή νοημοσύνη επεξεργάζεται πληροφορίες και προφέρει κάθε ελληνική λέξη "
        "με σταθερή και φυσική άρθρωση.",
    ),
    ("source_control", "Ελπίζουμε ότι η επόμενη δουλειά του θα είναι αντιπροσωπευτική για αυτόν."),
    ("barge_in", "Θα σταματήσω αμέσως μόλις αρχίσεις να μιλάς."),
    ("connection", "Η ασύρματη σύνδεση αποκαταστάθηκε χωρίς απώλεια δεδομένων."),
    ("weather", "Αύριο προβλέπονται ισχυρές βροχές και χαμηλότερη θερμοκρασία."),
    ("navigation", "Στρίψε δεξιά στη διασταύρωση και συνέχισε μέχρι την πλατεία."),
    ("security", "Η πρόσβαση απορρίφθηκε επειδή ο κωδικός επιβεβαίωσης έληξε."),
    ("numbers", "Υπάρχουν δεκατρία νέα μηνύματα και είκοσι επτά εκκρεμείς ειδοποιήσεις."),
    ("time", "Το αντίγραφο ασφαλείας θα ολοκληρωθεί σε περίπου τριάντα πέντε λεπτά."),
    ("names", "Η Αλεξάνδρα και ο Χρήστος θα συμμετάσχουν στη συνάντηση."),
    ("consonants", "Ο ξεναγός εξήγησε προσεκτικά την πολύπλοκη αρχιτεκτονική του κτιρίου."),
    ("clusters", "Η ψύχραιμη αξιολόγηση απέτρεψε μια βιαστική και λανθασμένη απόφαση."),
    ("question", "Θέλεις να ανοίξω το έγγραφο ή να δημιουργήσω ένα καινούργιο;"),
    ("confirmation", "Βεβαίως. Η αλλαγή αποθηκεύτηκε και μπορεί να αναιρεθεί."),
    (
        "long_sentence",
        "Μόλις ολοκληρωθεί η επεξεργασία, θα συγκρίνω τα αποτελέσματα και θα σου "
        "παρουσιάσω μόνο τις σημαντικότερες διαφορές.",
    ),
    ("closing", "Ολοκλήρωσα τον έλεγχο. Πες μου όταν θέλεις να συνεχίσουμε."),
)


class ArtifactSpec(NamedTuple):
    relative_path: str
    size: int
    sha256: str


MODEL_ARTIFACTS = (
    ArtifactSpec("LICENSE", 15_007, "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f"),
    ArtifactSpec("config.json", 174, "4099082b107a9d4029849ac76b89eca65e03732660969c2babe5bf308c7357f2"),
    ArtifactSpec(
        "onnx/duration_predictor.onnx",
        3_700_147,
        "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
    ),
    ArtifactSpec(
        "onnx/text_encoder.onnx",
        36_416_150,
        "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
    ),
    ArtifactSpec("onnx/tts.json", 8_253, "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09"),
    ArtifactSpec(
        "onnx/unicode_indexer.json",
        277_676,
        "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f",
    ),
    ArtifactSpec(
        "onnx/vector_estimator.onnx",
        256_534_781,
        "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c",
    ),
    ArtifactSpec(
        "onnx/vocoder.onnx",
        101_424_195,
        "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba",
    ),
    ArtifactSpec("voice_styles/F1.json", 292_046, "bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2"),
    ArtifactSpec("voice_styles/F2.json", 292_423, "7c722c6a72707b1a77f035d67f0d1351ba187738e06f7683e8c72b1df3477fc6"),
    ArtifactSpec("voice_styles/F3.json", 290_794, "12f6ef2573baa2defa1128069cb59f203e3ab67c92af77b42df8a0e3a2f7c6ab"),
    ArtifactSpec("voice_styles/F4.json", 291_808, "c2fa764c1225a76dfc3e2c73e8aa4f70d9ee48793860eb34c295fff01c2e032b"),
    ArtifactSpec("voice_styles/F5.json", 291_479, "45966e73316415626cf41a7d1c6f3b4c70dbc1ba2bee5c1978ef0ce33244fc8d"),
    ArtifactSpec("voice_styles/M1.json", 291_748, "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b"),
    ArtifactSpec("voice_styles/M2.json", 292_055, "b76cbf62bac707c710cf0ae5aba5e31eea1a6339a9734bfae33ab98499534a50"),
    ArtifactSpec("voice_styles/M3.json", 290_198, "ea1ac35ccb91b0d7ecad533a2fbd0eec10c91513d8951e3b25fbba99954e159b"),
    ArtifactSpec("voice_styles/M4.json", 291_522, "ca8eefad4fcd989c9379032ff3e50738adc547eeb5e221b82593a6d7b3bac303"),
    ArtifactSpec("voice_styles/M5.json", 291_469, "dd22b92740314321f8ae11c5e87f8dd60d060f15dd3a632b5adf77f471f77af2"),
)


@dataclass(frozen=True, slots=True)
class ModelValidation:
    ready: bool
    code: str
    files_present: int
    files_required: int


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _exact_distribution_version(name: str, expected: str) -> bool:
    try:
        return importlib.metadata.version(name) == expected
    except importlib.metadata.PackageNotFoundError:
        return False


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def validate_supertonic3_model_directory(model_directory: str | Path) -> ModelValidation:
    root = Path(model_directory).expanduser()
    required = len(MODEL_ARTIFACTS)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        return ModelValidation(False, "tts.supertonic3.model_directory_missing", 0, required)
    present = 0
    for artifact in MODEL_ARTIFACTS:
        path = root / artifact.relative_path
        try:
            valid = (
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == artifact.size
                and _sha256(path) == artifact.sha256
            )
        except OSError:
            valid = False
        if not valid:
            return ModelValidation(False, "tts.supertonic3.model_artifact_invalid", present, required)
        present += 1
    config = _read_json(root / "config.json")
    tts_config = _read_json(root / "onnx" / "tts.json")
    if not isinstance(config, dict) or not isinstance(tts_config, dict):
        return ModelValidation(False, "tts.supertonic3.model_schema_invalid", present, required)
    ae = tts_config.get("ae")
    if (
        config.get("model_name") != "Supertonic 3"
        or config.get("model_type") != "onnx"
        or tts_config.get("tts_version") != "v1.7.3"
        or tts_config.get("split") != "opensource-multilingual"
        or not isinstance(ae, dict)
        or ae.get("sample_rate") != SAMPLE_RATE
    ):
        return ModelValidation(False, "tts.supertonic3.model_schema_invalid", present, required)
    return ModelValidation(True, "tts.supertonic3.model_ready", required, required)


def supertonic3_pilot_preflight(
    model_directory: str | Path,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    python_version: tuple[int, int, int] | None = None,
    pointer_bits: int | None = None,
    dependency_probe: Callable[[str, str], bool] = _exact_distribution_version,
) -> dict[str, object]:
    selected_platform = sys.platform if platform_name is None else platform_name
    selected_machine = platform.machine() if machine is None else machine
    selected_version = sys.version_info[:3] if python_version is None else python_version
    selected_bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    host_ready = (
        selected_platform == "darwin"
        and selected_machine == "arm64"
        and selected_version[:2] == (3, 11)
        and selected_bits == 64
    )
    dependencies = {
        "supertonic": dependency_probe("supertonic", SUPERTONIC_VERSION),
        "onnxruntime": dependency_probe("onnxruntime", ONNXRUNTIME_VERSION),
        "numpy": dependency_probe("numpy", NUMPY_VERSION),
        "soundfile": dependency_probe("soundfile", SOUNDFILE_VERSION),
    }
    validation = validate_supertonic3_model_directory(model_directory)
    ready = host_ready and all(dependencies.values()) and validation.ready
    if not host_ready:
        code = "tts.supertonic3.host_unsupported"
    elif not all(dependencies.values()):
        code = "tts.supertonic3.dependencies_missing"
    else:
        code = validation.code
    return {
        "schema_version": 1,
        "ready": ready,
        "code": code,
        "host": {
            "platform": selected_platform,
            "machine": selected_machine,
            "python": ".".join(str(item) for item in selected_version),
            "bits": selected_bits,
        },
        "dependencies": dependencies,
        "model": {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "ready": validation.ready,
            "files_present": validation.files_present,
            "files_required": validation.files_required,
        },
        "network_used": False,
        "installation_performed": False,
        "runtime_started": False,
        "content_captured": False,
    }


def _deny_network_during_pilot(
    function: Callable[..., dict[str, object]],
) -> Callable[..., dict[str, object]]:
    @wraps(function)
    def guarded(*args: object, **kwargs: object) -> dict[str, object]:
        original_socket = socket.socket
        original_create_connection = socket.create_connection

        class OfflineSocket(original_socket):
            def connect(self, address: object) -> None:
                del address
                raise BackendError(
                    "tts.supertonic3.network_forbidden",
                    "Network access is forbidden during Supertonic qualification",
                    subsystem="tts",
                )

            def connect_ex(self, address: object) -> int:
                self.connect(address)
                return 1

        def deny_create_connection(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise BackendError(
                "tts.supertonic3.network_forbidden",
                "Network access is forbidden during Supertonic qualification",
                subsystem="tts",
            )

        socket.socket = OfflineSocket
        socket.create_connection = deny_create_connection
        previous = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")}
        os.environ.update({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
        try:
            return function(*args, **kwargs)
        finally:
            socket.socket = original_socket
            socket.create_connection = original_create_connection
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    return guarded


def _prepare_output_directory(output_directory: str | Path) -> Path:
    output = Path(output_directory).expanduser()
    if not output.is_absolute() or output.is_symlink():
        raise BackendError(
            "tts.supertonic3.output_invalid",
            "Supertonic qualification output path is invalid",
            subsystem="tts",
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise BackendError(
            "tts.supertonic3.output_not_empty",
            "Supertonic qualification output must be new or empty",
            subsystem="tts",
        )
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    return output


@_deny_network_during_pilot
def run_supertonic3_quality_qualification(
    model_directory: str | Path,
    output_directory: str | Path,
    *,
    voice: str = DEFAULT_VOICE,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    speed: float = DEFAULT_SPEED,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    model_path = Path(model_directory).expanduser()
    validation = validate_supertonic3_model_directory(model_path)
    if not validation.ready:
        raise BackendError(validation.code, "Supertonic model validation failed", subsystem="tts")
    if voice not in VOICE_NAMES:
        raise BackendError("tts.supertonic3.voice_invalid", "Supertonic voice is invalid", subsystem="tts")
    if (
        not isinstance(total_steps, int) or isinstance(total_steps, bool) or not 1 <= total_steps <= 100
        or not isinstance(speed, (int, float)) or isinstance(speed, bool) or not 0.7 <= speed <= 2.0
        or not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32 - len(QUALITY_FIXTURES)
    ):
        raise BackendError(
            "tts.supertonic3.settings_invalid",
            "Supertonic qualification settings are invalid",
            subsystem="tts",
        )
    output = _prepare_output_directory(output_directory)
    try:
        import numpy as np
        from supertonic import TTS
    except ImportError as exc:
        raise BackendError(
            "tts.supertonic3.dependencies_missing",
            "Supertonic qualification dependencies are unavailable",
            subsystem="tts",
        ) from exc

    load_started = time.perf_counter()
    tts = TTS(
        model="supertonic-3",
        model_dir=model_path,
        auto_download=False,
        intra_op_num_threads=4,
        inter_op_num_threads=1,
    )
    load_seconds = time.perf_counter() - load_started
    if tts.sample_rate != SAMPLE_RATE or tuple(tts.voice_style_names) != VOICE_NAMES:
        raise BackendError(
            "tts.supertonic3.runtime_schema_invalid",
            "Supertonic runtime did not expose the pinned model schema",
            subsystem="tts",
        )
    style = tts.get_voice_style(voice)
    samples: list[dict[str, object]] = []
    first_waveform = None
    for index, (fixture_id, text) in enumerate(QUALITY_FIXTURES, start=1):
        fixture_seed = seed + index
        np.random.seed(fixture_seed)
        started = time.perf_counter()
        waveform, duration = tts.synthesize(
            text,
            voice_style=style,
            total_steps=total_steps,
            speed=speed,
            max_chunk_length=300,
            silence_duration=0.2,
            lang="el",
        )
        inference_seconds = time.perf_counter() - started
        try:
            audio_seconds = float(duration[0])
            valid_audio = (
                0 < audio_seconds < float("inf")
                and waveform.ndim == 2 and waveform.shape[0] == 1 and waveform.size > 0
                and bool(np.isfinite(waveform).all())
            )
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            valid_audio = False
        if not valid_audio:
            raise BackendError("tts.supertonic3.audio_invalid", "Supertonic returned invalid audio", subsystem="tts")
        path = output / f"{index:02d}-{fixture_id}.wav"
        tts.save_audio(waveform, str(path))
        if first_waveform is None:
            first_waveform = waveform.copy()
        samples.append(
            {
                "fixture_id": fixture_id,
                "text": text,
                "seed": fixture_seed,
                "audio_seconds": round(audio_seconds, 6),
                "inference_seconds": round(inference_seconds, 6),
                "realtime_factor": round(inference_seconds / audio_seconds, 6),
                "output_file": path.name,
                "output_sha256": _sha256(path),
            }
        )

    np.random.seed(seed + 1)
    repeated, _ = tts.synthesize(
        QUALITY_FIXTURES[0][1],
        voice_style=style,
        total_steps=total_steps,
        speed=speed,
        max_chunk_length=300,
        silence_duration=0.2,
        lang="el",
    )
    deterministic = bool(np.array_equal(first_waveform, repeated))
    mean_rtf = sum(float(sample["realtime_factor"]) for sample in samples) / len(samples)
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    gates = {
        "pinned_artifacts": validation.ready,
        "all_twenty_fixtures_generated": len(samples) == len(QUALITY_FIXTURES),
        "deterministic_fixed_seed": deterministic,
        "load_time_within_limit": load_seconds <= MAX_LOAD_SECONDS,
        "mean_realtime_factor_within_limit": mean_rtf <= MAX_MEAN_REALTIME_FACTOR,
        "peak_rss_within_limit": peak_rss_bytes <= MAX_PEAK_RSS_BYTES,
    }
    return {
        "schema_version": 1,
        "ready": all(gates.values()),
        "code": (
            "tts.supertonic3.quality_generation_completed"
            if all(gates.values())
            else "tts.supertonic3.quality_generation_failed"
        ),
        "model": {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
        },
        "runtime": {
            "supertonic_version": SUPERTONIC_VERSION,
            "voice": voice,
            "language": "el",
            "sample_rate": SAMPLE_RATE,
            "total_steps": total_steps,
            "speed": speed,
            "seed": seed,
            "load_seconds": round(load_seconds, 6),
            "mean_realtime_factor": round(mean_rtf, 6),
            "peak_rss_bytes": peak_rss_bytes,
        },
        "automated_gates": gates,
        "samples": samples,
        "network_used": False,
        "asr_round_trip_measured": False,
        "human_voice_approved": False,
        "incremental_audio_streaming": False,
        "interrupt_during_generation": False,
        "llm_coexistence_measured": False,
        "production_backend_changed": False,
        "production_promotion_allowed": False,
    }


_DIGIT_WORDS = {
    "0": "μηδέν",
    "1": "ένα",
    "2": "δύο",
    "3": "τρία",
    "4": "τέσσερα",
    "5": "πέντε",
    "6": "έξι",
    "7": "επτά",
    "8": "οκτώ",
    "9": "εννέα",
}


def _normalized_words(text: str) -> list[str]:
    value = unicodedata.normalize("NFC", text).casefold()
    words = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
    return [_DIGIT_WORDS.get(word, word) for word in words]


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


def score_supertonic3_transcripts(transcripts: dict[str, str]) -> dict[str, object]:
    expected = {fixture_id for fixture_id, _ in QUALITY_FIXTURES}
    if set(transcripts) != expected or any(not isinstance(value, str) for value in transcripts.values()):
        raise BackendError(
            "tts.supertonic3.transcripts_incomplete",
            "Supertonic qualification transcripts are incomplete",
            subsystem="tts",
        )
    samples: list[dict[str, object]] = []
    total_errors = 0
    total_words = 0
    for fixture_id, reference_text in QUALITY_FIXTURES:
        reference = _normalized_words(reference_text)
        hypothesis = transcripts[fixture_id]
        errors = _word_error_distance(reference, _normalized_words(hypothesis))
        total_errors += errors
        total_words += len(reference)
        samples.append(
            {
                "fixture_id": fixture_id,
                "reference_words": len(reference),
                "word_errors": errors,
                "wer_percent": round(100.0 * errors / len(reference), 3),
                "transcript": hypothesis,
            }
        )
    micro_wer = 100.0 * total_errors / total_words
    pronunciation_gate = micro_wer <= MAX_ASR_MICRO_WER_PERCENT
    return {
        "schema_version": 1,
        "ready": True,
        "code": "tts.supertonic3.asr_scoring_completed",
        "fixture_count": len(samples),
        "total_reference_words": total_words,
        "total_word_errors": total_errors,
        "micro_wer_percent": round(micro_wer, 3),
        "near_perfect_pronunciation_gate": pronunciation_gate,
        "near_perfect_pronunciation_limit_percent": MAX_ASR_MICRO_WER_PERCENT,
        "samples": samples,
        "network_used": False,
        "human_voice_approved": False,
        "production_promotion_allowed": False,
    }


@_deny_network_during_pilot
def score_supertonic3_outputs(
    output_directory: str | Path,
    whisper_model_directory: str | Path,
) -> dict[str, object]:
    output = Path(output_directory).expanduser()
    model_directory = Path(whisper_model_directory).expanduser()
    if not output.is_absolute() or not output.is_dir() or output.is_symlink():
        raise BackendError("tts.supertonic3.output_invalid", "Qualification output is invalid", subsystem="tts")
    if not model_directory.is_absolute() or not model_directory.is_dir() or model_directory.is_symlink():
        raise BackendError(
            "tts.supertonic3.whisper_model_invalid",
            "Local Whisper model directory is invalid",
            subsystem="tts",
        )
    validation = manifest.validate_model_directory(str(model_directory), manifest.BASE_MODEL_MANIFEST)
    if not validation.ok:
        raise BackendError(validation.code, "Pinned Whisper model validation failed", subsystem="tts")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise BackendError(
            "tts.supertonic3.whisper_dependency_missing",
            "Local faster-whisper dependency is unavailable",
            subsystem="tts",
        ) from exc
    model = WhisperModel(str(model_directory), device="cpu", compute_type="int8", local_files_only=True)
    transcripts: dict[str, str] = {}
    for index, (fixture_id, _) in enumerate(QUALITY_FIXTURES, start=1):
        path = output / f"{index:02d}-{fixture_id}.wav"
        try:
            valid = path.is_file() and not path.is_symlink() and path.stat().st_size > 44
        except OSError:
            valid = False
        if not valid:
            raise BackendError(
                "tts.supertonic3.output_missing",
                "Supertonic qualification output is missing",
                subsystem="tts",
            )
        segments, _ = model.transcribe(
            str(path),
            language="el",
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        transcripts[fixture_id] = " ".join(segment.text.strip() for segment in segments).strip()
    report = score_supertonic3_transcripts(transcripts)
    report["asr"] = {
        "backend": "faster_whisper",
        "model_id": manifest.BASE_MODEL_MANIFEST.model_id,
        "model_revision": manifest.BASE_MODEL_MANIFEST.revision,
        "model_path_explicit": True,
        "compute_type": "int8",
    }
    return report
