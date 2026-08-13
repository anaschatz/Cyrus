"""Offline-first qualification harness for a future Chatterbox V3 backend.

This module is deliberately separate from the production ``TtsBackend``.  It
never installs packages or downloads model files.  Acquisition must happen in
an isolated environment after :func:`chatterbox_pilot_preflight` reports that
the host has enough disk headroom.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import resource
import shutil
import socket
import struct
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, NamedTuple

from ..errors import BackendError


CHATTERBOX_PACKAGE_VERSION = "0.1.7"
CHATTERBOX_SOURCE_REVISION = "65b18437192794391a0308a8f705b1e33e633948"
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


# Pinned artifacts used by ChatterboxMultilingualTTS.from_local(). The tokenizer
# is both hash-verified and structurally checked for the Greek language token.
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
    ArtifactSpec(
        "grapheme_mtl_merged_expanded_v1.json",
        69_989,
        "69632f47220a788a52ce2661d096453c5655e9bf25289d89a8d832c46ee07dbf",
    ),
)
TOKENIZER_FILENAME = "grapheme_mtl_merged_expanded_v1.json"
MODEL_DOWNLOAD_BYTES = sum(item.size for item in MODEL_ARTIFACTS)

PILOT_FIXTURES = (
    ("greek_greeting", "Καλημέρα. Είμαι ο Τζάρβις και είμαι έτοιμος να βοηθήσω."),
    ("greek_status", "Η σύνδεση ολοκληρώθηκε με επιτυχία."),
    ("greek_barge_in", "Θα σταματήσω αμέσως μόλις αρχίσεις να μιλάς."),
)
F1_REFERENCE_SHA256 = "72826663025609f540344a4adbab494d89405d0dfc44fe6202e124e7b6a1c192"
F1_QUALIFICATION_FIXTURES = (
    ("greeting", "Καλημέρα. Είμαι έτοιμη να σε βοηθήσω."),
    ("system_status", "Έλεγξα την κατάσταση του συστήματος και όλα λειτουργούν κανονικά."),
    ("calendar", "Η επόμενη συνάντησή σου είναι στις οκτώ και μισή."),
    ("reminder", "Αν θέλεις, μπορώ να σου υπενθυμίσω τις σημαντικές εργασίες."),
    ("articulation", "Η τεχνητή νοημοσύνη επεξεργάζεται πληροφορίες και προφέρει κάθε ελληνική λέξη με σταθερή και φυσική άρθρωση."),
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
    ("long_sentence", "Μόλις ολοκληρωθεί η επεξεργασία, θα συγκρίνω τα αποτελέσματα και θα σου παρουσιάσω μόνο τις σημαντικότερες διαφορές."),
    ("closing", "Ολοκλήρωσα τον έλεγχο. Πες μου όταν θέλεις να συνεχίσουμε."),
)
F1_MAX_STANDALONE_RSS_BYTES = 6 * GIB
F1_MAX_MEAN_REALTIME_FACTOR = 1.0
VOICE_CONVERSION_LABELS = (
    "serious-male",
    "serious-female",
    "serious-male-2",
    "serious-female-2",
    "serious-male-3",
    "serious-female-3",
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


def _chatterbox_source_revision_ready() -> bool:
    """Verify that the installed distribution came from the pinned V3 commit."""

    try:
        distribution = importlib.metadata.distribution("chatterbox-tts")
        direct_url = distribution.read_text("direct_url.json")
        parsed = json.loads(direct_url) if direct_url else {}
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, TypeError):
        return False
    vcs_info = parsed.get("vcs_info") if isinstance(parsed, dict) else None
    if not isinstance(vcs_info, dict):
        return False
    return (
        vcs_info.get("vcs") == "git"
        and vcs_info.get("commit_id") == CHATTERBOX_SOURCE_REVISION
        and vcs_info.get("requested_revision") == CHATTERBOX_SOURCE_REVISION
    )


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
    required = len(MODEL_ARTIFACTS)
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


def _validated_audio_path(value: str | Path, *, role: str) -> Path:
    path = Path(value).expanduser()
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise BackendError(
            f"tts.chatterbox.{role}_audio_missing",
            "Chatterbox pilot audio is unavailable",
            subsystem="tts",
        ) from exc
    invalid = (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or size <= 0
        or size > 25 * 1024 * 1024
        or path.suffix.lower() not in {".wav", ".flac", ".mp3", ".m4a"}
    )
    if not invalid and path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as stream:
                invalid = stream.getnchannels() <= 0 or stream.getframerate() <= 0 or stream.getnframes() <= 0
        except (OSError, EOFError, wave.Error):
            invalid = True
    if invalid:
        raise BackendError(
            f"tts.chatterbox.{role}_audio_invalid",
            "Chatterbox pilot audio is invalid",
            subsystem="tts",
        )
    return path


def _validated_f1_reference_audio(value: str | Path) -> Path:
    path = _validated_audio_path(value, role="reference")
    if path.suffix.lower() != ".wav" or _sha256(path) != F1_REFERENCE_SHA256:
        raise BackendError(
            "tts.chatterbox.f1_reference_mismatch",
            "Chatterbox F1 qualification requires the pinned STOMA F1 reference",
            subsystem="tts",
        )
    return path


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
    source_revision_probe: Callable[[], bool] = _chatterbox_source_revision_ready,
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
        "pkg_resources": bool(module_probe("pkg_resources")),
        "chatterbox_v3_source": bool(source_revision_probe()),
    }
    model = validate_chatterbox_model_directory(model_directory)
    dependencies_ready = all(dependencies.values())
    acquisition_required = (
        MODEL_DOWNLOAD_BYTES + MINIMUM_RUN_FREE_BYTES
        if dependencies_ready
        else MINIMUM_ACQUISITION_FREE_BYTES
    )
    acquisition_ready = host_ready and available >= acquisition_required
    run_ready = host_ready and available >= MINIMUM_RUN_FREE_BYTES and dependencies_ready and model.ready

    if not host_ready:
        code = "tts.chatterbox.host_unsupported"
    elif available < MINIMUM_RUN_FREE_BYTES:
        code = "tts.chatterbox.disk_space_critical"
    elif model.ready and dependencies_ready:
        code = "tts.chatterbox.pilot_ready"
    elif available < acquisition_required:
        code = "tts.chatterbox.disk_space_insufficient"
    elif not dependencies_ready:
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
            "acquisition_required_mib": acquisition_required // (1024 * 1024),
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
        "package": {
            "version": CHATTERBOX_PACKAGE_VERSION,
            "source_revision": CHATTERBOX_SOURCE_REVISION,
        },
        "network_used": False,
        "installation_performed": False,
        "runtime_started": False,
        "content_captured": False,
    }


def acquire_chatterbox_model(
    storage_root: str | Path,
    model_directory: str | Path,
) -> dict[str, object]:
    """Download only the pinned V3 runtime files and validate every binary."""

    root = Path(storage_root).expanduser()
    model_path = Path(model_directory).expanduser()
    report = chatterbox_pilot_preflight(root, model_path)
    if not report["acquisition_ready"]:
        raise BackendError(
            str(report["code"]),
            "Chatterbox model acquisition preflight failed",
            subsystem="tts",
        )
    try:
        resolved_root = root.resolve(strict=True)
        resolved_parent = model_path.parent.resolve(strict=True)
    except OSError as exc:
        raise BackendError(
            "tts.chatterbox.model_directory_invalid",
            "Chatterbox model directory is invalid",
            subsystem="tts",
        ) from exc
    if not model_path.is_absolute() or model_path.is_symlink() or not resolved_parent.is_relative_to(resolved_root):
        raise BackendError(
            "tts.chatterbox.model_directory_invalid",
            "Chatterbox model directory must be a direct private storage child",
            subsystem="tts",
        )

    validation = validate_chatterbox_model_directory(model_path)
    if validation.ready:
        return {
            "schema_version": 1,
            "ready": True,
            "code": "tts.chatterbox.model_already_ready",
            "network_used": False,
            "model": report["model"],
        }

    model_path.mkdir(mode=0o700, exist_ok=True)
    os.chmod(model_path, 0o700)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise BackendError(
            "tts.chatterbox.dependencies_missing",
            "Hugging Face Hub is unavailable",
            subsystem="tts",
        ) from exc

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        revision=MODEL_REVISION,
        allow_patterns=[item.filename for item in MODEL_ARTIFACTS],
        local_dir=model_path,
        max_workers=1,
    )
    validation = validate_chatterbox_model_directory(model_path)
    if not validation.ready:
        raise BackendError(
            validation.code,
            "Downloaded Chatterbox model failed validation",
            subsystem="tts",
        )
    for artifact in MODEL_ARTIFACTS:
        os.chmod(model_path / artifact.filename, 0o600)
    return {
        "schema_version": 1,
        "ready": True,
        "code": "tts.chatterbox.model_acquired",
        "network_used": True,
        "downloaded_bytes": MODEL_DOWNLOAD_BYTES,
        "model": {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "t3_model": T3_MODEL,
            "ready": validation.ready,
            "code": validation.code,
            "files_present": validation.files_present,
            "files_required": validation.files_required,
        },
    }


def _synchronize_mps(torch_module: object) -> None:
    mps = getattr(torch_module, "mps", None)
    synchronize = getattr(mps, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _configure_greek_only_tokenizer() -> None:
    """Prevent unrelated Chinese resources from loading during a Greek pilot."""

    from chatterbox.models.tokenizers.tokenizer import ChineseCangjieConverter

    def skip_cangjie_mapping(converter: object, model_dir: object = None) -> None:
        del model_dir
        setattr(converter, "word2cj", {})
        setattr(converter, "cj2word", {})

    def skip_chinese_segmenter(converter: object) -> None:
        setattr(converter, "segmenter", None)

    ChineseCangjieConverter._load_cangjie_mapping = skip_cangjie_mapping
    ChineseCangjieConverter._init_segmenter = skip_chinese_segmenter


def _deny_network_during_pilot(function: Callable[..., dict[str, object]]) -> Callable[..., dict[str, object]]:
    """Run a pilot call with Python socket creation and connects fail-closed."""

    @wraps(function)
    def guarded(*args: object, **kwargs: object) -> dict[str, object]:
        original_socket = socket.socket
        original_create_connection = socket.create_connection

        class OfflineSocket(original_socket):
            def connect(self, address: object) -> None:
                del address
                raise BackendError(
                    "tts.chatterbox.network_forbidden",
                    "Network access is forbidden during the Chatterbox pilot",
                    subsystem="tts",
                )

            def connect_ex(self, address: object) -> int:
                self.connect(address)
                return 1

        def deny_create_connection(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise BackendError(
                "tts.chatterbox.network_forbidden",
                "Network access is forbidden during the Chatterbox pilot",
                subsystem="tts",
            )

        socket.socket = OfflineSocket
        socket.create_connection = deny_create_connection
        try:
            return function(*args, **kwargs)
        finally:
            socket.socket = original_socket
            socket.create_connection = original_create_connection

    return guarded


def _mps_memory_bytes(torch_module: object, function_name: str) -> int | None:
    mps = getattr(torch_module, "mps", None)
    function = getattr(mps, function_name, None)
    if not callable(function):
        return None
    try:
        return int(function())
    except (RuntimeError, TypeError, ValueError):
        return None


def _run_chatterbox_synthesis_suite(
    model_directory: str | Path,
    output_directory: str | Path,
    reference_audio: str | Path | None,
    *,
    fixtures: tuple[tuple[str, str], ...],
    completion_code: str,
) -> dict[str, object]:
    model_path = Path(model_directory).expanduser()
    validation = validate_chatterbox_model_directory(model_path)
    if not validation.ready:
        raise BackendError(validation.code, "Chatterbox model directory is not ready", subsystem="tts")
    output = Path(output_directory).expanduser()
    if not output.is_absolute() or output.is_symlink():
        raise BackendError("tts.chatterbox.output_invalid", "Chatterbox pilot output is invalid", subsystem="tts")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    reference_path: Path | None = None
    if reference_audio is not None:
        reference_path = _validated_audio_path(reference_audio, role="reference")

    try:
        import torch
        import torchaudio
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as exc:  # pragma: no cover - exercised only in isolated pilot env
        raise BackendError("tts.chatterbox.dependencies_missing", "Chatterbox dependencies are unavailable", subsystem="tts") from exc
    if not torch.backends.mps.is_available():
        raise BackendError("tts.chatterbox.mps_unavailable", "Chatterbox MPS device is unavailable", subsystem="tts")

    _configure_greek_only_tokenizer()
    torch.manual_seed(0)
    load_started = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_local(model_path, device="mps", t3_model=T3_MODEL)
    _synchronize_mps(torch)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    if reference_path is not None:
        model.prepare_conditionals(str(reference_path))
        _synchronize_mps(torch)
    samples: list[dict[str, object]] = []
    peak_mps_allocated_bytes = _mps_memory_bytes(torch, "current_allocated_memory")
    peak_mps_driver_bytes = _mps_memory_bytes(torch, "driver_allocated_memory")
    for index, (fixture_id, text) in enumerate(fixtures, start=1):
        torch.manual_seed(10_000 + index)
        _synchronize_mps(torch)
        started = time.perf_counter()
        waveform = model.generate(text, language_id="el")
        _synchronize_mps(torch)
        generation_ms = (time.perf_counter() - started) * 1000.0
        peak_before = float(waveform.abs().max().item())
        normalization_gain = min(1.0, 0.95 / peak_before) if peak_before > 0.0 else 1.0
        if normalization_gain < 1.0:
            waveform = waveform * normalization_gain
        sample_count = int(waveform.shape[-1])
        duration_ms = sample_count * 1000.0 / float(model.sr)
        filename = f"{index:02d}-{fixture_id}.wav"
        destination = output / filename
        torchaudio.save(str(destination), waveform.detach().cpu(), model.sr)
        os.chmod(destination, 0o600)
        current_allocated = _mps_memory_bytes(torch, "current_allocated_memory")
        current_driver = _mps_memory_bytes(torch, "driver_allocated_memory")
        if current_allocated is not None:
            peak_mps_allocated_bytes = max(peak_mps_allocated_bytes or 0, current_allocated)
        if current_driver is not None:
            peak_mps_driver_bytes = max(peak_mps_driver_bytes or 0, current_driver)
        samples.append(
            {
                "fixture_id": fixture_id,
                "file": filename,
                "generation_ms": round(generation_ms, 3),
                "audio_duration_ms": round(duration_ms, 3),
                "realtime_factor": round(generation_ms / duration_ms, 4) if duration_ms else None,
                "peak_before_normalization": round(peak_before, 6),
                "normalization_gain": round(normalization_gain, 6),
            }
        )

    return {
        "schema_version": 1,
        "ready": True,
        "code": completion_code,
        "backend": "chatterbox_multilingual_v3",
        "device": "mps",
        "language": "el",
        "non_greek_initializers_disabled": True,
        "model": {"repo_id": MODEL_REPO_ID, "revision": MODEL_REVISION, "t3_model": T3_MODEL},
        "voice": {
            "mode": "reference" if reference_path is not None else "builtin",
            "reference_sha256": _sha256(reference_path) if reference_path is not None else None,
        },
        "model_load_ms": round(load_ms, 3),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "peak_mps_allocated_bytes": peak_mps_allocated_bytes,
        "peak_mps_driver_bytes": peak_mps_driver_bytes,
        "samples": samples,
        "network_used": False,
        "network_guard": "python_socket_fail_closed",
        "content_captured": False,
    }


@_deny_network_during_pilot
def run_chatterbox_pilot(
    model_directory: str | Path,
    output_directory: str | Path,
    reference_audio: str | Path | None = None,
) -> dict[str, object]:
    return _run_chatterbox_synthesis_suite(
        model_directory,
        output_directory,
        reference_audio,
        fixtures=PILOT_FIXTURES,
        completion_code="tts.chatterbox.pilot_completed",
    )


def _add_f1_qualification_gates(report: dict[str, object]) -> dict[str, object]:
    raw_samples = report.get("samples")
    samples = raw_samples if isinstance(raw_samples, list) else []
    realtime_factors = [
        float(sample["realtime_factor"])
        for sample in samples
        if isinstance(sample, dict) and sample.get("realtime_factor") is not None
    ]
    mean_realtime_factor = sum(realtime_factors) / len(realtime_factors) if realtime_factors else float("inf")
    voice = report.get("voice")
    reference_sha256 = voice.get("reference_sha256") if isinstance(voice, dict) else None
    peak_rss_bytes = report.get("peak_rss_bytes")
    rss_within_limit = (
        isinstance(peak_rss_bytes, int)
        and not isinstance(peak_rss_bytes, bool)
        and peak_rss_bytes <= F1_MAX_STANDALONE_RSS_BYTES
    )
    automated_gates = {
        "pinned_f1_reference": reference_sha256 == F1_REFERENCE_SHA256,
        "all_twenty_fixtures_generated": len(samples) == len(F1_QUALIFICATION_FIXTURES),
        "standalone_rss_within_limit": rss_within_limit,
        "mean_realtime_factor_within_limit": mean_realtime_factor <= F1_MAX_MEAN_REALTIME_FACTOR,
        "incremental_audio_streaming": False,
        "interrupt_during_generation": False,
    }
    report["qualification"] = {
        "profile": "stoma_f1_reference",
        "fixture_count": len(F1_QUALIFICATION_FIXTURES),
        "mean_realtime_factor": round(mean_realtime_factor, 4),
        "standalone_rss_limit_bytes": F1_MAX_STANDALONE_RSS_BYTES,
        "mean_realtime_factor_limit": F1_MAX_MEAN_REALTIME_FACTOR,
        "automated_gates": automated_gates,
        "automated_gates_passed": all(automated_gates.values()),
        "llm_coexistence_measured": False,
        "asr_round_trip_measured": False,
        "human_voice_approved": False,
        "production_promotion_allowed": False,
    }
    report["production_backend_changed"] = False
    return report


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


def score_f1_transcripts(transcripts: dict[str, str]) -> dict[str, object]:
    expected_ids = {fixture_id for fixture_id, _ in F1_QUALIFICATION_FIXTURES}
    if set(transcripts) != expected_ids or any(not isinstance(value, str) for value in transcripts.values()):
        raise BackendError(
            "tts.chatterbox.f1_transcripts_incomplete",
            "Chatterbox F1 qualification transcripts are incomplete",
            subsystem="tts",
        )
    samples: list[dict[str, object]] = []
    total_errors = 0
    total_words = 0
    for fixture_id, reference_text in F1_QUALIFICATION_FIXTURES:
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
        "code": "tts.chatterbox.f1_asr_scoring_completed",
        "fixture_count": len(samples),
        "total_reference_words": total_words,
        "total_word_errors": total_errors,
        "micro_wer_percent": round(micro_wer, 3),
        "samples": samples,
        "network_used": False,
        "human_voice_approved": False,
        "production_promotion_allowed": False,
    }


@_deny_network_during_pilot
def score_chatterbox_f1_outputs(
    output_directory: str | Path,
    whisper_model_directory: str | Path,
) -> dict[str, object]:
    output = Path(output_directory).expanduser()
    model_directory = Path(whisper_model_directory).expanduser()
    if not output.is_absolute() or not output.is_dir() or output.is_symlink():
        raise BackendError("tts.chatterbox.output_invalid", "Chatterbox pilot output is invalid", subsystem="tts")
    if not model_directory.is_absolute() or not model_directory.is_dir() or model_directory.is_symlink():
        raise BackendError(
            "tts.chatterbox.whisper_model_invalid",
            "Local Whisper model directory is invalid",
            subsystem="tts",
        )
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise BackendError(
            "tts.chatterbox.whisper_dependency_missing",
            "Local faster-whisper dependency is unavailable",
            subsystem="tts",
        ) from exc
    model = WhisperModel(str(model_directory), device="cpu", compute_type="int8", local_files_only=True)
    transcripts: dict[str, str] = {}
    for index, (fixture_id, _) in enumerate(F1_QUALIFICATION_FIXTURES, start=1):
        path = output / f"{index:02d}-{fixture_id}.wav"
        try:
            valid = path.is_file() and not path.is_symlink() and path.stat().st_size > 44
        except OSError:
            valid = False
        if not valid:
            raise BackendError(
                "tts.chatterbox.f1_output_missing",
                "Chatterbox F1 qualification output is missing",
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
    report = score_f1_transcripts(transcripts)
    report["asr"] = {"backend": "faster_whisper", "model_path_explicit": True, "compute_type": "int8"}
    return report


@_deny_network_during_pilot
def run_chatterbox_f1_qualification(
    model_directory: str | Path,
    output_directory: str | Path,
    reference_audio: str | Path,
) -> dict[str, object]:
    reference_path = _validated_f1_reference_audio(reference_audio)
    report = _run_chatterbox_synthesis_suite(
        model_directory,
        output_directory,
        reference_path,
        fixtures=F1_QUALIFICATION_FIXTURES,
        completion_code="tts.chatterbox.f1_qualification_completed",
    )
    return _add_f1_qualification_gates(report)


@_deny_network_during_pilot
def run_chatterbox_voice_conversion_pilot(
    model_directory: str | Path,
    output_directory: str | Path,
    source_audio: str | Path,
    target_voice_audio: str | Path,
    *,
    label: str,
) -> dict[str, object]:
    """Change speaker timbre while keeping the source audio's linguistic content."""

    model_path = Path(model_directory).expanduser()
    validation = validate_chatterbox_model_directory(model_path)
    if not validation.ready:
        raise BackendError(validation.code, "Chatterbox model directory is not ready", subsystem="tts")
    if label not in VOICE_CONVERSION_LABELS:
        raise BackendError(
            "tts.chatterbox.voice_conversion_label_invalid",
            "Chatterbox voice-conversion label is invalid",
            subsystem="tts",
        )
    source_path = _validated_audio_path(source_audio, role="source")
    target_path = _validated_audio_path(target_voice_audio, role="target")
    output = Path(output_directory).expanduser()
    if not output.is_absolute() or output.is_symlink():
        raise BackendError("tts.chatterbox.output_invalid", "Chatterbox pilot output is invalid", subsystem="tts")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)

    try:
        import torch
        import torchaudio
        from chatterbox.models.s3gen import S3Gen
        from chatterbox.vc import ChatterboxVC
    except ImportError as exc:  # pragma: no cover - isolated pilot only
        raise BackendError("tts.chatterbox.dependencies_missing", "Chatterbox dependencies are unavailable", subsystem="tts") from exc
    if not torch.backends.mps.is_available():
        raise BackendError("tts.chatterbox.mps_unavailable", "Chatterbox MPS device is unavailable", subsystem="tts")

    torch.manual_seed(20_001 if label.startswith("serious-male") else 20_002)
    load_started = time.perf_counter()
    s3gen = S3Gen()
    s3gen.load_state_dict(torch.load(model_path / "s3gen.pt", map_location="cpu", weights_only=True))
    s3gen.to("mps").eval()
    converter = ChatterboxVC(s3gen=s3gen, device="mps")
    _synchronize_mps(torch)
    load_ms = (time.perf_counter() - load_started) * 1000.0

    started = time.perf_counter()
    waveform = converter.generate(str(source_path), target_voice_path=str(target_path))
    _synchronize_mps(torch)
    generation_ms = (time.perf_counter() - started) * 1000.0
    if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.numel() == 0 or not bool(torch.isfinite(waveform).all()):
        raise BackendError("tts.chatterbox.audio_invalid", "Chatterbox voice conversion produced invalid audio", subsystem="tts")
    peak_before = float(waveform.abs().max().item())
    gain = min(1.0, 0.95 / peak_before) if peak_before > 0.0 else 1.0
    waveform = waveform * gain
    destination = output / f"greektts15-8bit-{label}.wav"
    torchaudio.save(str(destination), waveform.detach().cpu(), converter.sr)
    os.chmod(destination, 0o600)
    duration_ms = int(waveform.shape[-1]) * 1000.0 / float(converter.sr)

    return {
        "schema_version": 1,
        "ready": True,
        "code": "tts.chatterbox.voice_conversion_completed",
        "backend": "chatterbox_voice_conversion",
        "device": "mps",
        "language": "el",
        "label": label,
        "model": {"repo_id": MODEL_REPO_ID, "revision": MODEL_REVISION, "component": "s3gen"},
        "source_sha256": _sha256(source_path),
        "target_voice_sha256": _sha256(target_path),
        "model_load_ms": round(load_ms, 3),
        "generation_ms": round(generation_ms, 3),
        "audio_duration_ms": round(duration_ms, 3),
        "realtime_factor": round(generation_ms / duration_ms, 4) if duration_ms else None,
        "peak_before_normalization": round(peak_before, 6),
        "normalization_gain": round(gain, 6),
        "sample": {"file": destination.name, "sample_rate": converter.sr, "channels": 1},
        "watermarked": True,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "network_used": False,
        "network_guard": "python_socket_fail_closed",
        "production_backend_changed": False,
        "content_captured": False,
    }
