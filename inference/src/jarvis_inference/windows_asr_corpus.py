"""Strict metadata builder for the private Windows ASR qualification corpus.

The 200 WAV files and their exact UTF-8 reference sidecars are deliberately
never part of this package or repository.  This module sees their bytes only
long enough to validate format, duration and identity, then writes a
content-free manifest.  The native runner revalidates that manifest before it
delegates to the separately pinned Windows collector.
"""

from __future__ import annotations

import hashlib
import io
import os
import stat
import tempfile
import unicodedata
import wave
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from .errors import ApplianceError
from .manifest import _read_manifest_file
from .strict_json import canonical_bytes, parse_body


SCHEMA_VERSION = 1
CORPUS_ID = "cyrus-windows-asr-audio-corpus-v1"
FIXTURE_ID = "cyrus-windows-asr-coexistence-v1"
ENTRIES_PER_LOCALE = 100
COEXISTENCE_ENTRIES_PER_LOCALE = 50
ENTRY_COUNT = ENTRIES_PER_LOCALE * 2
MAX_MANIFEST_BYTES = 262_144
MAX_AUDIO_BYTES = 16 * 1024 * 1024
MAX_REFERENCE_BYTES = 4_096
MAX_REFERENCE_WORDS = 128
_LOCALES = (("el", "el-GR"), ("en", "en-US"))


class WindowsAsrCorpusError(ValueError):
    """A content-free corpus metadata validation error."""


def _is_local_windows_absolute(value: str) -> bool:
    path = PureWindowsPath(value)
    return bool(
        path.is_absolute()
        and len(path.drive) == 2
        and path.drive[0].isalpha()
        and path.drive[1] == ":"
    )


def _native_drive_type(root: str) -> int:  # pragma: no cover - native Windows only
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetDriveTypeW(root))
    except BaseException:
        return 0


def _windows_drive_is_local(
    value: str,
    *,
    drive_type_reader: Callable[[str], int] | None = None,
) -> bool:
    path = PureWindowsPath(value)
    if not _is_local_windows_absolute(value):
        return False
    reader = drive_type_reader or _native_drive_type
    try:
        return reader(path.anchor) in {2, 3, 5, 6}
    except BaseException:
        return False


def _is_reparse(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _validate_existing_chain(
    path: Path,
    *,
    final_directory: bool,
    name: str,
) -> None:
    if (
        not path.is_absolute()
        or any(component in {".", ".."} for component in path.parts)
        or (os.name == "nt" and not _windows_drive_is_local(str(path)))
    ):
        raise WindowsAsrCorpusError(f"unsafe {name} path")
    current = Path(path.anchor)
    components = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(components):
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise WindowsAsrCorpusError(f"unsafe {name} path") from exc
        if current.is_symlink() or _is_reparse(current):
            raise WindowsAsrCorpusError(f"unsafe {name} path")
        is_final = index == len(components) - 1
        if not is_final and not stat.S_ISDIR(metadata.st_mode):
            raise WindowsAsrCorpusError(f"unsafe {name} path")
        if is_final:
            expected = stat.S_ISDIR if final_directory else stat.S_ISREG
            if not expected(metadata.st_mode):
                raise WindowsAsrCorpusError(f"unsafe {name} path")


def _safe_regular(path: Path, *, maximum: int, name: str) -> bytes:
    _validate_existing_chain(path, final_directory=False, name=name)
    try:
        raw = _read_manifest_file(path, maximum=maximum)
    except (ApplianceError, OSError) as exc:
        raise WindowsAsrCorpusError(f"unsafe {name}") from exc
    if not raw:
        raise WindowsAsrCorpusError(f"unsafe {name}")
    return raw


def _resolve_under(root: Path, relative: str, name: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
    ):
        raise WindowsAsrCorpusError(f"unsafe {name} path")
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.as_posix() != relative
    ):
        raise WindowsAsrCorpusError(f"unsafe {name} path")
    if not root.is_absolute():
        raise WindowsAsrCorpusError(f"unsafe {name} path")
    _validate_existing_chain(root, final_directory=True, name=name)

    # Lexical containment is necessary but insufficient on Windows: a junction
    # in an intermediate component can redirect a seemingly relative entry
    # outside the private corpus.  Inspect every component without resolving
    # it, so neither symlinks nor reparse points are ever followed.
    current = root
    parts = candidate.parts
    for index, component in enumerate(parts):
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise WindowsAsrCorpusError(f"unsafe {name} path") from exc
        if current.is_symlink() or _is_reparse(current):
            raise WindowsAsrCorpusError(f"unsafe {name} path")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise WindowsAsrCorpusError(f"unsafe {name} path")
    return current


def _expected_entries() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for language, locale in _LOCALES:
        for ordinal in range(1, ENTRIES_PER_LOCALE + 1):
            case_id = f"asr-{language}-{ordinal:03d}"
            result.append(
                {
                    "case_id": case_id,
                    "locale": locale,
                    "relative_path": f"audio/{language}/{case_id}.wav",
                    "reference_relative_path": (
                        f"references/{language}/{case_id}.txt"
                    ),
                }
            )
    return result


def _identity(raw: bytes) -> tuple[int, str]:
    return len(raw), hashlib.sha256(raw).hexdigest()


def _wav_metadata(raw: bytes) -> int:
    try:
        with wave.open(io.BytesIO(raw), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frames = reader.getnframes()
            compression = reader.getcomptype()
            actual_bytes = len(reader.readframes(frames + 1))
    except (OSError, EOFError, wave.Error) as exc:
        raise WindowsAsrCorpusError("invalid WAV fixture") from exc
    if (
        channels != 1
        or sample_width != 2
        or sample_rate != 16_000
        or frames <= 0
        or compression != "NONE"
        or actual_bytes != frames * channels * sample_width
    ):
        raise WindowsAsrCorpusError("WAV fixture format mismatch")
    duration_ms = (frames * 1000 + sample_rate // 2) // sample_rate
    if not 1500 <= duration_ms <= 30000:
        raise WindowsAsrCorpusError("WAV fixture duration is out of bounds")
    return duration_ms


def normalize_words(text: str) -> tuple[str, ...]:
    """Return the frozen Unicode word sequence used for aggregate WER.

    NFC + casefold is followed by retaining only Unicode letters, numbers and
    combining marks.  Every other code point is a separator.  The result is
    deterministic, locale-independent and never persisted by the collector.
    """

    if not isinstance(text, str):
        raise WindowsAsrCorpusError("invalid transcript text")
    normalized = unicodedata.normalize("NFC", text).casefold()
    separated = "".join(
        character if unicodedata.category(character)[0] in {"L", "M", "N"} else " "
        for character in normalized
    )
    return tuple(separated.split())


def _reference_metadata(path: Path) -> tuple[int, str, int]:
    raw = _safe_regular(
        path,
        maximum=MAX_REFERENCE_BYTES,
        name="reference fixture",
    )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise WindowsAsrCorpusError("invalid reference fixture")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise WindowsAsrCorpusError("invalid reference fixture") from exc
    if (
        unicodedata.normalize("NFC", text) != text
        or text != text.strip()
        or any(unicodedata.category(character).startswith("C") for character in text)
    ):
        raise WindowsAsrCorpusError("invalid reference fixture")
    words = normalize_words(text)
    if not 1 <= len(words) <= MAX_REFERENCE_WORDS:
        raise WindowsAsrCorpusError("invalid reference fixture")
    return len(raw), hashlib.sha256(raw).hexdigest(), len(words)


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise WindowsAsrCorpusError(f"invalid {name} shape")
    return value


def _validate_template(document: Any) -> None:
    root = _exact(
        document,
        {
            "template_schema_version",
            "corpus_id",
            "fixture_id",
            "audio_format",
            "reference_format",
            "schedule",
            "layout",
        },
        "corpus template",
    )
    if (
        root["template_schema_version"] != 1
        or root["corpus_id"] != CORPUS_ID
        or root["fixture_id"] != FIXTURE_ID
        or root["audio_format"]
        != {
            "container": "wav",
            "sample_rate_hz": 16000,
            "channels": 1,
            "sample_width_bytes": 2,
            "minimum_duration_ms": 1500,
            "maximum_duration_ms": 30000,
        }
        or root["reference_format"]
        != {
            "encoding": "utf-8",
            "normalization": "NFC",
            "maximum_bytes": MAX_REFERENCE_BYTES,
            "maximum_words": MAX_REFERENCE_WORDS,
            "controls_allowed": False,
            "word_normalization": "unicode-lmn-casefold-v1",
        }
        or root["schedule"]
        != {
            "kind": "serial-continuous-from-audio-start",
            "inference_lead_in_ms": 2000,
            "order": "fixed-case-id-ascending",
            "coexistence_selection": "first-50-per-locale",
        }
        or root["layout"]
        != {
            "entries_per_locale": ENTRIES_PER_LOCALE,
            "coexistence_entries_per_locale": COEXISTENCE_ENTRIES_PER_LOCALE,
            "locales": {"el": "el-GR", "en": "en-US"},
            "path_pattern": "audio/{language}/asr-{language}-{ordinal:03d}.wav",
            "reference_path_pattern": (
                "references/{language}/asr-{language}-{ordinal:03d}.txt"
            ),
        }
    ):
        raise WindowsAsrCorpusError("corpus template contract mismatch")


def load_template(path: Path) -> dict[str, Any]:
    raw = _safe_regular(path, maximum=MAX_MANIFEST_BYTES, name="corpus template")
    value = parse_body(raw)
    _validate_template(value)
    return value


def build_corpus_manifest(corpus_root: Path, template: Path) -> dict[str, Any]:
    """Validate the private files and return their content-free manifest."""

    _validate_existing_chain(
        corpus_root,
        final_directory=True,
        name="corpus root",
    )
    _validate_template(load_template(template))
    entries: list[dict[str, Any]] = []
    for expected in _expected_entries():
        audio = _resolve_under(corpus_root, expected["relative_path"], "audio fixture")
        reference = _resolve_under(
            corpus_root,
            expected["reference_relative_path"],
            "reference fixture",
        )
        audio_raw = _safe_regular(
            audio,
            maximum=MAX_AUDIO_BYTES,
            name="audio fixture",
        )
        size_bytes, sha256 = _identity(audio_raw)
        reference_size, reference_sha256, reference_words = _reference_metadata(reference)
        entries.append(
            {
                **expected,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "duration_ms": _wav_metadata(audio_raw),
                "reference_size_bytes": reference_size,
                "reference_sha256": reference_sha256,
                "reference_word_count": reference_words,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "fixture_id": FIXTURE_ID,
        "audio_format": {
            "container": "wav",
            "sample_rate_hz": 16000,
            "channels": 1,
            "sample_width_bytes": 2,
            "minimum_duration_ms": 1500,
            "maximum_duration_ms": 30000,
        },
        "reference_format": {
            "encoding": "utf-8",
            "normalization": "NFC",
            "maximum_bytes": MAX_REFERENCE_BYTES,
            "maximum_words": MAX_REFERENCE_WORDS,
            "controls_allowed": False,
            "word_normalization": "unicode-lmn-casefold-v1",
        },
        "schedule": {
            "kind": "serial-continuous-from-audio-start",
            "inference_lead_in_ms": 2000,
            "order": "fixed-case-id-ascending",
            "coexistence_selection": "first-50-per-locale",
        },
        "entries": entries,
    }


def validate_corpus_manifest(path: Path, *, corpus_root: Path | None = None) -> dict[str, Any]:
    """Load a complete private manifest and optionally rehash its WAV files."""

    raw = _safe_regular(path, maximum=MAX_MANIFEST_BYTES, name="audio corpus")
    root = _exact(
        parse_body(raw),
        {
            "schema_version",
            "corpus_id",
            "fixture_id",
            "audio_format",
            "reference_format",
            "schedule",
            "entries",
        },
        "audio corpus",
    )
    if (
        root["schema_version"] != SCHEMA_VERSION
        or root["corpus_id"] != CORPUS_ID
        or root["fixture_id"] != FIXTURE_ID
        or root["audio_format"]
        != {
            "container": "wav",
            "sample_rate_hz": 16000,
            "channels": 1,
            "sample_width_bytes": 2,
            "minimum_duration_ms": 1500,
            "maximum_duration_ms": 30000,
        }
        or root["reference_format"]
        != {
            "encoding": "utf-8",
            "normalization": "NFC",
            "maximum_bytes": MAX_REFERENCE_BYTES,
            "maximum_words": MAX_REFERENCE_WORDS,
            "controls_allowed": False,
            "word_normalization": "unicode-lmn-casefold-v1",
        }
        or root["schedule"]
        != {
            "kind": "serial-continuous-from-audio-start",
            "inference_lead_in_ms": 2000,
            "order": "fixed-case-id-ascending",
            "coexistence_selection": "first-50-per-locale",
        }
        or not isinstance(root["entries"], list)
        or len(root["entries"]) != ENTRY_COUNT
    ):
        raise WindowsAsrCorpusError("audio corpus contract mismatch")
    expected = _expected_entries()
    for expected_entry, observed in zip(expected, root["entries"], strict=True):
        entry = _exact(
            observed,
            {
                "case_id",
                "locale",
                "relative_path",
                "reference_relative_path",
                "size_bytes",
                "sha256",
                "duration_ms",
                "reference_size_bytes",
                "reference_sha256",
                "reference_word_count",
            },
            "audio corpus entry",
        )
        if (
            {
                key: entry[key]
                for key in (
                    "case_id",
                    "locale",
                    "relative_path",
                    "reference_relative_path",
                )
            }
            != expected_entry
            or type(entry["size_bytes"]) is not int
            or not 0 < entry["size_bytes"] <= MAX_AUDIO_BYTES
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
            or type(entry["duration_ms"]) is not int
            or not 1500 <= entry["duration_ms"] <= 30000
            or type(entry["reference_size_bytes"]) is not int
            or not 0 < entry["reference_size_bytes"] <= MAX_REFERENCE_BYTES
            or not isinstance(entry["reference_sha256"], str)
            or len(entry["reference_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in entry["reference_sha256"]
            )
            or type(entry["reference_word_count"]) is not int
            or not 1 <= entry["reference_word_count"] <= MAX_REFERENCE_WORDS
        ):
            raise WindowsAsrCorpusError("audio corpus entry mismatch")
        if corpus_root is not None:
            audio = _resolve_under(corpus_root, entry["relative_path"], "audio fixture")
            audio_raw = _safe_regular(
                audio,
                maximum=MAX_AUDIO_BYTES,
                name="audio fixture",
            )
            size_bytes, sha256 = _identity(audio_raw)
            if size_bytes != entry["size_bytes"] or sha256 != entry["sha256"]:
                raise WindowsAsrCorpusError("audio corpus identity changed")
            if _wav_metadata(audio_raw) != entry["duration_ms"]:
                raise WindowsAsrCorpusError("audio corpus duration changed")
            reference = _resolve_under(
                corpus_root,
                entry["reference_relative_path"],
                "reference fixture",
            )
            reference_size, reference_sha256, reference_words = _reference_metadata(
                reference
            )
            if (
                reference_size != entry["reference_size_bytes"]
                or reference_sha256 != entry["reference_sha256"]
                or reference_words != entry["reference_word_count"]
            ):
                raise WindowsAsrCorpusError("reference corpus identity changed")
    return root


def coexistence_entries(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Select exactly IDs 001..050 for each locale from a validated manifest."""

    entries = document.get("entries")
    if not isinstance(entries, list) or len(entries) != ENTRY_COUNT:
        raise WindowsAsrCorpusError("audio corpus contract mismatch")
    by_id = {
        entry.get("case_id"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("case_id"), str)
    }
    expected_ids = [
        f"asr-{language}-{ordinal:03d}"
        for language, _locale in _LOCALES
        for ordinal in range(1, COEXISTENCE_ENTRIES_PER_LOCALE + 1)
    ]
    if len(by_id) != ENTRY_COUNT or any(case_id not in by_id for case_id in expected_ids):
        raise WindowsAsrCorpusError("coexistence corpus selection mismatch")
    selected = [by_id[case_id] for case_id in expected_ids]
    if [entry["case_id"] for entry in selected] != expected_ids:
        raise WindowsAsrCorpusError("coexistence corpus selection mismatch")
    return selected


def write_corpus_manifest(path: Path, document: dict[str, Any]) -> None:
    """Durably create one manifest without persisting audio or transcript data."""

    _validate_existing_chain(
        path.parent,
        final_directory=True,
        name="corpus manifest output",
    )
    if path.exists() or path.is_symlink() or _is_reparse(path):
        raise WindowsAsrCorpusError("unsafe corpus manifest output")
    raw = canonical_bytes(document)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise WindowsAsrCorpusError("audio corpus manifest is too large")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".audio-corpus-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if os.name != "nt":
                os.fchmod(stream.fileno(), 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
