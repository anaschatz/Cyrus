"""Offline-only model manifest validation."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ModelFileSpec:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_id: str
    revision: str
    engine: str
    files: tuple[ModelFileSpec, ...]


@dataclass(frozen=True, slots=True)
class ManifestValidationResult:
    ok: bool
    code: str
    message: str
    files_checked: int = 0
    artifact_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "ok": self.ok,
            "code": self.code,
            "message": self.message,
            "files_checked": self.files_checked,
        }
        if self.artifact_name is not None:
            result["artifact_name"] = self.artifact_name
        return result


BASE_MODEL_MANIFEST = ModelManifest(
    model_id="Systran/faster-whisper-base",
    revision="ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    engine="faster-whisper/ctranslate2",
    files=(
        ModelFileSpec(
            "config.json",
            2_309,
            "56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a",
        ),
        ModelFileSpec(
            "model.bin",
            145_217_532,
            "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
        ),
        ModelFileSpec(
            "tokenizer.json",
            2_203_239,
            "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
        ),
        ModelFileSpec(
            "vocabulary.txt",
            459_861,
            "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
        ),
    ),
)


LARGE_V3_TURBO_MODEL_MANIFEST = ModelManifest(
    model_id="dropbox-dash/faster-whisper-large-v3-turbo",
    revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    engine="faster-whisper/ctranslate2-fp16",
    files=(
        ModelFileSpec(
            "config.json",
            2_263,
            "b0253ea6c0d3bea6b1e19e91a02acfd3b53f4467362efcb5a3e6b16c9b3a9b7e",
        ),
        ModelFileSpec(
            "model.bin",
            1_617_884_929,
            "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
        ),
        ModelFileSpec(
            "preprocessor_config.json",
            340,
            "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
        ),
        ModelFileSpec(
            "tokenizer.json",
            2_710_337,
            "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd",
        ),
        ModelFileSpec(
            "vocabulary.json",
            1_068_114,
            "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1",
        ),
    ),
)


MODEL_PROFILE_BASE = "base"
MODEL_PROFILE_LARGE_V3_TURBO = "large-v3-turbo"
MODEL_MANIFESTS = MappingProxyType(
    {
        MODEL_PROFILE_BASE: BASE_MODEL_MANIFEST,
        MODEL_PROFILE_LARGE_V3_TURBO: LARGE_V3_TURBO_MODEL_MANIFEST,
    }
)


def model_manifest_for_profile(profile: str) -> ModelManifest:
    """Return one closed built-in manifest; runtime discovery is forbidden."""

    try:
        return MODEL_MANIFESTS[profile]
    except KeyError as exc:
        raise ValueError("Unsupported Whisper model profile") from exc


def _sha256(model_file: Path) -> str:
    digest = hashlib.sha256()
    with model_file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_FILE_ATTRIBUTE_REPARSE_POINT = getattr(
    stat,
    "FILE_ATTRIBUTE_REPARSE_POINT",
    0x400,
)


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(
        getattr(metadata, "st_file_attributes", 0)
        & _FILE_ATTRIBUTE_REPARSE_POINT
    )


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
    """Accept a drive-letter root only when Win32 does not classify it remote."""

    path = PureWindowsPath(value)
    if not _is_local_windows_absolute(value):
        return False
    reader = drive_type_reader or _native_drive_type
    try:
        drive_type = reader(path.anchor)
    except BaseException:
        return False
    # DRIVE_REMOVABLE/FIXED/CDROM/RAMDISK are local namespaces. Unknown,
    # missing roots and DRIVE_REMOTE fail closed before any model-path stat.
    return drive_type in {2, 3, 5, 6}


def _is_real_directory_chain(path: Path) -> bool:
    """Reject a link/reparse point in every existing path component."""

    if not path.is_absolute():
        return False
    if os.name != "nt":
        # ``reject_links=True`` is exercised by portable unit tests. macOS
        # temporary roots commonly traverse the system `/private` compatibility
        # symlink, which says nothing about the Windows path policy. Inspect the
        # selected directory and its caller-controlled parent here; native
        # Windows performs the complete drive-root walk below.
        for current in (path.parent, path):
            try:
                metadata = current.lstat()
            except OSError:
                return False
            if not stat.S_ISDIR(metadata.st_mode) or _is_reparse(metadata):
                return False
        return True
    if not _windows_drive_is_local(str(path)):
        return False
    current = Path(path.anchor)
    try:
        root_metadata = current.lstat()
    except OSError:
        return False
    if not stat.S_ISDIR(root_metadata.st_mode) or _is_reparse(root_metadata):
        return False
    for component in path.parts[1:]:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError:
            return False
        if not stat.S_ISDIR(metadata.st_mode) or _is_reparse(metadata):
            return False
    return True


def validate_model_directory(
    model_directory: str | Path | None,
    manifest: ModelManifest = BASE_MODEL_MANIFEST,
    *,
    reject_links: bool | None = None,
) -> ManifestValidationResult:
    """Validate an explicit directory without resolving or downloading anything.

    Native Windows defaults to an exact regular-file pack: neither the model
    directory nor any member may be a symlink or reparse point.  The explicit
    option lets Windows preflight tests exercise that policy on another host,
    while preserving the legacy cache-symlink behavior on macOS.
    """

    if model_directory is None or not str(model_directory).strip():
        return ManifestValidationResult(
            False,
            "model.directory_required",
            "An explicit local model directory is required",
        )
    strict_files = os.name == "nt" if reject_links is None else reject_links
    directory = Path(model_directory).expanduser()
    if os.name == "nt" and not _windows_drive_is_local(str(directory)):
        return ManifestValidationResult(
            False,
            "model.unsafe_directory",
            "The model directory must be on a local non-network drive",
        )
    try:
        directory_metadata = directory.lstat() if strict_files else directory.stat()
    except OSError:
        return ManifestValidationResult(
            False,
            "model.directory_missing",
            "The configured local model directory does not exist",
        )
    if strict_files and (
        not stat.S_ISDIR(directory_metadata.st_mode)
        or not _is_real_directory_chain(directory)
    ):
        return ManifestValidationResult(
            False,
            "model.unsafe_directory",
            "The model directory must be a real non-reparse directory",
        )
    if not stat.S_ISDIR(directory_metadata.st_mode):
        return ManifestValidationResult(
            False,
            "model.directory_missing",
            "The configured local model directory does not exist",
        )

    expected_names = {spec.name for spec in manifest.files}
    try:
        entries = tuple(directory.iterdir())
        if strict_files:
            actual_names = {entry.name for entry in entries}
        else:
            actual_names = {
                entry.name
                for entry in entries
                if not entry.name.startswith(".")
                and (entry.is_file() or entry.is_symlink())
            }
    except OSError:
        return ManifestValidationResult(
            False,
            "model.directory_unreadable",
            "The configured local model directory cannot be enumerated",
        )
    missing = sorted(expected_names - actual_names)
    if missing:
        return ManifestValidationResult(
            False,
            "model.missing_artifact",
            "The model pack is missing a required artifact",
            artifact_name=missing[0],
        )
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        return ManifestValidationResult(
            False,
            "model.unexpected_artifact",
            "The model pack contains an unexpected runtime artifact",
            artifact_name=unexpected[0],
        )

    checked = 0
    for spec in manifest.files:
        model_file = directory / spec.name
        try:
            file_stat = model_file.lstat() if strict_files else model_file.stat()
        except OSError:
            return ManifestValidationResult(
                False,
                "model.unreadable_artifact",
                "A model artifact cannot be read",
                files_checked=checked,
                artifact_name=spec.name,
            )
        if strict_files and (
            not stat.S_ISREG(file_stat.st_mode)
            or _is_reparse(file_stat)
        ):
            return ManifestValidationResult(
                False,
                "model.unsafe_artifact",
                "A model artifact must be a real non-reparse regular file",
                files_checked=checked,
                artifact_name=spec.name,
            )
        if file_stat.st_size != spec.size:
            return ManifestValidationResult(
                False,
                "model.size_mismatch",
                "A model artifact has the wrong size",
                files_checked=checked,
                artifact_name=spec.name,
            )
        try:
            actual_hash = _sha256(model_file)
        except OSError:
            return ManifestValidationResult(
                False,
                "model.unreadable_artifact",
                "A model artifact cannot be read",
                files_checked=checked,
                artifact_name=spec.name,
            )
        if actual_hash != spec.sha256:
            return ManifestValidationResult(
                False,
                "model.hash_mismatch",
                "A model artifact failed SHA-256 verification",
                files_checked=checked,
                artifact_name=spec.name,
            )
        checked += 1

    return ManifestValidationResult(
        True,
        "model.manifest_valid",
        "All local model artifacts match the pinned manifest",
        files_checked=checked,
    )
