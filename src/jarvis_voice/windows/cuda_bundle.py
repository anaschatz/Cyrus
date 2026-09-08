"""Strict identity contract for the private Windows CUDA DLL bundle.

The CUDA and cuDNN binaries are operator-supplied release inputs.  They are
never downloaded by the application or inferred from ``PATH``/``CUDA_PATH``.
This module validates the private canonical manifest before PyInstaller is
allowed to copy any DLL into a sealed executable.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CUDA_BUNDLE_SCHEMA = "cyrus.windows-cuda-runtime-bundle/1"
RUNNER_BUNDLE_SCHEMA = "cyrus.windows-qualification-runners/1"
MAX_MANIFEST_BYTES = 128 * 1024
MAX_DLL_BYTES = 4 * 1024 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_DLL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.dll$", re.ASCII)
_LOCAL_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]", re.ASCII)
_CUDA_RUNTIME_PREFIXES = (
    "cublas",
    "cudnn",
    "cudart",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvjitlink",
    "nvrtc",
)
_ALLOWED_AUXILIARY_DLLS = frozenset({"zlibwapi.dll"})
_FORBIDDEN_DRIVER_DLLS = frozenset({"nvcuda.dll", "nvml.dll"})
REQUIRED_CUDA_DLLS = frozenset(
    {
        "cublas64_12.dll",
        "cublaslt64_12.dll",
        "cudnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_ops64_9.dll",
    }
)
QUALIFICATION_RUNNERS = (
    "cyrus-windows-asr-collector.exe",
    "cyrus-windows-primary-profile-runner.exe",
    "native-fixture-runner.exe",
)


class CudaBundleError(ValueError):
    """Content-free validation error with a stable code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class CudaDll:
    name: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CudaBundle:
    manifest_path: Path
    manifest_sha256: str
    root: Path
    files: tuple[CudaDll, ...]


PathValidator = Callable[[Path, str], Path]


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CudaBundleError("cuda_bundle.duplicate_json_key")
        result[key] = value
    return result


def _read_canonical_document(
    path: Path, validator: PathValidator
) -> tuple[dict[str, Any], bytes]:
    checked = validator(path, "file")
    try:
        with checked.open("rb") as handle:
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise CudaBundleError("cuda_bundle.manifest_unreadable") from exc
    if not raw or len(raw) > MAX_MANIFEST_BYTES:
        raise CudaBundleError("cuda_bundle.manifest_size_invalid")
    try:
        document = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                CudaBundleError("cuda_bundle.json_constant_invalid")
            ),
        )
        if not isinstance(document, dict):
            raise CudaBundleError("cuda_bundle.manifest_schema_invalid")
        canonical = canonical_json_bytes(document)
    except CudaBundleError:
        raise
    except (ValueError, RecursionError) as exc:
        raise CudaBundleError("cuda_bundle.manifest_json_invalid") from exc
    if raw != canonical:
        raise CudaBundleError("cuda_bundle.manifest_not_canonical")
    return document, raw


def _hash_regular_file(path: Path, expected_size: int) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                raise CudaBundleError("cuda_bundle.file_size_mismatch")
            remaining = expected_size + 1
            while remaining and (chunk := handle.read(min(remaining, 1024 * 1024))):
                remaining -= len(chunk)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            current = path.stat(follow_symlinks=False)
    except CudaBundleError:
        raise
    except OSError as exc:
        raise CudaBundleError("cuda_bundle.file_unreadable") from exc
    identities = {
        (
            item.st_dev,
            item.st_ino,
            stat.S_IFMT(item.st_mode),
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        for item in (before, after, current)
    }
    if remaining != 1 or len(identities) != 1:
        raise CudaBundleError("cuda_bundle.file_changed_during_hash")
    return digest.hexdigest()


def _windows_attributes(path: Path) -> int:
    get_attributes = ctypes.WinDLL(  # type: ignore[attr-defined]
        "kernel32", use_last_error=True
    ).GetFileAttributesW
    get_attributes.argtypes = [ctypes.c_wchar_p]
    get_attributes.restype = ctypes.c_uint32
    attributes = int(get_attributes(str(path)))
    if attributes == 0xFFFFFFFF:
        raise CudaBundleError("cuda_bundle.path_unreadable")
    return attributes


def _windows_drive_type(root: str) -> int:
    get_drive_type = ctypes.WinDLL(  # type: ignore[attr-defined]
        "kernel32", use_last_error=True
    ).GetDriveTypeW
    get_drive_type.argtypes = [ctypes.c_wchar_p]
    get_drive_type.restype = ctypes.c_uint
    return int(get_drive_type(root))


def require_native_windows_path(path: Path, kind: str) -> Path:
    """Require an existing fixed-drive path with no reparse component."""

    if sys.platform != "win32":
        raise CudaBundleError("cuda_bundle.native_windows_required")
    if kind not in {"file", "directory"}:
        raise CudaBundleError("cuda_bundle.path_kind_invalid")
    raw = str(path)
    if (
        _LOCAL_DRIVE_PATH.match(raw) is None
        or raw.startswith(("\\\\", "//", "\\\\?\\", "\\\\.\\"))
        or not path.is_absolute()
    ):
        raise CudaBundleError("cuda_bundle.path_not_local_absolute")
    normalized = Path(os.path.abspath(raw))
    drive_root = normalized.anchor
    if not drive_root or _windows_drive_type(drive_root) != 3:  # DRIVE_FIXED
        raise CudaBundleError("cuda_bundle.drive_not_fixed_local")
    chain = tuple(reversed(normalized.parents)) + (normalized,)
    for component in chain:
        if _windows_attributes(component) & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
            raise CudaBundleError("cuda_bundle.reparse_path_forbidden")
    try:
        metadata = normalized.stat(follow_symlinks=False)
    except OSError as exc:
        raise CudaBundleError("cuda_bundle.path_unreadable") from exc
    expected = (
        stat.S_ISREG(metadata.st_mode)
        if kind == "file"
        else stat.S_ISDIR(metadata.st_mode)
    )
    if not expected:
        raise CudaBundleError("cuda_bundle.path_type_invalid")
    return normalized


def load_cuda_bundle(
    manifest_path: Path,
    *,
    path_validator: PathValidator = require_native_windows_path,
) -> CudaBundle:
    """Validate and hash an exact private CUDA/cuDNN DLL directory."""

    checked_manifest = path_validator(Path(manifest_path), "file")
    document, manifest_bytes = _read_canonical_document(
        checked_manifest, path_validator
    )
    if set(document) != {"schema", "root", "files"}:
        raise CudaBundleError("cuda_bundle.manifest_schema_invalid")
    if document.get("schema") != CUDA_BUNDLE_SCHEMA:
        raise CudaBundleError("cuda_bundle.manifest_schema_invalid")
    root_value = document.get("root")
    files_value = document.get("files")
    if not isinstance(root_value, str) or not isinstance(files_value, list):
        raise CudaBundleError("cuda_bundle.manifest_schema_invalid")
    root = path_validator(Path(root_value), "directory")
    try:
        if checked_manifest.parent.samefile(root):
            raise CudaBundleError("cuda_bundle.manifest_inside_bundle")
    except OSError as exc:
        raise CudaBundleError("cuda_bundle.path_unreadable") from exc

    parsed: list[CudaDll] = []
    previous = ""
    seen: set[str] = set()
    for raw in files_value:
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "size_bytes",
            "sha256",
        }:
            raise CudaBundleError("cuda_bundle.file_schema_invalid")
        name = raw.get("name")
        size_bytes = raw.get("size_bytes")
        sha256 = raw.get("sha256")
        if (
            not isinstance(name, str)
            or _DLL_NAME.fullmatch(name) is None
            or ".." in name
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or not 1 <= size_bytes <= MAX_DLL_BYTES
            or not isinstance(sha256, str)
            or _SHA256.fullmatch(sha256) is None
        ):
            raise CudaBundleError("cuda_bundle.file_schema_invalid")
        folded = name.casefold()
        if folded in seen or (previous and folded <= previous):
            raise CudaBundleError("cuda_bundle.file_order_invalid")
        if folded in _FORBIDDEN_DRIVER_DLLS:
            raise CudaBundleError("cuda_bundle.driver_dll_forbidden")
        if not (
            folded.startswith(_CUDA_RUNTIME_PREFIXES)
            or folded in _ALLOWED_AUXILIARY_DLLS
        ):
            raise CudaBundleError("cuda_bundle.dll_name_not_allowed")
        seen.add(folded)
        previous = folded
        candidate = path_validator(root / name, "file")
        if candidate.parent != root or candidate.name != name:
            raise CudaBundleError("cuda_bundle.file_path_invalid")
        actual_sha256 = _hash_regular_file(candidate, size_bytes)
        if actual_sha256 != sha256:
            raise CudaBundleError("cuda_bundle.file_hash_mismatch")
        parsed.append(CudaDll(name, candidate, size_bytes, sha256))
    if not REQUIRED_CUDA_DLLS.issubset(seen):
        raise CudaBundleError("cuda_bundle.required_dll_missing")
    try:
        directory_names = {item.name.casefold() for item in root.iterdir()}
    except OSError as exc:
        raise CudaBundleError("cuda_bundle.path_unreadable") from exc
    if directory_names != seen:
        raise CudaBundleError("cuda_bundle.directory_not_exact")
    return CudaBundle(
        manifest_path=checked_manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        root=root,
        files=tuple(parsed),
    )


def is_cuda_runtime_name(name: str) -> bool:
    folded = Path(name).name.casefold()
    return folded.endswith(".dll") and (
        folded.startswith(_CUDA_RUNTIME_PREFIXES) or folded in _ALLOWED_AUXILIARY_DLLS
    )


def build_cuda_manifest(
    root: Path,
    *,
    path_validator: PathValidator = require_native_windows_path,
) -> dict[str, Any]:
    """Hash a reviewed private directory into the canonical bundle schema."""

    checked_root = path_validator(root, "directory")
    try:
        entries = sorted(checked_root.iterdir(), key=lambda item: item.name.casefold())
    except OSError as exc:
        raise CudaBundleError("cuda_bundle.path_unreadable") from exc
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        name = entry.name
        folded = name.casefold()
        if (
            _DLL_NAME.fullmatch(name) is None
            or ".." in name
            or folded in seen
            or folded in _FORBIDDEN_DRIVER_DLLS
            or not (
                folded.startswith(_CUDA_RUNTIME_PREFIXES)
                or folded in _ALLOWED_AUXILIARY_DLLS
            )
        ):
            raise CudaBundleError("cuda_bundle.directory_entry_invalid")
        checked = path_validator(entry, "file")
        size = checked.stat().st_size
        if not 1 <= size <= MAX_DLL_BYTES:
            raise CudaBundleError("cuda_bundle.file_size_invalid")
        seen.add(folded)
        files.append(
            {
                "name": name,
                "size_bytes": size,
                "sha256": _hash_regular_file(checked, size),
            }
        )
    if not REQUIRED_CUDA_DLLS.issubset(seen):
        raise CudaBundleError("cuda_bundle.required_dll_missing")
    return {"schema": CUDA_BUNDLE_SCHEMA, "root": str(checked_root), "files": files}


def create_runner_manifest(
    directory: Path,
    cuda_manifest_sha256: str,
    *,
    path_validator: PathValidator = require_native_windows_path,
) -> dict[str, Any]:
    if _SHA256.fullmatch(cuda_manifest_sha256) is None:
        raise CudaBundleError("runner_bundle.cuda_manifest_digest_invalid")
    files: list[dict[str, Any]] = []
    for name in QUALIFICATION_RUNNERS:
        path = directory / name
        checked = path_validator(path, "file")
        size = checked.stat().st_size
        if size <= 0:
            raise CudaBundleError("runner_bundle.file_size_invalid")
        files.append(
            {
                "name": name,
                "size_bytes": size,
                "sha256": _hash_regular_file(checked, size),
            }
        )
    return {
        "schema": RUNNER_BUNDLE_SCHEMA,
        "cuda_runtime_manifest_sha256": cuda_manifest_sha256,
        "files": files,
    }


def verify_runner_manifest(
    manifest_path: Path,
    *,
    path_validator: PathValidator = require_native_windows_path,
) -> dict[str, Any]:
    checked = path_validator(manifest_path, "file")
    document, _manifest_bytes = _read_canonical_document(checked, path_validator)
    if set(document) != {"schema", "cuda_runtime_manifest_sha256", "files"}:
        raise CudaBundleError("runner_bundle.manifest_schema_invalid")
    if document.get("schema") != RUNNER_BUNDLE_SCHEMA:
        raise CudaBundleError("runner_bundle.manifest_schema_invalid")
    cuda_digest = document.get("cuda_runtime_manifest_sha256")
    if not isinstance(cuda_digest, str) or _SHA256.fullmatch(cuda_digest) is None:
        raise CudaBundleError("runner_bundle.manifest_schema_invalid")
    files = document.get("files")
    if not isinstance(files, list) or len(files) != len(QUALIFICATION_RUNNERS):
        raise CudaBundleError("runner_bundle.manifest_schema_invalid")
    expected_names = list(QUALIFICATION_RUNNERS)
    actual_names: list[str] = []
    for raw in files:
        if not isinstance(raw, dict) or set(raw) != {"name", "size_bytes", "sha256"}:
            raise CudaBundleError("runner_bundle.file_schema_invalid")
        name, size, sha256 = raw.get("name"), raw.get("size_bytes"), raw.get("sha256")
        if (
            not isinstance(name, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha256, str)
            or _SHA256.fullmatch(sha256) is None
        ):
            raise CudaBundleError("runner_bundle.file_schema_invalid")
        actual_names.append(name)
        runner = path_validator(checked.parent / name, "file")
        if runner.parent != checked.parent or runner.name != name:
            raise CudaBundleError("runner_bundle.file_path_invalid")
        try:
            actual_sha256 = _hash_regular_file(runner, size)
        except CudaBundleError as exc:
            raise CudaBundleError("runner_bundle.file_hash_mismatch") from exc
        if actual_sha256 != sha256:
            raise CudaBundleError("runner_bundle.file_hash_mismatch")
    if actual_names != expected_names:
        raise CudaBundleError("runner_bundle.file_set_invalid")
    return document
