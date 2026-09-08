"""Deterministic, offline integrity inventory for an onedir distribution."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from .errors import BackendError

MANIFEST_NAME = "distribution-manifest.json"
SCHEMA_VERSION = 1


def _existing_root(directory: str | Path) -> Path:
    try:
        root = Path(directory).resolve(strict=True)
    except OSError as exc:
        raise BackendError("packaging.distribution_missing", "Distribution directory does not exist", subsystem="packaging") from exc
    if not root.is_dir():
        raise BackendError("packaging.distribution_missing", "Distribution directory does not exist", subsystem="packaging")
    return root


@dataclass(frozen=True, slots=True)
class DistributionVerification:
    ok: bool
    code: str
    files_checked: int
    artifact_name: str | None = None
    manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"ok": self.ok, "code": self.code, "files_checked": self.files_checked}
        if self.artifact_name is not None:
            result["artifact_name"] = self.artifact_name
        if self.manifest_sha256 is not None:
            result["manifest_sha256"] = self.manifest_sha256
        return result


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unreadable(exc: OSError) -> NoReturn:
    raise BackendError("packaging.distribution_unreadable", "Distribution inventory could not be read", subsystem="packaging") from exc


def _distribution_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise BackendError("packaging.distribution_missing", "Distribution directory does not exist", subsystem="packaging")
    files: list[Path] = []

    for current, directories, names in os.walk(root, followlinks=False, onerror=_unreadable):
        current_path = Path(current)
        for name in directories + names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            try:
                metadata = path.lstat()
            except OSError as exc:
                _unreadable(exc)
            if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
                raise BackendError("packaging.symlink_forbidden", "Distribution symlinks are forbidden", subsystem="packaging", safe_context={"artifact_name": relative})
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise BackendError("packaging.artifact_type_invalid", "Distribution artifacts must be regular files", subsystem="packaging", safe_context={"artifact_name": relative})
            if relative != MANIFEST_NAME:
                files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def build_distribution_manifest(directory: str | Path) -> dict[str, object]:
    root = _existing_root(directory)
    records = []
    try:
        for path in _distribution_files(root):
            relative = path.relative_to(root).as_posix()
            records.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
    except OSError as exc:
        _unreadable(exc)
    return {"schema_version": SCHEMA_VERSION, "product": "jarvis-voice", "files": records}


def write_distribution_manifest(directory: str | Path) -> Path:
    root = _existing_root(directory)
    manifest = build_distribution_manifest(root)
    output = root / MANIFEST_NAME
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".jarvis-manifest-", dir=root, delete=False) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(output)
        return output
    except OSError as exc:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise BackendError("packaging.manifest_write_failed", "Distribution manifest could not be written", subsystem="packaging") from exc


def _safe_manifest_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.as_posix() != value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def verify_distribution(directory: str | Path) -> DistributionVerification:
    root = _existing_root(directory)
    manifest_path = root / MANIFEST_NAME
    try:
        metadata = manifest_path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise OSError("unsafe manifest type")
        manifest_bytes = manifest_path.read_bytes()
        raw = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise BackendError("packaging.manifest_unreadable", "Distribution manifest is missing or invalid", subsystem="packaging") from exc
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw.get("schema_version") != SCHEMA_VERSION or raw.get("product") != "jarvis-voice" or not isinstance(raw.get("files"), list):
        return DistributionVerification(False, "packaging.manifest_schema_invalid", 0)
    expected: dict[str, tuple[int, str]] = {}
    declared_order: list[str] = []
    for item in raw["files"]:
        if not isinstance(item, dict):
            return DistributionVerification(False, "packaging.manifest_schema_invalid", 0)
        relative = _safe_manifest_path(item.get("path"))
        size, digest = item.get("size"), item.get("sha256")
        try:
            digest_is_hex = isinstance(digest, str) and len(digest) == 64 and len(bytes.fromhex(digest)) == 32
        except ValueError:
            digest_is_hex = False
        if relative is None or relative in expected or not isinstance(size, int) or isinstance(size, bool) or size < 0 or not digest_is_hex:
            return DistributionVerification(False, "packaging.manifest_schema_invalid", 0, relative)
        expected[relative] = (size, digest)
        declared_order.append(relative)
    if declared_order != sorted(declared_order):
        return DistributionVerification(False, "packaging.manifest_schema_invalid", 0)
    actual_files = _distribution_files(root)
    actual = {path.relative_to(root).as_posix(): path for path in actual_files}
    missing = sorted(set(expected) - set(actual))
    if missing:
        return DistributionVerification(False, "packaging.artifact_missing", 0, missing[0])
    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        return DistributionVerification(False, "packaging.artifact_unexpected", 0, unexpected[0])
    checked = 0
    try:
        for relative in sorted(expected):
            expected_size, expected_hash = expected[relative]
            path = actual[relative]
            if path.stat().st_size != expected_size:
                return DistributionVerification(False, "packaging.artifact_size_mismatch", checked, relative)
            if _sha256(path) != expected_hash:
                return DistributionVerification(False, "packaging.artifact_hash_mismatch", checked, relative)
            checked += 1
    except OSError as exc:
        _unreadable(exc)
    return DistributionVerification(True, "packaging.distribution_valid", checked, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
