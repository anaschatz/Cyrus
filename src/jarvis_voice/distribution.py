"""Deterministic, offline integrity inventory for an onedir distribution."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

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

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"ok": self.ok, "code": self.code, "files_checked": self.files_checked}
        if self.artifact_name is not None:
            result["artifact_name"] = self.artifact_name
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_files(root: Path, manifest_name: str) -> list[Path]:
    if not root.is_dir():
        raise BackendError("packaging.distribution_missing", "Distribution directory does not exist", subsystem="packaging")
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            if (current_path / name).is_symlink():
                raise BackendError("packaging.symlink_forbidden", "Distribution symlinks are forbidden", subsystem="packaging", safe_context={"artifact_name": name})
        for name in names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative == manifest_name:
                continue
            if path.is_symlink():
                raise BackendError("packaging.symlink_forbidden", "Distribution symlinks are forbidden", subsystem="packaging", safe_context={"artifact_name": relative})
            if path.is_file():
                files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def build_distribution_manifest(directory: str | Path, *, manifest_name: str = MANIFEST_NAME) -> dict[str, object]:
    root = _existing_root(directory)
    records = []
    for path in _distribution_files(root, manifest_name):
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
    return {"schema_version": SCHEMA_VERSION, "product": "jarvis-voice", "files": records}


def write_distribution_manifest(directory: str | Path, *, manifest_name: str = MANIFEST_NAME) -> Path:
    root = _existing_root(directory)
    manifest = build_distribution_manifest(root, manifest_name=manifest_name)
    output = root / manifest_name
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
            Path(temporary_name).unlink(missing_ok=True)
        raise BackendError("packaging.manifest_write_failed", "Distribution manifest could not be written", subsystem="packaging") from exc


def _safe_manifest_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def verify_distribution(directory: str | Path, *, manifest_name: str = MANIFEST_NAME) -> DistributionVerification:
    root = _existing_root(directory)
    manifest_path = root / manifest_name
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("packaging.manifest_unreadable", "Distribution manifest is missing or invalid", subsystem="packaging") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION or raw.get("product") != "jarvis-voice" or not isinstance(raw.get("files"), list):
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
    actual_files = _distribution_files(root, manifest_name)
    actual = {path.relative_to(root).as_posix(): path for path in actual_files}
    missing = sorted(set(expected) - set(actual))
    if missing:
        return DistributionVerification(False, "packaging.artifact_missing", 0, missing[0])
    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        return DistributionVerification(False, "packaging.artifact_unexpected", 0, unexpected[0])
    checked = 0
    for relative in sorted(expected):
        expected_size, expected_hash = expected[relative]
        path = actual[relative]
        if path.stat().st_size != expected_size:
            return DistributionVerification(False, "packaging.artifact_size_mismatch", checked, relative)
        if _sha256(path) != expected_hash:
            return DistributionVerification(False, "packaging.artifact_hash_mismatch", checked, relative)
        checked += 1
    return DistributionVerification(True, "packaging.distribution_valid", checked)
