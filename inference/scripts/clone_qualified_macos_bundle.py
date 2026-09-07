#!/usr/bin/env python3
"""Clone a stopped, qualified macOS inference bundle into a clean directory.

The source is never modified.  The clone preserves every identity-bound byte
and every other ordinary bundle file, while dropping only unrecorded Finder
metadata or numeric conflict copies.  An unrecognized file below
``site-packages`` is not guessed away: it blocks the clone.

This command is intentionally outside the packaged ``jarvis_inference``
runtime.  Run it from a trusted maintenance environment with
``inference/src`` on ``PYTHONPATH`` and only while the source appliance is
stopped.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from jarvis_inference.limits import MANIFEST_BYTES
from jarvis_inference.errors import ApplianceError
from jarvis_inference.durable import generic_report_matches
from jarvis_inference.manifest import Manifest, _read_manifest_file, _verify_installed_wheel, load_manifest
from jarvis_inference.strict_json import canonical_bytes, parse_body
from jarvis_inference.python_environment import PYTHON_CONVENIENCE_LINKS, environment_digest, hash_file


REPORT_VERSION = "jarvis-qualified-bundle-clone/1"
QUALIFICATION_SUITE_VERSION = "jarvis-inference-qualification-v1"
_CONFLICT_COPY = re.compile(r"^.+ [2-9][0-9]*(?:\.[^./]+)?$", re.ASCII)
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_CANONICAL_NAME = re.compile(r"[-_.]+")
_CPYTHON_311_PYC = re.compile(
    r"^(?P<stem>.+)\.cpython-311(?:\.opt-[1-9][0-9]*)?\.pyc$",
    re.ASCII,
)
_PYTHON_CONVENIENCE_LINKS = {f"python/{path}": target for path, target in PYTHON_CONVENIENCE_LINKS.items()}
_MLX_SHARED_OWNER_VERSIONS = {
    "mlx": "0.32.0",
    "mlx-metal": "0.32.0",
}
_MLX_SHARED_WHEEL_MEMBERS = frozenset(
    {
        "mlx/__main__.py",
        "mlx/_reprlib_fix.py",
        "mlx/extension.py",
        "mlx/py.typed",
        "mlx/utils.py",
    }
)
_EXPECTED_EFFECTIVE_CONFIGURATION = {
    "blocked_write_seconds": 5,
    "host": "127.0.0.1",
    "port": 8766,
    "pre_output_retries": 1,
    "prefill_seconds": 120,
    "queue": False,
    "single_active_request": True,
    "stream_idle_seconds": 30,
    "total_request_seconds": 300,
}


class BundleCloneError(RuntimeError):
    """Fixed-code, content-free bundle clone failure."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    relative: str
    size_bytes: int
    sha256: str
    mode: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class WheelMemberIdentity:
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class WheelInspection:
    python_sources: dict[str, WheelMemberIdentity]
    shared_members: dict[str, WheelMemberIdentity]


@dataclass(frozen=True, slots=True)
class RecordOwnership:
    distribution: str
    row_path: str
    sha256: str | None
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class OmittedLink:
    relative: str
    target: str
    target_relative: str


@dataclass(frozen=True, slots=True)
class TreeSnapshot:
    root: Path
    files: dict[str, FileSnapshot]
    directories: frozenset[str]
    omitted_links: dict[str, OmittedLink]


@dataclass(frozen=True, slots=True)
class BundlePlan:
    tree: TreeSnapshot
    manifest_relative: str
    manifest_digest: str
    runtime_identity_digest: str
    record_files: frozenset[str]
    included_files: frozenset[str]
    excluded_files: frozenset[str]
    excluded_directories: frozenset[str]
    excluded_finder_files: int
    excluded_conflict_files: int
    excluded_python_cache_files: int
    omitted_python_runtime_links: int
    package_count: int


def _canonical_name(value: str) -> str:
    return _CANONICAL_NAME.sub("-", value).lower()


def _is_link_or_reparse_status(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _lstat(path: Path, code: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError:
        raise BundleCloneError(code) from None


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _assert_existing_path_has_no_links(path: Path, code: str) -> None:
    if not path.is_absolute():
        raise BundleCloneError(code)
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        metadata = _lstat(current, code)
        if _is_link_or_reparse_status(metadata):
            raise BundleCloneError(code)


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _safe_relative(value: str | Path, code: str) -> str:
    text = str(value)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or text.startswith(("/", "~"))
        or windows.is_absolute()
        or windows.drive
        or posix.is_absolute()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise BundleCloneError(code)
    return posix.as_posix()


def _hash_stream(stream: Any) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def _hash_file(path: Path) -> tuple[int, str]:
    try:
        return hash_file(path)
    except OSError:
        raise BundleCloneError("bundle_source_changed") from None


def _scan_tree(root: Path) -> TreeSnapshot:
    metadata = _lstat(root, "bundle_source_invalid")
    if _is_link_or_reparse_status(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise BundleCloneError("bundle_source_invalid")
    files: dict[str, FileSnapshot] = {}
    directories: set[str] = set()
    omitted_links: dict[str, OmittedLink] = {}

    def visit(directory: Path, relative_directory: PurePosixPath | None) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError:
            raise BundleCloneError("bundle_source_unreadable") from None
        for entry in entries:
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError:
                raise BundleCloneError("bundle_source_unreadable") from None
            relative_path = (
                PurePosixPath(entry.name)
                if relative_directory is None
                else relative_directory / entry.name
            )
            relative = relative_path.as_posix()
            attributes = getattr(status, "st_file_attributes", 0)
            if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise BundleCloneError("bundle_source_link_forbidden")
            if stat.S_ISLNK(status.st_mode):
                expected_target = _PYTHON_CONVENIENCE_LINKS.get(relative)
                if expected_target is None or os.name == "nt":
                    raise BundleCloneError("bundle_source_link_forbidden")
                try:
                    target = os.readlink(entry.path)
                except OSError:
                    raise BundleCloneError(
                        "bundle_python_convenience_link_invalid"
                    ) from None
                if target != expected_target:
                    raise BundleCloneError(
                        "bundle_python_convenience_link_invalid"
                    )
                target_path = Path(os.path.abspath(Path(entry.path).parent / target))
                try:
                    target_relative = target_path.relative_to(root).as_posix()
                    target_metadata = target_path.lstat()
                except (OSError, ValueError):
                    raise BundleCloneError(
                        "bundle_python_convenience_link_invalid"
                    ) from None
                if (
                    _is_link_or_reparse_status(target_metadata)
                    or not stat.S_ISREG(target_metadata.st_mode)
                ):
                    raise BundleCloneError(
                        "bundle_python_convenience_link_invalid"
                    )
                omitted_links[relative] = OmittedLink(
                    relative=relative,
                    target=target,
                    target_relative=target_relative,
                )
                continue
            if stat.S_ISDIR(status.st_mode):
                directories.add(relative)
                visit(Path(entry.path), relative_path)
                continue
            if not stat.S_ISREG(status.st_mode):
                raise BundleCloneError("bundle_source_special_file_forbidden")
            size, digest = _hash_file(Path(entry.path))
            if size != status.st_size:
                raise BundleCloneError("bundle_source_changed")
            files[relative] = FileSnapshot(
                relative=relative,
                size_bytes=size,
                sha256=digest,
                mode=stat.S_IMODE(status.st_mode),
                device=status.st_dev,
                inode=status.st_ino,
            )

    visit(root, None)
    if any(link.target_relative not in files for link in omitted_links.values()):
        raise BundleCloneError("bundle_python_convenience_link_invalid")
    return TreeSnapshot(
        root=root,
        files=files,
        directories=frozenset(directories),
        omitted_links=omitted_links,
    )


def _document_references(value: Any, candidates: set[str]) -> bool:
    if isinstance(value, str):
        return value in candidates
    if isinstance(value, dict):
        return any(_document_references(item, candidates) for item in value.values())
    if isinstance(value, list):
        return any(_document_references(item, candidates) for item in value)
    return False


def _any_bundle_identity_document_references(
    snapshot: TreeSnapshot,
    selected_manifest: str,
    selected_report: str,
    candidates: set[str],
) -> bool:
    if not candidates:
        return False
    selected = {selected_manifest, selected_report}
    for relative in sorted(snapshot.files):
        pure = PurePosixPath(relative)
        if len(pure.parts) != 1 or pure.suffix != ".json":
            continue
        name = pure.name
        if (
            relative not in selected
            and not name.startswith("manifest")
            and not name.startswith("qualification-report")
        ):
            continue
        if relative not in selected and _candidate_contamination(relative) is not None:
            continue
        raw = _read_limited(
            snapshot.root / name,
            MANIFEST_BYTES,
            "bundle_identity_document_invalid",
        )
        try:
            document = parse_body(raw)
        except Exception:
            raise BundleCloneError("bundle_identity_document_invalid") from None
        if _document_references(document, candidates):
            return True
    return False


def _read_limited(path: Path, maximum: int, code: str) -> bytes:
    try:
        return _read_manifest_file(path, maximum=maximum)
    except (ApplianceError, OSError):
        raise BundleCloneError(code) from None


def _artifact_relative(value: Any, code: str) -> str:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise BundleCloneError(code)
    relative = _safe_relative(value.get("path", ""), code)
    if (
        type(value.get("size_bytes")) is not int
        or value["size_bytes"] < 0
        or not isinstance(value.get("sha256"), str)
        or _SHA256.fullmatch(value["sha256"]) is None
    ):
        raise BundleCloneError(code)
    return relative


def _assert_artifact(
    snapshot: TreeSnapshot,
    value: Any,
    code: str,
) -> str:
    relative = _artifact_relative(value, code)
    observed = snapshot.files.get(relative)
    if (
        observed is None
        or observed.size_bytes != value["size_bytes"]
        or observed.sha256 != value["sha256"]
    ):
        raise BundleCloneError(code)
    return relative


def _artifact_bytes_equal(
    snapshot: TreeSnapshot,
    left_relative: str,
    right_relative: str,
) -> bool:
    if left_relative == right_relative:
        return True
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptors: list[int] = []
    try:
        for relative in (left_relative, right_relative):
            descriptor = os.open(
                snapshot.root / Path(*PurePosixPath(relative).parts), flags
            )
            descriptors.append(descriptor)
            metadata = os.fstat(descriptor)
            expected = snapshot.files[relative]
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_dev != expected.device
                or metadata.st_ino != expected.inode
                or metadata.st_size != expected.size_bytes
            ):
                return False
        while True:
            left = os.read(descriptors[0], 1024 * 1024)
            right = os.read(descriptors[1], 1024 * 1024)
            if left != right:
                return False
            if not left:
                break
        for descriptor, relative in zip(
            descriptors,
            (left_relative, right_relative),
            strict=True,
        ):
            metadata = os.fstat(descriptor)
            expected = snapshot.files[relative]
            if (
                metadata.st_dev != expected.device
                or metadata.st_ino != expected.inode
                or metadata.st_size != expected.size_bytes
            ):
                return False
        return True
    except (KeyError, OSError):
        return False
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _parse_distribution_metadata(raw: bytes) -> tuple[str, str]:
    if len(raw) > 1024 * 1024:
        raise BundleCloneError("bundle_distribution_metadata_invalid")
    try:
        message = BytesParser(policy=email_policy).parsebytes(raw)
    except Exception:
        raise BundleCloneError("bundle_distribution_metadata_invalid") from None
    names = message.get_all("Name", [])
    versions = message.get_all("Version", [])
    if len(names) != 1 or len(versions) != 1:
        raise BundleCloneError("bundle_distribution_metadata_invalid")
    name = names[0]
    version = versions[0]
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise BundleCloneError("bundle_distribution_metadata_invalid")
    return _canonical_name(name), version


def _decode_record_digest(value: str) -> str:
    algorithm, separator, encoded = value.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded:
        raise BundleCloneError("bundle_record_invalid")
    try:
        raw = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, binascii.Error):
        raise BundleCloneError("bundle_record_invalid") from None
    if len(raw) != hashlib.sha256().digest_size:
        raise BundleCloneError("bundle_record_invalid")
    return raw.hex()


def _record_target(
    root: Path,
    environment_root: Path,
    site_packages: Path,
    value: str,
) -> str:
    if not value or "\\" in value or "\x00" in value or "\n" in value or "\r" in value:
        raise BundleCloneError("bundle_record_invalid")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != value
    ):
        raise BundleCloneError("bundle_record_invalid")
    target = Path(os.path.abspath(site_packages.joinpath(*posix.parts)))
    try:
        target.relative_to(environment_root)
        relative = target.relative_to(root).as_posix()
    except ValueError:
        raise BundleCloneError("bundle_record_path_escape") from None
    return relative


def _validate_wheel(
    path: Path,
    expected_name: str,
    expected_version: str,
) -> WheelInspection:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)) or archive.testzip() is not None:
                raise BundleCloneError("bundle_wheel_invalid")
            metadata_candidates: list[zipfile.ZipInfo] = []
            python_sources: dict[str, WheelMemberIdentity] = {}
            shared_members: dict[str, WheelMemberIdentity] = {}
            for item in infos:
                pure = PurePosixPath(item.filename)
                mode = (item.external_attr >> 16) & 0xFFFF
                if (
                    not pure.parts
                    or item.filename.startswith("/")
                    or "\\" in item.filename
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or item.flag_bits & 1
                    or stat.S_ISLNK(mode)
                ):
                    raise BundleCloneError("bundle_wheel_invalid")
                if len(pure.parts) == 2 and item.filename.endswith(
                    ".dist-info/METADATA"
                ):
                    metadata_candidates.append(item)
                if item.filename.endswith(".py") or (
                    item.filename in _MLX_SHARED_WHEEL_MEMBERS
                ):
                    if item.is_dir():
                        raise BundleCloneError("bundle_wheel_invalid")
                    with archive.open(item) as stream:
                        size_bytes, digest = _hash_stream(stream)
                    if size_bytes != item.file_size:
                        raise BundleCloneError("bundle_wheel_invalid")
                    identity = WheelMemberIdentity(
                        size_bytes=size_bytes,
                        sha256=digest,
                    )
                    if item.filename.endswith(".py"):
                        python_sources[item.filename] = identity
                    if item.filename in _MLX_SHARED_WHEEL_MEMBERS:
                        shared_members[item.filename] = identity
            if len(metadata_candidates) != 1:
                raise BundleCloneError("bundle_wheel_invalid")
            metadata = metadata_candidates[0]
            if metadata.file_size > 1024 * 1024:
                raise BundleCloneError("bundle_wheel_invalid")
            name, version = _parse_distribution_metadata(archive.read(metadata))
            if name != expected_name or version != expected_version:
                raise BundleCloneError("bundle_wheel_identity_mismatch")
            return WheelInspection(
                python_sources=python_sources,
                shared_members=shared_members,
            )
    except BundleCloneError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError):
        raise BundleCloneError("bundle_wheel_invalid") from None


def _candidate_contamination(relative: str) -> str | None:
    for component in PurePosixPath(relative).parts:
        if component == ".DS_Store":
            return "finder"
        if _CONFLICT_COPY.fullmatch(component):
            return "conflict"
    return None


def _site_tail(relative: str, site_relative: str) -> PurePosixPath | None:
    try:
        return PurePosixPath(relative).relative_to(PurePosixPath(site_relative))
    except ValueError:
        return None


def _is_site_python_cache_payload(relative: str, site_relative: str) -> bool:
    tail = _site_tail(relative, site_relative)
    return tail is not None and (
        tail.suffix == ".pyc" or "__pycache__" in tail.parts
    )


def _site_python_cache_source(
    relative: str,
    site_relative: str,
) -> tuple[str, str] | None:
    tail = _site_tail(relative, site_relative)
    if tail is None or tail.parent.name != "__pycache__":
        return None
    match = _CPYTHON_311_PYC.fullmatch(tail.name)
    if match is None:
        return None
    source_tail = tail.parent.parent / f"{match.group('stem')}.py"
    if "__pycache__" in source_tail.parts:
        return None
    source_relative = (PurePosixPath(site_relative) / source_tail).as_posix()
    return source_relative, source_tail.as_posix()


def _gateway_digest(
    snapshot: TreeSnapshot,
    gateway_root: str,
    included: set[str],
) -> str:
    prefix = gateway_root.rstrip("/") + "/"
    candidates = [
        relative
        for relative in included
        if relative.startswith(prefix)
        and "/" not in relative[len(prefix) :]
        and relative.endswith(".py")
    ]
    digest = hashlib.sha256()
    for relative in sorted(candidates, key=lambda value: PurePosixPath(value).name):
        name = PurePosixPath(relative).name
        digest.update(name.encode("utf-8") + b"\0")
        path = snapshot.root / Path(*PurePosixPath(relative).parts)
        try:
            digest.update(path.read_bytes())
        except OSError:
            raise BundleCloneError("bundle_source_changed") from None
    if not candidates:
        raise BundleCloneError("bundle_gateway_identity_invalid")
    return digest.hexdigest()


def _load_report(path: Path) -> dict[str, Any]:
    raw = _read_limited(path, MANIFEST_BYTES, "bundle_qualification_report_invalid")
    try:
        value = parse_body(raw)
    except Exception:
        raise BundleCloneError("bundle_qualification_report_invalid") from None
    if not isinstance(value, dict):
        raise BundleCloneError("bundle_qualification_report_invalid")
    return value


def _validate_report_identity(
    snapshot: TreeSnapshot,
    manifest: Manifest,
    expected_packages: dict[str, dict[str, Any]],
    report: dict[str, Any],
    included: set[str],
) -> tuple[str, str]:
    required_report_fields = {
        "completed",
        "deterministic_seeds",
        "evidence_digest",
        "fixture_digest",
        "fixture_runner",
        "identity",
        "provenance",
        "report_schema_version",
        "results",
        "started",
        "status",
        "suite_version",
    }
    if (
        set(report) != required_report_fields
        or report.get("report_schema_version") != 1
        or report.get("status") != "pass"
        or report.get("suite_version") != QUALIFICATION_SUITE_VERSION
        or type(report.get("started")) is not int
        or type(report.get("completed")) is not int
        or report["started"] <= 0
        or report["completed"] < report["started"]
        or not isinstance(report.get("evidence_digest"), str)
        or _SHA256.fullmatch(report["evidence_digest"]) is None
        or not isinstance(report.get("fixture_digest"), str)
        or _SHA256.fullmatch(report["fixture_digest"]) is None
        or not isinstance(report.get("deterministic_seeds"), list)
        or not report["deterministic_seeds"]
        or any(type(value) is not int for value in report["deterministic_seeds"])
        or not isinstance(report.get("results"), dict)
    ):
        raise BundleCloneError("bundle_qualification_report_not_pass")
    identity = report.get("identity")
    if not isinstance(identity, dict) or identity.get("manifest_digest") != manifest.digest:
        raise BundleCloneError("bundle_runtime_identity_mismatch")
    if identity.get("effective_configuration") != _EXPECTED_EFFECTIVE_CONFIGURATION:
        raise BundleCloneError("bundle_runtime_identity_mismatch")
    python_identity = identity.get("python")
    if python_identity != {
        "implementation": manifest.python["implementation"],
        "version": manifest.python["version"],
    }:
        raise BundleCloneError("bundle_runtime_identity_mismatch")
    os_identity = identity.get("os")
    if (
        identity.get("architecture") not in {"arm64", "aarch64"}
        or not isinstance(os_identity, dict)
        or os_identity.get("system") != "Darwin"
        or not all(isinstance(os_identity.get(key), str) and os_identity[key] for key in ("release", "version"))
    ):
        raise BundleCloneError("bundle_runtime_identity_mismatch")
    hardware = identity.get("hardware_runtime")
    if (
        not isinstance(identity.get("cpu"), str)
        or not identity["cpu"]
        or not isinstance(hardware, dict)
        or type(hardware.get("ram_bytes")) is not int
        or hardware["ram_bytes"] <= 0
        or type(hardware.get("gpu_memory_bytes")) is not int
        or hardware["gpu_memory_bytes"] <= 0
        or any(
            not isinstance(hardware.get(key), str)
            or not hardware[key]
            or hardware[key] == "unknown"
            for key in (
                "gpu_device",
                "driver",
                "compute_runtime",
                "compute_capability",
            )
        )
    ):
        raise BundleCloneError("bundle_runtime_identity_mismatch")

    gateway_package = expected_packages.get("jarvis-inference")
    sdk_package = expected_packages.get("openai")
    gateway_runtime = identity.get("gateway_runtime")
    if (
        gateway_package is None
        or sdk_package is None
        or not isinstance(gateway_runtime, dict)
        or gateway_runtime.get("version") != gateway_package["version"]
        or gateway_runtime.get("openai_sdk_version") != sdk_package["version"]
    ):
        raise BundleCloneError("bundle_gateway_identity_invalid")
    wheel = gateway_runtime.get("wheel")
    expected_wheel = gateway_package["wheel"]
    wheel_relative = _assert_artifact(
        snapshot, wheel, "bundle_gateway_identity_invalid"
    )
    expected_wheel_relative = _assert_artifact(
        snapshot, expected_wheel, "bundle_gateway_identity_invalid"
    )
    if (
        PurePosixPath(wheel_relative).name
        != PurePosixPath(expected_wheel_relative).name
        or wheel["sha256"] != expected_wheel["sha256"]
        or wheel["size_bytes"] != expected_wheel["size_bytes"]
        or wheel_relative not in included
        or expected_wheel_relative not in included
        or not _artifact_bytes_equal(
            snapshot, wheel_relative, expected_wheel_relative
        )
    ):
        raise BundleCloneError("bundle_gateway_identity_invalid")

    site_relative = _safe_relative(
        manifest.python["site_packages"], "bundle_manifest_invalid"
    )
    gateway_root = f"{site_relative}/jarvis_inference"
    if identity.get("gateway_digest") != _gateway_digest(
        snapshot, gateway_root, included
    ):
        raise BundleCloneError("bundle_gateway_identity_would_change")

    fixture = report.get("fixture_runner")
    fixture_relative = _assert_artifact(
        snapshot, fixture, "bundle_fixture_runner_invalid"
    )
    if fixture_relative not in included:
        raise BundleCloneError("bundle_fixture_runner_invalid")
    try:
        identity_digest = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    except Exception:
        raise BundleCloneError("bundle_runtime_identity_mismatch") from None
    provenance = report.get("provenance")
    if (
        not isinstance(provenance, dict)
        or set(provenance)
        != {
            "challenge",
            "completed_unix_ns",
            "fixture_digest",
            "identity_digest",
            "monotonic_elapsed_ns",
            "started_unix_ns",
        }
        or provenance.get("identity_digest") != identity_digest
        or provenance.get("fixture_digest") != report["fixture_digest"]
        or not isinstance(provenance.get("challenge"), str)
        or _SHA256.fullmatch(provenance["challenge"]) is None
        or not isinstance(provenance.get("started_unix_ns"), str)
        or re.fullmatch(r"[1-9][0-9]{0,19}", provenance["started_unix_ns"])
        is None
        or not isinstance(provenance.get("completed_unix_ns"), str)
        or re.fullmatch(r"[1-9][0-9]{0,19}", provenance["completed_unix_ns"])
        is None
        or int(provenance["completed_unix_ns"])
        < int(provenance["started_unix_ns"])
        or type(provenance.get("monotonic_elapsed_ns")) is not int
        or provenance["monotonic_elapsed_ns"] <= 0
    ):
        raise BundleCloneError("bundle_qualification_provenance_invalid")
    if not generic_report_matches(manifest, identity, report):
        raise BundleCloneError("bundle_qualification_report_not_pass")
    return identity_digest, fixture_relative


def _inspect_bundle(
    root: Path,
    manifest_relative: str,
    *,
    allow_sanitizable_contamination: bool,
) -> BundlePlan:
    snapshot = _scan_tree(root)
    if snapshot.omitted_links and not allow_sanitizable_contamination:
        raise BundleCloneError("bundle_destination_not_sanitized")
    if manifest_relative not in snapshot.files:
        raise BundleCloneError("bundle_manifest_missing")
    manifest_path = root / Path(*PurePosixPath(manifest_relative).parts)
    try:
        manifest = load_manifest(manifest_path)
    except Exception:
        raise BundleCloneError("bundle_manifest_invalid") from None
    if manifest.platform != "macos-arm64" or manifest.root != root.resolve(strict=True):
        raise BundleCloneError("bundle_manifest_invalid")

    required: set[str] = {manifest_relative}
    for artifact in manifest.artifacts:
        value = {
            "path": artifact.path,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
        }
        required.add(
            _assert_artifact(snapshot, value, "bundle_artifact_identity_mismatch")
        )

    report_relative = manifest.qualification_report.relative_to(root).as_posix()
    if report_relative not in snapshot.files:
        raise BundleCloneError("bundle_qualification_report_invalid")
    required.add(report_relative)
    report = _load_report(manifest.qualification_report)
    if _any_bundle_identity_document_references(
        snapshot,
        manifest_relative,
        report_relative,
        set(snapshot.omitted_links),
    ):
        raise BundleCloneError("bundle_omitted_link_referenced")

    python = manifest.python
    site_relative = _safe_relative(
        python.get("site_packages", ""), "bundle_manifest_invalid"
    )
    executable_relative = _artifact_relative(
        python.get("executable"), "bundle_manifest_invalid"
    )
    environment_parts = PurePosixPath(executable_relative).parts
    if not environment_parts:
        raise BundleCloneError("bundle_manifest_invalid")
    environment_relative = environment_parts[0]
    site_packages = root / Path(*PurePosixPath(site_relative).parts)
    environment_root = root / environment_relative
    if site_relative not in snapshot.directories or environment_relative not in snapshot.directories:
        raise BundleCloneError("bundle_python_layout_invalid")

    package_values = python.get("packages")
    if not isinstance(package_values, list) or not package_values:
        raise BundleCloneError("bundle_manifest_invalid")
    expected_packages: dict[str, dict[str, Any]] = {}
    pinned_wheel_python_sources: dict[
        str, dict[str, WheelMemberIdentity]
    ] = {}
    pinned_wheel_shared_members: dict[
        str, dict[str, WheelMemberIdentity]
    ] = {}
    for value in package_values:
        if not isinstance(value, dict) or set(value) != {
            "name",
            "version",
            "wheel_filename",
            "wheel",
        }:
            raise BundleCloneError("bundle_manifest_invalid")
        name = value.get("name")
        version = value.get("version")
        wheel_filename = value.get("wheel_filename")
        if (
            not isinstance(name, str)
            or _canonical_name(name) != name
            or not isinstance(version, str)
            or not version
            or not isinstance(wheel_filename, str)
            or not wheel_filename.endswith(".whl")
            or name in expected_packages
        ):
            raise BundleCloneError("bundle_manifest_invalid")
        wheel_relative = _assert_artifact(
            snapshot, value["wheel"], "bundle_wheel_identity_mismatch"
        )
        if PurePosixPath(wheel_relative).name != wheel_filename:
            raise BundleCloneError("bundle_wheel_identity_mismatch")
        wheel_inspection = _validate_wheel(
            root / Path(*PurePosixPath(wheel_relative).parts),
            name,
            version,
        )
        pinned_wheel_python_sources[name] = wheel_inspection.python_sources
        pinned_wheel_shared_members[name] = wheel_inspection.shared_members
        expected_packages[name] = value

    exact_mlx_shared_profile = all(
        expected_packages.get(name, {}).get("version") == version
        for name, version in _MLX_SHARED_OWNER_VERSIONS.items()
    )
    if exact_mlx_shared_profile:
        for name in _MLX_SHARED_OWNER_VERSIONS:
            if (
                set(pinned_wheel_shared_members[name])
                != _MLX_SHARED_WHEEL_MEMBERS
            ):
                raise BundleCloneError("bundle_record_shared_ownership_invalid")
        for member in _MLX_SHARED_WHEEL_MEMBERS:
            identities = {
                pinned_wheel_shared_members[name][member]
                for name in _MLX_SHARED_OWNER_VERSIONS
            }
            if len(identities) != 1:
                raise BundleCloneError("bundle_record_shared_ownership_invalid")

    candidate_dist_infos: list[tuple[str, Path]] = []
    try:
        with os.scandir(site_packages) as iterator:
            site_children = sorted(iterator, key=lambda item: item.name)
    except OSError:
        raise BundleCloneError("bundle_python_layout_invalid") from None
    for child in site_children:
        try:
            metadata = child.stat(follow_symlinks=False)
        except OSError:
            raise BundleCloneError("bundle_python_layout_invalid") from None
        if _is_link_or_reparse_status(metadata):
            raise BundleCloneError("bundle_source_link_forbidden")
        if child.name.endswith(".dist-info"):
            relative = f"{site_relative}/{child.name}"
            if _candidate_contamination(relative) is not None:
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                raise BundleCloneError("bundle_distribution_metadata_invalid")
            candidate_dist_infos.append((relative, Path(child.path)))

    distributions: dict[str, tuple[str, Path, str]] = {}
    for dist_relative, dist_path in candidate_dist_infos:
        metadata_relative = f"{dist_relative}/METADATA"
        record_relative = f"{dist_relative}/RECORD"
        if metadata_relative not in snapshot.files or record_relative not in snapshot.files:
            raise BundleCloneError("bundle_distribution_metadata_invalid")
        name, version = _parse_distribution_metadata(
            _read_limited(
                root / Path(*PurePosixPath(metadata_relative).parts),
                1024 * 1024,
                "bundle_distribution_metadata_invalid",
            )
        )
        if name in distributions:
            raise BundleCloneError("bundle_distribution_duplicate")
        distributions[name] = (version, dist_path, record_relative)
    if set(distributions) != set(expected_packages):
        raise BundleCloneError("bundle_distribution_set_mismatch")

    record_files: set[str] = set()
    record_owners: dict[str, RecordOwnership] = {}
    record_owner_distributions: dict[str, set[str]] = {}
    mlx_shared_record_members: set[str] = set()
    for name, (version, dist_path, record_relative) in distributions.items():
        if version != expected_packages[name]["version"]:
            raise BundleCloneError("bundle_distribution_version_mismatch")
        raw = _read_limited(
            root / Path(*PurePosixPath(record_relative).parts),
            16 * 1024 * 1024,
            "bundle_record_invalid",
        )
        try:
            rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
        except (UnicodeError, csv.Error):
            raise BundleCloneError("bundle_record_invalid") from None
        if not rows or any(len(row) != 3 for row in rows):
            raise BundleCloneError("bundle_record_invalid")
        owned: set[str] = set()
        own_record_seen = False
        for row_path, encoded_digest, encoded_size in rows:
            relative = _record_target(
                root, environment_root, site_packages, row_path
            )
            if relative in owned:
                raise BundleCloneError("bundle_record_duplicate_path")
            observed = snapshot.files.get(relative)
            if relative in snapshot.omitted_links:
                raise BundleCloneError("bundle_omitted_link_referenced")
            if observed is None:
                raise BundleCloneError("bundle_record_file_missing")
            own_record = relative == record_relative
            row_digest: str | None = None
            row_size: int | None = None
            if not encoded_digest:
                if not own_record or encoded_size or own_record_seen:
                    raise BundleCloneError("bundle_record_invalid")
                own_record_seen = True
            else:
                row_digest = _decode_record_digest(encoded_digest)
                if row_digest != observed.sha256:
                    raise BundleCloneError("bundle_record_digest_mismatch")
                if encoded_size:
                    if re.fullmatch(r"0|[1-9][0-9]*", encoded_size) is None:
                        raise BundleCloneError("bundle_record_invalid")
                    row_size = int(encoded_size)
                    if row_size < 0 or row_size != observed.size_bytes:
                        raise BundleCloneError("bundle_record_size_mismatch")
            previous_owner = record_owners.get(relative)
            if previous_owner is not None:
                owner_pair = frozenset({previous_owner.distribution, name})
                wheel_member = row_path
                expected_relative = f"{site_relative}/{wheel_member}"
                if (
                    not exact_mlx_shared_profile
                    or owner_pair != frozenset(_MLX_SHARED_OWNER_VERSIONS)
                    or wheel_member not in _MLX_SHARED_WHEEL_MEMBERS
                    or relative != expected_relative
                    or previous_owner.row_path != wheel_member
                ):
                    raise BundleCloneError("bundle_record_duplicate_path")
                if (
                    row_digest is None
                    or row_size is None
                    or previous_owner.sha256 != row_digest
                    or previous_owner.size_bytes != row_size
                ):
                    raise BundleCloneError(
                        "bundle_record_shared_ownership_invalid"
                    )
                expected_wheel_identity = WheelMemberIdentity(
                    size_bytes=row_size,
                    sha256=row_digest,
                )
                if any(
                    pinned_wheel_shared_members[owner].get(wheel_member)
                    != expected_wheel_identity
                    for owner in _MLX_SHARED_OWNER_VERSIONS
                ):
                    raise BundleCloneError(
                        "bundle_record_shared_ownership_invalid"
                    )
                mlx_shared_record_members.add(wheel_member)
                record_owner_distributions[relative].add(name)
            else:
                record_owners[relative] = RecordOwnership(
                    distribution=name,
                    row_path=row_path,
                    sha256=row_digest,
                    size_bytes=row_size,
                )
                record_owner_distributions[relative] = {name}
            owned.add(relative)
            record_files.add(relative)
        if not own_record_seen:
            raise BundleCloneError("bundle_record_invalid")
        metadata_relative = f"{dist_path.relative_to(root).as_posix()}/METADATA"
        if metadata_relative not in owned:
            raise BundleCloneError("bundle_record_invalid")

    if exact_mlx_shared_profile and (
        mlx_shared_record_members != _MLX_SHARED_WHEEL_MEMBERS
    ):
        raise BundleCloneError("bundle_record_shared_ownership_invalid")

    exact_python_cache_profile = (
        python.get("implementation") == "CPython"
        and isinstance(python.get("version"), str)
        and re.fullmatch(r"3\.11\.[0-9]+", python["version"]) is not None
    )
    python_cache_files: set[str] = set()
    python_cache_directories = {
        relative
        for relative in snapshot.directories
        if (
            (tail := _site_tail(relative, site_relative)) is not None
            and "__pycache__" in tail.parts
        )
    }
    if python_cache_directories and not allow_sanitizable_contamination:
        raise BundleCloneError("bundle_destination_not_sanitized")
    for relative in sorted(snapshot.files):
        if not _is_site_python_cache_payload(relative, site_relative):
            continue
        contamination = _candidate_contamination(relative)
        if contamination is not None and relative not in record_files:
            continue
        if relative in record_files or not exact_python_cache_profile:
            raise BundleCloneError("bundle_generated_pycache_invalid")
        source = _site_python_cache_source(relative, site_relative)
        if source is None:
            raise BundleCloneError("bundle_generated_pycache_invalid")
        source_relative, wheel_member = source
        source_snapshot = snapshot.files.get(source_relative)
        owners = record_owner_distributions.get(source_relative)
        if source_snapshot is None or not owners:
            raise BundleCloneError("bundle_generated_pycache_invalid")
        expected_source_identity = WheelMemberIdentity(
            size_bytes=source_snapshot.size_bytes,
            sha256=source_snapshot.sha256,
        )
        if any(
            pinned_wheel_python_sources[owner].get(wheel_member)
            != expected_source_identity
            for owner in owners
        ):
            raise BundleCloneError("bundle_generated_pycache_invalid")
        python_cache_files.add(relative)
    if _any_bundle_identity_document_references(
        snapshot,
        manifest_relative,
        report_relative,
        python_cache_files | python_cache_directories,
    ):
        raise BundleCloneError("bundle_generated_pycache_referenced")

    for name, package in expected_packages.items():
        owned = {relative for relative, owners in record_owner_distributions.items() if name in owners}
        try:
            _verify_installed_wheel(manifest, package, owned)
        except ApplianceError:
            raise BundleCloneError("bundle_installed_wheel_mismatch") from None

    required.update(record_files)
    provisional_included = set(snapshot.files)
    excluded: set[str] = set()
    finder_count = 0
    conflict_count = 0
    python_cache_count = 0
    for relative in sorted(snapshot.files):
        if relative in python_cache_files:
            if not allow_sanitizable_contamination:
                raise BundleCloneError("bundle_destination_not_sanitized")
            excluded.add(relative)
            python_cache_count += 1
            continue
        kind = _candidate_contamination(relative)
        if kind is None or relative in required:
            continue
        if not allow_sanitizable_contamination:
            raise BundleCloneError("bundle_destination_not_sanitized")
        excluded.add(relative)
        if kind == "finder":
            finder_count += 1
        else:
            conflict_count += 1
    provisional_included.difference_update(excluded)

    if any(
        _is_site_python_cache_payload(relative, site_relative)
        for relative in provisional_included
    ):
        raise BundleCloneError("bundle_generated_pycache_invalid")

    site_prefix = site_relative.rstrip("/") + "/"
    unrecorded_site = {
        relative
        for relative in provisional_included
        if relative.startswith(site_prefix) and relative not in record_files
    }
    if unrecorded_site:
        raise BundleCloneError("bundle_unrecorded_site_packages_file")

    model_paths = {item["path"] for item in manifest.model["artifacts"]}
    first_parts = [PurePosixPath(value).parts[:2] for value in model_paths]
    if not first_parts or any(value != first_parts[0] for value in first_parts):
        raise BundleCloneError("bundle_model_tree_invalid")
    model_prefix = PurePosixPath(*first_parts[0]).as_posix().rstrip("/") + "/"
    copied_model_files = {
        relative
        for relative in provisional_included
        if relative.startswith(model_prefix)
    }
    if copied_model_files != model_paths:
        raise BundleCloneError("bundle_model_tree_invalid")

    identity_digest, fixture_relative = _validate_report_identity(
        snapshot,
        manifest,
        expected_packages,
        report,
        provisional_included,
    )
    required.add(fixture_relative)
    if fixture_relative not in provisional_included:
        raise BundleCloneError("bundle_fixture_runner_invalid")
    prefix = environment_relative + "/"
    installation_digest = environment_digest(
        (relative[len(prefix):], snapshot.files[relative].size_bytes,
         snapshot.files[relative].sha256, bool(snapshot.files[relative].mode & 0o111))
        for relative in provisional_included if relative.startswith(prefix)
    )
    if python.get("environment_sha256") != installation_digest:
        raise BundleCloneError("bundle_python_environment_mismatch")
    return BundlePlan(
        tree=snapshot,
        manifest_relative=manifest_relative,
        manifest_digest=manifest.digest,
        runtime_identity_digest=identity_digest,
        record_files=frozenset(record_files),
        included_files=frozenset(provisional_included),
        excluded_files=frozenset(excluded),
        excluded_directories=frozenset(python_cache_directories),
        excluded_finder_files=finder_count,
        excluded_conflict_files=conflict_count,
        excluded_python_cache_files=python_cache_count,
        omitted_python_runtime_links=len(snapshot.omitted_links),
        package_count=len(expected_packages),
    )


def _copy_regular(source: Path, destination: Path, expected: FileSnapshot) -> None:
    source_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_descriptor = -1
    destination_descriptor = -1
    try:
        source_descriptor = os.open(source, source_flags)
        source_metadata = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(source_metadata.st_mode)
            or source_metadata.st_dev != expected.device
            or source_metadata.st_ino != expected.inode
            or source_metadata.st_size != expected.size_bytes
        ):
            raise BundleCloneError("bundle_source_changed")
        mode = 0o700 if expected.mode & 0o111 else 0o600
        destination_descriptor = os.open(destination, destination_flags, mode)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
        if size != expected.size_bytes or digest.hexdigest() != expected.sha256:
            raise BundleCloneError("bundle_source_changed")
        source_after = os.fstat(source_descriptor)
        if (
            source_after.st_dev != expected.device
            or source_after.st_ino != expected.inode
            or source_after.st_size != expected.size_bytes
        ):
            raise BundleCloneError("bundle_source_changed")
        os.fchmod(destination_descriptor, mode)
        os.fsync(destination_descriptor)
    except BundleCloneError:
        raise
    except OSError:
        raise BundleCloneError("bundle_copy_failed") from None
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)


def _remove_created_tree(path: Path, root_identity: tuple[int, int]) -> bool:
    try:
        root_metadata = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if (
        _is_link_or_reparse_status(root_metadata)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or (root_metadata.st_dev, root_metadata.st_ino) != root_identity
    ):
        return False

    if not shutil.rmtree.avoids_symlink_attacks:
        return False
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != root_identity:
            return False
        # Anchor deletion to the directory we created, even if its path is replaced.
        with os.scandir(descriptor) as iterator:
            entries = list(iterator)
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.name, dir_fd=descriptor)
            else:
                os.unlink(entry.name, dir_fd=descriptor)
        current = path.lstat()
        if (current.st_dev, current.st_ino) != root_identity:
            return False
        path.rmdir()
    except OSError:
        return False
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return not os.path.lexists(path)


def _fsync_directories(root: Path, directories: Iterable[str]) -> None:
    ordered = sorted(directories, key=lambda value: len(PurePosixPath(value).parts), reverse=True)
    ordered.append("")
    for relative in ordered:
        path = root if not relative else root / Path(*PurePosixPath(relative).parts)
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError:
            raise BundleCloneError("bundle_copy_failed") from None
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def clone_qualified_bundle(
    source: str | Path,
    destination: str | Path,
    *,
    manifest_relative: str | Path = "manifest.json",
) -> dict[str, Any]:
    """Create one verified clean clone and return content-free evidence."""

    source_path = _absolute(source)
    destination_path = _absolute(destination)
    selected_manifest = _safe_relative(
        manifest_relative, "bundle_manifest_path_invalid"
    )
    _assert_existing_path_has_no_links(source_path, "bundle_source_link_forbidden")
    source_resolved = source_path.resolve(strict=True)
    if os.path.lexists(destination_path):
        raise BundleCloneError("bundle_destination_exists")
    destination_parent = destination_path.parent
    _assert_existing_path_has_no_links(
        destination_parent, "bundle_destination_parent_invalid"
    )
    parent_metadata = _lstat(
        destination_parent, "bundle_destination_parent_invalid"
    )
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise BundleCloneError("bundle_destination_parent_invalid")
    destination_resolved = destination_parent.resolve(strict=True) / destination_path.name
    if _is_within(destination_resolved, source_resolved) or _is_within(
        source_resolved, destination_resolved
    ):
        raise BundleCloneError("bundle_source_destination_overlap")

    plan = _inspect_bundle(
        source_resolved,
        selected_manifest,
        allow_sanitizable_contamination=True,
    )
    created = False
    root_identity: tuple[int, int] | None = None
    copied_bytes = 0
    created_directories: set[str] = set()
    try:
        os.mkdir(destination_path, 0o700)
        created = True
        root_metadata = destination_path.lstat()
        if _is_link_or_reparse_status(root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
            raise BundleCloneError("bundle_destination_create_failed")
        root_identity = (root_metadata.st_dev, root_metadata.st_ino)
        os.chmod(destination_path, 0o700)

        included = set(plan.included_files)
        needed_directories = {
            PurePosixPath(*PurePosixPath(relative).parts[:depth]).as_posix()
            for relative in included
            for depth in range(1, len(PurePosixPath(relative).parts))
        }
        needed_directories.update(
            relative
            for relative in plan.tree.directories
            if _candidate_contamination(relative) is None
            and relative not in plan.excluded_directories
        )
        for relative in sorted(
            needed_directories,
            key=lambda value: (len(PurePosixPath(value).parts), value),
        ):
            target = destination_path / Path(*PurePosixPath(relative).parts)
            os.mkdir(target, 0o700)
            os.chmod(target, 0o700)
            created_directories.add(relative)

        for relative in sorted(included):
            expected = plan.tree.files[relative]
            source_file = source_resolved / Path(*PurePosixPath(relative).parts)
            destination_file = destination_path / Path(*PurePosixPath(relative).parts)
            _copy_regular(source_file, destination_file, expected)
            copied_bytes += expected.size_bytes
        _fsync_directories(destination_path, created_directories)

        destination_plan = _inspect_bundle(
            destination_path,
            selected_manifest,
            allow_sanitizable_contamination=False,
        )
        if (
            destination_plan.manifest_digest != plan.manifest_digest
            or destination_plan.runtime_identity_digest
            != plan.runtime_identity_digest
            or destination_plan.included_files != plan.included_files
            or destination_plan.record_files != plan.record_files
            or destination_plan.excluded_files
            or destination_plan.excluded_directories
        ):
            raise BundleCloneError("bundle_destination_verification_failed")
        return {
            "bytes_copied": copied_bytes,
            "destination_created": True,
            "excluded": {
                "conflict_copy_files": plan.excluded_conflict_files,
                "finder_metadata_files": plan.excluded_finder_files,
                "generated_python_cache_files": plan.excluded_python_cache_files,
                "python_runtime_convenience_links": plan.omitted_python_runtime_links,
            },
            "files_copied": len(plan.included_files),
            "manifest_sha256": plan.manifest_digest,
            "network_used": False,
            "package_count": plan.package_count,
            "qualification_preserved": True,
            "ready": True,
            "record_file_count": len(plan.record_files),
            "requalification_required": False,
            "runtime_identity_sha256": plan.runtime_identity_digest,
            "source_modified": False,
            "version": REPORT_VERSION,
        }
    except BundleCloneError:
        if created and root_identity is not None:
            if not _remove_created_tree(destination_path, root_identity):
                raise BundleCloneError("bundle_rollback_incomplete") from None
        raise
    except OSError:
        if created and root_identity is not None:
            if not _remove_created_tree(destination_path, root_identity):
                raise BundleCloneError("bundle_rollback_incomplete") from None
        raise BundleCloneError("bundle_copy_failed") from None
    except KeyboardInterrupt:
        if created and root_identity is not None:
            if not _remove_created_tree(destination_path, root_identity):
                raise BundleCloneError("bundle_rollback_incomplete") from None
        raise
    except Exception:
        if created and root_identity is not None:
            if not _remove_created_tree(destination_path, root_identity):
                raise BundleCloneError("bundle_rollback_incomplete") from None
        raise BundleCloneError("bundle_clone_failed") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clone-qualified-macos-bundle")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest-relative", default="manifest.json")
    args = parser.parse_args(argv)
    try:
        report = clone_qualified_bundle(
            args.source,
            args.destination,
            manifest_relative=args.manifest_relative,
        )
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "error": {"code": "bundle_clone_interrupted", "retryable": True},
                    "ready": False,
                    "version": REPORT_VERSION,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 130
    except BundleCloneError as exc:
        print(
            json.dumps(
                {
                    "error": {"code": exc.code, "retryable": False},
                    "ready": False,
                    "version": REPORT_VERSION,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    except Exception:
        print(
            json.dumps(
                {
                    "error": {"code": "bundle_clone_failed", "retryable": False},
                    "ready": False,
                    "version": REPORT_VERSION,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
