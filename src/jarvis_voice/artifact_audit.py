"""Bounded, read-only discovery of already-local speech runtime artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .errors import BackendError
from .manifest import BASE_MODEL_MANIFEST, LARGE_V3_TURBO_MODEL_MANIFEST
from .qualification import write_report

AUDIT_SCHEMA_VERSION = 1
CT2_RUNTIME_FILE_SETS = (
    tuple(item.name for item in BASE_MODEL_MANIFEST.files),
    tuple(item.name for item in LARGE_V3_TURBO_MODEL_MANIFEST.files),
)
SKIP_DIRECTORY_NAMES = frozenset({
    ".git",
    ".Trash",
    "node_modules",
    "__pycache__",
    "Mail",
    "Messages",
    "Safari",
    "Keychains",
    "Calendars",
    "AddressBook",
    "Photos Library.photoslibrary",
})
WHISPER_CPP_PATTERN = re.compile(r"^(?:ggml-.+\.bin|(?:whisper|ggml).+\.gguf)$", re.IGNORECASE)
OPENAI_CHECKPOINT_PATTERN = re.compile(
    r"^(?:tiny|base|small|medium|turbo|large(?:-v[1-9][0-9]*)?)\.pt$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    kind: str
    path: Path
    compatibility: str
    artifacts: tuple[Path, ...]
    symlink: bool
    dataless: bool

    @property
    def candidate_id(self) -> str:
        encoded = f"{self.kind}\0{self.path}".encode("utf-8", "surrogatepass")
        return hashlib.sha256(encoded).hexdigest()[:20]


@dataclass(slots=True)
class ScanState:
    directories_scanned: int = 0
    files_seen: int = 0
    symlink_directories_skipped: int = 0
    privacy_directories_skipped: int = 0
    unreadable_entries: int = 0
    dataless_candidates: int = 0
    hash_failures: int = 0
    hashed_artifacts: int = 0
    truncated: bool = False


def _is_dataless(path: Path) -> bool:
    try:
        return bool(path.stat().st_flags & getattr(stat, "SF_DATALESS", 0))
    except (AttributeError, OSError):
        return False


def _validate_limits(max_depth: int, max_directories: int, max_candidates: int) -> None:
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or not 0 <= max_depth <= 32:
        raise BackendError("artifact.invalid_limit", "Audit depth limit is invalid", subsystem="artifact", safe_context={"field": "max_depth"})
    for name, value, maximum in (
        ("max_directories", max_directories, 1_000_000),
        ("max_candidates", max_candidates, 10_000),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
            raise BackendError("artifact.invalid_limit", "Audit limit is invalid", subsystem="artifact", safe_context={"field": name})


def _validate_roots(values: Iterable[str | Path]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        raw = Path(value)
        if not raw.is_absolute():
            raise BackendError("artifact.root_not_absolute", "Audit roots must be absolute", subsystem="artifact")
        try:
            root = raw.resolve(strict=True)
        except OSError as exc:
            raise BackendError("artifact.root_unavailable", "Audit root is unavailable", subsystem="artifact") from exc
        if root == Path(root.anchor):
            raise BackendError("artifact.root_too_broad", "Filesystem root cannot be audited", subsystem="artifact")
        if not root.is_dir():
            raise BackendError("artifact.root_not_directory", "Audit root must be a directory", subsystem="artifact")
        if root not in seen:
            roots.append(root)
            seen.add(root)
    if not roots:
        raise BackendError("artifact.root_required", "At least one audit root is required", subsystem="artifact")
    return roots


def _file_candidate(path: Path, *, symlink: bool) -> Candidate | None:
    if symlink:
        return None
    lower = path.name.lower()
    if WHISPER_CPP_PATTERN.fullmatch(path.name):
        return Candidate("whisper_cpp_model", path, "macos_whisper_cpp", (path,), False, _is_dataless(path))
    if OPENAI_CHECKPOINT_PATTERN.fullmatch(path.name):
        return Candidate("openai_whisper_checkpoint", path, "conversion_required", (path,), False, _is_dataless(path))
    if lower.startswith("silero") and lower.endswith(".onnx"):
        return Candidate("silero_vad_onnx", path, "shared_vad", (path,), False, _is_dataless(path))
    if path.name == "whisper-server" and os.access(path, os.X_OK):
        return Candidate("whisper_cpp_server", path, "macos_whisper_cpp", (path,), False, _is_dataless(path))
    return None


def discover_artifacts(
    roots: Iterable[str | Path],
    *,
    max_depth: int = 8,
    max_directories: int = 50_000,
    max_candidates: int = 100,
) -> tuple[list[Path], list[Candidate], ScanState]:
    _validate_limits(max_depth, max_directories, max_candidates)
    selected_roots = _validate_roots(roots)
    state = ScanState()
    candidates: dict[tuple[str, Path], Candidate] = {}
    visited_directories: set[tuple[int, int]] = set()
    stack: list[tuple[Path, int]] = [(root, 0) for root in reversed(selected_roots)]

    def add(candidate: Candidate) -> bool:
        key = (candidate.kind, candidate.path)
        if key in candidates:
            return True
        if len(candidates) >= max_candidates:
            state.truncated = True
            return False
        candidates[key] = candidate
        if candidate.dataless:
            state.dataless_candidates += 1
        return True

    while stack and not state.truncated:
        if state.directories_scanned >= max_directories:
            state.truncated = True
            break
        directory, depth = stack.pop()
        try:
            metadata = directory.stat()
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in visited_directories:
                continue
            visited_directories.add(identity)
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError:
            state.unreadable_entries += 1
            continue
        state.directories_scanned += 1

        file_entries: dict[str, os.DirEntry[str]] = {}
        child_directories: list[Path] = []
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in SKIP_DIRECTORY_NAMES:
                        state.privacy_directories_skipped += 1
                    elif depth < max_depth:
                        child_directories.append(Path(entry.path))
                    continue
                if entry.is_symlink() and entry.is_dir(follow_symlinks=True):
                    state.symlink_directories_skipped += 1
                    continue
                if entry.is_file(follow_symlinks=True):
                    state.files_seen += 1
                    file_entries[entry.name] = entry
                    candidate = _file_candidate(Path(entry.path), symlink=entry.is_symlink())
                    if candidate is not None and not add(candidate):
                        break
            except OSError:
                state.unreadable_entries += 1
        if state.truncated:
            break

        for runtime_files in CT2_RUNTIME_FILE_SETS:
            if all(name in file_entries for name in runtime_files):
                artifacts = tuple(
                    Path(file_entries[name].path) for name in runtime_files
                )
                candidate = Candidate(
                    "faster_whisper_ct2",
                    directory,
                    "windows_faster_whisper",
                    artifacts,
                    any(file_entries[name].is_symlink() for name in runtime_files),
                    any(_is_dataless(path) for path in artifacts),
                )
                if not add(candidate):
                    break
                break
        if state.truncated:
            break
        for child in reversed(child_directories):
            stack.append((child, depth + 1))
    ordered = sorted(candidates.values(), key=lambda item: (item.kind, item.path.as_posix()))
    return selected_roots, ordered, state


def _path_is_within(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_record(candidate: Candidate, roots: list[Path], *, hash_candidates: bool, state: ScanState) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    total_size = 0
    for path in candidate.artifacts:
        record: dict[str, object] = {"path": str(path), "symlink": path.is_symlink(), "dataless": _is_dataless(path)}
        try:
            size = path.stat().st_size
            record["size"] = size
            total_size += size
            if hash_candidates:
                if record["dataless"]:
                    record["hash_status"] = "blocked_dataless"
                    state.hash_failures += 1
                elif not _path_is_within(path, roots):
                    record["hash_status"] = "blocked_external_symlink"
                    state.hash_failures += 1
                else:
                    record["sha256"] = _sha256(path)
                    record["hash_status"] = "complete"
                    state.hashed_artifacts += 1
        except OSError:
            record["size"] = None
            record["hash_status"] = "unreadable" if hash_candidates else "not_requested"
            state.unreadable_entries += 1
            if hash_candidates:
                state.hash_failures += 1
        artifacts.append(record)
    record: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "kind": candidate.kind,
        "compatibility": candidate.compatibility,
        "path": str(candidate.path),
        "size": total_size,
        "symlink": candidate.symlink,
        "dataless": candidate.dataless,
        "artifacts": artifacts,
    }
    if candidate.kind == "faster_whisper_ct2":
        actual = {
            Path(str(item["path"])).name: (item.get("size"), item.get("sha256"))
            for item in artifacts
        }
        manifests = {
            "base": BASE_MODEL_MANIFEST,
            "large-v3-turbo": LARGE_V3_TURBO_MODEL_MANIFEST,
        }
        selected_profile, selected_manifest = next(
            (
                (profile, manifest)
                for profile, manifest in manifests.items()
                if {item.name for item in manifest.files} == set(actual)
            ),
            ("unknown", BASE_MODEL_MANIFEST),
        )
        expected = {
            item.name: (item.size, item.sha256)
            for item in selected_manifest.files
        }
        if not hash_candidates:
            match_status = "not_requested"
        elif all(item.get("hash_status") == "complete" for item in artifacts) and actual == expected:
            match_status = "match"
        else:
            match_status = "mismatch"
        record["pinned_manifest"] = {
            "status": match_status,
            "profile": selected_profile,
            "model_id": selected_manifest.model_id,
            "revision": selected_manifest.revision,
        }
    return record


def _next_action(counts: dict[str, int]) -> str:
    ct2 = counts.get("faster_whisper_ct2", 0)
    mac_model = counts.get("whisper_cpp_model", 0)
    server = counts.get("whisper_cpp_server", 0)
    vad = counts.get("silero_vad_onnx", 0)
    checkpoint = counts.get("openai_whisper_checkpoint", 0)
    if ct2 and mac_model and server and vad:
        return "artifact.cross_platform_candidates_found"
    if ct2 and not mac_model:
        return "artifact.windows_model_found_macos_model_missing"
    if mac_model and not server:
        return "artifact.macos_model_found_sidecar_missing"
    if checkpoint and not ct2 and not mac_model:
        return "artifact.conversion_would_be_required"
    if not ct2 and not mac_model and not checkpoint:
        return "artifact.whisper_not_found"
    return "artifact.candidates_require_review"


def run_artifact_audit(
    roots: Iterable[str | Path],
    *,
    output: str | Path,
    private_manifest: str | Path,
    max_depth: int = 8,
    max_directories: int = 50_000,
    max_candidates: int = 100,
    hash_candidates: bool = False,
    now: datetime | None = None,
) -> dict[str, object]:
    output_path = Path(output)
    private_path = Path(private_manifest)
    if not output_path.is_absolute() or not private_path.is_absolute():
        raise BackendError("artifact.output_not_absolute", "Audit outputs must be absolute", subsystem="artifact")
    if output_path.resolve() == private_path.resolve():
        raise BackendError("artifact.output_collision", "Audit outputs must be different files", subsystem="artifact")
    selected_roots, candidates, state = discover_artifacts(
        roots,
        max_depth=max_depth,
        max_directories=max_directories,
        max_candidates=max_candidates,
    )
    records = [
        _candidate_record(candidate, selected_roots, hash_candidates=hash_candidates, state=state)
        for candidate in candidates
    ]
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    private = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": timestamp,
        "roots": [str(root) for root in selected_roots],
        "candidates": records,
        "path_disclosure": True,
        "hash_candidates": hash_candidates,
        "network_used": False,
    }
    write_report(private_path, private)
    counts = {kind: 0 for kind in (
        "faster_whisper_ct2",
        "whisper_cpp_model",
        "openai_whisper_checkpoint",
        "silero_vad_onnx",
        "whisper_cpp_server",
    )}
    compatibility = {kind: 0 for kind in ("windows_faster_whisper", "macos_whisper_cpp", "conversion_required", "shared_vad")}
    for candidate in candidates:
        counts[candidate.kind] += 1
        compatibility[candidate.compatibility] += 1
    pinned_manifest_matches = sum(
        1
        for record in records
        if isinstance(record.get("pinned_manifest"), dict)
        and record["pinned_manifest"].get("status") == "match"  # type: ignore[union-attr]
    )
    completed = not state.truncated and state.unreadable_entries == 0 and state.hash_failures == 0
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": timestamp,
        "completed": completed,
        "code": "artifact.audit_complete" if completed else "artifact.audit_incomplete",
        "scan": {
            "roots": len(selected_roots),
            "directories": state.directories_scanned,
            "files": state.files_seen,
            "candidates": len(candidates),
            "unreadable": state.unreadable_entries,
            "dataless_candidates": state.dataless_candidates,
            "hash_failures": state.hash_failures,
            "hashed_artifacts": state.hashed_artifacts,
            "symlink_directories_skipped": state.symlink_directories_skipped,
            "privacy_directories_skipped": state.privacy_directories_skipped,
            "truncated": state.truncated,
            "max_depth": max_depth,
            "max_directories": max_directories,
            "max_candidates": max_candidates,
        },
        "counts": counts,
        "compatibility": compatibility,
        "pinned_manifest_matches": pinned_manifest_matches,
        "next_action": _next_action(counts),
        "private_manifest_written": True,
        "path_disclosure": False,
        "hash_candidates": hash_candidates,
        "model_content_read": state.hashed_artifacts > 0,
        "network_used": False,
    }
    write_report(output_path, summary)
    return summary
