from __future__ import annotations

import ctypes
import hashlib
import os
import re
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .strict_json import canonical_bytes
from .strict_json import parse_body
from .errors import ApplianceError
from .manifest import Manifest, _read_manifest_file, hash_file, resolve_local


class ExclusiveFileLock(AbstractContextManager["ExclusiveFileLock"]):
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self) -> "ExclusiveFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.handle.seek(0, os.SEEK_END) == 0:
                    self.handle.write(b"\0")
                    self.handle.flush()
                    os.fsync(self.handle.fileno())
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is None:
            return
        if os.name == "nt":
            import msvcrt
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


def durable_replace(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".qualification-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            if sys_platform_is_macos():
                _full_fsync(handle.fileno())
            else:
                os.fsync(handle.fileno())
        if os.name == "nt":
            _windows_replace(temporary, path)
        else:
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def sys_platform_is_macos() -> bool:
    import sys
    return sys.platform == "darwin"


def _full_fsync(descriptor: int) -> None:
    try:
        import fcntl
        command = getattr(fcntl, "F_FULLFSYNC")
        fcntl.fcntl(descriptor, command)
    except (AttributeError, OSError):
        os.fsync(descriptor)


def _windows_replace(source: Path, destination: Path) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if destination.exists():
        replace = kernel32.ReplaceFileW
        replace.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        replace.restype = ctypes.c_int
        ok = replace(str(destination), str(source), None, 0, None, None)
    else:
        move = kernel32.MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move.restype = ctypes.c_int
        ok = move(str(source), str(destination), 0x1 | 0x8)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())


def read_report(path: Path) -> dict[str, Any] | None:
    try:
        return parse_body(_read_manifest_file(path))
    except (OSError, ApplianceError, RecursionError):
        return None


def generic_report_matches(
    manifest: Manifest, identity: dict[str, Any], report: Any
) -> bool:
    if (
        not isinstance(report, dict)
        or type(report.get("report_schema_version")) is not int
        or report["report_schema_version"] != 1
        or report.get("status") != "pass"
        or report.get("identity") != identity
    ):
        return False
    runner = report.get("fixture_runner")
    if not isinstance(runner, dict) or set(runner) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        return False
    try:
        path = resolve_local(manifest.root, runner["path"], "fixture_runner")
        size, digest = hash_file(path)
    except (ApplianceError, OSError, TypeError):
        return False
    if not (
        type(runner["size_bytes"]) is int
        and runner["size_bytes"] == size
        and runner["sha256"] == digest
    ):
        return False
    # Import at admission time: qualification's runner also imports durable IO.
    from .qualification import evaluate_evidence, _load_windows_qualification_assets

    metadata = {
        "suite_version",
        "fixture_digest",
        "deterministic_seeds",
        "identity",
        "provenance",
    }
    if (
        set(report)
        != metadata
        | {
            "report_schema_version",
            "status",
            "started",
            "completed",
            "evidence_digest",
            "fixture_runner",
            "results",
        }
        or not isinstance(report["results"], dict)
        or metadata.intersection(report["results"])
        or type(report["started"]) is not int
        or type(report["completed"]) is not int
        or report["started"] <= 0
        or report["completed"] < report["started"]
        or not isinstance(report["evidence_digest"], str)
        or re.fullmatch(r"[0-9a-f]{64}", report["evidence_digest"]) is None
    ):
        return False
    try:
        provenance = report["provenance"]
        if (
            not isinstance(provenance["challenge"], str)
            or re.fullmatch(r"[0-9a-f]{64}", provenance["challenge"]) is None
        ):
            return False
        expected = {
            "challenge": provenance["challenge"],
            "identity_digest": hashlib.sha256(canonical_bytes(identity)).hexdigest(),
            "not_before_unix_ns": report["started"] * 1_000_000_000,
            "not_after_unix_ns": (report["completed"] + 1) * 1_000_000_000 - 1,
            "max_elapsed_ns": provenance["monotonic_elapsed_ns"],
        }
        if manifest.platform == "windows-x86_64":
            assets = _load_windows_qualification_assets(manifest)
            expected["windows_assets"] = assets
            if (
                report["fixture_digest"]
                != hashlib.sha256(
                    canonical_bytes({"runner": runner, "assets": assets})
                ).hexdigest()
            ):
                return False
        evaluate_evidence(
            manifest,
            {**report["results"], **{key: report[key] for key in metadata}},
            identity,
            expected,
        )
    except (
        ApplianceError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        return False
    return True
