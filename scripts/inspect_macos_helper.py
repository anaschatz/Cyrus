#!/usr/bin/env python3
"""Fail-closed inspection and manifest generation for a signed native helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path

from jarvis_orchestrator.macos_code_identity import code_identity_sha256


def inspect_helper(path: Path) -> dict[str, object]:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("helper path must be an absolute non-symlink path")
    selected = path.resolve(strict=True)
    metadata = selected.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or not os.access(selected, os.X_OK):
        raise ValueError("helper ownership or mode is invalid")
    file_probe = subprocess.run(
        ["/usr/bin/file", os.fspath(selected)], capture_output=True, check=False, text=True
    )
    if file_probe.returncode != 0 or "Mach-O" not in file_probe.stdout or "arm64" not in file_probe.stdout:
        raise ValueError("helper is not a native arm64 Mach-O executable")
    verified = subprocess.run(
        ["/usr/bin/codesign", "--verify", "--strict", "--verbose=2", os.fspath(selected)],
        capture_output=True,
        check=False,
    )
    details = subprocess.run(
        ["/usr/bin/codesign", "--display", "--verbose=4", os.fspath(selected)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    requirements = subprocess.run(
        ["/usr/bin/codesign", "--display", "--requirements", "-", os.fspath(selected)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    detail_text = details.stdout.decode("utf-8", errors="strict")
    if (
        verified.returncode != 0
        or details.returncode != 0
        or requirements.returncode != 0
        or "Signature=adhoc" in detail_text
        or "Authority=Developer ID Application:" not in detail_text
        or "TeamIdentifier=not set" in detail_text
    ):
        raise ValueError("helper lacks a qualified Developer ID signature")
    raw = selected.read_bytes()
    return {
        "schema_version": "jarvis-macos-helper-artifact/1",
        "binary_sha256": hashlib.sha256(raw).hexdigest(),
        "code_identity_sha256": code_identity_sha256(requirements.stdout),
        "size_bytes": len(raw),
    }


def write_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, sort_keys=True, separators=(",", ":"))
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_atomic(args.output, inspect_helper(args.helper))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
