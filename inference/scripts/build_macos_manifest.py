#!/usr/bin/env python3
"""Build a closed macOS inference manifest from an already sealed runtime.

This helper performs no download, installation, model load, or qualification.
It fails closed unless every installed distribution has exactly one matching
wheel and the model tree contains only regular files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import tempfile
import zipfile
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from typing import Any


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _hash(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _artifact(root: Path, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact must be a regular non-symlink file")
    relative = path.relative_to(root).as_posix()
    size, digest = _hash(path)
    return {"path": relative, "size_bytes": size, "sha256": digest}


def _wheel_identity(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file() or path.suffix != ".whl":
        raise ValueError("wheel must be a regular file")
    with zipfile.ZipFile(path) as archive:
        candidates = [
            item
            for item in archive.infolist()
            if item.filename.count("/") == 1 and item.filename.endswith(".dist-info/METADATA")
        ]
        if len(candidates) != 1 or candidates[0].file_size > 1_048_576 or candidates[0].flag_bits & 1:
            raise ValueError("wheel metadata is invalid")
        message = BytesParser(policy=email_policy).parsebytes(archive.read(candidates[0]))
    name, version = message.get("Name"), message.get("Version")
    if not isinstance(name, str) or not isinstance(version, str) or not name or not version:
        raise ValueError("wheel identity is invalid")
    return _canonical_name(name), version


def _installed(site: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = _canonical_name(distribution.metadata["Name"])
        if name in result:
            raise ValueError("duplicate installed distribution")
        if distribution.read_text("RECORD") is None:
            raise ValueError("installed distribution lacks RECORD")
        result[name] = distribution.version
    if not result:
        raise ValueError("no installed distributions")
    return result


def _python_identity(executable: Path) -> tuple[str, str]:
    code = (
        "import json,platform;"
        "print(json.dumps({'implementation':platform.python_implementation(),"
        "'version':platform.python_version()}))"
    )
    completed = subprocess.run(
        [str(executable), "-I", "-S", "-c", code],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    value = json.loads(completed.stdout) if completed.returncode == 0 else None
    if (
        not isinstance(value, dict)
        or value.get("implementation") != "CPython"
        or not isinstance(value.get("version"), str)
        or not value["version"].startswith("3.11.")
    ):
        raise ValueError("sealed Python identity is invalid")
    return value["implementation"], value["version"]


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build(root: Path, model_dir: Path, output: Path, context_window: int) -> dict[str, Any]:
    root = root.absolute()
    model_dir = model_dir.absolute()
    output = output.absolute()
    if not root.is_dir() or root.is_symlink() or output.parent != root:
        raise ValueError("runtime root or output path is invalid")
    try:
        model_relative = model_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("model directory escapes runtime root") from exc
    if len(model_relative.parts) != 2 or model_relative.parts[0] != "models" or not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError("model directory must be one dedicated models child")
    if isinstance(context_window, bool) or not 1024 <= context_window <= 262_144:
        raise ValueError("context window is invalid")

    executable = root / "python/bin/python3.11"
    site = root / "python/lib/python3.11/site-packages"
    worker = root / "runtime/mlx_worker_runtime.py"
    wheels_dir = root / "wheels"
    if executable.is_symlink() or not executable.is_file() or not site.is_dir() or not worker.is_file() or not wheels_dir.is_dir():
        raise ValueError("sealed runtime layout is incomplete")
    implementation, version = _python_identity(executable)

    wheel_map: dict[str, tuple[str, Path]] = {}
    for wheel in sorted(wheels_dir.glob("*.whl"), key=lambda item: item.name):
        name, wheel_version = _wheel_identity(wheel)
        if name in wheel_map:
            raise ValueError("duplicate wheel distribution")
        wheel_map[name] = (wheel_version, wheel)
    installed = _installed(site)
    if set(wheel_map) != set(installed):
        raise ValueError("wheelhouse and installed distribution sets differ")

    packages: list[dict[str, Any]] = []
    for name in sorted(installed):
        installed_version = installed[name]
        wheel_version, wheel = wheel_map[name]
        if installed_version != wheel_version:
            raise ValueError("installed and wheel versions differ")
        packages.append({
            "name": name,
            "version": installed_version,
            "wheel_filename": wheel.name,
            "wheel": _artifact(root, wheel),
        })

    model_files: list[Path] = []
    for path in sorted(model_dir.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError("model tree contains a symlink")
        if path.is_file():
            model_files.append(path)
        elif not path.is_dir():
            raise ValueError("model tree contains a non-regular entry")
    if not model_files:
        raise ValueError("model tree is empty")

    manifest = {
        "manifest_schema_version": 1,
        "alias": "jarvis-local",
        "platform": "macos-arm64",
        "backend": {"type": "mlx-worker", "worker": _artifact(root, worker)},
        "model": {
            "format": "mlx",
            "context_window": context_window,
            "artifacts": [_artifact(root, path) for path in model_files],
        },
        "python": {
            "implementation": implementation,
            "version": version,
            "executable": _artifact(root, executable),
            "site_packages": "python/lib/python3.11/site-packages",
            "packages": packages,
        },
        "capabilities": {
            "text": True,
            "stream": True,
            "tools": False,
            "json_object": False,
            "json_schema": False,
        },
        "qualification_report": "qualification-report.json",
    }
    _write(output, manifest)
    return {
        "ready": True,
        "code": "materialization.macos_inference_manifest_written",
        "model_files": len(model_files),
        "packages": len(packages),
        "network_used": False,
        "model_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="build-macos-inference-manifest")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-window", type=int, default=32768)
    args = parser.parse_args()
    try:
        result = build(args.root, args.model_dir, args.output, args.context_window)
    except (OSError, ValueError, zipfile.BadZipFile, csv.Error, json.JSONDecodeError):
        print(json.dumps({"ready": False, "code": "materialization.macos_inference_manifest_failed"}, separators=(",", ":")))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
