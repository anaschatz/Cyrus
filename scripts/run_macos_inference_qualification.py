#!/usr/bin/env python3
"""Run native macOS qualification with a private ephemeral API key."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("qualification input must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous-report", type=Path, required=True)
    parser.add_argument("--fixture-runner", type=Path, required=True)
    args = parser.parse_args()

    manifest = _object(args.manifest)
    previous = _object(args.previous_report)
    identity = previous.get("identity")
    python = manifest.get("python")
    if not isinstance(identity, dict) or not isinstance(python, dict):
        raise SystemExit("qualification identity/runtime is incomplete")
    hardware = identity.get("hardware_runtime")
    executable_artifact = python.get("executable")
    packages = python.get("packages")
    if (
        not isinstance(hardware, dict)
        or not isinstance(executable_artifact, dict)
        or not isinstance(packages, list)
    ):
        raise SystemExit("qualification hardware/runtime is incomplete")
    gateway_package = next(
        (
            item
            for item in packages
            if isinstance(item, dict) and item.get("name") == "jarvis-inference"
        ),
        None,
    )
    if not isinstance(gateway_package, dict) or not isinstance(gateway_package.get("wheel"), dict):
        raise SystemExit("gateway wheel is missing")
    executable = args.manifest.parent / str(executable_artifact["path"])
    wheel_path = str(gateway_package["wheel"]["path"])

    environment = {
        key: os.environ[key]
        for key in ("PATH", "TMPDIR", "LANG", "LC_ALL")
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "JARVIS_QUAL_API_KEY": secrets.token_urlsafe(32),
            "JARVIS_GATEWAY_WHEEL": wheel_path,
            "JARVIS_QUAL_GPU_DEVICE": str(hardware["gpu_device"]),
            "JARVIS_QUAL_GPU_MEMORY_BYTES": str(hardware["gpu_memory_bytes"]),
            "JARVIS_QUAL_GPU_DRIVER": str(hardware["driver"]),
            "JARVIS_QUAL_COMPUTE_RUNTIME": str(hardware["compute_runtime"]),
            "JARVIS_QUAL_COMPUTE_CAPABILITY": str(hardware["compute_capability"]),
        }
    )
    argv = [
        str(executable),
        "-I",
        "-m",
        "jarvis_inference",
        "qualify",
        "--manifest",
        str(args.manifest),
        "--fixture-runner",
        str(args.fixture_runner),
    ]
    os.execve(executable, argv, environment)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
