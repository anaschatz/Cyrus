#!/usr/bin/env python3
"""Exec the sealed, qualified macOS inference appliance without exposing secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("configuration document must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--orchestrator-config", type=Path, required=True)
    args = parser.parse_args()

    manifest = _load_object(args.manifest)
    report = _load_object(args.qualification_report)
    config = _load_object(args.orchestrator_config)
    identity = report.get("identity")
    if report.get("status") != "pass" or not isinstance(identity, dict):
        raise SystemExit("qualification report is not a pass")
    canonical_manifest = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if identity.get("manifest_digest") != hashlib.sha256(canonical_manifest).hexdigest():
        raise SystemExit("qualification report does not match manifest")
    hardware = identity.get("hardware_runtime")
    gateway = identity.get("gateway_runtime")
    python = manifest.get("python")
    if not isinstance(hardware, dict) or not isinstance(gateway, dict) or not isinstance(python, dict):
        raise SystemExit("qualification identity is incomplete")
    executable_artifact = python.get("executable")
    wheel = gateway.get("wheel")
    if not isinstance(executable_artifact, dict) or not isinstance(wheel, dict):
        raise SystemExit("qualified runtime artifacts are incomplete")
    executable = args.manifest.parent / str(executable_artifact["path"])
    bearer = config.get("inference_bearer_token")
    if not isinstance(bearer, str) or len(bearer.encode("utf-8")) < 32:
        raise SystemExit("private inference bearer token is invalid")

    environment = {
        key: os.environ[key]
        for key in ("PATH", "TMPDIR", "LANG", "LC_ALL")
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "JARVIS_INFERENCE_API_KEY": bearer,
            "JARVIS_GATEWAY_WHEEL": str(wheel["path"]),
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
        "serve",
        "--manifest",
        str(args.manifest),
    ]
    os.execve(executable, argv, environment)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
