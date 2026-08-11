#!/usr/bin/env python3
"""Create one private local Orchestrator config without exposing bearer keys."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="materialize-orchestrator-config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    database = args.database.absolute()
    if output.exists() or not output.parent.is_dir() or output.parent.is_symlink() or not database.parent.is_dir():
        print(json.dumps({"ready": False, "code": "materialization.private_config_target_invalid"}, separators=(",", ":")))
        return 2
    value = {
        "database_path": str(database),
        "voice_host": "127.0.0.1",
        "voice_port": 8765,
        "control_host": "127.0.0.1",
        "control_port": 8767,
        "control_bearer_token": secrets.token_urlsafe(48),
        "inference_base_url": "http://127.0.0.1:8766/v1",
        "inference_bearer_token": secrets.token_urlsafe(48),
        "model_alias": "jarvis-local",
        "model_timeout_seconds": 120,
        "default_speech_locale": "el-GR",
        "active_budget_seconds": 120,
        "absolute_lifetime_seconds": 900,
        "approval_ttl_seconds": 600,
        "reconnect_ttl_seconds": 30,
        "delivery_ttl_seconds": 30,
        "model_request_limit": 4,
        "output_token_limit": 2048,
        "memory_enabled": True,
        "memory_timeout_seconds": 0.25,
        "memory_recall_limit": 5,
        "memory_shadow_enabled": False,
        "memory_shadow_timeout_seconds": 30,
    }
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".orchestrator-", dir=output.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    print(json.dumps({
        "ready": True,
        "code": "materialization.private_orchestrator_config_written",
        "memory_enabled": True,
        "path_disclosure": False,
        "secrets_disclosed": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
