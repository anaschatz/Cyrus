#!/usr/bin/env python3
"""Create one private local Orchestrator config without exposing bearer keys."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import tempfile
from pathlib import Path

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.windows_private_file import (
    WindowsPrivateFileError,
    harden_windows_private_path,
    validate_windows_private_path,
)

from jarvis_voice.voice_auth import VoiceAuthError, load_voice_auth_key


def _print_result(value: dict[str, object]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _remove_if_present(
    path: Path | None, identity: tuple[int, int] | None = None
) -> None:
    if path is None:
        return
    try:
        if identity is not None:
            current = path.lstat()
            if (current.st_dev, current.st_ino) != identity:
                return
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(prog="materialize-orchestrator-config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--voice-auth-key", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    database = args.database.absolute()
    voice_auth_key = args.voice_auth_key
    if (
        output.exists()
        or not voice_auth_key.is_absolute()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
        or not database.parent.is_dir()
        or not voice_auth_key.is_file()
        or voice_auth_key.is_symlink()
    ):
        _print_result(
            {
                "ready": False,
                "code": "materialization.private_config_target_invalid",
            }
        )
        return 2
    if os.name == "nt":  # pragma: no cover - native Windows CI
        try:
            validate_windows_private_path(output.parent)
        except WindowsPrivateFileError:
            _print_result(
                {
                    "ready": False,
                    "code": "materialization.private_config_target_invalid",
                }
            )
            return 2
    try:
        load_voice_auth_key(voice_auth_key)
    except VoiceAuthError:
        _print_result(
            {
                "ready": False,
                "code": "materialization.voice_auth_key_invalid",
            }
        )
        return 2
    temporary: Path | None = None
    published_identity: tuple[int, int] | None = None
    descriptor: int | None = None
    try:
        value = {
            "database_path": str(database),
            "voice_host": "127.0.0.1",
            "voice_port": 8765,
            "voice_auth_key_file": str(voice_auth_key),
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
            "memory_enabled": False,
            "memory_timeout_seconds": 0.25,
            "memory_recall_limit": 5,
            "memory_shadow_enabled": False,
            "memory_shadow_timeout_seconds": 30,
        }
        raw = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode(
                "utf-8"
            )
            + b"\n"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".orchestrator-", dir=output.parent
        )
        temporary = Path(temporary_name)
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "wb")
        descriptor = None
        with stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":  # pragma: no cover - native Windows CI
            harden_windows_private_path(temporary)
        OrchestratorConfig.load(temporary)
        staged = temporary.lstat()
        os.link(temporary, output)
        published_identity = (staged.st_dev, staged.st_ino)
        temporary.unlink()
        temporary = None
        OrchestratorConfig.load(output)
    except Exception as exc:  # noqa: BLE001 - emit one stable content-free failure
        _remove_if_present(temporary)
        if published_identity is not None:
            _remove_if_present(output, published_identity)
        code = (
            "materialization.private_config_validation_failed"
            if isinstance(exc, (ValueError, WindowsPrivateFileError))
            else "materialization.private_config_write_failed"
        )
        _print_result({"ready": False, "code": code})
        return 2
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    _print_result(
        {
            "ready": True,
            "code": "materialization.private_orchestrator_config_written",
            "memory_enabled": False,
            "path_disclosure": False,
            "secrets_disclosed": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
