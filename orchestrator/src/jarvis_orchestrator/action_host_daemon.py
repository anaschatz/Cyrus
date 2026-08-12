"""Standalone macOS Action Host v2 process entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .action_host_security import load_instance_token
from .action_host_qualification import (
    current_macos_build,
    load_native_qualification,
    readiness_digest,
)
from .action_host_v2_server import ActionHostResultJournalV2, ActionHostV2Server
from .macos_action_adapters import MacOSActionAdapterConfig, MacOSNativeAdapterV1
from .jcs import jcs_text
from .scheduler_crypto import platform_key_provider
from .tool_contracts import ToolRegistryV1


@dataclass(frozen=True, slots=True)
class ActionHostDaemonConfig:
    enabled: bool
    socket_path: Path
    token_path: Path
    journal_path: Path
    reference_store_path: Path
    notes_path: Path
    native_helper_path: Path
    native_helper_sha256: str
    native_helper_code_identity_sha256: str
    qualification_file: Path | None
    qualification_sha256: str
    tool_roots: Mapping[str, str]
    tool_app_ids: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "ActionHostDaemonConfig":
        raw = path.read_bytes()
        if not raw or len(raw) > 64 * 1024:
            raise ValueError("action host configuration size is invalid")
        text = raw.decode("utf-8", errors="strict")
        if text.startswith("\ufeff"):
            raise ValueError("action host configuration BOM is forbidden")

        def unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in items:
                if key in result:
                    raise ValueError("action host configuration has a duplicate key")
                result[key] = item
            return result

        value = json.loads(
            text,
            object_pairs_hook=unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("action host configuration has a non-finite number")
            ),
        )
        fields = {
            "enabled",
            "socket_path",
            "token_path",
            "journal_path",
            "reference_store_path",
            "notes_path",
            "native_helper_path",
            "native_helper_sha256",
            "native_helper_code_identity_sha256",
            "qualification_file",
            "qualification_sha256",
            "tool_roots",
            "tool_app_ids",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("action host configuration is not closed")
        roots = value["tool_roots"]
        apps = value["tool_app_ids"]
        if not isinstance(roots, dict) or not isinstance(apps, list):
            raise ValueError("action host registry configuration is invalid")
        selected = cls(
            enabled=value["enabled"],
            socket_path=Path(value["socket_path"]),
            token_path=Path(value["token_path"]),
            journal_path=Path(value["journal_path"]),
            reference_store_path=Path(value["reference_store_path"]),
            notes_path=Path(value["notes_path"]),
            native_helper_path=Path(value["native_helper_path"]),
            native_helper_sha256=str(value["native_helper_sha256"]),
            native_helper_code_identity_sha256=str(value["native_helper_code_identity_sha256"]),
            qualification_file=(
                None if value["qualification_file"] is None
                else Path(value["qualification_file"])
            ),
            qualification_sha256=str(value["qualification_sha256"]),
            tool_roots={str(key): str(item) for key, item in roots.items()},
            tool_app_ids=tuple(str(item) for item in apps),
        )
        for candidate in (
            selected.socket_path,
            selected.token_path,
            selected.journal_path,
            selected.reference_store_path,
            selected.notes_path,
            selected.native_helper_path,
            *map(Path, selected.tool_roots.values()),
        ):
            if not candidate.is_absolute():
                raise ValueError("action host paths must be absolute")
        if not isinstance(selected.enabled, bool):
            raise ValueError("action host enabled must be boolean")
        for digest in (
            selected.native_helper_sha256,
            selected.native_helper_code_identity_sha256,
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("action host helper digest is invalid")
        if selected.enabled:
            if selected.qualification_file is None or not selected.qualification_file.is_absolute():
                raise ValueError("enabled action host requires an absolute qualification file")
            if len(selected.qualification_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in selected.qualification_sha256
            ):
                raise ValueError("enabled action host qualification digest is invalid")
        elif selected.qualification_file is not None or selected.qualification_sha256:
            raise ValueError("disabled action host cannot activate qualification")
        return selected


def initialize_token(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists() or path.is_symlink():
        load_instance_token(path)
        return
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        token = os.urandom(32)
        written = os.write(descriptor, token)
        if written != len(token):
            raise OSError("short token write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def serve(config: ActionHostDaemonConfig) -> None:
    if not config.enabled:
        raise RuntimeError("tool.disabled")
    if sys.platform != "darwin":
        raise RuntimeError("Action Host v2 native runtime requires macOS")
    load_instance_token(config.token_path)
    state_root = config.journal_path.parent / "keys"
    auth_keys = platform_key_provider(store_role="action-host-auth", state_root=state_root)
    auth_keys.active_version()
    registry = ToolRegistryV1.build(
        root_ids=tuple(config.tool_roots), app_ids=config.tool_app_ids
    )
    adapter = MacOSNativeAdapterV1(
        config=MacOSActionAdapterConfig(
            helper_path=config.native_helper_path,
            helper_sha256=config.native_helper_sha256,
            helper_code_identity_sha256=config.native_helper_code_identity_sha256,
            roots=config.tool_roots,
            app_ids=config.tool_app_ids,
            state_path=config.reference_store_path,
            notes_path=config.notes_path,
        ),
        registry=registry,
        auth_keys=auth_keys,
    )
    try:
        assert config.qualification_file is not None
        qualification = load_native_qualification(
            config.qualification_file,
            expected_sha256=config.qualification_sha256,
            registry_sha256=registry.digest,
            helper_binary_sha256=config.native_helper_sha256,
            helper_code_identity_sha256=config.native_helper_code_identity_sha256,
            expected_os_build=current_macos_build(),
        )
        readiness = adapter.readiness()
        if (
            readiness_digest(readiness) != qualification.readiness_sha256
            or tuple(registry.names) != qualification.qualified_tools
            or any(not readiness.supports(tool_id) for tool_id in registry.names)
        ):
            raise RuntimeError("tool.readiness_changed")
    except BaseException:
        adapter.close()
        raise
    journal = ActionHostResultJournalV2(config.journal_path, auth_keys)
    server = ActionHostV2Server(
        socket_path=config.socket_path,
        token_path=config.token_path,
        auth_keys=auth_keys,
        registry=registry,
        journal=journal,
        adapter=adapter,
    )
    stop = threading.Event()
    prior: dict[int, Any] = {}
    for selected in (signal.SIGINT, signal.SIGTERM):
        prior[selected] = signal.signal(selected, lambda *_: stop.set())
    try:
        server.serve_forever(stop)
    finally:
        for selected, handler in prior.items():
            signal.signal(selected, handler)
        adapter.close()
        journal.close()


def request_permissions(config: ActionHostDaemonConfig) -> None:
    """Run the only setup path permitted to display native TCC prompts."""

    if sys.platform != "darwin":
        raise RuntimeError("Action Host native permission setup requires macOS")
    auth_keys = platform_key_provider(
        store_role="action-host-auth", state_root=config.journal_path.parent / "keys"
    )
    registry = ToolRegistryV1.build(
        root_ids=tuple(config.tool_roots), app_ids=config.tool_app_ids
    )
    adapter = MacOSNativeAdapterV1(
        config=MacOSActionAdapterConfig(
            helper_path=config.native_helper_path,
            helper_sha256=config.native_helper_sha256,
            helper_code_identity_sha256=config.native_helper_code_identity_sha256,
            roots=config.tool_roots,
            app_ids=config.tool_app_ids,
            state_path=config.reference_store_path,
            notes_path=config.notes_path,
            timeout_seconds=300.0,
        ),
        registry=registry,
        auth_keys=auth_keys,
    )
    try:
        print(jcs_text(adapter.request_permissions().to_dict()))
    finally:
        adapter.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis-action-host")
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--request-permissions", action="store_true")
    args = parser.parse_args(argv)
    config = ActionHostDaemonConfig.load(args.config)
    if args.initialize:
        initialize_token(config.token_path)
        platform_key_provider(
            store_role="action-host-auth", state_root=config.journal_path.parent / "keys"
        ).active_version()
        return 0
    if args.request_permissions:
        request_permissions(config)
        return 0
    serve(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
