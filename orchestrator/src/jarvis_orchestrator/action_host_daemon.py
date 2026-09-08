"""Standalone macOS Action Host v2 process entrypoint."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .action_host_security import load_instance_token
from .action_host_qualification import (
    current_macos_build,
    load_native_qualification,
    readiness_digest,
    read_native_evidence_bytes,
)
from .action_host_v2_server import ActionHostResultJournalV2, ActionHostV2Server
from .macos_action_adapters import MacOSActionAdapterConfig, MacOSNativeAdapterV1
from .config import _parse_strict_json
from .private_state import ensure_private_directory
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
        value = _parse_strict_json(read_native_evidence_bytes(path))
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
        string_fields = fields - {"enabled", "qualification_file", "tool_roots", "tool_app_ids"}
        if any(not isinstance(value[name], str) for name in string_fields) or (
            value["qualification_file"] is not None and not isinstance(value["qualification_file"], str)
        ):
            raise ValueError("action host paths and digests must be strings")
        roots = value["tool_roots"]
        apps = value["tool_app_ids"]
        if (not isinstance(roots, dict) or any(not isinstance(item, str) for item in roots.values())
                or not isinstance(apps, list) or any(not isinstance(item, str) for item in apps)):
            raise ValueError("action host registry configuration is invalid")
        selected = cls(
            enabled=value["enabled"],
            socket_path=Path(value["socket_path"]),
            token_path=Path(value["token_path"]),
            journal_path=Path(value["journal_path"]),
            reference_store_path=Path(value["reference_store_path"]),
            notes_path=Path(value["notes_path"]),
            native_helper_path=Path(value["native_helper_path"]),
            native_helper_sha256=value["native_helper_sha256"],
            native_helper_code_identity_sha256=value["native_helper_code_identity_sha256"],
            qualification_file=(
                None if value["qualification_file"] is None
                else Path(value["qualification_file"])
            ),
            qualification_sha256=value["qualification_sha256"],
            tool_roots=dict(roots),
            tool_app_ids=tuple(apps),
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
    ensure_private_directory(path.parent, create=True)
    if path.exists() or path.is_symlink():
        load_instance_token(path)
        return
    # Publish complete, durable bytes without replacing another initializer's token.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".token-") as temporary:
        temporary.write(os.urandom(32))
        temporary.flush()
        os.fsync(temporary.fileno())
        try:
            os.link(temporary.name, path)
        except FileExistsError:
            load_instance_token(path)


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
    with ExitStack() as cleanup:
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
        cleanup.callback(adapter.close)
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
        journal = ActionHostResultJournalV2(config.journal_path, auth_keys)
        cleanup.callback(journal.close)
        server = ActionHostV2Server(
            socket_path=config.socket_path,
            token_path=config.token_path,
            auth_keys=auth_keys,
            registry=registry,
            journal=journal,
            adapter=adapter,
        )
        stop = threading.Event()
        for selected in (signal.SIGINT, signal.SIGTERM):
            prior = signal.signal(selected, lambda *_: stop.set())
            cleanup.callback(signal.signal, selected, prior)
        server.serve_forever(stop)


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
    parser = argparse.ArgumentParser(prog="cyrus-action-host")
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
