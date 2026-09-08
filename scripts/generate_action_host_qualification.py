#!/usr/bin/env python3
"""Generate a native Action Host qualification from signed, explicit evidence."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from inspect_macos_helper import inspect_helper
from jarvis_orchestrator.action_host_daemon import ActionHostDaemonConfig
from jarvis_orchestrator.action_host_qualification import (
    build_native_qualification,
    current_macos_build,
    read_native_evidence_bytes,
)
from jarvis_orchestrator.macos_action_adapters import (
    MacOSActionAdapterConfig,
    MacOSNativeAdapterV1,
)
from jarvis_orchestrator.scheduler_crypto import platform_key_provider
from jarvis_orchestrator.tool_contracts import ToolRegistryV1
from jarvis_orchestrator.jcs import jcs_bytes
from jarvis_orchestrator.config import _parse_strict_json


def utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_test_manifest(path: Path, expected_sha256: str, tool_ids: tuple[str, ...]) -> str:
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("test manifest digest is invalid")
    value = _parse_strict_json(read_native_evidence_bytes(path))
    if not isinstance(value, dict) or set(value) != {"contract", "baseline_commit", "completed_at_utc", "results"}:
        raise ValueError("test manifest is not closed")
    if value["contract"] != "jarvis.action-host-native-tests/1":
        raise ValueError("test manifest contract is invalid")
    if not isinstance(value["baseline_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["baseline_commit"]):
        raise ValueError("test manifest baseline is invalid")
    if not isinstance(value["completed_at_utc"], str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
        value["completed_at_utc"],
    ):
        raise ValueError("test manifest completion time is invalid")
    datetime.strptime(value["completed_at_utc"], "%Y-%m-%dT%H:%M:%S.%fZ")
    results = value["results"]
    if not isinstance(results, list) or len(results) != len(tool_ids):
        raise ValueError("test manifest does not cover the registry")
    observed: list[str] = []
    for result in results:
        if not isinstance(result, dict) or set(result) != {"tool_id", "status", "evidence_sha256"}:
            raise ValueError("test result is not closed")
        if result["status"] != "PASS" or not isinstance(result["evidence_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", result["evidence_sha256"]):
            raise ValueError("native tool test did not pass")
        observed.append(result["tool_id"])
    if tuple(observed) != tool_ids:
        raise ValueError("test manifest ordering or registry identity is invalid")
    digest = hashlib.sha256(jcs_bytes(value)).hexdigest()
    if digest != expected_sha256:
        raise ValueError("test manifest digest mismatch")
    return digest


def write_atomic(path: Path, value: dict[str, object]) -> None:
    payload = jcs_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            os.fchmod(target.fileno(), 0o600)
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest-sha256", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = ActionHostDaemonConfig.load(args.config)
    helper = inspect_helper(config.native_helper_path)
    if (
        helper["binary_sha256"] != config.native_helper_sha256
        or helper["code_identity_sha256"] != config.native_helper_code_identity_sha256
    ):
        raise ValueError("signed helper identity differs from host configuration")
    registry = ToolRegistryV1.build(
        root_ids=tuple(config.tool_roots), app_ids=config.tool_app_ids
    )
    test_sha = load_test_manifest(
        args.test_manifest, args.test_manifest_sha256, registry.names
    )
    keys = platform_key_provider(
        store_role="action-host-auth", state_root=config.journal_path.parent / "keys"
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
        auth_keys=keys,
    )
    try:
        readiness = adapter.readiness()
    finally:
        adapter.close()
    unsupported = [tool_id for tool_id in registry.names if not readiness.supports(tool_id)]
    if unsupported:
        raise ValueError(f"native readiness is incomplete: {unsupported[0]}")
    now = datetime.now(UTC)
    qualification = build_native_qualification(
        generated_at_utc=utc_text(now),
        expires_at_utc=args.expires_at,
        os_build=current_macos_build(),
        registry_sha256=registry.digest,
        helper_binary_sha256=config.native_helper_sha256,
        helper_code_identity_sha256=config.native_helper_code_identity_sha256,
        readiness=readiness,
        test_manifest_sha256=test_sha,
        qualified_tools=registry.names,
    )
    write_atomic(args.output, qualification.to_dict())
    print(qualification.qualification_sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
