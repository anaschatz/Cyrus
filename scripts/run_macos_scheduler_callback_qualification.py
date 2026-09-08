#!/usr/bin/env python3
"""Interactive physical qualification for the signed macOS scheduler host."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from contextlib import closing, suppress
from datetime import UTC, datetime
from pathlib import Path

from jarvis_voice.scheduler_callback import NativeSchedulerCallback
from jarvis_voice.scheduler_compatibility import load_scheduler_json
from jarvis_voice.scheduler_callback_qualification import REQUIRED_CHECKS, SCHEMA_VERSION
from jarvis_voice.scheduler_native import (
    MACOS_SCHEDULER_HOST_ADAPTER_SHA256,
    MacOSSchedulerHostAdapter,
)
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.scheduler_client import SchedulerClientController
from jarvis_voice.scheduler_journal import SchedulerClientJournal
from jarvis_voice.scheduler_runtime import SchedulerNativeRuntime
from jarvis_orchestrator.scheduler_crypto import (
    InMemorySchedulerKeyProvider,
    SchedulerContentCipher,
)


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} invalid")
    return value


def _wait_action(
    adapter: MacOSSchedulerHostAdapter,
    *,
    expected: str,
    notification_id: str,
    timeout_seconds: float,
) -> bytes:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for raw in adapter.drain_callbacks():
            callback = NativeSchedulerCallback.verify(raw, key=adapter.callback_key)
            if callback.notification_id == notification_id and callback.action == expected:
                return raw
        time.sleep(0.05)
    raise TimeoutError(f"{expected.lower()} callback was not received")


def _confirm(prompt: str) -> None:
    if input(prompt + " Type PASS: ").strip() != "PASS":
        raise RuntimeError("physical check was not confirmed")


def _verify_reconnect_replay(raw_callback: bytes, *, key: bytes, notification_id: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scheduler.sqlite3"
        cipher = SchedulerContentCipher(InMemorySchedulerKeyProvider())
        first = None
        for _ in range(2):
            with closing(SchedulerClientJournal(path, cipher=cipher)) as journal:
                runtime = SchedulerNativeRuntime(
                    controller=SchedulerClientController(
                        session_id=str(uuid.uuid4()), journal=journal,
                        approval_arbiter=GlobalApprovalController(),
                    ),
                    delivery=object(), capabilities=(), manifest_sha256="0" * 64,
                )
                if first is None:
                    journal.stage(
                        event_id=str(uuid.uuid4()), semantic_sha256="1" * 64,
                        event_type="scheduler.notification.requested",
                        payload={
                            "notification_id": notification_id, "occurrence_id": "2" * 64,
                            "job_id": str(uuid.uuid4()), "job_revision": 1, "delivery_generation": 0,
                        },
                    )
                    runtime.ingest_native_callback(raw_callback, key=key)
                actions = runtime.pump_native_callbacks()
                if len(actions) != 1:
                    raise RuntimeError("callback reconnect must emit exactly one action")
                if first is not None and (
                    actions[0].event_id != first.event_id
                    or actions[0].payload != first.payload
                    or actions[0].session_id == first.session_id
                ):
                    raise RuntimeError("callback reconnect identity or payload was not preserved")
                first = actions[0]


def write_atomic(path: Path, value: dict[str, object]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            Path(temporary).unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--callback-timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if not math.isfinite(args.callback_timeout_seconds) or args.callback_timeout_seconds <= 0:
        raise ValueError("callback timeout must be finite and positive")
    artifact, artifact_digest = load_scheduler_json(args.artifact_manifest)
    if not isinstance(artifact, dict) or set(artifact) != {
        "schema_version", "binary_sha256", "code_identity_sha256", "size_bytes"
    } or artifact["schema_version"] != "jarvis-macos-helper-artifact/1":
        raise ValueError("artifact manifest invalid")
    helper_sha = _sha(artifact["binary_sha256"], "helper digest")
    identity_sha = _sha(artifact["code_identity_sha256"], "helper identity")
    if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] <= 0:
        raise ValueError("artifact size invalid")
    if args.helper.stat().st_size != artifact["size_bytes"]:
        raise ValueError("artifact size mismatch")
    checks = {name: "fail" for name in REQUIRED_CHECKS}
    adapter = MacOSSchedulerHostAdapter(
        helper_path=os.fspath(args.helper),
        helper_sha256=helper_sha,
        helper_code_identity_sha256=identity_sha,
    )
    checks["signed_identity"] = "pass"
    try:
        test_id = hashlib.sha256(os.urandom(32)).hexdigest()
        first = adapter.post(
            notification_id=test_id,
            title="JARVIS qualification",
            body="Physical notification delivery test.",
            silent=False,
        )
        if not first.accepted:
            raise RuntimeError(first.reason_code)
        checks["permission_probe"] = "pass"
        _confirm("Confirm that the JARVIS qualification notification appeared.")
        checks["post"] = "pass"
        replaced = adapter.post(
            notification_id=test_id,
            title="JARVIS qualification updated",
            body="Replacement test.",
            silent=False,
        )
        if not replaced.accepted:
            raise RuntimeError(replaced.reason_code)
        _confirm("Confirm that only the updated qualification notification remains, with no duplicate.")
        checks["replace"] = "pass"
        if not adapter.remove(test_id).accepted:
            raise RuntimeError("remove failed")
        _confirm("Confirm that the updated qualification notification is gone from Notification Center.")
        checks["remove"] = "pass"

        silent_id = hashlib.sha256(os.urandom(32)).hexdigest()
        if not adapter.post(
            notification_id=silent_id,
            title="JARVIS silent qualification",
            body="This notification must make no sound.",
            silent=True,
        ).accepted:
            raise RuntimeError("silent post failed")
        _confirm("Confirm that the silent qualification notification appeared without sound.")
        checks["silent"] = "pass"
        adapter.remove(silent_id)

        snooze_id = hashlib.sha256(os.urandom(32)).hexdigest()
        if not adapter.post(
            notification_id=snooze_id,
            title="JARVIS callback qualification",
            body="Click Snooze 10 minutes now.",
            silent=True,
        ).accepted:
            raise RuntimeError("snooze post failed")
        snooze_raw = _wait_action(
            adapter, expected="SNOOZE", notification_id=snooze_id,
            timeout_seconds=args.callback_timeout_seconds,
        )
        checks["snooze_callback"] = "pass"
        checks["callback_authentication"] = "pass"
        _verify_reconnect_replay(
            snooze_raw, key=adapter.callback_key, notification_id=snooze_id
        )
        checks["reconnect_replay"] = "pass"

        cancel_id = hashlib.sha256(os.urandom(32)).hexdigest()
        if not adapter.post(
            notification_id=cancel_id,
            title="JARVIS callback qualification",
            body="Click Cancel now.",
            silent=True,
        ).accepted:
            raise RuntimeError("cancel post failed")
        _wait_action(
            adapter, expected="CANCEL", notification_id=cancel_id,
            timeout_seconds=args.callback_timeout_seconds,
        )
        checks["cancel_callback"] = "pass"
    finally:
        adapter.close()
    if any(value != "pass" for value in checks.values()):
        raise RuntimeError("qualification checks incomplete")
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "platform": "macos",
        "ready": True,
        "native_helper_sha256": helper_sha,
        "native_helper_code_identity_sha256": identity_sha,
        "notification_adapter_sha256": MACOS_SCHEDULER_HOST_ADAPTER_SHA256,
        "checks": checks,
        "qualified_at_utc": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "artifact_manifest_sha256": artifact_digest,
    }
    digest = write_atomic(args.output, value)
    print(json.dumps({"qualification_path": os.fspath(args.output), "qualification_sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
