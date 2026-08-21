from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_orchestrator.macos_beta_lease import (
    LEASE_MAX_AGE_NS,
    ProcessSnapshot,
    activate_live_session_lease,
    live_session_lease_path,
    validate_live_session_lease,
)
from jarvis_orchestrator.stack_config import (
    STACK_SCHEMA_VERSION,
    PrivateStackConfig,
    write_private_stack_config,
)
from jarvis_voice.voice_auth import create_voice_auth_key_file


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class Process:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class MacOSBetaLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root.chmod(0o700)
        self.auth = self.root / "voice-auth.json"
        create_voice_auth_key_file(self.auth, key=b"l" * 32)
        self.voice = self.private_json(
            "voice.json",
            {
                "orchestrator_url": "ws://127.0.0.1:8765",
                "voice_auth_key_file": str(self.auth),
            },
        )
        self.orchestrator = self.private_json(
            "orchestrator.json",
            {
                "database_path": str(self.root / "state.sqlite3"),
                "control_bearer_token": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
                "inference_bearer_token": "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
                "voice_auth_key_file": str(self.auth),
            },
        )
        self.runner = self.root / "runner"
        self.runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.runner.chmod(0o700)
        manifest_document = {
            "alias": "jarvis-local",
            "manifest_schema_version": 1,
        }
        self.manifest = self.private_json("manifest.json", manifest_document)
        self.qualification = self.private_json(
            "qualification.json",
            {
                "identity": {
                    "manifest_digest": hashlib.sha256(
                        canonical(manifest_document)
                    ).hexdigest()
                },
                "status": "pass",
            },
        )
        self.stack_path = self.root / "jarvis.private.json"
        write_private_stack_config(
            self.stack_path,
            {
                "schema_version": STACK_SCHEMA_VERSION,
                "voice_config_path": str(self.voice),
                "orchestrator_config_path": str(self.orchestrator),
                "inference": {
                    "runner_path": str(self.runner),
                    "manifest_path": str(self.manifest),
                    "qualification_report_path": str(self.qualification),
                },
            },
        )
        self.stack = PrivateStackConfig.load(self.stack_path)
        self.supervisor_pid = os.getpid()
        self.snapshots = {
            self.supervisor_pid: ProcessSnapshot(
                self.supervisor_pid, 99, os.getuid(), "a" * 64
            ),
            4101: ProcessSnapshot(4101, self.supervisor_pid, os.getuid(), "b" * 64),
            4102: ProcessSnapshot(4102, self.supervisor_pid, os.getuid(), "c" * 64),
            4103: ProcessSnapshot(4103, self.supervisor_pid, os.getuid(), "d" * 64),
        }
        self.listeners = {
            8765: frozenset({4102}),
            8766: frozenset({4101}),
            8767: frozenset({4102}),
        }
        self.now = [1_800_000_000_000_000_000]
        self.children = {
            "inference": Process(4101),
            "orchestrator": Process(4102),
            "voice": Process(4103),
        }
        self.commands = {
            "inference": (
                str(self.runner.resolve()),
                "--manifest",
                str(self.manifest.resolve()),
                "--qualification-report",
                str(self.qualification.resolve()),
                "--orchestrator-config",
                str(self.orchestrator.resolve()),
            ),
            "orchestrator": ("/runtime/python", "-m", "jarvis_orchestrator"),
            "voice": ("/runtime/python", "-m", "jarvis_voice"),
        }

    def private_json(self, name: str, value: object) -> Path:
        path = self.root / name
        path.write_bytes(canonical(value) + b"\n")
        path.chmod(0o600)
        return path

    def inspect(self, pid: int) -> ProcessSnapshot | None:
        return self.snapshots.get(pid)

    def listener_pids(self, port: int) -> frozenset[int] | None:
        return self.listeners.get(port)

    def activate(self):
        return activate_live_session_lease(
            self.stack,
            self.children,
            self.commands,
            inspect_process=self.inspect,
            listener_pids=self.listener_pids,
            now_ns=lambda: self.now[0],
        )

    def valid(self) -> bool:
        return validate_live_session_lease(
            self.stack,
            inspect_process=self.inspect,
            listener_pids=self.listener_pids,
            now_ns=lambda: self.now[0],
            platform="darwin",
        )

    def test_signed_fresh_lease_binds_exact_processes_and_listeners(self) -> None:
        lease = self.activate()

        self.assertTrue(self.valid())
        path = live_session_lease_path(self.stack)
        self.assertTrue(path.is_file())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        raw = path.read_text(encoding="utf-8")
        self.assertNotIn(
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8", raw
        )
        self.assertNotIn(
            "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8", raw
        )

        lease.close()
        self.assertFalse(path.exists())
        self.assertFalse(self.valid())

    def test_stale_or_future_lease_fails_closed(self) -> None:
        self.activate()
        self.now[0] += LEASE_MAX_AGE_NS + 1
        self.assertFalse(self.valid())

        self.now[0] -= LEASE_MAX_AGE_NS + 2_000_000_000
        self.assertFalse(self.valid())

    def test_pid_reuse_and_listener_replacement_fail_closed(self) -> None:
        self.activate()
        self.snapshots[4101] = ProcessSnapshot(
            4101, self.supervisor_pid, os.getuid(), "e" * 64
        )
        self.assertFalse(self.valid())

        self.snapshots[4101] = ProcessSnapshot(
            4101, self.supervisor_pid, os.getuid(), "b" * 64
        )
        self.listeners[8766] = frozenset({9999})
        self.assertFalse(self.valid())

    def test_heartbeat_tolerates_one_transient_listener_probe_failure(self) -> None:
        lease = self.activate()
        self.now[0] += 500_000_000
        calls = 0
        sleeps: list[float] = []

        def flaky_listener(port: int) -> frozenset[int] | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return None
            return self.listeners.get(port)

        lease.listener_pids = flaky_listener
        lease.sleep = sleeps.append
        lease.heartbeat()

        self.assertEqual(sleeps, [0.05])
        self.assertTrue(self.valid())

    def test_heartbeat_persistent_listener_mismatch_still_fails_closed(self) -> None:
        lease = self.activate()
        self.now[0] += 500_000_000
        sleeps: list[float] = []
        lease.listener_pids = lambda _port: frozenset({9999})
        lease.sleep = sleeps.append

        with self.assertRaisesRegex(RuntimeError, "live_lease_process_mismatch"):
            lease.heartbeat()

        self.assertEqual(sleeps, [0.05, 0.05])

    def test_tampered_or_arbitrary_lease_fails_closed(self) -> None:
        self.assertFalse(self.valid())
        self.activate()
        path = live_session_lease_path(self.stack)
        value = json.loads(path.read_text(encoding="utf-8"))
        value["deployment_sha256"] = "0" * 64
        path.write_bytes(canonical(value) + b"\n")
        path.chmod(0o600)

        self.assertFalse(self.valid())


if __name__ == "__main__":
    unittest.main()
