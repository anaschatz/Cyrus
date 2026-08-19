"""Authenticated, content-free liveness proof for a foreground macOS beta run.

The lease is not a replacement for the inference identity endpoint.  It only
bridges the one already-qualified legacy appliance which explicitly returns
404/405 for that endpoint, and only while the foreground supervisor and the
exact process instances it spawned still own the expected loopback listeners.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import OrchestratorConfig
from .stack_config import PrivateStackConfig


LEASE_SCHEMA_VERSION = "jarvis-macos-beta-live-lease/1"
LEASE_MAX_AGE_NS = 3_000_000_000
LEASE_FUTURE_TOLERANCE_NS = 1_000_000_000
LEASE_HEARTBEAT_INTERVAL_NS = 500_000_000
_MAX_LEASE_BYTES = 32 * 1024
_COMPONENTS = ("inference", "orchestrator", "voice")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class PidProcess(Protocol):
    @property
    def pid(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    pid: int
    parent_pid: int
    uid: int
    start_token: str

    def canonical(self) -> dict[str, object]:
        return {
            "parent_pid": self.parent_pid,
            "pid": self.pid,
            "start_token": self.start_token,
            "uid": self.uid,
        }


ProcessInspector = Callable[[int], ProcessSnapshot | None]
ListenerInspector = Callable[[int], frozenset[int] | None]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _command_sha256(command: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical(list(command))).hexdigest()


def live_session_lease_path(stack: PrivateStackConfig) -> Path:
    binding = hashlib.sha256(str(stack.source_path).encode("utf-8")).hexdigest()[:24]
    return stack.source_path.parent / f".jarvis-beta-{binding}.lease.json"


def _safe_parent(path: Path) -> bool:
    try:
        metadata = path.parent.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and (os.name == "nt" or metadata.st_uid == os.getuid())
        and not metadata.st_mode & 0o022
    )


def _inspect_process(pid: int) -> ProcessSnapshot | None:
    if sys.platform != "darwin" or type(pid) is not int or pid <= 1:
        return None
    try:
        result = subprocess.run(
            (
                "/bin/ps",
                "-p",
                str(pid),
                "-o",
                "uid=",
                "-o",
                "ppid=",
                "-o",
                "lstart=",
                "-o",
                "stat=",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=1.0,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        if result.returncode != 0 or len(result.stdout) > 4096:
            return None
        fields = result.stdout.decode("utf-8", errors="strict").strip().split()
        # uid, ppid, then exactly five lstart fields, followed by process state.
        if len(fields) != 8 or fields[7].startswith("Z"):
            return None
        uid = int(fields[0])
        parent_pid = int(fields[1])
        started = " ".join(fields[2:7])
        return ProcessSnapshot(
            pid=pid,
            parent_pid=parent_pid,
            uid=uid,
            start_token=hashlib.sha256(started.encode("ascii")).hexdigest(),
        )
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return None


def _listener_pids(port: int) -> frozenset[int] | None:
    if sys.platform != "darwin" or type(port) is not int or not 1 <= port <= 65535:
        return None
    try:
        result = subprocess.run(
            (
                "/usr/sbin/lsof",
                "-nP",
                "-t",
                f"-iTCP:{port}",
                "-sTCP:LISTEN",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=1.0,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        if result.returncode != 0 or len(result.stdout) > 4096:
            return None
        values: set[int] = set()
        for line in result.stdout.decode("ascii", errors="strict").splitlines():
            if not re.fullmatch(r"[1-9][0-9]*", line):
                return None
            values.add(int(line))
        return frozenset(values) if values else None
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return None


def _auth_key(stack: PrivateStackConfig) -> bytes:
    from jarvis_voice.voice_auth import load_voice_auth_key

    config = OrchestratorConfig.load(stack.orchestrator_config_path)
    if config.voice_auth_key_file is None:
        raise RuntimeError("live_lease_auth_unavailable")
    return load_voice_auth_key(config.voice_auth_key_file).key


def _processes_match(
    core: Mapping[str, Any],
    *,
    inspect_process: ProcessInspector,
    listener_pids: ListenerInspector,
) -> bool:
    supervisor = core.get("supervisor")
    components = core.get("components")
    listeners = core.get("listeners")
    if (
        not isinstance(supervisor, Mapping)
        or not isinstance(components, Mapping)
        or set(components) != set(_COMPONENTS)
        or not isinstance(listeners, Mapping)
        or set(listeners) != {"8765", "8766", "8767"}
    ):
        return False

    def snapshot(value: Any) -> ProcessSnapshot | None:
        if not isinstance(value, Mapping) or set(value) != {
            "parent_pid",
            "pid",
            "start_token",
            "uid",
        }:
            return None
        pid = value.get("pid")
        parent_pid = value.get("parent_pid")
        uid = value.get("uid")
        start_token = value.get("start_token")
        if (
            type(pid) is not int
            or pid <= 1
            or type(parent_pid) is not int
            or parent_pid < 0
            or type(uid) is not int
            or uid < 0
            or not isinstance(start_token, str)
            or _HEX64.fullmatch(start_token) is None
        ):
            return None
        return ProcessSnapshot(pid, parent_pid, uid, start_token)

    expected_supervisor = snapshot(supervisor)
    if expected_supervisor is None or expected_supervisor.uid != os.getuid():
        return False
    if inspect_process(expected_supervisor.pid) != expected_supervisor:
        return False
    expected_components: dict[str, ProcessSnapshot] = {}
    for name in _COMPONENTS:
        value = components[name]
        if not isinstance(value, Mapping) or set(value) != {"command_sha256", "process"}:
            return False
        command_digest = value.get("command_sha256")
        expected = snapshot(value.get("process"))
        if (
            not isinstance(command_digest, str)
            or _HEX64.fullmatch(command_digest) is None
            or expected is None
            or expected.uid != os.getuid()
            or expected.parent_pid != expected_supervisor.pid
            or inspect_process(expected.pid) != expected
        ):
            return False
        expected_components[name] = expected
    if len({item.pid for item in expected_components.values()}) != len(_COMPONENTS):
        return False
    expected_listener_pids = {
        "8765": expected_components["orchestrator"].pid,
        "8766": expected_components["inference"].pid,
        "8767": expected_components["orchestrator"].pid,
    }
    if listeners != expected_listener_pids:
        return False
    return all(
        listener_pids(int(port)) == frozenset({pid})
        for port, pid in expected_listener_pids.items()
    )


def _write_private(path: Path, value: Mapping[str, Any]) -> None:
    if not _safe_parent(path):
        raise RuntimeError("live_lease_parent_invalid")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = _canonical(value) + b"\n"
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            path.chmod(0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _signed(core: Mapping[str, Any], key: bytes) -> dict[str, Any]:
    return {
        **core,
        "mac": hmac.new(key, _canonical(core), hashlib.sha256).hexdigest(),
    }


def _read_signed(path: Path, key: bytes) -> dict[str, Any] | None:
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > _MAX_LEASE_BYTES
            or (os.name != "nt" and (before.st_uid != os.getuid() or before.st_mode & 0o077))
        ):
            return None
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                return None
            raw = os.read(descriptor, _MAX_LEASE_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(raw) > _MAX_LEASE_BYTES or raw.startswith(b"\xef\xbb\xbf"):
            return None

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for name, value in items:
                if name in result:
                    raise ValueError
                result[name] = value
            return result

        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        if not isinstance(value, dict) or "mac" not in value:
            return None
        received_mac = value.pop("mac")
        if (
            not isinstance(received_mac, str)
            or _HEX64.fullmatch(received_mac) is None
            or not hmac.compare_digest(
                received_mac,
                hmac.new(key, _canonical(value), hashlib.sha256).hexdigest(),
            )
        ):
            return None
        return value
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None


@dataclass(slots=True)
class LiveSessionLease:
    path: Path
    key: bytes
    core: dict[str, Any]
    inspect_process: ProcessInspector = _inspect_process
    listener_pids: ListenerInspector = _listener_pids
    now_ns: Callable[[], int] = time.time_ns
    _last_write_ns: int = 0

    def heartbeat(self, *, force: bool = False) -> None:
        current = self.now_ns()
        if not force and current - self._last_write_ns < LEASE_HEARTBEAT_INTERVAL_NS:
            return
        if not _processes_match(
            self.core,
            inspect_process=self.inspect_process,
            listener_pids=self.listener_pids,
        ):
            raise RuntimeError("live_lease_process_mismatch")
        self.core["heartbeat_unix_ns"] = current
        _write_private(self.path, _signed(self.core, self.key))
        self._last_write_ns = current

    def close(self) -> None:
        current = _read_signed(self.path, self.key)
        if current is not None and hmac.compare_digest(
            str(current.get("session_nonce", "")),
            str(self.core.get("session_nonce", "")),
        ):
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def activate_live_session_lease(
    stack: PrivateStackConfig,
    children: Mapping[str, PidProcess],
    commands: Mapping[str, tuple[str, ...]],
    *,
    inspect_process: ProcessInspector = _inspect_process,
    listener_pids: ListenerInspector = _listener_pids,
    now_ns: Callable[[], int] = time.time_ns,
) -> LiveSessionLease:
    if set(children) != set(_COMPONENTS) or set(commands) != set(_COMPONENTS):
        raise RuntimeError("live_lease_components_invalid")
    supervisor = inspect_process(os.getpid())
    if supervisor is None or supervisor.uid != os.getuid():
        raise RuntimeError("live_lease_supervisor_invalid")
    component_values: dict[str, Any] = {}
    for name in _COMPONENTS:
        pid = children[name].pid
        current = inspect_process(pid)
        if (
            current is None
            or current.uid != os.getuid()
            or current.parent_pid != supervisor.pid
        ):
            raise RuntimeError("live_lease_child_invalid")
        component_values[name] = {
            "command_sha256": _command_sha256(commands[name]),
            "process": current.canonical(),
        }
    expected_manifest, expected_runtime = stack.expected_inference_runtime_identity()
    created = now_ns()
    core: dict[str, Any] = {
        "components": component_values,
        "created_unix_ns": created,
        "deployment_sha256": stack.sha256,
        "expected_inference_manifest_sha256": expected_manifest,
        "expected_inference_runtime_sha256": expected_runtime,
        "heartbeat_unix_ns": created,
        "listeners": {
            "8765": component_values["orchestrator"]["process"]["pid"],
            "8766": component_values["inference"]["process"]["pid"],
            "8767": component_values["orchestrator"]["process"]["pid"],
        },
        "schema_version": LEASE_SCHEMA_VERSION,
        "session_nonce": secrets.token_hex(32),
        "startup_completed": True,
        "supervisor": supervisor.canonical(),
    }
    lease = LiveSessionLease(
        live_session_lease_path(stack),
        _auth_key(stack),
        core,
        inspect_process,
        listener_pids,
        now_ns,
    )
    lease.heartbeat(force=True)
    return lease


def validate_live_session_lease(
    stack: PrivateStackConfig,
    *,
    inspect_process: ProcessInspector = _inspect_process,
    listener_pids: ListenerInspector = _listener_pids,
    now_ns: Callable[[], int] = time.time_ns,
    platform: str = sys.platform,
) -> bool:
    if platform != "darwin":
        return False
    try:
        key = _auth_key(stack)
        core = _read_signed(live_session_lease_path(stack), key)
        if core is None or set(core) != {
            "components",
            "created_unix_ns",
            "deployment_sha256",
            "expected_inference_manifest_sha256",
            "expected_inference_runtime_sha256",
            "heartbeat_unix_ns",
            "listeners",
            "schema_version",
            "session_nonce",
            "startup_completed",
            "supervisor",
        }:
            return False
        expected_manifest, expected_runtime = stack.expected_inference_runtime_identity()
        created = core.get("created_unix_ns")
        heartbeat = core.get("heartbeat_unix_ns")
        nonce = core.get("session_nonce")
        current = now_ns()
        inference_command = (
            str(stack.inference.runner_path),
            "--manifest",
            str(stack.inference.manifest_path),
            "--qualification-report",
            str(stack.inference.qualification_report_path),
            "--orchestrator-config",
            str(stack.orchestrator_config_path),
        )
        inference = core.get("components", {}).get("inference", {})
        if (
            core.get("schema_version") != LEASE_SCHEMA_VERSION
            or core.get("startup_completed") is not True
            or core.get("deployment_sha256") != stack.sha256
            or core.get("expected_inference_manifest_sha256") != expected_manifest
            or core.get("expected_inference_runtime_sha256") != expected_runtime
            or not isinstance(nonce, str)
            or _HEX64.fullmatch(nonce) is None
            or type(created) is not int
            or type(heartbeat) is not int
            or created > heartbeat
            or current - heartbeat > LEASE_MAX_AGE_NS
            or heartbeat - current > LEASE_FUTURE_TOLERANCE_NS
            or not isinstance(inference, Mapping)
            or inference.get("command_sha256") != _command_sha256(inference_command)
        ):
            return False
        return _processes_match(
            core,
            inspect_process=inspect_process,
            listener_pids=listener_pids,
        )
    except Exception:
        return False


__all__ = [
    "LEASE_MAX_AGE_NS",
    "LEASE_SCHEMA_VERSION",
    "LiveSessionLease",
    "ProcessSnapshot",
    "activate_live_session_lease",
    "live_session_lease_path",
    "validate_live_session_lease",
]
