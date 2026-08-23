"""Foreground-only supervisor for one-host macOS beta sessions.

The supervisor is deliberately not a service manager.  It never installs an
autostart item and it never detaches.  The calling terminal owns the complete
Inference -> Orchestrator -> Voice process tree for the lifetime of the beta
session.
"""

from __future__ import annotations

import asyncio
import ctypes
import errno
import functools
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from .config import OrchestratorConfig
from .inference import (
    InferenceControlClient,
    InferenceIdentityEndpointUnavailable,
)
from .service import (
    BETA_SESSION_SCHEMA_VERSION,
    ENGLISH_ONLY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    valid_voice_connection_tracker_id,
)
from .stack_config import PrivateStackConfig


SESSION_REPORT_VERSION = "jarvis-macos-beta-session/1"
RECOVERY_REPORT_VERSION = "jarvis-macos-beta-recovery/1"
SESSION_INTERLOCK_VERSION = "jarvis-macos-beta-interlock/1"
SESSION_INTERLOCK_FILENAME = ".jarvis-macos-beta-unclean.json"
REQUIRED_PORTS = (8765, 8766, 8767)
_COMPONENTS = ("inference", "orchestrator", "voice")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_MAX_INTERLOCK_BYTES = 16 * 1024
_STOP_REASONS = {"operator_requested", "signal", "keyboard_interrupt"}
_ERROR_CODES = {
    "beta_preflight_failed",
    "command_prefix_invalid",
    "executable_invalid",
    "fixed_loopback_endpoints_required",
    "inference_warmup_failed",
    "live_lease_activation_failed",
    "live_lease_runtime_failed",
    "macos_required",
    "private_stack_invalid",
    "required_port_occupied",
    "session_directory_create_failed",
    "session_failed",
    "session_interlock_failed",
    "session_parent_invalid",
    "session_recovery_unsafe",
    "session_relaunch_blocked",
    "session_shutdown_incomplete",
    "session_timing_invalid",
    "signal_handlers_unavailable",
    "stop_reason_invalid",
    *(
        f"{component}_{suffix}"
        for component in _COMPONENTS
        for suffix in (
            "readiness_timeout",
            "runtime_failed",
            "spawn_failed",
            "startup_failed",
        )
    ),
}


class MacOSBetaSessionError(RuntimeError):
    """Content-free foreground-session failure."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_CODES:
            code = "session_failed"
        super().__init__(code)
        self.code = code


class ChildProcess(Protocol):
    """Small process surface used by the deterministic supervisor core."""

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout_seconds: float) -> bool: ...


class SessionLease(Protocol):
    def heartbeat(self, *, force: bool = False) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CommandPrefix:
    """An explicit executable and fixed arguments preceding component args.

    Installed entry points use no ``arguments``.  A source run can instead use
    an absolute interpreter with ``("-m", "jarvis_voice")`` or
    ``("-m", "jarvis_orchestrator")``.
    """

    executable: Path
    arguments: tuple[str, ...] = ()

    def command(self, *arguments: str) -> tuple[str, ...]:
        return (str(self.executable), *self.arguments, *arguments)


@dataclass(frozen=True, slots=True)
class ProbeContext:
    component: str
    stack: PrivateStackConfig
    orchestrator: OrchestratorConfig
    process: ChildProcess
    elapsed_seconds: float


ReadinessProbe = Callable[[ProbeContext], bool]
SpawnProcess = Callable[[str, tuple[str, ...], BinaryIO, Path], ChildProcess]
PortAvailable = Callable[[int], bool]
ListenerPids = Callable[[int], frozenset[int] | None]
PreflightReady = Callable[[PrivateStackConfig], bool]
InferenceWarmup = Callable[[PrivateStackConfig, OrchestratorConfig], bool]
ActivateLiveLease = Callable[
    [PrivateStackConfig, Mapping[str, ChildProcess], Mapping[str, tuple[str, ...]]],
    SessionLease,
]


_PROC_PGRP_ONLY = 2
_PROC_PIDTBSDINFO = 3
_SZOMB = 5
_MAX_PROCESS_LIST_BYTES = 1024 * 1024


class _ProcBsdInfo(ctypes.Structure):
    """The stable public ``proc_bsdinfo`` layout from macOS ``libproc``."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


class _SigValue(ctypes.Union):
    _fields_ = [
        ("sival_int", ctypes.c_int),
        ("sival_ptr", ctypes.c_void_p),
    ]


class _SigInfo(ctypes.Structure):
    """The stable public Darwin ``siginfo_t`` layout used by ``waitid``."""

    _fields_ = [
        ("si_signo", ctypes.c_int),
        ("si_errno", ctypes.c_int),
        ("si_code", ctypes.c_int),
        ("si_pid", ctypes.c_int),
        ("si_uid", ctypes.c_uint32),
        ("si_status", ctypes.c_int),
        ("si_addr", ctypes.c_void_p),
        ("si_value", _SigValue),
        ("si_band", ctypes.c_long),
        ("reserved", ctypes.c_ulong * 7),
    ]


@dataclass(frozen=True, slots=True)
class _ProcessGroupMember:
    pid: int
    process_group_id: int
    uid: int
    status: int
    start_token: str = ""


@dataclass(frozen=True, slots=True)
class _ProcessIdentity:
    pid: int
    process_group_id: int
    uid: int
    status: int
    start_token: str


ProcessGroupInspector = Callable[
    [int], tuple[_ProcessGroupMember, ...] | None
]
ProcessIdentityInspector = Callable[[int], _ProcessIdentity | None]
GroupSignal = Callable[[int, int], None]
PeekChildExit = Callable[[int], int | None]


def _process_start_token(info: _ProcBsdInfo) -> str:
    raw = f"{info.pbi_start_tvsec}:{info.pbi_start_tvusec}".encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _inspect_process_identity(pid: int) -> _ProcessIdentity | None:
    """Return a PID identity tied to its kernel start time, never PID alone."""

    if sys.platform != "darwin" or type(pid) is not int or pid <= 1:
        return None
    try:
        library = _libproc()
        info = _ProcBsdInfo()
        size = library.proc_pidinfo(
            pid,
            _PROC_PIDTBSDINFO,
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if size == 0:
            return None
        if size != ctypes.sizeof(info) or info.pbi_pid != pid:
            return None
        return _ProcessIdentity(
            pid=pid,
            process_group_id=info.pbi_pgid,
            uid=info.pbi_uid,
            status=info.pbi_status,
            start_token=_process_start_token(info),
        )
    except (AttributeError, OSError, ValueError):
        return None


@functools.cache
def _libproc() -> Any:
    if sys.platform != "darwin":
        raise OSError("libproc_unavailable")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_listpids.argtypes = (
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    library.proc_listpids.restype = ctypes.c_int
    library.proc_pidinfo.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    library.proc_pidinfo.restype = ctypes.c_int
    return library


@functools.cache
def _libc() -> Any:
    if sys.platform != "darwin":
        raise OSError("libc_waitid_unavailable")
    library = ctypes.CDLL(None, use_errno=True)
    library.waitid.argtypes = (
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.POINTER(_SigInfo),
        ctypes.c_int,
    )
    library.waitid.restype = ctypes.c_int
    return library


def _inspect_process_group(
    process_group_id: int,
) -> tuple[_ProcessGroupMember, ...] | None:
    """Return exact macOS group membership, or ``None`` on uncertainty."""

    if (
        sys.platform != "darwin"
        or type(process_group_id) is not int
        or process_group_id <= 1
    ):
        return None
    try:
        library = _libproc()
        required = library.proc_listpids(
            _PROC_PGRP_ONLY,
            process_group_id,
            None,
            0,
        )
        if required < 0 or required > _MAX_PROCESS_LIST_BYTES:
            return None
        buffer_size = max(4096, required + 4096)
        if buffer_size > _MAX_PROCESS_LIST_BYTES:
            return None
        pid_count = buffer_size // ctypes.sizeof(ctypes.c_int)
        pid_buffer = (ctypes.c_int * pid_count)()
        written = library.proc_listpids(
            _PROC_PGRP_ONLY,
            process_group_id,
            pid_buffer,
            ctypes.sizeof(pid_buffer),
        )
        if (
            written < 0
            or written >= ctypes.sizeof(pid_buffer)
            or written % ctypes.sizeof(ctypes.c_int)
        ):
            return None
        members: list[_ProcessGroupMember] = []
        seen: set[int] = set()
        for pid in pid_buffer[: written // ctypes.sizeof(ctypes.c_int)]:
            if pid <= 1 or pid in seen:
                continue
            seen.add(pid)
            info = _ProcBsdInfo()
            size = library.proc_pidinfo(
                pid,
                _PROC_PIDTBSDINFO,
                0,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            # A member can disappear between the two libproc calls. It is
            # safe to omit it; a later pass observes every survivor.
            if size == 0:
                try:
                    if os.getpgid(pid) == process_group_id:
                        return None
                except ProcessLookupError:
                    pass
                except OSError:
                    return None
                continue
            if (
                size != ctypes.sizeof(info)
                or info.pbi_pid != pid
                or info.pbi_pgid != process_group_id
            ):
                return None
            members.append(
                _ProcessGroupMember(
                    pid=pid,
                    process_group_id=info.pbi_pgid,
                    uid=info.pbi_uid,
                    status=info.pbi_status,
                    start_token=_process_start_token(info),
                )
            )
        return tuple(sorted(members, key=lambda item: item.pid))
    except (AttributeError, OSError, ValueError):
        return None


def _peek_child_exit(pid: int) -> int | None:
    """Observe a direct child's exit without reaping its PID/PGID anchor."""

    info = _SigInfo()
    ctypes.set_errno(0)
    result = _libc().waitid(
        os.P_PID,
        pid,
        ctypes.byref(info),
        os.WEXITED | os.WNOHANG | os.WNOWAIT,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.ECHILD:
            raise ChildProcessError(error_number, "child_identity_lost")
        raise OSError(error_number, "child_exit_observation_failed")
    if info.si_pid == 0:
        return None
    if info.si_pid != pid:
        raise OSError(errno.ECHILD, "child_identity_mismatch")
    if info.si_code == os.CLD_EXITED:
        return int(info.si_status)
    if info.si_code in {os.CLD_KILLED, os.CLD_DUMPED}:
        return -int(info.si_status)
    return None


class _PopenProcess:
    """A process-group-owning subprocess handle.

    Every component starts a fresh process group.  TERM/KILL therefore also
    reaches native helper children instead of leaving them behind when the
    foreground owner exits.
    """

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        group_inspector: ProcessGroupInspector = _inspect_process_group,
        group_signal: GroupSignal = os.killpg,
        peek_child_exit: PeekChildExit = _peek_child_exit,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._process = process
        self._process_group_id = process.pid
        self._group_inspector = group_inspector
        self._group_signal = group_signal
        self._peek_child_exit = peek_child_exit
        self._monotonic = monotonic
        self._sleep = sleep
        self._observed_returncode: int | None = None
        self._identity_lost = process.returncode is not None
        self._group_confirmed_drained = False

    @property
    def pid(self) -> int:
        return self._process.pid

    def poll(self) -> int | None:
        if self._process.returncode is not None:
            self._identity_lost = True
            return self._process.returncode
        if self._observed_returncode is not None:
            return self._observed_returncode
        try:
            observed = self._peek_child_exit(self._process.pid)
        except ChildProcessError:
            # Something outside this handle reaped the leader. We may report
            # the exit, but must never signal its now-reusable numeric PGID.
            self._identity_lost = True
            return self._process.poll()
        except OSError:
            # Do not reap or guess. Shutdown remains fail-closed and bounded.
            return None
        if observed is not None:
            self._observed_returncode = observed
        return observed

    def _refresh_identity_anchor(self) -> bool:
        if not self._identity_is_anchored():
            return False
        try:
            observed = self._peek_child_exit(self._process.pid)
        except ChildProcessError:
            self._identity_lost = True
            self._process.poll()
            return False
        except OSError:
            return False
        if observed is not None:
            self._observed_returncode = observed
        return True

    def _identity_is_anchored(self) -> bool:
        return not self._identity_lost and self._process.returncode is None

    def _group_drained(self) -> bool | None:
        if self._group_confirmed_drained:
            return True
        if not self._refresh_identity_anchor():
            return None
        try:
            members = self._group_inspector(self._process_group_id)
        except Exception:
            return None
        if members is None:
            return None
        if not members:
            # An empty group is conclusive only after waitid observed the
            # leader exit while deliberately retaining its unreaped PID.
            return self._observed_returncode is not None
        if any(
            item.process_group_id != self._process_group_id
            or item.pid <= 1
            for item in members
        ):
            return None
        return all(item.status == _SZOMB for item in members)

    def _signal_group(self, value: int) -> None:
        # Popen.returncode remains unset even after waitid observes an exit.
        # The unreaped leader therefore pins the numeric PID/PGID and makes a
        # group signal safe even when helpers outlive that leader. If an
        # external actor already reaped it, fail closed rather than risk
        # signalling a recycled, unrelated process group.
        if (
            self._group_confirmed_drained
            or not self._refresh_identity_anchor()
        ):
            return
        try:
            self._group_signal(self._process_group_id, value)
        except ProcessLookupError:
            return

    def terminate(self) -> None:
        self._signal_group(signal.SIGTERM)

    def kill(self) -> None:
        self._signal_group(signal.SIGKILL)

    def wait(self, timeout_seconds: float) -> bool:
        deadline = self._monotonic() + max(0.0, timeout_seconds)
        while True:
            self.poll()
            drained = self._group_drained()
            if drained is True:
                try:
                    self._process.wait(timeout=0.0)
                except (ChildProcessError, subprocess.TimeoutExpired):
                    return False
                self._group_confirmed_drained = True
                return True
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return False
            self._sleep(min(0.02, remaining))


def _spawn_process(
    _component: str,
    command: tuple[str, ...],
    log: BinaryIO,
    session_directory: Path,
) -> ChildProcess:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    environment.pop("CYRUS_VOICE_TRACE_TIMINGS", None)
    if _component == "voice":
        environment["CYRUS_VOICE_TRACE_TIMINGS"] = "1"
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        environment.pop(name, None)
    process = subprocess.Popen(
        command,
        cwd=session_directory,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        start_new_session=True,
    )
    return _PopenProcess(process)


def _port_available(port: int) -> bool:
    candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        candidate.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        candidate.close()


def _listener_pids(port: int) -> frozenset[int] | None:
    """Return exact listening PIDs, empty when absent, or ``None`` if unknown.

    A bind probe is intentionally unsuitable after shutdown: macOS can retain
    a fully closed loopback connection in TIME_WAIT and reject an immediate
    bind even though no process owns a listening socket.
    """

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
        if len(result.stdout) > 4096:
            return None
        if result.returncode == 1 and not result.stdout:
            return frozenset()
        if result.returncode != 0:
            return None
        values: set[int] = set()
        for line in result.stdout.decode("ascii", errors="strict").splitlines():
            if not re.fullmatch(r"[1-9][0-9]*", line):
                return None
            values.add(int(line))
        return frozenset(values)
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return None


def _beta_preflight_ready(stack: PrivateStackConfig) -> bool:
    try:
        from .beta_preflight import evaluate_macos_beta_preflight

        report = evaluate_macos_beta_preflight(stack)
        return report.get("technical_beta_preflight_ready") is True
    except Exception:
        return False


def _activate_live_lease(
    stack: PrivateStackConfig,
    children: Mapping[str, ChildProcess],
    commands: Mapping[str, tuple[str, ...]],
) -> SessionLease:
    from .macos_beta_lease import activate_live_session_lease

    return activate_live_session_lease(stack, children, commands)


def _inference_ready(context: ProbeContext) -> bool:
    try:
        client = InferenceControlClient(
            context.orchestrator.inference_base_url,
            context.orchestrator.inference_bearer_token,
            timeout_seconds=2.0,
        )
        capabilities = client.capabilities()
        if not (capabilities.text and capabilities.stream):
            return False
        try:
            identity = client.runtime_identity()
        except InferenceIdentityEndpointUnavailable:
            # A pre-identity-endpoint appliance is permitted only inside this
            # foreground supervisor: it exclusively bound a previously-free
            # port by launching the exact digest-checked runner from the
            # validated stack and continuously owns/monitors that child.
            return context.process.poll() is None
        expected_manifest, expected_runtime = (
            context.stack.expected_inference_runtime_identity()
        )
        return (
            identity.manifest_digest == expected_manifest
            and identity.runtime_identity_sha256 == expected_runtime
        )
    except Exception:
        return False


def _inference_warmup(
    _stack: PrivateStackConfig,
    config: OrchestratorConfig,
) -> bool:
    """Pay the one-time local generation cost before Voice becomes ready."""

    try:
        parsed = urllib.parse.urlparse(config.inference_base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or parsed.path.rstrip("/") != "/v1"
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            return False
        body = json.dumps(
            {
                "model": config.model_alias,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            ENGLISH_ONLY_SYSTEM_PROMPT
                            if config.response_language == "en"
                            else SYSTEM_PROMPT
                        ),
                    },
                    {"role": "user", "content": "Reply with only this word: ready"}
                ],
                "max_completion_tokens": 16,
                "temperature": 0,
                "top_p": 1,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{parsed.port}/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {config.inference_bearer_token}",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "Host": f"127.0.0.1:{parsed.port}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        parts: list[str] = []
        finish_reason: str | None = None
        done = False
        total_bytes = 0
        with urllib.request.urlopen(request, timeout=10.0) as response:
            if (
                response.status != 200
                or response.headers.get("Content-Type")
                != "text/event-stream; charset=utf-8"
            ):
                return False
            while total_bytes <= 256 * 1024:
                line = response.readline(64 * 1024 + 1)
                if not line:
                    break
                total_bytes += len(line)
                if len(line) > 64 * 1024 or total_bytes > 256 * 1024:
                    return False
                if not line.startswith(b"data: "):
                    if line.strip():
                        return False
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                event = json.loads(payload.decode("utf-8", errors="strict"))
                if not isinstance(event, dict):
                    return False
                choices = event.get("choices")
                if choices == []:
                    continue
                if (
                    not isinstance(choices, list)
                    or len(choices) != 1
                    or not isinstance(choices[0], dict)
                ):
                    return False
                choice = choices[0]
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    return False
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        return False
                    parts.append(content)
                    if len("".join(parts).encode("utf-8")) > 64:
                        return False
                candidate = choice.get("finish_reason")
                if candidate is not None:
                    if not isinstance(candidate, str):
                        return False
                    finish_reason = candidate
        return (
            done
            and finish_reason == "stop"
            and "".join(parts).strip().casefold() == "ready"
        )
    except (
        OSError,
        TimeoutError,
        UnicodeError,
        ValueError,
        urllib.error.URLError,
    ):
        return False


def _orchestrator_http_ready(config: OrchestratorConfig) -> bool:
    request = urllib.request.Request(
        f"http://127.0.0.1:{config.control_port}/readyz",
        headers={"Host": f"127.0.0.1:{config.control_port}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            body = response.read(4 * 1024 + 1)
        return (
            response.status == 200
            and len(body) <= 4 * 1024
            and json.loads(body.decode("utf-8", errors="strict"))
            == {"ready": True}
        )
    except (
        OSError,
        TimeoutError,
        UnicodeError,
        ValueError,
        urllib.error.URLError,
    ):
        return False


async def _authenticated_gateway_ready(config: OrchestratorConfig) -> bool:
    try:
        from jarvis_voice.voice_auth import load_voice_auth_key
        from jarvis_voice.voice_auth_transport import (
            AUTH_PROBE_FRAME,
            AUTH_PROBE_PATH,
            AUTH_PROBE_SUCCESS_FRAME,
            authenticate_async_client,
        )
        from websockets.asyncio.client import connect

        if config.voice_auth_key_file is None:
            return False
        auth_key = load_voice_auth_key(config.voice_auth_key_file)

        async with connect(
            f"ws://127.0.0.1:{config.voice_port}{AUTH_PROBE_PATH}",
            origin=None,
            compression=None,
            proxy=None,
            open_timeout=2.0,
            close_timeout=1.0,
            max_size=4 * 1024,
            max_queue=4,
        ) as websocket:
            await authenticate_async_client(
                websocket,
                auth_key,
                AUTH_PROBE_FRAME,
            )
            reply = await asyncio.wait_for(websocket.recv(), timeout=2.0)
        return reply == AUTH_PROBE_SUCCESS_FRAME
    except Exception:
        return False


def _orchestrator_ready(context: ProbeContext) -> bool:
    if not _orchestrator_http_ready(context.orchestrator):
        return False
    try:
        return asyncio.run(_authenticated_gateway_ready(context.orchestrator))
    except RuntimeError:
        return False


def _voice_liveness_ready(context: ProbeContext) -> bool:
    if context.elapsed_seconds < 0.25 or context.process.poll() is not None:
        return False
    request = urllib.request.Request(
        f"http://127.0.0.1:{context.orchestrator.control_port}/v1/beta-session",
        headers={
            "Accept": "application/json",
            "Authorization": (
                f"Bearer {context.orchestrator.control_bearer_token}"
            ),
            "Host": f"127.0.0.1:{context.orchestrator.control_port}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            body = response.read(4 * 1024 + 1)
        payload = json.loads(body.decode("utf-8", errors="strict"))
        return (
            response.status == 200
            and len(body) <= 4 * 1024
            and isinstance(payload, dict)
            and set(payload)
            == {
                "core_ready",
                "effective_capabilities",
                "schema_version",
                "voice_client_connected",
                "voice_connection_generation",
                "voice_connection_tracker_id",
            }
            and payload.get("schema_version")
            == BETA_SESSION_SCHEMA_VERSION
            and payload.get("core_ready") is True
            and payload.get("voice_client_connected") is True
            and type(payload.get("voice_connection_generation")) is int
            and payload["voice_connection_generation"] >= 0
            and valid_voice_connection_tracker_id(
                payload.get("voice_connection_tracker_id")
            )
            and isinstance(payload.get("effective_capabilities"), dict)
            and set(payload["effective_capabilities"])
            == {
                "cloud",
                "memory_retrospective",
                "memory_shadow",
                "research",
                "tools",
            }
            and all(
                type(value) is bool
                for value in payload["effective_capabilities"].values()
            )
        )
    except (
        OSError,
        TimeoutError,
        UnicodeError,
        ValueError,
        urllib.error.URLError,
    ):
        return False


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _stack_interlock_bindings(stack: PrivateStackConfig) -> tuple[str, str]:
    identity = stack.deployment_identity()
    components = identity.get("components")
    stack_digest = (
        components.get("stack_manifest_sha256")
        if isinstance(components, Mapping)
        else None
    )
    deployment_digest = identity.get("deployment_sha256")
    if (
        not isinstance(stack_digest, str)
        or _HEX64.fullmatch(stack_digest) is None
        or not isinstance(deployment_digest, str)
        or _HEX64.fullmatch(deployment_digest) is None
    ):
        raise ValueError("stack_interlock_identity_invalid")
    return stack_digest, deployment_digest


def _open_private_session_parent(path: Path) -> int:
    """Open one exact owner-only directory without following its final link."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise OSError(errno.EINVAL, "session_parent_invalid")
    try:
        resolved = path.resolve(strict=True)
        before = path.lstat()
    except OSError:
        raise OSError(errno.EINVAL, "session_parent_invalid") from None
    if resolved != path or stat.S_ISLNK(before.st_mode):
        raise OSError(errno.EINVAL, "session_parent_invalid")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(before.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)
            or current.st_uid != os.getuid()
            or current.st_mode & 0o777 != 0o700
        ):
            raise OSError(errno.EPERM, "session_parent_invalid")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "interlock_write_failed")
        offset += written


def _write_private_at(
    parent_descriptor: int,
    name: str,
    value: Mapping[str, Any],
    *,
    exclusive: bool,
) -> None:
    payload = _canonical(value) + b"\n"
    if len(payload) > _MAX_INTERLOCK_BYTES:
        raise OSError(errno.EFBIG, "interlock_too_large")
    target = name
    temporary: str | None = None
    if not exclusive:
        temporary = f".{name}.{secrets.token_hex(16)}.tmp"
        target = temporary
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(
        target,
        flags,
        0o600,
        dir_fd=parent_descriptor,
    )
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o777 != 0o600
            or metadata.st_nlink != 1
        ):
            raise OSError(errno.EPERM, "interlock_file_invalid")
    except BaseException:
        os.close(descriptor)
        try:
            os.unlink(target, dir_fd=parent_descriptor)
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    if temporary is not None:
        try:
            os.replace(
                temporary,
                name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise
    os.fsync(parent_descriptor)


def _read_private_at(
    parent_descriptor: int,
    name: str,
) -> dict[str, Any] | None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o777 != 0o600
            or metadata.st_nlink != 1
            or not 1 <= metadata.st_size <= _MAX_INTERLOCK_BYTES
        ):
            raise OSError(errno.EPERM, "interlock_file_invalid")
        chunks: list[bytes] = []
        remaining = _MAX_INTERLOCK_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(4096, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > _MAX_INTERLOCK_BYTES:
            raise OSError(errno.EFBIG, "interlock_too_large")
        parsed = json.loads(payload.decode("utf-8", errors="strict"))
        if not isinstance(parsed, dict):
            raise ValueError("interlock_document_invalid")
        return parsed
    finally:
        os.close(descriptor)


def _identity_document(identity: _ProcessIdentity) -> dict[str, object]:
    return {
        "leader_pid": identity.pid,
        "process_group_id": identity.process_group_id,
        "start_token": identity.start_token,
        "uid": identity.uid,
    }


def _parse_identity_document(value: Any) -> _ProcessIdentity | None:
    if not isinstance(value, Mapping) or set(value) != {
        "leader_pid",
        "process_group_id",
        "start_token",
        "uid",
    }:
        return None
    pid = value.get("leader_pid")
    process_group_id = value.get("process_group_id")
    start_token = value.get("start_token")
    uid = value.get("uid")
    if (
        type(pid) is not int
        or pid <= 1
        or type(process_group_id) is not int
        or process_group_id <= 1
        or type(uid) is not int
        or uid < 0
        or not isinstance(start_token, str)
        or _HEX64.fullmatch(start_token) is None
    ):
        return None
    return _ProcessIdentity(pid, process_group_id, uid, 0, start_token)


def _parse_interlock_document(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) != {
        "components",
        "deployment_identity_sha256",
        "session_id",
        "spawn_pending",
        "stack_identity_sha256",
        "supervisor",
        "version",
    }:
        return None
    if value.get("version") != SESSION_INTERLOCK_VERSION:
        return None
    for name in (
        "deployment_identity_sha256",
        "session_id",
        "stack_identity_sha256",
    ):
        candidate = value.get(name)
        if not isinstance(candidate, str) or _HEX64.fullmatch(candidate) is None:
            return None
    supervisor = _parse_identity_document(value.get("supervisor"))
    if supervisor is None or supervisor.uid != os.getuid():
        return None
    components = value.get("components")
    if not isinstance(components, Mapping):
        return None
    names = set(components)
    if names not in (
        set(),
        {"inference"},
        {"inference", "orchestrator"},
        set(_COMPONENTS),
    ):
        return None
    parsed_components: dict[str, dict[str, object]] = {}
    identities: list[_ProcessIdentity] = []
    for component in _COMPONENTS:
        if component not in components:
            continue
        identity = _parse_identity_document(components[component])
        if (
            identity is None
            or identity.uid != os.getuid()
            or identity.pid != identity.process_group_id
        ):
            return None
        identities.append(identity)
        parsed_components[component] = _identity_document(identity)
    if len({identity.pid for identity in identities}) != len(identities):
        return None
    pending = value.get("spawn_pending")
    if pending is not None:
        if not isinstance(pending, str) or pending not in _COMPONENTS:
            return None
        if len(parsed_components) >= len(_COMPONENTS):
            return None
        expected_pending = _COMPONENTS[len(parsed_components)]
        if pending != expected_pending:
            return None
    return {
        "components": parsed_components,
        "deployment_identity_sha256": value["deployment_identity_sha256"],
        "session_id": value["session_id"],
        "spawn_pending": pending,
        "stack_identity_sha256": value["stack_identity_sha256"],
        "supervisor": _identity_document(supervisor),
        "version": SESSION_INTERLOCK_VERSION,
    }


class _DurableSessionInterlock:
    def __init__(
        self,
        parent_descriptor: int,
        document: dict[str, Any],
    ) -> None:
        self._parent_descriptor = parent_descriptor
        self._document = document
        self._closed = False

    @classmethod
    def create(
        cls,
        parent: Path,
        stack: PrivateStackConfig,
        inspect_process: ProcessIdentityInspector,
    ) -> _DurableSessionInterlock:
        descriptor = _open_private_session_parent(parent)
        try:
            supervisor = inspect_process(os.getpid())
            if (
                supervisor is None
                or supervisor.pid != os.getpid()
                or supervisor.uid != os.getuid()
                or _HEX64.fullmatch(supervisor.start_token) is None
            ):
                raise OSError(errno.EPERM, "supervisor_identity_unavailable")
            stack_digest, deployment_digest = _stack_interlock_bindings(stack)
            document = {
                "components": {},
                "deployment_identity_sha256": deployment_digest,
                "session_id": secrets.token_hex(32),
                "spawn_pending": None,
                "stack_identity_sha256": stack_digest,
                "supervisor": _identity_document(supervisor),
                "version": SESSION_INTERLOCK_VERSION,
            }
            _write_private_at(
                descriptor,
                SESSION_INTERLOCK_FILENAME,
                document,
                exclusive=True,
            )
            return cls(descriptor, document)
        except BaseException:
            os.close(descriptor)
            raise

    def _replace(self, document: dict[str, Any]) -> None:
        if self._closed:
            raise OSError(errno.EBADF, "interlock_closed")
        current = _read_private_at(
            self._parent_descriptor,
            SESSION_INTERLOCK_FILENAME,
        )
        if current != self._document:
            raise OSError(errno.EPERM, "interlock_identity_changed")
        _write_private_at(
            self._parent_descriptor,
            SESSION_INTERLOCK_FILENAME,
            document,
            exclusive=False,
        )
        self._document = document

    def mark_spawn_pending(self, component: str) -> None:
        if component not in _COMPONENTS:
            raise OSError(errno.EINVAL, "interlock_component_invalid")
        expected = _COMPONENTS[len(self._document["components"])]
        if self._document["spawn_pending"] is not None or component != expected:
            raise OSError(errno.EINVAL, "interlock_spawn_order_invalid")
        self._replace({**self._document, "spawn_pending": component})

    def record_process(self, component: str, identity: _ProcessIdentity) -> None:
        if (
            self._document["spawn_pending"] != component
            or identity.uid != os.getuid()
            or identity.pid != identity.process_group_id
            or _HEX64.fullmatch(identity.start_token) is None
        ):
            raise OSError(errno.EPERM, "interlock_process_identity_invalid")
        components = dict(self._document["components"])
        components[component] = _identity_document(identity)
        self._replace(
            {
                **self._document,
                "components": components,
                "spawn_pending": None,
            }
        )

    def clear(self) -> None:
        if self._closed:
            raise OSError(errno.EBADF, "interlock_closed")
        current = _read_private_at(
            self._parent_descriptor,
            SESSION_INTERLOCK_FILENAME,
        )
        if current != self._document:
            raise OSError(errno.EPERM, "interlock_identity_changed")
        os.unlink(SESSION_INTERLOCK_FILENAME, dir_fd=self._parent_descriptor)
        os.fsync(self._parent_descriptor)

    def close(self) -> None:
        if not self._closed:
            os.close(self._parent_descriptor)
            self._closed = True


@dataclass(slots=True)
class SessionDependencies:
    spawn: SpawnProcess = _spawn_process
    port_available: PortAvailable = _port_available
    listener_pids: ListenerPids = _listener_pids
    preflight_ready: PreflightReady = _beta_preflight_ready
    inference_ready: ReadinessProbe = _inference_ready
    inference_warmup: InferenceWarmup = _inference_warmup
    orchestrator_ready: ReadinessProbe = _orchestrator_ready
    voice_ready: ReadinessProbe = _voice_liveness_ready
    activate_live_lease: ActivateLiveLease = _activate_live_lease
    process_identity: ProcessIdentityInspector = _inspect_process_identity
    process_group: ProcessGroupInspector = _inspect_process_group
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    platform: str = field(default_factory=lambda: sys.platform)
    signal_api: Any = signal


@dataclass(frozen=True, slots=True)
class MacOSBetaSessionReport:
    reason_code: str
    startup_completed: bool

    def __post_init__(self) -> None:
        if self.reason_code not in _STOP_REASONS or type(self.startup_completed) is not bool:
            raise MacOSBetaSessionError("session_failed")

    def canonical(self) -> dict[str, object]:
        return {
            "reason_code": self.reason_code,
            "shutdown_complete": True,
            "startup_completed": self.startup_completed,
            "status": "stopped",
            "version": SESSION_REPORT_VERSION,
        }


@dataclass(frozen=True, slots=True)
class MacOSBetaRecoveryReport:
    interlock_present: bool
    recovered: bool

    def canonical(self) -> dict[str, object]:
        return {
            "interlock_present": self.interlock_present,
            "recovered": self.recovered,
            "safe_to_relaunch": True,
            "version": RECOVERY_REPORT_VERSION,
        }


def _identity_matches(
    current: _ProcessIdentity,
    expected: _ProcessIdentity,
) -> bool:
    return (
        current.pid == expected.pid
        and current.process_group_id == expected.process_group_id
        and current.uid == expected.uid
        and current.start_token == expected.start_token
    )


def recover_macos_beta_session(
    *,
    stack_path: str | Path,
    session_parent: str | Path,
    dependencies: SessionDependencies | None = None,
) -> MacOSBetaRecoveryReport:
    """Clear a stale interlock only after proving the old session is absent.

    Recovery is deliberately verification-only. It never signals a PID or a
    process group; an active, reused, or uncertain numeric identity is left for
    explicit operator investigation.
    """

    selected = dependencies or SessionDependencies()
    if selected.platform != "darwin":
        raise MacOSBetaSessionError("macos_required")
    try:
        stack = PrivateStackConfig.load(Path(stack_path).expanduser())
        stack_digest, deployment_digest = _stack_interlock_bindings(stack)
    except Exception:
        raise MacOSBetaSessionError("private_stack_invalid") from None
    try:
        parent_descriptor = _open_private_session_parent(
            Path(session_parent).expanduser()
        )
    except Exception:
        raise MacOSBetaSessionError("session_parent_invalid") from None
    try:
        try:
            raw = _read_private_at(
                parent_descriptor,
                SESSION_INTERLOCK_FILENAME,
            )
        except Exception:
            raise MacOSBetaSessionError("session_recovery_unsafe") from None
        if raw is None:
            return MacOSBetaRecoveryReport(
                interlock_present=False,
                recovered=False,
            )
        document = _parse_interlock_document(raw)
        if (
            document is None
            or document["stack_identity_sha256"] != stack_digest
            or document["deployment_identity_sha256"] != deployment_digest
            or document["spawn_pending"] is not None
        ):
            raise MacOSBetaSessionError("session_recovery_unsafe")
        supervisor = _parse_identity_document(document["supervisor"])
        assert supervisor is not None
        try:
            current_supervisor = selected.process_identity(supervisor.pid)
        except Exception:
            raise MacOSBetaSessionError("session_recovery_unsafe") from None
        if current_supervisor is not None and _identity_matches(
            current_supervisor,
            supervisor,
        ):
            raise MacOSBetaSessionError("session_recovery_unsafe")
        for component in _COMPONENTS:
            raw_identity = document["components"].get(component)
            if raw_identity is None:
                continue
            expected = _parse_identity_document(raw_identity)
            assert expected is not None
            try:
                current = selected.process_identity(expected.pid)
                members = selected.process_group(expected.process_group_id)
            except Exception:
                raise MacOSBetaSessionError("session_recovery_unsafe") from None
            # PID equality is never enough: only the original start token can
            # identify the recorded leader. Any extant numeric group is still
            # unsafe because it may contain an owned helper or a recycled group.
            if (
                current is not None
                and _identity_matches(current, expected)
            ) or members is None or bool(members):
                raise MacOSBetaSessionError("session_recovery_unsafe")
        for port in REQUIRED_PORTS:
            try:
                listeners = selected.listener_pids(port)
            except Exception:
                listeners = None
            if listeners != frozenset():
                raise MacOSBetaSessionError("session_recovery_unsafe")
        try:
            if (
                _read_private_at(
                    parent_descriptor,
                    SESSION_INTERLOCK_FILENAME,
                )
                != raw
            ):
                raise OSError(errno.EPERM, "interlock_identity_changed")
            os.unlink(SESSION_INTERLOCK_FILENAME, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        except Exception:
            raise MacOSBetaSessionError("session_recovery_unsafe") from None
        return MacOSBetaRecoveryReport(
            interlock_present=True,
            recovered=True,
        )
    finally:
        os.close(parent_descriptor)


class _StopRequested(Exception):
    pass


class MacOSBetaSession:
    """Own and supervise one complete local macOS CYRUS beta session."""

    def __init__(
        self,
        *,
        stack_path: str | Path,
        orchestrator_command: CommandPrefix,
        voice_command: CommandPrefix,
        session_parent: str | Path,
        dependencies: SessionDependencies | None = None,
        startup_timeout_seconds: float = 60.0,
        shutdown_grace_seconds: float = 5.0,
        kill_timeout_seconds: float = 2.0,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        self.stack_path = Path(stack_path).expanduser()
        self.orchestrator_command = orchestrator_command
        self.voice_command = voice_command
        self.session_parent = Path(session_parent).expanduser()
        self.dependencies = dependencies or SessionDependencies()
        self.startup_timeout_seconds = startup_timeout_seconds
        self.shutdown_grace_seconds = shutdown_grace_seconds
        self.kill_timeout_seconds = kill_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_requested = threading.Event()
        self._stop_reason = "operator_requested"
        self._children: dict[str, ChildProcess] = {}
        self._logs: dict[str, BinaryIO] = {}
        self._session_directory: Path | None = None
        self._interlock: _DurableSessionInterlock | None = None

    @property
    def session_directory(self) -> Path | None:
        """Private diagnostic location; intentionally absent from reports."""

        return self._session_directory

    def request_stop(self, reason_code: str = "operator_requested") -> None:
        if reason_code not in _STOP_REASONS:
            raise MacOSBetaSessionError("stop_reason_invalid")
        self._stop_reason = reason_code
        self._stop_requested.set()

    def _validate_settings(self) -> tuple[PrivateStackConfig, OrchestratorConfig]:
        if self.dependencies.platform != "darwin":
            raise MacOSBetaSessionError("macos_required")
        for value in (
            self.startup_timeout_seconds,
            self.shutdown_grace_seconds,
            self.kill_timeout_seconds,
            self.poll_interval_seconds,
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise MacOSBetaSessionError("session_timing_invalid")
        try:
            stack = PrivateStackConfig.load(self.stack_path)
            orchestrator = OrchestratorConfig.load(stack.orchestrator_config_path)
        except Exception:
            raise MacOSBetaSessionError("private_stack_invalid") from None
        if (
            orchestrator.voice_host != "127.0.0.1"
            or orchestrator.voice_port != 8765
            or orchestrator.control_host != "127.0.0.1"
            or orchestrator.control_port != 8767
        ):
            raise MacOSBetaSessionError("fixed_loopback_endpoints_required")
        self._validate_command(self.orchestrator_command)
        self._validate_command(self.voice_command)
        self._validate_executable(stack.inference.runner_path)
        self._validate_session_parent()
        self._refuse_existing_interlock()
        try:
            preflight_ready = self.dependencies.preflight_ready(stack)
        except Exception:
            preflight_ready = False
        if not preflight_ready:
            raise MacOSBetaSessionError("beta_preflight_failed")
        for port in REQUIRED_PORTS:
            try:
                available = self.dependencies.port_available(port)
            except Exception:
                available = False
            if not available:
                raise MacOSBetaSessionError("required_port_occupied")
        return stack, orchestrator

    @staticmethod
    def _validate_executable(path: Path) -> None:
        if not isinstance(path, Path):
            raise MacOSBetaSessionError("executable_invalid")
        if not path.is_absolute():
            raise MacOSBetaSessionError("executable_invalid")
        try:
            resolved = path.resolve(strict=True)
            metadata = resolved.stat()
        except OSError:
            raise MacOSBetaSessionError("executable_invalid") from None
        if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
            raise MacOSBetaSessionError("executable_invalid")

    @classmethod
    def _validate_command(cls, command: CommandPrefix) -> None:
        if not isinstance(command, CommandPrefix):
            raise MacOSBetaSessionError("command_prefix_invalid")
        cls._validate_executable(command.executable)
        if any(
            not isinstance(item, str)
            or not item
            or "\x00" in item
            or len(item.encode("utf-8")) > 4096
            for item in command.arguments
        ):
            raise MacOSBetaSessionError("command_prefix_invalid")

    def _validate_session_parent(self) -> None:
        try:
            descriptor = _open_private_session_parent(self.session_parent)
        except Exception:
            raise MacOSBetaSessionError("session_parent_invalid") from None
        os.close(descriptor)

    def _refuse_existing_interlock(self) -> None:
        try:
            descriptor = _open_private_session_parent(self.session_parent)
        except Exception:
            raise MacOSBetaSessionError("session_parent_invalid") from None
        try:
            try:
                os.stat(
                    SESSION_INTERLOCK_FILENAME,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return
            except OSError:
                # An unreadable, dangling, or otherwise uncertain directory
                # entry is itself a durable fail-closed interlock.
                raise MacOSBetaSessionError("session_relaunch_blocked") from None
            raise MacOSBetaSessionError("session_relaunch_blocked")
        finally:
            os.close(descriptor)

    def _create_interlock(self, stack: PrivateStackConfig) -> None:
        try:
            self._interlock = _DurableSessionInterlock.create(
                self.session_parent,
                stack,
                self.dependencies.process_identity,
            )
        except FileExistsError:
            raise MacOSBetaSessionError("session_relaunch_blocked") from None
        except Exception:
            raise MacOSBetaSessionError("session_interlock_failed") from None

    def _create_private_session_files(self) -> None:
        try:
            directory = Path(
                tempfile.mkdtemp(
                    prefix=".jarvis-macos-beta-",
                    dir=self.session_parent,
                )
            )
            directory.chmod(0o700)
            metadata = directory.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or (os.name != "nt" and metadata.st_mode & 0o077)
            ):
                raise OSError
            self._session_directory = directory
            for component in _COMPONENTS:
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(directory / f"{component}.log", flags, 0o600)
                try:
                    os.fchmod(descriptor, 0o600)
                    self._logs[component] = os.fdopen(
                        descriptor,
                        "wb",
                        buffering=0,
                    )
                except BaseException:
                    os.close(descriptor)
                    raise
        except Exception:
            self._close_logs()
            raise MacOSBetaSessionError("session_directory_create_failed") from None

    @staticmethod
    def _commands(
        stack: PrivateStackConfig,
        orchestrator_command: CommandPrefix,
        voice_command: CommandPrefix,
    ) -> dict[str, tuple[str, ...]]:
        return {
            "inference": (
                str(stack.inference.runner_path),
                "--manifest",
                str(stack.inference.manifest_path),
                "--qualification-report",
                str(stack.inference.qualification_report_path),
                "--orchestrator-config",
                str(stack.orchestrator_config_path),
            ),
            "orchestrator": orchestrator_command.command(
                "--config",
                str(stack.orchestrator_config_path),
                "serve",
            ),
            "voice": voice_command.command(
                "run-macos",
                "--config",
                str(stack.voice_config_path),
            ),
        }

    def _spawn(self, component: str, command: tuple[str, ...]) -> ChildProcess:
        assert self._session_directory is not None
        if self._interlock is None:
            raise MacOSBetaSessionError("session_interlock_failed")
        try:
            self._interlock.mark_spawn_pending(component)
        except BaseException:
            raise MacOSBetaSessionError("session_interlock_failed") from None
        try:
            process = self.dependencies.spawn(
                component,
                command,
                self._logs[component],
                self._session_directory,
            )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise MacOSBetaSessionError(f"{component}_spawn_failed") from None
        self._children[component] = process
        try:
            identity = self.dependencies.process_identity(process.pid)
            if identity is None:
                raise OSError(errno.ESRCH, "process_identity_unavailable")
            self._interlock.record_process(component, identity)
        except BaseException:
            raise MacOSBetaSessionError("session_interlock_failed") from None
        return process

    def _wait_ready(
        self,
        component: str,
        process: ChildProcess,
        probe: ReadinessProbe,
        stack: PrivateStackConfig,
        orchestrator: OrchestratorConfig,
    ) -> None:
        started = self.dependencies.monotonic()
        deadline = started + self.startup_timeout_seconds
        while True:
            if self._stop_requested.is_set():
                raise _StopRequested
            if process.poll() is not None:
                raise MacOSBetaSessionError(f"{component}_startup_failed")
            now = self.dependencies.monotonic()
            context = ProbeContext(
                component=component,
                stack=stack,
                orchestrator=orchestrator,
                process=process,
                elapsed_seconds=max(0.0, now - started),
            )
            try:
                ready = probe(context)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                ready = False
            if ready and process.poll() is None:
                return
            if now >= deadline:
                raise MacOSBetaSessionError(f"{component}_readiness_timeout")
            self.dependencies.sleep(
                min(self.poll_interval_seconds, max(0.0, deadline - now))
            )

    def _monitor(self, lease: SessionLease) -> str:
        while True:
            if self._stop_requested.is_set():
                return self._stop_reason
            for component in _COMPONENTS:
                process = self._children[component]
                if process.poll() is not None:
                    raise MacOSBetaSessionError(f"{component}_runtime_failed")
            try:
                lease.heartbeat()
            except Exception:
                raise MacOSBetaSessionError("live_lease_runtime_failed") from None
            self.dependencies.sleep(self.poll_interval_seconds)

    def _shutdown(self) -> None:
        incomplete = False
        try:
            for component in reversed(_COMPONENTS):
                process = self._children.get(component)
                if process is None:
                    continue
                exited = False
                try:
                    process.terminate()
                except BaseException:
                    pass
                try:
                    exited = process.wait(self.shutdown_grace_seconds)
                except BaseException:
                    exited = False
                if not exited:
                    try:
                        process.kill()
                    except BaseException:
                        pass
                    try:
                        exited = process.wait(self.kill_timeout_seconds)
                    except BaseException:
                        exited = False
                try:
                    still_running = process.poll() is None
                except BaseException:
                    still_running = True
                if not exited or still_running:
                    incomplete = True
            if self._interlock is not None:
                for port in REQUIRED_PORTS:
                    try:
                        listeners = self.dependencies.listener_pids(port)
                    except BaseException:
                        listeners = None
                    if listeners != frozenset():
                        incomplete = True
                if not incomplete:
                    try:
                        self._interlock.clear()
                    except BaseException:
                        incomplete = True
        finally:
            self._close_logs()
            if self._interlock is not None:
                self._interlock.close()
                self._interlock = None
        if incomplete:
            raise MacOSBetaSessionError("session_shutdown_incomplete")

    def _close_logs(self) -> None:
        for log in self._logs.values():
            try:
                log.close()
            except OSError:
                pass
        self._logs.clear()

    def _install_signal_handlers(self) -> dict[int, Any]:
        previous: dict[int, Any] = {}

        def stop(_signum: int, _frame: Any) -> None:
            self.request_stop("signal")

        try:
            for value in (
                self.dependencies.signal_api.SIGINT,
                self.dependencies.signal_api.SIGTERM,
            ):
                previous[value] = self.dependencies.signal_api.signal(value, stop)
        except Exception:
            for value, handler in previous.items():
                try:
                    self.dependencies.signal_api.signal(value, handler)
                except Exception:
                    pass
            raise MacOSBetaSessionError("signal_handlers_unavailable") from None
        return previous

    def _restore_signal_handlers(self, previous: dict[int, Any]) -> None:
        for value, handler in previous.items():
            try:
                self.dependencies.signal_api.signal(value, handler)
            except Exception:
                pass

    def run(self) -> MacOSBetaSessionReport:
        """Run until operator signal or a child failure, always reaping children."""

        stack, orchestrator = self._validate_settings()
        commands = self._commands(
            stack,
            self.orchestrator_command,
            self.voice_command,
        )
        self._create_private_session_files()
        previous_handlers: dict[int, Any] = {}
        startup_completed = False
        primary_error: BaseException | None = None
        live_lease: SessionLease | None = None
        reason_code = "operator_requested"
        try:
            self._create_interlock(stack)
            previous_handlers = self._install_signal_handlers()
            inference = self._spawn("inference", commands["inference"])
            self._wait_ready(
                "inference",
                inference,
                self.dependencies.inference_ready,
                stack,
                orchestrator,
            )
            for _warmup_index in range(2):
                try:
                    inference_warm = self.dependencies.inference_warmup(
                        stack,
                        orchestrator,
                    )
                except Exception:
                    inference_warm = False
                if not inference_warm:
                    raise MacOSBetaSessionError("inference_warmup_failed")
            orchestrator_process = self._spawn(
                "orchestrator", commands["orchestrator"]
            )
            self._wait_ready(
                "orchestrator",
                orchestrator_process,
                self.dependencies.orchestrator_ready,
                stack,
                orchestrator,
            )
            voice = self._spawn("voice", commands["voice"])
            self._wait_ready(
                "voice",
                voice,
                self.dependencies.voice_ready,
                stack,
                orchestrator,
            )
            try:
                live_lease = self.dependencies.activate_live_lease(
                    stack,
                    self._children,
                    commands,
                )
            except Exception:
                raise MacOSBetaSessionError("live_lease_activation_failed") from None
            startup_completed = True
            reason_code = self._monitor(live_lease)
        except _StopRequested:
            reason_code = self._stop_reason
        except KeyboardInterrupt:
            reason_code = "keyboard_interrupt"
        except BaseException as exc:
            primary_error = exc
        finally:
            if live_lease is not None:
                try:
                    live_lease.close()
                except Exception:
                    if primary_error is None:
                        primary_error = MacOSBetaSessionError(
                            "live_lease_runtime_failed"
                        )
            try:
                self._shutdown()
            except MacOSBetaSessionError as exc:
                # A surviving owned process group is the operator's immediate
                # safety condition and must never be hidden by the runtime
                # failure that originally triggered cleanup.
                if (
                    exc.code == "session_shutdown_incomplete"
                    or primary_error is None
                ):
                    primary_error = exc
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
            self._restore_signal_handlers(previous_handlers)
        if primary_error is not None:
            if isinstance(primary_error, MacOSBetaSessionError):
                raise primary_error
            raise MacOSBetaSessionError("session_failed") from None
        return MacOSBetaSessionReport(
            reason_code=reason_code,
            startup_completed=startup_completed,
        )


__all__ = [
    "CommandPrefix",
    "MacOSBetaSession",
    "MacOSBetaSessionError",
    "MacOSBetaRecoveryReport",
    "MacOSBetaSessionReport",
    "ProbeContext",
    "RECOVERY_REPORT_VERSION",
    "SESSION_REPORT_VERSION",
    "SESSION_INTERLOCK_FILENAME",
    "SESSION_INTERLOCK_VERSION",
    "SessionDependencies",
    "recover_macos_beta_session",
]
