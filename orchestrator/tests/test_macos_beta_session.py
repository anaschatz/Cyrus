from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import OrchestratorError
from jarvis_orchestrator.inference import InferenceIdentityEndpointUnavailable
from jarvis_orchestrator.macos_beta_session import (
    CommandPrefix,
    MacOSBetaRecoveryReport,
    MacOSBetaSession,
    MacOSBetaSessionError,
    ProbeContext,
    SESSION_INTERLOCK_FILENAME,
    SessionDependencies,
    _PopenProcess,
    _ProcessIdentity,
    _ProcessGroupMember,
    _inference_ready,
    _inspect_process_group,
    _voice_liveness_ready,
    recover_macos_beta_session,
)
from jarvis_orchestrator.stack_cli import main as stack_main
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


class FakeHttpResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _maximum: int | None = None) -> bytes:
        return self.body


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.on_sleep = None

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds
        if self.on_sleep is not None:
            self.on_sleep()


class FakeSignal:
    SIGINT = 2
    SIGTERM = 15

    def __init__(self) -> None:
        self.handlers: dict[int, object] = {
            self.SIGINT: "original-int",
            self.SIGTERM: "original-term",
        }

    def signal(self, value: int, handler: object) -> object:
        previous = self.handlers[value]
        self.handlers[value] = handler
        return previous


class FakeProcess:
    def __init__(
        self,
        component: str,
        events: list[str],
        *,
        ignore_term: bool = False,
        ignore_kill: bool = False,
    ) -> None:
        self.component = component
        self.pid = {
            "inference": 4101,
            "orchestrator": 4102,
            "voice": 4103,
        }[component]
        self.events = events
        self.ignore_term = ignore_term
        self.ignore_kill = ignore_kill
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.events.append(f"terminate:{self.component}")
        if not self.ignore_term:
            self.returncode = -15

    def kill(self) -> None:
        self.events.append(f"kill:{self.component}")
        if not self.ignore_kill:
            self.returncode = -9

    def wait(self, timeout_seconds: float) -> bool:
        self.events.append(f"wait:{self.component}")
        return self.returncode is not None


class FakeSpawner:
    def __init__(
        self,
        *,
        fail_component: str | None = None,
        ignore_term: bool = False,
        ignore_kill: bool = False,
    ) -> None:
        self.fail_component = fail_component
        self.ignore_term = ignore_term
        self.ignore_kill = ignore_kill
        self.events: list[str] = []
        self.commands: dict[str, tuple[str, ...]] = {}
        self.processes: dict[str, FakeProcess] = {}
        self.logs: dict[str, Path] = {}

    def __call__(
        self,
        component: str,
        command: tuple[str, ...],
        log: BinaryIO,
        _session_directory: Path,
    ) -> FakeProcess:
        self.events.append(f"spawn:{component}")
        self.commands[component] = command
        self.logs[component] = _session_directory / f"{component}.log"
        if self.fail_component == component:
            raise OSError("private detail must not escape")
        process = FakeProcess(
            component,
            self.events,
            ignore_term=self.ignore_term,
            ignore_kill=self.ignore_kill,
        )
        self.processes[component] = process
        return process


class FakeLease:
    def __init__(self, *, heartbeat_fails: bool = False) -> None:
        self.heartbeat_fails = heartbeat_fails
        self.heartbeat_count = 0
        self.closed = False

    def heartbeat(self, *, force: bool = False) -> None:
        del force
        self.heartbeat_count += 1
        if self.heartbeat_fails:
            raise RuntimeError("private detail")

    def close(self) -> None:
        self.closed = True


class FakePopenHandle:
    def __init__(self, *, pid: int = 8101, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = 7
        return self.returncode


class PopenProcessGroupTests(unittest.TestCase):
    def test_exited_leader_keeps_identity_anchor_until_helper_is_stopped(self) -> None:
        popen = FakePopenHandle()
        helper_running = True
        signals: list[tuple[int, int]] = []

        def inspect(pgid: int) -> tuple[_ProcessGroupMember, ...]:
            members = [
                _ProcessGroupMember(popen.pid, pgid, os.getuid(), 5),
            ]
            if helper_running:
                members.append(
                    _ProcessGroupMember(popen.pid + 1, pgid, os.getuid(), 2)
                )
            return tuple(members)

        def signal_group(pgid: int, value: int) -> None:
            nonlocal helper_running
            signals.append((pgid, value))
            helper_running = False

        process = _PopenProcess(
            popen,  # type: ignore[arg-type]
            group_inspector=inspect,
            group_signal=signal_group,
            peek_child_exit=lambda _pid: 7,
        )

        self.assertEqual(process.poll(), 7)
        self.assertIsNone(popen.returncode, "waitid observation must not reap leader")
        process.terminate()
        self.assertTrue(process.wait(0.1))

        self.assertEqual(signals, [(popen.pid, signal.SIGTERM)])
        self.assertEqual(popen.returncode, 7)
        self.assertEqual(popen.wait_calls, 1)

    def test_reaped_identity_never_signals_reused_unrelated_group(self) -> None:
        popen = FakePopenHandle(returncode=9)
        signals: list[tuple[int, int]] = []
        unrelated = (
            _ProcessGroupMember(popen.pid + 100, popen.pid, os.getuid(), 2),
        )
        process = _PopenProcess(
            popen,  # type: ignore[arg-type]
            group_inspector=lambda _pgid: unrelated,
            group_signal=lambda pgid, value: signals.append((pgid, value)),
            peek_child_exit=lambda _pid: 9,
        )

        process.terminate()
        process.kill()

        self.assertEqual(signals, [])
        self.assertFalse(process.wait(0.0))

    def test_external_reap_loses_anchor_before_any_group_signal(self) -> None:
        popen = FakePopenHandle()
        signals: list[tuple[int, int]] = []

        def identity_lost(_pid: int) -> int | None:
            raise ChildProcessError("already reaped")

        process = _PopenProcess(
            popen,  # type: ignore[arg-type]
            group_inspector=lambda _pgid: (
                _ProcessGroupMember(
                    popen.pid + 100,
                    popen.pid,
                    os.getuid(),
                    2,
                ),
            ),
            group_signal=lambda pgid, value: signals.append((pgid, value)),
            peek_child_exit=identity_lost,
        )

        process.terminate()
        process.kill()

        self.assertEqual(signals, [])

    @unittest.skipUnless(sys.platform == "darwin", "macOS libproc required")
    def test_real_group_cleanup_stops_helper_after_leader_crash(self) -> None:
        # This is the same containment shape as Voice crashing while an
        # inherited-group /usr/bin/say child is still active.
        script = (
            "import subprocess; "
            "child=subprocess.Popen(['/bin/sleep','30'], "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
            "stderr=subprocess.DEVNULL, close_fds=True); "
            "print(child.pid, flush=True); raise SystemExit(23)"
        )
        parent = subprocess.Popen(
            (sys.executable, "-c", script),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            text=True,
        )
        process = _PopenProcess(parent)  # type: ignore[arg-type]
        try:
            assert parent.stdout is not None
            helper_pid = int(parent.stdout.readline().strip())
            deadline = time.monotonic() + 2.0
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(process.poll(), 23)
            self.assertIsNone(parent.returncode)
            self.assertEqual(os.getpgid(helper_pid), parent.pid)

            process.terminate()
            self.assertTrue(process.wait(2.0))
            remaining = _inspect_process_group(parent.pid)
            self.assertIsNotNone(remaining)
            self.assertFalse(
                any(member.status != 5 for member in remaining or ()),
            )
        finally:
            if parent.stdout is not None:
                parent.stdout.close()
            # It is safe to address this numeric group only while the original
            # leader is still our unreaped child.
            if parent.returncode is None:
                try:
                    os.killpg(parent.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    parent.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass


class MacOSBetaSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.root.chmod(0o700)
        self.auth = self.root / "voice-auth.json"
        create_voice_auth_key_file(self.auth, key=b"a" * 32)
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
                "control_bearer_token": "Y2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2M",
                "inference_bearer_token": "aWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWk",
                "voice_auth_key_file": str(self.auth),
            },
        )
        self.runner = self.executable("inference-runner")
        self.manifest_document = {
            "alias": "jarvis-local",
            "manifest_schema_version": 1,
        }
        self.manifest = self.private_json("manifest.json", self.manifest_document)
        self.qualification = self.private_json(
            "qualification.json",
            {
                "status": "pass",
                "identity": {
                    "manifest_digest": hashlib.sha256(
                        canonical(self.manifest_document)
                    ).hexdigest()
                },
            },
        )
        self.stack = self.root / "jarvis.private.json"
        write_private_stack_config(
            self.stack,
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
        self.orchestrator_executable = self.executable("orchestrator-command")
        self.voice_executable = self.executable("voice-command")
        self.sessions = self.root / "sessions"
        self.sessions.mkdir(mode=0o700)

    def private_json(self, name: str, value: object) -> Path:
        path = self.root / name
        path.write_bytes(canonical(value) + b"\n")
        path.chmod(0o600)
        return path

    def executable(self, name: str) -> Path:
        path = self.root / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o700)
        return path

    def dependencies(
        self,
        spawner: FakeSpawner,
        clock: FakeClock,
        *,
        inference_ready=lambda _context: True,
        orchestrator_ready=lambda _context: True,
        voice_ready=lambda _context: True,
        port_available=lambda _port: True,
        listener_pids=lambda _port: frozenset(),
        preflight_ready=lambda _stack: True,
        activate_live_lease=lambda _stack, _children, _commands: FakeLease(),
        process_identity=None,
        process_group=None,
    ) -> SessionDependencies:
        def fake_process_identity(pid: int) -> _ProcessIdentity | None:
            if pid == os.getpid():
                return _ProcessIdentity(
                    pid,
                    os.getpgrp(),
                    os.getuid(),
                    2,
                    hashlib.sha256(f"fake:{pid}".encode()).hexdigest(),
                )
            for process in spawner.processes.values():
                if process.pid == pid:
                    return _ProcessIdentity(
                        pid,
                        pid,
                        os.getuid(),
                        2,
                        hashlib.sha256(f"fake:{pid}".encode()).hexdigest(),
                    )
            return None

        def fake_process_group(
            process_group_id: int,
        ) -> tuple[_ProcessGroupMember, ...]:
            for process in spawner.processes.values():
                if process.pid == process_group_id and process.poll() is None:
                    return (
                        _ProcessGroupMember(
                            process.pid,
                            process.pid,
                            os.getuid(),
                            2,
                            hashlib.sha256(
                                f"fake:{process.pid}".encode()
                            ).hexdigest(),
                        ),
                    )
            return ()

        return SessionDependencies(
            spawn=spawner,
            port_available=port_available,
            listener_pids=listener_pids,
            preflight_ready=preflight_ready,
            inference_ready=inference_ready,
            orchestrator_ready=orchestrator_ready,
            voice_ready=voice_ready,
            activate_live_lease=activate_live_lease,
            process_identity=process_identity or fake_process_identity,
            process_group=process_group or fake_process_group,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            platform="darwin",
            signal_api=FakeSignal(),
        )

    def session(
        self,
        dependencies: SessionDependencies,
        **values: object,
    ) -> MacOSBetaSession:
        return MacOSBetaSession(
            stack_path=self.stack,
            orchestrator_command=CommandPrefix(
                self.orchestrator_executable,
                ("-m", "jarvis_orchestrator"),
            ),
            voice_command=CommandPrefix(
                self.voice_executable,
                ("-m", "jarvis_voice"),
            ),
            session_parent=self.sessions,
            dependencies=dependencies,
            startup_timeout_seconds=float(values.get("startup_timeout_seconds", 0.3)),
            shutdown_grace_seconds=0.1,
            kill_timeout_seconds=0.1,
            poll_interval_seconds=0.1,
        )

    def test_starts_in_dependency_order_and_stops_in_reverse_order(self) -> None:
        spawner = FakeSpawner()
        clock = FakeClock()
        current: MacOSBetaSession | None = None

        def voice_ready(_context: object) -> bool:
            assert current is not None
            current.request_stop()
            return True

        dependencies = self.dependencies(
            spawner,
            clock,
            voice_ready=voice_ready,
        )
        current = self.session(dependencies)

        report = current.run().canonical()

        self.assertEqual(
            [event for event in spawner.events if not event.startswith("wait:")],
            [
                "spawn:inference",
                "spawn:orchestrator",
                "spawn:voice",
                "terminate:voice",
                "terminate:orchestrator",
                "terminate:inference",
            ],
        )
        self.assertEqual(report["status"], "stopped")
        self.assertTrue(report["startup_completed"])
        self.assertTrue(report["shutdown_complete"])
        self.assertNotIn(str(self.root), json.dumps(report))
        self.assertNotIn("secret", json.dumps(report))

        self.assertEqual(
            spawner.commands["inference"],
            (
                str(self.runner.resolve()),
                "--manifest",
                str(self.manifest.resolve()),
                "--qualification-report",
                str(self.qualification.resolve()),
                "--orchestrator-config",
                str(self.orchestrator.resolve()),
            ),
        )
        self.assertEqual(
            spawner.commands["orchestrator"],
            (
                str(self.orchestrator_executable),
                "-m",
                "jarvis_orchestrator",
                "--config",
                str(self.orchestrator.resolve()),
                "serve",
            ),
        )
        self.assertEqual(
            spawner.commands["voice"],
            (
                str(self.voice_executable),
                "-m",
                "jarvis_voice",
                "run-macos",
                "--config",
                str(self.voice.resolve()),
            ),
        )

    def test_private_session_directory_and_logs_have_owner_only_modes(self) -> None:
        spawner = FakeSpawner()
        clock = FakeClock()
        current: MacOSBetaSession | None = None

        def stop_after_voice(_context: object) -> bool:
            assert current is not None
            current.request_stop()
            return True

        current = self.session(
            self.dependencies(spawner, clock, voice_ready=stop_after_voice)
        )
        current.run()

        assert current.session_directory is not None
        if os.name != "nt":
            self.assertEqual(current.session_directory.stat().st_mode & 0o777, 0o700)
            for path in spawner.logs.values():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_interlock_is_durable_and_owner_only_before_first_spawn(self) -> None:
        spawner = FakeSpawner()
        clock = FakeClock()
        current: MacOSBetaSession | None = None
        observed = False

        def checked_spawn(
            component: str,
            command: tuple[str, ...],
            log: BinaryIO,
            session_directory: Path,
        ) -> FakeProcess:
            nonlocal observed
            path = self.sessions / SESSION_INTERLOCK_FILENAME
            self.assertTrue(path.exists())
            self.assertGreaterEqual(fsync.call_count, 2)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            document = json.loads(path.read_text(encoding="utf-8"))
            serialized = json.dumps(document, sort_keys=True)
            self.assertEqual(document["spawn_pending"], component)
            self.assertNotIn(str(self.root), serialized)
            self.assertNotIn(
                "Y2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2M",
                serialized,
            )
            self.assertEqual(
                set(document),
                {
                    "components",
                    "deployment_identity_sha256",
                    "session_id",
                    "spawn_pending",
                    "stack_identity_sha256",
                    "supervisor",
                    "version",
                },
            )
            observed = True
            return spawner(component, command, log, session_directory)

        def stop_after_voice(_context: object) -> bool:
            assert current is not None
            current.request_stop()
            return True

        dependencies = self.dependencies(
            spawner,
            clock,
            voice_ready=stop_after_voice,
        )
        dependencies.spawn = checked_spawn
        current = self.session(dependencies)
        with patch(
            "jarvis_orchestrator.macos_beta_session.os.fsync",
            wraps=os.fsync,
        ) as fsync:
            current.run()

        self.assertTrue(observed)
        self.assertGreaterEqual(fsync.call_count, 2)
        self.assertFalse((self.sessions / SESSION_INTERLOCK_FILENAME).exists())

    def test_session_parent_requires_exact_owner_only_mode(self) -> None:
        self.sessions.chmod(0o750)
        spawner = FakeSpawner()
        session = self.session(self.dependencies(spawner, FakeClock()))

        with self.assertRaises(MacOSBetaSessionError) as caught:
            session.run()

        self.assertEqual(caught.exception.code, "session_parent_invalid")
        self.assertEqual(spawner.events, [])

    def test_session_parent_rejects_symlinked_path(self) -> None:
        linked = self.root / "linked-sessions"
        linked.symlink_to(self.sessions, target_is_directory=True)
        spawner = FakeSpawner()
        session = MacOSBetaSession(
            stack_path=self.stack,
            orchestrator_command=CommandPrefix(
                self.orchestrator_executable,
                ("-m", "jarvis_orchestrator"),
            ),
            voice_command=CommandPrefix(
                self.voice_executable,
                ("-m", "jarvis_voice"),
            ),
            session_parent=linked,
            dependencies=self.dependencies(spawner, FakeClock()),
            startup_timeout_seconds=0.3,
            shutdown_grace_seconds=0.1,
            kill_timeout_seconds=0.1,
            poll_interval_seconds=0.1,
        )

        with self.assertRaises(MacOSBetaSessionError) as caught:
            session.run()

        self.assertEqual(caught.exception.code, "session_parent_invalid")
        self.assertEqual(spawner.events, [])

    def test_readiness_failure_rolls_back_every_started_stage(self) -> None:
        cases = {
            "inference": ["inference"],
            "orchestrator": ["orchestrator", "inference"],
            "voice": ["voice", "orchestrator", "inference"],
        }
        for failed, reverse_order in cases.items():
            with self.subTest(failed=failed):
                spawner = FakeSpawner()
                clock = FakeClock()
                probes = {
                    "inference_ready": (
                        (lambda _context: False)
                        if failed == "inference"
                        else (lambda _context: True)
                    ),
                    "orchestrator_ready": (
                        (lambda _context: False)
                        if failed == "orchestrator"
                        else (lambda _context: True)
                    ),
                    "voice_ready": (
                        (lambda _context: False)
                        if failed == "voice"
                        else (lambda _context: True)
                    ),
                }
                session = self.session(
                    self.dependencies(spawner, clock, **probes),
                    startup_timeout_seconds=0.2,
                )

                with self.assertRaises(MacOSBetaSessionError) as caught:
                    session.run()

                self.assertEqual(caught.exception.code, f"{failed}_readiness_timeout")
                self.assertEqual(
                    [
                        event.removeprefix("terminate:")
                        for event in spawner.events
                        if event.startswith("terminate:")
                    ],
                    reverse_order,
                )
                self.assertEqual(str(caught.exception), caught.exception.code)

    def test_static_beta_preflight_failure_spawns_nothing(self) -> None:
        spawner = FakeSpawner()
        clock = FakeClock()
        session = self.session(
            self.dependencies(
                spawner,
                clock,
                preflight_ready=lambda _stack: False,
            )
        )

        with self.assertRaises(MacOSBetaSessionError) as caught:
            session.run()

        self.assertEqual(caught.exception.code, "beta_preflight_failed")
        self.assertEqual(spawner.events, [])

    def test_spawn_failure_rolls_back_previously_started_components(self) -> None:
        expected = {
            "inference": [],
            "orchestrator": ["inference"],
            "voice": ["orchestrator", "inference"],
        }
        for failed, reverse_order in expected.items():
            with self.subTest(failed=failed):
                spawner = FakeSpawner(fail_component=failed)
                clock = FakeClock()
                session = self.session(self.dependencies(spawner, clock))

                with self.assertRaises(MacOSBetaSessionError) as caught:
                    session.run()

                self.assertEqual(caught.exception.code, f"{failed}_spawn_failed")
                self.assertEqual(
                    [
                        event.removeprefix("terminate:")
                        for event in spawner.events
                        if event.startswith("terminate:")
                    ],
                    reverse_order,
                )

    def test_term_then_kill_is_bounded_and_reverse_ordered(self) -> None:
        spawner = FakeSpawner(ignore_term=True)
        clock = FakeClock()
        current: MacOSBetaSession | None = None

        def stop_after_voice(_context: object) -> bool:
            assert current is not None
            current.request_stop("signal")
            return True

        current = self.session(
            self.dependencies(spawner, clock, voice_ready=stop_after_voice)
        )
        report = current.run().canonical()

        self.assertEqual(
            [event for event in spawner.events if not event.startswith("spawn:")],
            [
                "terminate:voice",
                "wait:voice",
                "kill:voice",
                "wait:voice",
                "terminate:orchestrator",
                "wait:orchestrator",
                "kill:orchestrator",
                "wait:orchestrator",
                "terminate:inference",
                "wait:inference",
                "kill:inference",
                "wait:inference",
            ],
        )
        self.assertEqual(report["reason_code"], "signal")

    def test_sigterm_and_keyboard_interrupt_use_the_same_cleanup_path(self) -> None:
        for interruption in ("sigterm", "keyboard"):
            with self.subTest(interruption=interruption):
                spawner = FakeSpawner()
                clock = FakeClock()
                signals = FakeSignal()

                def voice_ready(_context: object) -> bool:
                    if interruption == "keyboard":
                        raise KeyboardInterrupt
                    handler = signals.handlers[signals.SIGTERM]
                    assert callable(handler)
                    handler(signals.SIGTERM, None)
                    return True

                dependencies = self.dependencies(
                    spawner,
                    clock,
                    voice_ready=voice_ready,
                )
                dependencies.signal_api = signals
                session = self.session(dependencies)

                report = session.run().canonical()

                self.assertEqual(
                    report["reason_code"],
                    "keyboard_interrupt" if interruption == "keyboard" else "signal",
                )
                self.assertEqual(
                    [event for event in spawner.events if event.startswith("terminate:")],
                    [
                        "terminate:voice",
                        "terminate:orchestrator",
                        "terminate:inference",
                    ],
                )
                self.assertEqual(signals.handlers[signals.SIGINT], "original-int")
                self.assertEqual(signals.handlers[signals.SIGTERM], "original-term")

    def test_occupied_required_port_refuses_to_spawn(self) -> None:
        spawner = FakeSpawner()
        clock = FakeClock()
        session = self.session(
            self.dependencies(
                spawner,
                clock,
                port_available=lambda port: port != 8766,
            )
        )

        with self.assertRaises(MacOSBetaSessionError) as caught:
            session.run()

        self.assertEqual(caught.exception.code, "required_port_occupied")
        self.assertEqual(spawner.events, [])
        self.assertIsNone(session.session_directory)

    def test_runtime_child_crash_stops_remaining_processes(self) -> None:
        spawner = FakeSpawner()
        clock = FakeClock()

        def crash_orchestrator() -> None:
            if set(spawner.processes) == {"inference", "orchestrator", "voice"}:
                spawner.processes["orchestrator"].returncode = 7
                clock.on_sleep = None

        clock.on_sleep = crash_orchestrator
        session = self.session(self.dependencies(spawner, clock))

        with self.assertRaises(MacOSBetaSessionError) as caught:
            session.run()

        self.assertEqual(caught.exception.code, "orchestrator_runtime_failed")
        self.assertEqual(
            [event for event in spawner.events if event.startswith("terminate:")],
            ["terminate:voice", "terminate:orchestrator", "terminate:inference"],
        )
        self.assertIsNotNone(spawner.processes["orchestrator"].poll())
        self.assertIsNotNone(spawner.processes["voice"].poll())
        self.assertIsNotNone(spawner.processes["inference"].poll())

    def test_shutdown_incomplete_dominates_primary_runtime_failure(self) -> None:
        spawner = FakeSpawner(ignore_term=True, ignore_kill=True)
        clock = FakeClock()

        def crash_orchestrator() -> None:
            if set(spawner.processes) == {"inference", "orchestrator", "voice"}:
                spawner.processes["orchestrator"].returncode = 7
                clock.on_sleep = None

        clock.on_sleep = crash_orchestrator
        session = self.session(self.dependencies(spawner, clock))

        with self.assertRaises(MacOSBetaSessionError) as caught:
            session.run()

        self.assertEqual(caught.exception.code, "session_shutdown_incomplete")
        self.assertEqual(str(caught.exception), "session_shutdown_incomplete")
        self.assertEqual(
            [event for event in spawner.events if event.startswith("kill:")],
            ["kill:voice", "kill:inference"],
        )
        interlock = self.sessions / SESSION_INTERLOCK_FILENAME
        self.assertTrue(interlock.exists())
        self.assertEqual(interlock.stat().st_mode & 0o777, 0o600)

        second_spawner = FakeSpawner()
        second = self.session(
            self.dependencies(second_spawner, FakeClock())
        )
        with self.assertRaises(MacOSBetaSessionError) as relaunch:
            second.run()
        self.assertEqual(relaunch.exception.code, "session_relaunch_blocked")
        self.assertEqual(second_spawner.events, [])

    def test_shutdown_ignores_time_wait_when_no_listener_remains(self) -> None:
        spawner = FakeSpawner()
        clock = FakeClock()
        current: MacOSBetaSession | None = None
        bind_probes: list[int] = []

        def port_available(port: int) -> bool:
            bind_probes.append(port)
            return True

        def voice_ready(_context: object) -> bool:
            assert current is not None
            current.request_stop()
            return True

        current = self.session(
            self.dependencies(
                spawner,
                clock,
                voice_ready=voice_ready,
                port_available=port_available,
                listener_pids=lambda _port: frozenset(),
            )
        )

        report = current.run().canonical()

        self.assertTrue(report["shutdown_complete"])
        self.assertEqual(bind_probes, [8765, 8766, 8767])
        self.assertFalse((self.sessions / SESSION_INTERLOCK_FILENAME).exists())

    def _leave_incomplete_interlock(
        self,
    ) -> tuple[FakeSpawner, SessionDependencies]:
        spawner = FakeSpawner(ignore_term=True, ignore_kill=True)
        clock = FakeClock()
        current: MacOSBetaSession | None = None

        def stop_after_voice(_context: object) -> bool:
            assert current is not None
            current.request_stop()
            return True

        dependencies = self.dependencies(
            spawner,
            clock,
            voice_ready=stop_after_voice,
        )
        current = self.session(dependencies)
        with self.assertRaises(MacOSBetaSessionError) as caught:
            current.run()
        self.assertEqual(caught.exception.code, "session_shutdown_incomplete")
        return spawner, dependencies

    def test_explicit_recovery_clears_only_after_absence_is_proven(self) -> None:
        self._leave_incomplete_interlock()
        recovery_dependencies = self.dependencies(FakeSpawner(), FakeClock())
        recovery_dependencies.process_identity = lambda _pid: None
        recovery_dependencies.process_group = lambda _pgid: ()

        report = recover_macos_beta_session(
            stack_path=self.stack,
            session_parent=self.sessions,
            dependencies=recovery_dependencies,
        ).canonical()

        self.assertEqual(
            report,
            {
                "interlock_present": True,
                "recovered": True,
                "safe_to_relaunch": True,
                "version": "jarvis-macos-beta-recovery/1",
            },
        )
        self.assertFalse((self.sessions / SESSION_INTERLOCK_FILENAME).exists())

    def test_recovery_uses_listener_absence_not_bind_availability(self) -> None:
        self._leave_incomplete_interlock()
        recovery = self.dependencies(
            FakeSpawner(),
            FakeClock(),
            port_available=lambda _port: False,
            listener_pids=lambda _port: frozenset(),
        )
        recovery.process_identity = lambda _pid: None
        recovery.process_group = lambda _pgid: ()

        report = recover_macos_beta_session(
            stack_path=self.stack,
            session_parent=self.sessions,
            dependencies=recovery,
        ).canonical()

        self.assertTrue(report["safe_to_relaunch"])
        self.assertFalse((self.sessions / SESSION_INTERLOCK_FILENAME).exists())

    def test_recovery_refuses_an_exact_live_process_identity(self) -> None:
        spawner, original = self._leave_incomplete_interlock()
        inspect_original = original.process_identity

        def without_supervisor(pid: int) -> _ProcessIdentity | None:
            if pid == os.getpid():
                return None
            return inspect_original(pid)

        original.process_identity = without_supervisor
        with self.assertRaises(MacOSBetaSessionError) as caught:
            recover_macos_beta_session(
                stack_path=self.stack,
                session_parent=self.sessions,
                dependencies=original,
            )

        self.assertEqual(caught.exception.code, "session_recovery_unsafe")
        self.assertTrue(spawner.processes)
        self.assertTrue((self.sessions / SESSION_INTERLOCK_FILENAME).exists())

    def test_recovery_does_not_trust_a_reused_pid_without_start_token(self) -> None:
        _spawner, original = self._leave_incomplete_interlock()
        document = json.loads(
            (self.sessions / SESSION_INTERLOCK_FILENAME).read_text(encoding="utf-8")
        )
        leader = document["components"]["inference"]
        reused_pid = leader["leader_pid"]
        reused = _ProcessIdentity(
            reused_pid,
            reused_pid,
            os.getuid(),
            2,
            hashlib.sha256(b"different-start").hexdigest(),
        )

        def inspect(pid: int) -> _ProcessIdentity | None:
            if pid == reused_pid:
                return reused
            return None

        original.process_identity = inspect
        original.process_group = (
            lambda pgid: (
                _ProcessGroupMember(
                    reused_pid,
                    pgid,
                    os.getuid(),
                    2,
                    reused.start_token,
                ),
            )
            if pgid == reused_pid
            else ()
        )

        with self.assertRaises(MacOSBetaSessionError) as caught:
            recover_macos_beta_session(
                stack_path=self.stack,
                session_parent=self.sessions,
                dependencies=original,
            )

        self.assertEqual(caught.exception.code, "session_recovery_unsafe")
        self.assertTrue((self.sessions / SESSION_INTERLOCK_FILENAME).exists())

    def test_recovery_refuses_an_exposed_or_malformed_interlock(self) -> None:
        self._leave_incomplete_interlock()
        path = self.sessions / SESSION_INTERLOCK_FILENAME
        path.chmod(0o644)
        recovery = self.dependencies(FakeSpawner(), FakeClock())
        recovery.process_identity = lambda _pid: None
        recovery.process_group = lambda _pgid: ()

        with self.assertRaises(MacOSBetaSessionError) as caught:
            recover_macos_beta_session(
                stack_path=self.stack,
                session_parent=self.sessions,
                dependencies=recovery,
            )

        self.assertEqual(caught.exception.code, "session_recovery_unsafe")
        self.assertTrue(path.exists())

    def test_recovery_refuses_a_remaining_listener(self) -> None:
        self._leave_incomplete_interlock()
        recovery = self.dependencies(
            FakeSpawner(),
            FakeClock(),
            listener_pids=lambda port: (
                frozenset({9999}) if port == 8767 else frozenset()
            ),
        )
        recovery.process_identity = lambda _pid: None
        recovery.process_group = lambda _pgid: ()

        with self.assertRaises(MacOSBetaSessionError) as caught:
            recover_macos_beta_session(
                stack_path=self.stack,
                session_parent=self.sessions,
                dependencies=recovery,
            )

        self.assertEqual(caught.exception.code, "session_recovery_unsafe")
        self.assertTrue((self.sessions / SESSION_INTERLOCK_FILENAME).exists())

    def test_beta_recover_cli_is_an_explicit_content_free_operation(self) -> None:
        output = io.StringIO()
        with (
            patch(
                "jarvis_orchestrator.macos_beta_session.recover_macos_beta_session",
                return_value=MacOSBetaRecoveryReport(
                    interlock_present=True,
                    recovered=True,
                ),
            ) as recover,
            redirect_stdout(output),
        ):
            exit_code = stack_main(
                [
                    "--config",
                    str(self.stack),
                    "beta-recover",
                    "--session-parent",
                    str(self.sessions),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(recover.call_count, 1)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "interlock_present": True,
                "recovered": True,
                "safe_to_relaunch": True,
                "version": "jarvis-macos-beta-recovery/1",
            },
        )

    def test_live_lease_activation_failure_rolls_back_all_children(self) -> None:
        spawner = FakeSpawner()
        clock = FakeClock()

        def fail_lease(_stack, _children, _commands):
            raise RuntimeError("private detail")

        session = self.session(
            self.dependencies(
                spawner,
                clock,
                activate_live_lease=fail_lease,
            )
        )

        with self.assertRaises(MacOSBetaSessionError) as caught:
            session.run()

        self.assertEqual(caught.exception.code, "live_lease_activation_failed")
        self.assertEqual(
            [event for event in spawner.events if event.startswith("terminate:")],
            ["terminate:voice", "terminate:orchestrator", "terminate:inference"],
        )

    def test_live_lease_failure_is_fail_closed_and_removed(self) -> None:
        spawner = FakeSpawner()
        clock = FakeClock()
        lease = FakeLease(heartbeat_fails=True)
        session = self.session(
            self.dependencies(
                spawner,
                clock,
                activate_live_lease=lambda _stack, _children, _commands: lease,
            )
        )

        with self.assertRaises(MacOSBetaSessionError) as caught:
            session.run()

        self.assertEqual(caught.exception.code, "live_lease_runtime_failed")
        self.assertTrue(lease.closed)
        self.assertEqual(
            [event for event in spawner.events if event.startswith("terminate:")],
            ["terminate:voice", "terminate:orchestrator", "terminate:inference"],
        )

    def test_non_macos_host_fails_before_validation_or_spawn(self) -> None:
        spawner = FakeSpawner()
        clock = FakeClock()
        dependencies = self.dependencies(spawner, clock)
        dependencies.platform = "linux"
        session = self.session(dependencies)

        with self.assertRaises(MacOSBetaSessionError) as caught:
            session.run()

        self.assertEqual(caught.exception.code, "macos_required")
        self.assertEqual(spawner.events, [])

    def test_legacy_inference_fallback_requires_explicit_missing_route(self) -> None:
        process = FakeProcess("inference", [])
        stack = PrivateStackConfig.load(self.stack)
        config = OrchestratorConfig.load(self.orchestrator)
        context = ProbeContext("inference", stack, config, process, 1.0)
        with patch(
            "jarvis_orchestrator.macos_beta_session.InferenceControlClient"
        ) as client:
            client.return_value.capabilities.return_value = SimpleNamespace(
                text=True, stream=True
            )
            client.return_value.runtime_identity.side_effect = OrchestratorError(
                "service_unavailable", status=503, retryable=True
            )
            self.assertFalse(_inference_ready(context))

            client.return_value.runtime_identity.side_effect = (
                InferenceIdentityEndpointUnavailable()
            )
            self.assertTrue(_inference_ready(context))

    def test_voice_liveness_requires_exact_beta_session_v2(self) -> None:
        process = FakeProcess("voice", [])
        stack = PrivateStackConfig.load(self.stack)
        config = OrchestratorConfig.load(self.orchestrator)
        context = ProbeContext("voice", stack, config, process, 1.0)
        valid = {
            "core_ready": True,
            "effective_capabilities": {
                "cloud": False,
                "memory_retrospective": False,
                "memory_shadow": False,
                "research": False,
                "tools": False,
            },
            "schema_version": "jarvis-orchestrator-beta-session/2",
            "voice_client_connected": True,
            "voice_connection_generation": 1,
            "voice_connection_tracker_id": (
                "00000000-0000-4000-8000-000000000001"
            ),
        }
        with patch(
            "jarvis_orchestrator.macos_beta_session.urllib.request.urlopen",
            return_value=FakeHttpResponse(canonical(valid)),
        ):
            self.assertTrue(_voice_liveness_ready(context))

        invalid_documents = (
            {**valid, "schema_version": "jarvis-orchestrator-beta-session/1"},
            {
                key: value
                for key, value in valid.items()
                if key != "voice_connection_generation"
            },
            {**valid, "voice_connection_tracker_id": "not-a-uuid"},
            {**valid, "core_ready": False},
        )
        for document in invalid_documents:
            with self.subTest(document=document):
                with patch(
                    "jarvis_orchestrator.macos_beta_session.urllib.request.urlopen",
                    return_value=FakeHttpResponse(canonical(document)),
                ):
                    self.assertFalse(_voice_liveness_ready(context))


if __name__ == "__main__":
    unittest.main()
