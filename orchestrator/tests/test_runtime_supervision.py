from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator.background import BackgroundServiceManager
from jarvis_orchestrator.control import ControlApplication
from jarvis_orchestrator.domain import OrchestratorError
from jarvis_orchestrator.gateway import VoiceGateway
from jarvis_orchestrator.operations import OperationalError
from jarvis_orchestrator.private_state import (
    create_or_replace_private_file,
    ensure_private_directory,
    unlink_private_file,
)
from jarvis_orchestrator.runtime_supervisor import (
    RUNTIME_HARD_EXIT_STATUS,
    RuntimeFailure,
    RuntimeProcessDeadline,
    RuntimeShutdownBudget,
    RuntimeSignalGuard,
    RuntimeSupervisor,
    SupervisedControlRuntime,
)
from jarvis_orchestrator.scheduler_worker import SchedulerWorker

from jarvis_voice.voice_auth import VoiceAuthKey

AUTH_KEY = VoiceAuthKey(1, b"k" * 32)


class _FailingSchedulerStore:
    def expire_approvals(self, **_kwargs: object) -> int:
        raise RuntimeError("private scheduler failure payload")


class _ReadySchedulerStore:
    def expire_approvals(self, **_kwargs: object) -> int:
        return 0

    def retry_key_blocked_deliveries(self, **_kwargs: object) -> int:
        return 0

    def fire_due(self, **_kwargs: object) -> list[object]:
        return []

    def expire_deliveries(self, **_kwargs: object) -> int:
        return 0

    def purge_content(self, **_kwargs: object) -> int:
        return 0


class _FakeServer:
    def __init__(self) -> None:
        self.closed = asyncio.Event()

    def close(self) -> None:
        self.closed.set()

    async def wait_closed(self) -> None:
        await self.closed.wait()


class _CancellationResistantServer:
    def __init__(self) -> None:
        self.release = asyncio.Event()

    def close(self) -> None:
        return

    async def wait_closed(self) -> None:
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                continue


class _SweepService:
    def __init__(self, *, fail: bool) -> None:
        self.config = SimpleNamespace(voice_host="127.0.0.1", voice_port=0)
        self.execution_scheduler: object | None = None
        self.fail = fail

    def set_execution_scheduler(self, scheduler: object | None) -> None:
        self.execution_scheduler = scheduler

    def sweep(self) -> None:
        if self.fail:
            raise RuntimeError("private sweep failure payload")


class _FakeHttpd:
    daemon_threads = True
    block_on_close = False

    def __init__(self) -> None:
        self.closed = False

    def server_close(self) -> None:
        self.closed = True


class _DrainingControlServer:
    def __init__(self) -> None:
        self.httpd = _FakeHttpd()
        self.handler_started = threading.Event()
        self.handler_release = threading.Event()
        self.handler_done = threading.Event()
        self.shutdown_requested = threading.Event()
        self.handler: threading.Thread | None = None
        self.events: list[str] = []

    def serve_forever(self) -> None:
        def handle() -> None:
            self.handler_started.set()
            self.handler_release.wait()
            self.events.append("handler_done")
            self.handler_done.set()

        self.handler = threading.Thread(target=handle, name="test-control-handler")
        self.handler.start()
        self.shutdown_requested.wait()

    def shutdown(self) -> None:
        self.shutdown_requested.set()
        assert self.handler is not None
        self.handler.join()


class _FailingShutdownControlServer:
    def __init__(self) -> None:
        self.httpd = _FakeHttpd()
        self.release = threading.Event()

    def serve_forever(self) -> None:
        self.release.wait()

    def shutdown(self) -> None:
        self.release.set()
        raise RuntimeError("private shutdown failure payload")


class RuntimeSupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_stays_closed_until_all_acknowledgements_and_seal(
        self,
    ) -> None:
        supervisor = RuntimeSupervisor()
        release = asyncio.Event()

        async def worker() -> None:
            supervisor.acknowledge_ready("barrier_worker")
            await release.wait()

        supervisor.start_task(
            "barrier_worker",
            worker,
            failure_code="runtime_barrier_worker_failed",
        )
        self.assertFalse(supervisor.health()["ready"])
        self.assertTrue(await supervisor.finalize_readiness(asyncio.Event()))
        self.assertTrue(supervisor.health()["ready"])
        with self.assertRaisesRegex(
            RuntimeError, "runtime_supervisor_registration_sealed"
        ):
            supervisor.start_task(
                "late_worker",
                worker,
                failure_code="runtime_late_worker_failed",
            )
        release.set()
        supervisor.begin_shutdown()
        await supervisor.shutdown()

    async def test_controlled_stop_before_ack_never_opens_admission(self) -> None:
        supervisor = RuntimeSupervisor()

        async def worker() -> None:
            await asyncio.Event().wait()

        supervisor.start_task(
            "blocked_worker",
            worker,
            failure_code="runtime_blocked_worker_failed",
        )
        stop_requested = asyncio.Event()
        stop_requested.set()
        self.assertFalse(await supervisor.finalize_readiness(stop_requested))
        self.assertFalse(supervisor.health()["ready"])
        await supervisor.shutdown()

    async def test_missing_ack_fails_closed_with_stable_barrier_timeout(self) -> None:
        supervisor = RuntimeSupervisor(shutdown_timeout_seconds=0.05)

        async def worker() -> None:
            await asyncio.Event().wait()

        supervisor.start_task(
            "silent_worker",
            worker,
            failure_code="runtime_silent_worker_failed",
        )
        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.finalize_readiness(asyncio.Event())
        self.assertEqual(observed.exception.code, "runtime_startup_barrier_timeout")
        self.assertFalse(supervisor.health()["ready"])
        await supervisor.shutdown()

    async def test_critical_exception_uses_stable_content_free_failure(self) -> None:
        supervisor = RuntimeSupervisor()

        async def fail() -> None:
            raise RuntimeError("secret provider response")

        supervisor.start_task(
            "example_worker",
            fail,
            failure_code="runtime_example_failed",
        )
        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.wait(asyncio.Event())
        self.assertEqual(observed.exception.code, "runtime_example_failed")
        health = supervisor.health()
        self.assertEqual(health["state"], "FAILED")
        self.assertNotIn("secret", repr(health))
        await supervisor.shutdown()

    async def test_unexpected_clean_component_exit_is_a_failure(self) -> None:
        supervisor = RuntimeSupervisor()

        async def return_early() -> None:
            return

        supervisor.start_task(
            "early_exit",
            return_early,
            failure_code="runtime_early_exit_failed",
        )
        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.wait(asyncio.Event())
        self.assertEqual(observed.exception.code, "runtime_early_exit_failed")
        await supervisor.shutdown()

    async def test_first_tick_failure_is_detected_before_readiness_boundary(
        self,
    ) -> None:
        supervisor = RuntimeSupervisor()

        async def fail_during_first_tick() -> None:
            raise RuntimeError("private startup failure payload")

        supervisor.start_task(
            "startup_worker",
            fail_during_first_tick,
            failure_code="runtime_startup_worker_failed",
        )
        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.first_tick()
        self.assertEqual(
            observed.exception.code,
            "runtime_startup_worker_failed",
        )
        await supervisor.shutdown()

    async def test_failed_runtime_forces_control_readiness_to_503(self) -> None:
        supervisor = RuntimeSupervisor()

        async def fail() -> None:
            raise RuntimeError("private component payload")

        supervisor.start_task(
            "readiness_worker",
            fail,
            failure_code="runtime_readiness_worker_failed",
        )
        with self.assertRaises(RuntimeFailure):
            await supervisor.first_tick()
        service = SimpleNamespace(
            readiness_snapshot=lambda: {
                "core_ready": True,
                "inference_ready": True,
                "memory_ready": True,
                "memory_requirement_satisfied": True,
                "memory_retrospective_ready": False,
                "memory_state": "READY",
                "storage_ready": True,
            },
            scheduler=None,
        )
        application = ControlApplication(
            SimpleNamespace(),  # type: ignore[arg-type]
            service,  # type: ignore[arg-type]
            runtime_health=supervisor.health,
        )

        self.assertEqual(application.get("/readyz"), (503, {"ready": False}))
        status, readiness = application.get("/v1/readiness")
        self.assertEqual(status, 200)
        self.assertFalse(readiness["core_ready"])
        self.assertFalse(readiness["runtime_ready"])
        self.assertEqual(readiness["runtime_state"], "FAILED")
        self.assertEqual(
            readiness["runtime_failure_code"],
            "runtime_readiness_worker_failed",
        )
        await supervisor.shutdown()

    async def test_control_readiness_fails_closed_without_runtime_health(self) -> None:
        service = SimpleNamespace(
            readiness_snapshot=lambda: {
                "core_ready": True,
                "inference_ready": True,
                "memory_ready": True,
                "memory_requirement_satisfied": True,
                "memory_retrospective_ready": False,
                "memory_state": "READY",
                "storage_ready": True,
            },
            scheduler=None,
        )
        application = ControlApplication(
            SimpleNamespace(),  # type: ignore[arg-type]
            service,  # type: ignore[arg-type]
        )

        self.assertEqual(application.get("/readyz"), (503, {"ready": False}))
        _, readiness = application.get("/v1/readiness")
        self.assertFalse(readiness["core_ready"])
        self.assertFalse(readiness["runtime_ready"])
        self.assertEqual(readiness["runtime_state"], "UNSUPERVISED")
        self.assertEqual(
            readiness["runtime_failure_code"],
            "runtime_health_unavailable",
        )

    async def test_memory_failure_is_not_reported_as_inference_failure(self) -> None:
        calls = 0

        def snapshot() -> dict[str, bool | str]:
            nonlocal calls
            calls += 1
            return {
                "core_ready": False,
                "inference_ready": True,
                "memory_ready": False,
                "memory_requirement_satisfied": False,
                "memory_retrospective_ready": False,
                "memory_state": "BLOCKED",
                "storage_ready": True,
            }

        application = ControlApplication(
            SimpleNamespace(),  # type: ignore[arg-type]
            SimpleNamespace(  # type: ignore[arg-type]
                readiness_snapshot=snapshot,
                scheduler=None,
            ),
            runtime_health=lambda: (True, "READY", None),
        )

        status, readiness = application.get("/v1/readiness")
        self.assertEqual(status, 200)
        self.assertEqual(calls, 1)
        self.assertFalse(readiness["core_ready"])
        self.assertFalse(readiness["memory_ready"])
        self.assertEqual(readiness["memory_state"], "BLOCKED")
        self.assertTrue(readiness["inference_ready"])
        self.assertTrue(readiness["storage_ready"])

    async def test_control_business_admission_returns_retryable_503_prebarrier(
        self,
    ) -> None:
        application = ControlApplication(
            SimpleNamespace(),  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            business_admission=lambda: False,
        )
        application.require_business_admission("/readyz")
        application.require_business_admission("/v1/readiness")
        with self.assertRaises(OrchestratorError) as observed:
            application.require_business_admission("/v1/conversations")
        self.assertEqual(observed.exception.code, "runtime_not_ready")
        self.assertEqual(observed.exception.status, 503)
        self.assertTrue(observed.exception.retryable)

    async def test_controlled_shutdown_cancels_and_reaps_every_owned_task(self) -> None:
        supervisor = RuntimeSupervisor()
        running = asyncio.Event()
        stopped = asyncio.Event()

        async def worker() -> None:
            running.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        supervisor.start_task(
            "long_running",
            worker,
            failure_code="runtime_long_running_failed",
        )
        await running.wait()
        controlled_stop = asyncio.Event()
        controlled_stop.set()
        await supervisor.wait(controlled_stop)
        await supervisor.shutdown()
        await stopped.wait()
        self.assertTrue(all(task.done() for task in supervisor.tasks.values()))
        self.assertEqual(supervisor.health()["state"], "STOPPED")

    async def test_shutdown_timeout_is_bounded_and_stable(self) -> None:
        running = asyncio.Event()
        release = asyncio.Event()
        exit_statuses: list[int] = []

        def terminate(status: int) -> None:
            exit_statuses.append(status)
            release.set()

        supervisor = RuntimeSupervisor(
            shutdown_timeout_seconds=0.05,
            process_terminator=terminate,
        )

        async def stubborn_worker() -> None:
            running.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

        supervisor.start_task(
            "stubborn_worker",
            stubborn_worker,
            failure_code="runtime_stubborn_failed",
        )
        await running.wait()
        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.shutdown()
        self.assertEqual(observed.exception.code, "runtime_shutdown_timeout")
        self.assertEqual(
            supervisor.health()["failure_code"], "runtime_shutdown_timeout"
        )
        self.assertEqual(exit_statuses, [RUNTIME_HARD_EXIT_STATUS])
        self.assertTrue(all(task.done() for task in supervisor.tasks.values()))

    async def test_cancellation_resistant_cleanup_cannot_reach_finalizer(self) -> None:
        release = asyncio.Event()
        exit_statuses: list[int] = []

        def terminate(status: int) -> None:
            exit_statuses.append(status)
            release.set()

        supervisor = RuntimeSupervisor(
            shutdown_timeout_seconds=0.05,
            process_terminator=terminate,
        )

        async def stubborn_cleanup() -> None:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.run_bounded_cleanup(
                "test_cleanup",
                stubborn_cleanup,
                failure_code="runtime_test_cleanup_failed",
                timeout_code="runtime_test_cleanup_timeout",
            )
        self.assertEqual(observed.exception.code, "runtime_test_cleanup_timeout")
        self.assertEqual(exit_statuses, [RUNTIME_HARD_EXIT_STATUS])
        self.assertFalse(
            any(
                task.get_name() == "jarvis-runtime-test_cleanup-cleanup"
                for task in asyncio.all_tasks()
                if task is not asyncio.current_task()
            )
        )

    async def test_total_shutdown_budget_is_cumulative_across_slow_stages(
        self,
    ) -> None:
        exit_statuses: list[int] = []
        supervisor = RuntimeSupervisor(
            shutdown_timeout_seconds=1.0,
            process_terminator=exit_statuses.append,
        )
        budget = RuntimeShutdownBudget(0.10)
        loop = asyncio.get_running_loop()
        started = loop.time()

        await supervisor.run_bounded_cleanup(
            "first_slow_stage",
            lambda: asyncio.sleep(0.06),
            failure_code="runtime_first_slow_stage_failed",
            timeout_code="runtime_first_slow_stage_timeout",
            timeout_seconds=budget.remaining(),
        )
        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.run_bounded_cleanup(
                "second_slow_stage",
                lambda: asyncio.sleep(0.08),
                failure_code="runtime_second_slow_stage_failed",
                timeout_code="runtime_second_slow_stage_timeout",
                timeout_seconds=budget.remaining(),
            )

        self.assertEqual(
            observed.exception.code,
            "runtime_second_slow_stage_timeout",
        )
        self.assertEqual(exit_statuses, [RUNTIME_HARD_EXIT_STATUS])
        self.assertLess(loop.time() - started, 0.25)
        self.assertFalse(
            any(
                task.get_name().endswith("_slow_stage-cleanup")
                for task in asyncio.all_tasks()
                if task is not asyncio.current_task()
            )
        )

    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "POSIX signal required")
    async def test_signal_guard_restores_custom_handler_and_absorbs_second_signal(
        self,
    ) -> None:
        selected = signal.SIGUSR1
        original = signal.getsignal(selected)
        prior_calls: list[int] = []

        def prior_handler(number: int, _frame: object) -> None:
            prior_calls.append(number)

        signal.signal(selected, prior_handler)
        stop_requested = asyncio.Event()
        guard = RuntimeSignalGuard(
            asyncio.get_running_loop(),
            stop_requested,
            selected_signals=(selected,),
        )
        try:
            guard.install()
            signal.raise_signal(selected)
            await asyncio.sleep(0.02)
            self.assertTrue(stop_requested.is_set())
            guard.begin_cleanup()
            signal.raise_signal(selected)
            await asyncio.sleep(0.02)
            self.assertEqual(guard.request_count, 2)
            self.assertEqual(prior_calls, [])
            guard.restore()
            self.assertIs(signal.getsignal(selected), prior_handler)
            signal.raise_signal(selected)
            self.assertEqual(prior_calls, [selected])
        finally:
            guard.restore()
            signal.signal(selected, original)

    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "POSIX signal required")
    async def test_signal_guard_fails_closed_without_replacing_loop_handler(
        self,
    ) -> None:
        selected = signal.SIGUSR1
        original = signal.getsignal(selected)
        loop = asyncio.get_running_loop()
        prior_called = asyncio.Event()
        loop.add_signal_handler(selected, prior_called.set)
        guard = RuntimeSignalGuard(
            loop,
            asyncio.Event(),
            selected_signals=(selected,),
        )
        try:
            with self.assertRaisesRegex(
                RuntimeError, "runtime_signal_handler_conflict"
            ):
                guard.install()
            signal.raise_signal(selected)
            await asyncio.wait_for(prior_called.wait(), timeout=0.25)
        finally:
            guard.restore()
            loop.remove_signal_handler(selected)
            signal.signal(selected, original)

    async def test_external_process_deadline_can_be_disarmed(self) -> None:
        exit_statuses: list[int] = []
        deadline = RuntimeProcessDeadline(
            0.05,
            process_terminator=exit_statuses.append,
        )
        deadline.begin()
        deadline.disarm()
        await asyncio.sleep(0.08)
        self.assertEqual(exit_statuses, [])
        self.assertFalse(deadline.fired)

    def test_external_process_deadline_bounds_synchronous_event_loop_stall(
        self,
    ) -> None:
        script = (
            "import time\n"
            "from jarvis_orchestrator.runtime_supervisor import "
            "RuntimeProcessDeadline\n"
            "deadline = RuntimeProcessDeadline(0.10)\n"
            "deadline.begin()\n"
            "time.sleep(5)\n"
        )
        started = time.monotonic()
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            timeout=2.0,
        )
        self.assertEqual(completed.returncode, RUNTIME_HARD_EXIT_STATUS)
        self.assertLess(time.monotonic() - started, 1.5)

    async def test_control_handlers_drain_before_store_close_boundary(self) -> None:
        supervisor = RuntimeSupervisor()
        server = _DrainingControlServer()
        runtime = SupervisedControlRuntime(
            server,
            supervisor,
            shutdown_timeout_seconds=0.25,
        )
        runtime.start()
        while not server.handler_started.is_set():
            await asyncio.sleep(0.01)
        self.assertTrue(await supervisor.finalize_readiness(asyncio.Event()))
        supervisor.begin_shutdown()

        async def release_handler() -> None:
            await asyncio.sleep(0.02)
            server.handler_release.set()

        release_task = asyncio.create_task(release_handler())
        await runtime.stop()
        await release_task
        server.events.append("store_close")
        await supervisor.shutdown()
        self.assertEqual(server.events, ["handler_done", "store_close"])
        self.assertFalse(server.httpd.daemon_threads)
        self.assertTrue(server.httpd.block_on_close)
        self.assertTrue(server.httpd.closed)

    async def test_control_registration_failure_cannot_leak_server_thread(
        self,
    ) -> None:
        supervisor = RuntimeSupervisor()
        supervisor.begin_shutdown()
        server = _DrainingControlServer()
        runtime = SupervisedControlRuntime(
            server,
            supervisor,
            shutdown_timeout_seconds=0.25,
        )

        with self.assertRaisesRegex(RuntimeError, "runtime_supervisor_stopping"):
            runtime.start()

        self.assertIsNone(runtime._thread)
        self.assertTrue(server.httpd.closed)
        self.assertFalse(server.handler_started.is_set())
        await supervisor.shutdown()

    async def test_control_drain_timeout_uses_hard_exit_seam_then_joins(self) -> None:
        server = _DrainingControlServer()
        exit_statuses: list[int] = []

        def terminate(status: int) -> None:
            exit_statuses.append(status)
            server.handler_release.set()

        supervisor = RuntimeSupervisor(
            shutdown_timeout_seconds=0.05,
            process_terminator=terminate,
        )
        runtime = SupervisedControlRuntime(
            server,
            supervisor,
            shutdown_timeout_seconds=0.05,
        )
        runtime.start()
        while not server.handler_started.is_set():
            await asyncio.sleep(0.01)
        supervisor.begin_shutdown()
        with self.assertRaises(RuntimeFailure) as observed:
            await runtime.stop()
        self.assertEqual(
            observed.exception.code,
            "runtime_control_shutdown_timeout",
        )
        self.assertEqual(exit_statuses, [RUNTIME_HARD_EXIT_STATUS])
        self.assertTrue(server.handler_done.is_set())
        await supervisor.shutdown()

    async def test_control_shutdown_failure_uses_safe_hard_exit_seam(self) -> None:
        exit_statuses: list[int] = []
        supervisor = RuntimeSupervisor(
            shutdown_timeout_seconds=0.10,
            process_terminator=exit_statuses.append,
        )
        runtime = SupervisedControlRuntime(
            _FailingShutdownControlServer(),
            supervisor,
            shutdown_timeout_seconds=0.10,
        )
        runtime.start()
        supervisor.begin_shutdown()

        with self.assertRaises(RuntimeFailure) as observed:
            await runtime.stop()

        self.assertEqual(
            observed.exception.code,
            "runtime_control_shutdown_failed",
        )
        self.assertEqual(exit_statuses, [RUNTIME_HARD_EXIT_STATUS])
        self.assertIsNone(runtime._thread)
        await supervisor.shutdown()

    async def test_private_state_error_is_not_reported_as_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            manager = BackgroundServiceManager(
                config_path=config,
                database_path=root / "state.sqlite3",
            )
            with (
                patch(
                    "jarvis_orchestrator.background.ProcessLock.acquire",
                    side_effect=OperationalError("private_state_file_owner_invalid"),
                ),
                self.assertRaisesRegex(
                    OperationalError,
                    "private_state_file_owner_invalid",
                ),
            ):
                manager.status()
            with patch(
                "jarvis_orchestrator.background.ProcessLock.acquire",
                side_effect=OperationalError("service_already_running"),
            ):
                self.assertTrue(manager.status()["running"])

    async def test_stop_arriving_during_background_install_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            manager = BackgroundServiceManager(
                config_path=config,
                database_path=root / "state.sqlite3",
            )
            ensure_private_directory(manager.paths["state"], create=True)

            def stop_during_install() -> None:
                create_or_replace_private_file(manager.paths["stop"], b"stop\n")

            with (
                patch.object(
                    manager,
                    "_write_runner",
                    side_effect=stop_during_install,
                ),
                patch.object(manager, "_write_macos_plist"),
                patch.object(
                    manager,
                    "_run",
                    return_value=True,
                ),
                patch.object(
                    manager,
                    "status",
                    return_value={"running": False},
                ),
                patch(
                    "jarvis_orchestrator.background.sys.platform",
                    "darwin",
                ),
            ):
                manager.install()
            self.assertTrue(manager.paths["stop"].is_file())

    async def test_background_stop_rejects_transient_unlocked_restart_gap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            manager = BackgroundServiceManager(
                config_path=config,
                database_path=root / "state.sqlite3",
            )
            ensure_private_directory(manager.paths["state"], create=True)

            def complete_cleanup(_seconds: float) -> None:
                unlink_private_file(manager.paths["stop"], missing_ok=False)

            with (
                patch.object(manager, "_is_running", return_value=False) as running,
                patch.object(
                    manager,
                    "status",
                    return_value={"running": False},
                ),
                patch(
                    "jarvis_orchestrator.background.time.sleep",
                    side_effect=complete_cleanup,
                ) as sleep,
            ):
                self.assertEqual(
                    manager.stop(timeout_seconds=0.25),
                    {"running": False},
                )
            self.assertEqual(running.call_count, 2)
            sleep.assert_called_once()

    async def test_scheduler_crash_reaches_process_supervisor(self) -> None:
        supervisor = RuntimeSupervisor()
        worker = SchedulerWorker(_FailingSchedulerStore())  # type: ignore[arg-type]
        worker.start(supervisor=supervisor)
        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.wait(asyncio.Event())
        self.assertEqual(observed.exception.code, "runtime_scheduler_failed")
        supervisor.begin_shutdown()
        await worker.stop()
        await supervisor.shutdown()

    async def test_scheduler_acknowledges_only_after_first_successful_tick(
        self,
    ) -> None:
        supervisor = RuntimeSupervisor()
        worker = SchedulerWorker(_ReadySchedulerStore())  # type: ignore[arg-type]
        worker.start(supervisor=supervisor)
        self.assertFalse(supervisor.health()["ready"])
        self.assertTrue(await supervisor.finalize_readiness(asyncio.Event()))
        self.assertTrue(supervisor.health()["ready"])
        supervisor.begin_shutdown()
        await worker.stop()
        await supervisor.shutdown()

    async def test_gateway_sweeper_crash_reaches_process_supervisor(self) -> None:
        supervisor = RuntimeSupervisor()
        service = _SweepService(fail=True)
        gateway = VoiceGateway(service, auth_key=AUTH_KEY)  # type: ignore[arg-type]
        server = _FakeServer()
        with patch(
            "websockets.asyncio.server.serve",
            new=AsyncMock(return_value=server),
        ):
            await gateway.start(supervisor=supervisor)
        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.wait(asyncio.Event())
        self.assertEqual(
            observed.exception.code,
            "runtime_gateway_sweeper_failed",
        )
        supervisor.begin_shutdown()
        await gateway.stop()
        await supervisor.shutdown()
        self.assertIsNone(service.execution_scheduler)
        self.assertTrue(all(task.done() for task in supervisor.tasks.values()))

    async def test_gateway_listener_and_sweeper_acknowledge_real_startup(
        self,
    ) -> None:
        supervisor = RuntimeSupervisor()
        service = _SweepService(fail=False)
        gateway = VoiceGateway(service, auth_key=AUTH_KEY)  # type: ignore[arg-type]
        server = _FakeServer()
        with patch(
            "websockets.asyncio.server.serve",
            new=AsyncMock(return_value=server),
        ):
            await gateway.start(supervisor=supervisor)
        self.assertFalse(supervisor.health()["ready"])
        self.assertTrue(await supervisor.finalize_readiness(asyncio.Event()))
        self.assertTrue(supervisor.health()["ready"])
        supervisor.begin_shutdown()
        await gateway.stop()
        await supervisor.shutdown()

    async def test_gateway_start_failure_clears_execution_scheduler(self) -> None:
        service = _SweepService(fail=False)
        gateway = VoiceGateway(service, auth_key=AUTH_KEY)  # type: ignore[arg-type]
        with (
            patch(
                "websockets.asyncio.server.serve",
                new=AsyncMock(side_effect=RuntimeError("listener bind failed")),
            ),
            self.assertRaisesRegex(RuntimeError, "listener bind failed"),
        ):
            await gateway.start(supervisor=RuntimeSupervisor())
        self.assertIsNone(service.execution_scheduler)
        self.assertIsNone(gateway._server)
        self.assertIsNone(gateway._sweeper)
        self.assertIsNone(gateway._server_monitor)

    async def test_gateway_failed_start_wait_closed_is_bounded(self) -> None:
        service = _SweepService(fail=False)
        gateway = VoiceGateway(service, auth_key=AUTH_KEY)  # type: ignore[arg-type]
        server = _CancellationResistantServer()
        exit_statuses: list[int] = []

        def terminate(status: int) -> None:
            exit_statuses.append(status)
            server.release.set()

        supervisor = RuntimeSupervisor(
            shutdown_timeout_seconds=0.05,
            process_terminator=terminate,
        )
        deadline = supervisor.process_deadline(0.25)
        original_start_task = supervisor.start_task
        registrations = 0

        def fail_second_registration(*args: object, **kwargs: object):
            nonlocal registrations
            registrations += 1
            if registrations == 2:
                raise RuntimeError("listener registration failed")
            return original_start_task(*args, **kwargs)

        try:
            with (
                patch(
                    "websockets.asyncio.server.serve",
                    new=AsyncMock(return_value=server),
                ),
                patch.object(
                    supervisor,
                    "start_task",
                    side_effect=fail_second_registration,
                ),
                self.assertRaises(RuntimeFailure) as observed,
            ):
                await gateway.start(
                    supervisor=supervisor,
                    shutdown_deadline=deadline,
                )
            self.assertEqual(
                observed.exception.code,
                "runtime_gateway_startup_cleanup_timeout",
            )
            self.assertEqual(exit_statuses, [RUNTIME_HARD_EXIT_STATUS])
            self.assertIsNone(gateway._server)
            self.assertIsNone(gateway._sweeper)
            self.assertIsNone(service.execution_scheduler)
        finally:
            deadline.disarm()

    async def test_gateway_listener_exit_reaches_process_supervisor(self) -> None:
        supervisor = RuntimeSupervisor()
        service = _SweepService(fail=False)
        gateway = VoiceGateway(service, auth_key=AUTH_KEY)  # type: ignore[arg-type]
        server = _FakeServer()
        with patch(
            "websockets.asyncio.server.serve",
            new=AsyncMock(return_value=server),
        ):
            await gateway.start(supervisor=supervisor)
        server.close()
        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.wait(asyncio.Event())
        self.assertEqual(
            observed.exception.code,
            "runtime_gateway_listener_failed",
        )
        supervisor.begin_shutdown()
        await gateway.stop()
        await supervisor.shutdown()


if __name__ == "__main__":
    unittest.main()
