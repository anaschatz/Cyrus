"""Fail-closed supervision for critical Orchestrator runtime components.

The supervisor deliberately records only bounded component names and stable
reason codes.  Exception messages are never exposed through its health
snapshot because background failures can otherwise become a transcript or
provider-payload disclosure path.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from types import FrameType
from typing import Any

RUNTIME_HEALTH_VERSION = "jarvis-runtime-health/1"
RUNTIME_HARD_EXIT_STATUS = 70
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class RuntimeFailure(RuntimeError):
    """A critical component failed and the process must exit non-zero."""

    def __init__(
        self,
        code: str,
        *,
        component: str,
        cause: BaseException | None = None,
    ) -> None:
        _require_identifier(code, field="failure code")
        _require_identifier(component, field="component")
        super().__init__(code)
        self.code = code
        self.component = component
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class RuntimeComponentHealth:
    component: str
    state: str
    failure_code: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "component": self.component,
            "failure_code": self.failure_code,
            "state": self.state,
        }


class RuntimeShutdownBudget:
    """One monotonic deadline shared by every sequential cleanup stage."""

    def __init__(self, total_seconds: float) -> None:
        if not 0.05 <= total_seconds <= 30.0:
            raise ValueError("total_seconds must be between 50ms and 30s")
        self.deadline = time.monotonic() + total_seconds

    def remaining(self) -> float:
        # Zero is intentional: the receiving bounded operation gets one event
        # loop turn to reap work that already completed, otherwise it invokes
        # the hard-exit seam while it still owns the unfinished task/thread.
        return max(0.0, self.deadline - time.monotonic())


class RuntimeProcessDeadline:
    """A wall-clock process deadline that does not depend on the event loop.

    The daemon watchdog is intentionally independent of asyncio.  Once armed,
    it remains live across async adapter cleanup and synchronous store closing;
    only the outer lifecycle owner may disarm it after every cleanup boundary
    has completed successfully.
    """

    def __init__(
        self,
        total_seconds: float,
        *,
        process_terminator: Callable[[int], object] = os._exit,
        exit_status: int = RUNTIME_HARD_EXIT_STATUS,
    ) -> None:
        if not 0.05 <= total_seconds <= 30.0:
            raise ValueError("total_seconds must be between 50ms and 30s")
        self.total_seconds = total_seconds
        self._process_terminator = process_terminator
        self.exit_status = exit_status
        self._lock = threading.Lock()
        self._disarmed = threading.Event()
        self._deadline: float | None = None
        self._watchdog: threading.Thread | None = None
        self._fired = False

    @property
    def started(self) -> bool:
        with self._lock:
            return self._deadline is not None

    @property
    def fired(self) -> bool:
        with self._lock:
            return self._fired

    def begin(self) -> None:
        with self._lock:
            if self._deadline is not None:
                return
            if self._disarmed.is_set():
                raise RuntimeError("runtime_process_deadline_disarmed")
            self._deadline = time.monotonic() + self.total_seconds
            watchdog = threading.Thread(
                target=self._watch,
                name="jarvis-runtime-process-deadline",
                daemon=True,
            )
            self._watchdog = watchdog
            watchdog.start()

    def remaining(self) -> float:
        with self._lock:
            deadline = self._deadline
        if deadline is None:
            return self.total_seconds
        return max(0.0, deadline - time.monotonic())

    def disarm(self) -> None:
        self._disarmed.set()

    def _watch(self) -> None:
        remaining = self.remaining()
        if self._disarmed.wait(remaining):
            return
        with self._lock:
            if self._disarmed.is_set():
                return
            self._fired = True
        self._process_terminator(self.exit_status)


def _require_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"runtime {field} is invalid")
    return value


def _finish_cleanup_before_cancellation(operation):
    """Keep bounded cleanup owned even when its caller is cancelled repeatedly."""

    @wraps(operation)
    async def finish(*args, **kwargs):
        task = asyncio.create_task(operation(*args, **kwargs))
        cancelled = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancelled = exc
        result = task.result()
        if cancelled is not None:
            raise cancelled
        return result

    return finish


class RuntimeSupervisor:
    """Own critical tasks, surface their failures, and bound their shutdown."""

    def __init__(
        self,
        *,
        shutdown_timeout_seconds: float = 3.0,
        process_terminator: Callable[[int], object] = os._exit,
    ) -> None:
        if not 0.05 <= shutdown_timeout_seconds <= 30.0:
            raise ValueError("shutdown_timeout_seconds must be between 50ms and 30s")
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self._process_terminator = process_terminator
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._component_health: dict[str, RuntimeComponentHealth] = {}
        self._failure: RuntimeFailure | None = None
        self._failure_event = asyncio.Event()
        self._readiness_changed = asyncio.Event()
        self._ready_components: set[str] = set()
        self._registrations_sealed = False
        self._ready = False
        self._stopping = False
        self._stopped = False

    @property
    def failure(self) -> RuntimeFailure | None:
        return self._failure

    @property
    def tasks(self) -> Mapping[str, asyncio.Task[None]]:
        """A read-only view used by lifecycle owners and focused diagnostics."""

        return dict(self._tasks)

    def process_deadline(self, total_seconds: float) -> RuntimeProcessDeadline:
        return RuntimeProcessDeadline(
            total_seconds,
            process_terminator=self._process_terminator,
        )

    def business_admission_allowed(self) -> bool:
        return (
            self._ready
            and self._failure is None
            and not self._stopping
            and not self._stopped
        )

    def start_task(
        self,
        component: str,
        runner: Callable[[], Awaitable[Any]],
        *,
        failure_code: str,
    ) -> asyncio.Task[None]:
        component = _require_identifier(component, field="component")
        failure_code = _require_identifier(failure_code, field="failure code")
        if self._stopping or self._stopped:
            raise RuntimeError("runtime_supervisor_stopping")
        if self._registrations_sealed:
            raise RuntimeError("runtime_supervisor_registration_sealed")
        existing = self._tasks.get(component)
        if existing is not None:
            raise RuntimeError("runtime_component_already_registered")
        self._component_health[component] = RuntimeComponentHealth(
            component=component,
            state="RUNNING",
            failure_code=None,
        )
        task = asyncio.create_task(
            self._run_component(component, runner, failure_code),
            name=f"jarvis-runtime-{component}",
        )
        self._tasks[component] = task

        def completed(task: asyncio.Task[None]) -> None:
            # A task cancelled before its first tick never enters the runner.
            if task.cancelled() and not self._stopping:
                self._record_failure(component, failure_code)

        task.add_done_callback(completed)
        return task

    def acknowledge_ready(self, component: str) -> None:
        """Acknowledge one component's first successful tick or bind."""

        component = _require_identifier(component, field="component")
        if self._stopping or self._stopped:
            return
        if component not in self._tasks:
            raise RuntimeError("runtime_component_not_registered")
        health = self._component_health[component]
        if health.state != "RUNNING":
            raise RuntimeError("runtime_component_not_running")
        self._ready_components.add(component)
        self._readiness_changed.set()

    async def finalize_readiness(
        self,
        stop_requested: asyncio.Event,
        *,
        timeout_seconds: float | None = None,
    ) -> bool:
        """Seal registrations and open admission after every component ACK.

        ``False`` means a controlled stop arrived before readiness.  A critical
        failure or bounded startup-barrier timeout is raised instead.
        """

        if self._registrations_sealed:
            raise RuntimeError("runtime_supervisor_registration_sealed")
        if not self._tasks:
            raise RuntimeError("runtime_supervisor_has_no_components")
        self._registrations_sealed = True
        timeout = self._selected_timeout(timeout_seconds)

        async def wake_for_stop() -> None:
            await stop_requested.wait()
            self._readiness_changed.set()

        async def wait_for_acknowledgements() -> bool:
            while True:
                if self._failure is not None:
                    raise self._failure
                if stop_requested.is_set():
                    return False
                if self._ready_components == set(self._tasks):
                    self._ready = True
                    return True
                self._readiness_changed.clear()
                # No task can interleave between the clear and these checks;
                # the repeat prevents a notification from being lost.
                if self._failure is not None or stop_requested.is_set():
                    continue
                if self._ready_components == set(self._tasks):
                    continue
                await self._readiness_changed.wait()

        stop_bridge = asyncio.create_task(
            wake_for_stop(), name="jarvis-runtime-startup-stop-bridge"
        )
        try:
            try:
                return await asyncio.wait_for(
                    wait_for_acknowledgements(), timeout=timeout
                )
            except TimeoutError as exc:
                self._record_failure(
                    "supervisor",
                    "runtime_startup_barrier_timeout",
                    cause=exc,
                )
                assert self._failure is not None
                raise self._failure
        finally:
            stop_bridge.cancel()
            await asyncio.gather(stop_bridge, return_exceptions=True)

    async def _run_component(
        self,
        component: str,
        runner: Callable[[], Awaitable[Any]],
        failure_code: str,
    ) -> None:
        cause: BaseException | None = None
        try:
            await runner()
        except asyncio.CancelledError as exc:
            cause = exc
            if not self._stopping:
                self._record_failure(component, failure_code, cause=exc)
            raise
        except BaseException as exc:  # noqa: BLE001 - critical task boundary
            cause = exc
            self._record_failure(component, failure_code, cause=exc)
            if self._stopping:
                raise RuntimeFailure(
                    failure_code, component=component, cause=exc
                ) from exc
        else:
            if not self._stopping:
                self._record_failure(component, failure_code)
        finally:
            current = self._component_health[component]
            if current.state != "FAILED":
                self._component_health[component] = RuntimeComponentHealth(
                    component=component,
                    state="STOPPED" if self._stopping else "FAILED",
                    failure_code=None if self._stopping else failure_code,
                )
                if not self._stopping and cause is None:
                    self._record_failure(component, failure_code)

    def _record_failure(
        self,
        component: str,
        failure_code: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        self._component_health[component] = RuntimeComponentHealth(
            component=component,
            state="FAILED",
            failure_code=failure_code,
        )
        if self._failure is None:
            self._failure = RuntimeFailure(
                failure_code,
                component=component,
                cause=cause,
            )
            self._failure_event.set()
            self._ready = False
            self._readiness_changed.set()

    async def wait(self, stop_requested: asyncio.Event) -> None:
        """Return for a controlled stop or raise for the first critical failure."""

        if self._failure is not None:
            raise self._failure
        stop_waiter = asyncio.create_task(
            stop_requested.wait(), name="jarvis-runtime-stop-event"
        )
        failure_waiter = asyncio.create_task(
            self._failure_event.wait(), name="jarvis-runtime-failure-event"
        )
        try:
            await asyncio.wait(
                (stop_waiter, failure_waiter),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            stop_waiter.cancel()
            failure_waiter.cancel()
            await asyncio.gather(stop_waiter, failure_waiter, return_exceptions=True)
        if self._failure is not None:
            raise self._failure

    async def first_tick(self) -> None:
        """Let newly started components run once before readiness is exposed."""

        await asyncio.sleep(0)
        # Drain completion callbacks for tasks cancelled before entering their runner.
        await asyncio.gather(
            *(task for task in self._tasks.values() if task.done()),
            return_exceptions=True,
        )
        if self._failure is not None:
            raise self._failure

    def begin_shutdown(self) -> None:
        self._ready = False
        self._registrations_sealed = True
        self._stopping = True
        for component, current in tuple(self._component_health.items()):
            if current.state == "RUNNING":
                self._component_health[component] = RuntimeComponentHealth(
                    component=component,
                    state="STOPPING",
                    failure_code=None,
                )

    @_finish_cleanup_before_cancellation
    async def shutdown(self, *, timeout_seconds: float | None = None) -> None:
        """Cancel every owned task and wait no longer than the configured bound."""

        timeout = self._selected_timeout(timeout_seconds)
        self.begin_shutdown()
        pending = tuple(task for task in self._tasks.values() if not task.done())
        for task in pending:
            task.cancel()
        if pending:
            done, unfinished = await asyncio.wait(
                pending,
                timeout=timeout,
            )
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            if unfinished:
                for task in unfinished:
                    task.cancel()
                timeout_failure = RuntimeFailure(
                    "runtime_shutdown_timeout",
                    component="supervisor",
                )
                self._component_health["supervisor"] = RuntimeComponentHealth(
                    component="supervisor",
                    state="FAILED",
                    failure_code=timeout_failure.code,
                )
                if self._failure is None:
                    self._failure = timeout_failure
                    self._failure_event.set()
                # Production's os._exit never returns, so asyncio.run cannot
                # enter its unbounded finalizer with cancellation-resistant
                # owned tasks.  Tests inject a seam that releases the task;
                # only then may this method return control to the event loop.
                self._process_terminator(RUNTIME_HARD_EXIT_STATUS)
                recovered, still_unfinished = await asyncio.wait(
                    unfinished,
                    timeout=self.shutdown_timeout_seconds,
                )
                if recovered:
                    await asyncio.gather(*recovered, return_exceptions=True)
                if still_unfinished:
                    os._exit(RUNTIME_HARD_EXIT_STATUS)
                    raise AssertionError("os._exit returned")  # pragma: no cover
                raise timeout_failure
        completed = tuple(task for task in self._tasks.values() if task.done())
        results = await asyncio.gather(*completed, return_exceptions=True)
        for component, current in tuple(self._component_health.items()):
            if current.state != "FAILED":
                self._component_health[component] = RuntimeComponentHealth(
                    component=component,
                    state="STOPPED",
                    failure_code=None,
                )
        self._stopped = True
        for result in results:
            if isinstance(result, RuntimeFailure):
                raise result

    @_finish_cleanup_before_cancellation
    async def run_bounded_cleanup(
        self,
        component: str,
        runner: Callable[[], Awaitable[Any]],
        *,
        failure_code: str,
        timeout_code: str,
        timeout_seconds: float | None = None,
    ) -> None:
        """Run cleanup without ever returning a cancellation-resistant task."""

        component = _require_identifier(component, field="component")
        failure_code = _require_identifier(failure_code, field="failure code")
        timeout_code = _require_identifier(timeout_code, field="failure code")
        timeout = self._selected_timeout(timeout_seconds)

        async def cleanup() -> None:
            try:
                await runner()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # Normalize before asyncio handles SystemExit.
                self._record_failure(component, failure_code, cause=exc)
                raise RuntimeFailure(
                    failure_code, component=component, cause=exc
                ) from exc

        task = asyncio.create_task(
            cleanup(),
            name=f"jarvis-runtime-{component}-cleanup",
        )
        done, unfinished = await asyncio.wait(
            (task,),
            timeout=timeout,
        )
        if unfinished:
            task.cancel()
            timeout_failure = self.request_hard_exit(
                timeout_code,
                component=component,
            )
            recovered, still_unfinished = await asyncio.wait(
                unfinished,
                timeout=self.shutdown_timeout_seconds,
            )
            if recovered:
                await asyncio.gather(*recovered, return_exceptions=True)
            if still_unfinished:
                os._exit(RUNTIME_HARD_EXIT_STATUS)
                raise AssertionError("os._exit returned")  # pragma: no cover
            raise timeout_failure
        assert done == {task}
        try:
            task.result()
        except asyncio.CancelledError as exc:
            self._record_failure(component, failure_code, cause=exc)
            raise RuntimeFailure(
                failure_code, component=component, cause=exc
            ) from exc

    def _selected_timeout(self, timeout_seconds: float | None) -> float:
        if timeout_seconds is None:
            return self.shutdown_timeout_seconds
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        return min(timeout_seconds, self.shutdown_timeout_seconds)

    def health(self) -> dict[str, Any]:
        if self._failure is not None:
            state = "FAILED"
        elif self._stopped:
            state = "STOPPED"
        elif self._stopping:
            state = "STOPPING"
        else:
            state = "RUNNING"
        components = [
            self._component_health[name].to_dict()
            for name in sorted(self._component_health)
        ]
        return {
            "component_count": len(components),
            "components": components,
            "failure_code": self._failure.code if self._failure is not None else None,
            "ready": state == "RUNNING" and self.business_admission_allowed(),
            "state": state,
            "version": RUNTIME_HEALTH_VERSION,
        }

    def request_hard_exit(
        self,
        failure_code: str,
        *,
        component: str,
        cause: BaseException | None = None,
    ) -> RuntimeFailure:
        """Terminate production, or return a failure after an injected test seam."""

        failure_code = _require_identifier(failure_code, field="failure code")
        component = _require_identifier(component, field="component")
        failure = RuntimeFailure(failure_code, component=component, cause=cause)
        self._component_health[component] = RuntimeComponentHealth(
            component=component,
            state="FAILED",
            failure_code=failure_code,
        )
        if self._failure is None:
            self._failure = failure
            self._failure_event.set()
        self._ready = False
        self._readiness_changed.set()
        self._process_terminator(RUNTIME_HARD_EXIT_STATUS)
        return failure


class SupervisedControlRuntime:
    """Run and fully drain a threaded HTTP server before durable state closes."""

    def __init__(
        self,
        server: Any,
        supervisor: RuntimeSupervisor,
        *,
        shutdown_timeout_seconds: float,
    ) -> None:
        self.server = server
        self.supervisor = supervisor
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self._thread: threading.Thread | None = None
        self._serve_failure: list[BaseException] = []

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("runtime_control_already_started")
        # ThreadingMixIn.server_close joins all non-daemon request handlers.
        self.server.httpd.daemon_threads = False
        self.server.httpd.block_on_close = True
        thread = threading.Thread(
            target=self._serve,
            name="jarvis-control",
            daemon=False,
        )
        self._thread = thread
        try:
            self.supervisor.start_task(
                "control_server",
                lambda: self._monitor(thread),
                failure_code="runtime_control_failed",
            )
        except BaseException:
            self.server.httpd.server_close()
            self._thread = None
            raise
        try:
            thread.start()
        except BaseException:
            self.server.httpd.server_close()
            self._thread = None
            raise

    def _serve(self) -> None:
        try:
            self.server.serve_forever()
        except BaseException as exc:  # noqa: BLE001 - supervised thread boundary
            self._serve_failure.append(exc)

    async def _monitor(self, thread: threading.Thread) -> None:
        if thread.is_alive():
            self.supervisor.acknowledge_ready("control_server")
        while thread.is_alive():
            await asyncio.sleep(0.05)
        if self._serve_failure:
            raise self._serve_failure[0]

    @staticmethod
    async def _wait_for_thread(
        thread: threading.Thread,
        *,
        deadline: float,
    ) -> bool:
        loop = asyncio.get_running_loop()
        while thread.is_alive() and loop.time() < deadline:
            await asyncio.sleep(0.02)
        return not thread.is_alive()

    @_finish_cleanup_before_cancellation
    async def stop(self, *, timeout_seconds: float | None = None) -> None:
        if self._thread is None:
            return
        selected_timeout = (
            self.shutdown_timeout_seconds
            if timeout_seconds is None
            else min(timeout_seconds, self.shutdown_timeout_seconds)
        )
        if selected_timeout < 0:
            raise ValueError("timeout_seconds must be non-negative")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + selected_timeout
        shutdown_failure: list[BaseException] = []

        def shutdown_server() -> None:
            try:
                self.server.shutdown()
            except BaseException as exc:  # noqa: BLE001 - shutdown thread boundary
                shutdown_failure.append(exc)
            finally:
                try:
                    # ControlServer normally closes internally.  Repeating the
                    # close is idempotent and guarantees ThreadingMixIn joins
                    # request handlers even when its shutdown wrapper failed.
                    self.server.httpd.server_close()
                except BaseException as exc:  # noqa: BLE001 - drain boundary
                    shutdown_failure.append(exc)

        shutdown_thread = threading.Thread(
            target=shutdown_server,
            name="jarvis-control-shutdown",
            daemon=True,
        )
        shutdown_thread.start()
        hard_failure: RuntimeFailure | None = None
        if not await self._wait_for_thread(shutdown_thread, deadline=deadline):
            hard_failure = self.supervisor.request_hard_exit(
                "runtime_control_shutdown_timeout",
                component="control_server",
            )
            recovery_deadline = loop.time() + self.shutdown_timeout_seconds
            if not await self._wait_for_thread(
                shutdown_thread,
                deadline=recovery_deadline,
            ):
                os._exit(RUNTIME_HARD_EXIT_STATUS)
                raise AssertionError("os._exit returned")  # pragma: no cover
        if shutdown_failure and hard_failure is None:
            hard_failure = self.supervisor.request_hard_exit(
                "runtime_control_shutdown_failed",
                component="control_server",
                cause=shutdown_failure[0],
            )
        if not await self._wait_for_thread(self._thread, deadline=deadline):
            if hard_failure is None:
                hard_failure = self.supervisor.request_hard_exit(
                    "runtime_control_shutdown_timeout",
                    component="control_server",
                )
            recovery_deadline = loop.time() + self.shutdown_timeout_seconds
            if not await self._wait_for_thread(
                self._thread,
                deadline=recovery_deadline,
            ):
                os._exit(RUNTIME_HARD_EXIT_STATUS)
                raise AssertionError("os._exit returned")  # pragma: no cover
        self._thread = None
        if hard_failure is not None:
            raise hard_failure


class RuntimeSignalGuard:
    """Install reversible stop handlers and absorb repeat signals during cleanup."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        stop_requested: asyncio.Event,
        *,
        selected_signals: tuple[signal.Signals, ...] = (
            signal.SIGINT,
            signal.SIGTERM,
        ),
    ) -> None:
        if not selected_signals or len(set(selected_signals)) != len(selected_signals):
            raise ValueError("selected_signals must be unique and non-empty")
        self.loop = loop
        self.stop_requested = stop_requested
        self.selected_signals = selected_signals
        self._previous: dict[signal.Signals, Any] = {}
        self._installed = False
        self._cleanup_started = False
        self._request_count = 0

    @property
    def request_count(self) -> int:
        return self._request_count

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("runtime_signal_guard_already_installed")
        self._cleanup_started = False
        self._request_count = 0
        loop_handlers = getattr(self.loop, "_signal_handlers", None)
        if isinstance(loop_handlers, Mapping) and any(
            selected_signal in loop_handlers
            for selected_signal in self.selected_signals
        ):
            raise RuntimeError("runtime_signal_handler_conflict")
        try:
            for selected_signal in self.selected_signals:
                self._previous[selected_signal] = signal.getsignal(selected_signal)
                signal.signal(selected_signal, self._fallback_handler)
            self._installed = True
        except BaseException:
            self.restore()
            raise

    def request_stop(self) -> None:
        self._request_count += 1
        if self._cleanup_started:
            return
        if not self.stop_requested.is_set():
            self.stop_requested.set()

    def begin_cleanup(self) -> None:
        self._cleanup_started = True

    def _fallback_handler(
        self,
        _signal_number: int,
        _frame: FrameType | None,
    ) -> None:
        self.loop.call_soon_threadsafe(self.request_stop)

    def restore(self) -> None:
        if not self._previous:
            self._installed = False
            return
        for selected_signal in self.selected_signals:
            if selected_signal not in self._previous:
                continue
            signal.signal(selected_signal, self._previous[selected_signal])
        self._previous.clear()
        self._installed = False


__all__ = [
    "RUNTIME_HARD_EXIT_STATUS",
    "RUNTIME_HEALTH_VERSION",
    "RuntimeComponentHealth",
    "RuntimeFailure",
    "RuntimeProcessDeadline",
    "RuntimeShutdownBudget",
    "RuntimeSignalGuard",
    "RuntimeSupervisor",
    "SupervisedControlRuntime",
]
