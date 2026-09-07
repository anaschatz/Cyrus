from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.lifecycle import LifecycleStatus, VoiceLifecycleController


def wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition timed out")


class FakeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str | None, str | None]] = []

    def write(self, event: str, *, status: str | None = None, code: str | None = None) -> None:
        self.records.append((event, status, code))


class BlockingRuntime:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stop_count = 0

    def run(self) -> None:
        self.entered.set()
        self.release.wait(timeout=2)

    def request_stop(self) -> None:
        self.stop_count += 1
        self.release.set()


class LifecycleTests(unittest.TestCase):
    def test_stop_can_retry_while_failed_runtime_is_still_alive(self) -> None:
        runtime = BlockingRuntime()

        def request_stop() -> None:
            runtime.stop_count += 1
            if runtime.stop_count == 1:
                raise OSError("temporary cancellation failure")
            runtime.release.set()

        runtime.request_stop = request_stop
        controller = VoiceLifecycleController(lambda: runtime, shutdown_timeout=0.5)
        controller.start()
        self.assertTrue(runtime.entered.wait(1))
        try:
            self.assertFalse(controller.stop())
            self.assertEqual(controller.snapshot.error_code, "runtime.stop_failed")
            self.assertTrue(controller.stop())
            self.assertEqual(runtime.stop_count, 2)
            self.assertEqual(controller.snapshot.status, LifecycleStatus.STOPPED)
        finally:
            runtime.release.set()
            thread = controller._thread
            if thread is not None:
                thread.join(1)

    def test_thread_start_failure_is_reported_and_can_be_retried(self) -> None:
        for phase in ("construction", "start"):
            with self.subTest(phase=phase):
                runtime = BlockingRuntime()
                logger = FakeLogger()
                controller = VoiceLifecycleController(lambda: runtime, logger=logger, shutdown_timeout=0.5)
                target = "jarvis_voice.lifecycle.threading.Thread" + (".start" if phase == "start" else "")
                try:
                    with mock.patch(target, side_effect=RuntimeError("private start failure")):
                        self.assertFalse(controller.start())
                    self.assertEqual(controller.snapshot.status, LifecycleStatus.ERROR)
                    self.assertEqual(controller.snapshot.error_code, "runtime.start_failed")
                    self.assertIsNone(controller._thread)
                    self.assertFalse(controller.stop())
                    self.assertNotIn("private start failure", repr(logger.records))
                    self.assertTrue(controller.start())
                    self.assertTrue(runtime.entered.wait(1.0))
                    self.assertTrue(controller.stop())
                    self.assertEqual(controller.snapshot.status, LifecycleStatus.STOPPED)
                finally:
                    runtime.release.set()
                    if controller._thread is not None and controller._thread.ident is not None:
                        controller._thread.join(timeout=1.0)

    def test_start_stop_and_shutdown_are_bounded_and_idempotent(self) -> None:
        runtime = BlockingRuntime(); logger = FakeLogger()
        controller = VoiceLifecycleController(lambda: runtime, logger=logger, shutdown_timeout=0.5)
        self.assertTrue(controller.start())
        self.assertTrue(runtime.entered.wait(timeout=1))
        wait_for(lambda: controller.snapshot.status == LifecycleStatus.RUNNING)
        self.assertFalse(controller.start())
        self.assertTrue(controller.stop())
        wait_for(lambda: controller.snapshot.status == LifecycleStatus.STOPPED)
        self.assertEqual(runtime.stop_count, 1)
        controller.shutdown()
        self.assertEqual(runtime.stop_count, 1)
        self.assertTrue(any(record[0] == "voice.lifecycle.running" for record in logger.records))

    def test_typed_and_unhandled_failures_store_only_codes(self) -> None:
        for factory, expected in (
            (lambda: (_ for _ in ()).throw(BackendError("runtime.known", "secret", subsystem="runtime")), "runtime.known"),
            (lambda: (_ for _ in ()).throw(RuntimeError("private transcript")), "runtime.unhandled_failure"),
        ):
            logger = FakeLogger(); controller = VoiceLifecycleController(factory, logger=logger)
            controller.start()
            wait_for(lambda: controller.snapshot.status == LifecycleStatus.ERROR)
            self.assertEqual(controller.snapshot.error_code, expected)
            self.assertNotIn("secret", repr(logger.records))
            self.assertNotIn("private transcript", repr(logger.records))

    def test_shutdown_timeout_is_typed_and_can_be_recovered_by_worker_exit(self) -> None:
        runtime = BlockingRuntime()

        def ignore_stop() -> None:
            runtime.stop_count += 1

        runtime.request_stop = ignore_stop  # type: ignore[method-assign]
        controller = VoiceLifecycleController(lambda: runtime, shutdown_timeout=0.01)
        controller.start(); self.assertTrue(runtime.entered.wait(timeout=1))
        self.assertFalse(controller.stop())
        self.assertEqual(controller.snapshot.error_code, "runtime.shutdown_timeout")
        runtime.release.set()

    def test_stop_during_factory_startup_is_not_lost(self) -> None:
        factory_entered = threading.Event(); factory_release = threading.Event()
        runtime = BlockingRuntime()

        def factory() -> BlockingRuntime:
            factory_entered.set(); factory_release.wait(timeout=1); return runtime

        controller = VoiceLifecycleController(factory, shutdown_timeout=0.5)
        controller.start(); self.assertTrue(factory_entered.wait(timeout=1))
        stopped: list[bool] = []
        stopper = threading.Thread(target=lambda: stopped.append(controller.stop()))
        stopper.start(); factory_release.set(); stopper.join(timeout=1)
        self.assertEqual(stopped, [True])
        self.assertEqual(runtime.stop_count, 1)
        wait_for(lambda: controller.snapshot.status == LifecycleStatus.STOPPED)


if __name__ == "__main__":
    unittest.main()
