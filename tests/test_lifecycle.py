from __future__ import annotations

import threading
import time
import unittest

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
