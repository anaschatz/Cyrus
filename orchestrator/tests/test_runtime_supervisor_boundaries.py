from __future__ import annotations

import asyncio
import subprocess
import sys
import unittest

import _bootstrap  # noqa: F401
from jarvis_orchestrator.runtime_supervisor import (
    RuntimeFailure,
    RuntimeSupervisor,
    SupervisedControlRuntime,
)
from test_runtime_supervision import _DrainingControlServer


class SupervisorBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_before_first_tick_records_failure(self):
        supervisor = RuntimeSupervisor()
        task = supervisor.start_task(
            "worker", lambda: asyncio.sleep(10), failure_code="runtime_worker_failed"
        )
        task.cancel()
        try:
            with self.assertRaises(RuntimeFailure):
                await supervisor.first_tick()
            self.assertIsNotNone(supervisor.failure)
            self.assertEqual(supervisor.health()["state"], "FAILED")
        finally:
            await supervisor.shutdown()

    async def test_shutdown_propagates_worker_cleanup_failure(self):
        supervisor = RuntimeSupervisor()
        started = asyncio.Event()

        async def worker():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                raise ValueError("private cleanup failure")

        supervisor.start_task("worker", worker, failure_code="runtime_worker_failed")
        await started.wait()
        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.shutdown()
        self.assertEqual(observed.exception.code, "runtime_worker_failed")
        self.assertNotIn("private", repr(supervisor.health()))
        self.assertEqual(supervisor.health()["state"], "FAILED")
        self.assertTrue(all(task.done() for task in supervisor.tasks.values()))

    async def test_outer_cancellation_cannot_abandon_owned_cleanup(self):
        for operation in ("cleanup", "shutdown"):
            with self.subTest(operation=operation):
                supervisor = RuntimeSupervisor(shutdown_timeout_seconds=0.5)
                started, release, finished = (
                    asyncio.Event(),
                    asyncio.Event(),
                    asyncio.Event(),
                )

                async def cleanup():
                    started.set()
                    await release.wait()
                    finished.set()

                if operation == "cleanup":
                    runner = supervisor.run_bounded_cleanup(
                        "cleanup",
                        cleanup,
                        failure_code="runtime_cleanup_failed",
                        timeout_code="runtime_cleanup_timeout",
                    )
                else:

                    async def worker():
                        try:
                            await asyncio.Event().wait()
                        finally:
                            await cleanup()

                    supervisor.start_task(
                        "worker", worker, failure_code="runtime_worker_failed"
                    )
                    await supervisor.first_tick()
                    runner = supervisor.shutdown()
                owner = asyncio.create_task(runner)
                await started.wait()
                try:
                    owner.cancel()
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    self.assertFalse(
                        owner.done(), "cleanup owner returned with live work"
                    )
                    owner.cancel()
                    await asyncio.sleep(0)
                finally:
                    release.set()
                    await asyncio.gather(owner, return_exceptions=True)
                    await asyncio.sleep(0)
                self.assertTrue(finished.is_set())
                self.assertTrue(owner.cancelled())

    async def test_control_cancellation_waits_for_handler_drain(self):
        supervisor = RuntimeSupervisor(shutdown_timeout_seconds=0.5)
        server = _DrainingControlServer()
        runtime = SupervisedControlRuntime(
            server, supervisor, shutdown_timeout_seconds=0.5
        )
        runtime.start()
        while not server.handler_started.is_set():
            await asyncio.sleep(0.001)
        supervisor.begin_shutdown()
        owner = asyncio.create_task(runtime.stop())
        while not server.shutdown_requested.is_set():
            await asyncio.sleep(0.001)
        try:
            owner.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertFalse(
                owner.done(), "control owner returned before handler drained"
            )
        finally:
            server.handler_release.set()
            await asyncio.gather(owner, return_exceptions=True)
            await supervisor.shutdown()
        self.assertTrue(owner.cancelled())
        self.assertTrue(server.handler_done.is_set())
        self.assertIsNone(runtime._thread)

    async def test_cleanup_accepts_awaitables_and_normalizes_synchronous_failure(self):
        supervisor = RuntimeSupervisor()

        def fail():
            raise ValueError("private synchronous failure")

        with self.assertRaises(RuntimeFailure) as observed:
            await supervisor.run_bounded_cleanup(
                "cleanup",
                fail,
                failure_code="runtime_cleanup_failed",
                timeout_code="runtime_cleanup_timeout",
            )
        self.assertEqual(observed.exception.code, "runtime_cleanup_failed")
        self.assertNotIn("private", repr(supervisor.health()))
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        await supervisor.run_bounded_cleanup(
            "cleanup",
            lambda: future,
            failure_code="runtime_cleanup_failed",
            timeout_code="runtime_cleanup_timeout",
        )

    def test_cleanup_process_control_exception_is_normalized_inside_task(self):
        source = """
import asyncio
from jarvis_orchestrator.runtime_supervisor import RuntimeFailure, RuntimeSupervisor
async def run():
    async def cleanup():
        raise SystemExit(7)
    supervisor = RuntimeSupervisor()
    try:
        await supervisor.run_bounded_cleanup("cleanup", cleanup,
            failure_code="runtime_cleanup_failed", timeout_code="runtime_cleanup_timeout")
    except RuntimeFailure as exc:
        assert exc.code == "runtime_cleanup_failed"
        assert supervisor.health()["state"] == "FAILED"
    else:
        raise AssertionError("cleanup failure was swallowed")
asyncio.run(run())
"""
        result = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, timeout=2
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
