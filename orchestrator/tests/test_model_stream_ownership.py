from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

import _bootstrap  # noqa: F401
from jarvis_orchestrator.domain import (
    ModelChunk,
    ModelRequest,
    ModelUsage,
    OrchestratorError,
)
from jarvis_orchestrator.model import (
    BudgetedModelPort,
    active_dispatch_reservation,
    consume_dispatch_reservation,
)
from jarvis_orchestrator.pydantic_adapter import (
    BudgetedPydanticModel,
    PydanticAgentModelPort,
)


class Store:
    def __init__(self):
        self.reconciled = []
        self.dispatch_error = None

    def reserve_model(self, *args, **kwargs):
        return "reservation"

    def mark_model_dispatched(self, *args):
        if self.dispatch_error:
            raise self.dispatch_error

    def reconcile_model(self, *args):
        self.reconciled.append(args[-1])


class Model:
    def __init__(self):
        self.closed = False
        self.permits = []

    async def stream(self, request):
        try:
            for text in ("first", "second"):
                self.permits.append(active_dispatch_reservation())
                yield ModelChunk(text, ModelUsage(output_tokens=2))
        finally:
            self.permits.append(active_dispatch_reservation())
            self.closed = True


class ModelStreamOwnershipTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return ModelRequest("task", 1, "final", "system", (), 16)

    async def test_early_close_reconciles_and_closes_delegate_through_both_wrappers(
        self,
    ):
        for outer in (False, True):
            with self.subTest(outer=outer):
                store, model = Store(), Model()
                port = BudgetedModelPort(store, model)
                if outer:
                    port = PydanticAgentModelPort(
                        None,
                        None,
                        store,
                        tools_capable=False,
                        json_schema_capable=False,
                        registered_tools=(),
                        final_stream_port=port,
                    )
                stream = port.stream(self.request())
                await anext(stream)
                await stream.aclose()
                self.assertEqual(store.reconciled, [None])
                self.assertTrue(model.closed)
                self.assertIsNone(active_dispatch_reservation())

    async def test_dispatch_permission_is_scoped_to_delegate_not_consumer(self):
        store, model = Store(), Model()
        stream = BudgetedModelPort(store, model).stream(self.request())
        try:
            async for chunk in stream:
                self.assertIsNone(active_dispatch_reservation())
                with self.assertRaises(OrchestratorError):
                    consume_dispatch_reservation()
        finally:
            await stream.aclose()
        self.assertEqual(store.reconciled, [2])
        self.assertEqual(model.permits, ["reservation"] * 3)

    async def test_close_in_another_task_restores_each_task_context(self):
        store, model = Store(), Model()
        stream = BudgetedModelPort(store, model).stream(self.request())
        await asyncio.create_task(anext(stream))
        await asyncio.create_task(stream.aclose())
        self.assertIsNone(active_dispatch_reservation())
        self.assertTrue(model.closed)
        self.assertEqual(store.reconciled, [None])

    async def test_failed_dispatch_mark_still_reconciles_reserved_budget(self):
        store = Store()
        store.dispatch_error = OrchestratorError("stale_epoch", status=409)
        stream = BudgetedModelPort(store, Model()).stream(self.request())
        with self.assertRaisesRegex(OrchestratorError, "stale_epoch"):
            await anext(stream)
        self.assertEqual(store.reconciled, [None])

    async def test_pydantic_reservation_handles_generator_exit(self):
        store = Store()
        wrapper = object.__new__(BudgetedPydanticModel)
        wrapper._jarvis_store = store
        wrapper._jarvis_context_provider = lambda: SimpleNamespace(
            task_id="task",
            epoch=1,
            purpose="final",
            output_allowance=16,
            timeout_seconds=1,
        )
        manager = wrapper._reservation({})
        await manager.__aenter__()
        await manager.gen.aclose()
        self.assertEqual(store.reconciled, [None])
        self.assertIsNone(active_dispatch_reservation())

    async def test_pydantic_failed_dispatch_mark_releases_reservation(self):
        store = Store()
        store.dispatch_error = OrchestratorError("stale_epoch", status=409)
        wrapper = object.__new__(BudgetedPydanticModel)
        wrapper._jarvis_store = store
        wrapper._jarvis_context_provider = lambda: SimpleNamespace(
            task_id="task",
            epoch=1,
            purpose="final",
            output_allowance=16,
            timeout_seconds=1,
        )
        with self.assertRaisesRegex(OrchestratorError, "stale_epoch"):
            async with wrapper._reservation({}):
                self.fail("dispatch proceeded after mark failed")
        self.assertEqual(store.reconciled, [None])
        self.assertIsNone(active_dispatch_reservation())

    async def test_early_close_reconciles_the_real_sqlite_reservation(self):
        from test_model_speech_schema import ModelBudgetTests

        fixture = ModelBudgetTests()
        self.addCleanup(fixture.doCleanups)
        store = fixture.make_store()
        task = fixture.live_executing_task(store)
        stream = BudgetedModelPort(store, Model()).stream(
            ModelRequest(task.task_id, task.epoch, "final", "system", (), 16)
        )
        await anext(stream)
        await stream.aclose()
        with store._lock:
            row = store._connection.execute(
                "SELECT state, actual_output_tokens FROM reservations WHERE task_id=?",
                (task.task_id,),
            ).fetchone()
        self.assertEqual(tuple(row), ("RECONCILED", 16))


if __name__ == "__main__":
    unittest.main()
