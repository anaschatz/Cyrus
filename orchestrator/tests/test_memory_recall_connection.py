from __future__ import annotations

import asyncio
import sqlite3
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.memory_crypto import (
    InMemoryMemoryEpochKeyProvider,
    MemoryEnvelopeCipherV1,
)
from jarvis_orchestrator.memory_runtime import InjectedMemoryCryptoRuntime
from test_memory_storage_v10 import _complete_command


class MemoryRecallConnectionTests(StoreCase, unittest.TestCase):
    def test_recall_closes_connection_after_setup_query_failure_and_success(self):
        provider = InMemoryMemoryEpochKeyProvider()
        runtime = InjectedMemoryCryptoRuntime(
            provider.install_id, MemoryEnvelopeCipherV1(provider)
        )
        store = self.make_store(memory_crypto=runtime)
        store.prepare_memory_storage(now=NOW)
        task, _ = _complete_command(store, "Remember that violet bicycle.")
        original = sqlite3.connect
        for boundary in ("query_only", "busy_timeout", "query", "success"):
            with self.subTest(boundary=boundary):
                connections = []

                def connect(*args, **kwargs):
                    # Inspect real closure from the test thread after the worker.
                    connection = original(*args, check_same_thread=False, **kwargs)
                    connections.append(connection)
                    self.addCleanup(connection.close)

                    def authorize(action, arg1, arg2, database, trigger):
                        denied = (
                            action == sqlite3.SQLITE_PRAGMA and arg1 == boundary
                            or action == sqlite3.SQLITE_READ and boundary == "query"
                        )
                        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK

                    connection.set_authorizer(authorize)
                    return connection

                async def recall():
                    return await store.search_memory_records(
                        conversation_id=task.conversation_id,
                        query="violet", limit=1, now=NOW,
                    )

                with patch.object(sqlite3, "connect", side_effect=connect):
                    if boundary == "success":
                        self.assertEqual(len(asyncio.run(recall())), 1)
                    else:
                        with self.assertRaises(sqlite3.DatabaseError):
                            asyncio.run(recall())
                self.assertEqual(len(connections), 1)
                with self.assertRaises(sqlite3.ProgrammingError):
                    connections[0].execute("SELECT 1")
                self.assertFalse(store._connection.in_transaction)


if __name__ == "__main__":
    unittest.main()
