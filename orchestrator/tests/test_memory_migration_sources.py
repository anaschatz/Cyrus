from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401
from helpers import NOW, StoreCase
from jarvis_orchestrator.memory_crypto import MemoryEnvelopeCipherV1
from jarvis_orchestrator.memory_runtime import InjectedMemoryCryptoRuntime
from jarvis_orchestrator.store import MemoryStorageUnavailable
from test_memory_storage_v10 import (
    TransactionSpyProvider,
    _complete_disabled_source,
    _insert_legacy_record,
)


class MemoryMigrationSourceTests(StoreCase, unittest.TestCase):
    def test_missing_source_binding_preserves_legacy_data_before_bootstrap(self):
        for table in ("canonical_turns", "tasks", "inbox"):
            with self.subTest(table=table):
                provider = TransactionSpyProvider()
                runtime = InjectedMemoryCryptoRuntime(
                    provider.install_id, MemoryEnvelopeCipherV1(provider)
                )
                store = self.make_store(memory_crypto=runtime)
                source = _complete_disabled_source(store)
                legacy = _insert_legacy_record(
                    store, source, content="violet bicycle", created_at=NOW
                )
                # Canonical turns have no incoming FK. The other cases model a
                # legacy database damaged by a writer without FK enforcement.
                connection = store._connection
                if table != "canonical_turns":
                    connection.execute("PRAGMA foreign_keys=OFF")
                connection.execute(
                    f"DELETE FROM {table} WHERE task_id=?", (source.task_id,)
                )
                connection.execute("PRAGMA foreign_keys=ON")
                with self.assertRaisesRegex(
                    MemoryStorageUnavailable, "memory_legacy_record_invalid"
                ):
                    store.prepare_memory_storage(now=NOW)
                self.assertEqual(provider.calls, [])
                self.assertEqual(
                    connection.execute(
                        "SELECT content FROM memory_records_v9_legacy "
                        "WHERE record_id=?", (legacy["record_id"],)
                    ).fetchone()[0],
                    legacy["content"],
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM memory_crypto_operations"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    store.memory_storage_status()["storage_state"], "LEGACY_PENDING"
                )


if __name__ == "__main__":
    unittest.main()
