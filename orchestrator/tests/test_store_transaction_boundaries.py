from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from jarvis_orchestrator import store as module
from jarvis_orchestrator.store import OrchestratorStore


class StoreTransactionBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_failed_initialization_closes_every_opened_connection(self):
        original = sqlite3.connect
        for boundary in ("pragma", "begin", "commit", "final_private_check"):
            with self.subTest(boundary=boundary):
                connections = []

                def connect(*args, **kwargs):
                    connection = original(*args, **kwargs)
                    connections.append(connection)
                    self.addCleanup(connection.close)

                    def authorize(action, arg1, arg2, database, trigger):
                        denied = (
                            boundary == "pragma"
                            and action == sqlite3.SQLITE_PRAGMA
                            or boundary == "begin"
                            and action == sqlite3.SQLITE_TRANSACTION
                            and arg1 == "BEGIN"
                            or boundary == "commit"
                            and action == sqlite3.SQLITE_TRANSACTION
                            and arg1 == "COMMIT"
                        )
                        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK

                    connection.set_authorizer(authorize)
                    return connection

                verify = module.assert_private_database
                checks = 0

                def private_check(path):
                    nonlocal checks
                    checks += 1
                    if boundary == "final_private_check" and checks == 2:
                        raise RuntimeError("synthetic private check failure")
                    return verify(path)

                with (
                    patch.object(module.sqlite3, "connect", side_effect=connect),
                    patch.object(
                        module, "assert_private_database", side_effect=private_check
                    ),
                ):
                    with self.assertRaises((sqlite3.DatabaseError, RuntimeError)):
                        OrchestratorStore(self.root / (boundary + ".sqlite3"))
                self.assertEqual(len(connections), 1)
                with self.assertRaises(sqlite3.ProgrammingError):
                    connections[0].execute("SELECT 1")

    def test_invalid_memory_runtime_fails_before_creating_database(self):
        path = self.root / "invalid.sqlite3"
        with self.assertRaisesRegex(TypeError, "memory crypto runtime is invalid"):
            OrchestratorStore(path, memory_crypto=object())
        self.assertFalse(path.exists())

    def make_store(self, name):
        store = OrchestratorStore(self.root / name)
        self.addCleanup(store.close)
        store._connection.execute("CREATE TABLE review_parent(id INTEGER PRIMARY KEY)")
        store._connection.execute(
            "CREATE TABLE review_child(parent_id INTEGER REFERENCES review_parent(id) DEFERRABLE INITIALLY DEFERRED)"
        )
        return store

    def test_failed_commit_rolls_back_and_allows_the_next_transaction(self):
        for name in ("_read_transaction", "_transaction"):
            with self.subTest(name=name):
                store = self.make_store(name + ".sqlite3")
                transaction = getattr(store, name)
                with self.assertRaises(sqlite3.IntegrityError):
                    with transaction() as connection:
                        connection.execute("INSERT INTO review_child VALUES(99)")
                self.assertFalse(store._connection.in_transaction)
                self.assertEqual(
                    store._connection.execute(
                        "SELECT count(*) FROM review_child"
                    ).fetchone()[0],
                    0,
                )
                with transaction() as connection:
                    connection.execute("INSERT INTO review_parent VALUES(99)")
                    connection.execute("INSERT INTO review_child VALUES(99)")
                self.assertEqual(
                    store._connection.execute(
                        "SELECT count(*) FROM review_child"
                    ).fetchone()[0],
                    1,
                )

    def test_sqlite_auto_rollback_preserves_the_original_constraint_error(self):
        for name in ("_read_transaction", "_transaction"):
            with self.subTest(name=name):
                store = self.make_store("auto" + name + ".sqlite3")
                store._connection.execute("INSERT INTO review_parent VALUES(99)")
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "UNIQUE constraint failed"
                ):
                    with getattr(store, name)() as connection:
                        connection.execute(
                            "INSERT OR ROLLBACK INTO review_parent VALUES(99)"
                        )
                self.assertFalse(store._connection.in_transaction)
                with store._transaction() as connection:
                    connection.execute("INSERT INTO review_parent VALUES(100)")

    def test_failed_begin_does_not_rollback_an_existing_transaction(self):
        for name in ("_read_transaction", "_transaction"):
            with self.subTest(name=name):
                store = self.make_store("nested" + name + ".sqlite3")
                with store._transaction() as connection:
                    connection.execute("INSERT INTO review_parent VALUES(99)")
                    with self.assertRaises(sqlite3.OperationalError):
                        with getattr(store, name)():
                            self.fail("nested BEGIN succeeded")
                    self.assertTrue(connection.in_transaction)
                    self.assertEqual(
                        connection.execute(
                            "SELECT count(*) FROM review_parent"
                        ).fetchone()[0],
                        1,
                    )
                self.assertEqual(
                    connection.execute("SELECT count(*) FROM review_parent").fetchone()[
                        0
                    ],
                    1,
                )


if __name__ == "__main__":
    unittest.main()
