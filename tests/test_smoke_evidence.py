from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


class SmokeEvidenceTests(unittest.TestCase):
    def evidence_functions(self):
        root = Path(__file__).resolve().parents[1] / "scripts"
        for name, function in (
            ("smoke_orchestrator_memory", "_task_evidence"),
            ("smoke_orchestrator_text", "_durable_evidence"),
        ):
            spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            yield name, getattr(module, function)

    def test_missing_evidence_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, read in self.evidence_functions():
                with self.subTest(script=name):
                    path = Path(directory) / f"{name}.sqlite3"
                    with self.assertRaises(sqlite3.OperationalError):
                        read(path, "turn")
                    self.assertFalse(path.exists())

    def test_existing_database_with_uri_characters_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence #?.sqlite3"
            with closing(sqlite3.connect(path)) as database:
                database.executescript("""
                    CREATE TABLE tasks(task_id, state, epoch, conversation_id);
                    CREATE TABLE canonical_turns(task_id, turn_id, status);
                    CREATE TABLE task_events(task_id, event_type, payload_json, event_sequence);
                    CREATE TABLE memory_records(conversation_id);
                    CREATE TABLE reservations(task_id, state, actual_output_tokens, created_at);
                    INSERT INTO tasks VALUES('task', 'COMPLETED', 1, 'conversation');
                    INSERT INTO canonical_turns VALUES('task', 'turn', 'COMPLETED');
                    INSERT INTO memory_records VALUES('conversation');
                    INSERT INTO reservations VALUES('task', 'RECONCILED', 2, 1);
                """)
            before = path.read_bytes()
            for name, read in self.evidence_functions():
                with self.subTest(script=name):
                    result = read(path, "turn")
                    self.assertEqual(result["task_state"], "COMPLETED")
                    if name.endswith("memory"):
                        self.assertEqual(result["memory_record_count"], 1)
                    else:
                        self.assertTrue(result["usage_reconciled"])
            self.assertEqual(path.read_bytes(), before)
